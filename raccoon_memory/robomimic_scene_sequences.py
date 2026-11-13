"""Episode-aligned scene sequences for an unmodified robomimic recurrent core.

Only the stored 82 features are observations. Times, labels, and the residuals
needed to reproduce the original recorded command are metadata, never inputs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from raccoon_memory import scene_sequence_data as source


HORIZON = 10
OBS_KEY = "scene"
FEATURE_COUNT = 82
ACTION_COUNT = 5
EXPECTED_DATA_SHA256 = "bbde3f4209b59c851b53cda9e92737d1ba9fbdd8ab1617d3b2585299838b7e27"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_recorded_command(command: dict) -> tuple[np.ndarray, list[float]]:
    """Encode one actual command and keep roundoff needed for exact recovery.

    The official model accepts float32 tensors. A residual stored outside model
    inputs makes the original float64 joint values exactly recoverable despite
    float32 quantization. No clipping or target inference is performed.
    """
    q = command.get("joint_target_deg")
    grip = command.get("gripper_target")
    if not isinstance(q, list) or len(q) != 4 or grip not in {"closed", "open"}:
        raise ValueError("expected four joint targets and an explicit gripper target")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in q):
        raise ValueError("joint targets must be finite numbers")
    if type(command.get("gripper_opening_m")) not in (int, float):
        raise ValueError("recorded gripper opening must be a number")
    if command.get("gripper_opening_m") != (0.0 if grip == "closed" else 0.04):
        raise ValueError("recorded gripper opening disagrees with target")
    original = [float(value) for value in q]
    if not all(np.isfinite(original)):
        raise ValueError("joint targets must be finite")
    normalized = np.asarray([value / 180.0 for value in original], dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("joint targets cannot be represented in float32")
    residual = [value - float(encoded) * 180.0 for value, encoded in zip(original, normalized)]
    result = np.empty(ACTION_COUNT, dtype=np.float32)
    result[:4] = normalized
    result[4] = -1.0 if grip == "closed" else 1.0
    return result, residual


def decode_recorded_command(encoded: np.ndarray, residual: list[float]) -> dict:
    """Recover a recorded label; this is not a decoder for predicted gripper values."""
    action = np.asarray(encoded)
    if (action.shape != (ACTION_COUNT,) or not np.isfinite(action).all() or len(residual) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in residual)):
        raise ValueError("invalid encoded recorded command")
    if float(action[4]) not in (-1.0, 1.0):
        raise ValueError("recorded gripper labels must be exactly -1 or +1")
    grip = "closed" if float(action[4]) == -1.0 else "open"
    return {"joint_target_deg": [float(action[i]) * 180.0 + float(residual[i]) for i in range(4)],
            "gripper_target": grip, "gripper_opening_m": 0.0 if grip == "closed" else 0.04}


def iter_episodes(path: Path | str, split: str | None = None,
                  expected_sha256: str | None = EXPECTED_DATA_SHA256) -> Iterator[dict]:
    """Read only complete episode records, in file order, without changing them."""
    path = Path(path)
    if split not in (None, "train", "validation"):
        raise ValueError("split must be train, validation, or None")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("input gzip digest differs from the fixed intermediate")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"empty episode at line {line_no}")
            episode = json.loads(line)
            source.validate_episode(episode)
            if split is None or episode["split"] == split:
                yield episode


def episode_windows(episode: dict) -> Iterator[dict]:
    """Partition at source record indices 0, 10, ... and right-pad each chunk.

    Every chunk begins at the official online reset phase. Student prefix and
    terminal records consume their actual observation positions but never
    acquire an action label. A terminal-only chunk is retained.
    """
    source.validate_episode(episode)
    records = episode["records"]
    for start in range(0, len(records), HORIZON):
        rows = records[start:start + HORIZON]
        observations = np.zeros((HORIZON, FEATURE_COUNT), dtype=np.float32)
        actions = np.zeros((HORIZON, ACTION_COUNT), dtype=np.float32)
        observation_mask = np.zeros(HORIZON, dtype=np.bool_)
        action_mask = np.zeros(HORIZON, dtype=np.bool_)
        terminal_mask = np.zeros(HORIZON, dtype=np.bool_)
        terminal_label = np.full(HORIZON, -1, dtype=np.int8)
        metadata = []
        for offset, record in enumerate(rows):
            observations[offset] = np.asarray(record["features"], dtype=np.float32)
            observation_mask[offset] = True
            action_mask[offset] = record["servo_supervision_valid"]
            terminal_mask[offset] = record["terminal_supervision_valid"]
            if terminal_mask[offset]:
                terminal_label[offset] = int(record["terminal_label"])
            residual = None
            original = record["servo_target"]
            if action_mask[offset]:
                actions[offset], residual = encode_recorded_command(original)
                recovered = decode_recorded_command(actions[offset], residual)
                if any(recovered[key] != original[key] for key in recovered):
                    raise ValueError("recorded command is not exactly recoverable")
            metadata.append({"record_index": start + offset, "kind": record["kind"],
                             "source_time_seconds": record["source_time_seconds"],
                             "original_servo_target": original, "joint_roundoff_residual": residual})
        metadata.extend([None] * (HORIZON - len(rows)))
        yield {"observations": observations, "actions": actions, "action_mask": action_mask,
               "observation_mask": observation_mask, "terminal_label": terminal_label,
               "terminal_mask": terminal_mask,
               "metadata": {"episode_id": episode["episode_id"], "split": episode["split"],
                            "source_kind": episode["source_kind"], "group": episode["group"],
                            "parent_group": episode.get("parent_group"),
                            "source_lineage": episode.get("source_lineage"),
                            "source_episode_line_sha256": episode.get("source_episode_line_sha256"),
                            "start_record_index": start, "source_reset_before_record_index": 0,
                            "records": metadata}}


class SceneSequenceIterableDataset(IterableDataset):
    """Streaming deterministic chunks; sampling and source weights remain unset."""

    def __init__(self, path: Path | str, split: str, expected_sha256: str | None = EXPECTED_DATA_SHA256):
        super().__init__()
        if split not in ("train", "validation"):
            raise ValueError("only train and validation are available")
        self.path, self.split, self.expected_sha256 = Path(path), split, expected_sha256

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("worker sharding is not configured; use num_workers=0")
        for episode in iter_episodes(self.path, self.split, self.expected_sha256):
            yield from episode_windows(episode)


def collate_windows(windows: list[dict]) -> dict:
    """Place only the 82 stored features in the model observation dictionary."""
    if not windows:
        raise ValueError("cannot collate an empty batch")
    stack = lambda name: torch.from_numpy(np.stack([window[name] for window in windows]))
    obs_mask, action_mask, terminal_mask = (stack(name) for name in ("observation_mask", "action_mask", "terminal_mask"))
    if torch.any(action_mask & ~obs_mask) or torch.any(terminal_mask & ~obs_mask):
        raise ValueError("padding cannot have supervision")
    if torch.any((~obs_mask[:, :-1]) & obs_mask[:, 1:]):
        raise ValueError("observations must be right-padded")
    return {"obs": {OBS_KEY: stack("observations")}, "actions": stack("actions"),
            "observation_mask": obs_mask, "action_mask": action_mask,
            "terminal_label": stack("terminal_label"), "terminal_mask": terminal_mask,
            "metadata": [window["metadata"] for window in windows]}
