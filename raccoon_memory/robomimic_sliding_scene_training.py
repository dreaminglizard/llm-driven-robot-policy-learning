"""State and accounting helpers for the independent sliding-window run.

This module does not change the official policy or the frozen aligned-window
runner.  The two PCG64 streams and the torch CPU RNG are explicit state.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from raccoon_memory.robomimic_scene_training import model_weight_sha256
from raccoon_memory.robomimic_sliding_scene_data import fixed_candidate_draw

RESUME_SCHEMA = "raccoon-robomimic-sliding-scene-resume.v1"
SOURCES = ("physical", "teacher", "direct")


def candidate_rng(seed):
    entropy = hashlib.sha256(f"raccoon-sliding-start-v1|{seed}".encode("ascii")).digest()[:16]
    return np.random.Generator(np.random.PCG64(int.from_bytes(entropy, "big")))


def checked_draw(base_sampler, rng, metadata, original_row, candidate_row):
    original = base_sampler.draw(len(original_row))
    if not np.array_equal(original, original_row):
        raise ValueError("actual original draw differs from frozen parent, ID or original window")
    candidate = fixed_candidate_draw(original, metadata, rng)
    if not np.array_equal(candidate, candidate_row):
        raise ValueError("actual sliding start draw differs from the frozen table")
    return candidate


def empty_counts():
    return {"updates": 0, "sampled_windows": 0, "valid_actions": 0,
            "by_source": {s: {"windows": 0, "valid_actions": 0,
                               "first_command": 0, "eleventh_command": 0} for s in SOURCES},
            "start_positions": [0] * 20}


def count_batch(counts, draws, actual_action_mask, metadata, *, include_update=True):
    """Count only materialized action labels, including direct-prefix ordinals."""
    mask = np.asarray(actual_action_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(draws), 10):
        raise ValueError("action mask must be a boolean [batch,10] array")
    positions = draws["start"].astype(np.int16)[:, None] + np.arange(10, dtype=np.int16)
    within = positions < 20
    bounded = np.minimum(positions, 19)
    action = metadata["absolute_action_mask"][draws["unit"], draws["variant"]]
    expected_mask = within & np.take_along_axis(action, bounded, axis=1)
    if not np.array_equal(mask, expected_mask) or not np.array_equal(mask.sum(axis=1), draws["valid_actions"]):
        raise ValueError("materialized action labels disagree with the frozen window")
    ordinal = np.take_along_axis(np.cumsum(action, axis=1), bounded, axis=1)
    sources = np.where(draws["unit"] < 621, 0, np.where(draws["unit"] < 645, 1, 2))
    counts["updates"] += int(include_update)
    counts["sampled_windows"] += len(draws)
    counts["valid_actions"] += int(mask.sum())
    counts["start_positions"] = (np.asarray(counts["start_positions"]) +
                                  np.bincount(draws["start"], minlength=20)).tolist()
    for source_id, source in enumerate(SOURCES):
        selected = sources == source_id
        target = counts["by_source"][source]
        target["windows"] += int(selected.sum())
        target["valid_actions"] += int(mask[selected].sum())
        target["first_command"] += int((mask[selected] & (ordinal[selected] == 1)).sum())
        target["eleventh_command"] += int((mask[selected] & (ordinal[selected] == 11)).sum())


def check_counts(counts, table, metadata, completed, batch_size):
    if counts["updates"] != completed or counts["sampled_windows"] != completed * batch_size:
        raise ValueError("checkpoint progress and actual sampling counts disagree")
    reconstructed = empty_counts()
    flat = table[:completed].reshape(-1)
    for begin in range(0, len(flat), 8192):
        draws = flat[begin:begin + 8192]
        positions = draws["start"].astype(np.int16)[:, None] + np.arange(10, dtype=np.int16)
        action = metadata["absolute_action_mask"][draws["unit"], draws["variant"]]
        mask = (positions < 20) & np.take_along_axis(action, np.minimum(positions, 19), axis=1)
        count_batch(reconstructed, draws, mask, metadata, include_update=False)
    reconstructed["updates"] = completed
    if reconstructed != counts:
        raise ValueError("checkpoint cumulative action or sample counts differ from the frozen table")


def to_model_batch(materialized):
    def tensor(name):
        return torch.from_numpy(np.array(materialized[name], copy=True))
    return {"obs": {"scene": tensor("observations")}, "actions": tensor("actions"),
            "action_mask": tensor("action_mask"), "observation_mask": tensor("observation_mask")}


def save_checkpoint(path, *, model, config, base_sampler, sliding_rng, identity, completed, counts, metrics):
    if type(completed) is not int or completed < 0:
        raise ValueError("invalid checkpoint update count")
    steps = {int(s["step"].item()) for s in model.optimizers["policy"].state.values() if "step" in s}
    if steps != (set() if completed == 0 else {completed}):
        raise ValueError("optimizer progress and checkpoint update disagree")
    payload = {"schema": RESUME_SCHEMA, "identity": identity, "config": config.dump(),
               "updates_completed": completed, "model": model.serialize(),
               "model_weight_sha256": model_weight_sha256(model),
               "torch_rng_state": torch.get_rng_state(),
               "baseline_rng_state": base_sampler.state_dict(),
               "sliding_rng_state": copy.deepcopy(sliding_rng.bit_generator.state),
               "cumulative": copy.deepcopy(counts), "block_metrics": copy.deepcopy(metrics)}
    path = Path(path)
    if path.exists():
        raise FileExistsError("recovery checkpoint already exists")
    partial = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def load_checkpoint(path, *, model, config, base_sampler, sliding_rng, identity):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("schema") != RESUME_SCHEMA or payload.get("identity") != identity
            or payload.get("config") != config.dump()):
        raise ValueError("candidate recovery identity or model config differs")
    model.deserialize(payload["model"], load_optimizers=True)
    if model_weight_sha256(model) != payload["model_weight_sha256"]:
        raise ValueError("recovered model weights differ")
    steps = {int(s["step"].item()) for s in model.optimizers["policy"].state.values() if "step" in s}
    completed = payload["updates_completed"]
    if steps != (set() if completed == 0 else {completed}):
        raise ValueError("recovered Adam state and update count disagree")
    if payload["sliding_rng_state"].get("bit_generator") != "PCG64":
        raise ValueError("wrong candidate RNG")
    base_sampler.load_state_dict(payload["baseline_rng_state"])
    sliding_rng.bit_generator.state = copy.deepcopy(payload["sliding_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload
