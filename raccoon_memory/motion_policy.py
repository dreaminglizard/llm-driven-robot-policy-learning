"""Offline waypoint imitation; this module never contacts a robot or service.

Inputs are deliberately enumerated below. A terminal label is generated only
from the last post-action observation, and is not a recorded device command.
Outputs are event-based waypoint deltas, not periodic torque or position
feedback. Speed is not learned; a controller must supply its own limits.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ACTIONS = ("move", "open", "close", "done")
MOVE, OPEN, CLOSE, DONE = range(4)
JOINT_SCALE_DEG = 180.0
TASK_SCALE = 4.0
HIDDEN_SIZE = 64
FEATURE_NAMES = (
    "joint_1_deg/180", "joint_2_deg/180", "joint_3_deg/180", "joint_4_deg/180",
    "sensor_valid", "moving",
    "gripper_command_none", "gripper_command_open", "gripper_command_closed",
    "gripper_reported_none", "gripper_reported_open", "gripper_reported_closed",
    "gripper_reported_valid", "gripper_feedback_fresh",
    "source_slot/4", "destination_slot/4", "source_level/4", "destination_level/4",
)
TASK_FIELDS = ("source_slot", "destination_slot", "source_level", "destination_level")
LIMITATION = (
    "Metrics predict the next recorded action on recorded observations; they are not "
    "closed-loop or physical success rates. No object pose or visual feedback is available. "
    "Outputs are event-based waypoints, not fixed-period or torque control. Speed is not "
    "learned and must follow controller limits. Generated done labels are not recorded commands."
)


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required(mapping: dict, key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing {label}.{key}")
    return mapping[key]


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} is not finite")


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must have four joint angles")
    return [_finite_number(item, label) for item in value]


def _flag(obs: dict, key: str) -> bool:
    value = _required(obs, key, "observation")
    if type(value) is not bool:
        raise ValueError(f"observation.{key} must be boolean")
    return value


def encode_features(observation: dict, task: dict) -> list[float]:
    """Encode only the declared current-state/task whitelist; no target is read.

    An explicitly absent previous command and explicitly invalid gripper report
    have distinct categories. Missing keys, missing joints, stale joint sensors,
    and inconsistent gripper validity are rejected rather than imputed.
    """
    obs = _mapping(observation, "observation")
    task = _mapping(task, "task")
    if _required(obs, "schema_version", "observation") != "PolicyObservation.phase1":
        raise ValueError("unsupported observation schema")
    joints = _vector(_required(obs, "joints_deg", "observation"), "observation.joints_deg")
    valid, moving = _flag(obs, "sensor_valid"), _flag(obs, "moving")
    report_valid, report_fresh = _flag(obs, "gripper_reported_valid"), _flag(obs, "gripper_feedback_fresh")
    if not valid:
        raise ValueError("joint sensor is invalid")
    command = _required(obs, "gripper_command", "observation")
    report = _required(obs, "gripper_reported_state", "observation")
    if command not in (None, "open", "closed") or report not in (None, "open", "closed"):
        raise ValueError("unsupported gripper state")
    if (report_valid and (not report_fresh or report is None)) or (not report_valid and report is not None):
        raise ValueError("gripper report and validity disagree")
    task_values = []
    for key in TASK_FIELDS:
        value = _required(task, key, "task")
        if type(value) is not int or value < 1:
            raise ValueError(f"task.{key} must be a positive integer")
        task_values.append(_finite_number(value, f"task.{key}") / TASK_SCALE)
    if task["source_slot"] == task["destination_slot"]:
        raise ValueError("source and destination slots must differ")
    result = [value / JOINT_SCALE_DEG for value in joints]
    result.extend((float(valid), float(moving)))
    result.extend(float(command == value) for value in (None, "open", "closed"))
    result.extend(float(report == value) for value in (None, "open", "closed"))
    result.extend((float(report_valid), float(report_fresh)))
    result.extend(task_values)
    if len(result) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in result):
        raise ValueError("non-finite encoded feature")
    return result


def _time(observation: dict, label: str) -> float:
    value = _finite_number(_required(observation, "observed_monotonic", label), f"{label}.observed_monotonic")
    if value < 0:
        raise ValueError(f"{label}.observed_monotonic must be nonnegative")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _label(action: dict, observation: dict, applied: Any, status: Any) -> tuple[int, list[float]]:
    action = _mapping(action, "action")
    kind = _required(action, "kind", "action")
    if kind == "move_joints":
        target = _vector(_required(action, "joints_deg", "action"), "action.joints_deg")
        speed = _finite_number(_required(action, "speed_pct", "action"), "action.speed_pct")
        if not 0 < speed <= 100:
            raise ValueError("action.speed_pct must be in (0, 100]")
        if not ((status == "completed" and applied is True) or (status == "already_at_target" and applied is False)):
            raise ValueError("motion result was not confirmed")
        prior = _vector(observation["joints_deg"], "observation.joints_deg")
        delta = [(after - before) / JOINT_SCALE_DEG for before, after in zip(prior, target)]
        if not all(math.isfinite(v) for v in delta):
            raise ValueError("non-finite joint delta")
        return MOVE, delta
    if kind == "set_gripper":
        state = _required(action, "state", "action")
        if state not in ("open", "closed"):
            raise ValueError("unsupported gripper action")
        if status != "gripper_reported" or applied is not True:
            raise ValueError("gripper result was not confirmed")
        return (OPEN if state == "open" else CLOSE), [0.0] * 4
    raise ValueError(f"unsupported action kind: {kind!r}")


@dataclass
class MotionEpisode:
    episode_id: str
    features: list[list[float]]
    actions: list[int]
    deltas: list[list[float]]
    quality: dict
    terminal: dict


def build_episodes(records: list[dict]) -> list[MotionEpisode]:
    """Validate every episode and all command/observation identifiers globally."""
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must contain at least one episode")
    episode_ids: set[str] = set()
    command_ids: set[str] = set()
    observation_ids: set[str] = set()
    episodes = []
    for index, record in enumerate(records):
        record = _mapping(record, f"episode[{index}]")
        _finite_tree(record, f"episode[{index}]")
        episode_id = _identifier(_required(record, "episode_id", "episode"), "episode_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        task = _mapping(_required(record, "task", "episode"), "task")
        quality = _mapping(_required(record, "quality", "episode"), "quality")
        transitions = _required(record, "transitions", "episode")
        if not isinstance(transitions, list) or not transitions:
            raise ValueError(f"{episode_id}: empty transitions")
        features, actions, deltas = [], [], []
        previous_end = None
        for position, transition in enumerate(transitions):
            label = f"{episode_id}.transitions[{position}]"
            transition = _mapping(transition, label)
            obs = _mapping(_required(transition, "observation", label), f"{label}.observation")
            next_obs = _mapping(_required(transition, "next_observation", label), f"{label}.next_observation")
            features.append(encode_features(obs, task))
            # Validate all post-action observations, but do not expose them to the
            # corresponding action predictor. Only the final one labels done.
            encode_features(next_obs, task)
            before, after = _time(obs, "observation"), _time(next_obs, "next_observation")
            if not before < after or (previous_end is not None and before < previous_end):
                raise ValueError(f"{label}: observation times are not in action order")
            previous_end = after
            provenance = _mapping(_required(transition, "provenance", label), f"{label}.provenance")
            command = _identifier(_required(provenance, "command_id", "provenance"), "command_id")
            if command in command_ids:
                raise ValueError(f"duplicate command_id: {command}")
            command_ids.add(command)
            for field in ("input_observation_id", "result_observation_id"):
                obs_id = _identifier(_required(provenance, field, "provenance"), field)
                if obs_id in observation_ids:
                    raise ValueError(f"duplicate observation_id: {obs_id}")
                observation_ids.add(obs_id)
            action, delta = _label(_required(transition, "action", label), obs,
                                   _required(transition, "applied", label),
                                   _required(transition, "status", label))
            actions.append(action)
            deltas.append(delta)
        features.append(encode_features(next_obs, task))
        actions.append(DONE)
        deltas.append([0.0] * 4)
        terminal = {"episode_id": episode_id, "action": "done", "label_origin": "generated_terminal",
                    "observation_source": "last_transition.next_observation",
                    "observation_id": transitions[-1]["provenance"]["result_observation_id"]}
        episodes.append(MotionEpisode(episode_id, features, actions, deltas, quality, terminal))
    return episodes


def parse_jsonl(raw: bytes) -> list[dict]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("dataset must be UTF-8 JSONL") from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid dataset line {number}: {exc}") from exc
        records.append(value)
    return records


def select_splits(episodes: list[MotionEpisode], train_ids: list[str], test_ids: list[str]) -> tuple[list[MotionEpisode], list[MotionEpisode]]:
    by_id = {episode.episode_id: episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate episode IDs")
    for name, ids in (("train", train_ids), ("test", test_ids)):
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{name} episode list is empty")
        for value in ids:
            _identifier(value, f"{name} episode ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {name} episode ID")
        missing = set(ids) - set(by_id)
        if missing:
            raise ValueError(f"unknown {name} episode IDs: {sorted(missing)}")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"train/test episodes overlap: {sorted(overlap)}")
    return [by_id[value] for value in train_ids], [by_id[value] for value in test_ids]


def class_weights(episodes: list[MotionEpisode]) -> tuple[torch.Tensor, list[int]]:
    counts = Counter(action for episode in episodes for action in episode.actions)
    if counts[MOVE] == 0:
        raise ValueError("training split needs at least one motion command")
    observed = len(counts)
    total = sum(counts.values())
    values = [total / (observed * counts[i]) if counts[i] else 0.0 for i in range(len(ACTIONS))]
    return torch.tensor(values, dtype=torch.float32), [counts[i] for i in range(len(ACTIONS))]


def collate(episodes: list[MotionEpisode], device: torch.device) -> dict[str, torch.Tensor]:
    if not episodes:
        raise ValueError("cannot collate an empty split")
    lengths = torch.tensor([len(ep.actions) for ep in episodes], dtype=torch.long)
    width = int(lengths.max())
    inputs = torch.zeros((len(episodes), width, len(FEATURE_NAMES)), dtype=torch.float32)
    actions = torch.full((len(episodes), width), -100, dtype=torch.long)
    deltas = torch.zeros((len(episodes), width, 4), dtype=torch.float32)
    mask = torch.zeros((len(episodes), width), dtype=torch.bool)
    for row, ep in enumerate(episodes):
        length = len(ep.actions)
        inputs[row, :length] = torch.tensor(ep.features, dtype=torch.float32)
        actions[row, :length] = torch.tensor(ep.actions, dtype=torch.long)
        deltas[row, :length] = torch.tensor(ep.deltas, dtype=torch.float32)
        mask[row, :length] = True
    if not torch.isfinite(inputs).all() or not torch.isfinite(deltas).all():
        raise ValueError("values overflowed the model's finite float32 range")
    return {"features": inputs.to(device), "actions": actions.to(device), "deltas": deltas.to(device),
            "mask": mask.to(device), "lengths": lengths}


class MotionPolicy(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE, output_parameterization: str = "delta"):
        super().__init__()
        if output_parameterization not in ("delta", "absolute"):
            raise ValueError("output parameterization must be delta or absolute")
        self.hidden_size = hidden_size
        self.output_parameterization = output_parameterization
        self.gru = nn.GRU(len(FEATURE_NAMES), hidden_size, batch_first=True)
        self.action_head = nn.Linear(hidden_size, len(ACTIONS))
        self.delta_head = nn.Linear(hidden_size, 4)

    def _deltas(self, head_output: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # The first four whitelisted features are current joint degrees / 180.
        # Absolute heads predict target degrees / 180; all public outputs remain
        # normalized deltas so the original labels and loss stay unchanged.
        if self.output_parameterization == "absolute":
            return head_output - features[..., :4]
        return head_output

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Each forward starts with GRU's zero hidden state. Packed sequences keep
        # independent episode rows and exclude padded events from recurrence.
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.shape[1])
        return self.action_head(encoded), self._deltas(self.delta_head(encoded), features)

    def step(self, features: torch.Tensor, hidden: torch.Tensor | None = None, *, return_head_output: bool = False):
        """Return logits, normalized deltas and hidden; optionally also the raw joint head."""
        if features.ndim != 3 or features.shape[1] != 1 or features.shape[2] != len(FEATURE_NAMES):
            raise ValueError("step features must have shape (batch, 1, feature_count)")
        encoded, hidden = self.gru(features, hidden)
        raw = self.delta_head(encoded)
        outputs = (self.action_head(encoded), self._deltas(raw, features), hidden)
        return (*outputs, raw) if return_head_output else outputs


def masked_loss(logits: torch.Tensor, predicted_deltas: torch.Tensor, actions: torch.Tensor,
                target_deltas: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not mask.any():
        raise ValueError("loss requires at least one unpadded event")
    ce = F.cross_entropy(logits[mask], actions[mask], weight=weights)
    moves = mask & (actions == MOVE)
    mse = F.mse_loss(predicted_deltas[moves], target_deltas[moves]) if moves.any() else predicted_deltas[mask].sum() * 0.0
    return ce + mse, ce, mse


@torch.no_grad()
def evaluate(model: MotionPolicy, batch: dict[str, torch.Tensor]) -> dict:
    model.eval()
    logits, deltas = model(batch["features"], batch["lengths"])
    if not torch.isfinite(logits).all() or not torch.isfinite(deltas).all():
        raise ValueError("non-finite evaluation output")
    mask, target = batch["mask"], batch["actions"]
    predicted = logits.argmax(dim=-1)
    confusion = torch.zeros((len(ACTIONS), len(ACTIONS)), dtype=torch.long)
    for actual, guess in zip(target[mask].cpu().tolist(), predicted[mask].cpu().tolist()):
        confusion[actual, guess] += 1
    moves = mask & (target == MOVE)
    count = int(moves.sum().item())
    error = (deltas[moves] - batch["deltas"][moves]).abs() * JOINT_SCALE_DEG
    return {
        "event_count": int(mask.sum().item()), "move_event_count": count,
        "recorded_action_count": int(mask.sum().item()) - int((mask & (target == DONE)).sum().item()),
        "generated_done_count": int((mask & (target == DONE)).sum().item()),
        "action_accuracy": float((predicted[mask] == target[mask]).float().mean().item()),
        "action_cross_entropy_unweighted": float(F.cross_entropy(logits[mask], target[mask]).item()),
        "joint_delta_mae_deg": float(error.mean().item()) if count else None,
        "joint_delta_mae_deg_per_joint": error.mean(dim=0).cpu().tolist() if count else None,
        "confusion_matrix": {"labels": list(ACTIONS), "rows": "actual", "columns": "predicted",
                             "counts": confusion.tolist()},
    }
