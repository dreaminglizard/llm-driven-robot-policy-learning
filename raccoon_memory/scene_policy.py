"""Shared history-window policies for registered-scene manipulation.

The three architectures receive the same past/current feature window and a
left-padding mask. Their outputs are action logits and absolute joint targets
normalized by 180 degrees. This module has no robot or network integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features


ARCHITECTURES = ("mlp", "gru", "transformer", "slot_shared")
MAX_PARAMETERS = 250_000
RECORDED_METRIC_SCOPE = (
    "Recorded-observation next-action accuracy and target error only; "
    "not closed-loop or physical success. Validation holds out registered "
    "layout/task groups; shared teacher references are not independent teachers."
)


def split_slot_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rearrange the existing 82 fields into four 14-value rows and 26 globals.

    Rows contain XY, bottom/top five-way IDs, source flag and destination flag.
    Globals contain arm/gripper, source/destination levels, target and active IDs.
    This rearrangement is invertible before the shared encoder compresses rows.
    """
    if not isinstance(features, torch.Tensor) or features.ndim < 1 or features.shape[-1] != 82:
        raise ValueError("scene features must have a final dimension of 82")
    occupancy = features[..., 30:70].reshape(*features.shape[:-1], 4, 10)
    coordinates = features[..., 74:82].reshape(*features.shape[:-1], 4, 2)
    rows = torch.cat((coordinates, occupancy, features[..., 14:18].unsqueeze(-1),
                      features[..., 18:22].unsqueeze(-1)), dim=-1)
    globals_ = torch.cat((features[..., :14], features[..., 22:30], features[..., 70:74]), dim=-1)
    return rows, globals_


def join_slot_features(rows: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
    """Invert split_slot_features, preserving row order and every input value."""
    if (not isinstance(rows, torch.Tensor) or rows.ndim < 2 or rows.shape[-2:] != (4, 14)
            or not isinstance(globals_, torch.Tensor) or globals_.shape != (*rows.shape[:-2], 26)
            or rows.device != globals_.device or rows.dtype != globals_.dtype):
        raise ValueError("slot rows and globals must have matching leading dimensions and 4x14/26 values")
    return torch.cat((globals_[..., :14], rows[..., 12], rows[..., 13], globals_[..., 14:22],
                      rows[..., 2:12].reshape(*rows.shape[:-2], 40), globals_[..., 22:26],
                      rows[..., :2].reshape(*rows.shape[:-2], 8)), dim=-1)


@dataclass
class SceneEpisode:
    episode_id: str
    split_group: str
    source_references: Any
    declared_split: str | None
    split_plan_sha256: str | None
    features: list[list[float]]
    actions: list[int]
    absolute_targets: list[list[float]]
    terminal: dict


def split_group_for(registered_scene, task) -> str:
    """Group by count, source/destination levels and unordered other counts."""
    before, goal = scene_features.transfer_task(registered_scene, task)
    other = sorted(len(stack) for index, stack in enumerate(before, 1)
                   if index not in (goal["source_slot"], goal["destination_slot"]))
    total = sum(map(len, before))
    return f"n{total}_s{goal['source_level']}_d{goal['destination_level']}_other{other[0]}{other[1]}"


def _nonempty_id(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def build_episodes(records: list[dict]) -> list[SceneEpisode]:
    """Build prefix-only observations and a final generated terminal label.

    Intermediate next_observation values are not read. The final one is used
    only to form the generated done observation; no future action is supplied.
    Source references are retained as evidence, not encoded as inputs.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must contain at least one episode")
    episodes, episode_ids, command_ids, observation_ids = [], set(), set(), set()
    required_task = ("source_slot", "destination_slot", "source_level", "destination_level", "object_id")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("episode must be an object")
        episode_id = _nonempty_id(record.get("episode_id"), "episode_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        registered_scene, task = record.get("registered_scene"), record.get("task")
        if not isinstance(task, dict) or any(key not in task for key in required_task):
            raise ValueError(f"{episode_id}: task requires slots, levels and object_id")
        expected_group = split_group_for(registered_scene, task)
        split_group = _nonempty_id(record.get("split_group"), "split_group")
        if split_group != expected_group:
            raise ValueError(f"{episode_id}: split_group must be {expected_group}")
        sources = record.get("source_references")
        if not isinstance(sources, (dict, list)) or not sources:
            raise ValueError(f"{episode_id}: source_references must be a nonempty object or array")
        quality = record.get("quality", {})
        if not isinstance(quality, dict):
            raise ValueError(f"{episode_id}: quality must be an object when provided")
        declared_split = quality.get("split")
        if declared_split is not None and declared_split not in ("train", "validation", "test", "pilot"):
            raise ValueError(f"{episode_id}: unsupported declared split")
        split_plan_hash = quality.get("split_plan_sha256")
        if split_plan_hash is not None:
            if (not isinstance(split_plan_hash, str) or len(split_plan_hash) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in split_plan_hash)):
                raise ValueError(f"{episode_id}: split_plan_sha256 must be a SHA-256 digest")
            split_plan_hash = split_plan_hash.lower()
        if "nominal_slot_xy_mm" not in record:
            raise ValueError(f"{episode_id}: nominal_slot_xy_mm is required")
        nominal_xy = record["nominal_slot_xy_mm"]
        transitions = record.get("transitions")
        if not isinstance(transitions, list) or not transitions:
            raise ValueError(f"{episode_id}: transitions must be a nonempty list")
        features, actions, targets = [], [], []
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise ValueError(f"{episode_id}: transition must be an object")
            observation = transition.get("observation")
            features.append(scene_features.encode_scene_features(observation, registered_scene, task, nominal_xy))
            provenance = transition.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{episode_id}: transition provenance is required")
            command = _nonempty_id(provenance.get("command_id"), "command_id")
            if command in command_ids:
                raise ValueError(f"duplicate command_id: {command}")
            command_ids.add(command)
            for name in ("input_observation_id", "result_observation_id"):
                obs_id = _nonempty_id(provenance.get(name), name)
                if obs_id in observation_ids:
                    raise ValueError(f"duplicate observation_id: {obs_id}")
                observation_ids.add(obs_id)
            kind, _ = motion_policy._label(transition.get("action"), observation,
                                           transition.get("applied"), transition.get("status"))
            target = [0.0] * 4
            if kind == motion_policy.MOVE:
                degrees = motion_policy._vector(transition["action"]["joints_deg"], "action.joints_deg")
                if any(not low <= angle <= high for angle, (low, high) in zip(degrees, JOINT_LIMITS)):
                    raise ValueError(f"{episode_id}: joint target is outside joint limits")
                target = [angle / motion_policy.JOINT_SCALE_DEG for angle in degrees]
            actions.append(kind)
            targets.append(target)
        # This is the only next_observation read anywhere in this builder.
        terminal_obs = transitions[-1].get("next_observation")
        features.append(scene_features.encode_scene_features(terminal_obs, registered_scene, task, nominal_xy))
        actions.append(motion_policy.DONE)
        targets.append([0.0] * 4)
        terminal = {"episode_id": episode_id, "action": "done", "label_origin": "generated_terminal",
                    "observation_source": "last_transition.next_observation",
                    "observation_id": transitions[-1]["provenance"]["result_observation_id"]}
        episodes.append(SceneEpisode(episode_id, split_group, sources, declared_split, split_plan_hash,
                                     features, actions, targets, terminal))
    return episodes


def select_splits(episodes: list[SceneEpisode], train_ids: list[str], validation_ids: list[str]):
    by_id = {episode.episode_id: episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate episode IDs")
    for label, ids in (("train", train_ids), ("validation", validation_ids)):
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{label} IDs must be a nonempty list")
        for item in ids:
            _nonempty_id(item, f"{label} episode ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {label} episode ID")
        missing = set(ids) - set(by_id)
        if missing:
            raise ValueError(f"unknown {label} episode IDs: {sorted(missing)}")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation episode IDs overlap")
    train = [by_id[item] for item in train_ids]
    validation = [by_id[item] for item in validation_ids]
    for label, selected in (("train", train), ("validation", validation)):
        conflicting = [e.episode_id for e in selected if e.declared_split not in (None, label)]
        if conflicting:
            raise ValueError(f"declared split conflicts with {label}: {conflicting}")
    plan_hashes = {e.split_plan_sha256 for e in train + validation}
    if len(plan_hashes) > 1:
        raise ValueError("selected episodes must use the same declared split plan")
    overlap = {item.split_group for item in train} & {item.split_group for item in validation}
    if overlap:
        raise ValueError(f"train and validation split_group overlap: {sorted(overlap)}")
    return train, validation


def collate_windows(episodes: list[SceneEpisode], history_length: int = scene_features.HISTORY_LENGTH) -> dict:
    if not episodes:
        raise ValueError("cannot collate an empty split")
    windows, masks, actions, targets, episode_ids, decision_indices = [], [], [], [], [], []
    for episode in episodes:
        history = []
        for index, (features, action, target) in enumerate(zip(episode.features, episode.actions,
                                                               episode.absolute_targets)):
            history.append(features)
            window, mask = scene_features.history_window(history, length=history_length)
            windows.append(window)
            masks.append(mask)
            actions.append(action)
            targets.append(target)
            episode_ids.append(episode.episode_id)
            decision_indices.append(index)
    return {"windows": torch.tensor(windows, dtype=torch.float32),
            "masks": torch.tensor(masks, dtype=torch.bool),
            "actions": torch.tensor(actions, dtype=torch.long),
            "absolute_targets": torch.tensor(targets, dtype=torch.float32),
            "episode_ids": episode_ids, "decision_indices": decision_indices}


class ScenePolicy(nn.Module):
    """Compare flat and shared-slot policies on identical observations."""

    def __init__(self, architecture: str = "gru", hidden_size: int = 64,
                 history_length: int = scene_features.HISTORY_LENGTH):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(history_length) is not int or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        if architecture == "transformer" and hidden_size % 4:
            raise ValueError("transformer hidden_size must be divisible by four")
        self.architecture = architecture
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.feature_count = len(scene_features.FEATURE_NAMES)
        if architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(history_length * (self.feature_count + 1), hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )
        elif architecture == "slot_shared":
            self.slot_encoder = nn.Sequential(nn.Linear(14, 18), nn.ReLU(), nn.Linear(18, 18), nn.ReLU())
            self.encoder = nn.Sequential(nn.Linear(history_length * 81, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size), nn.ReLU())
        elif architecture == "gru":
            self.encoder = nn.GRU(self.feature_count, hidden_size, batch_first=True)
        else:
            self.input_projection = nn.Linear(self.feature_count, hidden_size)
            self.position_embedding = nn.Embedding(history_length, hidden_size)
            layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4,
                                               dim_feedforward=hidden_size * 2,
                                               dropout=0.0, activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.action_head = nn.Linear(hidden_size, len(motion_policy.ACTIONS))
        self.joint_head = nn.Linear(hidden_size, 4)
        if sum(parameter.numel() for parameter in self.parameters()) > MAX_PARAMETERS:
            raise ValueError(f"model must have at most {MAX_PARAMETERS} parameters")

    def _validate(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if (not isinstance(windows, torch.Tensor) or not windows.is_floating_point()
                or windows.ndim != 3 or windows.shape[1:] != (self.history_length, self.feature_count)):
            raise ValueError("windows must have shape (batch, history_length, feature_count)")
        if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                or masks.shape != windows.shape[:2] or masks.device != windows.device):
            raise ValueError("masks must be a boolean tensor matching batch and history dimensions")
        if not windows.shape[0] or not bool(torch.isfinite(windows).all()):
            raise ValueError("windows must be nonempty and finite")
        if not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any()):
            raise ValueError("each mask must be left-padded and include its current observation")
        return masks.sum(dim=1)

    def _compact(self, windows: torch.Tensor, lengths: torch.Tensor):
        """Move valid suffixes to the front; trailing slots are ignored."""
        time = torch.arange(self.history_length, device=windows.device).unsqueeze(0)
        positions = (time + self.history_length - lengths.unsqueeze(1)).clamp(max=self.history_length - 1)
        compact = windows.gather(1, positions.unsqueeze(-1).expand(-1, -1, self.feature_count))
        valid = time < lengths.unsqueeze(1)
        return compact.masked_fill(~valid.unsqueeze(-1), 0.0), valid, positions

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = self._validate(windows, masks)
        if self.architecture == "mlp":
            masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
            encoded = self.encoder(torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1))
        elif self.architecture == "slot_shared":
            rows, globals_ = split_slot_features(windows)
            embedded = self.slot_encoder(rows)
            pooled = embedded.sum(dim=-2)
            source = (embedded * rows[..., 12:13]).sum(dim=-2)
            destination = (embedded * rows[..., 13:14]).sum(dim=-2)
            values = torch.cat((pooled, source, destination, globals_), dim=-1)
            values = values.masked_fill(~masks.unsqueeze(-1), 0.0)
            encoded = self.encoder(torch.cat((values, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1))
        else:
            compact, valid, positions = self._compact(windows, lengths)
            if self.architecture == "gru":
                packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, hidden = self.encoder(packed)
                encoded = hidden[-1]
            else:
                values = self.input_projection(compact) + self.position_embedding(positions)
                causal = torch.triu(torch.ones((self.history_length, self.history_length), dtype=torch.bool,
                                              device=windows.device), diagonal=1)
                sequence = self.encoder(values, mask=causal, src_key_padding_mask=~valid)
                encoded = sequence[torch.arange(windows.shape[0], device=windows.device), lengths - 1]
        return self.action_head(encoded), self.joint_head(encoded)
