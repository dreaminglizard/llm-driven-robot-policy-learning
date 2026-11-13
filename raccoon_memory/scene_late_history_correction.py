"""Frozen 145-feature policy with an independent long-history joint correction.

The only inference inputs remain the original 82-feature window and mask.  The
new head uses the same frozen 64-vector plus current 81 non-moving features as
the existing early head.  It is applied only for H>=4 and predicted MOVE; no
teacher action, future observation, privileged position or clipping is used.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from raccoon_memory import motion_policy
from raccoon_memory import scene_current_observation_correction as current
from raccoon_memory import scene_early_history_correction as early


HEAD_FEATURE_COUNT = current.HEAD_FEATURE_COUNT
MIN_LATE_HISTORY = early.MAX_CORRECTED_HISTORY + 1
CHECKPOINT_FORMAT = "raccoon-scene-late-history-correction.v1"


class LateTrainingOutputs(NamedTuple):
    features: torch.Tensor
    action_logits: torch.Tensor
    base_joints: torch.Tensor
    early_correction: torch.Tensor
    late_correction: torch.Tensor
    history_lengths: torch.Tensor


class SceneLateHistoryCorrectionPolicy(nn.Module):
    """Freeze the full existing policy and train only a new 145-to-4 head."""

    def __init__(self):
        super().__init__()
        current._check_contract()
        self.architecture = "late_history_current_observation_correction_mlp"
        self.history_length = current.HISTORY_LENGTH
        self.feature_count = current.FEATURE_COUNT
        self.hidden_size = current.HIDDEN_SIZE
        self.frozen_policy = current.SceneCurrentObservationCorrectionPolicy()
        self.frozen_policy.requires_grad_(False)
        self.frozen_policy.eval()
        self.late_head = nn.Linear(HEAD_FEATURE_COUNT, 4)
        nn.init.zeros_(self.late_head.weight)
        nn.init.zeros_(self.late_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the entire original policy in evaluation mode in every caller.
        self.frozen_policy.eval()
        return self

    def head_features(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return detached [batch,145] existing-policy features after validation."""
        return self.frozen_policy.head_features(windows, masks)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> LateTrainingOutputs:
        """Expose ungated late output for fitting without accepting labels as inputs."""
        with torch.no_grad():
            features, logits, joints, lengths = self.frozen_policy._frozen_outputs(windows, masks)
            early_correction = self.frozen_policy.correction_head(features)
        late_correction = self.late_head(features)
        return LateTrainingOutputs(features, logits, joints, early_correction, late_correction, lengths)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep early behavior; apply the late head only at H>=4 and predicted MOVE.

        Nonfinite selected output is returned for the existing evaluator to
        classify. Nonfinite unused corrections do not change other actions or
        the other history region. Nothing is clipped or replaced.
        """
        out = self.training_outputs(windows, masks)
        is_move = out.action_logits.argmax(dim=1) == motion_policy.MOVE
        early_applies = (out.history_lengths < MIN_LATE_HISTORY) & is_move
        late_applies = (out.history_lengths >= MIN_LATE_HISTORY) & is_move
        existing = torch.where(early_applies.unsqueeze(1), out.base_joints + out.early_correction, out.base_joints)
        joints = torch.where(late_applies.unsqueeze(1), out.base_joints + out.late_correction, existing)
        return out.action_logits, joints


def initialize_from_current(source: current.SceneCurrentObservationCorrectionPolicy) -> SceneLateHistoryCorrectionPolicy:
    """Copy and freeze the current policy independently; initialize 584 values to zero."""
    if type(source) is not current.SceneCurrentObservationCorrectionPolicy:
        raise ValueError("source must be the existing 145-input current-observation correction policy")
    # Reuse its complete checkpoint validation without sharing any source tensor.
    payload = current.checkpoint_payload(source)
    parameter = source.correction_head.weight
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneLateHistoryCorrectionPolicy()
    model = model.to(device=parameter.device, dtype=parameter.dtype)
    model.frozen_policy.load_state_dict(payload["state_dict"], strict=True)
    model.frozen_policy.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract() -> dict:
    """Keep this model distinct from both previously published formats."""
    return {"format": CHECKPOINT_FORMAT, "architecture": "late_history_current_observation_correction_mlp",
            "frozen_policy_contract": current.checkpoint_contract(), "history_length": current.HISTORY_LENGTH,
            "external_feature_count": current.FEATURE_COUNT, "hidden_size": current.HIDDEN_SIZE,
            "head_feature_count": HEAD_FEATURE_COUNT, "joint_count": 4, "joint_scale_degrees": 180.0,
            "head_input": "frozen_encoding_64_then_current_observation_without_moving_81",
            "early_gate": "history_1_to_3_and_predicted_move_existing_head",
            "late_gate": "history_4_to_8_and_predicted_move_new_head",
            "output": "absolute_base_plus_applicable_correction_no_clipping"}


def checkpoint_payload(model: SceneLateHistoryCorrectionPolicy) -> dict:
    """Return independent CPU tensors; caller binds training provenance separately."""
    if type(model) is not SceneLateHistoryCorrectionPolicy:
        raise ValueError("late history correction policy required")
    if any(parameter.requires_grad for parameter in model.frozen_policy.parameters()):
        raise ValueError("entire existing policy must remain frozen")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneLateHistoryCorrectionPolicy:
    """Validate this explicit format and restore the original policy frozen."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("late history correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "late_head.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["late_head.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneLateHistoryCorrectionPolicy().to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != weight.dtype
                or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.frozen_policy.requires_grad_(False)
    model.eval()
    return model
