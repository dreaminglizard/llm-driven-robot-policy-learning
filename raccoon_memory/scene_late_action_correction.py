"""Trainable late action correction and joint head on the frozen BC policy.

The original 82-feature windows and masks remain the only policy inputs.
The existing frozen core removes ``moving`` internally.  The new action head
and copied late joint head both use only the final valid 81-feature frame.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from raccoon_memory import motion_policy
from raccoon_memory import scene_current_late_nonlinear as late
from raccoon_memory import scene_current_observation_correction as current


MIN_LATE_HISTORY = late.MIN_LATE_HISTORY
CHECKPOINT_FORMAT = "raccoon-scene-late-action-joint-correction.v1"


class LateActionOutputs(NamedTuple):
    """Values from one decision; the ungated joint value is for supervision."""

    features81: torch.Tensor
    base_logits: torch.Tensor
    action_correction: torch.Tensor
    action_logits: torch.Tensor
    base_joints: torch.Tensor
    early_correction: torch.Tensor
    late_correction: torch.Tensor
    late_ungated_joints: torch.Tensor
    history_lengths: torch.Tensor
    action_correction_gate: torch.Tensor
    early_gate: torch.Tensor
    late_gate: torch.Tensor
    joints: torch.Tensor


class SceneLateActionCorrectionPolicy(nn.Module):
    """Freeze the prior core; train its copied late joint and new action heads."""

    def __init__(self):
        super().__init__()
        current._check_contract()
        self.architecture = "late_action_joint_correction_mlp"
        self.history_length = current.HISTORY_LENGTH
        self.feature_count = current.FEATURE_COUNT
        self.hidden_size = current.HIDDEN_SIZE

        # Construction must not alter the caller's RNG.  The initialization
        # helper below replaces the independently constructed joint weights.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.frozen_policy = current.SceneCurrentObservationCorrectionPolicy()
            self.late_head = nn.Sequential(
                nn.Linear(current.CURRENT_FEATURE_COUNT, late.LATE_HIDDEN_SIZE, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(late.LATE_HIDDEN_SIZE, 4, dtype=torch.float32),
            )
            self.action_head = nn.Linear(current.CURRENT_FEATURE_COUNT, len(motion_policy.ACTIONS), dtype=torch.float32)
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
        self.frozen_policy.requires_grad_(False)
        self.frozen_policy.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_policy.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.frozen_policy.requires_grad_(False)
        return self

    def head_features(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return the validated last frame without its ``moving`` column."""
        return self.frozen_policy.head_features(windows, masks)[:, current.HIDDEN_SIZE:]

    def diagnostics(self, windows: torch.Tensor, masks: torch.Tensor) -> LateActionOutputs:
        """Compute corrected actions and their corresponding joint gate once."""
        with torch.no_grad():
            features145, base_logits, base_joints, lengths = self.frozen_policy._frozen_outputs(windows, masks)
            early_correction = self.frozen_policy.correction_head(features145)
            features81 = features145[:, current.HIDDEN_SIZE:]

        action_correction = self.action_head(features81)
        late_correction = self.late_head(features81)
        action_gate = lengths >= MIN_LATE_HISTORY
        action_logits = torch.where(action_gate.unsqueeze(1), base_logits + action_correction, base_logits)
        is_move = action_logits.argmax(dim=1) == motion_policy.MOVE
        early_gate = (lengths < MIN_LATE_HISTORY) & is_move
        late_gate = action_gate & is_move
        late_ungated_joints = base_joints + late_correction
        original_joints = torch.where(early_gate.unsqueeze(1), base_joints + early_correction, base_joints)
        joints = torch.where(late_gate.unsqueeze(1), late_ungated_joints, original_joints)
        return LateActionOutputs(features81, base_logits, action_correction, action_logits, base_joints,
                                 early_correction, late_correction, late_ungated_joints, lengths,
                                 action_gate, early_gate, late_gate, joints)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> LateActionOutputs:
        """Return the same action scores and ungated teacher-MOVE proposal."""
        return self.diagnostics(windows, masks)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.diagnostics(windows, masks)
        return out.action_logits, out.joints


def initialize_from_late(source: late.SceneCurrentLateNonlinearPolicy) -> SceneLateActionCorrectionPolicy:
    """Independently copy the prior frozen policy and learned late joint head."""
    if type(source) is not late.SceneCurrentLateNonlinearPolicy:
        raise ValueError("source must be the existing current-observation late policy")
    # The prior contract validates all tensor shapes, dtypes and finite values.
    payload = late.checkpoint_payload(source)
    parameter = source.late_head[0].weight
    model = SceneLateActionCorrectionPolicy().to(device=parameter.device, dtype=parameter.dtype)
    state = payload["state_dict"]
    frozen = {key.removeprefix("frozen_policy."): value for key, value in state.items()
              if key.startswith("frozen_policy.")}
    joint = {key.removeprefix("late_head."): value for key, value in state.items()
             if key.startswith("late_head.")}
    model.frozen_policy.load_state_dict(frozen, strict=True)
    model.late_head.load_state_dict(joint, strict=True)
    model.frozen_policy.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract() -> dict:
    """Describe the distinct action/joint correction checkpoint format."""
    current._check_contract()
    return {
        "format": CHECKPOINT_FORMAT,
        "architecture": "late_action_joint_correction_mlp",
        "source_late_policy_contract": late.checkpoint_contract(),
        "history_length": current.HISTORY_LENGTH,
        "external_feature_count": current.FEATURE_COUNT,
        "current_feature_columns": list(current.CURRENT_COLUMNS),
        "current_feature_count": current.CURRENT_FEATURE_COUNT,
        "actions": list(motion_policy.ACTIONS),
        "action_head": "linear_81_4_weight_and_bias_initially_zero",
        "late_joint_head": "independent_copy_of_source_linear_81_64_relu_linear_64_4",
        "action_gate": "history_4_to_8_before_corrected_argmax",
        "joint_gate": "history_4_to_8_and_corrected_predicted_move",
        "early_gate": "history_1_to_3_and_original_predicted_move",
        "joint_supervision": "ungated_base_plus_late_for_teacher_move_history_4_to_8",
        "joint_scale_degrees": 180.0,
        "output": "absolute_base_plus_applicable_correction_no_clipping",
    }


def checkpoint_payload(model: SceneLateActionCorrectionPolicy) -> dict:
    """Return independent CPU tensors after checking the frozen core."""
    if type(model) is not SceneLateActionCorrectionPolicy:
        raise ValueError("late action and joint correction policy required")
    if any(p.requires_grad for p in model.frozen_policy.parameters()):
        raise ValueError("the entire existing core must remain frozen")
    if any(module.training for module in model.frozen_policy.modules()):
        raise ValueError("the entire existing core must remain in evaluation mode")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneLateActionCorrectionPolicy:
    """Strictly restore this format without sharing stored tensor storage."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("late action and joint correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "action_head.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["action_head.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    model = SceneLateActionCorrectionPolicy().to(dtype=weight.dtype)
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
