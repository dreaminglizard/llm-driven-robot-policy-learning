"""Late action and joint corrections with a frozen history representation.

The policy retains the existing 82-feature window, masks, validation and
execution gates.  The 64 history features come from the same frozen encoder
call as the original action logits and joint output.  The new, bias-free
projections start at zero and are separate parameters from the copied heads.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from raccoon_memory import motion_policy
from raccoon_memory import scene_current_observation_correction as current
from raccoon_memory import scene_late_action_correction as previous


MIN_LATE_HISTORY = previous.MIN_LATE_HISTORY
CHECKPOINT_FORMAT = "raccoon-scene-late-history-action-joint-correction.v1"


class LateHistoryActionOutputs(NamedTuple):
    """Values from one policy call, including its ungated training proposal."""

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
    history_features: torch.Tensor
    action_history_correction: torch.Tensor
    joint_history_projection: torch.Tensor


class SceneLateHistoryActionCorrectionPolicy(previous.SceneLateActionCorrectionPolicy):
    """Retain both old heads and add separate trainable history projections."""

    def __init__(self):
        super().__init__()
        self.architecture = "late_history_action_joint_correction_mlp"
        # Do not advance the caller's random generator: both matrices are zero.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.action_history = nn.Linear(current.HIDDEN_SIZE, len(motion_policy.ACTIONS), bias=False,
                                            dtype=torch.float32)
            self.joint_history = nn.Linear(current.HIDDEN_SIZE, current.HIDDEN_SIZE, bias=False,
                                           dtype=torch.float32)
            nn.init.zeros_(self.action_history.weight)
            nn.init.zeros_(self.joint_history.weight)
        # The inherited frozen core is constructed from modules that may use
        # the global default dtype.  Match every floating parameter to the
        # existing late head even when the caller changed that default.
        self.to(dtype=self.late_head[0].weight.dtype)

    def diagnostics(self, windows: torch.Tensor, masks: torch.Tensor) -> LateHistoryActionOutputs:
        """Use one validated frozen call for the base outputs and both inputs."""
        with torch.no_grad():
            features145, base_logits, base_joints, lengths = self.frozen_policy._frozen_outputs(windows, masks)
            early_correction = self.frozen_policy.correction_head(features145)
            history_features = features145[:, :current.HIDDEN_SIZE]
            features81 = features145[:, current.HIDDEN_SIZE:]

        action_history_correction = self.action_history(history_features)
        joint_history_projection = self.joint_history(history_features)
        action_correction = self.action_head(features81) + action_history_correction
        joint_hidden = self.late_head[0](features81) + joint_history_projection
        late_correction = self.late_head[2](self.late_head[1](joint_hidden))
        action_gate = lengths >= MIN_LATE_HISTORY
        action_logits = torch.where(action_gate.unsqueeze(1), base_logits + action_correction, base_logits)
        is_move = action_logits.argmax(dim=1) == motion_policy.MOVE
        early_gate = (lengths < MIN_LATE_HISTORY) & is_move
        late_gate = action_gate & is_move
        late_ungated_joints = base_joints + late_correction
        original_joints = torch.where(early_gate.unsqueeze(1), base_joints + early_correction, base_joints)
        joints = torch.where(late_gate.unsqueeze(1), late_ungated_joints, original_joints)
        return LateHistoryActionOutputs(
            features81, base_logits, action_correction, action_logits, base_joints,
            early_correction, late_correction, late_ungated_joints, lengths,
            action_gate, early_gate, late_gate, joints, history_features,
            action_history_correction, joint_history_projection,
        )


def initialize_from_action_policy(
    source: previous.SceneLateActionCorrectionPolicy,
) -> SceneLateHistoryActionCorrectionPolicy:
    """Copy the 83 comparison's zero-action starting policy independently.

    The caller must additionally bind its checkpoint to the intended BC 1020
    provenance.  A trained, nonzero action correction is rejected here.
    """
    if type(source) is not previous.SceneLateActionCorrectionPolicy:
        raise ValueError("source must be the existing late action correction policy")
    payload = previous.checkpoint_payload(source)
    state = payload["state_dict"]
    if (torch.count_nonzero(state["action_head.weight"]).item() != 0
            or torch.count_nonzero(state["action_head.bias"]).item() != 0):
        raise ValueError("source action correction must be the zero-initialized starting policy")
    parameter = source.late_head[0].weight
    model = SceneLateHistoryActionCorrectionPolicy().to(device=parameter.device, dtype=parameter.dtype)
    target = model.state_dict()
    extra = {"action_history.weight", "joint_history.weight"}
    if set(target) - set(state) != extra or set(state) - set(target):
        raise ValueError("source and new parameter names differ")
    model.load_state_dict({**state, **{name: target[name] for name in extra}}, strict=True)
    model.frozen_policy.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract() -> dict:
    """Bind the full output, history source and optimizer-facing structure."""
    current._check_contract()
    return {
        "format": CHECKPOINT_FORMAT,
        "architecture": "late_history_action_joint_correction_mlp",
        "source_policy_contract": previous.checkpoint_contract(),
        "history_source": "first_64_frozen_features_from_the_same_validated_encoder_call",
        "current_source": "last_valid_frame_without_moving_81",
        "action_history": "bias_free_linear_64_4_zero_initialized",
        "joint_history": "bias_free_linear_64_64_zero_initialized_added_before_existing_relu",
        "copied_late_head": "linear_81_64_relu_linear_64_4",
        "action_gate": "history_4_to_8_before_corrected_argmax",
        "joint_gate": "history_4_to_8_and_same_call_corrected_predicted_move",
        "early_gate": "history_1_to_3_and_original_predicted_move",
        "joint_supervision": "ungated_base_plus_late_for_teacher_move_history_4_to_8",
        "joint_scale_degrees": 180.0,
        "output": "absolute_base_plus_applicable_correction_no_clipping",
    }


def checkpoint_payload(model: SceneLateHistoryActionCorrectionPolicy) -> dict:
    """Return independently owned CPU tensors for a strictly bound policy."""
    if type(model) is not SceneLateHistoryActionCorrectionPolicy:
        raise ValueError("late history action and joint policy required")
    if any(p.requires_grad for p in model.frozen_policy.parameters()):
        raise ValueError("the entire existing core must remain frozen")
    if any(module.training for module in model.frozen_policy.modules()):
        raise ValueError("the entire existing core must remain in evaluation mode")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneLateHistoryActionCorrectionPolicy:
    """Reject incompatible, partial, mixed-dtype or nonfinite checkpoints."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("late history action and joint checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "action_head.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["action_head.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    model = SceneLateHistoryActionCorrectionPolicy().to(dtype=weight.dtype)
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
