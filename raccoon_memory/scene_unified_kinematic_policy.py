"""Trainable BC1020-based policy with observable kinematic context.

One fixed model produces all four action scores and absolute joint proposals.
It accepts the existing causal 8x82 observation window and an independently
verified controller target, in degrees.  The added geometric features use
nominal registered slot/level information and current joint FK only.  This
module does not read objects, contacts, privileged state, or execute actions.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import (motion_policy, scene_current_late_nonlinear as baseline,
                            scene_current_observation_correction as current,
                            scene_fk_loss, scene_no_moving_policy, scene_relative_policy)

CHECKPOINT_FORMAT = "raccoon-unified-bc1020-kinematic-policy.v1"
JOINT_BRANCH_CHECKPOINT_FORMAT = "raccoon-unified-bc1020-kinematic-joint-branch.v1"
HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE, CONTEXT_COUNT = 8, 82, 64, 10
JOINT_BRANCH_INPUT_COUNT, JOINT_BRANCH_HIDDEN_SIZE = 22, 32
PREVIOUS_TARGET_LIMIT_TOLERANCE_DEG = 1e-7


class UnifiedOutputs(NamedTuple):
    action_logits: torch.Tensor
    absolute_joints: torch.Tensor
    ungated_move_joints: torch.Tensor
    history_lengths: torch.Tensor
    nominal_relative_xyz: torch.Tensor
    previous_target_normalized: torch.Tensor


class UnifiedKinematicPolicy(nn.Module):
    """The entire copied BC computation is trainable; no policy is switched.

    The ten new context values enter the first encoder preactivation through a
    bias-free zero-start projection.  This preserves the original calculation
    exactly at initialization without freezing its encoder or output heads.
    The optional joint branch adds a zero-start 22-32-4 observable kinematic
    correction to the MOVE proposal only. The caller must establish that the
    previous control value came from a returned command or a validated initial
    control.
    """

    def __init__(self, *, joint_branch=False):
        super().__init__()
        if type(joint_branch) is not bool:
            raise ValueError("joint_branch must be an explicit boolean")
        baseline.current._check_contract()
        self.architecture = ("unified_bc1020_kinematic_mlp64_joint_branch32" if joint_branch
                             else "unified_bc1020_kinematic_mlp64")
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.base = scene_no_moving_policy.SceneNoMovingPolicy()
        self.early_head = nn.Linear(current.HEAD_FEATURE_COUNT, 4)
        self.late_head = nn.Sequential(nn.Linear(current.CURRENT_FEATURE_COUNT, HIDDEN_SIZE), nn.ReLU(),
                                       nn.Linear(HIDDEN_SIZE, 4))
        self.context_input = nn.Linear(CONTEXT_COUNT, HIDDEN_SIZE, bias=False)
        nn.init.zeros_(self.context_input.weight)
        self.register_buffer("_current_columns", torch.tensor(current.CURRENT_COLUMNS, dtype=torch.long), persistent=False)
        self.joint_branch = None
        if joint_branch:
            self.joint_branch = nn.Sequential(nn.Linear(JOINT_BRANCH_INPUT_COUNT, JOINT_BRANCH_HIDDEN_SIZE),
                                              nn.ReLU(), nn.Linear(JOINT_BRANCH_HIDDEN_SIZE, 4))
            nn.init.zeros_(self.joint_branch[2].weight)
            nn.init.zeros_(self.joint_branch[2].bias)

    def _context(self, windows, masks, previous_target_degrees):
        parameter = self.base.encoder[0].weight
        if (not isinstance(windows, torch.Tensor) or windows.dtype != parameter.dtype or
                windows.device != parameter.device):
            raise ValueError("window dtype and device must match the policy")
        extended = scene_relative_policy.extend_scene_features(windows, masks)
        if (not isinstance(previous_target_degrees, torch.Tensor) or
                previous_target_degrees.shape != (windows.shape[0], 4) or
                previous_target_degrees.dtype != parameter.dtype or
                previous_target_degrees.device != parameter.device or
                not bool(torch.isfinite(previous_target_degrees).all())):
            raise ValueError("previous applied target must be finite degrees with shape [batch,4]")
        bounds = previous_target_degrees.new_tensor(JOINT_LIMITS)
        if bool(((previous_target_degrees < bounds[:, 0] - PREVIOUS_TARGET_LIMIT_TOLERANCE_DEG) |
                 (previous_target_degrees > bounds[:, 1] + PREVIOUS_TARGET_LIMIT_TOLERANCE_DEG)).any()):
            raise ValueError("previous applied target exceeds joint limits")
        relative = extended[:, -1, FEATURE_COUNT:]
        normalized = previous_target_degrees / motion_policy.JOINT_SCALE_DEG
        return relative, normalized

    def training_outputs(self, windows, masks, previous_target_degrees):
        relative, prior = self._context(windows, masks, previous_target_degrees)
        lengths = self.base._validate(windows, masks)
        original = self.base._project_validated(windows, masks)
        first = self.base.encoder[0](original) + self.context_input(torch.cat((relative, prior), dim=1))
        encoded = self.base.encoder[3](self.base.encoder[2](self.base.encoder[1](first)))
        logits, base_joints = self.base.action_head(encoded), self.base.joint_head(encoded)
        current81 = windows[:, -1, :].index_select(1, self._current_columns)
        early = self.early_head(torch.cat((encoded, current81), dim=1))
        late = self.late_head(current81)
        ungated = base_joints + torch.where((lengths < baseline.MIN_LATE_HISTORY).unsqueeze(1), early, late)
        if self.joint_branch is not None:
            observation = windows[:, -1, :]
            joint_context = torch.cat((relative, prior, observation[:, :4], observation[:, 6:14]), dim=1)
            ungated = ungated + self.joint_branch(joint_context)
        predicted_move = logits.argmax(dim=1) == motion_policy.MOVE
        joints = torch.where(predicted_move.unsqueeze(1), ungated, base_joints)
        return UnifiedOutputs(logits, joints, ungated, lengths, relative, prior)

    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        out = self.training_outputs(windows, masks, previous_target_degrees)
        return out.action_logits, out.absolute_joints

    def forward(self, windows, masks, previous_target_degrees=None):
        if previous_target_degrees is None:
            raise ValueError("previous applied target is required; it must not be imputed")
        return self.forward_with_control_state(windows, masks, previous_target_degrees)


def initialize_from_bc1020(source, *, joint_branch=False):
    """Copy all BC1020 tensors into independent, trainable model parameters."""
    if type(source) is not baseline.SceneCurrentLateNonlinearPolicy:
        raise ValueError("BC1020 current-late policy is required")
    parameters = tuple(source.parameters())
    if not parameters or any(p.dtype != parameters[0].dtype or p.device != parameters[0].device or
                             not bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError("source tensors must be finite and have one dtype and device")
    with torch.random.fork_rng(devices=[]):
        result = UnifiedKinematicPolicy(joint_branch=joint_branch).to(dtype=parameters[0].dtype,
                                                                       device=parameters[0].device)
    result.base.load_state_dict(source.frozen_policy.base.state_dict(), strict=True)
    result.early_head.load_state_dict(source.frozen_policy.correction_head.state_dict(), strict=True)
    result.late_head.load_state_dict(source.late_head.state_dict(), strict=True)
    result.requires_grad_(True)
    result.train(source.training)
    return result


def supervised_loss(outputs, actions, target_joints_normalized, *, joint_weight, fk_weight):
    """Action CE plus teacher-MOVE ungated joint and nominal-FK losses."""
    if not isinstance(outputs, UnifiedOutputs):
        raise ValueError("outputs must come from the same policy call")
    n = outputs.action_logits.shape[0]
    if (not n or outputs.action_logits.shape != (n, 4) or outputs.ungated_move_joints.shape != (n, 4) or
            not isinstance(actions, torch.Tensor) or actions.shape != (n,) or actions.dtype != torch.long or
            actions.device != outputs.action_logits.device or bool(((actions < 0) | (actions > 3)).any()) or
            not isinstance(target_joints_normalized, torch.Tensor) or target_joints_normalized.shape != (n, 4) or
            target_joints_normalized.dtype != outputs.ungated_move_joints.dtype or
            target_joints_normalized.device != outputs.ungated_move_joints.device or
            type(joint_weight) not in (int, float) or type(fk_weight) not in (int, float) or
            not 0 <= joint_weight < float("inf") or not 0 <= fk_weight < float("inf")):
        raise ValueError("invalid action, joint, device or explicit loss-weight contract")
    move = actions == motion_policy.MOVE
    if (not bool(torch.isfinite(outputs.action_logits).all()) or
            not bool(torch.isfinite(outputs.ungated_move_joints).all()) or
            not bool(torch.isfinite(target_joints_normalized[move]).all())):
        raise ValueError("nonfinite supervised output or target")
    ce = F.cross_entropy(outputs.action_logits, actions)
    mse = (F.mse_loss(outputs.ungated_move_joints[move], target_joints_normalized[move]) if bool(move.any())
           else outputs.ungated_move_joints.sum() * 0.0)
    fk_total, fk_position, fk_axis = scene_fk_loss.move_fk_loss(outputs.ungated_move_joints,
                                                                 target_joints_normalized, move)
    return {"total": ce + float(joint_weight) * mse + float(fk_weight) * fk_total,
            "action_ce": ce, "joint_mse": mse, "nominal_fk": fk_total,
            "nominal_fk_position": fk_position, "nominal_fk_axis": fk_axis,
            "teacher_move_count": int(move.sum().item())}


def checkpoint_contract(*, joint_branch=False):
    if type(joint_branch) is not bool:
        raise ValueError("joint_branch must be an explicit boolean")
    contract = {"format": CHECKPOINT_FORMAT, "initial_source": baseline.CHECKPOINT_FORMAT,
            "architecture": "trainable_bc1020_computation_plus_zero_start_encoder_context",
            "window": [HISTORY_LENGTH, FEATURE_COUNT], "history_mask": "left_padded_bool",
            "input_context": "source_and_destination_nominal_center_minus_current_fk_xyz_div300_then_previous_target_degrees_div180",
            "previous_target_provenance": "caller_verifies_returned_servo_or_validated_initial_control",
            "missing_previous_target": "reject", "action_names": list(motion_policy.ACTIONS),
            "joint_output": "absolute_degrees_div180", "teacher_move_loss": "ungated_by_predicted_action"}
    if joint_branch:
        contract["format"] = JOINT_BRANCH_CHECKPOINT_FORMAT
        contract["architecture"] = "trainable_bc1020_kinematic_plus_zero_start_joint_branch"
        contract["joint_branch"] = {
            "architecture": [JOINT_BRANCH_INPUT_COUNT, JOINT_BRANCH_HIDDEN_SIZE, 4],
            "input": ["source_and_destination_nominal_relative_xyz_div300[6]",
                      "previous_applied_target_degrees_div180[4]",
                      "current_joint_degrees_div180[4]", "current_window_columns_6_to_13[8]"],
            "output": "move_joint_proposal_correction_degrees_div180",
        }
    return contract


def checkpoint_payload(model):
    if type(model) is not UnifiedKinematicPolicy:
        raise ValueError("unified policy is required")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("all model parameters must be trainable")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("model tensors must be finite")
    return {"contract": checkpoint_contract(joint_branch=model.joint_branch is not None), "state_dict": state}


def load_checkpoint(payload, *, device="cpu"):
    if not isinstance(payload, dict):
        raise ValueError("unified checkpoint contract differs")
    if payload.get("contract") == checkpoint_contract():
        joint_branch = False
    elif payload.get("contract") == checkpoint_contract(joint_branch=True):
        joint_branch = True
    else:
        raise ValueError("unified checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "base.encoder.0.weight" not in state:
        raise ValueError("unified state dictionary is required")
    weight = state["base.encoder.0.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu" or
            weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]):
        model = UnifiedKinematicPolicy(joint_branch=joint_branch).to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected) or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != weight.dtype or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())
            for name, value in state.items()):
        raise ValueError("checkpoint tensor names, shapes, dtype or values differ")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.requires_grad_(True)
    model.eval()
    return model
