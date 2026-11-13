"""One trainable policy with teacher-target partitions and MOVE corrections.

The fixed centers are caller-provided training-target labels, in joint degrees
divided by 180. The original variants use them only to partition supervision;
their MOVE proposals add learned corrections to the common policy proposal.
The optional center-residual variant instead adds each correction directly to
its fixed center, which is then the basis of the student's selected proposal.
Centers are not observations or live object positions.

This module returns raw proposals before the existing one-degree guard. It
does not implement RL sampling. Future stochastic collection must record its
actual sampled center and raw joint sample and evaluate their conditional
probabilities; a guarded command is not that raw sample.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_fk_loss, scene_unified_kinematic_policy as unified

CHECKPOINT_FORMAT = "raccoon-unified-codebook64-policy.v1"
NONLINEAR_CHECKPOINT_FORMAT = "raccoon-unified-codebook64-nonlinear-selector.v1"
CENTER_RESIDUAL_CHECKPOINT_FORMAT = "raccoon-unified-codebook64-center-residual.v1"
CENTER_COUNT = 64
KINEMATIC_FEATURE_COUNT = 22
HEAD_FEATURE_COUNT = unified.HIDDEN_SIZE + KINEMATIC_FEATURE_COUNT
SELECTOR_HIDDEN_SIZE = 128


class CodebookOutputs(NamedTuple):
    action_logits: torch.Tensor
    absolute_joints: torch.Tensor
    ungated_move_joints: torch.Tensor
    history_lengths: torch.Tensor
    nominal_relative_xyz: torch.Tensor
    previous_target_normalized: torch.Tensor
    center_logits: torch.Tensor
    selected_center: torch.Tensor
    move_proposals: torch.Tensor
    base_ungated_move_joints: torch.Tensor


def _validate_centers(centers, *, dtype=None, device=None):
    if (not isinstance(centers, torch.Tensor) or centers.shape != (CENTER_COUNT, 4)
            or centers.dtype not in (torch.float32, torch.float64)
            or (dtype is not None and centers.dtype != dtype)
            or (device is not None and centers.device != device)
            or not bool(torch.isfinite(centers).all())):
        raise ValueError("centers must be finite floating [64,4] and match policy dtype and device")
    degrees = centers * motion_policy.JOINT_SCALE_DEG
    bounds = centers.new_tensor(JOINT_LIMITS)
    tolerance = unified.PREVIOUS_TARGET_LIMIT_TOLERANCE_DEG
    if bool(((degrees < bounds[:, 0] - tolerance) | (degrees > bounds[:, 1] + tolerance)).any()):
        raise ValueError("training target centers must be within joint limits")
    if len(torch.unique(centers.detach(), dim=0)) != CENTER_COUNT:
        raise ValueError("training target centers must be distinct")


def nearest_center_indices(targets_normalized, centers_normalized):
    """Choose nearest fixed training center; ties choose the first index."""
    if (not isinstance(targets_normalized, torch.Tensor) or targets_normalized.ndim != 2
            or targets_normalized.shape[1] != 4 or targets_normalized.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(targets_normalized).all())):
        raise ValueError("MOVE targets must be finite floating [batch,4]")
    _validate_centers(centers_normalized, dtype=targets_normalized.dtype, device=targets_normalized.device)
    distances = (targets_normalized[:, None, :] - centers_normalized[None, :, :]).square().sum(dim=2)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("target-to-center distances must be finite")
    return distances.argmin(dim=1)


def _add_preserving_zero(expanded, corrections):
    """Add broadcastable tensors while preserving signed-zero baseline bits.

    Both branches have the derivative of base + correction. This retains
    gradients for zero-initialized heads, including when a baseline is -0.0.
    """
    summed = expanded + corrections
    negative_zero = (expanded == 0) & torch.signbit(expanded) & (corrections == 0)
    reversed_sum = -((-expanded) - corrections)
    return torch.where(negative_zero, reversed_sum, summed)


def _add_corrections(base, corrections):
    return _add_preserving_zero(base.unsqueeze(1), corrections)


class CodebookPolicy(unified.UnifiedKinematicPolicy):
    """Shared unified policy plus 64 learned scores and four corrections each."""

    def __init__(self, centers_normalized, *, nonlinear_selector=False, center_residual=False):
        if type(nonlinear_selector) is not bool or type(center_residual) is not bool:
            raise ValueError("policy variant selectors must be explicit booleans")
        if center_residual and not nonlinear_selector:
            raise ValueError("center-residual output requires a nonlinear selector")
        _validate_centers(centers_normalized)
        super().__init__()
        self.architecture = ("unified_kinematic_target_partition64_center_residual" if center_residual else
                             "unified_kinematic_target_partition64_nonlinear_selector" if nonlinear_selector else
                             "unified_kinematic_target_partition64")
        self.center_residual = center_residual
        self.register_buffer("centers_normalized", centers_normalized.detach().clone())
        if center_residual:
            self.register_buffer("_center_residual_contract_marker", centers_normalized.new_ones(1))
        self.center_head = nn.Linear(HEAD_FEATURE_COUNT, CENTER_COUNT)
        self.correction_head = nn.Linear(HEAD_FEATURE_COUNT, CENTER_COUNT * 4)
        nn.init.zeros_(self.center_head.weight)
        nn.init.zeros_(self.center_head.bias)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self.selector_residual = None
        if nonlinear_selector:
            self.selector_residual = nn.Sequential(nn.Linear(HEAD_FEATURE_COUNT, SELECTOR_HIDDEN_SIZE), nn.ReLU(),
                                                   nn.Linear(SELECTOR_HIDDEN_SIZE, CENTER_COUNT))
            nn.init.zeros_(self.selector_residual[2].weight)
            nn.init.zeros_(self.selector_residual[2].bias)

    def selector_parameters(self):
        """Return the trainable selector tensors without changing their flags."""
        linear = tuple(self.center_head.parameters())
        return linear if self.selector_residual is None else linear + tuple(self.selector_residual.parameters())

    def _shared_features(self, windows, masks, previous_target_degrees):
        relative, prior = self._context(windows, masks, previous_target_degrees)
        lengths = self.base._validate(windows, masks)
        original = self.base._project_validated(windows, masks)
        first = self.base.encoder[0](original) + self.context_input(torch.cat((relative, prior), dim=1))
        encoded = self.base.encoder[3](self.base.encoder[2](self.base.encoder[1](first)))
        observation = windows[:, -1, :]
        head_input = torch.cat((encoded, relative, prior, observation[:, :4], observation[:, 6:14]), dim=1)
        return relative, prior, lengths, encoded, head_input

    def correction_features(self, windows, masks, previous_target_degrees):
        """Return the exact 86 observable values consumed by the correction head."""
        return self._shared_features(windows, masks, previous_target_degrees)[-1]

    def training_outputs(self, windows, masks, previous_target_degrees):
        # The common encoder and the public correction-feature method share one
        # computation; no teacher target or physical object state is read.
        relative, prior, lengths, encoded, head_input = self._shared_features(windows, masks, previous_target_degrees)
        logits, base_joints = self.base.action_head(encoded), self.base.joint_head(encoded)
        current81 = windows[:, -1, :].index_select(1, self._current_columns)
        early = self.early_head(torch.cat((encoded, current81), dim=1))
        late = self.late_head(current81)
        base_move = base_joints + torch.where((lengths < unified.baseline.MIN_LATE_HISTORY).unsqueeze(1), early, late)
        center_logits = self.center_head(head_input)
        if self.selector_residual is not None:
            residual = self.selector_residual(head_input)
            center_logits = _add_corrections(center_logits, residual.unsqueeze(1)).squeeze(1)
        corrections = self.correction_head(head_input).reshape(-1, CENTER_COUNT, 4)
        proposals = (_add_preserving_zero(self.centers_normalized.unsqueeze(0), corrections) if self.center_residual
                     else _add_corrections(base_move, corrections))
        if not all(bool(torch.isfinite(value).all()) for value in (logits, base_joints, base_move, center_logits, proposals)):
            raise ValueError("policy logits and raw MOVE proposals must be finite")
        selected = center_logits.argmax(dim=1)
        rows = torch.arange(len(windows), device=windows.device)
        student_move = proposals[rows, selected]
        joints = torch.where((logits.argmax(dim=1) == motion_policy.MOVE).unsqueeze(1), student_move, base_joints)
        return CodebookOutputs(logits, joints, student_move, lengths, relative, prior,
                               center_logits, selected, proposals, base_move)


def initialize_from_unified(source, centers_normalized):
    """Copy a branch-free unified policy and add independently trainable heads."""
    if type(source) is not unified.UnifiedKinematicPolicy or source.joint_branch is not None:
        raise ValueError("the branch-free unified policy is required")
    parameters = tuple(source.parameters())
    if (not parameters or parameters[0].dtype not in (torch.float32, torch.float64)
            or any(p.dtype != parameters[0].dtype or p.device != parameters[0].device
                   or not bool(torch.isfinite(p).all()) for p in parameters)):
        raise ValueError("source tensors must be finite and have one floating dtype and device")
    _validate_centers(centers_normalized, dtype=parameters[0].dtype, device=parameters[0].device)
    with torch.random.fork_rng(devices=[]):
        model = CodebookPolicy(centers_normalized).to(dtype=parameters[0].dtype, device=parameters[0].device)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    expected_missing = {"centers_normalized", "center_head.weight", "center_head.bias",
                        "correction_head.weight", "correction_head.bias"}
    if set(missing) != expected_missing or unexpected:
        raise ValueError("unified source tensors do not match the common policy")
    model.requires_grad_(True)
    model.train(source.training)
    return model


def with_nonlinear_selector(source):
    """Independently copy a linear codebook model and add a zero-start residual."""
    if type(source) is not CodebookPolicy or source.joint_branch is not None or source.selector_residual is not None:
        raise ValueError("a linear-selector codebook policy is required")
    parameters = tuple(source.parameters())
    if (not parameters or parameters[0].dtype not in (torch.float32, torch.float64)
            or any(p.dtype != parameters[0].dtype or p.device != parameters[0].device
                   or not bool(torch.isfinite(p).all()) for p in parameters)):
        raise ValueError("source tensors must be finite and have one floating dtype and device")
    _validate_centers(source.centers_normalized, dtype=parameters[0].dtype, device=parameters[0].device)
    with torch.random.fork_rng(devices=[]):
        model = CodebookPolicy(source.centers_normalized, nonlinear_selector=True).to(
            dtype=parameters[0].dtype, device=parameters[0].device)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if set(missing) != {"selector_residual.0.weight", "selector_residual.0.bias",
                        "selector_residual.2.weight", "selector_residual.2.bias"} or unexpected:
        raise ValueError("linear codebook source tensors do not match the policy")
    model.requires_grad_(True)
    model.train(source.training)
    return model


def with_center_residual(source):
    """Copy a nonlinear-selector model and reset only its correction head.

    Initial MOVE proposals become the fixed centers, not the source model's
    proposals. The source is not modified and all parameters remain trainable.
    """
    if (type(source) is not CodebookPolicy or source.joint_branch is not None
            or source.selector_residual is None or source.center_residual):
        raise ValueError("a common-proposal codebook policy with nonlinear selector is required")
    parameters = tuple(source.parameters())
    if (not parameters or parameters[0].dtype not in (torch.float32, torch.float64)
            or any(p.dtype != parameters[0].dtype or p.device != parameters[0].device
                   or not bool(torch.isfinite(p).all()) for p in parameters)):
        raise ValueError("source tensors must be finite and have one floating dtype and device")
    _validate_centers(source.centers_normalized, dtype=parameters[0].dtype, device=parameters[0].device)
    with torch.random.fork_rng(devices=[]):
        model = CodebookPolicy(source.centers_normalized, nonlinear_selector=True, center_residual=True).to(
            dtype=parameters[0].dtype, device=parameters[0].device)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if set(missing) != {"_center_residual_contract_marker"} or unexpected:
        raise ValueError("nonlinear-selector source tensors do not match the policy")
    with torch.no_grad():
        model.correction_head.weight.zero_()
        model.correction_head.bias.zero_()
    model.requires_grad_(True)
    model.train(source.training)
    return model


def supervised_loss(outputs, actions, target_joints_normalized, centers_normalized, *, joint_weight, fk_weight):
    """Action CE, teacher-center CE, and teacher-center MOVE joint/FK losses.

    Student-selected and teacher-center errors are returned separately as
    diagnostics; only the latter is used for joint and FK supervision.
    """
    if not isinstance(outputs, CodebookOutputs):
        raise ValueError("outputs must come from the codebook policy")
    n = outputs.action_logits.shape[0]
    if (not n or outputs.action_logits.shape != (n, 4) or outputs.center_logits.shape != (n, CENTER_COUNT)
            or outputs.move_proposals.shape != (n, CENTER_COUNT, 4)
            or outputs.ungated_move_joints.shape != (n, 4)
            or not isinstance(actions, torch.Tensor) or actions.shape != (n,) or actions.dtype != torch.long
            or actions.device != outputs.action_logits.device or bool(((actions < 0) | (actions > 3)).any())
            or not isinstance(target_joints_normalized, torch.Tensor) or target_joints_normalized.shape != (n, 4)
            or target_joints_normalized.dtype != outputs.ungated_move_joints.dtype
            or target_joints_normalized.device != outputs.ungated_move_joints.device
            or type(joint_weight) not in (int, float) or type(fk_weight) not in (int, float)
            or not 0 <= joint_weight < float("inf") or not 0 <= fk_weight < float("inf")):
        raise ValueError("invalid action, target, device or explicit loss-weight contract")
    _validate_centers(centers_normalized, dtype=outputs.ungated_move_joints.dtype,
                      device=outputs.ungated_move_joints.device)
    move = actions == motion_policy.MOVE
    if (not bool(torch.isfinite(outputs.action_logits).all()) or not bool(torch.isfinite(outputs.center_logits).all())
            or not bool(torch.isfinite(outputs.move_proposals).all())
            or not bool(torch.isfinite(target_joints_normalized[move]).all())):
        raise ValueError("nonfinite supervised output or MOVE target")
    action_ce = F.cross_entropy(outputs.action_logits, actions)
    move_count = int(move.sum().item())
    if move_count:
        targets = target_joints_normalized[move]
        teacher_center = nearest_center_indices(targets, centers_normalized)
        center_ce = F.cross_entropy(outputs.center_logits[move], teacher_center)
        proposals = outputs.move_proposals[move]
        teacher_joints = proposals[torch.arange(move_count, device=proposals.device), teacher_center]
        student_joints = outputs.ungated_move_joints[move]
        joint_mse = F.mse_loss(teacher_joints, targets)
        fk_total, fk_position, fk_axis = scene_fk_loss.move_fk_loss(
            teacher_joints, targets, torch.ones(move_count, dtype=torch.bool, device=targets.device))
        with torch.no_grad():
            student_mse = F.mse_loss(student_joints, targets)
            student_mae_deg = (student_joints - targets).abs().mean() * motion_policy.JOINT_SCALE_DEG
            teacher_mae_deg = (teacher_joints - targets).abs().mean() * motion_policy.JOINT_SCALE_DEG
            center_accuracy = (outputs.selected_center[move] == teacher_center).float().mean()
    else:
        center_ce = outputs.center_logits.sum() * 0.0
        joint_mse = outputs.move_proposals.sum() * 0.0
        fk_total = fk_position = fk_axis = joint_mse
        student_mse = student_mae_deg = teacher_mae_deg = center_accuracy = joint_mse.detach()
    return {"total": action_ce + center_ce + float(joint_weight) * joint_mse + float(fk_weight) * fk_total,
            "action_ce": action_ce, "center_ce": center_ce, "joint_mse": joint_mse,
            "nominal_fk": fk_total, "nominal_fk_position": fk_position, "nominal_fk_axis": fk_axis,
            "teacher_move_count": move_count, "student_selected_joint_mse": student_mse,
            "teacher_selected_joint_mse": joint_mse.detach(), "student_selected_joint_mae_deg": student_mae_deg,
            "teacher_selected_joint_mae_deg": teacher_mae_deg, "student_center_accuracy": center_accuracy}


def checkpoint_contract(*, nonlinear_selector=False, center_residual=False):
    if type(nonlinear_selector) is not bool or type(center_residual) is not bool:
        raise ValueError("policy variant selectors must be explicit booleans")
    if center_residual and not nonlinear_selector:
        raise ValueError("center-residual output requires a nonlinear selector")
    contract = {"format": CHECKPOINT_FORMAT, "base_contract": unified.checkpoint_contract(),
            "center_count": CENTER_COUNT, "center_provenance": "caller_supplies_fixed_training_move_targets_only",
            "center_units": "absolute_joint_degrees_div180", "center_role": "teacher_label_partition_only",
            "center_assignment": "nearest_normalized_joint_target_squared_l2_ties_first",
            "head_input": ["unified_history_encoding[64]", "nominal_relative_xyz_div300[6]",
                           "previous_applied_target_degrees_div180[4]", "current_joint_degrees_div180[4]",
                           "current_window_columns_6_to_13[8]"],
            "move_proposal": "unified_ungated_move_plus_learned_center_specific_correction",
            "student_selection": "highest_center_score_ties_first", "teacher_selection": "loss_and_diagnostics_only",
            "output_scope": "raw_pre_guard_absolute_joint_degrees_div180",
            "future_rl": "record_sampled_center_and_raw_joint_sample_with_conditional_log_probability"}
    if nonlinear_selector:
        contract["format"] = NONLINEAR_CHECKPOINT_FORMAT
        contract["selector_residual"] = {"architecture": [HEAD_FEATURE_COUNT, SELECTOR_HIDDEN_SIZE, CENTER_COUNT],
                                         "activation": "relu", "initialization": "last_linear_zero",
                                         "output": "added_to_student_center_logits"}
    if center_residual:
        contract["format"] = CENTER_RESIDUAL_CHECKPOINT_FORMAT
        contract["center_role"] = "teacher_label_partition_and_inference_move_proposal_base"
        contract["move_proposal"] = "fixed_normalized_center_plus_learned_center_specific_correction"
    return contract


def checkpoint_payload(model):
    if type(model) is not CodebookPolicy or model.joint_branch is not None:
        raise ValueError("a codebook policy is required")
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("all policy parameters must be trainable")
    parameter = model.base.encoder[0].weight
    _validate_centers(model.centers_normalized, dtype=parameter.dtype, device=parameter.device)
    if model.center_residual and (not hasattr(model, "_center_residual_contract_marker") or
            not torch.equal(model._center_residual_contract_marker, parameter.new_ones(1))):
        raise ValueError("center-residual contract marker differs")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("model tensors must be finite")
    return {"contract": checkpoint_contract(nonlinear_selector=model.selector_residual is not None,
                                             center_residual=model.center_residual),
            "state_dict": state}


def load_checkpoint(payload, *, device="cpu"):
    if not isinstance(payload, dict):
        raise ValueError("codebook checkpoint contract differs")
    if payload.get("contract") == checkpoint_contract():
        nonlinear_selector, center_residual = False, False
    elif payload.get("contract") == checkpoint_contract(nonlinear_selector=True):
        nonlinear_selector, center_residual = True, False
    elif payload.get("contract") == checkpoint_contract(nonlinear_selector=True, center_residual=True):
        nonlinear_selector, center_residual = True, True
    else:
        raise ValueError("codebook checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "base.encoder.0.weight" not in state or "centers_normalized" not in state:
        raise ValueError("codebook state dictionary and centers are required")
    weight = state["base.encoder.0.weight"]
    centers = state["centers_normalized"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    _validate_centers(centers, dtype=weight.dtype, device=weight.device)
    with torch.random.fork_rng(devices=[]):
        model = CodebookPolicy(centers, nonlinear_selector=nonlinear_selector,
                               center_residual=center_residual).to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected) or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != weight.dtype or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())
            for name, value in state.items()):
        raise ValueError("checkpoint tensor names, shapes, dtype or values differ")
    if center_residual and not torch.equal(state["_center_residual_contract_marker"], weight.new_ones(1)):
        raise ValueError("center-residual contract marker differs")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.requires_grad_(True)
    model.eval()
    return model
