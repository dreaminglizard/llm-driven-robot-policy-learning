"""A small, causal student that proposes robot-frame joint targets directly.

The only observation features are those in :mod:`scene_features`: current
robot feedback and the *registered* symbolic scene, task and nominal slot
coordinates.  Registration is information supplied to the controller; it is
not the simulator's actual object pose or a claim that the registered scene is
correct.  The remaining features are the last confirmed applied physical
action, its held joint target, and transfer/registration boundary indicators.
No ground-truth object state, hidden corrector state, teacher proposal, future
observation, or data-source identifier belongs in the model input.

The action head uses ``motion_policy.ACTIONS`` order.  The joint head always
returns four absolute robot-coordinate joint angles in degrees, bounded by the
existing joint limits.  They are MOVE proposals; for OPEN, CLOSE and DONE the
caller must ignore them.  There is no inverse frame transform, attraction,
action-dependent correction, or physical execution in this module.

For supervision, MOVE joint targets must be the *actually applied physical
command*, identified by the caller from a confirmed application record.  A
raw teacher proposal, attempted/unconfirmed command, or inferred correction
is not such a target.  No OPEN, CLOSE or DONE joint target is fitted.  The
explicit mask in ``supervised_loss`` cannot itself prove provenance; the data
producer must establish it before marking an entry as applied.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_features


FORMAT = "raccoon-scene-applied-action-student.v1"
HIDDEN_SIZE = 64
# DONE never applies a physical command, so it cannot be the last applied one.
PREVIOUS_APPLIED_ACTIONS = (None, "move", "open", "close")
FEATURE_NAMES = (
    *scene_features.FEATURE_NAMES,
    *(f"previous_applied_{name}" for name in ("none", "move", "open", "close")),
    *(f"previous_applied_joint_{i}_deg/180" for i in range(1, 5)),
    "transfer_start",
    "registration_start",
)
FEATURE_COUNT = len(FEATURE_NAMES)
PREVIOUS_ACTION_SLICE = slice(len(scene_features.FEATURE_NAMES), len(scene_features.FEATURE_NAMES) + 4)
PREVIOUS_TARGET_SLICE = slice(PREVIOUS_ACTION_SLICE.stop, PREVIOUS_ACTION_SLICE.stop + 4)
TRANSFER_START_INDEX = PREVIOUS_TARGET_SLICE.stop
REGISTRATION_START_INDEX = TRANSFER_START_INDEX + 1


def _applied_target(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("previous applied joint target must contain four angles in degrees")
    result = []
    for value, (low, high) in zip(values, JOINT_LIMITS):
        if type(value) not in (int, float):
            raise ValueError("previous applied joint target must be finite")
        try:
            angle = float(value)
        except (OverflowError, ValueError):
            raise ValueError("previous applied joint target must be finite") from None
        if not math.isfinite(angle):
            raise ValueError("previous applied joint target must be finite")
        if not low - 1e-7 <= angle <= high + 1e-7:
            raise ValueError("previous applied joint target exceeds robot joint limits")
        result.append(angle)
    return result


def encode_step(observation, registered_scene, task, *, slot_xy_mm,
                previous_applied_action, previous_applied_target_deg,
                transfer_start, registration_start):
    """Encode one current observation with no access to later observations.

    ``registered_scene``, ``task`` and ``slot_xy_mm`` must be the controller's
    current registration, not measurements of hidden object truth.  A previous
    action is MOVE/OPEN/CLOSE only after confirmed application; ``None`` means
    no previous physical action is known.  At registration, the target still
    has to come from a validated initial control or an actually applied prior
    command; it must not be guessed from a model proposal.  DONE is not a
    physical action.  Additional dictionary fields are never read as features.

    ``transfer_start`` marks a new transfer within a registration without
    clearing memory.  ``registration_start`` marks the first observation after
    registration/re-registration and clears memory before processing that row.
    The current ``moving`` feedback is already present in the scene features.
    """
    if previous_applied_action not in PREVIOUS_APPLIED_ACTIONS:
        raise ValueError("previous physical action must be move, open, close, or None")
    if type(transfer_start) is not bool or type(registration_start) is not bool:
        raise ValueError("transfer and registration boundaries must be booleans")
    target = _applied_target(previous_applied_target_deg)
    features = scene_features.encode_scene_features(observation, registered_scene, task, slot_xy_mm)
    features.extend(float(previous_applied_action == action) for action in PREVIOUS_APPLIED_ACTIONS)
    features.extend(value / motion_policy.JOINT_SCALE_DEG for value in target)
    features.extend((float(transfer_start), float(registration_start)))
    return features


class PolicyOutputs(NamedTuple):
    action_logits: torch.Tensor       # [batch, time, 4], in ACTIONS order
    move_joints_deg: torch.Tensor     # [batch, time, 4], absolute robot angles
    state: torch.Tensor               # [1, batch, HIDDEN_SIZE]


class LossOutputs(NamedTuple):
    total: torch.Tensor
    action: torch.Tensor
    move: torch.Tensor
    action_count: int
    move_count: int


class AppliedActionStudent(nn.Module):
    """Single forward-only GRU, shared across every transfer in a registration.

    ``forward`` consumes encoded rows [B,T,F] and a boolean validity mask
    [B,T].  Padding can occur anywhere and neither advances nor resets memory;
    its outputs are zero.  With no incoming state, each nonempty batch member
    must start with a registration boundary.  A later registration boundary
    resets that member immediately before the row; a transfer boundary does
    not.  Pass the returned state to continue a sequence, retaining or
    detaching it as appropriate for the caller's training procedure.
    """

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(FEATURE_COUNT, HIDDEN_SIZE, batch_first=True)
        self.action_head = nn.Linear(HIDDEN_SIZE, len(motion_policy.ACTIONS))
        self.joint_head = nn.Linear(HIDDEN_SIZE, 4)
        limits = torch.tensor(JOINT_LIMITS, dtype=torch.float32)
        self.register_buffer("joint_center_deg", limits.mean(dim=1))
        self.register_buffer("joint_half_range_deg", (limits[:, 1] - limits[:, 0]) / 2)

    def forward(self, features, valid_mask, state=None):
        parameter = self.action_head.weight
        if (not isinstance(features, torch.Tensor) or features.ndim != 3 or
                features.shape[0] < 1 or features.shape[1] < 1 or
                features.shape[2] != FEATURE_COUNT or features.dtype != parameter.dtype or
                features.device != parameter.device):
            raise ValueError("features must be [batch,time,FEATURE_COUNT] on the model dtype and device")
        batch, steps = features.shape[:2]
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (batch, steps) or
                valid_mask.dtype != torch.bool or valid_mask.device != features.device):
            raise ValueError("validity mask must be boolean [batch,time] on the feature device")
        present = features[valid_mask]
        if not bool(torch.isfinite(present).all()):
            raise ValueError("valid inputs must be finite")
        previous_actions = present[:, PREVIOUS_ACTION_SLICE]
        if (bool(((previous_actions != 0) & (previous_actions != 1)).any()) or
                bool((previous_actions.sum(dim=1) != 1).any())):
            raise ValueError("previous applied action must be one-hot")
        boundaries = present[:, [TRANSFER_START_INDEX, REGISTRATION_START_INDEX]]
        if bool(((boundaries != 0) & (boundaries != 1)).any()):
            raise ValueError("boundary inputs must be binary")
        previous_targets = present[:, PREVIOUS_TARGET_SLICE] * motion_policy.JOINT_SCALE_DEG
        limits = previous_targets.new_tensor(JOINT_LIMITS)
        if bool(((previous_targets < limits[:, 0] - 1e-4) |
                 (previous_targets > limits[:, 1] + 1e-4)).any()):
            raise ValueError("previous applied target exceeds robot joint limits")
        if state is None:
            has_rows = valid_mask.any(dim=1)
            first = valid_mask.to(torch.int64).argmax(dim=1)
            first_boundary = features[torch.arange(batch, device=features.device), first,
                                      REGISTRATION_START_INDEX]
            if bool((has_rows & (first_boundary != 1)).any()):
                raise ValueError("a sequence without prior state must begin at registration")
            hidden = features.new_zeros((1, batch, HIDDEN_SIZE))
        else:
            if (not isinstance(state, torch.Tensor) or state.shape != (1, batch, HIDDEN_SIZE) or
                    state.dtype != features.dtype or state.device != features.device or
                    not bool(torch.isfinite(state).all())):
                raise ValueError("state must be finite [1,batch,HIDDEN_SIZE] on the model dtype and device")
            hidden = state

        logits, joints = [], []
        for step in range(steps):
            valid = valid_mask[:, step]
            # Sanitizing padding also prevents NaNs in ignored cells from
            # contaminating backward gradients through a masked GRU call.
            row = torch.where(valid[:, None], features[:, step], torch.zeros_like(features[:, step]))
            reset = valid & (row[:, REGISTRATION_START_INDEX] == 1)
            before = torch.where(reset[None, :, None], torch.zeros_like(hidden), hidden)
            _, candidate = self.gru(row[:, None, :], before)
            hidden = torch.where(valid[None, :, None], candidate, hidden)
            action = self.action_head(hidden[0])
            target = self.joint_center_deg + self.joint_half_range_deg * torch.tanh(self.joint_head(hidden[0]))
            logits.append(torch.where(valid[:, None], action, torch.zeros_like(action)))
            joints.append(torch.where(valid[:, None], target, torch.zeros_like(target)))
        return PolicyOutputs(torch.stack(logits, dim=1), torch.stack(joints, dim=1), hidden)


class RegistrationSession:
    """Independent inference memory for one registration at a time.

    Create one session per concurrent episode.  Call ``start_registration``
    when registration changes; transfer changes alone do not reset it.  The
    caller supplies confirmed last-applied control on every step: this class
    does not treat its own proposal as executed or communicate with a robot.
    """

    def __init__(self, model):
        if not isinstance(model, AppliedActionStudent):
            raise ValueError("an AppliedActionStudent is required")
        self.model = model
        self.start_registration()

    def start_registration(self):
        self._state = None
        self._first = True

    @torch.no_grad()
    def step(self, observation, registered_scene, task, *, slot_xy_mm,
             previous_applied_action, previous_applied_target_deg, transfer_start):
        row = encode_step(observation, registered_scene, task, slot_xy_mm=slot_xy_mm,
                          previous_applied_action=previous_applied_action,
                          previous_applied_target_deg=previous_applied_target_deg,
                          transfer_start=transfer_start, registration_start=self._first)
        parameter = self.model.action_head.weight
        features = torch.tensor(row, dtype=parameter.dtype, device=parameter.device).reshape(1, 1, -1)
        valid = torch.ones((1, 1), dtype=torch.bool, device=parameter.device)
        output = self.model(features, valid, self._state)
        self._state = output.state.detach()
        self._first = False
        return output.action_logits[0, 0], output.move_joints_deg[0, 0]


def supervised_loss(outputs, actions, applied_move_targets_deg, *, valid_mask,
                    label_mask, applied_move_mask, joint_weight=1.0):
    """Cross entropy and MOVE-only MSE, with joint errors scaled by 180 degrees.

    ``label_mask`` selects known action labels, independent of predictions.
    ``applied_move_mask`` selects only labeled MOVE rows for which the data
    producer confirmed physical application and supplied the actual applied
    target.  Other target cells may be NaN and are never read.  A MOVE without
    confirmed application can contribute action loss but cannot contribute
    joint loss.  This function trusts the producer's explicit provenance mask.
    """
    if not isinstance(outputs, PolicyOutputs):
        raise ValueError("outputs must come from AppliedActionStudent")
    logits, predicted = outputs.action_logits, outputs.move_joints_deg
    if (logits.ndim != 3 or logits.shape[-1] != 4 or predicted.shape != logits.shape or
            logits.dtype != predicted.dtype or logits.device != predicted.device):
        raise ValueError("invalid action or joint output shape")
    shape = logits.shape[:2]
    if (not isinstance(actions, torch.Tensor) or actions.shape != shape or
            actions.dtype != torch.long or actions.device != logits.device or
            not isinstance(applied_move_targets_deg, torch.Tensor) or
            applied_move_targets_deg.shape != predicted.shape or
            applied_move_targets_deg.dtype != predicted.dtype or
            applied_move_targets_deg.device != predicted.device):
        raise ValueError("invalid action labels or applied joint targets")
    for mask in (valid_mask, label_mask, applied_move_mask):
        if (not isinstance(mask, torch.Tensor) or mask.shape != shape or
                mask.dtype != torch.bool or mask.device != logits.device):
            raise ValueError("all masks must be boolean [batch,time] on the output device")
    if (bool((label_mask & ~valid_mask).any()) or bool((applied_move_mask & ~label_mask).any()) or
            bool(((actions[label_mask] < 0) | (actions[label_mask] >= 4)).any()) or
            bool((actions[applied_move_mask] != motion_policy.MOVE).any())):
        raise ValueError("applied MOVE rows must be a subset of valid labeled MOVE rows")
    if type(joint_weight) not in (int, float) or not math.isfinite(joint_weight) or joint_weight < 0:
        raise ValueError("joint_weight must be a finite nonnegative number")
    actual = applied_move_targets_deg[applied_move_mask]
    if (not bool(torch.isfinite(logits[label_mask]).all()) or
            not bool(torch.isfinite(predicted[applied_move_mask]).all()) or
            not bool(torch.isfinite(actual).all())):
        raise ValueError("supervised values must be finite")
    limits = actual.new_tensor(JOINT_LIMITS)
    if bool(((actual < limits[:, 0] - 1e-4) | (actual > limits[:, 1] + 1e-4)).any()):
        raise ValueError("applied MOVE target exceeds robot joint limits")
    action_count, move_count = int(label_mask.sum()), int(applied_move_mask.sum())
    action_loss = (F.cross_entropy(logits[label_mask], actions[label_mask]) if action_count
                   else logits[label_mask].sum())
    move_loss = (F.mse_loss(predicted[applied_move_mask] / motion_policy.JOINT_SCALE_DEG,
                            actual / motion_policy.JOINT_SCALE_DEG) if move_count
                 else predicted[applied_move_mask].sum())
    return LossOutputs(action_loss + joint_weight * move_loss, action_loss, move_loss,
                       action_count, move_count)
