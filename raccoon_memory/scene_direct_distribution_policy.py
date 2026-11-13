"""Two direct-demonstration output heads on the same public recurrent student.

The shared body is the task-frame student's public feature encoder and LSTM.
The output alternatives predict the same absolute, applied robot joint targets:
a bounded point estimate or twelve autoregressive hexadecimal digits. This
module does not execute actions, choose a transition, or observe object poses.
OPEN/CLOSE/DONE have no joint target. Digit teacher forcing is used only for
confirmed MOVE labels; prediction always feeds back the model's own digits.
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_task_frame_student as task_frame


FORMAT = "raccoon-direct-distribution-policy.v1"
RADIX = 16
DIGITS_PER_JOINT = 3
TOKEN_COUNT = 4 * DIGITS_PER_JOINT
LEVELS = RADIX ** DIGITS_PER_JOINT
START_TOKEN = RADIX
DIGIT_HIDDEN_SIZE = 64
DIGIT_EMBEDDING_SIZE = 32


def _limits(reference):
    return reference.new_tensor(JOINT_LIMITS)


def joints_to_digits(joints_deg):
    """Round each bounded joint to one of 4096 endpoint-inclusive levels.

    The flattened order is joint 1 through 4, most significant digit first.
    Non-finite or out-of-range targets are rejected, never silently clipped.
    """
    if (not isinstance(joints_deg, torch.Tensor) or joints_deg.dtype not in (torch.float32, torch.float64)
            or joints_deg.ndim < 1 or joints_deg.shape[-1] != 4
            or not bool(torch.isfinite(joints_deg).all())):
        raise ValueError("finite float32/float64 joint targets with last dimension four are required")
    limits = _limits(joints_deg)
    if bool(((joints_deg < limits[:, 0]) | (joints_deg > limits[:, 1])).any()):
        raise ValueError("joint target exceeds its limits")
    normalized = (joints_deg - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    value = torch.floor(normalized * (LEVELS - 1) + 0.5).to(torch.long)
    digits = torch.stack((value // (RADIX * RADIX), (value // RADIX) % RADIX,
                          value % RADIX), dim=-1)
    return digits.reshape(*joints_deg.shape[:-1], TOKEN_COUNT)


def digits_to_joints(digits, *, dtype=torch.float32):
    """Decode twelve base-16 digits to bounded robot joint degrees."""
    if (not isinstance(digits, torch.Tensor) or digits.ndim < 1 or digits.shape[-1] != TOKEN_COUNT
            or digits.dtype != torch.long or bool(((digits < 0) | (digits >= RADIX)).any())
            or dtype not in (torch.float32, torch.float64)):
        raise ValueError("twelve integer hexadecimal digits and float32/float64 dtype are required")
    grouped = digits.reshape(*digits.shape[:-1], 4, DIGITS_PER_JOINT)
    values = grouped[..., 0] * RADIX * RADIX + grouped[..., 1] * RADIX + grouped[..., 2]
    values = values.to(dtype=dtype)
    limits = _limits(values)
    return limits[:, 0] + values / (LEVELS - 1) * (limits[:, 1] - limits[:, 0])


class PublicBackbone(task_frame.TaskFrameStudent):
    """Reuse the existing slot encoder, action head, LSTM and mask semantics."""

    def __init__(self, include_object_ids=True):
        super().__init__(include_object_ids=include_object_ids)
        # Existing forward emits this head at every valid recurrent step. The
        # identity exposes the recurrent context without copying its logic.
        self.pose_head = nn.Identity()


class DigitDecoder(nn.Module):
    """Causal joint-major digit model P(d_i | public history, d_<i)."""

    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(task_frame.HIDDEN_SIZE, DIGIT_HIDDEN_SIZE)
        self.tokens = nn.Embedding(RADIX + 1, DIGIT_EMBEDDING_SIZE)
        self.positions = nn.Embedding(TOKEN_COUNT, DIGIT_EMBEDDING_SIZE)
        self.cell = nn.GRUCell(DIGIT_EMBEDDING_SIZE, DIGIT_HIDDEN_SIZE)
        self.readout = nn.Linear(DIGIT_HIDDEN_SIZE, RADIX)

    def _step(self, previous, position, hidden):
        value = self.tokens(previous) + self.positions.weight[position]
        hidden = self.cell(value, hidden)
        return self.readout(hidden), hidden

    def _initial(self, context):
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or context.shape[-1] != task_frame.HIDDEN_SIZE
                or not bool(torch.isfinite(context).all())):
            raise ValueError("finite recurrent contexts [rows,96] are required")
        return torch.tanh(self.initial(context))

    def teacher_forced(self, context, target_digits):
        """Return logits under shifted target prefixes, for MOVE loss only."""
        if (not isinstance(target_digits, torch.Tensor) or target_digits.dtype != torch.long
                or target_digits.shape != (context.shape[0], TOKEN_COUNT)
                or target_digits.device != context.device
                or bool(((target_digits < 0) | (target_digits >= RADIX)).any())):
            raise ValueError("matching target digits [rows,12] are required")
        hidden = self._initial(context)
        previous = torch.full((context.shape[0],), START_TOKEN, dtype=torch.long, device=context.device)
        logits = []
        for position in range(TOKEN_COUNT):
            output, hidden = self._step(previous, position, hidden)
            logits.append(output)
            previous = target_digits[:, position]
        return torch.stack(logits, dim=1)

    def generate(self, context):
        """Greedily decode; each next step receives the preceding prediction."""
        hidden = self._initial(context)
        previous = torch.full((context.shape[0],), START_TOKEN, dtype=torch.long, device=context.device)
        digits = []
        for position in range(TOKEN_COUNT):
            logits, hidden = self._step(previous, position, hidden)
            previous = logits.argmax(dim=-1)
            digits.append(previous)
        return torch.stack(digits, dim=1)


class Prediction(NamedTuple):
    action_logits: torch.Tensor
    actions: torch.Tensor              # -1 on padded rows
    move_joints_deg: torch.Tensor      # NaN unless the predicted action is MOVE
    move_digits: torch.Tensor | None  # -1 outside MOVE, only for digit policy
    state: tuple[torch.Tensor, torch.Tensor]


class DistributionPolicy(nn.Module):
    """Policy candidate with an explicit output kind and no execution adapter."""

    def __init__(self, kind, *, backbone=None, include_object_ids=True):
        super().__init__()
        if kind not in ("point", "digits"):
            raise ValueError("kind must be point or digits")
        if type(include_object_ids) is not bool:
            raise ValueError("include_object_ids must be boolean")
        if backbone is not None and (type(backbone) is not PublicBackbone
                                     or backbone.include_object_ids != include_object_ids):
            raise ValueError("backbone type or ID condition differs")
        self.kind = kind
        self.backbone = PublicBackbone(include_object_ids) if backbone is None else copy.deepcopy(backbone)
        self.head = nn.Linear(task_frame.HIDDEN_SIZE, 4) if kind == "point" else DigitDecoder()

    def encode(self, features, valid_mask, state=None):
        return self.backbone(features, valid_mask, state)

    def point_joints(self, context):
        if self.kind != "point":
            raise ValueError("point output is available only for the point policy")
        limits = _limits(context)
        return limits.mean(-1) + torch.tanh(self.head(context)) * ((limits[:, 1] - limits[:, 0]) / 2)

    def predict(self, features, valid_mask, state=None):
        """Predict action and, only on predicted MOVE rows, bounded targets.

        In particular, this interface accepts no labels or ground-truth digits.
        An action proposal is not an executed command or success assessment.
        """
        out = self.encode(features, valid_mask, state)
        actions = out.action_logits.argmax(dim=-1).masked_fill(~valid_mask, -1)
        moves = valid_mask & (actions == motion_policy.MOVE)
        joints = features.new_full((*valid_mask.shape, 4), float("nan"))
        digits = None
        if self.kind == "point":
            if bool(moves.any()):
                joints[moves] = self.point_joints(out.move_pose[moves])
        else:
            digits = torch.full((*valid_mask.shape, TOKEN_COUNT), -1, dtype=torch.long, device=features.device)
            if bool(moves.any()):
                predicted_digits = self.head.generate(out.move_pose[moves])
                digits[moves] = predicted_digits
                joints[moves] = digits_to_joints(predicted_digits, dtype=features.dtype)
        return Prediction(out.action_logits, actions, joints, digits, out.state)


class Loss(NamedTuple):
    total: torch.Tensor
    action: torch.Tensor
    move: torch.Tensor
    action_count: int
    move_count: int


def supervised_loss(model, batch, *, move_weight):
    """Labeled action CE plus applied-MOVE point MSE or digit CE.

    Context and non-MOVE rows advance memory but never supply joint labels.
    Batch metadata is not passed to the model. Provenance must be verified by
    the loader; the masks alone cannot establish that a command was applied.
    """
    import math

    if not isinstance(model, DistributionPolicy):
        raise ValueError("a distribution policy is required")
    if type(move_weight) not in (int, float) or not math.isfinite(move_weight) or move_weight <= 0:
        raise ValueError("MOVE weight must be positive and finite")
    features, valid = batch["features"], batch["valid"]
    actions, targets = batch["actions"], batch["targets"]
    labels, moves = batch["labels"], batch["applied_moves"]
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or valid.shape != features.shape[:2] or valid.device != features.device
            or any(not isinstance(m, torch.Tensor) or m.dtype != torch.bool or m.shape != valid.shape
                   or m.device != features.device for m in (labels, moves))
            or not isinstance(actions, torch.Tensor) or actions.dtype != torch.long
            or actions.shape != valid.shape or actions.device != features.device
            or not isinstance(targets, torch.Tensor) or targets.dtype != features.dtype
            or targets.device != features.device or targets.shape != (*valid.shape, 4)):
        raise ValueError("batch tensors or masks differ")
    if (bool((labels & ~valid).any()) or not torch.equal(moves, labels & (actions == motion_policy.MOVE))
            or bool(((actions[labels] < 0) | (actions[labels] >= len(motion_policy.ACTIONS))).any())
            or bool((actions[~labels] != -1).any()) or not bool(torch.isnan(targets[~moves]).all())):
        raise ValueError("action, applied MOVE, context or padding masks differ")
    if bool(moves.any()):
        # Validate original applied targets even for the continuous branch.
        digits = joints_to_digits(targets[moves])
    out = model.encode(features, valid)
    if not bool(torch.isfinite(out.action_logits[valid]).all() and torch.isfinite(out.move_pose[valid]).all()):
        raise ValueError("non-finite model outputs")
    action = (F.cross_entropy(out.action_logits[labels], actions[labels]) if bool(labels.any())
              else out.action_logits[labels].sum())
    if not bool(moves.any()):
        move = out.move_pose[moves].sum()
    elif model.kind == "point":
        move = F.mse_loss(model.point_joints(out.move_pose[moves]) / 180., targets[moves] / 180.)
    else:
        logits = model.head.teacher_forced(out.move_pose[moves], digits)
        move = F.cross_entropy(logits.reshape(-1, RADIX), digits.reshape(-1))
    return Loss(action + float(move_weight) * move, action, move, int(labels.sum()), int(moves.sum()))


def model_contract(kind, include_object_ids=True):
    if kind not in ("point", "digits") or type(include_object_ids) is not bool:
        raise ValueError("unsupported model condition")
    return {"format": FORMAT, "kind": kind, "include_object_ids": include_object_ids,
            "features": "task_frame_public_92", "body": "task_frame_shared_slot_encoder_lstm96",
            "actions": list(motion_policy.ACTIONS), "output": "absolute_applied_robot_joint_degrees",
            "joint_limits_deg": [list(pair) for pair in JOINT_LIMITS],
            "point": "tanh_bounded" if kind == "point" else None,
            "digits": ({"base": RADIX, "digits_per_joint": DIGITS_PER_JOINT, "levels": LEVELS,
                        "order": "joint_major_most_significant_first", "endpoint_inclusive": True,
                        "rounding": "nearest_half_up", "training": "teacher_forced",
                        "inference": "autoregressive_greedy_self_feedback", "hidden_size": DIGIT_HIDDEN_SIZE,
                        "embedding_size": DIGIT_EMBEDDING_SIZE} if kind == "digits" else None)}
