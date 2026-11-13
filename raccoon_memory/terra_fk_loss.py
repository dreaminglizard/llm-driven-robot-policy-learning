"""Nominal TCP auxiliary loss for directly demonstrated Terra MOVE targets.

Both joint tensors are absolute targets in degrees divided by 180.  The
teacher tensor must come from the direct demonstration; this module does not
read object poses, policy outputs, or demonstration files.  Its TCP is the
nominal point between the gripper pads, not a measurement of an object or of
the real robot.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import torch

from raccoon_memory import motion_policy, scene_fk_loss


DEFAULT_SCALE_MM = 10.0


class TerraFKLoss(NamedTuple):
    """Weighted total and unweighted, dimensionless loss components."""

    total: torch.Tensor
    horizontal: torch.Tensor
    height: torch.Tensor
    below_teacher_height: torch.Tensor


def nominal_tcp_mm(joint_targets: torch.Tensor) -> torch.Tensor:
    """Compute the differentiable nominal TCP for normalized joint targets.

    ``joint_targets`` has shape ``[..., 4]`` and uses the same degree/180
    normalization as the Terra absolute joint targets.  No angle clipping or
    quantization is applied.
    """
    if (not isinstance(joint_targets, torch.Tensor) or joint_targets.ndim < 1
            or joint_targets.shape[-1] != 4 or not joint_targets.is_floating_point()):
        raise ValueError("joint targets must be a floating tensor with last dimension 4")
    tcp, _ = scene_fk_loss.nominal_forward(joint_targets * motion_policy.JOINT_SCALE_DEG)
    return tcp


def _number(name: str, value: Real, *, positive: bool) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        constraint = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {constraint} number")
    return float(value)


def move_fk_loss(
    predicted_joints: torch.Tensor,
    teacher_joints: torch.Tensor,
    teacher_move_mask: torch.Tensor,
    *,
    horizontal_weight: float = 1.0,
    height_weight: float = 1.0,
    below_height_weight: float = 0.0,
    horizontal_scale_mm: float = DEFAULT_SCALE_MM,
    height_scale_mm: float = DEFAULT_SCALE_MM,
) -> TerraFKLoss:
    """Compare predicted and directly demonstrated MOVE TCPs in nominal FK.

    Both joint tensors have matching shape ``[..., 4]``, dtype and device;
    ``teacher_move_mask`` is a boolean tensor of shape ``[...]`` selecting
    teacher MOVE labels, not the student's predicted action.  Non-MOVE rows
    do not contribute.  The teacher target is detached from autograd.

    Horizontal error is the mean squared X/Y error divided by
    ``horizontal_scale_mm**2``.  Height error is the mean squared Z error
    divided by ``height_scale_mm**2``.  The optional one-sided component
    penalizes a predicted TCP below the demonstrated TCP.  It is zero by
    default and applies to every selected MOVE if enabled; it is not a
    detector for grasp phase, object position or stack height.

    Each component is averaged over selected rows.  An empty MOVE selection
    returns graph-connected zeros.  The caller chooses how this auxiliary
    loss is weighted against action and joint losses.
    """
    if (not isinstance(predicted_joints, torch.Tensor) or predicted_joints.ndim < 1
            or predicted_joints.shape[-1] != 4 or not predicted_joints.is_floating_point()
            or not isinstance(teacher_joints, torch.Tensor)
            or teacher_joints.shape != predicted_joints.shape
            or teacher_joints.dtype != predicted_joints.dtype
            or teacher_joints.device != predicted_joints.device
            or not isinstance(teacher_move_mask, torch.Tensor)
            or teacher_move_mask.dtype != torch.bool
            or teacher_move_mask.shape != predicted_joints.shape[:-1]
            or teacher_move_mask.device != predicted_joints.device):
        raise ValueError("expected matching floating [...,4] joints and a boolean [...] teacher MOVE mask")

    xy_weight = _number("horizontal_weight", horizontal_weight, positive=False)
    z_weight = _number("height_weight", height_weight, positive=False)
    low_weight = _number("below_height_weight", below_height_weight, positive=False)
    xy_scale = _number("horizontal_scale_mm", horizontal_scale_mm, positive=True)
    z_scale = _number("height_scale_mm", height_scale_mm, positive=True)

    selected = predicted_joints[teacher_move_mask]
    if selected.shape[0] == 0:
        # Index before summing so NaNs in ignored rows cannot poison the zero.
        zero = selected.sum() * 0.0
        return TerraFKLoss(zero, zero, zero, zero)

    predicted_tcp = nominal_tcp_mm(selected)
    teacher_tcp = nominal_tcp_mm(teacher_joints[teacher_move_mask].detach())
    difference = predicted_tcp - teacher_tcp
    horizontal = (difference[..., :2] / xy_scale).square().mean()
    normalized_height = difference[..., 2] / z_scale
    height = normalized_height.square().mean()
    below_height = torch.relu(-normalized_height).square().mean()
    total = xy_weight * horizontal + z_weight * height + low_weight * below_height
    return TerraFKLoss(total, horizontal, height, below_height)
