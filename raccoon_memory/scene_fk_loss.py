"""Differentiable nominal TCP and tool-axis loss for scene-policy training.

This is the same idealized geometry as ``Geometry.forward``.  It uses only
predicted and demonstrated joint targets; it is not a physical-state sensor,
collision check, or estimate of real-device positioning error.
"""
from __future__ import annotations

import math

import torch

from raccoon_control.kinematics import Geometry


POSITION_SCALE_MM = 10.0
AXIS_CHORD_SCALE = 2.0 * math.sin(math.radians(5.0))
JOINT_SCALE_DEG = 180.0
NOMINAL_GEOMETRY = Geometry()


def nominal_forward(joints_deg: torch.Tensor, geometry: Geometry = NOMINAL_GEOMETRY):
    """Return nominal world TCP in mm and unit tool-forward axis.

    The input can have any leading dimensions, followed by four joint angles
    in degrees. No joint clipping, quantization or teacher information is used.
    """
    if (not isinstance(joints_deg, torch.Tensor) or joints_deg.ndim < 1
            or joints_deg.shape[-1] != 4 or not joints_deg.is_floating_point()):
        raise ValueError("joint angles must be a floating tensor with last dimension 4")
    q1, q2, q3, q4 = torch.deg2rad(joints_deg).unbind(dim=-1)
    pitch = q2 + q3 + q4 + math.pi / 2.0
    radius = (-geometry.upper_mm * torch.sin(q2) - geometry.forearm_mm * torch.sin(q2 + q3)
              + geometry.tool_forward_mm * torch.cos(pitch) - geometry.tool_vertical_mm * torch.sin(pitch))
    z = (geometry.base_mm + geometry.upper_mm * torch.cos(q2) + geometry.forearm_mm * torch.cos(q2 + q3)
         + geometry.tool_forward_mm * torch.sin(pitch) + geometry.tool_vertical_mm * torch.cos(pitch))
    sin_yaw, cos_yaw, cos_pitch = torch.sin(q1), torch.cos(q1), torch.cos(pitch)
    tcp_mm = torch.stack((-radius * sin_yaw, radius * cos_yaw, z), dim=-1)
    tool_axis = torch.stack((-sin_yaw * cos_pitch, cos_yaw * cos_pitch, torch.sin(pitch)), dim=-1)
    return tcp_mm, tool_axis


def move_fk_loss(predicted: torch.Tensor, targets: torch.Tensor, move_mask: torch.Tensor):
    """Compute total, position and axis MSE on move targets normalized by 180.

    Each MSE averages over selected moves and its three vector components.
    Empty selections return graph-connected zeros, with no loss on other actions.
    """
    if (not isinstance(predicted, torch.Tensor) or not isinstance(targets, torch.Tensor)
            or predicted.ndim != 2 or predicted.shape[-1] != 4 or targets.shape != predicted.shape
            or not predicted.is_floating_point() or targets.dtype != predicted.dtype
            or targets.device != predicted.device or not isinstance(move_mask, torch.Tensor)
            or move_mask.dtype != torch.bool or move_mask.shape != predicted.shape[:1]
            or move_mask.device != predicted.device):
        raise ValueError("expected matching floating [batch,4] targets and a boolean [batch] move mask")
    if not bool(move_mask.any()):
        zero = predicted.sum() * 0.0
        return zero, zero, zero
    pred_position, pred_axis = nominal_forward(predicted[move_mask] * JOINT_SCALE_DEG)
    target_position, target_axis = nominal_forward(targets[move_mask] * JOINT_SCALE_DEG)
    position = ((pred_position - target_position) / POSITION_SCALE_MM).square().mean()
    axis = ((pred_axis - target_axis) / AXIS_CHORD_SCALE).square().mean()
    return position + axis, position, axis
