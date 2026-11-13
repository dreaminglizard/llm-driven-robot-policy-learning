"""Training-only BC action distillation with one nominal placement target.

No simulator objects, contacts, hidden stage, or runtime model switching are
used. The supplied BC output is always the only action target. A MOVE target
may be replaced only when both teachers say MOVE and the observable nominal
geometry identifies the destination's final placement approach.
"""
from __future__ import annotations

import math
from typing import NamedTuple
import torch
from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_fk_loss


# The same tolerances apply to all slots, levels, and object identities.
CURRENT_DESTINATION_XY_MM = 30.0
BC_TARGET_DESTINATION_XY_MM = 25.0
CURRENT_BELOW_GOAL_MM = 10.0
CURRENT_ABOVE_GOAL_MM = 55.0
BC_TARGET_BELOW_GOAL_MM = 10.0
BC_TARGET_ABOVE_GOAL_MM = 30.0
PLACEMENT_BLEND = 0.25


class ConsensusTargets(NamedTuple):
    action_probabilities: torch.Tensor
    bc_actions: torch.Tensor
    joints: torch.Tensor
    correction: torch.Tensor
    both_move: torch.Tensor


def nominal_placement_joints(registered_xy, registered_z):
    """Vectorized inverse of the same fixed nominal placement geometry."""
    radius = torch.linalg.vector_norm(registered_xy, dim=1)
    arm_r, arm_z = radius - 80., registered_z - 76.5
    cosine = (arm_r.square() + arm_z.square() - 20000.) / 20000.
    if bool((radius < 1e-6).any()) or bool(((cosine < -1.-1e-6) | (cosine > 1.+1e-6)).any()):
        raise ValueError("nominal placement is not reachable")
    q3 = -torch.acos(cosine.clamp(-1., 1.))
    q2 = torch.atan2(-arm_r, arm_z) - torch.atan2(100. * torch.sin(q3), 100. + 100. * torch.cos(q3))
    q1 = torch.atan2(-registered_xy[:, 0], registered_xy[:, 1])
    q4 = -math.pi / 2. - q2 - q3
    normalized = torch.stack((q1, q2, q3, q4), 1) / math.pi
    bounds = normalized.new_tensor(JOINT_LIMITS) / 180.
    if bool(((normalized < bounds[:, 0] - 1e-6) | (normalized > bounds[:, 1] + 1e-6)).any()):
        raise ValueError("nominal placement exceeds joint limits")
    return normalized


def compose(windows, bc_logits, bc_joints, nominal_actions):
    """Return one action distribution and one joint target per observation.

    Inputs are full student observation windows, unmodified BC logits and
    normalized absolute joint proposals, and separately computed observable
    nominal-teacher actions. The placement target is fixed by registered slot
    and level, and is supervised only where BC predicts MOVE.
    """
    if not isinstance(windows, torch.Tensor) or windows.ndim != 3 or windows.shape[1:] != (8, 82):
        raise ValueError("expected [batch,8,82] student windows")
    n = windows.shape[0]
    if (n == 0 or bc_logits.shape != (n, 4) or bc_joints.shape != (n, 4) or
            nominal_actions.shape != (n,) or nominal_actions.dtype != torch.long or
            any(t.device != windows.device for t in (bc_logits, bc_joints, nominal_actions)) or
            any(not bool(torch.isfinite(t).all()) for t in (windows, bc_logits, bc_joints))):
        raise ValueError("incompatible or nonfinite teacher outputs")
    row = windows[:, -1]
    destination = row[:, 18:22]
    level = row[:, 24:26]
    if (not bool(((destination == 0) | (destination == 1)).all()) or
            not bool((destination.sum(1) == 1).all()) or
            not bool(((level == 0) | (level == 1)).all()) or not bool((level.sum(1) == 1).all())):
        raise ValueError("destination slot and level must be one-hot")
    dst = destination.argmax(1)
    coords = row[:, 74:82].reshape(n, 4, 2) * 300.
    registered_xy = coords[torch.arange(n, device=windows.device), dst]
    registered_z = 15. + 20. * level.argmax(1).to(row.dtype)
    measured_xyz, _ = scene_fk_loss.nominal_forward(row[:, :4] * 180.)
    bc_xyz, _ = scene_fk_loss.nominal_forward(bc_joints * 180.)
    placement_joints = nominal_placement_joints(registered_xy, registered_z)
    norm = lambda value: torch.linalg.vector_norm(value, dim=1)
    bc_actions = bc_logits.argmax(1)
    both_move = (bc_actions == motion_policy.MOVE) & (nominal_actions == motion_policy.MOVE)
    correction = (both_move & (row[:, 8] == 1) &
                  (norm(measured_xyz[:, :2] - registered_xy) <= CURRENT_DESTINATION_XY_MM) &
                  (measured_xyz[:, 2] >= registered_z - CURRENT_BELOW_GOAL_MM) &
                  (measured_xyz[:, 2] <= registered_z + CURRENT_ABOVE_GOAL_MM) &
                  (norm(bc_xyz[:, :2] - registered_xy) <= BC_TARGET_DESTINATION_XY_MM) &
                  (bc_xyz[:, 2] >= registered_z - BC_TARGET_BELOW_GOAL_MM) &
                  (bc_xyz[:, 2] <= registered_z + BC_TARGET_ABOVE_GOAL_MM))
    # One fixed target per MOVE: a modest uniform step from the baseline
    # proposal toward the registered nominal placement solution.
    corrected = bc_joints + PLACEMENT_BLEND * (placement_joints - bc_joints)
    targets = torch.where(correction[:, None], corrected, bc_joints)
    return ConsensusTargets(bc_logits.softmax(1), bc_actions, targets, correction, both_move)
