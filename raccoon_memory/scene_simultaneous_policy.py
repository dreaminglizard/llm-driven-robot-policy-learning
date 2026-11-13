"""Isolated scene policy with simultaneous joint, gripper and stop outputs.

This module defines no hardware or simulator access. Joint outputs are absolute
targets normalized by 180 degrees. STOP has no joint or gripper supervision.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from raccoon_memory import motion_policy, scene_features, scene_policy

CHECKPOINT_FORMAT = "raccoon-scene-simultaneous-joints-gripper.v1"
CHECKPOINT_EXCLUSIVE_FORMAT = "raccoon-scene-exclusive-control-comparison.v1"
CONTINUE, DONE = 0, 1
OPEN, CLOSED = 0, 1
GRIPPER_NAMES = ("open", "closed")
STOP_NAMES = ("continue", "done")


class SimultaneousScenePolicy(nn.Module):
    """MLP64 on the existing 82 features and eight masked history positions."""

    def __init__(self):
        super().__init__()
        self.history_length, self.feature_count, self.hidden_size = 8, 82, 64
        if len(scene_features.FEATURE_NAMES) != self.feature_count:
            raise ValueError("existing scene feature contract differs")
        self.encoder = nn.Sequential(nn.Linear(8 * 83, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.joint_head = nn.Linear(64, 4)
        self.gripper_head = nn.Linear(64, 2)
        self.stop_head = nn.Linear(64, 2)
        if sum(p.numel() for p in self.parameters()) > scene_policy.MAX_PARAMETERS:
            raise ValueError("parameter limit exceeded")

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        if (not isinstance(windows, torch.Tensor) or not windows.is_floating_point()
                or windows.ndim != 3 or windows.shape[1:] != (8, 82) or not windows.shape[0]
                or not bool(torch.isfinite(windows).all())):
            raise ValueError("windows must be finite (batch, 8, 82) floating tensors")
        if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                or masks.shape != windows.shape[:2] or masks.device != windows.device
                or not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any())):
            raise ValueError("masks must be left-padded and include the current observation")
        masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
        encoded = self.encoder(torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1))
        return self.joint_head(encoded), self.gripper_head(encoded), self.stop_head(encoded)


def paired_random_models(seed: int):
    """Create numerically paired fresh models without modifying caller RNG state.

    The exclusive action head rows 0:2 initialize open/closed and rows 2:4
    initialize continue/done. These random rows have no inherited semantics.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        exclusive = scene_policy.ScenePolicy("mlp", 64, 8)
        simultaneous = SimultaneousScenePolicy()
    simultaneous.encoder.load_state_dict(exclusive.encoder.state_dict(), strict=True)
    simultaneous.joint_head.load_state_dict(exclusive.joint_head.state_dict(), strict=True)
    with torch.no_grad():
        simultaneous.gripper_head.weight.copy_(exclusive.action_head.weight[:2])
        simultaneous.gripper_head.bias.copy_(exclusive.action_head.bias[:2])
        simultaneous.stop_head.weight.copy_(exclusive.action_head.weight[2:])
        simultaneous.stop_head.bias.copy_(exclusive.action_head.bias[2:])
    if sum(p.numel() for p in exclusive.parameters()) != sum(p.numel() for p in simultaneous.parameters()):
        raise ValueError("paired architectures have different parameter counts")
    return exclusive, simultaneous


def simultaneous_loss(joints, gripper_logits, stop_logits, joint_target, gripper_target, done_target,
                      joint_weight=100.0):
    """Train stop everywhere; train both commanded targets only on continue rows."""
    n = len(done_target)
    if (joints.shape != (n, 4) or joint_target.shape != (n, 4)
            or gripper_logits.shape != (n, 2) or stop_logits.shape != (n, 2)
            or gripper_target.shape != (n,) or done_target.shape != (n,)
            or done_target.dtype != torch.long or gripper_target.dtype != torch.long or n == 0
            or not bool(((done_target == CONTINUE) | (done_target == DONE)).all())):
        raise ValueError("simultaneous training tensors have incompatible shapes or classes")
    active = done_target == CONTINUE
    if (not bool(torch.isfinite(joints).all() and torch.isfinite(gripper_logits).all() and torch.isfinite(stop_logits).all())
            or not bool(torch.isfinite(joint_target[active]).all())
            or not bool(((gripper_target[active] == OPEN) | (gripper_target[active] == CLOSED)).all())):
        raise ValueError("invalid nonterminal target or nonfinite output")
    stop_ce = F.cross_entropy(stop_logits, done_target)
    if bool(active.any()):
        grip_ce = F.cross_entropy(gripper_logits[active], gripper_target[active])
        joint_mse = F.mse_loss(joints[active], joint_target[active])
    else:
        grip_ce = gripper_logits.sum() * 0
        joint_mse = joints.sum() * 0
    return {"total": stop_ce + grip_ce + joint_weight * joint_mse,
            "stop_ce": stop_ce, "gripper_ce": grip_ce, "joint_mse": joint_mse,
            "continue_count": int(active.sum())}


def exclusive_loss(logits, joints, actions, original_joint_target, joint_weight=100.0):
    """Preserve the existing exclusive CE plus move-only joint regression."""
    n = len(actions)
    if (not n or logits.shape != (n, 4) or joints.shape != (n, 4)
            or original_joint_target.shape != (n, 4) or actions.shape != (n,) or actions.dtype != torch.long
            or not bool(((actions >= 0) & (actions < 4)).all())):
        raise ValueError("exclusive training tensors have incompatible shapes or classes")
    move = actions == motion_policy.MOVE
    if not bool(torch.isfinite(logits).all() and torch.isfinite(joints).all() and torch.isfinite(original_joint_target[move]).all()):
        raise ValueError("nonfinite exclusive prediction or move target")
    action_ce = F.cross_entropy(logits, actions)
    joint_mse = F.mse_loss(joints[move], original_joint_target[move]) if bool(move.any()) else joints.sum() * 0
    return {"total": action_ce + joint_weight * joint_mse, "action_ce": action_ce,
            "joint_mse": joint_mse, "move_count": int(move.sum())}


def validate_checkpoint(payload, expected_format):
    """Reject old checkpoints and prevent the two new contracts being swapped."""
    if expected_format not in (CHECKPOINT_FORMAT, CHECKPOINT_EXCLUSIVE_FORMAT):
        raise ValueError("unknown checkpoint contract")
    if (not isinstance(payload, dict) or payload.get("format") != expected_format
            or payload.get("checkpoint_update") != 3000 or payload.get("selection") != "fixed_final"
            or payload.get("model_config") != {"architecture": "mlp", "hidden_size": 64, "features": 82,
                                              "history_length": 8, "joint_parameterization": "absolute_degrees_div_180"}
            or payload.get("feature_schema") != scene_features.SCHEMA
            or payload.get("feature_names") != list(scene_features.FEATURE_NAMES)
            or payload.get("exclusive_action_names") != list(motion_policy.ACTIONS)
            or payload.get("gripper_names") != list(GRIPPER_NAMES) or payload.get("stop_names") != list(STOP_NAMES)
            or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("checkpoint format, model contract or final update differs")
    model = SimultaneousScenePolicy() if expected_format == CHECKPOINT_FORMAT else scene_policy.ScenePolicy("mlp", 64, 8)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("checkpoint contains nonfinite weights")
    return model
