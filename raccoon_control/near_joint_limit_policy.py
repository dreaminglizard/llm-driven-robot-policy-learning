"""Development-only small joint-limit correction for an existing student.

This hand-written intervention clips a MOVE target only if every exceeded
joint is within one degree of its declared bound. The tolerance was selected
after observing development failures; it is not a learned or hardware policy.
The historical event key ``applied_joints_deg`` is the target returned by this
policy, not evidence that an executor applied a command. GuardedPolicy exposes
separate proposal and applied-command fields.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy

MAX_OVERSHOOT_DEG = 1.0


class NearJointLimitPolicy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.history_length = model.history_length
        self.feature_count = model.feature_count
        self.events = []
        self.eval()

    def forward(self, windows, masks):
        logits, angles = self.model(windows, masks)
        if (tuple(logits.shape) != (1, 4) or tuple(angles.shape) != (1, 4) or
                not bool(torch.isfinite(logits).all() and torch.isfinite(angles).all())):
            self.events.append({"status": "invalid_output_unchanged", "clamped": False})
            return logits, angles
        action = int(logits.argmax(dim=1).item())
        event = {"model_action": motion_policy.ACTIONS[action], "clamped": False}
        if action == motion_policy.MOVE:
            proposed = [float(value) * 180.0 for value in angles[0].tolist()]
            excess = [max(low - value, 0.0, value - high)
                      for value, (low, high) in zip(proposed, JOINT_LIMITS)]
            maximum = max(excess)
            event.update(proposed_joints_deg=proposed, max_overshoot_deg=maximum)
            if 0.0 < maximum <= MAX_OVERSHOOT_DEG:
                bounded = angles.clone()
                for index, (value, (low, high)) in enumerate(zip(proposed, JOINT_LIMITS)):
                    if value < low:
                        endpoint = torch.tensor(low / 180.0, dtype=angles.dtype, device=angles.device)
                        bounded[0, index] = torch.nextafter(endpoint, torch.full_like(endpoint, math.inf))
                    elif value > high:
                        endpoint = torch.tensor(high / 180.0, dtype=angles.dtype, device=angles.device)
                        bounded[0, index] = torch.nextafter(endpoint, torch.full_like(endpoint, -math.inf))
                actual = [float(value) * 180.0 for value in bounded[0].tolist()]
                if not all(low <= value <= high for value, (low, high) in zip(actual, JOINT_LIMITS)):
                    raise ValueError("inward-rounded target is still out of range")
                angles = bounded
                event.update(clamped=True, applied_joints_deg=actual)
        self.events.append(event)
        return logits, angles
