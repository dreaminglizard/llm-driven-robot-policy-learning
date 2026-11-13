"""A recorded adapter for the existing one-degree virtual joint guard.

The underlying policy is called once. A target returned by this adapter is a
proposal to the executor; only the executor can confirm an applied command.
This development rule is not a validated real-device controller.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from .near_joint_limit_policy import NearJointLimitPolicy, MAX_OVERSHOOT_DEG


class _InvokeOnce(nn.Module):
    """Forward optional controller state without changing either model API."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.history_length = policy.history_length
        self.feature_count = policy.feature_count
        self.previous_target = None
        self.use_context = False
        self.last_output = None

    def forward(self, windows, masks):
        contextual = getattr(self.policy, "forward_with_control_state", None)
        if self.use_context and callable(contextual):
            out = contextual(windows, masks, self.previous_target)
        else:
            out = self.policy(windows, masks)
        self.last_output = out
        return out


class GuardedPolicy(nn.Module):
    """Run the existing guard and retain the original and guarded outputs."""
    def __init__(self, policy):
        super().__init__()
        self._dispatch = _InvokeOnce(policy)
        self._guard = NearJointLimitPolicy(self._dispatch)
        self.history_length = policy.history_length
        self.feature_count = policy.feature_count
        self.events = []
        self.eval()

    @property
    def policy(self):
        return self._dispatch.policy

    def _call(self, windows, masks, previous, use_context):
        if use_context:
            if (not isinstance(previous, torch.Tensor) or tuple(previous.shape) != (1, 4)
                    or not bool(torch.isfinite(previous).all())):
                raise ValueError("previous applied target must be one finite four-joint tensor")
            values = previous.detach().cpu().tolist()[0]
            if not all(low <= value <= high for value, (low, high) in zip(values, JOINT_LIMITS)):
                raise ValueError("previous applied target is outside the joint limits")
        self._dispatch.previous_target = previous
        self._dispatch.use_context = use_context
        self._dispatch.last_output = None
        count = len(self._guard.events)
        try:
            returned = self._guard(windows, masks)
            if len(self._guard.events) != count + 1 or self._dispatch.last_output is None:
                raise ValueError("guard did not record exactly one model return")
            raw_logits, raw_angles = self._dispatch.last_output
            logits, angles = returned
            event = dict(self._guard.events[-1])
            # The legacy guard's "applied" value is only its returned proposal.
            # A downstream command return, not this model call, confirms application.
            event.pop("applied_joints_deg", None)
            event["raw_logits"] = raw_logits.detach().cpu().tolist()
            event["raw_normalized_joint_output"] = raw_angles.detach().cpu().tolist()
            event["guard_returned_normalized_joint_output"] = angles.detach().cpu().tolist()
            event["context_forwarded"] = bool(use_context and callable(
                getattr(self.policy, "forward_with_control_state", None)))
            event["previous_applied_target_deg"] = previous.detach().cpu().tolist()[0] if use_context else None
            event["max_allowed_overshoot_deg"] = MAX_OVERSHOOT_DEG
            event["actual_applied_joints_deg"] = None
            if (tuple(raw_logits.shape) == (1, 4) and tuple(raw_angles.shape) == (1, 4) and
                    bool(torch.isfinite(raw_logits).all() and torch.isfinite(raw_angles).all())):
                event["original_proposal_joints_deg"] = [float(q) * 180.0 for q in raw_angles[0].tolist()]
                action = int(raw_logits.argmax(dim=1).item())
                result = [float(q) * 180.0 for q in angles[0].tolist()]
                event["guard_returned_target_deg"] = result if action == 0 else None
                if action == 0:
                    event["status"] = ("within_limits" if all(lo <= q <= hi for q, (lo, hi) in zip(result, JOINT_LIMITS))
                                       else "outside_guard_range_no_valid_target")
            else:
                event["original_proposal_joints_deg"] = None
                event["guard_returned_target_deg"] = None
                event["status"] = "invalid_output_no_valid_target"
            self.events.append(event)
            return logits, angles
        finally:
            self._dispatch.previous_target = None
            self._dispatch.use_context = False

    def forward(self, windows, masks):
        return self._call(windows, masks, None, False)

    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        """Pass the last confirmed applied target unchanged when supported."""
        return self._call(windows, masks, previous_target_degrees, True)

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        callback = getattr(self.policy, "on_scene_registered", None)
        if callable(callback):
            callback(scene, nominal_slot_xy_mm)

    def on_transfer_started(self, before, task, applied_targets):
        callback = getattr(self.policy, "on_transfer_started", None)
        if callable(callback):
            callback(before, task, applied_targets)

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        callback = getattr(self.policy, "on_control_applied", None)
        if callable(callback):
            callback(action, transmitted_target_deg, applied_target_deg, gripper_opening_m)

    def on_expected_transfer_committed(self, before, task, after):
        callback = getattr(self.policy, "on_expected_transfer_committed", None)
        if callable(callback):
            callback(before, task, after)

    def conditional_pose_state(self):
        callback = getattr(self.policy, "conditional_pose_state", None)
        return callback() if callable(callback) else None
