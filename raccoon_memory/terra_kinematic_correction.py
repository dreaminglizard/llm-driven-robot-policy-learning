"""Trace one Terra student through the existing public nominal correction.

The same student is called exactly once per decision.  The canonical frame,
conditional no-slip tracker and grasp/placement calculations are inherited
unchanged; the existing joint guard remains the outermost command guard.
Nothing in this module reads a simulator, an object pose, or another policy.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from raccoon_memory import motion_policy
from raccoon_memory.scene_belief_canonical_policy import BeliefCanonicalPolicy
from raccoon_memory.scene_placement_belief_policy import GraspPlacementBeliefCanonicalPolicy


ARMS = ("raw", "canonical_only", "canonical_grasp_placement")
GRASP_TCP_ABOVE_CENTER_MM = 5.0
TRACE_CHANGE_TOLERANCE_DEG = 1e-4


def _degrees(angles):
    return [float(value) * motion_policy.JOINT_SCALE_DEG for value in angles.detach().cpu()[0].tolist()]


def _changed(left, right):
    if left is None or right is None:
        return None
    return any(not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > TRACE_CHANGE_TOLERANCE_DEG
               for a, b in zip(left, right, strict=True))


class StudentProbe(nn.Module):
    """Expose the stateless Terra interface expected by the existing wrapper."""

    history_length, feature_count = 8, 82

    def __init__(self, student):
        super().__init__()
        if (not isinstance(student, nn.Module)
                or (getattr(student, "history_length", None), getattr(student, "feature_count", None)) != (8, 82)):
            raise ValueError("one Terra student with the public 8-by-82 interface is required")
        self.student = student
        self.calls = 0
        self.last = None
        self.last_logits = None

    def forward_with_control_state(self, windows, masks, previous):
        # The Terra student itself does not take previous targets as an input.
        self.calls += 1
        logits, angles = self.student(windows, masks)
        if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 4)
                or not isinstance(angles, torch.Tensor) or tuple(angles.shape) != (1, 4)):
            raise ValueError("the single student must return one action and four joint proposals")
        self.last_logits = logits.detach().clone()
        self.last = {
            "action_logits": logits.detach().cpu().tolist()[0],
            "action": motion_policy.ACTIONS[int(logits.argmax(dim=1).item())],
            "joints_in_student_frame_deg": _degrees(angles),
            "input_window": windows.detach().cpu().tolist()[0],
            "input_mask": masks.detach().cpu().tolist()[0],
            "input_previous_target_deg": previous.detach().cpu().tolist()[0],
        }
        return logits, angles


class RawStudent(nn.Module):
    """Pass the unmodified registered-frame proposal to the same outer guard."""

    history_length, feature_count = 8, 82

    def __init__(self, student):
        super().__init__()
        self.reference = StudentProbe(student)
        self.last_trace = None

    def forward_with_control_state(self, windows, masks, previous):
        self.last_trace = None
        before = self.reference.calls
        logits, target = self.reference.forward_with_control_state(windows, masks, previous)
        if self.reference.calls != before + 1:
            raise ValueError("exactly one student evaluation is required")
        action = self.reference.last["action"]
        proposal = _degrees(target) if action == "move" else None
        self.last_trace = {"student": copy.deepcopy(self.reference.last),
                           "input_frame": "registered",
                           "nominal_correction": {"enabled": False,
                               "frame_mapped_joints_deg": proposal,
                               "frame_mapping_changed": False if proposal is not None else None,
                               "corrected_joints_deg": proposal,
                               "changed_after_frame_mapping": False if proposal is not None else None,
                               "diagnostic": None}}
        return logits, target


class _TraceCanonical:
    """Capture boundaries around the inherited, unmodified computations."""

    def _forward_in_frame(self, windows, masks, previous, *, source_anchor=None):
        logits, joints = super()._forward_in_frame(windows, masks, previous, source_anchor=source_anchor)
        self._mapped_proposal = joints.detach().clone()
        return logits, joints

    def forward_with_control_state(self, windows, masks, previous):
        self.last_trace, self._mapped_proposal = None, None
        before = self.reference.calls
        logits, corrected = super().forward_with_control_state(windows, masks, previous)
        if (self.reference.calls != before + 1 or self.reference.last is None
                or self._mapped_proposal is None
                or not torch.allclose(logits, self.reference.last_logits, rtol=0, atol=0, equal_nan=True)):
            raise ValueError("correction changed the student's action logits or call count")
        action = self.reference.last["action"]
        mapped = _degrees(self._mapped_proposal) if action == "move" else None
        proposal = _degrees(corrected) if action == "move" else None
        diagnostic = copy.deepcopy(self.last_diagnostic)
        self.last_trace = {"student": copy.deepcopy(self.reference.last),
                           "input_frame": "registered" if diagnostic.get("identity_registration") else "canonical",
                           "nominal_correction": {"enabled": self.grasp_placement_enabled,
                               "frame_mapped_joints_deg": mapped,
                               "frame_mapping_changed": _changed(self.reference.last["joints_in_student_frame_deg"], mapped),
                               "corrected_joints_deg": proposal,
                               "changed_after_frame_mapping": _changed(mapped, proposal),
                               "diagnostic": diagnostic}}
        return logits, corrected


class CanonicalOnly(_TraceCanonical, BeliefCanonicalPolicy):
    """Control arm: retain the strong configuration's frame and belief only."""

    grasp_placement_enabled = False

    def __init__(self, student):
        super().__init__(StudentProbe(student), center_lateral_on_close=True, joint_limited_pitch=True)
        self.last_trace = None
        self._mapped_proposal = None


class CanonicalGraspPlacement(_TraceCanonical, GraspPlacementBeliefCanonicalPolicy):
    """Use the original strong configuration's full nominal correction."""

    grasp_placement_enabled = True

    def __init__(self, student):
        super().__init__(StudentProbe(student), grasp_tcp_above_center_mm=GRASP_TCP_ABOVE_CENTER_MM)
        self.last_trace = None
        self._mapped_proposal = None


class TracedGuard(GuardedPolicy):
    """Preserve guard behavior and add separate student/correction evidence."""

    def on_transfer_started(self, before, task, applied_targets):
        # The saved strong runtime's guard did not forward this later-added hook.
        pass

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        # The saved strong runtime's guard did not forward this later-added hook.
        pass

    def _call(self, windows, masks, previous, use_context):
        count = len(self.events)
        result = super()._call(windows, masks, previous, use_context)
        trace = self.policy.last_trace
        if not isinstance(trace, dict) or len(self.events) != count + 1:
            raise ValueError("the one-student correction trace is missing")
        self.events[-1]["student_decision"] = copy.deepcopy(trace["student"])
        self.events[-1]["student_input_frame"] = trace["input_frame"]
        self.events[-1]["nominal_correction"] = copy.deepcopy(trace["nominal_correction"])
        return result


def make_guard(student, arm):
    """Create one of three comparison arms around the same frozen student."""
    if arm not in ARMS:
        raise ValueError("unknown development comparison arm")
    if student.training or any(p.requires_grad for p in student.parameters()):
        raise ValueError("the single Terra student must be frozen in evaluation mode")
    cls = {"raw": RawStudent, "canonical_only": CanonicalOnly,
           "canonical_grasp_placement": CanonicalGraspPlacement}[arm]
    return TracedGuard(cls(student)).eval()
