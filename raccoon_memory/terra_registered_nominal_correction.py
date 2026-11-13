"""Nominal grasp/placement correction in the Terra student's registered frame.

The student receives the original public feature window and is called once.
Only its selected MOVE proposal is subsequently changed by the existing
conditional nominal correction; the existing joint guard remains outside.
This is a distinct development experiment, not the full strong configuration.
"""
from __future__ import annotations

from raccoon_memory import terra_kinematic_correction as traced


class RegisteredGraspPlacement(traced.CanonicalGraspPlacement):
    """Reuse the existing tracker and corrections without changing the frame."""

    def _forward_in_frame(self, windows, masks, previous, *, source_anchor=None):
        # This is the same frame treatment as RegistrationNominalCorrection:
        # the student sees the unchanged registered observation and mask.
        self.last_diagnostic = {"frame_transform": "none", "output_rejection": None}
        logits, joints = self.reference.forward_with_control_state(windows, masks, previous)
        self._mapped_proposal = joints.detach().clone()
        return logits, joints

    def forward_with_control_state(self, windows, masks, previous):
        logits, joints = super().forward_with_control_state(windows, masks, previous)
        self.last_trace["input_frame"] = "registered"
        return logits, joints


def make_guard(student):
    """Wrap the same frozen student in one correction and one original guard."""
    if student.training or any(parameter.requires_grad for parameter in student.parameters()):
        raise ValueError("the single Terra student must be frozen in evaluation mode")
    return traced.TracedGuard(RegisteredGraspPlacement(student)).eval()
