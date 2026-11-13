"""Separate contextual and range-limited students with registered correction."""
from __future__ import annotations

import torch

from raccoon_memory import motion_policy
from raccoon_memory import terra_kinematic_correction as traced
from raccoon_memory import terra_registered_nominal_correction as registered


class ContextualStudentProbe(traced.StudentProbe):
    """Pass the executor's actual previous target to the contextual student."""

    def forward_with_control_state(self, windows, masks, previous):
        self.calls += 1
        logits, angles = self.student.forward_with_control_state(windows, masks, previous)
        if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 4)
                or not isinstance(angles, torch.Tensor) or tuple(angles.shape) != (1, 4)):
            raise ValueError("the contextual student must return one action and four joint proposals")
        self.last_logits = logits.detach().clone()
        self.last = {"action_logits": logits.detach().cpu().tolist()[0],
                     "action": motion_policy.ACTIONS[int(logits.argmax(dim=1).item())],
                     "joints_in_student_frame_deg": traced._degrees(angles),
                     "input_window": windows.detach().cpu().tolist()[0],
                     "input_mask": masks.detach().cpu().tolist()[0],
                     "input_previous_target_deg": previous.detach().cpu().tolist()[0]}
        return logits, angles


class RegisteredContextual(registered.RegisteredGraspPlacement):
    def __init__(self, student):
        if not callable(getattr(student, "forward_with_control_state", None)):
            raise ValueError("the unified student requires its previous-control interface")
        super().__init__(student)
        self.reference = ContextualStudentProbe(student)


def make_guard(student, candidate_kind):
    if student.training or any(parameter.requires_grad for parameter in student.parameters()):
        raise ValueError("the single candidate must be frozen in evaluation mode")
    if candidate_kind == "unified":
        policy = RegisteredContextual(student)
    elif candidate_kind == "feasible":
        policy = registered.RegisteredGraspPlacement(student)
    else:
        raise ValueError("unknown student architecture")
    return traced.TracedGuard(policy).eval()
