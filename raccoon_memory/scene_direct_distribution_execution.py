"""Execute a direct-only distribution student through the common session API.

The candidate selects the action and the absolute MOVE target. This adapter
only supplies public history and executor-confirmed prior control. It does
not run IK, replace actions, select another policy, or inspect physical task
assessments. Use the ordinary joint guard outside this adapter.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy
from .scene_applied_action_policy import AppliedActionPolicy
from .scene_direct_distribution_policy import DistributionPolicy


class DistributionExecutionPolicy(AppliedActionPolicy):
    """Keep one recurrent candidate across transfers and await command returns."""

    def __init__(self, student):
        nn.Module.__init__(self)
        if not isinstance(student, DistributionPolicy):
            raise TypeError("a direct distribution student is required")
        # Register the entire candidate exactly once, including its MOVE head.
        self.student = student
        self._scene = self._slots = None
        self._reset_runtime()
        self.eval()

    @property
    def model(self):
        # The inherited public-row validator uses only action_head's dtype and
        # device. All inference and serialization use the full `student`.
        return self.student.backbone

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self._task is None or self._previous_target is None:
            raise ValueError("confirmed transfer-start control is required")
        if self._pending_action is not None or self._done:
            raise ValueError("the previous decision still requires command return or expected commit")
        row = self._row(windows, masks, previous_target_degrees)
        output = self.student.predict(row, masks[:, -1:], self._state)
        logits = output.action_logits[:, 0]
        if (tuple(logits.shape) != (1, len(motion_policy.ACTIONS))
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("distribution student returned invalid action logits")
        index = int(logits.argmax(dim=-1).item())
        if output.actions.shape != (1, 1) or int(output.actions[0, 0]) != index:
            raise ValueError("distribution prediction disagrees with its action logits")
        action = motion_policy.ACTIONS[index]
        if action == "move":
            joints = output.move_joints_deg[:, 0]
            limits = joints.new_tensor(JOINT_LIMITS)
            if (tuple(joints.shape) != (1, 4) or not bool(torch.isfinite(joints).all())
                    or bool(((joints < limits[:, 0]) | (joints > limits[:, 1])).any())):
                raise ValueError("distribution student returned invalid MOVE joints")
        else:
            # Non-MOVE heads deliberately contain NaN: they supply no target.
            # The executor holds its last confirmed joints for OPEN/CLOSE/DONE.
            joints = row.new_tensor([self._previous_target])
        if (not isinstance(output.state, tuple) or len(output.state) != 2
                or any(not isinstance(value, torch.Tensor)
                       or value.shape != (1, 1, self.student.backbone.lstm.hidden_size)
                       or value.device != row.device or value.dtype != row.dtype
                       or not bool(torch.isfinite(value).all()) for value in output.state)):
            raise ValueError("distribution student returned invalid recurrent state")
        self._state = tuple(value.detach() for value in output.state)
        self._first_registration_row = self._first_transfer_row = False
        if action == "done":
            self._done = True
        else:
            self._pending_action = action
        return logits, joints / motion_policy.JOINT_SCALE_DEG


def load_verified_checkpoint(path, expected_sha256):
    """Reuse the trainer's checked artifact loader without fitting a model.

    The caller still binds current runtime sources and owns development/final
    eligibility. Loading a checkpoint does not establish physical performance.
    """
    from scripts.train_direct_distribution_pair import load_checkpoint

    model, payload = load_checkpoint(path, expected_sha256)
    return DistributionExecutionPolicy(model), payload
