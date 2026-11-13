"""Attach the existing nominal MOVE correction to the frozen registration GRU.

Only the registration student is evaluated.  Its actual public input, logits,
hidden state, and command callbacks remain unchanged.  The inherited continuous
grasp/placement functions act once on MOVE targets in the original registered
frame; the existing joint guard remains outside this module's correction layer.
This is a development-only simulated execution adapter, not a new checkpoint.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import motion_policy, scene_oracle_free_session as session
from . import scene_registration_distilled_policy as student_policy
from .scene_placement_belief_policy import GraspPlacementBeliefCanonicalPolicy


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/architecture_research/20261103_public_registration_distilled_seed20261101_v1/candidate.pt"
CHECKPOINT_SHA256 = "61a66c3497a8192ebe2946d734fb1c36f342f03f42f7fc209f6ef901de737585"
GRASP_TCP_ABOVE_CENTER_MM = 5.0


class RegistrationNominalCorrection(GraspPlacementBeliefCanonicalPolicy):
    """Existing correction functions with a single, untransformed student input."""

    def __init__(self, adapter):
        if not isinstance(adapter, student_policy.RegistrationAdapter):
            raise ValueError("the registration student's actual callback adapter is required")
        super().__init__(adapter, grasp_tcp_above_center_mm=GRASP_TCP_ABOVE_CENTER_MM)
        self.last_motion_trace = None
        self._raw_student = None

    @property
    def registration_adapter(self):
        return self.reference

    def _forward_in_frame(self, windows, masks, previous, *, source_anchor=None):
        # The distilled student already consumes the real registered frame.
        # Its inputs and recurrent memory must not be canonicalized or patched.
        self.last_diagnostic = {"frame_transform": "none", "output_rejection": None}
        logits, angles = self.reference.forward_with_control_state(windows, masks, previous)
        self._raw_student = (logits.detach().clone(), angles.detach().clone())
        return logits, angles

    def forward_with_control_state(self, windows, masks, previous):
        self._raw_student = None
        logits, corrected = super().forward_with_control_state(windows, masks, previous)
        if self._raw_student is None or not torch.equal(logits, self._raw_student[0]):
            raise ValueError("the correction changed student logits or did not call the student once")
        is_move = int(logits.argmax()) == motion_policy.MOVE
        raw = [float(q) * 180. for q in self._raw_student[1][0].tolist()] if is_move else None
        changed = [float(q) * 180. for q in corrected[0].tolist()] if is_move else None
        self.last_motion_trace = {"student_action": motion_policy.ACTIONS[int(logits.argmax())],
                                  "raw_student_joints_deg": raw, "corrected_joints_deg": changed,
                                  "actual_applied_joints_deg": None}
        return logits, corrected

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        super().on_scene_registered(scene, nominal_slot_xy_mm)
        self.reference.on_scene_registered(scene, nominal_slot_xy_mm)
        self.last_motion_trace = self._raw_student = None

    def on_transfer_started(self, before, task, applied_targets):
        self.reference.on_transfer_started(before, task, applied_targets)

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        if self.last_motion_trace is None or self.last_motion_trace["student_action"] != action:
            raise ValueError("control callback has no matching student decision")
        self.reference.on_control_applied(action, transmitted_target_deg, applied_target_deg, gripper_opening_m)
        self.last_motion_trace["actual_applied_joints_deg"] = [float(x) for x in applied_target_deg]

    def on_expected_transfer_committed(self, before, task, after):
        self.reference.on_expected_transfer_committed(before, task, after)
        super().on_expected_transfer_committed(before, task, after)


class RegistrationNominalGuard(GuardedPolicy):
    """Include the raw and corrected MOVE in the existing guarded-command log."""

    def __init__(self, correction):
        if not isinstance(correction, RegistrationNominalCorrection):
            raise ValueError("the one-student nominal correction is required")
        super().__init__(correction)

    def _call(self, windows, masks, previous, use_context):
        result = super()._call(windows, masks, previous, use_context)
        trace = self.policy.last_motion_trace
        if not isinstance(trace, dict):
            raise ValueError("the MOVE correction did not publish a decision trace")
        self.events[-1]["raw_student_joints_deg"] = trace["raw_student_joints_deg"]
        self.events[-1]["corrected_joints_deg"] = trace["corrected_joints_deg"]
        return result

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        if not self.events or self.events[-1].get("model_action") != action:
            raise ValueError("the guard has no matching decision for the applied command")
        super().on_control_applied(action, transmitted_target_deg, applied_target_deg, gripper_opening_m)
        self.events[-1]["actual_applied_joints_deg"] = [float(x) for x in applied_target_deg]


def load_guarded_checkpoint(path=CHECKPOINT):
    """Load exactly the frozen 1,400-update registration student once."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("the fixed registration student fingerprint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    training = payload.get("training", {})
    if training.get("updates") != 1400 or training.get("bc_updates") != 1200 or training.get("fk_updates") != 200:
        raise ValueError("the checkpoint training contract differs")
    student = student_policy.load_checkpoint(payload)
    student.requires_grad_(False)
    return RegistrationNominalGuard(RegistrationNominalCorrection(student_policy.RegistrationAdapter(student))).eval()


class RegistrationNominalSceneSession(session.OracleFreeSceneModelSession):
    """Wire the existing virtual executor's confirmed public return observation."""

    def __init__(self, sim, memory, student, **kwargs):
        if not isinstance(student, RegistrationNominalGuard):
            raise ValueError("the single-student nominal correction and joint guard are required")
        super().__init__(sim, memory, student, **kwargs)
        student.policy.registration_adapter.return_joint_observer = lambda: session._robot_observation(sim, None)["joints_deg"]
