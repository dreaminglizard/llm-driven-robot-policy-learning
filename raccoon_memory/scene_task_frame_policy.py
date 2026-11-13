"""Confirmed-control adapter with exact-pose nominal IK for TaskFrameStudent.

The learned policy alone selects MOVE/OPEN/CLOSE/DONE. A MOVE is rejected if
the requested world XYZ and unwrapped pitch cannot both be achieved within
joint bounds. No nearest-pitch repair, scene attraction or object sensing is
used. Control history advances only after the executor's validated return.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import motion_policy
from .scene_applied_action_policy import AppliedActionPolicy
from . import scene_task_frame_student as student


class PoseIKError(ValueError):
    """A proposed Cartesian MOVE has no exact permitted nominal solution."""


def exact_pose_joints(world_pose, *, geometry=None, reference_joints_deg=None):
    """Enumerate both signed radial branches, preserving XYZ and pitch.

    The negative-elbow joint range fixes the elbow branch. Base yaw can still
    differ by 180 degrees when the arm's signed TCP radius is negative. If
    both solutions satisfy every bound, prefer the one closest to the public
    last applied joints; otherwise prefer the positive-radius solution. The
    zero-radius yaw singularity is explicitly unsupported rather than guessed.
    """
    if (not isinstance(world_pose, (tuple, list)) or len(world_pose) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in world_pose)):
        raise PoseIKError("task-frame IK requires finite world XYZ and pitch")
    geometry = Geometry() if geometry is None else geometry
    if (not isinstance(geometry, Geometry) or any(not math.isfinite(value) for value in
            (geometry.base_mm, geometry.upper_mm, geometry.forearm_mm,
             geometry.tool_forward_mm, geometry.tool_vertical_mm))
            or min(geometry.upper_mm, geometry.forearm_mm) <= 0):
        raise PoseIKError("finite nominal geometry with positive arm lengths is required")
    if reference_joints_deg is not None and (not isinstance(reference_joints_deg, (list, tuple))
            or len(reference_joints_deg) != 4 or any(type(q) not in (int, float) or not math.isfinite(q)
            or not low <= q <= high for q, (low, high) in zip(reference_joints_deg, JOINT_LIMITS))):
        raise PoseIKError("IK branch reference must be confirmed joints within limits")
    x, y, z, degrees = map(float, world_pose)
    radius = math.hypot(x, y)
    if radius < 1e-6:
        raise PoseIKError("task-frame IK rejects the undefined zero-radius base yaw")
    pitch = math.radians(degrees)
    solutions = []
    for sign in (1, -1):
        yaw = math.degrees(math.atan2(-sign * x, sign * y))
        if not JOINT_LIMITS[0][0] <= yaw <= JOINT_LIMITS[0][1]:
            continue
        arm_r = sign * radius - geometry.tool_forward_mm * math.cos(pitch) + geometry.tool_vertical_mm * math.sin(pitch)
        arm_z = z - geometry.base_mm - geometry.tool_forward_mm * math.sin(pitch) - geometry.tool_vertical_mm * math.cos(pitch)
        cosine = (arm_r**2 + arm_z**2 - geometry.upper_mm**2 - geometry.forearm_mm**2) / (2 * geometry.upper_mm * geometry.forearm_mm)
        if not math.isfinite(cosine) or not -1 - 1e-10 <= cosine <= 1 + 1e-10:
            continue
        elbow = -math.acos(max(-1., min(1., cosine)))
        shoulder = math.atan2(-arm_r, arm_z) - math.atan2(geometry.forearm_mm * math.sin(elbow),
            geometry.upper_mm + geometry.forearm_mm * math.cos(elbow))
        for turns in (-1, 0, 1):
            q2 = math.degrees(shoulder) + 360 * turns
            q3 = math.degrees(elbow)
            joints = [yaw, q2, q3, degrees - 90.0 - q2 - q3]
            if any(q < low - 1e-7 or q > high + 1e-7 for q, (low, high) in zip(joints, JOINT_LIMITS)):
                continue
            # Remove numerical boundary roundoff only, never repair a pose.
            joints = [max(low, min(high, q)) for q, (low, high) in zip(joints, JOINT_LIMITS)]
            error = math.dist(geometry.forward(joints), world_pose[:3])
            pitch_error = sum(joints[1:]) + 90.0 - degrees
            if error <= 1e-5 and abs(pitch_error) <= 1e-7:
                cost = (sum((q - r)**2 for q, r in zip(joints, reference_joints_deg))
                        if reference_joints_deg is not None else 0.)
                solutions.append((cost, sign != 1, joints, error, pitch_error, sign))
    if not solutions:
        raise PoseIKError("task-frame exact-pose IK rejected the proposed MOVE: no joint-bounded XYZ/pitch solution")
    _, _, joints, error, pitch_error, sign = min(solutions, key=lambda item: (item[0], item[1], item[2]))
    return joints, {"world_pose_mm_deg": list(world_pose), "solver_position_error_mm": error,
                    "solver_pitch_error_deg": pitch_error, "pitch_adjusted": False,
                    "signed_radius_branch": sign, "valid_solution_count": len(solutions),
                    "branch_selection": "nearest_confirmed_joints" if reference_joints_deg is not None
                                        else "positive_radius_first"}


class TaskFramePolicy(AppliedActionPolicy):
    def __init__(self, model):
        nn.Module.__init__(self)
        if not isinstance(model, student.TaskFrameStudent):
            raise TypeError("a TaskFrameStudent is required")
        self.model = model
        self._scene = self._slots = None
        self._reset_runtime()
        self.eval()

    def _reset_runtime(self):
        super()._reset_runtime()
        self.pose_events = []

    def _nominal_error(self, target, world_pose):
        geometry = Geometry(*self.model.nominal_constants[:5].cpu().tolist())
        return {"position_error_mm": math.dist(geometry.forward(target), world_pose[:3]),
                "pitch_error_deg": sum(target[1:]) + 90.0 - world_pose[3],
                "joints_deg": list(target), "physical_measurement": False}

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        super().on_control_applied(action, transmitted_target_deg, applied_target_deg, gripper_opening_m)
        if action == "move":
            event = self.pose_events[-1]
            event["confirmed_command_nominal_error"] = self._nominal_error(
                applied_target_deg, event["ik"]["world_pose_mm_deg"])

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self._task is None or self._previous_target is None:
            raise ValueError("confirmed transfer-start control is required")
        if self._pending_action is not None or self._done:
            raise ValueError("the previous decision still requires command return or expected commit")
        row = self._row(windows, masks, previous_target_degrees)
        output = self.model(row, masks[:, -1:], self._state)
        logits, pose = output.action_logits[:, 0], output.move_pose[:, 0]
        if (logits.shape != (1, 4) or pose.shape != (1, 4)
                or not bool(torch.isfinite(logits).all() and torch.isfinite(pose).all())):
            raise ValueError("task-frame student returned invalid action or pose")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        target = list(self._previous_target)
        event = {"action": action, "relative_pose_normalized": pose[0].cpu().tolist(), "ik": None}
        if action == "move":
            # Decode in double precision to avoid an extra float32 rounding of
            # registered mm and predicted values before the exact IK check.
            constants = self.model.nominal_constants.double()
            world = student.pose_to_world(row[0, 0].double(), pose[0].double(), constants).cpu().tolist()
            try:
                target, diagnostic = exact_pose_joints(world, geometry=Geometry(*constants[:5].cpu().tolist()),
                                                      reference_joints_deg=self._previous_target)
            except PoseIKError as exc:
                event.update(status="ik_rejected_no_command", world_pose_mm_deg=world, error=str(exc))
                self.pose_events.append(event)
                raise
            event.update(status="exact_pose_proposal", ik=diagnostic)
        else:
            event["status"] = "non_move_pose_ignored"
        normalized_target = logits.new_tensor([target]) / motion_policy.JOINT_SCALE_DEG
        if action == "move":
            returned = [float(q) * motion_policy.JOINT_SCALE_DEG for q in normalized_target[0].cpu().tolist()]
            event["solver_joints_deg"] = target
            event["policy_returned_nominal_error"] = self._nominal_error(returned, world)
            event["confirmed_command_nominal_error"] = None
        self.pose_events.append(event)
        self._state = tuple(value.detach() for value in output.state)
        self._first_registration_row = self._first_transfer_row = False
        if action == "done":
            self._done = True
        else:
            self._pending_action = action
        # Existing executor/guard API uses normalized joint angles. Only the
        # subsequent command-return callback may mark this proposal applied.
        return logits, normalized_target


def load_checkpoint(payload):
    return TaskFramePolicy(student.load_checkpoint(payload))
