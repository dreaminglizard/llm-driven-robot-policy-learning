"""Simulation-only direct position and signed-velocity commands.

This adapter makes no intermediate arm targets. Its velocity term is a
feed-forward addition to the existing MuJoCo position servo, clipped together
with that servo's force at each physics step. It is not a hardware controller.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from raccoon_control.kinematics import JOINT_LIMITS
from . import scene_features, task_catalog
from .native_approach_session import PublicCustomSlotsSceneMotionSim


PERIOD_SECONDS = 0.02
PHYSICS_STEPS = 10
# Research envelope matching the policy's velocity scale, not a hardware limit.
MAX_VELOCITY_DEG_S = 180.0


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must have four values")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} must contain finite numbers")
    return [float(x) for x in value]


def validate_proposal(value):
    """Validate the entire model proposal before changing any control."""
    if not isinstance(value, dict):
        raise ValueError("student proposal must be an object")
    joints = _vector(value.get("joint_target_deg"), "joint target")
    velocity = _vector(value.get("joint_velocity_deg_s"), "joint velocity")
    if any(not lo <= q <= hi for q, (lo, hi) in zip(joints, JOINT_LIMITS, strict=True)):
        raise ValueError("joint target outside the virtual joint limits")
    if any(abs(v) > MAX_VELOCITY_DEG_S for v in velocity):
        raise ValueError("joint velocity outside the virtual research envelope")
    if value.get("gripper") not in ("open", "closed") or type(value.get("done")) is not bool:
        raise ValueError("invalid gripper or DONE")
    period = value.get("step_seconds")
    if type(period) not in (int, float) or not math.isfinite(period) or not math.isclose(
            period, PERIOD_SECONDS, abs_tol=1e-12, rel_tol=0):
        raise ValueError("student period differs from 0.02 seconds")
    return {"joint_target_deg": joints, "joint_velocity_deg_s": velocity,
            "gripper": value["gripper"], "done": value["done"], "step_seconds": PERIOD_SECONDS}


def _json_safe(value):
    """Preserve rejected output evidence without writing non-standard JSON."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return f"<unsupported {type(value).__name__}>"


class DirectVelocityServo:
    """Apply one validated student proposal for exactly ten 0.002 s steps.

    The existing actuator already contains position feedback and velocity
    damping. Desired velocity adds ``kv * desired_velocity`` in joint torque.
    Each tick measures the existing actuator force with the selected target,
    then adds only the difference between that force and their clipped sum.
    The four arm external forces must initially be zero and are cleared when
    the command returns or raises; non-arm external inputs are not touched.
    """

    def __init__(self, sim):
        if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
            raise TypeError("the public-only virtual simulator is required")
        self.sim = sim
        model = sim.model
        if not math.isclose(float(model.opt.timestep), .002, abs_tol=1e-12, rel_tol=0):
            raise ValueError("the direct adapter requires a 0.002 s physics timestep")
        self.dofs = [int(model.jnt_dofadr[j]) for j in sim._joint_ids]
        if len(set(self.dofs)) != 4:
            raise ValueError("arm joints must have distinct scalar degrees of freedom")
        self.kv, self.limits = [], []
        for joint, actuator in zip(sim._joint_ids, sim._arm_actuators, strict=True):
            # qfrc_applied is joint torque. Reject transmissions for which a
            # scalar actuator force does not equal that same joint torque.
            gear = np.asarray(model.actuator_gear[actuator], dtype=float)
            force_range = np.asarray(model.actuator_forcerange[actuator], dtype=float)
            bias = np.asarray(model.actuator_biasprm[actuator], dtype=float)
            if (int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                    or int(model.actuator_trnid[actuator, 0]) != joint
                    or not np.array_equal(gear, [1., 0., 0., 0., 0., 0.])
                    or not bool(model.actuator_forcelimited[actuator])
                    or not np.isfinite(force_range).all() or force_range[0] >= force_range[1]
                    or int(model.actuator_dyntype[actuator]) != int(mujoco.mjtDyn.mjDYN_NONE)
                    or int(model.actuator_gaintype[actuator]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                    or int(model.actuator_biastype[actuator]) != int(mujoco.mjtBias.mjBIAS_AFFINE)
                    or not np.isfinite(bias).all() or bias[2] >= 0):
                raise ValueError("unsupported arm servo transmission or force configuration")
            self.kv.append(float(-bias[2]))
            self.limits.append(force_range.copy())
        self.limits = np.asarray(self.limits)

    def apply(self, proposal, *, tick_observer=None):
        command = validate_proposal(proposal)
        if command["done"]:
            raise ValueError("DONE is a termination, not a motor command")
        sim, data = self.sim, self.sim.data
        if sim.layout_status == "uncertain":
            raise RuntimeError("cannot command an uncertain virtual scene")
        if not np.array_equal(data.qfrc_applied[self.dofs], np.zeros(4)):
            raise ValueError("arm external forces must be zero before a direct command")
        for actuator, angle in zip(sim._arm_actuators, command["joint_target_deg"], strict=True):
            data.ctrl[actuator] = math.radians(angle)
        opening = .04 if command["gripper"] == "open" else 0.
        for actuator in sim._finger_actuators:
            data.ctrl[actuator] = opening / 2.
        desired = np.radians(command["joint_velocity_deg_s"]) * np.asarray(self.kv)
        try:
            for step in range(PHYSICS_STEPS):
                # Skip the extra forward call for zero velocity to preserve the
                # original position servo's exact integration behavior.
                if np.any(desired):
                    mujoco.mj_forward(sim.model, data)
                    original = np.asarray(data.actuator_force[sim._arm_actuators]).copy()
                    total = np.clip(original + desired, self.limits[:, 0], self.limits[:, 1])
                    data.qfrc_applied[self.dofs] = total - original
                else:
                    original = total = None
                if tick_observer is not None:
                    tick_observer(step, original, total, np.asarray(data.qfrc_applied[self.dofs]).copy())
                mujoco.mj_step(sim.model, data)
        finally:
            # No virtual time elapses between this clear and the next proposal.
            # A DONE hold therefore starts with the original zero-speed servo.
            data.qfrc_applied[self.dofs] = 0.
        mujoco.mj_forward(sim.model, data)
        return command


class PublicObservation:
    """Only encoder/finger readings; estimate moving from past public angles."""

    def __init__(self, sim):
        self.sim, self.previous = sim, None

    def read(self, gripper):
        sim = self.sim
        joints = [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids]
        opening = float(sum(sim.data.qpos[sim.model.jnt_qposadr[j]] for j in sim._finger_ids))
        now = float(sim.data.time)
        if not all(math.isfinite(x) for x in [*joints, opening, now]):
            raise ValueError("invalid public robot observation")
        moving = False
        if self.previous is not None:
            previous_time, previous_joints = self.previous
            elapsed = now - previous_time
            if elapsed <= 0:
                raise ValueError("public observation time did not advance")
            moving = max(abs(a-b) / elapsed for a, b in zip(joints, previous_joints, strict=True)) > .1
        self.previous = (now, joints)
        return {"schema_version": "PolicyObservation.phase1", "source": "mujoco_simulation_derived",
                "simulated": True, "observed_monotonic": now, "joints_deg": joints,
                "sensor_valid": True, "moving": moving, "gripper_command": gripper,
                "gripper_reported_state": "open" if opening >= .03 else "closed",
                "gripper_reported_valid": True, "gripper_feedback_fresh": True}


def run_direct_transfer(runtime, sim, before, task, *, max_ticks):
    """Run without physics assessment; DONE preserves the last applied targets."""
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError("max_ticks must be a positive integer")
    if not math.isclose(float(runtime.step_seconds), PERIOD_SECONDS, abs_tol=1e-12, rel_tol=0):
        raise ValueError("student runtime period differs from 0.02 seconds")
    actuation = getattr(runtime, "actuation_semantics", "position_target_with_velocity_force")
    terminal = getattr(runtime, "terminal_semantics", "done_no_command")
    if actuation != "position_target_with_velocity_force":
        raise ValueError("checkpoint requires a different actuation adapter")
    if terminal not in ("done_no_command", "apply_then_finish"):
        raise ValueError("unsupported terminal semantics")
    before, task = scene_features.transfer_task(before, task)
    if task_catalog.scene(sim._committed_layout) != before or sim.layout_status != "registered":
        raise ValueError("initial expected layout differs")
    servo = DirectVelocityServo(sim)
    observation = PublicObservation(sim)
    sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
    rows, gripper = [], "open"
    for index in range(max_ticks):
        obs = observation.read(gripper)
        raw_output = None
        try:
            raw_output = runtime.step(obs)
            if (not isinstance(raw_output, dict)
                    or raw_output.get("terminal_semantics", terminal) != terminal
                    or raw_output.get("actuation_semantics", actuation) != actuation):
                raise ValueError("runtime output semantics differ from its checkpoint")
            output = validate_proposal(raw_output)
        except Exception as exc:
            return {"termination": "invalid_or_failed_student_output", "error_type": type(exc).__name__,
                    "error": str(exc), "rejected_observation": obs,
                    "rejected_proposal": _json_safe(raw_output), "ticks": rows, "physical_success": None}
        row = {"index": index, "observation": obs, "proposal": output}
        rows.append(row)
        if output["done"] and terminal == "done_no_command":
            row["result"] = "done_no_command_velocity_zero"
            return {"termination": "model_done", "ticks": rows, "physical_success": None}
        try:
            # The existing servo accepts commands only, so terminal status is
            # recorded separately and never changes the motor command itself.
            servo.apply({**output, "done": False})
        except Exception as exc:
            row["result"] = "command_error"
            return {"termination": "command_error", "error_type": type(exc).__name__,
                    "error": str(exc), "ticks": rows, "physical_success": None}
        row["result"] = "applied_for_10_physics_steps"
        gripper = output["gripper"]
        if output["done"]:
            row["result"] = "applied_last_10_physics_steps_then_done_velocity_zero"
            return {"termination": "model_done", "ticks": rows, "physical_success": None}
    return {"termination": "tick_limit", "ticks": rows, "physical_success": None}
