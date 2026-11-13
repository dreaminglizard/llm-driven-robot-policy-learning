"""Simulation-only actuator-level integration of student position and speed.

Each student's 20ms proposal supplies a *current position-servo target* and
a signed angular velocity. The actuator integrates that velocity for ten 2ms
physics steps; it does not generate an independent path or object trajectory.
The target is an actuator setpoint, not a separately submitted teacher angle.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from raccoon_control.kinematics import JOINT_LIMITS
from . import scene_features, task_catalog
from .native_approach_session import PublicCustomSlotsSceneMotionSim
from .scene_direct_velocity_sim import (PERIOD_SECONDS, PHYSICS_STEPS, PublicObservation,
                                        _json_safe, validate_proposal)


ACTUATION_SEMANTICS = 'integrated_velocity_with_position_target'
POSITION_LABEL = 'current_applied_position_servo_target'
TERMINAL_SEMANTICS = frozenset(('done_no_command', 'apply_then_finish'))


class IntegratedVelocityServo:
    """Consume the explicit current servo target and integrate velocity at 2ms."""

    def __init__(self, sim, *, near_limit_guard=False):
        if type(near_limit_guard) is not bool:
            raise ValueError('near_limit_guard must be a boolean')
        if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
            raise TypeError('the public-only virtual simulator is required')
        if not math.isclose(float(sim.model.opt.timestep), .002, abs_tol=1e-12, rel_tol=0):
            raise ValueError('the integrated adapter requires a 0.002 s physics timestep')
        self.sim = sim
        self.near_limit_guard = near_limit_guard
        if near_limit_guard:
            # Use the existing virtual guard's threshold without adding its
            # torch dependency to the original, guard-disabled simulator path.
            from raccoon_control.near_joint_limit_policy import MAX_OVERSHOOT_DEG
            self.max_guard_overshoot_deg = MAX_OVERSHOOT_DEG
        # Set only for a guard decision that modifies or rejects an endpoint.
        # It is a proposal/decision record, not evidence of actual joint motion.
        self.last_limit_guard = None
        self.dofs = [int(sim.model.jnt_dofadr[j]) for j in sim._joint_ids]
        if len(set(self.dofs)) != 4:
            raise ValueError('arm joints must have distinct scalar degrees of freedom')

    def apply(self, proposal, *, tick_observer=None):
        self.last_limit_guard = None
        command = validate_proposal(proposal)
        # Both endpoints are checked before any control changes. Each axis is
        # monotone while one directly supplied velocity is held constant.
        start = command['joint_target_deg']
        speed = command['joint_velocity_deg_s']
        end = [q + PERIOD_SECONDS * v for q, v in zip(start, speed, strict=True)]
        if any(not lo <= q <= hi for q, (lo, hi) in zip(end, JOINT_LIMITS, strict=True)):
            if not self.near_limit_guard:
                raise ValueError('integrated joint target outside the virtual joint limits')
            excess = [max(lo - q, 0., q - hi) for q, (lo, hi) in zip(end, JOINT_LIMITS, strict=True)]
            largest = max(excess)
            self.last_limit_guard = {
                'status': 'clamped' if largest <= self.max_guard_overshoot_deg else 'exceeds_guard_range_rejected',
                'max_allowed_overshoot_deg': self.max_guard_overshoot_deg,
                'raw_end_deg': list(end), 'overshoot_deg': excess,
                'bounded_end_deg': [min(max(q, lo), hi) for q, (lo, hi) in zip(end, JOINT_LIMITS, strict=True)],
            }
            if largest > self.max_guard_overshoot_deg:
                raise ValueError('integrated joint target outside the virtual joint limits')
        sim, data = self.sim, self.sim.data
        if sim.layout_status == 'uncertain':
            raise RuntimeError('cannot command an uncertain virtual scene')
        if not np.array_equal(data.qfrc_applied[self.dofs], np.zeros(4)):
            raise ValueError('arm external forces must be zero before an integrated command')
        for actuator in sim._finger_actuators:
            data.ctrl[actuator] = .02 if command['gripper'] == 'open' else 0.
        for tick in range(PHYSICS_STEPS):
            elapsed = (tick + 1) * float(sim.model.opt.timestep)
            target = [q + elapsed * v for q, v in zip(start, speed, strict=True)]
            if self.last_limit_guard is not None:
                target = [min(max(q, lo), hi) for q, (lo, hi) in zip(target, JOINT_LIMITS, strict=True)]
            for actuator, angle in zip(sim._arm_actuators, target, strict=True):
                data.ctrl[actuator] = math.radians(angle)
            if tick_observer is not None:
                tick_observer(tick, list(target))
            mujoco.mj_step(sim.model, data)
        mujoco.mj_forward(sim.model, data)
        return command


def run_integrated_transfer(runtime, sim, before, task, *, max_ticks, near_limit_guard=False):
    """Run a direct-command checkpoint without any live physical assessment."""
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError('max_ticks must be a positive integer')
    if not math.isclose(float(runtime.step_seconds), PERIOD_SECONDS, abs_tol=1e-12, rel_tol=0):
        raise ValueError('student runtime period differs from 0.02 seconds')
    actuation = getattr(runtime, 'actuation_semantics', None)
    label = getattr(runtime, 'direct_position_label', None)
    terminal = getattr(runtime, 'terminal_semantics', None)
    if actuation != ACTUATION_SEMANTICS or label != POSITION_LABEL:
        raise ValueError('checkpoint does not declare the integrated current-servo-target contract')
    if terminal not in TERMINAL_SEMANTICS:
        raise ValueError('checkpoint does not declare supported terminal semantics')
    before, task = scene_features.transfer_task(before, task)
    if task_catalog.scene(sim._committed_layout) != before or sim.layout_status != 'registered':
        raise ValueError('initial expected layout differs')
    servo, observation = IntegratedVelocityServo(sim, near_limit_guard=near_limit_guard), PublicObservation(sim)
    sim.begin_transfer(before, task['source_slot'], task['destination_slot'])
    rows, gripper = [], 'open'
    for index in range(max_ticks):
        obs, raw = observation.read(gripper), None
        try:
            raw = runtime.step(obs)
            if (not isinstance(raw, dict) or raw.get('actuation_semantics') != actuation
                    or raw.get('direct_position_label') != label or raw.get('terminal_semantics') != terminal):
                raise ValueError('runtime output semantics differ from its checkpoint')
            output = validate_proposal(raw)
        except Exception as exc:
            return {'termination': 'invalid_or_failed_student_output', 'error_type': type(exc).__name__,
                    'error': str(exc), 'rejected_observation': obs, 'rejected_proposal': _json_safe(raw),
                    'ticks': rows, 'physical_success': None}
        row = {'index': index, 'observation': obs, 'proposal': output}
        rows.append(row)
        if output['done'] and terminal == 'done_no_command':
            row['result'] = 'done_no_command_velocity_zero'
            return {'termination': 'model_done', 'ticks': rows, 'physical_success': None}
        try:
            servo.apply(output)
        except Exception as exc:
            row['result'] = 'command_error'
            if servo.last_limit_guard is not None:
                row['joint_limit_guard'] = servo.last_limit_guard
            return {'termination': 'command_error', 'error_type': type(exc).__name__, 'error': str(exc),
                    'ticks': rows, 'physical_success': None}
        if servo.last_limit_guard is not None:
            row['joint_limit_guard'] = servo.last_limit_guard
        gripper = output['gripper']
        row['after_time_seconds'] = float(sim.data.time)
        row['observed_after_joints_deg'] = [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]]))
                                            for j in sim._joint_ids]
        row['applied_target_deg'] = [math.degrees(float(sim.data.ctrl[a])) for a in sim._arm_actuators]
        row['result'] = 'applied_for_10_physics_steps'
        if output['done']:
            row['result'] = 'applied_last_10_physics_steps_then_done_velocity_zero'
            return {'termination': 'model_done', 'ticks': rows, 'physical_success': None}
    return {'termination': 'tick_limit', 'ticks': rows, 'physical_success': None}
