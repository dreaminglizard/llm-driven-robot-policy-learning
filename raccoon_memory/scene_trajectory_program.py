"""Continuous, simulation-only programs for the 50 Hz trajectory student.

The existing v2 coordinator and symbolic planner own the program and GRU
memory. Each transfer gets a fresh single-transfer policy history, while one
public-only MuJoCo instance, its actuator targets, and the GRU remain alive.
The expected scene is advanced on DONE, not on a live physical assessment.
This research adapter cannot connect to motors or a microphone.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math

from . import scene_recall, scene_trajectory_policy
from . import scene_oracle_free_session as oracle
from .native_approach_session import MEMORY, MEMORY_SHA256, PublicCustomSlotsSceneMotionSim
from .scene_direct_integrated_velocity_sim import (
    ACTUATION_SEMANTICS, POSITION_LABEL, TERMINAL_SEMANTICS, IntegratedVelocityServo,
)
from .scene_direct_velocity_sim import PERIOD_SECONDS, PublicObservation, _json_safe, validate_proposal


def load_pinned_gru128():
    """Load the existing repaired GRU, without loading or replacing its old student."""
    data = MEMORY.read_bytes()
    if hashlib.sha256(data).hexdigest() != MEMORY_SHA256:
        raise ValueError('the existing GRU128 checkpoint fingerprint differs')
    model = scene_recall.load_scene_memory(io.BytesIO(data), device='cpu')
    if model.hidden_size != 128:
        raise ValueError('the existing memory checkpoint is not GRU128')
    return scene_recall.SceneRecall(model.eval().requires_grad_(False))


def _gripper_command(applied_targets):
    opening = applied_targets['gripper_opening_m']
    if math.isclose(opening, .04, abs_tol=1e-9, rel_tol=0):
        return 'open'
    if math.isclose(opening, 0., abs_tol=1e-9, rel_tol=0):
        return 'closed'
    raise ValueError('the applied finger target is not an open or closed command')


def _robot_boundary(sim):
    """Robot/control diagnostics, never used as privileged model or plan input."""
    return {
        'time_seconds': float(sim.data.time),
        'joints_deg': [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids],
        # Retained to audit physical continuity only. Policy moving is derived
        # causally by PublicObservation from public joint positions and time.
        'joint_velocity_deg_s_diagnostic_only': [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]]))
                                                 for j in sim._joint_ids],
        'applied_targets': oracle._servo_targets(sim),
        'finger_opening_m': float(sum(sim.data.qpos[sim.model.jnt_qposadr[j]] for j in sim._finger_ids)),
    }


class TrajectoryTransferRunner:
    """Bind one observation clock and actuator to a single live simulator."""

    def __init__(self, sim, *, runtime_factory=scene_trajectory_policy.TrajectoryPolicyRuntime):
        if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
            raise TypeError('a public-only virtual simulator is required')
        self.sim = sim
        self.observation = PublicObservation(sim)
        self.servo = IntegratedVelocityServo(sim)
        self.runtime_factory = runtime_factory

    def __call__(self, model, sim, before, task, *, max_decisions, device, cancel,
                 applied_targets, decisions, slot_xy_mm):
        if sim is not self.sim or device != 'cpu':
            raise ValueError('trajectory runner must retain its CPU virtual simulator')
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError('max_decisions must be positive')
        oracle._require_servo_targets(sim, applied_targets)
        if self.observation.previous is None:
            # The training contract labels the first observation as stationary
            # only when that initial condition was explicitly verified.
            velocities = _robot_boundary(sim)['joint_velocity_deg_s_diagnostic_only']
            if any(not math.isfinite(v) or abs(v) > .1 for v in velocities):
                raise ValueError('initial stationary observation has not been verified')
        runtime = self.runtime_factory(model, registered_scene=before, task=task,
                                       nominal_slot_xy_mm=slot_xy_mm)
        if (not math.isclose(float(runtime.step_seconds), PERIOD_SECONDS, abs_tol=1e-12, rel_tol=0)
                or runtime.actuation_semantics != ACTUATION_SEMANTICS
                or runtime.direct_position_label != POSITION_LABEL
                or runtime.terminal_semantics not in TERMINAL_SEMANTICS):
            raise ValueError('trajectory checkpoint does not match the integrated 20 ms contract')
        if cancel.is_set():
            raise InterruptedError('request cancelled before transfer')
        initial = _robot_boundary(sim)
        gripper = _gripper_command(applied_targets)
        sim.begin_transfer(before, task['source_slot'], task['destination_slot'])

        def result(termination, **extra):
            return {'termination': termination, 'decisions': decisions, 'physical_success': None,
                    'teacher_assistance_count': 0, 'actual_hardware': False,
                    'history_scope': 'fresh_per_transfer_real_public_observations_only',
                    'initial_robot_boundary': initial, 'end_robot_boundary': _robot_boundary(sim), **extra}

        for index in range(max_decisions):
            if cancel.is_set():
                raise InterruptedError('request cancelled before trajectory decision')
            oracle._require_servo_targets(sim, applied_targets)
            obs = self.observation.read(gripper)
            raw = None
            try:
                raw = runtime.step(obs)
                if (not isinstance(raw, dict) or raw.get('actuation_semantics') != runtime.actuation_semantics
                        or raw.get('direct_position_label') != runtime.direct_position_label
                        or raw.get('terminal_semantics') != runtime.terminal_semantics):
                    raise ValueError('student output semantics differ from the checkpoint')
                command = validate_proposal(raw)
            except Exception as exc:
                return result('invalid_or_failed_student_output', error_type=type(exc).__name__,
                              error=str(exc), rejected_observation=obs, rejected_proposal=_json_safe(raw))
            row = {'index': index, 'observation': obs, 'proposal': command,
                   'previous_applied_targets': copy.deepcopy(applied_targets)}
            decisions.append(row)
            if command['done'] and runtime.terminal_semantics == 'done_no_command':
                row['result'] = 'done_no_command_velocity_zero'
                return result('model_done')
            if cancel.is_set():
                raise InterruptedError('request cancelled before applying trajectory command')
            try:
                def check_cancel(_index, _target):
                    if cancel.is_set():
                        raise InterruptedError('request cancelled during trajectory command')
                self.servo.apply(command, tick_observer=check_cancel)
            except InterruptedError:
                raise
            except Exception as exc:
                row['result'] = 'command_error'
                return result('command_error', error_type=type(exc).__name__, error=str(exc))
            applied_targets.update(oracle._servo_targets(sim))
            gripper = command['gripper']
            row['applied_targets'] = copy.deepcopy(applied_targets)
            row['after_time_seconds'] = float(sim.data.time)
            row['result'] = ('applied_last_10_physics_steps_then_done_velocity_zero' if command['done']
                             else 'applied_for_10_physics_steps')
            if command['done']:
                # Advance the observation's causal reference through the final
                # 20 ms without feeding a second sample into the old policy.
                self.observation.read(gripper)
                return result('model_done')
        return result('tick_limit')


class TrajectoryProgramSession(oracle.OracleFreeSceneModelSession):
    """Use existing program/GRU semantics with the new trajectory runner."""

    def _hold_observation_hook(self, index, steps):
        # As in the direct demonstrator, the next transfer's moving flag is
        # based on the final 20 ms of the passive hold, not its whole second.
        dt = float(self.sim.model.opt.timestep)
        tail_steps = round(PERIOD_SECONDS / dt)
        if tail_steps < 1 or not math.isclose(tail_steps * dt, PERIOD_SECONDS, abs_tol=1e-9):
            raise ValueError('physics timestep cannot represent the observation period')
        if index == steps - tail_steps:
            runner = self._unscored_transfer_runner
            runner.observation.read(_gripper_command(oracle._servo_targets(self.sim)))

    def _move(self, action, **kwargs):
        super()._move(action, **kwargs)
        event = next(event for event in reversed(self.events) if event.get('event') == 'transfer')
        event['after_hold_robot_boundary'] = _robot_boundary(self.sim)
        event['memory_state_version_after'] = self.memory.state_version
        event['boundary_note'] = ('same simulated physical state and applied controls; '
                                  'scene is expected only; policy observation history restarts')


def make_coordinator(sim, model, *, max_ticks, memory=None, development_groups=None,
                     runtime_factory=scene_trajectory_policy.TrajectoryPolicyRuntime):
    """Create one offline program session; do not reset or advance ``sim``.

    Default movement groups are the existing allowed train groups. An explicit
    development_groups value is checked by the existing session against its
    train/validation split; it cannot silently admit reserved test groups.
    """
    if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
        raise TypeError('a public-only virtual simulator is required')
    memory = load_pinned_gru128() if memory is None else memory
    if not isinstance(memory, scene_recall.SceneRecall):
        raise TypeError('a scene recall runtime is required')
    runner = TrajectoryTransferRunner(sim, runtime_factory=runtime_factory)
    session = TrajectoryProgramSession(sim, memory, model, device='cpu', max_decisions=max_ticks,
                                       development_groups=development_groups,
                                       unscored_transfer_runner=runner)
    return oracle.OracleFreeSceneProgramCoordinator(session)
