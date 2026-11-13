"""Simulation-only collection of explicitly submitted timed velocity commands.

The four submitted speeds are integrated at each physics step into the existing
position servos. There is no trajectory planner, interpolator, teacher policy,
object-position or contact query in this execution path. One decision can yield
many 50 Hz samples; simulator observations are never attributed to the actor.
"""
from __future__ import annotations

import math
import uuid

import mujoco

from raccoon_control.kinematics import check_joints
from . import scene_features
from .scene_motion_sim import SceneMotionSim


SAMPLE_SECONDS = 0.02
INITIAL_REST_TOLERANCE_DEG_S = 1e-9


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


class DirectVelocityExecutionError(RuntimeError):
    """An accepted batch stopped during execution; ``result`` is partial."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class DirectVelocityDemo:
    """Execute explicit velocity/gripper decisions in one virtual scene."""

    def __init__(self, sim, registered_scene, task, slot_xy_mm, *, tick_seconds=0.1,
                 max_joint_speed_deg_s=30.0, max_chunk_steps=10,
                 max_chunk_seconds=60.0, max_command_seconds=10.0):
        if not isinstance(sim, SceneMotionSim) or sim.metadata.get('simulated') is not True:
            raise ValueError('a simulated SceneMotionSim is required; hardware is not supported')
        self.sim = sim
        self.scene, self.task = scene_features.transfer_task(registered_scene, task)
        if not isinstance(slot_xy_mm, (list, tuple)) or len(slot_xy_mm) != 4:
            raise ValueError('four registered slot coordinates are required')
        self.slots = []
        for xy in slot_xy_mm:
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError('each slot requires two coordinates')
            self.slots.append([_finite(v, 'slot coordinate') for v in xy])
        self.physics_seconds = _finite(sim.model.opt.timestep, 'physics timestep')
        if self.physics_seconds <= 0:
            raise ValueError('physics timestep must be positive')
        self.physics_steps = round(SAMPLE_SECONDS / self.physics_seconds)
        if self.physics_steps < 1 or not math.isclose(self.physics_steps * self.physics_seconds,
                SAMPLE_SECONDS, abs_tol=1e-9, rel_tol=0):
            raise ValueError('sample period must be a multiple of the simulator timestep')
        self.max_speed = _finite(max_joint_speed_deg_s, 'max_joint_speed_deg_s')
        self.max_chunk_seconds = _finite(max_chunk_seconds, 'max_chunk_seconds')
        self.max_command_seconds = _finite(max_command_seconds, 'max_command_seconds')
        if (self.max_speed <= 0 or type(max_chunk_steps) is not int or not 1 <= max_chunk_steps <= 1000
                or not SAMPLE_SECONDS <= self.max_command_seconds <= 60
                or not SAMPLE_SECONDS <= self.max_chunk_seconds <= 600):
            raise ValueError('invalid simulation-only speed, duration or chunk limit')
        self.max_chunk_steps = max_chunk_steps
        self.tick_seconds = _finite(tick_seconds, 'tick_seconds')
        self._duration_samples(self.tick_seconds)
        self.session_id = uuid.uuid4().hex
        self.revision = 0
        self.decision_count = 0
        self.sample_count = 0
        self.last_sample_id = None
        self.previous_action = None
        # There is no earlier sampled angle at session start. Require and
        # record a verifiably stationary simulated start instead of inventing
        # the first moving flag from an unavailable observation history.
        self.initial_joint_velocity_deg_s = [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]]))
                                             for j in sim._joint_ids]
        if (not all(math.isfinite(v) for v in self.initial_joint_velocity_deg_s)
                or any(abs(v) > INITIAL_REST_TOLERANCE_DEG_S for v in self.initial_joint_velocity_deg_s)):
            raise ValueError('initial simulated joint velocities must be zero for the first public moving flag')
        self.moving = False
        self.finished = False
        self.stopped = False
        self.gripper = 'open' if sum(float(sim.data.ctrl[i]) for i in sim._finger_actuators) >= .025 else 'closed'
        self.sim.begin_transfer(self.scene, self.task['source_slot'], self.task['destination_slot'])

    def _duration_samples(self, value):
        duration = _finite(value, 'duration_seconds')
        count = round(duration / SAMPLE_SECONDS)
        if (duration <= 0 or duration > self.max_command_seconds or count < 1
                or not math.isclose(count * SAMPLE_SECONDS, duration, abs_tol=1e-9, rel_tol=0)):
            raise ValueError('duration must be a positive 0.02 second multiple within the command limit')
        return count

    def observation(self):
        """Public scene and simulated joint/finger observations, no object truth."""
        sim = self.sim
        joints = [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids]
        velocity = [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]])) for j in sim._joint_ids]
        opening = float(sum(sim.data.qpos[sim.model.jnt_qposadr[j]] for j in sim._finger_ids))
        return {'observation_id': f'{self.session_id}:{self.revision}',
                'schema_version': 'PolicyObservation.phase1',
                'simulated': True, 'time_seconds': float(sim.data.time),
                'joints_deg': joints, 'joint_velocity_deg_s': velocity,
                'joint_velocity_source': 'simulator_observation_not_submitted_command',
                'sensor_valid': True, 'moving': self.moving,
                'applied_target_deg': [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators],
                'gripper_command': self.gripper, 'gripper_opening_m': opening,
                'gripper_reported_state': 'open' if opening >= .03 else 'closed',
                'gripper_reported_valid': True, 'gripper_feedback_fresh': True,
                'registered_scene': [list(stack) for stack in self.scene],
                'task': dict(self.task), 'registered_slot_xy_mm': [list(x) for x in self.slots],
                'physical_confirmation': 'not_inspected'}

    def _require_active(self, observation_id):
        if self.finished or self.stopped or self.sim.layout_status != 'in_progress':
            raise RuntimeError('the simulated transfer is not active')
        if observation_id != f'{self.session_id}:{self.revision}':
            raise ValueError('the observation is stale')

    def submit(self, observation_id, commands, *, on_progress=None):
        """Preflight all decisions atomically, then execute and record each sample.

        An optional callback receives ``accepted`` and ``sample`` events. It can
        persist progress before a process interruption; completed samples in the
        result have the same IDs and must not be counted twice.
        """
        self._require_active(observation_id)
        if not isinstance(commands, list) or not 1 <= len(commands) <= self.max_chunk_steps:
            raise ValueError('invalid command batch size')
        current = [math.degrees(float(self.sim.data.ctrl[i])) for i in self.sim._arm_actuators]
        check_joints(current)
        prepared, total_samples = [], 0
        for index, row in enumerate(commands):
            if (not isinstance(row, dict) or set(row) not in
                    ({'joint_velocity_deg_s', 'gripper'}, {'joint_velocity_deg_s', 'gripper', 'duration_seconds'})):
                raise ValueError('each decision requires four velocities, a gripper and optional duration_seconds')
            velocity = row['joint_velocity_deg_s']
            if not isinstance(velocity, (list, tuple)) or len(velocity) != 4:
                raise ValueError('four joint velocities are required')
            velocity = [_finite(value, 'joint velocity') for value in velocity]
            if any(abs(value) > self.max_speed for value in velocity):
                raise ValueError('a velocity exceeds the simulation-only limit')
            if row['gripper'] not in ('hold', 'open', 'closed'):
                raise ValueError('gripper must be hold, open or closed')
            count = self._duration_samples(row.get('duration_seconds', self.tick_seconds))
            total_samples += count
            if total_samples * SAMPLE_SECONDS > self.max_chunk_seconds + 1e-9:
                raise ValueError('command batch exceeds the simulation-only duration limit')
            # Each axis is monotone under a constant speed, so its two ends
            # bound the entire command. No simulator state changes in preflight.
            current = list(check_joints([q + count * SAMPLE_SECONDS * v for q, v in zip(current, velocity)]))
            submitted = {'joint_velocity_deg_s': velocity, 'gripper': row['gripper']}
            if 'duration_seconds' in row:
                submitted['duration_seconds'] = _finite(row['duration_seconds'], 'duration_seconds')
            prepared.append({'decision_id': f'{self.session_id}:decision:{self.decision_count + index + 1}',
                             'submitted': submitted, 'duration_seconds': count * SAMPLE_SECONDS,
                             'duration_source': ('submitted_duration_seconds' if 'duration_seconds' in row
                                                 else 'configured_legacy_tick_seconds'),
                             'sample_count': count})
        if on_progress is not None:
            on_progress('accepted', {'observation_id': observation_id, 'decisions': prepared,
                                     'expected_samples': total_samples})

        result = {'schema': 'raccoon-direct-velocity-chunk.v2', 'simulated': True,
                  'source': 'submitted_commands_actor_unverified', 'status': 'executing',
                  'steps': [], 'samples': [], 'physical_success': None}
        current_step, physics_done = None, 0
        try:
            for command in prepared:
                self.decision_count += 1
                submitted, count = command['submitted'], command['sample_count']
                velocity, grip = submitted['joint_velocity_deg_s'], submitted['gripper']
                start_target = [math.degrees(float(self.sim.data.ctrl[i])) for i in self.sim._arm_actuators]
                current_step = {**command, 'status': 'executing', 'samples_completed': 0,
                                'before': self.observation()}
                result['steps'].append(current_step)
                for offset in range(count):
                    before = self.observation()
                    features = scene_features.encode_scene_features(before, self.scene, self.task, self.slots)
                    previous = self.previous_action
                    if grip != 'hold':
                        self.gripper = grip
                    for actuator in self.sim._finger_actuators:
                        self.sim.data.ctrl[actuator] = .02 if self.gripper == 'open' else 0.
                    physics_done = 0
                    for physics in range(self.physics_steps):
                        elapsed = (offset * self.physics_steps + physics + 1) * self.physics_seconds
                        target = [q + elapsed * v for q, v in zip(start_target, velocity)]
                        for actuator, angle in zip(self.sim._arm_actuators, target):
                            self.sim.data.ctrl[actuator] = math.radians(angle)
                        mujoco.mj_step(self.sim.model, self.sim.data)
                        physics_done += 1
                    mujoco.mj_forward(self.sim.model, self.sim.data)
                    self.revision += 1
                    after = self.observation()
                    self.moving = max(abs(a-b) / SAMPLE_SECONDS for a, b in
                                      zip(after['joints_deg'], before['joints_deg'])) > .1
                    after['moving'] = self.moving
                    sample_id = f'{self.session_id}:sample:{self.sample_count + 1}'
                    sample = {'sample_id': sample_id, 't_s': round(self.sample_count * SAMPLE_SECONDS, 8),
                              'dt_s': SAMPLE_SECONDS, 'features': features,
                              'previous_action': previous,
                              'observed_joint_deg': list(before['joints_deg']),
                              'observed_joint_velocity_deg_s': list(before['joint_velocity_deg_s']),
                              'observed_gripper_command': before['gripper_command'],
                              'decision_id': command['decision_id'], 'decision_sample_index': offset,
                              'decision_sample_count': count,
                              'submitted': dict(submitted),
                              'submitted_duration_seconds': submitted.get('duration_seconds'),
                              'effective_duration_seconds': command['duration_seconds'],
                              'duration_source': command['duration_source'],
                              'applied_position_servo_target_deg': list(after['applied_target_deg']),
                              'target_joint_deg': list(after['joints_deg']),
                              'target_velocity_deg_s': list(velocity), 'target_gripper': self.gripper,
                              'terminal': False, 'next_observation': after}
                    self.sample_count += 1
                    self.last_sample_id = sample_id
                    self.previous_action = {'decision_id': command['decision_id'],
                                            'joint_velocity_deg_s': list(velocity),
                                            'gripper_command': self.gripper}
                    current_step['samples_completed'] += 1
                    result['samples'].append(sample)
                    if on_progress is not None:
                        on_progress('sample', sample)
                current_step.update(status='completed', after=self.observation(),
                                    applied_position_servo_target_deg=self.observation()['applied_target_deg'])
            result.update(status='completed', observation=self.observation(),
                          decisions_submitted=len(prepared), decisions_completed=len(prepared),
                          samples_completed=len(result['samples']))
            return result
        except BaseException as exc:
            self.sim.mark_layout_uncertain('direct velocity batch interrupted after partial execution')
            self.stopped = True
            self.revision += 1
            if current_step is not None:
                current_step.update(status='interrupted', physics_steps_in_incomplete_sample=physics_done,
                                    after=self.observation())
            result.update(status='interrupted', decisions_submitted=len(prepared),
                          decisions_completed=sum(step['status'] == 'completed' for step in result['steps']),
                          samples_completed=len(result['samples']), observation=self.observation(),
                          error_type=type(exc).__name__, error=str(exc), physical_success=None)
            raise DirectVelocityExecutionError('direct velocity batch interrupted', result) from exc

    def finish(self, observation_id):
        """Record explicit DONE. Assessment is a separate post-finish operation."""
        self._require_active(observation_id)
        self.finished = True
        return {'source': 'explicit_finish', 'observation_id': observation_id,
                'time_seconds': float(self.sim.data.time), 'last_sample_id': self.last_sample_id,
                'last_sample_terminal': self.last_sample_id is not None,
                'decisions': self.decision_count, 'samples': self.sample_count}

    def stop(self, observation_id, reason='explicit_stop'):
        """Stop without DONE, assessment, or further virtual physics steps."""
        self._require_active(observation_id)
        self.stopped = True
        self.sim.mark_layout_uncertain(reason)
        return {'reason': reason, 'completed': False, 'decisions': self.decision_count,
                'samples': self.sample_count, 'observation': self.observation(), 'physical_success': None}
