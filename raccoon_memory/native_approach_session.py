"""The source-approach student in the existing, simulation-only v2 session.

Only declared slots, symbolic scene and robot/control observations reach the
student. DONE commits an *expected* scene; the saved snapshots can be scored
only after the request ends. This module cannot connect to physical hardware.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
from pathlib import Path
import threading
import weakref

import torch
from torch import nn

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import motion_policy, scene_features, scene_recall, task_catalog
from . import scene_oracle_free_session as runtime
from . import terra_native_approach_policy as approach
from .scene_custom_slots_sim import CustomSlotsSceneMotionSim
from .scene_ramped_motion_sim import FINGER_OPEN_HALF_SECOND, FINGER_OPEN_RAMP_SECONDS, RampedSceneMotionSim

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / 'artifacts/architecture_research/astra_native_geometry_source_approach_v1/mixed/candidate.pt'
STUDENT_SHA256 = '06f198d501c93c2e247d8e55c839a88015e1fdaa9f79941540b2d1278f3a5c89'
MEMORY = ROOT / 'artifacts/architecture_research/astra_terra_weight_repair_v1/gru128_decoder_margin_repair_v1/model.pt'
MEMORY_SHA256 = 'e8fdf96cb266c094afb90a02bc8953e18f68ca13f8aa03bea2709dea0ac06682'
START = 4
_SIMULATOR_OWNERS = weakref.WeakKeyDictionary()
_SIMULATOR_OWNERS_LOCK = threading.Lock()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _close(a, b, tolerance=1e-7):
    return len(a) == len(b) and all(math.isclose(float(x), float(y), abs_tol=tolerance, rel_tol=0)
                                      for x, y in zip(a, b))


class PublicCustomSlotsSceneMotionSim(CustomSlotsSceneMotionSim):
    """Return robot data only from commands; assessment uses a separate sim."""
    def __init__(self, *args, **kwargs):
        self._public_only = False
        super().__init__(*args, **kwargs)
        self._public_only = True

    def _objects(self):
        if self._public_only:
            raise RuntimeError('object state is unavailable during the live session')
        return super()._objects()

    def _contacts(self):
        if self._public_only:
            raise RuntimeError('contact state is unavailable during the live session')
        return super()._contacts()

    def _assessment_sample(self, *args, **kwargs):
        raise RuntimeError('live assessment is unavailable')

    def evaluate_transfer(self, *args, **kwargs):
        raise RuntimeError('live assessment is unavailable')

    def reset(self, *args, **kwargs):
        raise RuntimeError('live reset is unavailable')

    def load_state(self, *args, **kwargs):
        raise RuntimeError('live state restoration is unavailable')

    def observation(self):
        observation = runtime._robot_observation(self, None)
        return {'source': 'mujoco_simulation', 'simulated': True,
                'time_seconds': observation['observed_monotonic'],
                'joints_deg': observation['joints_deg'], 'tcp_mm': observation['tcp_mm'],
                'gripper_opening_m': observation['gripper_opening_m']}


class AppliedHistoryPolicy(nn.Module):
    """Align each public observation with the last *applied* target and action."""
    history_length, feature_count = 8, 82

    def __init__(self, student):
        super().__init__()
        self.student = student
        self.on_scene_registered(None, None)

    def on_scene_registered(self, _scene, _slots):
        self.current_target = self.previous_action = self.pending = self.last_context = None
        self.committed, self.count = [], 0

    def on_transfer_started(self, _before, _task, applied_targets):
        target = applied_targets.get('joints_deg')
        _need(isinstance(target, (tuple, list)) and runtime._in_range(target), 'invalid applied start target')
        self.current_target, self.previous_action = list(target), START
        self.committed, self.count, self.pending, self.last_context = [], 0, None, None

    def forward_with_control_state(self, windows, masks, previous):
        _need(self.current_target is not None and isinstance(windows, torch.Tensor)
              and windows.shape == (1, 8, 82) and windows.dtype == torch.float32
              and isinstance(masks, torch.Tensor) and masks.shape == (1, 8) and masks.dtype == torch.bool
              and isinstance(previous, torch.Tensor) and previous.shape == (1, 4)
              and bool(torch.isfinite(windows).all() and torch.isfinite(previous).all()), 'invalid public inputs')
        n = min(self.count + 1, 8)
        _need(masks[0].tolist() == [False] * (8-n) + [True] * n
              and _close(previous[0].tolist(), self.current_target, 1e-4), 'applied control history differs')
        entries = (self.committed + [(list(self.current_target), self.previous_action)])[-8:]
        padding = 8 - len(entries)
        prior = [[0.] * 4 for _ in range(padding)] + [[x / 180. for x in q] for q, _ in entries]
        actions = [-1] * padding + [a for _, a in entries]
        p = torch.tensor([prior], dtype=windows.dtype, device=windows.device)
        a = torch.tensor([actions], dtype=torch.long, device=windows.device)
        logits, joints = self.student(windows, masks, p, a)
        _need(logits.shape == (1, 4) and joints.shape == (1, 4)
              and bool(torch.isfinite(logits).all() and torch.isfinite(joints).all()), 'invalid student output')
        pending = {'window': windows.detach().cpu().tolist(), 'mask': masks.detach().cpu().tolist(),
                   'target': list(self.current_target), 'prior_action': self.previous_action,
                   'predicted_action': int(logits.argmax(-1).item())}
        if self.pending is not None:
            _need(pending == self.pending, 'new decision before command confirmation')
        else:
            self.pending = pending
        self.last_context = {'previous_targets_normalized': p[0].detach().cpu().tolist(),
                             'previous_actions': actions, 'confirmed_previous_commands': self.count}
        return logits, joints

    def forward(self, _windows, _masks):
        raise ValueError('confirmed previous applied control is required')

    def on_control_applied(self, action, transmitted, applied, _opening):
        _need(self.pending is not None and action in motion_policy.ACTIONS[:-1]
              and motion_policy.ACTIONS[self.pending['predicted_action']] == action
              and runtime._in_range(transmitted) and runtime._in_range(applied) and _close(transmitted, applied),
              'command was not confirmed applied')
        self.committed = (self.committed + [(list(self.current_target), self.previous_action)])[-7:]
        self.current_target, self.previous_action = list(applied), motion_policy.ACTIONS.index(action)
        self.count += 1
        self.pending = None


def native_policy_decision(model, observation, before, task, slots, history, previous_target):
    """Produce one guarded student decision from public observations only.

    Both the virtual executor and the controller adapter use this function.
    Their observations and last applied targets still come from their own
    backends, and an output is only a proposal until that backend confirms it.
    """
    features = scene_features.encode_scene_features(observation, before, task, slots)
    history.append(features)
    window, mask = scene_features.history_window(history, 8)
    with torch.inference_mode():
        logits, joints = model.forward_with_control_state(
            torch.tensor([window], dtype=torch.float32),
            torch.tensor([mask], dtype=torch.bool),
            torch.tensor([previous_target], dtype=torch.float32))
    action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
    return features, mask, logits, joints, action


def native_command_target(action, joints, previous_target):
    """Translate a guarded proposal without claiming that it was applied."""
    return [float(value) * 180. for value in joints[0].tolist()] if action == 'move' else list(previous_target)


def _apply(sim, target, opening, action, *, cancel=None):
    _need(isinstance(sim, PublicCustomSlotsSceneMotionSim)
          and type(sim).command is RampedSceneMotionSim.command
          and (sim.ramp_seconds, sim.threshold_degrees, sim.ramp_shape) == (.5, 0., 'minimum_jerk')
          and float(sim.model.opt.timestep) == .002, 'different virtual control profile')
    demo_timing = getattr(sim, '_control_timing_profile', 'app_2s') == 'demo_1s'
    duration = 1. if demo_timing or action != 'move' else 2.
    ramp = .5 if demo_timing or action != 'move' else (2. if opening == 0. else 1.)
    start = runtime._servo_targets(sim)
    if action == 'move':
        _need(start['gripper_opening_m'] == opening, 'MOVE must preserve the confirmed gripper target')
    cartesian = None
    path = None
    if getattr(sim, '_straight_descent_enabled', False) and action == 'move' and opening == 0.:
        from . import straight_descent
        task = sim.pending_transfer
        _need(isinstance(task, dict), 'checked descent requires the current expected transfer')
        destination = task['destination_slot'] - 1
        try:
            path, cartesian = straight_descent.prepare(
                start['joints_deg'], target, sim.slot_xy_mm[destination], bool(task['before_layout'][destination]), cancel)
        except Exception as exc:
            sim.mark_layout_uncertain(f'virtual_descent_preparation_error: {type(exc).__name__}: {exc}')
            raise
    if cancel is not None and cancel.is_set():
        sim.mark_layout_uncertain('virtual_command_cancelled_before_motion')
        raise InterruptedError('cancelled before virtual command')
    beginning = float(sim.data.time)
    _need(getattr(sim, '_checked_straight_descent', None) is None, 'another virtual descent is active')
    try:
        sim.ramp_seconds = ramp
        sim._demo_one_second_move = demo_timing and action == 'move'
        sim._checked_straight_descent = path
        sim._checked_descent_cancel = cancel
        returned = sim.command(target, opening, seconds=duration)
    finally:
        sim.ramp_seconds = .5
        sim._demo_one_second_move = False
        sim._checked_straight_descent = None
        sim._checked_descent_cancel = None
    elapsed = float(returned['time_seconds']) - beginning
    _need(returned.get('source') == 'mujoco_simulation' and returned.get('simulated') is True
          and math.isclose(elapsed, duration, abs_tol=1e-6, rel_tol=0), 'virtual command did not return normally')
    runtime._require_servo_targets(sim, {'joints_deg': target, 'gripper_opening_m': opening})
    actual = runtime._servo_targets(sim)['joints_deg']
    timing = {'requested_seconds': duration, 'sim_elapsed_seconds': elapsed, 'ramp_seconds': ramp,
              'closed_full_2s': action == 'move' and opening == 0. and duration == 2. and ramp == 2.,
              'initial_control_target_deg': start['joints_deg']}
    if action == 'move':
        closed_move = start['gripper_opening_m'] == 0. and opening == 0.
        timing.update(closed_move_profile=sim.closed_move_profile, closed_move_curve_applied=closed_move,
                      arm_interpolation_profile=sim.closed_move_profile if closed_move else sim.ramp_shape,
                      move_start_gripper_target_m=start['gripper_opening_m'])
        if cartesian is not None:
            timing['cartesian'] = cartesian
            if cartesian['changed']:
                timing['arm_interpolation_profile'] = 'nominal_straight_descent_smoothstep'
    if action == 'open':
        configured_ramp = FINGER_OPEN_RAMP_SECONDS if sim.finger_open_profile == FINGER_OPEN_HALF_SECOND else 0.
        finger_ramp = configured_ramp if start['gripper_opening_m'] < opening else 0.
        timing.update(finger_open_profile=sim.finger_open_profile, finger_open_start_m=start['gripper_opening_m'],
                      finger_open_target_m=opening, finger_open_configured_ramp_seconds=configured_ramp,
                      finger_open_ramp_seconds=finger_ramp,
                      finger_open_endpoint_hold_seconds=duration - finger_ramp)
    return timing, actual


def run_native_transfer(model, sim, before, task, *, max_decisions, device, cancel,
                        applied_targets, decisions=None, slot_xy_mm=None):
    if decisions is None:
        decisions = []
    before, task = scene_features.transfer_task(before, task)
    slots = runtime._registered_slots_for_sim(sim, slot_xy_mm)
    _need(device == 'cpu' and max_decisions == 60 and task_catalog.scene(sim._committed_layout) == before
          and sim.layout_status == 'registered', 'different virtual session preconditions')
    runtime._require_servo_targets(sim, applied_targets)
    start = runtime._servo_targets(sim)
    opening = start['gripper_opening_m']
    _need(opening >= .025 and runtime._robot_observation(sim, 'open')['gripper_opening_m'] >= .025,
          'transfer did not start open')
    if cancel.is_set():
        raise InterruptedError('cancelled before transfer')
    model.on_transfer_started(before, task, copy.deepcopy(start))
    sim.begin_transfer(before, task['source_slot'], task['destination_slot'])
    support_guard = None
    if getattr(sim, '_support_height_guard_enabled', False):
        from .support_height_guard import SupportHeightGuard
        support_guard = SupportHeightGuard(task, slots[task['destination_slot'] - 1])
    # Forecasts copy this robot-observation-only state when a command starts.
    sim._active_support_height_guard = support_guard
    history, gripper, last = [], 'open', list(start['joints_deg'])
    for index in range(max_decisions):
        if cancel.is_set():
            raise InterruptedError('cancelled before decision')
        runtime._require_servo_targets(sim, applied_targets)
        observation = runtime._robot_observation(sim, gripper)
        features, mask, logits, joints, action = native_policy_decision(
            model, observation, before, task, slots, history, last)
        guard_event = model.events[-1]
        row = {'index': index, 'policy_features': features, 'valid_history': mask,
               'action_logits': logits.tolist(), 'absolute_joint_output': joints.tolist(),
               'previous_applied_target_deg': list(last), 'action': action,
               'transfer_start_applied_target_deg': list(start['joints_deg']),
               'control_context': copy.deepcopy(model.policy.last_context), 'guard': copy.deepcopy(guard_event)}
        decisions.append(row)
        if cancel.is_set():
            row['result'] = 'no_command'
            raise InterruptedError('cancelled before command')
        runtime._require_servo_targets(sim, applied_targets)
        if action == 'done':
            sim._active_support_height_guard = None
            row['result'] = 'done_no_command'
            return {'termination': 'model_done', 'decisions': decisions, 'physical_success': None,
                    'teacher_assistance_count': 0, 'actual_hardware': False}
        target = native_command_target(action, joints, last)
        next_opening = (.04 if action == 'open' else 0.) if action in ('open', 'close') else opening
        if not runtime._in_range(target):
            sim._active_support_height_guard = None
            row['result'] = 'no_command'
            return {'termination': 'out_of_range_target', 'decisions': decisions, 'physical_success': None,
                    'teacher_assistance_count': 0, 'actual_hardware': False}
        if support_guard is not None:
            try:
                target, adjustment = support_guard.apply(
                    action=action, proposed_target_deg=target, previous_target_deg=last,
                    gripper_opening_m=opening, observed_tcp_mm=observation['tcp_mm'])
            except Exception as exc:
                row['result'] = 'no_command'
                sim.mark_layout_uncertain(f'virtual_support_height_error: {type(exc).__name__}: {exc}')
                raise
            if adjustment is not None:
                row['support_height_guard'] = adjustment
        seconds = (1. if getattr(sim, '_control_timing_profile', 'app_2s') == 'demo_1s'
                   or action != 'move' else 2.)
        row['attempted_command'] = {'joints_deg': target, 'gripper_opening_m': next_opening,
                                    'seconds': seconds, 'source': 'shared_student_policy'}
        row['result'] = 'command_unconfirmed'
        timing, actual = _apply(sim, target, next_opening, action, cancel=cancel)
        _need(runtime._in_range(actual) and _close(target, actual), 'applied control target differs')
        row['duration_profile'], row['result'] = timing, 'command_applied'
        row['command'] = row.pop('attempted_command')
        guard_event['actual_applied_joints_deg'] = list(actual)
        row['guard']['actual_applied_joints_deg'] = list(actual)
        last, opening = list(actual), next_opening
        if action in ('open', 'close'):
            gripper = 'open' if action == 'open' else 'closed'
        applied_targets.update(joints_deg=list(actual), gripper_opening_m=opening)
        model.on_control_applied(action, target, actual, opening)
    sim._active_support_height_guard = None
    return {'termination': 'decision_limit', 'decisions': decisions, 'physical_success': None,
            'teacher_assistance_count': 0, 'actual_hardware': False}


def load_session(sim, *, cancel=None, student_path=STUDENT, memory_path=MEMORY):
    """Bind the pinned latest student and GRU to the established v2 API."""
    if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
        raise TypeError('a public-only virtual simulator is required')
    with _SIMULATOR_OWNERS_LOCK:
        owner = _SIMULATOR_OWNERS.get(sim)
        if owner is not None and owner() is not None:
            raise RuntimeError('this simulator is already bound to a live session')
    student_bytes, memory_bytes = Path(student_path).read_bytes(), Path(memory_path).read_bytes()
    _need(hashlib.sha256(student_bytes).hexdigest() == STUDENT_SHA256, 'student checkpoint differs')
    _need(hashlib.sha256(memory_bytes).hexdigest() == MEMORY_SHA256, 'GRU checkpoint differs')
    student = approach.load_student_checkpoint(io.BytesIO(student_bytes), device='cpu').requires_grad_(False)
    memory_model = scene_recall.load_scene_memory(io.BytesIO(memory_bytes), device='cpu').requires_grad_(False)
    _need(memory_model.hidden_size == 128, 'expected the GRU128 memory')
    guard = GuardedPolicy(AppliedHistoryPolicy(student)).eval()
    split, _, _ = runtime.scene_demo_data.read_split_plan(runtime.SPLIT_PLAN)
    allowed = set(split['train_groups']) | set(split['validation_groups'])
    session = runtime.OracleFreeSceneModelSession(sim, scene_recall.SceneRecall(memory_model), guard, max_decisions=60,
        development_groups=allowed, slot_xy_mm=sim.slot_xy_mm, cancel=cancel,
        unscored_transfer_runner=run_native_transfer)
    # The historical split gates development execution; it does not describe
    # this candidate's actual training roster.
    session.checkpoint_train_groups = None
    coordinator = runtime.OracleFreeSceneProgramCoordinator(session)
    # The check and claim are atomic even when callers load the same simulator
    # concurrently. A live session, including an externally retained one,
    # prevents a second independent lock and memory from owning its simulator.
    with _SIMULATOR_OWNERS_LOCK:
        owner = _SIMULATOR_OWNERS.get(sim)
        if owner is not None and owner() is not None:
            raise RuntimeError('this simulator is already bound to a live session')
        _SIMULATOR_OWNERS[sim] = weakref.ref(session)
    return coordinator
