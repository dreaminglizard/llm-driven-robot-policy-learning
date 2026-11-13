"""Detached current-transfer forecast using the student's own public inputs.

The live policy only records its bounded input tensors.  A private copy runs
the existing virtual forecast engine; it cannot call the controller or inspect
real object locations.  The result is conditional, not a safety check.
"""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import uuid

import torch
from torch import nn

from raccoon_control.kinematics import Geometry, quantize_joints
from . import motion_policy
from .demo_forecast import _FORECAST_GATE, _forecast
from .support_height_guard import SupportHeightGuard


class _CapturedStudent(nn.Module):
    def __init__(self, student, forecast):
        super().__init__()
        self.student, self.forecast = student, forecast

    def forward(self, windows, masks, prior, actions):
        try:
            context = self.forecast.take_decision_context()
        except Exception:
            context = None
            try:
                self.forecast.invalidate('unavailable')
            except Exception:
                pass
        result = self.student(windows, masks, prior, actions)
        try:
            self.forecast.capture(context, windows, masks, prior, actions, result[0], result[1])
        except Exception:
            # A display-only feature must never change the live decision.
            try:
                self.forecast.capture_failed(context)
            except Exception:
                pass
        return result


class _Cancelled:
    def __init__(self, owner, generation):
        self.owner, self.generation = owner, generation

    def is_set(self):
        with self.owner._lock:
            changed = self.owner._closed or self.owner._generation != self.generation
        return changed or self.owner.session.cancel.is_set()


class ControllerForecast:
    """Keep only the latest command's detached, bounded forecast."""
    def __init__(self, controller, coordinator, *, mock):
        self.controller, self.session = controller, coordinator.session
        self.session_token = uuid.uuid4().hex
        self.mock = mock
        self._student = deepcopy(self.session.student).requires_grad_(False).eval()
        self._lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._thread = None
        self._job = None
        self._decision_context = None
        self._capture = None
        self._staged_segment = None
        self._segment_plan = None
        self._future_result = None
        self._transfer = None
        self._close_z = None
        self._result = {'segments': [], 'complete': False, 'reason': 'no_active_command',
                        'scope': 'current_transfer', 'basis_command_id': None}
        self.session.student = _CapturedStudent(self.session.student, self)
        controller._forecast = self

    def begin_decision(self, observation, before, task, slots, transfer, decision_index):
        """Copy the actual inputs for one live decision, never an event's task."""
        context = {'observation': {key: deepcopy(observation.get(key))
                                  for key in ('joints_deg', 'tcp_mm', 'gripper_command')},
                   'before': deepcopy(before), 'task': deepcopy(task), 'slots': deepcopy(slots),
                   'thread': threading.get_ident(), 'decision_index': decision_index}
        with self._lock:
            self._clear_locked('unavailable')
            if self._closed or self.session.cancel.is_set():
                return
            if self._transfer is not transfer:
                self._transfer, self._close_z = transfer, None
            context['generation'] = self._generation
            self._decision_context = context

    def take_decision_context(self):
        with self._lock:
            context = self._decision_context
            if context is None:
                # For example, replayed historical inputs are not a live
                # decision and must not reuse the preceding capture.
                self._clear_locked('unavailable')
                return None
            if context['thread'] != threading.get_ident():
                return None
            self._decision_context = None
            return context

    def capture_failed(self, context):
        with self._lock:
            if context is not None and context['generation'] == self._generation:
                self._clear_locked('unavailable')

    def capture(self, context, windows, masks, prior, actions, logits, joints):
        if context is None:
            return
        mask = masks.detach().cpu().tolist()[0]
        encoded = windows.detach().cpu().tolist()
        prior = prior.detach().cpu().tolist()[0]
        actions = actions.detach().cpu().tolist()[0]
        n = sum(bool(value) for value in mask)
        index = context['decision_index']
        if (n < 1 or mask != [False] * (8-n) + [True] * n
                or type(index) is not int or not 0 <= index < self.session.max_decisions):
            self.capture_failed(context)
            return
        current = [float(value) * 180. for value in prior[-1]]
        previous = int(actions[-1])
        state = {'current_target': current, 'previous_action': previous,
                 'committed': [([float(value) * 180. for value in q], int(a))
                               for q, a in zip(prior[-n:-1], actions[-n:-1])],
                 'count': n-1, 'last_context': None,
                 'pending': {'window': encoded, 'mask': [mask], 'target': list(current),
                             'prior_action': previous,
                             'predicted_action': int(logits.argmax(-1).item())}}
        with self._lock:
            if (self._closed or self.session.cancel.is_set()
                    or context['generation'] != self._generation):
                return
            self._capture = {'before': context['before'], 'task': context['task'],
                             'slots': context['slots'], 'model_snapshot': state,
                             'observation': context['observation'],
                             'student_target': [float(value) * 180. for value in joints.detach().cpu().tolist()[0]],
                             'remaining_decisions': self.session.max_decisions - index - 1}

    def stage_segments(self, command_id, segment_index, waypoints):
        """Stage the executor's exact, ordered stop-per-waypoint plan.

        The next command is checked against this copy; it never directs or
        authorizes a movement. A changed plan is not silently stitched to an
        older virtual rollout.
        """
        try:
            if (not isinstance(command_id, str) or not command_id
                    or type(segment_index) is not int or not isinstance(waypoints, list)
                    or not 2 <= len(waypoints) <= 64 or not 0 <= segment_index < len(waypoints)):
                raise ValueError('invalid segmented display plan')
            points = []
            for row in waypoints:
                if (not isinstance(row, (list, tuple)) or len(row) != 4
                        or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) for value in row)):
                    raise ValueError('invalid display waypoint')
                points.append(list(row))
            steps = [list(quantize_joints(row)[0]) for row in points]
        except (TypeError, ValueError, OverflowError):
            self.invalidate('unavailable')
            return
        with self._lock:
            if self._closed or self.session.cancel.is_set():
                self._clear_locked('cancelled')
                return
            plan = self._segment_plan
            if segment_index == 0:
                if plan is not None or self._capture is None:
                    self._clear_locked('unavailable')
                    return
            elif (plan is None or not plan['finished'] or plan['active_id'] is not None
                    or segment_index != plan['index'] + 1 or steps != plan['steps']):
                self._clear_locked('unavailable')
                return
            self._staged_segment = {'command_id': command_id, 'index': segment_index,
                                    'waypoints': points, 'steps': steps}

    def command_started(self, command, trajectory):
        with self._lock:
            if self._closed or self.session.cancel.is_set():
                self._clear_locked('cancelled')
                return
            staged, self._staged_segment = self._staged_segment, None
            plan = self._segment_plan
            if staged is not None:
                valid = (staged['command_id'] == command['command_id']
                         and command.get('kind') == 'move_joints'
                         and command.get('target_joints_deg') is not None
                         and list(quantize_joints(command['target_joints_deg'])[0])
                             == staged['steps'][staged['index']])
                if not valid:
                    self._clear_locked('unavailable')
                    self._result['basis_command_id'] = command['command_id']
                    return
                if staged['index'] > 0:
                    if (plan is None or not plan['finished'] or plan['active_id'] is not None
                            or staged['index'] != plan['index'] + 1 or staged['steps'] != plan['steps']):
                        self._clear_locked('unavailable')
                        self._result['basis_command_id'] = command['command_id']
                        return
                    plan.update(index=staged['index'], active_id=command['command_id'], finished=False)
                    self._set_segment_result_locked()
                    return
            elif plan is not None:
                # A retry, correction or unrelated command is not one of the
                # staged decision waypoints and cannot reuse this rollout.
                self._clear_locked('unavailable')
                self._result['basis_command_id'] = command['command_id']
                return
            self._generation += 1
            generation = self._generation
            self._job = None
            self._result = {'segments': [], 'complete': False, 'reason': 'unavailable',
                            'scope': 'current_transfer', 'basis_command_id': command['command_id']}
            if (command.get('kind') in ('move_joints', 'blended_joint_path') and not self.mock
                    and self.controller.config.backend == 'sdk'):
                # This is the requested controller command, before admission.
                # The shadow forecast remains virtual and does not model SDK
                # acceleration, independent joint timing, or physical arrival.
                control_mode = ('sdk_blended_position_targets' if command['kind'] == 'blended_joint_path'
                                else 'sdk_single_position_target')
                self._result.update(basis_control_mode=control_mode,
                                    path_timing_basis='conditional_virtual_only',
                                    actual_sdk_path_or_arrival_time_confirmed=False)
            # Only the first command after a decision can use its observation.
            # Later waypoints, retries and corrections need a new decision.
            captured, self._capture = self._capture, None
            if self._closed or captured is None or captured['observation'] is None:
                return
            action = motion_policy.ACTIONS[captured['model_snapshot']['pending']['predicted_action']]
            kind = command['kind']
            if ((action == 'move' and kind not in ('move_joints', 'blended_joint_path', 'mock_joint_trajectory'))
                    or (action in ('open', 'close') and
                        (kind != 'set_gripper' or command['gripper_state'] != ('open' if action == 'open' else 'closed')))
                    or action == 'done'):
                return
            observation = captured['observation']
            prior_gripper = observation.get('gripper_command')
            if prior_gripper not in ('open', 'closed') or observation.get('tcp_mm') is None:
                return
            if action == 'close':
                self._close_z = float(observation['tcp_mm'][2])
            height_guard = None
            mock_descent = self.mock and bool(getattr(self.session, 'execute_mock_virtual_descent', False))
            if mock_descent:
                height_guard = SupportHeightGuard(captured['task'],
                    captured['slots'][captured['task']['destination_slot'] - 1])
                height_guard.close_observed_z_mm = self._close_z
                if prior_gripper == 'closed' and self._close_z is None:
                    return
            # The existing shadow uses the default tool geometry; refuse to
            # forecast with a differently configured physical tool.
            if (self.controller.config.tool_forward_mm, self.controller.config.tool_vertical_mm) != (80., -6.):
                return
            current = captured['model_snapshot']['current_target']
            if action == 'move':
                target = (list(trajectory[-1]) if trajectory else
                          list(quantize_joints(command['target_joints_deg'])[1]))
                if staged is not None:
                    if staged['steps'][-1] != list(quantize_joints(captured['student_target'])[0]):
                        return
                    target = list(staged['waypoints'][-1])
                elif list(quantize_joints(target)[0]) != list(quantize_joints(captured['student_target'])[0]):
                    return
            else:
                if staged is not None:
                    return
                target = list(current)
            opening = .04 if (action == 'open' or (action == 'move' and prior_gripper == 'open')) else 0.
            observed_opening = .04 if prior_gripper == 'open' else 0.
            job = {'number': command['command_id'],
                   'observation': {'joints_deg': observation['joints_deg'],
                                   'gripper_opening_m': observed_opening},
                   'targets': {'joints_deg': list(current), 'gripper_opening_m': observed_opening},
                   'student_template': self._student, 'model_snapshot': captured['model_snapshot'],
                   'before': captured['before'], 'task': captured['task'], 'slots': captured['slots'],
                   'planned': None,
                   # The bounded policy history can be shorter than the run
                   # index after clipping or discarded decisions. Keep those
                   # two counts separate when limiting future decisions.
                   'max_decisions': captured['model_snapshot']['count'] + 1 + captured['remaining_decisions'],
                   'target': target, 'opening': opening,
                   'finger_profile': 'minimum_jerk_0p5s' if self.mock else 'instant',
                   'move_profile': 'smoothstep' if self.mock else 'minimum_jerk',
                   'straight_descent': mock_descent, 'support_height_guard_enabled': mock_descent,
                   'support_height_guard': height_guard, 'program_snapshot': None}
            self._result['reason'] = 'computing'
            if staged is not None:
                self._segment_plan = {'generation': generation, 'origin_id': command['command_id'],
                    'active_id': command['command_id'], 'index': 0, 'finished': False,
                    'waypoints': staged['waypoints'], 'steps': staged['steps'],
                    'control_basis': {key: self._result[key] for key in (
                        'basis_control_mode', 'path_timing_basis',
                        'actual_sdk_path_or_arrival_time_confirmed') if key in self._result}}
                self._future_result = {'segments': [], 'complete': False, 'reason': 'computing'}
                self._set_segment_result_locked()
            self._job = (generation, job)
            if self._thread is None:
                if not _FORECAST_GATE.acquire(blocking=False):
                    self._job = None
                    self._result['reason'] = 'busy'
                    if self._segment_plan is not None:
                        self._future_result['reason'] = 'busy'
                    return
                self._thread = threading.Thread(target=self._work, daemon=True, name='controller-forecast')
                try:
                    self._thread.start()
                except BaseException:
                    self._thread = None
                    self._job = None
                    _FORECAST_GATE.release()
                    raise

    def _set_segment_result_locked(self):
        plan = self._segment_plan
        self._result = {**deepcopy(self._future_result), **plan['control_basis'],
                        'scope': 'current_transfer', 'basis_command_id': plan['active_id'],
                        'source': 'conditional_virtual_forecast_from_student_inputs',
                        'forecast_assumption': 'virtual_direct_move_to_student_endpoint',
                        'actual_path_or_arrival_time_confirmed': False}

    def _publish(self, generation, command_id, value):
        with self._lock:
            if (self._closed or generation != self._generation or self.session.cancel.is_set()):
                return
            plan = self._segment_plan
            if plan is not None and plan['generation'] == generation and plan['origin_id'] == command_id:
                self._future_result = deepcopy(value)
                if plan['active_id'] is not None:
                    self._set_segment_result_locked()
                return
            control_basis = {key: self._result[key] for key in (
                'basis_control_mode', 'path_timing_basis', 'actual_sdk_path_or_arrival_time_confirmed')
                if key in self._result}
            self._result = {**deepcopy(value), **control_basis, 'scope': 'current_transfer',
                            'basis_command_id': command_id,
                            'source': 'conditional_virtual_forecast_from_student_inputs'}

    def _work(self):
        released = False
        try:
            while True:
                with self._lock:
                    item, self._job = self._job, None
                    if self._closed or item is None:
                        self._thread = None
                        _FORECAST_GATE.release()
                        released = True
                        return
                generation, job = item
                command_id = job['number']
                job['_emit_prefix'] = lambda value: self._publish(generation, command_id, value)
                try:
                    result = _forecast(job, _Cancelled(self, generation))
                except InterruptedError:
                    continue
                except Exception:
                    result = {'segments': [], 'complete': False, 'reason': 'unavailable'}
                self._publish(generation, command_id, result)
        finally:
            if not released:
                with self._lock:
                    self._thread = None
                    self._job = None
                _FORECAST_GATE.release()

    def _clear_locked(self, reason):
        self._generation += 1
        self._job = None
        self._decision_context = None
        self._capture = None
        self._staged_segment = None
        self._segment_plan = None
        self._future_result = None
        self._result = {'segments': [], 'complete': False, 'reason': reason,
                        'scope': 'current_transfer', 'basis_command_id': None}

    def invalidate(self, reason='unavailable'):
        with self._lock:
            self._clear_locked(reason)

    def command_finished(self, command_id, response=None):
        with self._lock:
            plan = self._segment_plan
            if plan is not None and plan['active_id'] == command_id:
                success = (isinstance(response, dict) and response.get('command_id') == command_id
                           and response.get('replayed') is False
                           and ((response.get('status') == 'completed' and response.get('applied') is True)
                                or (response.get('status') == 'already_at_target'
                                    and response.get('applied') is False)))
                if (not success or self.session.cancel.is_set()
                        or plan['index'] + 1 == len(plan['waypoints'])):
                    self._clear_locked('no_active_command' if success else 'unavailable')
                    return
                plan.update(active_id=None, finished=True)
                self._result = {'segments': [], 'complete': False, 'reason': 'no_active_command',
                                'scope': 'current_transfer', 'basis_command_id': None}
                return
            if self._result.get('basis_command_id') == command_id:
                self._generation += 1
                self._job = None
                self._result = {'segments': [], 'complete': False, 'reason': 'no_active_command',
                                'scope': 'current_transfer', 'basis_command_id': None}

    def snapshot(self):
        with self._lock:
            if self.session.cancel.is_set() and self._result.get('reason') != 'cancelled':
                self._clear_locked('cancelled')
            result = deepcopy(self._result)
            plan = self._segment_plan
            if (plan is None or plan['active_id'] is None
                    or result.get('basis_command_id') != plan['active_id']):
                return result
            generation, active_id, index = self._generation, plan['active_id'], plan['index']
            waypoints = deepcopy(plan['waypoints'])
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        nominal = []
        for start, target in zip(waypoints[index:-1], waypoints[index + 1:]):
            if start == target:
                continue
            nominal.append([list(geometry.forward([a + (b-a) * i / 12
                           for a, b in zip(start, target)])) for i in range(13)])
        with self._lock:
            if (generation != self._generation or self.session.cancel.is_set()
                    or self._segment_plan is None or self._segment_plan['active_id'] != active_id):
                return deepcopy(self._result)
        result['segments'] = nominal + result.get('segments', [])
        result['nominal_remaining_segment_count'] = len(nominal)
        return result

    def close(self, *, wait=False):
        with self._lock:
            self._closed = True
            self._generation += 1
            self._job = None
            self._decision_context = None
            self._capture = None
            self._staged_segment = None
            self._segment_plan = None
            self._future_result = None
            thread = self._thread
            self._result = {'segments': [], 'complete': False, 'reason': 'closed',
                            'scope': 'current_transfer', 'basis_command_id': None}
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.)
        return thread is None or not thread.is_alive()
