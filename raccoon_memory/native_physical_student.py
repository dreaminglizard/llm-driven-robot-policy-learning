"""Closed-loop student/controller adapter; it never creates or opens a device.

The caller owns registration, the controller, and the evidence for the initial
applied target.  This is not connected to the application or its virtual-only
session.  A model DONE is not evidence of an object transfer or physical success.
"""
from __future__ import annotations

import math
import copy
import threading
import time
import uuid

import torch

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, JOINT_LIMITS, STEPS_PER_DEGREE, quantize_joints
from raccoon_control.near_joint_limit_guard import GuardedPolicy
from raccoon_control.paths import vertical_segments
from . import motion_policy, scene_features, straight_descent
from .native_approach_session import AppliedHistoryPolicy, native_command_target, native_policy_decision
from .robot_bridge import DEFAULT_BLENDED_MAX_VELOCITY_DEG_S, SlotCalibration, TransferCompiler
from .support_height_guard import SupportHeightGuard
from .experimental_transport_adjustment import lowered_grasp_target, source_floor_target, validated_bias_target

_MOCK_SOURCE_RETURN_LIFT_Z_MM = 134.
_MOCK_SOURCE_RETURN_CEILING_MARGIN_MM = 5.
# Development-only nominal TCP tolerances; these are not object measurements.
_MOCK_SOURCE_RETURN_XY_TOLERANCE_MM = 3.
_MOCK_SOURCE_RETURN_Z_BELOW_MM = 3.
_MOCK_SOURCE_RETURN_Z_ABOVE_MM = 20.
_MOCK_VERTICAL_REPLAN_WALL_SECONDS = 30.


class _NotSentHistoryPolicy(AppliedHistoryPolicy):
    """Discard an untransmitted decision without inventing a control event."""

    def discard_unconfirmed_move(self):
        """Keep the last confirmed action and goal when a MOVE has no confirmed endpoint."""
        if (self.pending is None or self.pending['predicted_action'] != motion_policy.MOVE
                or self.current_target is None):
            raise ValueError('no pending MOVE decision')
        self.pending = None

    def on_move_not_sent(self):
        self.discard_unconfirmed_move()

    def on_move_retained(self, nominal_target=None):
        """Advance one policy step while reusing a confirmed existing goal."""
        if (self.pending is None or self.pending['predicted_action'] != motion_policy.MOVE
                or self.current_target is None):
            raise ValueError('no pending MOVE decision')
        self.committed = (self.committed + [(list(self.current_target), self.previous_action)])[-7:]
        if nominal_target is not None:
            self.current_target = _joints(nominal_target)
        self.previous_action = motion_policy.MOVE
        self.count += 1
        self.pending = None

    def on_decision_not_sent(self):
        if self.pending is None or self.current_target is None:
            raise ValueError('no pending policy decision')
        self.pending = None

    def on_move_interrupted(self, stationary_reference):
        """Record a sent MOVE and observed stop, never a reached target."""
        if (self.pending is None or self.pending['predicted_action'] != motion_policy.MOVE
                or self.current_target is None):
            raise ValueError('no pending MOVE decision')
        self.committed = (self.committed + [(list(self.current_target), self.previous_action)])[-7:]
        self.current_target = list(stationary_reference)
        self.previous_action = motion_policy.MOVE
        self.count += 1
        self.pending = None

    def on_open_setdown_superseded(self, control_reference):
        """Record a real executor MOVE; the student's OPEN was never sent."""
        if self.pending is None or self.pending['predicted_action'] != motion_policy.OPEN:
            raise ValueError('no pending OPEN decision')
        self.committed = (self.committed + [(list(self.current_target), self.previous_action)])[-7:]
        self.current_target = list(control_reference)
        self.previous_action = motion_policy.MOVE
        self.count += 1
        self.pending = None


class _Abort(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _joints(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(not _number(item) for item in value)):
        raise _Abort('invalid_joints', 'four finite joint angles are required')
    result = [float(item) for item in value]
    if not all(low <= q <= high for q, (low, high) in zip(result, JOINT_LIMITS)):
        raise _Abort('joint_limit', 'joint angles are outside the permitted range')
    return result


def _close(left, right, tolerance=1e-7):
    return len(left) == len(right) and all(abs(a-b) <= tolerance for a, b in zip(left, right))


class _EndpointPreservingGeometry:
    """Use the Cartesian subdivision without changing the student's joint goal."""

    def __init__(self, geometry, target_xyz, requested, start_joints):
        self.geometry, self.target_xyz, self.requested = geometry, target_xyz, requested
        self.start_xyz = geometry.forward(start_joints)
        self.start_pitch = 90. + sum(start_joints[1:])
        self.target_pitch = 90. + sum(requested[1:])

    def forward(self, joints):
        return self.geometry.forward(joints)

    def inverse(self, xyz):
        if _close(xyz, self.target_xyz, 1e-7):
            return tuple(self.requested)
        delta = [b-a for a, b in zip(self.start_xyz, self.target_xyz)]
        length_squared = sum(value * value for value in delta)
        fraction = (sum((value-start)*change for value, start, change in zip(xyz, self.start_xyz, delta))
                    / length_squared if length_squared else 0.)
        pitch = self.start_pitch + max(0., min(1., fraction)) * (self.target_pitch - self.start_pitch)
        return self.geometry.inverse(xyz, pitch)


class NativePhysicalStudent:
    """Use an injected controller and student for one registered transfer.

    The initial policy reference is either a caller-confirmed applied target
    or an explicitly marked stationary observation; the latter is never reported
    as an applied target. Confirmed applied targets must be quantized unless the
    explicitly enabled mock trajectory previously applied a simulated target.
    The controller must already be started, and the caller must have established
    the registration and permitted workspace.  This class neither loads a model
    checkpoint nor starts, closes, recovers, or connects a controller.
    """

    def __init__(self, controller, student, *, speed_pct, max_decisions=60,
                 enable_arrival_correction=False, enable_blended_path=False, enable_release_setdown=False,
                 compare_virtual_descent=False, execute_mock_virtual_descent=False,
                 path_calibration=None, enable_experimental_segmented_path=False,
                 enable_mock_source_return=False, enable_mock_interrupt_retarget=False,
                 enable_runtime_source_return=False, enable_mock_nominal_trim=False,
                 require_retarget_release_check=False,
                 enable_mock_layered_grasp_lowering=False, enable_mock_early_source_return=False,
                 enable_mock_open_drift_reconcile=False, enable_mock_vertical_replan=False,
                 enable_mock_move_release_upper_separation=False):
        limits = controller.config.limits
        for flag_name, flag_value in (
                ('enable_blended_path', enable_blended_path),
                ('enable_release_setdown', enable_release_setdown),
                ('enable_arrival_correction', enable_arrival_correction),
                ('compare_virtual_descent', compare_virtual_descent),
                ('execute_mock_virtual_descent', execute_mock_virtual_descent),
                ('enable_experimental_segmented_path', enable_experimental_segmented_path),
                ('enable_mock_source_return', enable_mock_source_return),
                ('enable_mock_interrupt_retarget', enable_mock_interrupt_retarget),
                ('enable_runtime_source_return', enable_runtime_source_return),
                ('require_retarget_release_check', require_retarget_release_check),
                ('enable_mock_nominal_trim', enable_mock_nominal_trim)):
            if type(flag_value) is not bool:
                raise ValueError(f'{flag_name} must be boolean')
        if any(type(flag) is not bool for flag in (
                enable_mock_layered_grasp_lowering, enable_mock_early_source_return,
                enable_mock_open_drift_reconcile, enable_mock_vertical_replan,
                enable_mock_move_release_upper_separation)):
            raise ValueError('mock development flags must be boolean')
        if enable_blended_path and not enable_arrival_correction:
            raise ValueError('blended paths require the common arrival contract')
        if enable_release_setdown and (not enable_arrival_correction or not enable_experimental_segmented_path):
            raise ValueError('release setdown requires calibrated paths and common arrival feedback')
        if enable_arrival_correction and enable_mock_nominal_trim:
            raise ValueError('common arrival correction and mock nominal trim are exclusive')
        if enable_arrival_correction and (path_calibration is None or not enable_experimental_segmented_path
                                          or compare_virtual_descent or execute_mock_virtual_descent):
            raise ValueError('arrival correction requires calibrated segmented paths')
        if (not _number(speed_pct) or not 1 <= speed_pct <= limits.max_speed_pct
                or type(max_decisions) is not int or not 1 <= max_decisions <= 60):
            raise ValueError('invalid speed or decision limit')
        if ((enable_mock_layered_grasp_lowering or enable_mock_early_source_return or enable_mock_open_drift_reconcile
             or enable_mock_vertical_replan)
                and not enable_mock_nominal_trim):
            raise ValueError('mock development options require mock nominal trim')
        if enable_mock_move_release_upper_separation and (not enable_mock_nominal_trim or not enable_mock_source_return):
            raise ValueError('mock MOVE upper separation requires mock nominal trim and source return')
        if enable_mock_vertical_replan and not enable_mock_source_return:
            raise ValueError('mock vertical replanning requires mock source return')
        if enable_mock_early_source_return and not enable_mock_source_return:
            raise ValueError('mock early return requires mock source return')
        if enable_mock_nominal_trim and (controller.config.backend != 'mock'
                or controller.config.hardware_enabled is not False or controller.config.port is not None
                or path_calibration is None or not enable_experimental_segmented_path
                or enable_runtime_source_return
                or compare_virtual_descent or execute_mock_virtual_descent):
            raise ValueError('nominal trim requires an isolated mock controller and calibrated segmented paths')
        if enable_runtime_source_return and (enable_mock_source_return or enable_mock_interrupt_retarget):
            raise ValueError('runtime and mock source return modes are exclusive')
        if enable_runtime_source_return and (path_calibration is None or not enable_experimental_segmented_path
                or compare_virtual_descent or execute_mock_virtual_descent
                or not callable(getattr(controller, 'interrupt_motion', None))
                or not callable(getattr(controller, 'resume_interrupted_motion', None))):
            raise ValueError('runtime source return requires calibrated paths and controller interruption')
        if enable_mock_interrupt_retarget and (not enable_mock_source_return
                or not callable(getattr(controller, 'interrupt_mock_motion', None))
                or not callable(getattr(controller, 'resume_mock_interrupted_motion', None))):
            raise ValueError('mock interruption requires source return and controller support')
        if enable_mock_source_return and (controller.config.backend != 'mock'
                or controller.config.hardware_enabled is not False or controller.config.port is not None):
            raise ValueError('source return is an explicit mock-only experiment')
        if enable_mock_source_return and (compare_virtual_descent or execute_mock_virtual_descent):
            raise ValueError('source return cannot use the separate mock virtual-descent profile')
        if enable_experimental_segmented_path and path_calibration is None:
            raise ValueError('experimental segmented path requires explicit path calibration')
        if ((compare_virtual_descent or execute_mock_virtual_descent) and
                (controller.config.backend != 'mock' or controller.config.hardware_enabled is not False)):
            raise ValueError('virtual descent requires a mock-only controller')
        if (execute_mock_virtual_descent and
                getattr(controller, 'supports_mock_joint_trajectory', False) is not True):
            raise ValueError('controller lacks the mock trajectory capability')
        if controller.config.mode != 'calibration' or not controller.config.workspace_ready:
            raise ValueError('controller workspace and calibration mode must be ready')
        if path_calibration is not None:
            if not isinstance(path_calibration, SlotCalibration):
                raise ValueError('path calibration must be a SlotCalibration')
            if execute_mock_virtual_descent and enable_experimental_segmented_path:
                raise ValueError('mock trajectory execution does not enforce physical path calibration')
            path_calibration.require_ready(controller.config)
            if speed_pct > path_calibration.speed_pct:
                raise ValueError('speed exceeds the calibrated path speed')
        self.controller = controller
        self.model = GuardedPolicy(_NotSentHistoryPolicy(student)).eval()
        self.speed_pct = float(speed_pct)
        self.max_decisions = max_decisions
        self.enable_arrival_correction = enable_arrival_correction
        self.enable_blended_path = enable_blended_path
        self.enable_release_setdown = enable_release_setdown
        self.compare_virtual_descent = compare_virtual_descent
        self.execute_mock_virtual_descent = execute_mock_virtual_descent
        self.path_calibration = path_calibration
        self.enable_experimental_segmented_path = enable_experimental_segmented_path
        self.enable_mock_source_return = enable_mock_source_return
        self.enable_mock_interrupt_retarget = enable_mock_interrupt_retarget
        self.enable_runtime_source_return = enable_runtime_source_return
        self.require_retarget_release_check = require_retarget_release_check
        self.enable_mock_nominal_trim = enable_mock_nominal_trim
        self.enable_mock_layered_grasp_lowering = enable_mock_layered_grasp_lowering
        self.enable_mock_early_source_return = enable_mock_early_source_return
        self.enable_mock_open_drift_reconcile = enable_mock_open_drift_reconcile
        self.enable_mock_vertical_replan = enable_mock_vertical_replan
        self.enable_mock_move_release_upper_separation = enable_mock_move_release_upper_separation
        self._source_return_enabled = enable_mock_source_return or enable_runtime_source_return
        self._interrupt_retarget_enabled = enable_mock_interrupt_retarget or enable_runtime_source_return
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        # Serializes admission of a source-return request with the final call
        # of an original MOVE. It is deliberately never held by cancel().
        self._mock_move_dispatch_lock = threading.Lock()
        self._mock_move_dispatch_owner = None
        self._source_return_admission_done = threading.Event()
        self._source_return_admission_done.set()
        self._source_return_admission_waiting = threading.Event()
        self._cancelled = threading.Event()
        self._lease = None
        self._stop_error = None
        self._source_return_state = None
        self._direct_previous_destination_slot = None
        self._forecast_run_token = None

    def _publish_decision_context(self, observation, before, task, slots, decision_index):
        # The optional display observes the same public inputs as this policy.
        # It must never affect the policy or the controller's actual commands.
        try:
            callback = getattr(self.controller, '_student_decision_context', None)
            if callable(callback):
                callback(observation, before, task, slots, self._forecast_run_token, decision_index)
        except Exception:
            pass

    def _publish_segment_plan(self, command_id, segment_index, segments):
        # This is only a copy of the already planned endpoints. It cannot
        # change command selection, admission, or policy history.
        try:
            callback = getattr(self.controller, '_student_segment_plan', None)
            if callable(callback):
                callback(command_id, segment_index, [list(part['effective']) for part in segments])
        except Exception:
            pass

    def mock_source_return_status(self):
        """Snapshot the experimental request, separately from its application."""
        with self._state_lock:
            state = self._source_return_state
            status = ('unsupported' if not self._source_return_enabled else
                      'inactive' if state is None or state['phase'] != 'running' else
                      'applied' if state['activation_decision_index'] is not None else
                      'requested' if state['pending'] or state['switched'] else
                      'waiting' if state['admission_waiting'] else 'inactive')
            progress = {'status': status,
                    'source_slot': state['source_slot'] if state is not None else None,
                    'staged_redirect_available': bool(
                        state is not None and state['phase'] == 'running'
                        and not state['pending'] and not state['switched']
                        and not state['admission_waiting'] and not state['opened_after_close']
                        and not (state['ever_closed'] and state['inflight_action'] in ('open', 'done'))
                        and ((state['gripper'] == 'open' and not state['ever_closed']
                              and not state['close_dispatch_started']) or state['gripper'] == 'closed')
                        and not self._cancelled.is_set()),
                    'activation_decision_index': (state['activation_decision_index']
                                                  if state is not None else None),
                    'physical_success': None}
            if state is not None and state.get('direct_task') is not None:
                progress.update(mode='direct_held_transfer',
                                destination_slot=state['direct_task']['destination_slot'])
            return progress

    def request_mock_source_return(self, *, source_slot, cancel_event=None, replacement_task=None):
        """Request a mock-only switch, optionally interrupting an active MOVE."""
        if not self._source_return_enabled:
            return {'status': 'retarget_unsupported', 'reason': 'mock_source_return_disabled'}

        def rejection(state, *, waiting=False):
            config = self.controller.config
            if (not self.enable_runtime_source_return and
                    (config.backend != 'mock' or config.hardware_enabled is not False or config.port is not None)):
                return {'status': 'retarget_unsupported', 'reason': 'mock_controller_required'}
            if (self._cancelled.is_set() or (cancel_event is not None and cancel_event.is_set())
                    or state is None or state['phase'] != 'running'):
                return {'status': 'retarget_rejected', 'reason': 'transfer_not_active'}
            if type(source_slot) is not int or source_slot != state['source_slot']:
                return {'status': 'retarget_rejected', 'reason': 'source_slot_mismatch'}
            if state['pending'] or state['switched'] or (state['admission_waiting'] and not waiting):
                return {'status': 'retarget_rejected', 'reason': 'duplicate_source_return'}
            if state['opened_after_close'] or (state['ever_closed'] and state['inflight_action'] in ('open', 'done')):
                return {'status': 'retarget_rejected', 'reason': 'gripper_opening_or_opened'}
            if state['gripper'] != 'closed' and not (state['gripper'] == 'open' and not state['ever_closed']):
                return {'status': 'retarget_rejected', 'reason': 'gripper_not_closed'}
            return None

        with self._state_lock:
            state = self._source_return_state
            pregrasp = (self.enable_mock_layered_grasp_lowering and state is not None
                        and state.get('phase') == 'running' and state.get('pregrasp_lowering') is True
                        and state.get('gripper') == 'open' and type(source_slot) is int
                        and source_slot == state.get('source_slot')
                        and not self._cancelled.is_set()
                        and not (cancel_event is not None and cancel_event.is_set()))
            if pregrasp:
                state['pregrasp_cancel_requested'] = True
        if pregrasp:
            stopped = self.cancel()
            if not isinstance(stopped, dict) or stopped.get('confirmed') is not True:
                raise ControlError('INTERRUPTION_UNCONFIRMED', 'Pregrasp stop could not be confirmed')
            if cancel_event is not None and cancel_event.is_set():
                return {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
            return {'status': 'retarget_requested', 'reason': 'pregrasp_stopped', 'source_slot': source_slot,
                    'physical_success': None}
        close_in_flight = False
        with self._state_lock:
            state = self._source_return_state
            denied = rejection(state)
            if denied is not None:
                return denied
            direct_task = None
            if replacement_task is not None and self.enable_runtime_source_return:
                try:
                    _, direct_task = scene_features.transfer_task(state['registered_scene'], replacement_task)
                    original = state['original_task']
                    if (direct_task['source_slot'] != source_slot
                            or direct_task['destination_slot'] in (source_slot, original['destination_slot'])
                            or direct_task['destination_slot'] not in self.path_calibration.enabled_slots
                            or direct_task['destination_level'] > self.path_calibration.max_height):
                        raise ValueError('invalid replacement task')
                except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                    return {'status': 'retarget_rejected', 'reason': 'destination_infeasible'}
            # A synchronous callback inside apply_action cannot wait for its
            # own return, nor safely claim that the controller sent it.
            if self._mock_move_dispatch_owner == threading.get_ident():
                return {'status': 'retarget_rejected', 'reason': 'reentrant_dispatch_request'}
            if state['gripper'] == 'open' and not state['ever_closed']:
                if state.get('close_dispatch_started'):
                    close_in_flight = True
                else:
                    # Ordered with the final CLOSE admission under this lock.
                    state['pregrasp_cancel_requested'] = True
                    state['pregrasp_return_requested'] = True
            if not close_in_flight:
                state['admission_waiting'] = True
                self._source_return_admission_done.clear()
                self._source_return_admission_waiting.set()
                command_id = state.get('active_command_id') if self._interrupt_retarget_enabled else None
                lease = self._lease
        # A pending task change may supersede the rest of the original plan,
        # even when no active command needs to be interrupted.
        try:
            invalidate = getattr(self.controller, '_invalidate_student_forecast', None)
            if callable(invalidate):
                invalidate('retargeting')
        except Exception:
            pass
        if close_in_flight:
            # An admitted CLOSE may already have been delivered. Stop it but
            # leave the transfer uncertain rather than claim the scene stayed.
            stopped = self.cancel()
            if not isinstance(stopped, dict) or stopped.get('confirmed') is not True:
                raise ControlError('INTERRUPTION_UNCONFIRMED', 'CLOSE delivery or stop is unconfirmed')
            return {'status': 'retarget_requested', 'reason': 'close_in_flight_stop_requested',
                    'source_slot': source_slot, 'physical_success': None}
        try:
            if command_id is not None and lease is not None:
                try:
                    interrupt = (self.controller.interrupt_motion if self.enable_runtime_source_return
                                 else self.controller.interrupt_mock_motion)
                    interruption = interrupt(lease, command_id)
                except ControlError as exc:
                    if exc.code != 'COMMAND_ALREADY_FINISHED':
                        # Only explicit pre-admission refusals establish that
                        # this interruption did not reach the controller. A
                        # lost reply or device/server error may follow a sent
                        # stop; never let the old transfer continue in that
                        # uncertain state or claim the request was rejected.
                        if exc.code not in {'CLOSED', 'LEASE_REQUIRED', 'LEASE_EXPIRED',
                                            'STOP_IN_PROGRESS', 'INTERRUPTION_MISMATCH',
                                            'NOT_ARMED', 'UNSUPPORTED_ACTION'}:
                            self.cancel()
                            raise
                        return {'status': 'retarget_rejected', 'reason': 'interruption_failed',
                                'error': exc.as_dict(), 'physical_success': None}
                except Exception:
                    self.cancel()
                    raise
                else:
                    if isinstance(interruption, dict) and interruption.get('superseded_by_stop') is True:
                        self._cancelled.set()
                        return {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
                    if (not isinstance(interruption, dict)
                            or interruption.get('command_id') != command_id
                            or interruption.get('superseded_by_stop') is not False
                            or not isinstance(interruption.get('stop'), dict)
                            or interruption['stop'].get('confirmed') is not True):
                        self.cancel()
                        raise ControlError('INTERRUPTION_UNCONFIRMED',
                                           'Interrupted command outcome could not be confirmed')
                    with self._state_lock:
                        current = self._source_return_state
                        denied = (rejection(current, waiting=True) if current is state else
                                  {'status': 'retarget_rejected', 'reason': 'transfer_not_active'})
                        if denied is not None:
                            return denied
                        state['pending'] = True
                        state['pending_cancel_event'] = cancel_event
                        state['pending_direct_task'] = (copy.deepcopy(direct_task)
                                                        if state['gripper'] == 'closed' else None)
                        state['interruption'] = interruption
                        return {'status': 'retarget_requested', 'reason': 'queued_after_interrupt',
                                'source_slot': source_slot, 'physical_success': None}
            with self._mock_move_dispatch_lock:
                with self._state_lock:
                    current = self._source_return_state
                    denied = (rejection(current, waiting=True) if current is state else
                              {'status': 'retarget_rejected', 'reason': 'transfer_not_active'})
                    if denied is not None:
                        return denied
                    state['pending'] = True
                    state['pending_cancel_event'] = cancel_event
                    state['pending_direct_task'] = (copy.deepcopy(direct_task)
                                                    if state['gripper'] == 'closed' else None)
                    return {'status': 'retarget_requested', 'reason': 'queued_for_decision_boundary',
                            'source_slot': source_slot, 'physical_success': None}
        finally:
            with self._state_lock:
                if (not state['pending'] and not state['switched'] and not self._cancelled.is_set()):
                    state['pregrasp_cancel_requested'] = False
                    state['pregrasp_return_requested'] = False
                state['admission_waiting'] = False
                if self._source_return_state is state:
                    self._source_return_admission_waiting.clear()
                    self._source_return_admission_done.set()

    def _source_return_decision(self, observation, before, task, slots, history, previous, decision_index):
        """Use a policy-only return context while retaining actual control history."""
        if self.enable_runtime_source_return:
            context_scene, context_task = self._return_policy_context(before, task)
            features = scene_features.encode_scene_features(observation, context_scene, context_task, slots)
            self._publish_decision_context(observation, context_scene, context_task, slots, decision_index)
        else:
            features = scene_features.encode_scene_features(observation, before, task, slots)
            source = task['source_slot']
            level = len(before[source - 1])
            for index, name in enumerate(scene_features.FEATURE_NAMES):
                if name.startswith('destination_slot_'):
                    features[index] = float(name == f'destination_slot_{source}')
                elif name.startswith('destination_level_'):
                    features[index] = float(name == f'destination_level_{level}')
            # The experimental mock edits individual features rather than
            # encoding a full task. There is no matching scene to forecast.
            self._publish_decision_context(observation, None, None, slots, decision_index)
        history.append(features)
        window, mask = scene_features.history_window(history, 8)
        with torch.inference_mode():
            logits, joints = self.model.forward_with_control_state(
                torch.tensor([window], dtype=torch.float32),
                torch.tensor([mask], dtype=torch.bool),
                torch.tensor([previous], dtype=torch.float32))
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        return features, mask, logits, joints, action

    @staticmethod
    def _return_policy_context(before, task):
        """A valid reverse-task context, never an observed/committed layout.

        The carried object is assigned to the original destination solely to
        encode a known reverse task. Actual registration and release checks
        continue using ``before`` and the original source slot.
        """
        from . import task_catalog
        projected = task_catalog.transfer(before, task['source_slot'], task['destination_slot'])
        return scene_features.transfer_task(projected, {
            'source_slot': task['destination_slot'], 'destination_slot': task['source_slot']})

    def _source_return_open_pose(self, observation, applied, task, slots):
        """Compare confirmed joints with the registered *nominal* source pose."""
        source = task['source_slot'] - 1
        calibration = self.path_calibration
        if calibration is not None:
            if not _close(slots[source], calibration.slot_xy_mm[source]):
                raise _Abort('source_return_calibration_mismatch', 'source slot differs from calibration')
            table, height, grip = (calibration.table_z_mm, calibration.cylinder_height_mm,
                                   calibration.grip_offset_mm)
            reference = 'registered_calibration'
        else:
            # The mock scene/teacher uses a 0 mm table, 20 mm cylinders and a
            # 15 mm TCP grip offset. It is not a physical calibration.
            table, height, grip = 0., 20., 15.
            reference = 'mock_nominal_geometry'
        expected = [float(slots[source][0]), float(slots[source][1]),
                    table + (task['source_level'] - 1) * height + grip]
        geometry = self._geometry()
        target = list(geometry.forward(_joints(applied)))
        observed = list(geometry.forward(_joints(observation['joints_deg'])))
        target_xy = math.dist(target[:2], expected[:2])
        observed_xy = math.dist(observed[:2], expected[:2])
        target_z, observed_z = target[2] - expected[2], observed[2] - expected[2]
        return {'nominal_tcp_mm': expected, 'target_tcp_mm': target, 'observed_tcp_mm': observed,
                'target_xy_error_mm': target_xy, 'observed_xy_error_mm': observed_xy,
                'target_z_offset_mm': target_z, 'observed_z_offset_mm': observed_z,
                'xy_tolerance_mm': _MOCK_SOURCE_RETURN_XY_TOLERANCE_MM,
                'z_below_mm': _MOCK_SOURCE_RETURN_Z_BELOW_MM,
                'z_above_mm': _MOCK_SOURCE_RETURN_Z_ABOVE_MM,
                'geometry_source': reference,
                'matches': max(target_xy, observed_xy) <= _MOCK_SOURCE_RETURN_XY_TOLERANCE_MM
                           and all(-_MOCK_SOURCE_RETURN_Z_BELOW_MM <= offset <= _MOCK_SOURCE_RETURN_Z_ABOVE_MM
                                   for offset in (target_z, observed_z))}

    def _retarget_grip_pose(self, observation, applied, task, slots, *, release):
        """Check a nominal grip location; this is not an object observation."""
        pose_task = ({'source_slot': task['destination_slot'], 'source_level': task['destination_level']}
                     if release else task)
        pose = self._source_return_open_pose(observation, applied, pose_task, slots)
        # Keep the permitted height window narrower than a cylinder level.
        pose['z_below_mm'] = pose['z_above_mm'] = 5.
        pose['matches'] = (pose['matches'] and all(abs(pose[key]) <= 5. for key in
                           ('target_z_offset_mm', 'observed_z_offset_mm')))
        return pose

    def _source_return_lift_parameters(self, observation, applied):
        """Choose the configured runtime height and an observed starting pose."""
        geometry = self._geometry()
        start = _joints(observation['joints_deg']) if self.enable_runtime_source_return else _joints(applied)
        lift_z = (self.path_calibration.travel_z_mm if self.enable_runtime_source_return
                  else _MOCK_SOURCE_RETURN_LIFT_Z_MM)
        source = 'registered_calibration' if self.enable_runtime_source_return else 'mock_experiment'
        return geometry, geometry.forward(start), lift_z, source

    def _source_return_grasp_floor(self, proposed, task, slots):
        """Respect the robot pose recorded at the original confirmed grasp."""
        with self._state_lock:
            reference = self._source_return_state.get('grasp_reference')
        if not isinstance(reference, dict):
            raise _Abort('grasp_reference_unknown', 'the original robot grasp pose is unavailable')
        geometry = self._geometry()
        xyz = geometry.forward(proposed)
        if math.dist(xyz[:2], slots[task['source_slot'] - 1]) > self.path_calibration.vertical_corridor_mm:
            return proposed, None
        floor = reference['nominal_tcp_mm'][2]
        if xyz[2] >= floor:
            return proposed, None
        pitch = 90. + sum(proposed[1:])
        adjusted = _joints(geometry.inverse((xyz[0], xyz[1], floor), pitch))
        return adjusted, {'kind': 'recorded_robot_grasp_height', 'original_tcp_mm': list(xyz),
            'adjusted_tcp_mm': list(geometry.forward(adjusted)), 'grasp_reference': copy.deepcopy(reference),
            'object_pose_observed': False}

    def _mock_source_return_target_allowed(self, nominal, candidate, task, slots):
        """Keep a source-adjacent release approach inside the existing OPEN bounds."""
        calibration = self.path_calibration
        source_xy = slots[task['source_slot'] - 1]
        height = (calibration.table_z_mm + (task['source_level'] - 1) * calibration.cylinder_height_mm
                  + calibration.grip_offset_mm)
        geometry = self._geometry()
        nominal_tcp, candidate_tcp = geometry.forward(nominal), geometry.forward(candidate)
        if (math.dist(nominal_tcp[:2], source_xy) > _MOCK_SOURCE_RETURN_XY_TOLERANCE_MM
                or nominal_tcp[2] > height + _MOCK_SOURCE_RETURN_Z_ABOVE_MM):
            return True  # This is not yet a source-adjacent release approach.
        if math.dist(candidate_tcp[:2], source_xy) > _MOCK_SOURCE_RETURN_XY_TOLERANCE_MM:
            return False
        low, high = height - _MOCK_SOURCE_RETURN_Z_BELOW_MM, height + _MOCK_SOURCE_RETURN_Z_ABOVE_MM
        if self.enable_mock_move_release_upper_separation:
            # This guard is used only to prepare a checked MOVE. The actual
            # OPEN still checks both observed and sent heights independently.
            return low <= nominal_tcp[2] <= high and candidate_tcp[2] >= low
        return all(low <= point[2] <= high for point in (nominal_tcp, candidate_tcp))

    def _mock_bounded_nominal_retry(self, nominal, measured, applied, phase, task, slots, *, return_active):
        """Choose one checked correction; this does not transmit or allow extra retries."""
        correction = [max(-.5, min(.5, goal - observed)) for goal, observed in zip(nominal, measured)]
        # The release height and quantized path can admit a correction between
        # two coarser fractions; checking more candidates doesn't relax either.
        for fraction in (1., .75, .5, .375, .25, .125, 0.):
            proposed = [goal + fraction * offset for goal, offset in zip(nominal, correction)]
            steps, effective = quantize_joints(proposed)
            effective = list(effective)
            if (max(abs(a-b) for a, b in zip(effective, nominal)) > 1.0
                    or _close(effective, applied)
                    or (return_active and not self._mock_source_return_target_allowed(
                        nominal, effective, task, slots))):
                continue
            try:
                self._checked_joint_segment(measured, effective, phase, observed_start=True)
            except ControlError as exc:
                if exc.code not in {'ASYNC_ENVELOPE', 'VERTICAL_CORRIDOR',
                                    'PATH_OUTSIDE_WORKSPACE', 'TRANSFER_CLEARANCE'}:
                    raise
                continue
            return proposed, effective, steps, fraction
        raise _Abort('nominal_retry_unavailable', 'mock retry cannot produce a bounded checked target')

    @staticmethod
    def _mock_quiet_diagnostics(before, after):
        """Require present, unchanged mock event feedback; never infer missing values."""
        def diagnostics(observation):
            values = observation.get('device_diagnostics')
            if not isinstance(values, dict) or type(values.get('firmware_warning_raw')) is not int:
                raise _Abort('warning_unavailable', 'warning feedback is unavailable')
            if values['firmware_warning_raw'] != 0:
                raise _Abort('device_warning', 'warning blocks replanning')
            counts = values.get('collision_event_counts')
            if (not isinstance(counts, (tuple, list)) or len(counts) != 4
                    or any(type(item) is not int or item < 0 for item in counts)):
                raise _Abort('collision_feedback_unavailable', 'collision feedback is unavailable')
            return list(counts)
        before_counts, after_counts = diagnostics(before), diagnostics(after)
        if before_counts != after_counts:
            raise _Abort('device_collision_event', 'collision event changed')
        return after_counts

    def _reconcile_open_drift(self, before, after, task, slots, applied, *, runtime=False):
        """Check post-OPEN feedback as a path start, never as a new sent target."""
        after_counts = self._mock_quiet_diagnostics(before, after)
        if (not _number(before.get('sensor_sequence')) or not _number(after.get('sensor_sequence'))
                or after['sensor_sequence'] <= before['sensor_sequence']
                or after.get('sensor_valid') is not True or after.get('moving') is not False):
            raise _Abort('invalid_observation', 'mock OPEN requires fresh stationary joint feedback')
        start, end = _joints(before.get('joints_deg')), _joints(after.get('joints_deg'))
        config, calibration = self.controller.config, self.path_calibration
        if runtime and max(abs(a-b) for a,b in zip(start, end)) > (
                config.limits.joint_tolerance_deg + config.limits.observation_joint_drift_deg):
            raise _Abort('state_changed', 'post-OPEN change exceeds the bounded observation envelope')
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        start_xyz, end_xyz = geometry.forward(start), geometry.forward(end)
        centers = [slots[task[key] - 1] for key in ('source_slot', 'destination_slot')]
        center = min(centers, key=lambda xy: math.dist(xy, start_xyz[:2]) + math.dist(xy, end_xyz[:2]))
        if (math.dist(center, start_xyz[:2]) <= calibration.vertical_corridor_mm
                and math.dist(center, end_xyz[:2]) <= calibration.vertical_corridor_mm):
            phase = {'kind': 'vertical', 'vertical_xy_mm': center}
        elif min(start_xyz[2], end_xyz[2]) > calibration.transport_floor_mm:
            phase = {'kind': 'transport'}
        else:
            raise _Abort('open_drift_path_unavailable', 'mock OPEN drift has no checked path phase')
        evidence = self._checked_joint_segment(start, end, phase, observed_start=True)
        return {'kind': 'open_feedback_reconciliation' if runtime else 'mock_open_drift_reconciliation',
                'before_joints_deg': start, 'after_joints_deg': end,
                'last_sent_target_deg': list(applied), 'next_path_start_source': 'fresh_stationary_observation',
                'warning_raw': 0, 'collision_event_counts': after_counts, 'phase': phase, 'path': evidence,
                'continuous_contact_assessed': False, 'physical_success': None}

    _mock_reconcile_open_drift = _reconcile_open_drift

    def _mock_replan_source_descent(self, measured, last_sent, old_nominal, final_segment,
                                    phase, task, slots, *, remaining_capacity, deadline):
        """Preflight a bounded replacement path; no motion or object state is read."""
        if self._cancelled.is_set():
            raise _Abort('cancelled', 'cancelled before mock vertical replanning')
        config, calibration = self.controller.config, self.path_calibration
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        start_xyz, old_xyz = geometry.forward(measured), geometry.forward(old_nominal)
        final_requested = _joints(final_segment['requested'])
        final_effective = _joints(final_segment['effective'])
        final_xyz = geometry.forward(final_effective)
        source = slots[task['source_slot'] - 1]
        if (phase.get('kind') != 'vertical' or not final_xyz[2] < start_xyz[2] < old_xyz[2]
                or math.dist(start_xyz[:2], source) > calibration.vertical_corridor_mm
                or math.dist(final_xyz[:2], source) > calibration.vertical_corridor_mm
                or not self._mock_source_return_target_allowed(final_effective, final_effective, task, slots)):
            raise _Abort('vertical_replan_unavailable', 'stopped position is not on a checked source descent')
        last_error = None
        for count in (2, 4, 8, 16, 32, 64):
            if count > remaining_capacity:
                continue
            previous = list(measured)
            replacements = []
            try:
                for index in range(1, count + 1):
                    if self._cancelled.is_set():
                        raise _Abort('cancelled', 'cancelled while checking mock vertical replanning')
                    if time.monotonic() >= deadline:
                        raise _Abort('vertical_replan_timeout', 'mock vertical replanning exceeded its time budget')
                    xyz = tuple(a + (b-a)*index/count for a, b in zip(start_xyz, final_xyz))
                    requested = final_requested if index == count else _joints(geometry.inverse(xyz))
                    _, effective = quantize_joints(requested)
                    effective = list(effective)
                    if not self._mock_source_return_target_allowed(effective, effective, task, slots):
                        raise _Abort('vertical_replan_unavailable', 'replanned source target is outside release bounds')
                    path = self._checked_joint_segment(previous, effective, phase, observed_start=index == 1)
                    replacements.append({'requested': list(requested), 'effective': effective,
                                         'path': path, 'mock_replanned_no_retry': True})
                    previous = effective
                break
            except ControlError as exc:
                if exc.code not in {'ASYNC_ENVELOPE', 'VERTICAL_CORRIDOR'}:
                    raise
                last_error = exc
        else:
            raise _Abort('vertical_replan_unavailable', 'no bounded checked vertical replanning fits') from last_error
        nominal = replacements[0]['effective']
        proposal, fallback = validated_bias_target(nominal, last_sent, measured,
            lambda start, end: self._checked_joint_segment(start, end, phase, observed_start=True))
        steps, sent = quantize_joints(proposal)
        sent = list(sent)
        if (_close(sent, last_sent)
                or not self._mock_source_return_target_allowed(nominal, sent, task, slots)):
            raise _Abort('vertical_replan_unavailable', 'no bounded replacement target is available')
        return replacements, list(proposal), sent, steps, fallback

    def _virtual_descent_comparison(self, before, task, slot_xy_mm, start, target, gripper,
                                    *, start_source='last_confirmed_applied_target'):
        """Describe the virtual path without sending it to the mock controller.

        The virtual app and this comparison share ``straight_descent.prepare``.
        The start here is the controller's last confirmed target;
        the end is the guarded, unquantized model proposal. Neither the mock
        timing nor the controller's actual path is predicted by this result.
        """
        comparison = {'purpose': 'comparison_only', 'applied_to_controller': False,
                      'start_source': start_source,
                      'end_source': 'planned_unquantized_target_after_guard',
                      'virtual_profile': 'two_seconds_1000_ticks_smoothstep',
                      'start_target_deg': list(start), 'end_target_deg': list(target)}
        if gripper != 'closed':
            comparison.update(status='not_applicable', reason='gripper_not_closed')
            return None, comparison
        destination = task['destination_slot'] - 1
        path, info = straight_descent.prepare(start, target, slot_xy_mm[destination],
                                              bool(before[destination]), self._cancelled)
        comparison.update(status='prepared' if path is not None else 'unchanged_joint_path',
                          path_info=info)
        if path is not None:
            ticks = (1, 250, 500, 750, straight_descent.TICKS)
            comparison['samples'] = [{'tick': tick, 'joints_deg': [math.degrees(value) for value in
                                                                     path.targets_rad[tick - 1]]}
                                     for tick in ticks]
        return path, comparison

    def _geometry(self):
        """Return arm TCP geometry using the active controller dimensions."""
        config = self.controller.config
        return Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)

    def _within_target_tolerances(self, observed, target, *, allowed_tcp_mm=None):
        """Return whether observed joints satisfy both joint and TCP tolerances."""
        limits = self.controller.config.limits
        tcp_limit = limits.tcp_tolerance_mm if allowed_tcp_mm is None else allowed_tcp_mm
        geometry = self._geometry()
        return (_close(observed, target, limits.joint_tolerance_deg)
                and math.dist(geometry.forward(observed), geometry.forward(target)) <= tcp_limit)

    def _target_reached_flag(self, response_value, observed, target, *, allowed_tcp_mm=None):
        """Use the controller arrival flag when reported, else check joint and TCP bounds."""
        if response_value is not None:
            return response_value
        return self._within_target_tolerances(observed, target, allowed_tcp_mm=allowed_tcp_mm)

    def _has_pending_retarget_locked(self, *, return_active=False, include_pregrasp=False):
        """Check under _state_lock if a source-return or retarget admission is pending."""
        if not self._source_return_enabled or return_active or self._source_return_state is None:
            return False
        state = self._source_return_state
        return bool(state['pending'] or state['admission_waiting']
                    or (include_pregrasp and state['pregrasp_cancel_requested']))

    def _dispatch_guarded_action(self, request, *, command_id, return_active=False, include_pregrasp=False,
                                 track_active=True, on_before_pending=None, on_pending=None, on_before_send=None):
        """Dispatch a controller action serialized with retarget admission ownership."""
        with self._mock_move_dispatch_lock:
            pending_return = False
            if self._source_return_enabled and (not return_active or not track_active):
                with self._state_lock:
                    pending_return = self._has_pending_retarget_locked(
                        return_active=return_active if track_active else False,
                        include_pregrasp=include_pregrasp)
                    if track_active and not pending_return and not return_active:
                        self._source_return_state['active_command_id'] = command_id
                        self._mock_move_dispatch_owner = threading.get_ident()
            if on_before_pending is not None:
                on_before_pending()
            if pending_return:
                if on_pending is not None:
                    on_pending()
                return True, None
            try:
                if on_before_send is not None:
                    on_before_send()
                return False, self.controller.apply_action(request)
            finally:
                if track_active and self._source_return_enabled and not return_active:
                    with self._state_lock:
                        self._source_return_state['active_command_id'] = None
                        self._mock_move_dispatch_owner = None

    def _preempt_retained_before_send(self, lease, command_id, *, instance, seen, last_time, last_sequence,
                                      gripper, label):
        """Interrupt a retained SDK move before dispatching the next command and re-arm."""
        interruption = self.controller.interrupt_motion(lease, command_id)
        stop = interruption.get('stop') if isinstance(interruption, dict) else None
        if (not isinstance(interruption, dict)
                or interruption.get('instance_id') != instance
                or interruption.get('command_id') != command_id
                or interruption.get('superseded_by_stop') is not False
                or not isinstance(interruption.get('interruption_id'), str)
                or not interruption['interruption_id']
                or not isinstance(stop, dict) or stop.get('requested') is not True
                or stop.get('confirmed') is not True):
            raise _Abort('stop_unconfirmed', f'retarget stop before {label} is unconfirmed')
        stopped = self.controller.observe(lease)
        instance, last_time, last_sequence = self._observation(
            stopped, instance, seen, last_time, last_sequence,
            require_new_sample=True, expected_state='stopped')
        self._gripper(stopped, gripper)
        if (stopped['sensor_sequence'] < stop.get('confirmed_sensor_sequence', float('inf'))
                or stopped['observed_monotonic'] < stop.get('confirmed_sensor_monotonic', float('inf'))):
            raise _Abort('stop_unconfirmed', 'retarget feedback predates the stop')
        measured = _joints(stopped['joints_deg'])
        drift = self.controller.config.limits.observation_joint_drift_deg
        resumed = self.controller.resume_interrupted_motion(
            lease, interruption['interruption_id'], stopped['observation_id'])
        if (not isinstance(resumed, dict) or resumed.get('resumed') is not True
                or not _close(_joints(resumed.get('joints_deg')), measured, drift)):
            raise _Abort('stop_unconfirmed', 'retarget did not re-arm from the stopped observation')
        observation = self.controller.observe(lease)
        instance, last_time, last_sequence = self._observation(
            observation, instance, seen, last_time, last_sequence)
        self._gripper(observation, gripper)
        expected_joints = _joints(observation['joints_deg'])
        if not _close(expected_joints, measured, drift):
            raise _Abort('state_changed', 'arm changed after the retarget stop')
        return stop, measured, observation, expected_joints, instance, last_time, last_sequence

    def _observe_interrupted_stop(self, lease, response, requested, effective, steps,
                                  *, instance, seen, last_time, last_sequence, gripper, expected_joints,
                                  pending_cancel, subject, cancel_stage):
        """Validate an interrupted command reply and fetch fresh stopped joint feedback."""
        if self._cancelled.is_set() or (pending_cancel is not None and pending_cancel.is_set()):
            raise _Abort('cancelled', f'{cancel_stage} was cancelled')
        if response.get('status') != 'cancelled' or type(response.get('applied')) is not bool:
            raise _Abort('command_unconfirmed', f'{subject} outcome is unknown')
        sent = response['applied'] is True
        if sent:
            self._target_response(response, requested, effective, steps)
        stopped = self.controller.observe(lease)
        instance, last_time, last_sequence = self._observation(
            stopped, instance, seen, last_time, last_sequence,
            require_new_sample=True, expected_state='stopped')
        self._gripper(stopped, gripper)
        stopped_joints = _joints(stopped['joints_deg'])
        drift = self.controller.config.limits.observation_joint_drift_deg
        if not sent and not _close(stopped_joints, expected_joints, drift):
            raise _Abort('state_changed', f'joints changed before the {subject} was sent')
        return sent, stopped, stopped_joints, instance, last_time, last_sequence

    def _rearm_stopped_controller(self, lease, interruption, stopped, stopped_joints,
                                  *, instance, seen, last_time, last_sequence, gripper,
                                  pending_cancel, cancel_stage, rearm_error, post_drift_error,
                                  force_mock=False):
        """Resume from a validated stop and verify stationary post-resume feedback."""
        drift = self.controller.config.limits.observation_joint_drift_deg
        resume = (self.controller.resume_interrupted_motion
                  if self.enable_runtime_source_return and not force_mock
                  else self.controller.resume_mock_interrupted_motion)
        resumed = resume(lease, interruption['interruption_id'], stopped['observation_id'])
        if (not isinstance(resumed, dict) or resumed.get('resumed') is not True
                or not _close(_joints(resumed.get('joints_deg')), stopped_joints, drift)):
            raise _Abort('stop_unconfirmed', rearm_error)
        if self._cancelled.is_set() or (pending_cancel is not None and pending_cancel.is_set()):
            raise _Abort('cancelled', f'{cancel_stage} was cancelled before replanning')
        observation = self.controller.observe(lease)
        instance, last_time, last_sequence = self._observation(
            observation, instance, seen, last_time, last_sequence)
        self._gripper(observation, gripper)
        rearmed_joints = _joints(observation['joints_deg'])
        if not _close(rearmed_joints, stopped_joints, drift):
            raise _Abort('state_changed', post_drift_error)
        return observation, rearmed_joints, instance, last_time, last_sequence

    def cancel(self):
        """Request an immediate stop without taking ownership of the controller."""
        self._cancelled.set()
        with self._state_lock:
            if self._lease is None:
                return {'requested': False, 'reason': 'no_active_lease'}
            try:
                stop = self.controller.stop('STUDENT_TRANSFER_CANCELLED')
                if not isinstance(stop, dict) or stop.get('confirmed') is not True:
                    self._stop_error = 'stop_unconfirmed'
                return stop
            except Exception as exc:
                self._stop_error = type(exc).__name__
                return {'requested': True, 'confirmed': False, 'error': self._stop_error}

    def _observation(self, observation, instance, seen, last_time, last_sequence, *, require_new_sample=False,
                     expected_state='armed'):
        if not isinstance(observation, dict):
            raise _Abort('invalid_observation', 'controller observation is missing')
        identifier = observation.get('observation_id')
        observed = observation.get('observed_monotonic')
        age = observation.get('sensor_age_seconds')
        now = time.monotonic()
        if (not isinstance(identifier, str) or not identifier.strip() or identifier in seen
                or not _number(observed) or observed < 0 or observed < last_time
                or observed > now or now - observed > self.controller.config.limits.observation_max_age
                or not _number(age) or age < 0
                or age + (now-observed) > self.controller.config.limits.sensor_max_age
                or type(observation.get('sensor_sequence')) is not int
                or observation['sensor_sequence'] < 1
                or observation['sensor_sequence'] < last_sequence
                or (require_new_sample and observation['sensor_sequence'] <= last_sequence)
                or observation.get('schema_version') != 'PolicyObservation.phase1'
                or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                or observation.get('state') != expected_state):
            raise _Abort('invalid_observation', 'a new, fresh, stationary observation is required')
        identity = observation.get('instance_id')
        if (not isinstance(identity, str) or not identity.strip()
                or (instance is not None and identity != instance)
                or (getattr(self.controller, 'instance_id', None) is not None
                    and identity != self.controller.instance_id)):
            raise _Abort('controller_changed', 'the observation has a different controller identity')
        diagnostics = observation.get('device_diagnostics')
        warning = diagnostics.get('firmware_warning_raw') if isinstance(diagnostics, dict) else None
        simulated_without_warning = (isinstance(diagnostics, dict) and warning is None
                                     and self.controller.config.backend == 'mock'
                                     and observation.get('simulated') is True)
        if not simulated_without_warning and (type(warning) is not int or warning not in (0, 1)):
            raise _Abort('warning_unavailable', 'device warning feedback is unavailable')
        if warning == 1:
            raise _Abort('device_warning', 'the controller reports a device warning')
        _joints(observation.get('joints_deg'))
        seen.add(identifier)
        return identity, float(observed), observation['sensor_sequence']

    @staticmethod
    def _gripper(observation, expected):
        if (observation.get('gripper_command') != expected
                or observation.get('gripper_reported_state') != expected
                or observation.get('gripper_reported_valid') is not True
                or observation.get('gripper_feedback_fresh') is not True):
            raise _Abort('gripper_unconfirmed', 'fresh gripper state does not match the last confirmed command')

    def _within_stationary_arrival_tolerances(self, joints, stationary):
        limits = self.controller.config.limits
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        return (_close(joints, stationary['joints_deg'], limits.joint_tolerance_deg)
                and math.dist(geometry.forward(joints), geometry.forward(stationary['joints_deg']))
                    <= limits.tcp_tolerance_mm)

    def _end_motion_policy_observation(self, attempts, stationary):
        """Read the last measured motion from this policy MOVE and its same-target retry."""
        for attempt in reversed(attempts):
            completion = attempt.get('completion') or {}
            sample = completion.get('end_motion_observation')
            if not isinstance(sample, dict):
                continue
            frame = completion.get('host_frame') or {}
            if (sample.get('schema_version') != 'PolicyObservation.phase1'
                    or sample.get('instance_id') != stationary['instance_id']
                    or sample.get('sensor_valid') is not True or sample.get('moving') is not True
                    or sample.get('gripper_command') != 'closed' or sample.get('gripper_reported_state') != 'closed'
                    or sample.get('gripper_reported_valid') is not True or sample.get('gripper_feedback_fresh') is not True
                    or type(sample.get('sensor_sequence')) is not int
                    or type(completion.get('sensor_sequence')) is not int
                    or not 0 < sample['sensor_sequence'] < completion['sensor_sequence'] <= stationary['sensor_sequence']
                    or not all(_number(x) for x in (frame.get('monotonic'), sample.get('sensor_monotonic'),
                                                     completion.get('sensor_monotonic')))
                    or not frame['monotonic'] < sample['sensor_monotonic'] <= completion['sensor_monotonic']):
                return None, 'motion_sample_unconfirmed'
            joints = _joints(sample['joints_deg'])
            if not self._within_stationary_arrival_tolerances(joints, stationary):
                return None, 'motion_sample_differs_from_stationary_pose'
            geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                                tool_vertical_mm=self.controller.config.tool_vertical_mm)
            tcp = geometry.forward(joints)
            return {**sample, 'source_command_id': attempt['command_id'], 'tcp_mm': list(tcp)}, 'measured_end_motion'
        return None, 'no_eligible_loaded_motion_feedback'

    def _is_unstack_destination_descent(self, start_joints, end_joints, task, slots):
        if self.path_calibration is None or task['source_level'] <= task['destination_level']:
            return False
        config = self.controller.config
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        start, end = geometry.forward(start_joints), geometry.forward(end_joints)
        destination = slots[task['destination_slot'] - 1]
        corridor = self.path_calibration.vertical_corridor_mm
        return (start[2] > end[2] and math.dist(start[:2], destination) <= corridor
                and math.dist(end[:2], destination) <= corridor)

    def _reconsideration_superseded(self):
        # Only used in the original task, before any redirect has taken over.
        with self._state_lock:
            if self._cancelled.is_set():
                raise _Abort('cancelled', 'cancelled during stationary reconsideration')
            return self._source_return_enabled and (
                self._source_return_state['pending'] or self._source_return_state['admission_waiting'])

    def _current_motion_proposal(self, angles, previous):
        # Order a completed retarget admission and this original-task check.
        # No SDK I/O or model inference is performed while holding the lock.
        with self._state_lock:
            if (self._cancelled.is_set() or (self._source_return_enabled and
                    (self._source_return_state['pending'] or self._source_return_state['admission_waiting']))):
                return None
            proposed = _joints(native_command_target('move', angles, previous))
            self._require_original_student_endpoint(proposed)
            return proposed

    def _require_original_student_endpoint(self, proposed):
        event = self.model.events[-1] if self.model.events else {}
        original = event.get('original_proposal_joints_deg')
        if event.get('clamped') is True or not isinstance(original, list) or not _close(proposed, original):
            raise _Abort('student_endpoint_changed', 'the physical MOVE cannot replace the student endpoint')

    @staticmethod
    def _target_response(response, requested, effective, steps):
        target = response.get('target')
        if not isinstance(target, dict):
            raise _Abort('target_unconfirmed', 'controller omitted the prepared target')
        try:
            requested_reply = _joints(target.get('requested_joints_deg'))
            effective_reply = _joints(target.get('effective_target_deg'))
            reply_steps = target.get('steps')
            if (not isinstance(reply_steps, (tuple, list)) or len(reply_steps) != 4
                    or any(type(v) is not int for v in reply_steps)
                    or list(reply_steps) != list(steps)
                    or not _close(requested_reply, requested)
                    or not _close(effective_reply, effective)):
                raise ValueError('target mismatch')
        except (ValueError, _Abort):
            raise _Abort('target_unconfirmed', 'controller target differs from the requested quantized target') from None

    @staticmethod
    def _retained_move_response(response, steps, speed, before_sequence, observation):
        """Confirm an encoded target and a later sample, without asserting arrival."""
        completion, stop = response.get('completion'), response.get('stop')
        frame = completion.get('host_frame') if isinstance(completion, dict) else None
        latest = completion.get('latest_host_frame') if isinstance(completion, dict) else None
        sensor_time = observation['observed_monotonic'] - observation['sensor_age_seconds']
        if (not isinstance(completion, dict) or completion.get('mode') != 'position_target_retained'
                or type(response.get('nominal_arrival_reached')) is not bool
                or (response.get('status') == 'completed' and response.get('nominal_arrival_reached') is not True)
                or ('sent_target_reached' in response and type(response['sent_target_reached']) is not bool)
                or not isinstance(stop, dict) or stop.get('requested') is not False
                or stop.get('confirmed') is not False
                or not isinstance(completion.get('target_steps'), (list, tuple))
                or any(type(value) is not int for value in completion['target_steps'])
                or list(completion['target_steps']) != list(steps)
                or type(completion.get('sensor_sequence')) is not int
                or not before_sequence < completion['sensor_sequence'] <= observation['sensor_sequence']
                or not _number(completion.get('sensor_monotonic'))
                or completion['sensor_monotonic'] > sensor_time + 1e-6
                or not isinstance(frame, dict) or type(frame.get('frame_sequence')) is not int
                or frame['frame_sequence'] < 1 or type(frame.get('command_sequence')) is not int
                or not 0 <= frame['command_sequence'] <= 255 or frame.get('mode_hex') != '11'
                or type(frame.get('target_segment_start_sequence')) is not int
                or not 1 <= frame['target_segment_start_sequence'] <= frame['frame_sequence']
                or not isinstance(frame.get('joint_steps'), (list, tuple))
                or any(type(value) is not int for value in frame['joint_steps'])
                or list(frame['joint_steps']) != list(steps)
                or frame.get('speed_pct') != int(.5 + speed)
                or not _number(frame.get('monotonic'))
                or frame['monotonic'] > completion['sensor_monotonic']
                or not isinstance(latest, dict) or type(latest.get('frame_sequence')) is not int
                or latest['frame_sequence'] < frame['frame_sequence']
                or latest.get('target_segment_start_sequence') != frame['target_segment_start_sequence']
                or latest.get('command_sequence') != frame['command_sequence'] or latest.get('mode_hex') != '11'
                or latest.get('joint_steps') != frame['joint_steps'] or latest.get('speed_pct') != frame['speed_pct']
                or not _number(latest.get('monotonic')) or latest['monotonic'] < frame['monotonic']):
            raise _Abort('target_unconfirmed', 'retained position or its arrival evidence is unconfirmed')
        return {'command_id': response['command_id'], 'target_steps': list(frame['joint_steps']),
                'frame_sequence': latest['frame_sequence'], 'frame_monotonic': latest['monotonic'],
                'target_segment_start_sequence': frame['target_segment_start_sequence'],
                'sensor_sequence': completion['sensor_sequence'],
                'sensor_monotonic': completion['sensor_monotonic']}

    @staticmethod
    def _retained_gripper_response(response, retained, before_sequence, observation):
        """Bind a fresh gripper result to the last retained host position goal."""
        hold = response.get('arm_hold')
        sensor_time = observation['observed_monotonic'] - observation['sensor_age_seconds']
        if (not isinstance(hold, dict) or hold.get('motion_command_id') != retained['command_id']
                or not isinstance(hold.get('target_steps'), (list, tuple))
                or any(type(value) is not int for value in hold['target_steps'])
                or list(hold['target_steps']) != retained['target_steps']
                or hold.get('target_segment_start_sequence') != retained['target_segment_start_sequence']
                or type(hold.get('frame_sequence')) is not int
                or hold['frame_sequence'] <= retained['frame_sequence']
                or not _number(hold.get('frame_monotonic'))
                or hold['frame_monotonic'] < retained['frame_monotonic']
                or type(hold.get('sensor_sequence')) is not int
                or not max(before_sequence, retained['sensor_sequence']) < hold['sensor_sequence'] <= observation['sensor_sequence']
                or not _number(hold.get('sensor_monotonic'))
                or hold['sensor_monotonic'] <= hold['frame_monotonic']
                or hold['sensor_monotonic'] < retained['sensor_monotonic']
                or hold['sensor_monotonic'] > sensor_time + 1e-6):
            raise _Abort('target_unconfirmed', 'gripper action did not confirm the retained arm target')
        return {**retained, 'frame_sequence': hold['frame_sequence'], 'frame_monotonic': hold['frame_monotonic'],
                'sensor_sequence': hold['sensor_sequence'], 'sensor_monotonic': hold['sensor_monotonic']}

    @staticmethod
    def _retained_noop_response(response, retained, steps, before_sequence, observation):
        """Advance a no-transmission policy step only on fresh hold evidence."""
        sample = response.get('retained_position_observation')
        sensor_time = observation['observed_monotonic'] - observation['sensor_age_seconds']
        if (retained is None or response.get('retained_position_from_command_id') != retained['command_id']
                or list(steps) != retained['target_steps']
                or type(response.get('nominal_arrival_reached')) is not bool
                or ('sent_target_reached' in response and type(response['sent_target_reached']) is not bool)
                or not isinstance(sample, dict) or type(sample.get('sensor_sequence')) is not int
                or not max(before_sequence, retained['sensor_sequence']) < sample['sensor_sequence'] <= observation['sensor_sequence']
                or not _number(sample.get('sensor_monotonic'))
                or sample['sensor_monotonic'] < retained['sensor_monotonic']
                or sample['sensor_monotonic'] > sensor_time + 1e-6):
            raise _Abort('target_unconfirmed', 'the existing retained target was not freshly confirmed')
        return {**retained, 'sensor_sequence': sample['sensor_sequence'],
                'sensor_monotonic': sample['sensor_monotonic']}

    def _damped_correction_basis(self, response, correction, parent_id, nominal, measured, observation, speed,
                                 before_sequence, constraints):
        """Bind one damped correction to the retained original SDK target."""
        basis, stop, parent = response.get('correction_basis'), response.get('stop'), response.get('retained_parent')
        if not isinstance(basis, dict) or not isinstance(stop, dict) or not isinstance(parent, dict):
            raise _Abort('target_unconfirmed', 'damped compensation evidence is missing')
        nominal_steps, _ = quantize_joints(nominal)
        requested = _joints(correction.get('joints_deg'))
        compensated_steps, compensated = quantize_joints(requested)
        basis_joints = _joints(basis.get('joints_deg'))
        expected = [target + .5 * (target - actual) for target, actual in zip(nominal, basis_joints)]
        frame = basis.get('original_host_frame')
        latest = parent.get('latest_host_frame')
        sample_time = observation['observed_monotonic'] - observation['sensor_age_seconds']
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        joint_error = max(abs(a-b) for a, b in zip(nominal, basis_joints))
        tcp_error = math.dist(geometry.forward(nominal), geometry.forward(basis_joints))
        if (basis.get('kind') != 'damped_nominal_half' or basis.get('gain') != .5
                or basis.get('source') != 'retained_position_encoder_feedback'
                or basis.get('parent_command_id') != parent_id or stop.get('requested') is not False
                or stop.get('confirmed') is not False
                or type(basis.get('sensor_sequence')) is not int
                or not before_sequence < basis['sensor_sequence'] <= observation['sensor_sequence']
                or not _number(basis.get('sensor_monotonic'))
                or basis['sensor_monotonic'] > sample_time + 1e-6
                or not _close(basis_joints, measured, self.controller.config.limits.joint_tolerance_deg)
                or not _close(_joints(basis.get('nominal_target_deg')), nominal)
                or basis.get('nominal_steps') != list(nominal_steps)
                or not _close(requested, expected)
                or not _close(_joints(basis.get('unquantized_target_deg')), requested)
                or not _close(_joints(basis.get('compensated_target_deg')), compensated)
                or basis.get('compensated_steps') != list(compensated_steps)
                or list(compensated_steps) == list(nominal_steps)
                or max(abs(a-b) for a, b in zip(requested, nominal)) > 1.
                or max(abs(a-b) for a, b in zip(compensated, nominal)) > 1.
                or basis.get('minimum_z_mm') != constraints.get('minimum_z_mm')
                or not _number(basis.get('baseline_joint_error_deg'))
                or abs(basis['baseline_joint_error_deg'] - joint_error) > 1e-7
                or not _number(basis.get('baseline_tcp_error_mm'))
                or abs(basis['baseline_tcp_error_mm'] - tcp_error) > 1e-6
                or not isinstance(frame, dict) or frame.get('mode_hex') != '11'
                or frame.get('joint_steps') != list(nominal_steps)
                or frame.get('speed_pct') != int(.5 + speed)
                or type(frame.get('command_sequence')) is not int
                or type(frame.get('frame_sequence')) is not int or frame['frame_sequence'] < 1
                or type(frame.get('target_segment_start_sequence')) is not int
                or not 1 <= frame['target_segment_start_sequence'] <= frame['frame_sequence']
                or not _number(frame.get('monotonic')) or frame['monotonic'] > basis['sensor_monotonic']
                or parent.get('mode') != 'position_target_retained' or parent.get('command_id') != parent_id
                or type(parent.get('stop_epoch')) is not int or parent['stop_epoch'] < 0
                or basis.get('stop_epoch') != parent['stop_epoch']
                or parent.get('target_steps') != list(nominal_steps) or parent.get('host_frame') != frame
                or parent.get('sensor_sequence') != basis['sensor_sequence']
                or parent.get('sensor_monotonic') != basis['sensor_monotonic']
                or not isinstance(latest, dict) or type(latest.get('frame_sequence')) is not int
                or latest['frame_sequence'] < frame['frame_sequence']
                or latest.get('target_segment_start_sequence') != frame['target_segment_start_sequence']
                or basis.get('latest_host_frame') != latest
                or latest.get('command_sequence') != frame['command_sequence'] or latest.get('mode_hex') != '11'
                or latest.get('joint_steps') != frame['joint_steps'] or latest.get('speed_pct') != frame['speed_pct']
                or not _number(latest.get('monotonic')) or latest['monotonic'] < frame['monotonic']):
            raise _Abort('target_unconfirmed', 'damped compensation does not match the encoded nominal goal')
        return copy.deepcopy(basis)

    def _damped_compensation_completed(self, response, basis, nominal, sent, observation, tolerance_mm):
        """Confirm nominal arrival while retaining the distinct encoded setpoint."""
        proof, target = response.get('damped_compensation'), response.get('target')
        completion = response.get('completion')
        frame = completion.get('host_frame') if isinstance(completion, dict) else None
        nominal_target = target.get('nominal_arrival') if isinstance(target, dict) else None
        nominal_steps, _ = quantize_joints(nominal)
        sent_steps, _ = quantize_joints(sent)
        fields = ('kind', 'gain', 'parent_command_id', 'stop_epoch', 'sensor_sequence', 'sensor_monotonic', 'joints_deg', 'source',
                  'nominal_target_deg', 'nominal_steps', 'unquantized_target_deg', 'compensated_target_deg',
                  'compensated_steps', 'original_host_frame', 'latest_host_frame', 'baseline_joint_error_deg',
                  'baseline_tcp_error_mm', 'minimum_z_mm')
        if (not isinstance(proof, dict) or any(proof.get(key) != basis.get(key) for key in fields)
                or ('after_gripper_command_id' in basis
                    and proof.get('after_gripper_command_id') != basis['after_gripper_command_id'])
                or not isinstance(nominal_target, dict)
                or not isinstance(nominal_target.get('steps'), (list, tuple))
                or list(nominal_target['steps']) != list(nominal_steps)
                or not isinstance(target.get('steps'), (list, tuple)) or list(target['steps']) != list(sent_steps)
                or not isinstance(frame, dict) or type(frame.get('frame_sequence')) is not int
                or frame['frame_sequence'] <= basis['latest_host_frame']['frame_sequence']
                or type(frame.get('target_segment_start_sequence')) is not int
                or frame['target_segment_start_sequence'] <= basis['latest_host_frame']['frame_sequence']
                or not _number(frame.get('monotonic')) or frame['monotonic'] <= basis['sensor_monotonic']
                or type(completion.get('sensor_sequence')) is not int
                or completion['sensor_sequence'] <= basis['sensor_sequence']
                or response.get('nominal_arrival_reached') is not True
                or type(response.get('sent_target_reached')) is not bool
                or proof.get('compensated_target_reached') is not response['sent_target_reached']
                or not _number(proof.get('nominal_joint_error_after_deg'))
                or not _number(proof.get('nominal_tcp_error_after_mm'))
                or proof['nominal_joint_error_after_deg'] > basis['baseline_joint_error_deg'] + 1e-7
                or proof['nominal_tcp_error_after_mm'] > basis['baseline_tcp_error_mm'] + 1e-6):
            raise _Abort('target_unconfirmed', 'completed damped compensation evidence differs')
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        measured = _joints(observation['joints_deg'])
        if (max(abs(a-b) for a, b in zip(measured, nominal)) > self.controller.config.limits.joint_tolerance_deg
                or math.dist(geometry.forward(measured), geometry.forward(nominal)) > tolerance_mm
                or max(abs(a-b) for a, b in zip(measured, nominal)) > basis['baseline_joint_error_deg'] + 1e-7
                or math.dist(geometry.forward(measured), geometry.forward(nominal)) > basis['baseline_tcp_error_mm'] + 1e-6):
            raise _Abort('target_unconfirmed', 'the original target was not reached after compensation')
        return copy.deepcopy(proof)

    @staticmethod
    def _stationary_blend_as_single(command, observation):
        """Keep a zero-length SDK path as one explicit position request."""
        if command.get('kind') != 'blended_joint_path':
            return command
        observed_steps, _ = quantize_joints(_joints(observation['joints_deg']))
        waypoints = command['waypoints_deg']
        if not waypoints or any(quantize_joints(_joints(point))[0] != observed_steps for point in waypoints):
            return command
        result = {key: copy.deepcopy(command[key]) for key in
                  ('speed_pct', 'path_constraints', 'nominal_arrival_joints_deg', 'allow_nominal_correction')
                  if key in command}
        result.update(kind='move_joints', joints_deg=list(waypoints[-1]))
        return result

    @classmethod
    def _mock_nominal_response(cls, response, requested, effective, steps, nominal):
        cls._target_response(response, requested, effective, steps)
        target = response.get('target')
        reference = target.get('nominal_arrival') if isinstance(target, dict) else None
        if (not isinstance(reference, dict)
                or not _close(_joints(reference.get('effective_target_deg')), nominal)
                or response.get('arrival_reference') != 'nominal'):
            raise _Abort('target_unconfirmed', 'controller nominal target differs')
        if response.get('applied') is True:
            receipt = response.get('receipt')
            if (not isinstance(receipt, dict) or not isinstance(receipt.get('steps'), (tuple, list))
                    or any(type(value) is not int for value in receipt['steps'])
                    or list(receipt['steps']) != list(steps)
                    or not _close(_joints(receipt.get('effective_target_deg')), effective)):
                raise _Abort('target_unconfirmed', 'worker receipt does not match the sent target')

    @staticmethod
    def _blended_response(response, final_target):
        trace, receipt = response.get('blended_path'), response.get('receipt')
        if not isinstance(trace, dict) or not isinstance(receipt, dict):
            raise _Abort('blended_path_unconfirmed', 'continuous path application evidence is missing')
        if trace.get('scope') == 'sdk_host_stream_unverified':
            counts = [trace.get(key) for key in ('sdk_planned_count', 'sdk_submitted_count', 'sdk_encoded_count')]
            digests = [trace.get(key) for key in ('sdk_planned_sha256', 'sdk_submitted_sha256', 'sdk_encoded_sha256')]
            if (any(type(value) is not int for value in counts) or counts[0] <= 0
                    or not 0 <= counts[2] <= counts[1] <= counts[0]
                    or trace.get('intermediate_stop_count') != 0
                    or any(key not in trace or trace[key] is not None
                           for key in ('applied_count', 'applied_sha256', 'last_applied_joints_deg'))
                    or any(not isinstance(value, str) or len(value) != 64
                           or any(char not in '0123456789abcdef' for char in value) for value in digests)):
                raise _Abort('blended_path_unconfirmed', 'SDK host evidence or unknown physical application differs')
            for count, key in zip(counts[1:], ('last_sdk_submitted_target_deg', 'last_sdk_encoded_target_deg')):
                if count:
                    _joints(trace.get(key))
                elif trace.get(key) is not None:
                    raise _Abort('blended_path_unconfirmed', 'SDK empty prefix has an encoded target')
            required = response.get('status') in ('completed', 'settled_needs_nominal_correction',
                                                   'settled_needs_target_follow_up', 'settled')
            if required or trace.get('completed') is True:
                if (trace.get('completed') is not True or len(set(counts)) != 1
                        or len(set(digests)) != 1
                        or not _close(_joints(trace.get('last_sdk_encoded_target_deg')), final_target)
                        or not _close(_joints(trace.get('last_sdk_submitted_target_deg')), final_target)):
                    raise _Abort('blended_path_unconfirmed', 'SDK did not encode its complete scheduled sequence')
            elif response.get('status') == 'cancelled' and trace.get('completed') is not False:
                raise _Abort('blended_path_unconfirmed', 'SDK prefix completion is inconsistent')
            return copy.deepcopy(trace)
        scheduled, applied = trace.get('scheduled_count'), trace.get('applied_count')
        if (type(scheduled) is not int or type(applied) is not int
                or not 0 <= applied <= scheduled or scheduled <= 0
                or trace.get('intermediate_stop_count') != 0):
            raise _Abort('blended_path_unconfirmed', 'continuous path sample counts or stops differ')
        status = response.get('status')
        completed_required = status in ('completed', 'settled_needs_nominal_correction',
                                        'settled_needs_target_follow_up', 'settled')
        if completed_required and trace.get('completed') is not True:
            raise _Abort('blended_path_unconfirmed', 'continuous path was not completed')
        if trace.get('completed') is True:
            digest = trace.get('planned_sha256')
            if (trace.get('completed') is not True or applied != scheduled
                    or not isinstance(digest, str) or len(digest) != 64
                    or trace.get('applied_sha256') != digest
                    or not _close(_joints(trace.get('last_applied_joints_deg')), final_target)):
                raise _Abort('blended_path_unconfirmed', 'continuous path did not apply its exact full sequence')
        elif status == 'cancelled' and trace.get('completed') is not False:
            raise _Abort('blended_path_unconfirmed', 'interrupted path completion is inconsistent')
        return copy.deepcopy(trace)

    def _checked_joint_segment(self, start, target, phase, *, observed_start):
        """Apply the same workspace/envelope checks as controller admission.

        Planned starts after the first command may differ from the confirmed
        target by the controller's arrival tolerance; account for that before
        any part of a subdivided decision is sent. The controller still checks
        each command against its own current sample at admission.
        """
        if self._cancelled.is_set():
            raise _Abort('cancelled', 'cancelled while checking the path')
        return TransferCompiler(self.controller.config, self.path_calibration).check_segment(
            start, target, transport=phase['kind'] == 'transport',
            vertical_xy=phase.get('vertical_xy_mm'), observed_start=observed_start)

    def _mock_lower_before_close(self, observation, task, slots, *, lease, applied, confirmed_target,
                                 reference_source, instance, seen, last_time, last_sequence, progress):
        """Execute a separately logged mock-only manual approach before CLOSE."""
        config, calibration = self.controller.config, self.path_calibration
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        depth = 6. if task['source_level'] == 1 else 4.
        proposed, xyz = lowered_grasp_target(geometry, observation['joints_deg'],
            slots[task['source_slot'] - 1], calibration.vertical_corridor_mm, depth)
        _, effective = quantize_joints(proposed)
        segments, phase, split_reason = self._checked_move_segments(observation, proposed, effective, task)
        if phase.get('kind') != 'vertical':
            raise _Abort('pregrasp_path_invalid', 'mock pregrasp lowering must remain vertical')
        manual = {'kind': 'mock_layered_preclose_lowering', 'student_action': 'close',
                  'depth_mm': depth, 'nominal_goal_mm': list(xyz), 'split_reason': split_reason,
                  'segments': [], 'physical_success': None}
        progress['record'] = manual
        for index, segment in enumerate(segments):
            nominal = _joints(segment['effective'])
            measured = _joints(observation['joints_deg'])
            reference = list(applied) if confirmed_target else measured
            candidate, fallback = validated_bias_target(nominal, reference, measured,
                lambda start, end: self._checked_joint_segment(start, end, phase, observed_start=True))
            entry = {'nominal_target_deg': nominal, 'previous_sent_target_deg': list(applied) if confirmed_target else None,
                     'observed_before_deg': measured, 'fallback_reason': fallback, 'attempts': []}
            manual['segments'].append(entry)
            for attempt in range(2):
                if self._cancelled.is_set():
                    raise _Abort('cancelled', 'cancelled before mock pregrasp lowering')
                steps, sent = quantize_joints(candidate)
                sent = list(sent)
                self._checked_joint_segment(_joints(observation['joints_deg']), sent, phase, observed_start=True)
                age = time.monotonic() - observation['observed_monotonic']
                if (age > config.limits.observation_max_age
                        or observation['sensor_age_seconds'] + age > config.limits.sensor_max_age):
                    raise _Abort('observation_expired', 'mock pregrasp observation expired')
                action = {'kind': 'move_joints', 'joints_deg': list(candidate), 'speed_pct': self.speed_pct,
                          'nominal_arrival_joints_deg': nominal, 'allow_mock_nominal_retry': attempt == 0,
                          'path_constraints': {key: value for key, value in phase.items() if key != 'kind'}}
                if index + 1 < len(segments) and calibration.waypoint_tolerance_mm is not None:
                    action['arrival_tolerance_mm'] = calibration.waypoint_tolerance_mm
                command_id = uuid.uuid4().hex
                entry['attempts'].append({'command_id': command_id, 'requested_target_deg': list(candidate),
                                          'quantized_target_deg': sent, 'outcome': 'unknown'})
                progress['in_flight'] = progress['sent_or_unknown'] = True
                response = self.controller.apply_action({'lease_id': lease, 'command_id': command_id,
                    'observation_id': observation['observation_id'], 'action': action})
                if (not isinstance(response, dict) or response.get('command_id') != command_id
                        or response.get('replayed') is not False):
                    raise _Abort('reply_mismatch', 'mock pregrasp response cannot be confirmed')
                status = response.get('status')
                if status == 'cancelled' and response.get('applied') is False:
                    progress['in_flight'] = False
                    raise _Abort('cancelled', 'mock pregrasp was cancelled before transmission')
                applied_now = response.get('applied') is True
                no_op = status == 'already_at_target' and response.get('applied') is False
                if status not in ('completed', 'settled_needs_nominal_retry') and not no_op:
                    raise _Abort('command_unconfirmed', 'mock pregrasp command did not complete')
                self._mock_nominal_response(response, candidate, sent, steps, nominal)
                if applied_now and (not isinstance(response.get('stop'), dict)
                                    or response['stop'].get('confirmed') is not True):
                    raise _Abort('command_unconfirmed', 'mock pregrasp stop was not confirmed')
                if no_op and response.get('reason') != 'already_at_nominal_target':
                    raise _Abort('reply_mismatch', 'mock pregrasp no-op was not nominal')
                after = response.get('observation')
                instance, last_time, last_sequence = self._observation(
                    after, instance, seen, last_time, last_sequence, require_new_sample=applied_now)
                self._gripper(after, 'open')
                measured = _joints(after['joints_deg'])
                joint_error = max(abs(a-b) for a,b in zip(nominal, measured))
                tcp_error = math.dist(geometry.forward(nominal), geometry.forward(measured))
                tolerance = action.get('arrival_tolerance_mm', config.limits.tcp_tolerance_mm)
                if applied_now:
                    applied, confirmed_target = list(sent), True
                    reference_source = 'controller_confirmed_sent_target'
                    progress['applied_count'] += 1
                progress['in_flight'] = False
                entry['attempts'][-1].update(outcome=status, observed_after_deg=measured,
                                             applied_target_deg=list(applied) if confirmed_target else None)
                observation = after
                progress.update(observation=after, applied=list(applied), confirmed_target=confirmed_target,
                                reference_source=reference_source, instance=instance,
                                last_time=last_time, last_sequence=last_sequence)
                if self._cancelled.is_set():
                    raise _Abort('cancelled', 'cancelled after mock pregrasp segment')
                if status == 'settled_needs_nominal_retry':
                    if (attempt != 0 or response.get('mock_retry_permitted') is not True
                            or response.get('nominal_arrival_reached') is not False
                            or joint_error > 1.0 or tcp_error > 5.0):
                        raise _Abort('command_unconfirmed', 'mock pregrasp retry is not permitted')
                    candidate = [goal + max(-.5, min(.5, goal-observed))
                                 for goal, observed in zip(nominal, measured)]
                    _, retry_target = quantize_joints(candidate)
                    if (_close(retry_target, sent)
                            or max(abs(a-b) for a,b in zip(retry_target, nominal)) > 1.0):
                        raise _Abort('nominal_retry_unavailable', 'mock pregrasp retry is unavailable')
                    continue
                if (response.get('nominal_arrival_reached') is not True
                        or joint_error > config.limits.joint_tolerance_deg or tcp_error > tolerance):
                    raise _Abort('target_unconfirmed', 'mock pregrasp nominal arrival was not confirmed')
                progress['nominal'] = nominal
                progress['reference_source'] = ('controller_confirmed_nominal_move' if confirmed_target
                                                else reference_source)
                reference_source = progress['reference_source']
                break
            else:
                raise _Abort('nominal_retry_unavailable', 'mock pregrasp retry limit was reached')
        manual['completed'] = True
        return progress

    def _release_setdown_goal(self, observation, task, slots, grasp_reference, *, returning):
        """Use robot observations and registered levels, never object truth."""
        if not isinstance(grasp_reference, dict):
            raise _Abort('release_reference_unknown', 'confirmed grasp observation is unavailable')
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        grasp = geometry.forward(_joints(grasp_reference.get('joints_deg')))
        destination = task['source_slot'] if returning else task['destination_slot']
        level = task['source_level'] if returning else task['destination_level']
        target_z = grasp[2] + (level - task['source_level']) * self.path_calibration.cylinder_height_mm
        joints = _joints(observation['joints_deg'])
        xyz = geometry.forward(joints)
        if math.dist(xyz[:2], slots[destination - 1]) > self.path_calibration.vertical_corridor_mm:
            raise _Abort('release_not_at_destination', 'OPEN observation is outside the registered destination')
        tolerance = self.controller.config.limits.tcp_tolerance_mm
        if xyz[2] < target_z - tolerance:
            raise _Abort('release_below_reference', 'OPEN observation is below the recorded grasp-level reference')
        record = {'kind': 'recorded_grasp_registered_level_setdown', 'student_action': 'open',
                  'destination_slot': destination, 'source_level': task['source_level'], 'release_level': level,
                  'grasp_reference': copy.deepcopy(grasp_reference), 'before_tcp_mm': list(xyz),
                  'reference_height_mm': target_z, 'height_tolerance_mm': tolerance,
                  'object_pose_observed': False, 'support_contact_observed': False,
                  'assumption': 'payload remains in the recorded grasp relative pose', 'commands': []}
        if abs(xyz[2] - target_z) <= tolerance:
            record.update(completed=True, movement_needed=False)
            return None, record
        pitch = 90. + sum(joints[1:])
        goal = _joints(geometry.inverse((xyz[0], xyz[1], target_z), pitch))
        record.update(movement_needed=True, requested_target_deg=list(goal))
        return goal, record

    def _lower_before_open(self, observation, task, slots, grasp_reference, *, returning, lease,
                           applied, confirmed_target, reference_source, instance, seen,
                           last_time, last_sequence, progress):
        """Finish an executor-only descent while the original OPEN is pending."""
        config = self.controller.config
        if config.backend == 'sdk':
            raise _Abort('release_setdown_disabled',
                         'physical student OPEN does not add an executor descent')
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        progress.update(observation=observation, applied=list(applied), confirmed_target=confirmed_target,
                        reference_source=reference_source, instance=instance,
                        last_time=last_time, last_sequence=last_sequence, preempted=False)
        goal, record = self._release_setdown_goal(observation, task, slots, grasp_reference, returning=returning)
        progress['record'] = record
        if goal is None:
            return progress
        _, nominal = quantize_joints(goal)
        parts, phase, _ = self._checked_move_segments(observation, goal, nominal, task)
        destination_xy = slots[record['destination_slot'] - 1]
        if phase['kind'] != 'vertical' or not _close(phase.get('vertical_xy_mm', []), destination_xy):
            raise _Abort('release_path_invalid', 'setdown must stay in the actual destination corridor')
        if self.enable_blended_path:
            commands = [self._blended_waypoint_command(observation, parts, phase)[0]]
        else:
            commands = [{'kind': 'move_joints', 'joints_deg': list(part['requested']), 'speed_pct': self.speed_pct,
                         'path_constraints': {key: value for key, value in phase.items() if key != 'kind'},
                         'nominal_arrival_joints_deg': list(part['effective']), 'allow_nominal_correction': True}
                        for part in parts]

        def pending_return():
            if not self._source_return_enabled or returning:
                return False
            with self._state_lock:
                return self._source_return_state['pending'] or self._source_return_state['admission_waiting']

        for command_index, command in enumerate(commands):
            expected_pose = _joints(observation['joints_deg'])
            observation = self.controller.observe(lease)
            instance, last_time, last_sequence = self._observation(observation, instance, seen, last_time, last_sequence)
            self._gripper(observation, 'closed')
            if not _close(_joints(observation['joints_deg']), expected_pose, config.limits.observation_joint_drift_deg):
                raise _Abort('state_changed', 'joints changed while preparing release descent')
            retain_release = config.backend == 'sdk' and returning
            allow_release_compensation = retain_release and command_index + 1 == len(commands)
            release_compensation_basis = None
            if retain_release:
                command = self._stationary_blend_as_single(command, observation)
                command['completion_mode'] = 'retain_position_target'
                if allow_release_compensation:
                    command['allow_damped_compensation'] = True
            nominal = _joints(command['nominal_arrival_joints_deg'])
            for attempt in range(2):
                if self._cancelled.is_set():
                    raise _Abort('cancelled', 'cancelled before release descent')
                if pending_return():
                    progress.update(preempted=True, observation=observation, instance=instance,
                                    last_time=last_time, last_sequence=last_sequence)
                    return progress
                requested = _joints(command['waypoints_deg'][-1] if command['kind'] == 'blended_joint_path'
                                    else command['joints_deg'])
                steps, sent = quantize_joints(requested)
                if (allow_release_compensation and attempt == 0
                        and list(steps) != list(quantize_joints(nominal)[0])):
                    raise _Abort('segment_endpoint_mismatch', 'release compensation requires the original nominal goal')
                before_command_sequence = observation['sensor_sequence']
                command_id = uuid.uuid4().hex
                entry = {'command_id': command_id, 'action': copy.deepcopy(command), 'outcome': 'unknown'}
                record['commands'].append(entry)
                def mark_lower_in_flight():
                    progress['in_flight'] = progress['sent_or_unknown'] = True

                preempted, response = self._dispatch_guarded_action(
                    {'lease_id': lease, 'command_id': command_id,
                     'observation_id': observation['observation_id'], 'action': command},
                    command_id=command_id, return_active=returning,
                    on_before_send=mark_lower_in_flight)
                if preempted:
                    progress['preempted'] = True
                    return progress
                if (not isinstance(response, dict) or response.get('command_id') != command_id
                        or response.get('replayed') is not False):
                    raise _Abort('reply_mismatch', 'release descent response is unconfirmed')
                status, did_send = response.get('status'), response.get('applied') is True
                entry['outcome'] = status
                if status == 'cancelled' and self._cancelled.is_set():
                    progress['in_flight'] = response.get('applied') is not False
                    raise _Abort('cancelled', 'release descent was cancelled')
                if self._source_return_enabled and not returning:
                    while not self._source_return_admission_done.wait(.02):
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'release descent retarget admission was cancelled')
                if status == 'cancelled' and pending_return() and not self._cancelled.is_set():
                    with self._state_lock:
                        interruption = self._source_return_state['interruption']
                    if (not isinstance(interruption, dict) or interruption.get('command_id') != command_id
                            or type(response.get('applied')) is not bool):
                        raise _Abort('command_unconfirmed', 'release descent interruption is unconfirmed')
                    if did_send:
                        self._target_response(response, requested, sent, steps)
                        if command['kind'] == 'blended_joint_path':
                            entry['execution'] = self._blended_response(response, sent)
                        progress['applied_count'] += 1
                    stopped = self.controller.observe(lease)
                    instance, last_time, last_sequence = self._observation(
                        stopped, instance, seen, last_time, last_sequence, require_new_sample=True, expected_state='stopped')
                    self._gripper(stopped, 'closed')
                    stopped_joints = _joints(stopped['joints_deg'])
                    if not did_send and not _close(stopped_joints, observation['joints_deg'],
                                                   config.limits.observation_joint_drift_deg):
                        raise _Abort('state_changed', 'release pose changed before transmission')
                    resume = (self.controller.resume_interrupted_motion if self.enable_runtime_source_return
                              else self.controller.resume_mock_interrupted_motion)
                    resumed = resume(lease, interruption['interruption_id'], stopped['observation_id'])
                    if (not isinstance(resumed, dict) or resumed.get('resumed') is not True
                            or not _close(_joints(resumed.get('joints_deg')), stopped_joints,
                                          config.limits.observation_joint_drift_deg)):
                        raise _Abort('stop_unconfirmed', 'release descent did not rearm from its stopped pose')
                    observation = self.controller.observe(lease)
                    instance, last_time, last_sequence = self._observation(
                        observation, instance, seen, last_time, last_sequence)
                    self._gripper(observation, 'closed')
                    if not _close(_joints(observation['joints_deg']), stopped_joints,
                                  config.limits.observation_joint_drift_deg):
                        raise _Abort('state_changed', 'release descent changed after interrupted stop')
                    progress.update(in_flight=False, preempted=True, observation=observation,
                                    applied=stopped_joints if did_send else progress['applied'],
                                    confirmed_target=False if did_send else progress['confirmed_target'],
                                    reference_source='stationary_observation_after_setdown_interruption',
                                    instance=instance, last_time=last_time, last_sequence=last_sequence)
                    entry.update(stopped_joints_deg=stopped_joints, open_sent=False)
                    return progress
                no_op = status == 'already_at_target' and response.get('applied') is False
                if status not in ('completed', 'settled_needs_nominal_correction') and not no_op:
                    raise _Abort('command_unconfirmed', 'release descent did not complete')
                self._mock_nominal_response(response, requested, sent, steps, nominal)
                if status == 'settled_needs_nominal_correction' and allow_release_compensation:
                    if (not did_send or response.get('nominal_arrival_reached') is not False
                            or response.get('stop', {}).get('requested') is not False
                            or response.get('stop', {}).get('confirmed') is not False):
                        raise _Abort('stop_unconfirmed', 'release correction did not retain its parent target')
                elif did_send and (not retain_release or status != 'completed') and response.get('stop', {}).get('confirmed') is not True:
                    raise _Abort('stop_unconfirmed', 'release descent stop is unconfirmed')
                if command['kind'] == 'blended_joint_path' and did_send:
                    entry['execution'] = self._blended_response(response, sent)
                observation = response.get('observation')
                instance, last_time, last_sequence = self._observation(
                    observation, instance, seen, last_time, last_sequence,
                    require_new_sample=did_send or (retain_release and no_op))
                self._gripper(observation, 'closed')
                if retain_release and did_send and status == 'completed':
                    retained = self._retained_move_response(
                        response, steps, command['speed_pct'], before_command_sequence, observation)
                    retained.update(nominal_target_deg=list(nominal), speed_pct=command['speed_pct'],
                        path_constraints=copy.deepcopy(command.get('path_constraints', {})),
                        arrival_tolerance_mm=command.get('arrival_tolerance_mm'),
                        compensation_allowed=bool(allow_release_compensation),
                        compensation_used=release_compensation_basis is not None,
                        checked_phase=copy.deepcopy(phase))
                    progress['retained_move'] = retained
                    sent = [step / STEPS_PER_DEGREE for step in retained['target_steps']]
                    entry['completion'] = copy.deepcopy(response['completion'])
                    entry['sdk_host_encoded_target_deg'] = list(sent)
                elif retain_release and no_op:
                    progress['retained_move'] = self._retained_noop_response(
                        response, progress.get('retained_move'), steps, before_command_sequence, observation)
                    entry['retained_position_observation'] = copy.deepcopy(response['retained_position_observation'])
                elif retain_release and status == 'settled_needs_nominal_correction':
                    progress['retained_move'] = None
                progress.update(in_flight=False, observation=observation, instance=instance,
                                last_time=last_time, last_sequence=last_sequence)
                if did_send:
                    progress['applied_count'] += 1
                    progress.update(applied=list(sent), confirmed_target=True,
                                    reference_source='controller_confirmed_move')
                if status == 'settled_needs_nominal_correction':
                    correction = response.get('correction_action')
                    if (attempt or response.get('nominal_correction_permitted') is not True
                            or not isinstance(correction, dict) or correction.get('kind') != 'move_joints'
                            or correction.get('correction_of_command_id') != command_id
                            or correction.get('allow_nominal_correction', False) is not False
                            or not _close(_joints(correction.get('nominal_arrival_joints_deg')), nominal)
                            or correction.get('speed_pct') != command['speed_pct']
                            or correction.get('arrival_tolerance_mm') != command.get('arrival_tolerance_mm')
                            or correction.get('path_constraints') != {
                                key: list(value) if isinstance(value, tuple) else value
                                for key, value in command['path_constraints'].items()}):
                        raise _Abort('reply_mismatch', 'release arrival correction differs from the checked descent')
                    candidate = _joints(correction.get('joints_deg'))
                    if max(abs(a-b) for a, b in zip(quantize_joints(candidate)[1], nominal)) > 1.:
                        raise _Abort('reply_mismatch', 'release correction exceeds one degree')
                    self._checked_joint_segment(observation['joints_deg'], quantize_joints(candidate)[1], phase,
                                                observed_start=True)
                    if allow_release_compensation:
                        release_compensation_basis = self._damped_correction_basis(
                            response, correction, command_id, nominal, _joints(observation['joints_deg']),
                            observation, command['speed_pct'], before_command_sequence,
                            command.get('path_constraints', {}))
                        entry['damped_compensation_basis'] = copy.deepcopy(release_compensation_basis)
                    command = correction
                    if retain_release:
                        command['completion_mode'] = 'retain_position_target'
                    continue
                measured = _joints(observation['joints_deg'])
                if (response.get('nominal_arrival_reached') is not True
                        or not _close(measured, nominal, config.limits.joint_tolerance_deg)
                        or math.dist(geometry.forward(measured), geometry.forward(nominal)) > config.limits.tcp_tolerance_mm):
                    raise _Abort('target_unconfirmed', 'release nominal arrival is unconfirmed')
                if release_compensation_basis is not None:
                    entry['damped_compensation'] = self._damped_compensation_completed(
                        response, release_compensation_basis, nominal, sent, observation,
                        config.limits.tcp_tolerance_mm)
                if (retain_release and release_compensation_basis is None
                        and (not _close(measured, sent, config.limits.joint_tolerance_deg)
                        or math.dist(geometry.forward(measured), geometry.forward(sent)) > config.limits.tcp_tolerance_mm)):
                    raise _Abort('target_unconfirmed', 'release sent target arrival is unconfirmed')
                progress['nominal_target_deg'] = list(nominal)
                break
            else:
                raise _Abort('command_unconfirmed', 'release correction limit was reached')
        final = geometry.forward(_joints(observation['joints_deg']))
        if (abs(final[2] - record['reference_height_mm']) > record['height_tolerance_mm']
                or math.dist(final[:2], destination_xy) > self.path_calibration.vertical_corridor_mm):
            raise _Abort('release_pose_unconfirmed', 'release pose differs from the grasp-level reference')
        record.update(completed=True, final_observed_tcp_mm=list(final))
        return progress

    def _checked_move_segments(self, observation, requested, effective, task):
        """Preflight every subdivision and retain the exact student endpoint."""
        start = _joints(observation['joints_deg'])
        calibration = self.path_calibration
        config = self.controller.config
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                            tool_vertical_mm=config.tool_vertical_mm)
        start_xyz, target_xyz = geometry.forward(start), geometry.forward(effective)
        centers = [calibration.slot_xy_mm[task[key] - 1]
                   for key in ('source_slot', 'destination_slot')]
        # A held-object redirect may begin low in the previous destination's
        # registered corridor. The original student endpoint and all existing
        # vertical-envelope, workspace and ceiling checks remain unchanged.
        old_destination = self._direct_previous_destination_slot
        if type(old_destination) is int and old_destination in calibration.enabled_slots:
            centers.append(calibration.slot_xy_mm[old_destination - 1])
        center = min(centers, key=lambda xy: math.dist(xy, start_xyz[:2]) + math.dist(xy, target_xyz[:2]))
        corridor = calibration.vertical_corridor_mm
        if math.dist(center, start_xyz[:2]) <= corridor and math.dist(center, target_xyz[:2]) <= corridor:
            phase = {'kind': 'vertical', 'vertical_xy_mm': center, 'corridor_mm': corridor}
        elif min(start_xyz[2], target_xyz[2]) > calibration.transport_floor_mm:
            phase = {'kind': 'transport', 'minimum_z_mm': calibration.transport_floor_mm}
        else:
            raise ControlError('LOW_HORIZONTAL_PATH', 'A low horizontal student path requires separate planning')
        if calibration.path_ceiling_mm is not None:
            phase['maximum_z_mm'] = calibration.path_ceiling_mm
        try:
            evidence = self._checked_joint_segment(start, effective, phase, observed_start=True)
        except ControlError as exc:
            if exc.code not in {'ASYNC_ENVELOPE', 'VERTICAL_CORRIDOR'}:
                raise
            endpoint_geometry = _EndpointPreservingGeometry(geometry, target_xyz, requested, start)
            segments = vertical_segments(
                endpoint_geometry, start, target_xyz,
                lambda a, b: self._checked_joint_segment(a, b, phase, observed_start=False),
                allow_async_subdivision=True, compact=calibration.compact_waypoints)
            if not segments or not _close(segments[-1]['effective'], effective):
                raise _Abort('segment_endpoint_mismatch', 'subdivision changed the student target')
            return segments, phase, exc.as_dict()
        return [{'requested': requested, 'effective': effective, 'path': evidence}], phase, None

    def _blended_waypoint_command(self, observation, parts, phase):
        """Bundle one model MOVE without adding predictions or object inputs."""
        previous = _joints(observation['joints_deg'])
        # This is a C2 planning limit, not a measured physical servo speed.
        velocity_limit = (self.path_calibration.blended_max_velocity_deg_s if self.path_calibration is not None
                          else DEFAULT_BLENDED_MAX_VELOCITY_DEG_S)
        waypoints, durations, checks = [], [], []
        for index, part in enumerate(parts):
            target = _joints(part['effective'])
            # Keep the endpoint once. Repeating the last internal point as a
            # zero-length final segment would falsely require an inner stop.
            # A wholly stationary MOVE still retains its explicit endpoint.
            if _close(previous, target) and (index + 1 < len(parts) or waypoints):
                continue
            checks.append(self._checked_joint_segment(previous, target, phase, observed_start=not waypoints))
            delta = max(abs(a-b) for a,b in zip(previous, target))
            duration = max(.04, 1.875 * delta / velocity_limit, math.sqrt(6. * delta / 60.),
                           (60. * delta / 300.) ** (1. / 3.))
            waypoints.append(list(target))
            durations.append(duration)
            previous = target
        if not waypoints:
            raise _Abort('invalid_blended_path', 'a final waypoint is required')
        return {'kind': 'blended_joint_path', 'waypoints_deg': waypoints,
                'segment_durations_s': durations, 'max_velocity_deg_s': velocity_limit,
                'max_acceleration_deg_s2': 60., 'max_jerk_deg_s3': 300.,
                'sample_period_seconds': .002, 'speed_pct': self.speed_pct,
                'path_constraints': {key: value for key, value in phase.items() if key != 'kind'},
                'nominal_arrival_joints_deg': list(waypoints[-1]), 'allow_nominal_correction': True}, checks

    def _restore_interrupted_move(self, previous, before, task, slots, initial):
        """Recompute saved policy decisions without replaying motor commands.

        Only a partly executed MOVE can resume. Its original observation remains
        in policy history until its original final target is confirmed. The fresh
        physical pose is used exclusively to validate the remaining device path.
        """
        rows = previous.get('decisions') if isinstance(previous, dict) else None
        if (not isinstance(rows, list) or not rows or len(rows) >= self.max_decisions
                or rows[-1].get('action') != 'move'
                or rows[-1].get('result') != 'command_outcome_unknown'
                or previous.get('last_confirmed_gripper_command') not in ('open', 'closed')):
            raise _Abort('invalid_resume', 'a recorded interrupted MOVE is required')
        history, applied, gripper = [], list(initial), 'open'
        self.model.on_scene_registered(before, slots)
        self.model.on_transfer_started(before, task, {'joints_deg': list(initial)})
        for index, row in enumerate(rows):
            features = row.get('policy_features')
            if (row.get('index') != index or not isinstance(features, list)
                    or len(features) != len(scene_features.FEATURE_NAMES)
                    or any(not _number(v) for v in features)
                    or not _close(_joints(row.get('previous_applied_target_deg')), applied)):
                raise _Abort('invalid_resume', 'saved policy history is inconsistent')
            old_observation = row.get('end_motion_policy_observation')
            if old_observation is None:
                old_observation = {'schema_version': 'PolicyObservation.phase1',
                                   'joints_deg': [v * 180. for v in features[:4]],
                                   'sensor_valid': True, 'moving': False,
                                   'gripper_command': gripper, 'gripper_reported_state': gripper,
                                   'gripper_reported_valid': True, 'gripper_feedback_fresh': True}
            else:
                self._gripper(old_observation, gripper)
            if not _close(features, scene_features.encode_scene_features(old_observation, before, task, slots)):
                raise _Abort('invalid_resume', 'saved scene, task or gripper observation differs')
            history.append(features)
            window, mask = scene_features.history_window(history, 8)
            if mask != row.get('valid_history'):
                raise _Abort('invalid_resume', 'saved history mask differs')
            with torch.inference_mode():
                logits, angles = self.model.forward_with_control_state(
                    torch.tensor([window], dtype=torch.float32),
                    torch.tensor([mask], dtype=torch.bool),
                    torch.tensor([applied], dtype=torch.float32))
            action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
            proposed = _joints(native_command_target(action, angles, applied))
            if action != row.get('action') or (action == 'move' and
                    not _close(proposed, _joints(row.get('proposed_target_deg')), 1e-5)):
                raise _Abort('invalid_resume', 'the selected model no longer reproduces the saved decision')
            if index == len(rows) - 1:
                if gripper != previous['last_confirmed_gripper_command']:
                    raise _Abort('invalid_resume', 'saved gripper history differs')
                return history, gripper, angles, copy.deepcopy(row)
            _, effective = quantize_joints(proposed)
            if (row.get('result') != 'command_applied' or row.get('transmitted') is not True
                    or not _close(_joints(row.get('applied_target_deg')), effective)):
                raise _Abort('invalid_resume', 'only confirmed complete decisions may be restored')
            applied = list(effective)
            gripper = 'closed' if action == 'close' else 'open' if action == 'open' else gripper
            self.model.on_control_applied(action, list(applied), list(applied), None)
        raise _Abort('invalid_resume', 'missing interrupted decision')

    def _direct_contact_speed_pct(self, start_joints, target_joints, requested_speed_pct):
        """Cap SDK low-height descents/lifts to reduce j2 slip and j0 overshoot."""
        speed = float(requested_speed_pct)
        if speed <= 8. or start_joints is None or target_joints is None:
            return speed
        start = _joints(start_joints)
        target = _joints(target_joints)
        geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                            tool_vertical_mm=self.controller.config.tool_vertical_mm)
        if min(geometry.forward(start)[2], geometry.forward(target)[2]) >= 60.:
            return speed
        max_delta_deg = max(abs(b - a) for a, b in zip(start, target))
        # Hardware telemetry shows 0/544 settled stalls at 8% vs 20.5% at 30%;
        # cap >68 deg slews at 12% so duration stays well under the 30s timeout.
        capped = 8. if max_delta_deg <= 68. else 12.
        return float(min(speed, capped))

    def run(self, registered_scene, task, slot_xy_mm, *, initial_applied_target_deg,
            initial_target_source='caller_confirmed', initial_observation_instance_id=None,
            initial_observation_revision=None, initial_nominal_target_deg=None,
            resume_from=None, stopped_joints_deg=None, object_state_confirmed=False,
            on_release_confirmed=None, initial_previous_student_target_deg=None):
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError('a transfer is already running')
        self._forecast_run_token = object()
        result = {'termination': None, 'model_done': False, 'physical_success': None,
                  'layout_uncertain': False, 'current_target_known': False,
                  'initial_target_source': initial_target_source, 'decisions': [],
                  'last_encoded_student_target': None,
                  'applied_command_count': 0, 'last_confirmed_target_deg': None,
                  'current_control_reference_deg': None, 'current_control_reference_source': None,
                  'last_confirmed_gripper_command': None,
                  'control_profile': {'arm': 'controller_managed_joint_target',
                                      'gripper': 'controller_managed',
                                      'segmented_path': ('disabled_for_sdk_student_move'
                                                         if self.controller.config.backend == 'sdk' else
                                                         'experimental_calibrated' if self.enable_experimental_segmented_path
                                                         else 'disabled'),
                                      'waypoint_compaction': ('disabled_for_sdk_student_move'
                                                              if self.controller.config.backend == 'sdk' else
                                                              'checked_shortcuts' if self.enable_experimental_segmented_path
                                                              and self.path_calibration.compact_waypoints else 'disabled'),
                                      'support_height_guard': ('mock_only_nominal' if self.execute_mock_virtual_descent
                                                               else 'not_applied'),
                                      'waypoint_execution': ('sdk_single_student_target'
                                                             if self.controller.config.backend == 'sdk' else
                                                             'checked_transport_single_target_else_blended_C2'
                                                             if self.enable_blended_path else 'stop_per_waypoint'),
                                      'release_setdown': ('recorded_grasp_registered_levels'
                                                          if self.enable_release_setdown
                                                          and self.controller.config.backend != 'sdk'
                                                          else 'disabled'),
                                      'student_endpoint': ('normal_original_quantized_target_no_alternate_correction'
                                                           if self.controller.config.backend == 'sdk'
                                                           else 'configured_mock_behavior'),
                                      'virtual_descent': ('mock_trajectory_execution' if self.execute_mock_virtual_descent
                                                          else 'comparison_only' if self.compare_virtual_descent
                                                          else 'not_computed')}}
        lease = None
        sent_or_unknown = False
        in_flight = False
        unknown = False
        retained_move = None
        last_student_move = None
        preceding_move_attempts = None
        direct_cancel = None
        initial_observation = None
        self._direct_previous_destination_slot = None

        def cancelled():
            return self._cancelled.is_set() or (direct_cancel is not None and direct_cancel.is_set())

        def remember_no_motion(move_start_observation, attempts):
            # Preserve observations and delivery evidence without deciding
            # whether the registered object arrangement is still valid.
            result['no_motion_evidence'] = {
                'initial_observation': copy.deepcopy(initial_observation),
                'move_start_observation': copy.deepcopy(move_start_observation),
                'attempts': copy.deepcopy(attempts),
                'gripper_commands': [
                    {key: copy.deepcopy(item.get(key)) for key in
                     ('index', 'action', 'command_id', 'result', 'transmitted', 'gripper_delivery')}
                    for item in result['decisions'] if item.get('action') in ('open', 'close')],
                'end_episode': None,
                'post_stop_observation': None,
                'post_stop_observation_validated': False,
            }

        def remember_student_host_frame(steps, command_id, frame, receipt):
            if (not isinstance(frame, dict) or not isinstance(receipt, dict)
                    or type(receipt.get('encoding_after_frame_sequence')) is not int
                    or type(frame.get('frame_sequence')) is not int
                    or frame['frame_sequence'] <= receipt['encoding_after_frame_sequence']
                    or not _number(receipt.get('submitted_monotonic'))
                    or not _number(frame.get('monotonic'))
                    or frame['monotonic'] < receipt['submitted_monotonic']):
                result['last_encoded_student_target'] = None
                return
            result['last_encoded_student_target'] = {
                'joints_deg': [step / STEPS_PER_DEGREE for step in steps],
                'command_id': command_id, 'instance_id': instance,
                'host_frame': copy.deepcopy(frame), 'source': 'sdk_host_frame',
                'submission': {key: receipt[key] for key in
                               ('encoding_after_frame_sequence', 'submitted_monotonic')},
                'physical_arrival_confirmed': None,
            }

        try:
            before, task = scene_features.transfer_task(registered_scene, task)
            if self.enable_mock_source_return and (self.controller.config.backend != 'mock'
                    or self.controller.config.hardware_enabled is not False or self.controller.config.port is not None):
                raise _Abort('source_return_unsupported', 'the source-return experiment requires a mock controller')
            if self.enable_experimental_segmented_path:
                calibration = self.path_calibration
                if (not isinstance(slot_xy_mm, (list, tuple)) or len(slot_xy_mm) != 4
                        or any(not isinstance(xy, (list, tuple)) or len(xy) != 2
                               or not _close(xy, expected)
                               for xy, expected in zip(slot_xy_mm, calibration.slot_xy_mm))
                        or task['source_slot'] not in calibration.enabled_slots
                        or task['destination_slot'] not in calibration.enabled_slots
                        or max(map(len, before)) > calibration.max_height
                        or task['destination_level'] > calibration.max_height):
                    raise _Abort('path_calibration_mismatch', 'scene or slots exceed the supplied calibration')
            if initial_target_source not in ('caller_confirmed', 'controller_confirmed_move', 'stationary_observation',
                                             'controller_confirmed_nominal_move'):
                raise _Abort('initial_reference_unknown', 'unsupported initial control reference source')
            nominal_initial = initial_target_source == 'controller_confirmed_nominal_move'
            if nominal_initial and (not self.enable_mock_nominal_trim or initial_nominal_target_deg is None):
                raise _Abort('initial_reference_unknown', 'nominal start reference is available only in mock trim')
            if not nominal_initial and initial_nominal_target_deg is not None:
                raise _Abort('initial_reference_unknown', 'unexpected nominal start reference')
            observed_initial = initial_target_source == 'stationary_observation'
            if observed_initial and (not isinstance(initial_observation_instance_id, str)
                    or not initial_observation_instance_id.strip()
                    or type(initial_observation_revision) is not int or initial_observation_revision < 0):
                raise _Abort('initial_reference_unknown', 'stationary observation identity and revision are required')
            initial = _joints(initial_applied_target_deg)
            previous_student_target = None
            if initial_previous_student_target_deg is not None:
                if (self.controller.config.backend != 'sdk' or not observed_initial or resume_from is not None):
                    raise _Abort('initial_reference_unknown', 'previous student target requires an SDK stopped reference')
                previous_student_target = _joints(initial_previous_student_target_deg)
                _, previous_rounded = quantize_joints(previous_student_target)
                if not _close(previous_student_target, previous_rounded):
                    raise _Abort('initial_target_unknown', 'previous student target must match encoded joint steps')
            result['initial_student_target_source'] = (
                'previous_sdk_host_frame' if previous_student_target is not None else initial_target_source)
            initial_nominal = _joints(initial_nominal_target_deg) if nominal_initial else None
            if initial_nominal is not None:
                _, nominal_rounded = quantize_joints(initial_nominal)
                if (not _close(initial_nominal, nominal_rounded)
                        or max(abs(a-b) for a, b in zip(initial, initial_nominal)) > 1.0):
                    raise _Abort('initial_reference_unknown', 'nominal and sent start references are inconsistent')
            _, rounded = quantize_joints(initial)
            if not observed_initial and not self.execute_mock_virtual_descent and not _close(initial, rounded):
                raise _Abort('initial_target_unknown', 'initial target must already be a confirmed quantized target')
            restored = None
            if resume_from is not None:
                if (self.enable_mock_nominal_trim or observed_initial or object_state_confirmed is not True or not self.enable_experimental_segmented_path
                        or self.execute_mock_virtual_descent):
                    raise _Abort('invalid_resume', 'resume requires confirmed object state and calibrated position control')
                stopped = _joints(stopped_joints_deg)
                restored = self._restore_interrupted_move(resume_from, before, task, slot_xy_mm, initial)
                result.update(resumed=True, restored_history_decisions=len(resume_from['decisions']) - 1,
                              initial_target_source='prior_confirmed_decisions_and_current_stopped_feedback')
            if self._cancelled.is_set():
                raise _Abort('cancelled', 'cancelled before acquiring control')
            try:
                acquired = self.controller.acquire('native-student-transfer')
            except ControlError as exc:
                # NOT_READY is an explicit refusal before a lease is issued.
                # A lost reply or any other acquire error is still ambiguous.
                if exc.code == 'NOT_READY':
                    result['acquire_rejected'] = 'not_ready'
                    result['lease_state_uncertain'] = False
                else:
                    result['lease_state_uncertain'] = True
                raise
            except Exception:
                result['lease_state_uncertain'] = True
                raise
            if not isinstance(acquired, dict) or not isinstance(acquired.get('lease_id'), str) or not acquired['lease_id']:
                result['lease_state_uncertain'] = True
                raise _Abort('lease_unconfirmed', 'controller did not return a valid lease')
            lease = acquired['lease_id']
            with self._state_lock:
                self._lease = lease
            if self._cancelled.is_set():
                raise _Abort('cancelled', 'cancelled before observing')
            seen, instance, last_time, last_sequence = (set(), initial_observation_instance_id if observed_initial else None,
                                                        -1.0, 0)
            try:
                observation = self.controller.observe(lease)
            except Exception:
                unknown = True
                raise
            instance, last_time, last_sequence = self._observation(observation, instance, seen, last_time, last_sequence)
            self._gripper(observation, restored[1] if restored else 'open')
            if observed_initial and observation.get('revision') != initial_observation_revision + 1:
                raise _Abort('initial_reference_changed', 'controller ownership or command state changed after registration')
            tolerance = self.controller.config.limits.joint_tolerance_deg
            drift = self.controller.config.limits.observation_joint_drift_deg
            start_check = stopped if restored else initial_nominal if nominal_initial else initial
            if not _close(_joints(observation['joints_deg']), start_check,
                          drift if restored or observed_initial else tolerance + drift):
                raise _Abort('initial_target_mismatch', 'observed joints differ from the initial control reference')
            # The first observation is also the first decision observation.
            initial_observation = copy.deepcopy(observation)
            expected_joints = _joints(observation['joints_deg'])
            applied = _joints(resume_from['last_confirmed_target_deg']) if restored else list(initial)
            nominal_applied = list(initial_nominal) if initial_nominal is not None else None
            gripper = restored[1] if restored else 'open'
            reference_source = initial_target_source
            confirmed_target = not observed_initial
            result.update(current_target_known=not bool(restored) and confirmed_target,
                          last_confirmed_target_deg=list(applied) if confirmed_target else None,
                          current_control_reference_deg=list(applied),
                          current_control_reference_source=reference_source, last_confirmed_gripper_command=gripper)
            if self.enable_mock_nominal_trim:
                result['last_confirmed_nominal_target_deg'] = nominal_applied
            if not restored:
                self.model.on_scene_registered(before, slot_xy_mm)
                self.model.on_transfer_started(before, task, {'joints_deg': list(
                    previous_student_target if previous_student_target is not None else applied)})
            if self._source_return_enabled:
                with self._state_lock:
                    self._source_return_state = {'phase': 'running', 'source_slot': task['source_slot'],
                        'registered_scene': before, 'original_task': copy.deepcopy(task),
                        'pending_direct_task': None, 'direct_task': None,
                        'gripper': gripper, 'ever_closed': gripper == 'closed', 'opened_after_close': False,
                        'inflight_action': None, 'pending': False, 'pending_cancel_event': None,
                        'active_command_id': None, 'interruption': None,
                        'admission_waiting': False, 'switched': False,
                        'activation_decision_index': None,
                        'lift_status': ('student_controlled' if self.controller.config.backend == 'sdk'
                                        else 'not_started'),
                        'return_move_confirmed': False, 'open_after_switch': False,
                        'grasp_reference': None,
                        'open_pose_before': None, 'open_pose_after': None,
                        'pregrasp_lowering': False, 'pregrasp_cancel_requested': False,
                        'pregrasp_return_requested': False, 'pregrasp_return_applied': False,
                        'close_dispatch_started': False,
                        'early_return_active': None, 'early_return_phase': 0}
                    self._source_return_admission_waiting.clear()
                    self._source_return_admission_done.set()
                result['source_return_expected'] = False
            height_guard = (SupportHeightGuard(task, slot_xy_mm[task['destination_slot'] - 1])
                            if self.execute_mock_virtual_descent else None)
            history = restored[0] if restored else []
            start_index = len(resume_from['decisions']) - 1 if restored else 0
            mock_vertical_replans_used = 0
            grasp_reference = None
            for index in range(start_index, self.max_decisions):
                if cancelled():
                    raise _Abort('cancelled', 'transfer cancelled')
                return_active = False
                direct_active = False
                if self._source_return_enabled:
                    # Let a request waiting behind the previous MOVE reach its
                    # admission boundary before starting another decision.
                    while not self._source_return_admission_done.wait(.02):
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'transfer cancelled during retarget admission')
                    if self._cancelled.is_set():
                        raise _Abort('cancelled', 'transfer cancelled')
                    with self._state_lock:
                        state = self._source_return_state
                        if state['pending']:
                            invalidation = state['pending_cancel_event']
                            if invalidation is not None and invalidation.is_set():
                                if state['interruption'] is not None:
                                    raise _Abort('cancelled', 'interrupted retarget was invalidated')
                                state['pending'] = False
                                state['pending_cancel_event'] = None
                                state['pending_direct_task'] = None
                            else:
                                if (not self.enable_runtime_source_return and (self.controller.config.backend != 'mock'
                                        or self.controller.config.hardware_enabled is not False
                                        or self.controller.config.port is not None)):
                                    raise _Abort('source_return_unsupported', 'the mock controller changed')
                                state['pending'] = False
                                state['pending_cancel_event'] = None
                                state['switched'] = True
                                candidate = state['pending_direct_task']
                                state['pending_direct_task'] = None
                                if candidate is not None:
                                    if state['gripper'] != 'closed' or not state['ever_closed']:
                                        raise _Abort('retarget_gripper_unconfirmed',
                                                     'direct retarget requires the confirmed closed gripper')
                                    _, task = scene_features.transfer_task(before, candidate)
                                    state['direct_task'] = copy.deepcopy(task)
                                    self._direct_previous_destination_slot = state['original_task']['destination_slot']
                                    direct_cancel = invalidation
                        direct_active = state['switched'] and state['direct_task'] is not None
                        return_active = state['switched'] and not direct_active
                        pregrasp_return = (return_active and state['pregrasp_return_requested']
                                           and not state['ever_closed'])
                    if pregrasp_return:
                        if self._cancelled.is_set() or state['close_dispatch_started'] or state['gripper'] != 'open':
                            raise _Abort('pregrasp_unconfirmed', 'gripper state or CLOSE delivery is uncertain')
                        before_stop = observation
                        stopped = self.controller.stop('PREGRASP_RETARGET')
                        if not isinstance(stopped, dict) or stopped.get('confirmed') is not True:
                            raise _Abort('stop_unconfirmed', 'pregrasp retarget stop was not confirmed')
                        observation = self.controller.observe(lease)
                        instance, last_time, last_sequence = self._observation(
                            observation, instance, seen, last_time, last_sequence,
                            require_new_sample=True, expected_state='stopped')
                        self._gripper(observation, 'open')
                        if self.enable_runtime_source_return:
                            self._mock_quiet_diagnostics(before_stop, observation)
                        if (observation['sensor_sequence'] < stopped.get('confirmed_sensor_sequence', float('inf'))
                                or observation['observed_monotonic'] < stopped.get('confirmed_sensor_monotonic', float('inf'))):
                            raise _Abort('stop_unconfirmed', 'pregrasp observation predates the confirmed stop')
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'pregrasp retarget was cancelled')
                        with self._state_lock:
                            state['pregrasp_return_applied'] = True
                            state['activation_decision_index'] = index
                        expected_joints = _joints(observation['joints_deg'])
                        result.update(termination='pregrasp_source_return', model_done=False,
                                      pregrasp_stopped_joints_deg=list(expected_joints),
                                      last_confirmed_gripper_command='open')
                        break
                if index != start_index:
                    try:
                        observation = self.controller.observe(lease)
                    except Exception:
                        unknown = True
                        raise
                    instance, last_time, last_sequence = self._observation(observation, instance, seen, last_time, last_sequence)
                    self._gripper(observation, gripper)
                    observed_joints = _joints(observation['joints_deg'])
                    if (self.controller.config.backend != 'sdk'
                            and not _close(observed_joints, expected_joints, drift)):
                        raise _Abort('state_changed', 'joints changed since the previous confirmed observation')
                    if self.controller.config.backend == 'sdk':
                        if confirmed_target:
                            geometry = Geometry(tool_forward_mm=self.controller.config.tool_forward_mm,
                                                tool_vertical_mm=self.controller.config.tool_vertical_mm)
                            allowed_tcp = ((retained_move or {}).get('arrival_tolerance_mm')
                                           or self.controller.config.limits.tcp_tolerance_mm)
                            confirmed_target = (_close(observed_joints, applied, tolerance)
                                and math.dist(geometry.forward(observed_joints), geometry.forward(applied)) <= allowed_tcp)
                        if not confirmed_target:
                            applied = list(observed_joints)
                            if (reference_source != 'stationary_observation_after_interruption'
                                    or not _close(observed_joints, expected_joints, drift)):
                                reference_source = 'stationary_observation_unreached_target'
                            result.update(current_target_known=False, last_confirmed_target_deg=None,
                                          last_confirmed_nominal_target_deg=None,
                                          current_control_reference_deg=list(applied),
                                          current_control_reference_source=reference_source)
                    expected_joints = observed_joints
                manual_early_action = False
                original_return_action = None
                policy_previous_target = list(self.model.policy.current_target)
                measured_motion = None
                reconsidered_motion = None
                policy_observation_basis = 'stationary_controller_feedback'
                if (preceding_move_attempts and gripper == 'closed' and not return_active and not direct_active):
                    measured_motion, policy_observation_basis = self._end_motion_policy_observation(
                        preceding_move_attempts, observation)
                elif preceding_move_attempts:
                    policy_observation_basis = 'task_changed_use_stationary_feedback'
                preceding_move_attempts = None
                if restored and index == start_index:
                    angles, saved = restored[2:]
                    features, mask, action = saved['policy_features'], saved['valid_history'], 'move'
                elif return_active:
                    features, mask, logits, angles, action = self._source_return_decision(
                        observation, before, task, slot_xy_mm, history, policy_previous_target, index)
                    with self._state_lock:
                        if self._source_return_state['activation_decision_index'] is None:
                            self._source_return_state['activation_decision_index'] = index
                        if self.enable_mock_early_source_return:
                            state = self._source_return_state
                            if state['early_return_active'] is None:
                                config = self.controller.config
                                geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                                    tool_vertical_mm=config.tool_vertical_mm)
                                xyz = geometry.forward(_joints(observation['joints_deg']))
                                source_xy = slot_xy_mm[task['source_slot'] - 1]
                                grip_z = (self.path_calibration.table_z_mm
                                    + (task['source_level'] - 1) * self.path_calibration.cylinder_height_mm
                                    + self.path_calibration.grip_offset_mm)
                                state['early_return_active'] = (math.dist(xyz[:2], source_xy) <= 3.
                                    and grip_z - 3. <= xyz[2] <= grip_z + 30.)
                            if state['early_return_active']:
                                original_return_action = action
                                action = ('move', 'open', 'done')[min(state['early_return_phase'], 2)]
                                self.model.policy.pending['predicted_action'] = motion_policy.ACTIONS.index(action)
                                manual_early_action = True
                                result['manual_early_return_used'] = True
                else:
                    policy_observation = measured_motion or observation
                    self._publish_decision_context(policy_observation, before, task, slot_xy_mm, index)
                    features, mask, logits, angles, action = native_policy_decision(
                        self.model, policy_observation, before, task, slot_xy_mm, history, policy_previous_target)
                    proposed_motion = (self._current_motion_proposal(angles, policy_previous_target)
                                       if measured_motion is not None and action == 'move' else None)
                    if proposed_motion is not None:
                        _, encoded_motion = quantize_joints(proposed_motion)
                        if self._within_stationary_arrival_tolerances(encoded_motion, observation):
                            # A delayed suggestion can already be at the
                            # stopped arm. Reobserve before sending it; never
                            # commit the unsent proposal to the action history.
                            self.model.policy.on_decision_not_sent()
                            history.pop()
                            reconsidered_motion = {'observation': measured_motion,
                                'proposed_target_deg': proposed_motion, 'transmitted': False}
                            measured_motion = None
                            policy_observation_basis = 'motion_proposal_within_stationary_arrival_tolerances'
                            self._publish_decision_context(observation, before, task, slot_xy_mm, index)
                            obsolete = self._reconsideration_superseded()
                            confirmed_history_length = len(history)
                            if not obsolete:
                                try:
                                    features, mask, logits, angles, action = native_policy_decision(
                                        self.model, observation, before, task, slot_xy_mm, history, policy_previous_target)
                                except Exception:
                                    if not self._reconsideration_superseded():
                                        raise
                                    # A superseded inference cannot poison the
                                    # replacement's confirmed history.
                                    del history[confirmed_history_length:]
                                    if self.model.policy.pending is not None:
                                        self.model.policy.on_decision_not_sent()
                                    obsolete = True
                            if obsolete:
                                result['decisions'].append({'index': index, 'action': 'move',
                                    'result': 'superseded_before_send', 'transmitted': False,
                                    'observation_id': observation['observation_id'],
                                    'dispatch_observation_id': observation['observation_id'],
                                    'policy_features': features, 'valid_history': mask,
                                    'policy_observation_basis': 'measured_end_motion',
                                    'end_motion_policy_observation': reconsidered_motion['observation'],
                                    'proposed_target_deg': proposed_motion,
                                    'previous_student_target_deg': list(policy_previous_target)})
                                continue
                    if direct_active:
                        with self._state_lock:
                            if self._source_return_state['activation_decision_index'] is None:
                                self._source_return_state['activation_decision_index'] = index
                row = {'index': index, 'observation_id': observation['observation_id'],
                       'policy_features': features, 'valid_history': mask, 'action': action,
                       'device_warning_raw': observation['device_diagnostics'].get('firmware_warning_raw'),
                       'device_warning_assessed': observation['device_diagnostics'].get('firmware_warning_raw') is not None,
                       'previous_applied_target_deg': list(applied) if confirmed_target else None,
                       'previous_student_target_deg': list(policy_previous_target),
                       'previous_control_reference_deg': list(applied),
                       'previous_control_reference_source': reference_source}
                result['decisions'].append(row)
                if self.controller.config.backend == 'sdk':
                    row['policy_observation_basis'] = policy_observation_basis
                if measured_motion is not None:
                    row['end_motion_policy_observation'] = measured_motion
                    row['dispatch_observation_id'] = observation['observation_id']
                if reconsidered_motion is not None:
                    row['superseded_motion_proposal'] = reconsidered_motion
                if return_active:
                    row['experimental_source_return'] = True
                    if self.enable_runtime_source_return:
                        context_scene, context_task = self._return_policy_context(before, task)
                        row['return_policy_context'] = {
                            'kind': 'projected_reverse_transfer', 'scene': [list(s) for s in context_scene],
                            'task': context_task, 'observed_object_layout': False,
                            'registration_changed': False}
                if direct_active:
                    row['direct_retarget_context'] = {
                        'kind': 'same_registered_scene_replacement_task',
                        'task': copy.deepcopy(task), 'observed_object_layout': False,
                        'registration_changed': False}
                if manual_early_action:
                    row['manual_early_source_return'] = {'student_proposed_action': original_return_action,
                                                         'executed_action': action, 'physical_success': None}
                # A source return changes the task presented to the student,
                # but hardware still executes that student's own joint goal.
                preserve_student_endpoint = self.controller.config.backend == 'sdk'
                if self._source_return_enabled:
                    with self._state_lock:
                        superseded = (self._source_return_state['pending']
                                      or self._source_return_state['admission_waiting']) and not return_active
                        if not superseded:
                            self._source_return_state['inflight_action'] = (
                                'move' if action == 'open' and preserve_student_endpoint else action)
                    if superseded:
                        self.model.policy.on_decision_not_sent()
                        history.pop()
                        row.update(result='superseded_before_send', transmitted=False)
                        continue
                if restored and index == start_index:
                    row['restored_pending_decision'] = True
                    row['restored_policy_previous_target_deg'] = saved['previous_applied_target_deg']
                if self._cancelled.is_set():
                    row['result'] = 'cancelled_before_send'
                    raise _Abort('cancelled', 'cancelled before sending a command')
                if action == 'done':
                    if self._source_return_enabled:
                        with self._state_lock:
                            pregrasp_pending = self._source_return_state['pregrasp_cancel_requested']
                        if pregrasp_pending and not return_active:
                            self.model.policy.on_decision_not_sent()
                            history.pop()
                            row['result'] = 'superseded_before_send'
                            continue
                    row['result'] = 'model_done_no_command'
                    result.update(termination='model_done', model_done=True)
                    break
                if direct_active and action == 'close':
                    row['result'] = 'direct_retarget_unexpected_close'
                    raise _Abort('retarget_regrasp_unexpected',
                                 'direct retarget cannot issue another CLOSE')
                if action == 'close' and self.enable_mock_layered_grasp_lowering:
                    if self._source_return_enabled:
                        with self._state_lock:
                            self._source_return_state['pregrasp_lowering'] = True
                            self._source_return_state['pregrasp_cancel_requested'] = False
                    progress = {'applied_count': 0, 'sent_or_unknown': False, 'in_flight': False}
                    try:
                        progress = self._mock_lower_before_close(
                            observation, task, slot_xy_mm, lease=lease, applied=applied,
                            confirmed_target=confirmed_target, reference_source=reference_source,
                            instance=instance, seen=seen, last_time=last_time,
                            last_sequence=last_sequence, progress=progress)
                    finally:
                        sent_or_unknown = sent_or_unknown or progress['sent_or_unknown']
                        in_flight = progress['in_flight']
                        result['applied_command_count'] += progress['applied_count']
                        if 'record' in progress:
                            row['manual_preclose'] = progress['record']
                    observation, applied = progress['observation'], progress['applied']
                    confirmed_target, reference_source = progress['confirmed_target'], progress['reference_source']
                    instance, last_time, last_sequence = progress['instance'], progress['last_time'], progress['last_sequence']
                    expected_joints = _joints(observation['joints_deg'])
                    nominal_applied = progress.get('nominal')
                    result.update(current_target_known=confirmed_target,
                                  last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                  current_control_reference_deg=list(applied),
                                  current_control_reference_source=reference_source,
                                  last_confirmed_nominal_target_deg=nominal_applied)
                if (action == 'open' and gripper == 'closed' and self.enable_release_setdown
                        and not preserve_student_endpoint):
                    if self._source_return_enabled:
                        with self._state_lock:
                            self._source_return_state['inflight_action'] = 'move'
                    progress = {'applied_count': 0, 'sent_or_unknown': False, 'in_flight': False,
                                'retained_move': retained_move}
                    if self.model.events:
                        row['release_policy_output'] = {key: copy.deepcopy(self.model.events[-1].get(key))
                            for key in ('raw_logits', 'raw_normalized_joint_output', 'previous_applied_target_deg')}
                    try:
                        progress = self._lower_before_open(observation, task, slot_xy_mm, grasp_reference,
                            returning=return_active, lease=lease, applied=applied, confirmed_target=confirmed_target,
                            reference_source=reference_source, instance=instance, seen=seen,
                            last_time=last_time, last_sequence=last_sequence, progress=progress)
                    finally:
                        sent_or_unknown = sent_or_unknown or progress['sent_or_unknown']
                        in_flight = progress['in_flight']
                        result['applied_command_count'] += progress['applied_count']
                        if 'record' in progress:
                            row['release_setdown'] = progress['record']
                    observation, applied = progress['observation'], progress['applied']
                    retained_move = progress['retained_move']
                    confirmed_target, reference_source = progress['confirmed_target'], progress['reference_source']
                    instance, last_time, last_sequence = progress['instance'], progress['last_time'], progress['last_sequence']
                    expected_joints = _joints(observation['joints_deg'])
                    if 'nominal_target_deg' in progress:
                        nominal_applied = list(progress['nominal_target_deg'])
                        result['last_confirmed_nominal_target_deg'] = list(nominal_applied)
                    result.update(current_target_known=confirmed_target,
                                  last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                  current_control_reference_deg=list(applied),
                                  current_control_reference_source=reference_source)
                    with self._state_lock:
                        superseded = (progress['preempted'] or (self._source_return_enabled and not return_active
                            and (self._source_return_state['pending'] or self._source_return_state['admission_waiting'])))
                        if self._source_return_enabled and not superseded:
                            self._source_return_state['inflight_action'] = 'open'
                    if superseded:
                        if progress['applied_count']:
                            self.model.policy.on_open_setdown_superseded(applied)
                        else:
                            self.model.policy.on_decision_not_sent()
                            history.pop()
                        row.update(result='open_superseded_during_setdown', transmitted=bool(progress['applied_count']),
                                   executor_action='move' if progress['applied_count'] else None,
                                   open_sent=False, control_reference_deg=list(applied),
                                   control_reference_source=reference_source)
                        continue
                if action == 'open' and preserve_student_endpoint:
                    row['open_execution'] = 'same_student_goal_once_then_open'
                closing_reference = list(observation['joints_deg']) if action == 'close' else None
                proposed = _joints(native_command_target(action, angles, policy_previous_target))
                if action == 'move' and preserve_student_endpoint:
                    self._require_original_student_endpoint(proposed)
                forced_lift = False
                lift_center_waypoint = None
                if manual_early_action and action == 'move':
                    config = self.controller.config
                    geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                        tool_vertical_mm=config.tool_vertical_mm)
                    source_xy = slot_xy_mm[task['source_slot'] - 1]
                    grip_z = (self.path_calibration.table_z_mm
                        + (task['source_level'] - 1) * self.path_calibration.cylinder_height_mm
                        + self.path_calibration.grip_offset_mm)
                    row['manual_early_source_return']['student_proposed_target_deg'] = list(proposed)
                    row['manual_early_source_return']['nominal_goal_mm'] = [*source_xy, grip_z]
                    proposed = _joints(geometry.inverse((*source_xy, grip_z)))
                    with self._state_lock:
                        self._source_return_state['lift_status'] = 'early_return_bypassed'
                if return_active and action == 'open':
                    with self._state_lock:
                        if not self._source_return_state['return_move_confirmed']:
                            row['result'] = 'source_return_open_before_move'
                            raise _Abort('source_return_incomplete', 'source return has no confirmed return move')
                    before_open = self._source_return_open_pose(observation, applied, task, slot_xy_mm)
                    row['source_return_open_pose'] = {'before': before_open}
                    with self._state_lock:
                        self._source_return_state['open_pose_before'] = before_open
                    if (not confirmed_target or not before_open['matches']) and not preserve_student_endpoint:
                        row['result'] = 'source_return_open_not_at_source'
                        raise _Abort('source_return_not_at_source', 'return OPEN target or observation is outside the nominal source')
                if (self.require_retarget_release_check or direct_active) and action == 'open' and gripper == 'closed':
                    release_pose = self._retarget_grip_pose(observation, applied, task, slot_xy_mm, release=True)
                    row['retarget_release_pose'] = {'before': release_pose}
                    if (not confirmed_target or not release_pose['matches']) and not preserve_student_endpoint:
                        row['result'] = 'retarget_open_not_at_destination'
                        raise _Abort('retarget_release_unconfirmed',
                                     'redirected OPEN is outside the registered nominal destination')
                if self.require_retarget_release_check and action == 'close' and gripper == 'open':
                    grip_pose = self._retarget_grip_pose(observation, applied, task, slot_xy_mm, release=False)
                    row['retarget_grasp_pose'] = {'before': grip_pose}
                    # A post-stop stationary reference is verified against a
                    # fresh observation at transfer start. It is a valid
                    # control reference even though it is not an applied goal.
                    if (not confirmed_target and reference_source != 'stationary_observation') or not grip_pose['matches']:
                        row['result'] = 'retarget_close_not_at_source'
                        raise _Abort('retarget_grasp_unconfirmed',
                                     'redirected CLOSE is outside the registered nominal source')
                if return_active and action == 'move' and not manual_early_action and not preserve_student_endpoint:
                    with self._state_lock:
                        needs_lift = self._source_return_state['lift_status'] == 'not_started'
                    if needs_lift:
                        config = self.controller.config
                        # Runtime re-planning starts from the latest stationary
                        # encoder observation, not an interrupted servo goal.
                        geometry, (x, y, z), lift_z, lift_source = self._source_return_lift_parameters(observation, applied)
                        if z < lift_z:
                            ceilings = []
                            if config.limits.workspace_max_mm is not None:
                                ceilings.append(config.limits.workspace_max_mm[2])
                            if (self.enable_experimental_segmented_path
                                    and self.path_calibration.path_ceiling_mm is not None):
                                ceilings.append(self.path_calibration.path_ceiling_mm)
                            ceiling = min(ceilings) if ceilings else None
                            if (ceiling is None or not _number(ceiling)
                                    or lift_z + _MOCK_SOURCE_RETURN_CEILING_MARGIN_MM > ceiling):
                                with self._state_lock:
                                    self._source_return_state['lift_status'] = 'insufficient_ceiling_margin'
                                row['result'] = 'source_return_lift_unavailable'
                                row['source_return_lift'] = {'from_z_mm': z, 'to_z_mm': lift_z,
                                    'ceiling_mm': ceiling, 'required_margin_mm': _MOCK_SOURCE_RETURN_CEILING_MARGIN_MM,
                                    'source': lift_source}
                                raise _Abort('source_return_lift_unavailable',
                                             'mock lift lacks the required nominal ceiling margin')
                            try:
                                lift_xy = (x, y)
                                if self.enable_runtime_source_return:
                                    centers = [slot_xy_mm[task[key] - 1]
                                               for key in ('source_slot', 'destination_slot')]
                                    center = min(centers, key=lambda value: math.dist((x, y), value))
                                    if math.dist((x, y), center) <= self.path_calibration.vertical_corridor_mm:
                                        lift_xy = tuple(center)
                                lift = _joints(geometry.inverse((*lift_xy, lift_z)))
                                # This is the executor's separately requested
                                # return lift, not a replacement student goal.
                                # Reach the registered corridor center while
                                # rising by the already configured clearance.
                                if self.enable_runtime_source_return and not _close(lift_xy, (x, y)):
                                    intermediate_z = min(z + self.path_calibration.clearance_mm, lift_z)
                                    if intermediate_z < lift_z:
                                        start_pitch = 90. + sum(observation['joints_deg'][1:])
                                        end_pitch = 90. + sum(lift[1:])
                                        fraction = (intermediate_z - z) / (lift_z - z)
                                        pitch = start_pitch + fraction * (end_pitch - start_pitch)
                                        lift_center_waypoint = _joints(
                                            geometry.inverse((*lift_xy, intermediate_z), pitch))
                            except Exception as exc:
                                with self._state_lock:
                                    self._source_return_state['lift_status'] = 'inverse_failed'
                                row['result'] = 'source_return_lift_unavailable'
                                raise _Abort('source_return_lift_unavailable', type(exc).__name__) from None
                            row['student_proposed_target_deg'] = list(proposed)
                            row['source_return_lift'] = {'from_z_mm': z, 'to_z_mm': lift_z,
                                'ceiling_mm': ceiling, 'nominal_margin_mm': ceiling - lift_z,
                                'source': lift_source}
                            if lift_center_waypoint is not None:
                                row['source_return_lift'].update(
                                    corridor_center_xy_mm=list(lift_xy),
                                    center_waypoint_z_mm=intermediate_z)
                            proposed = lift
                            forced_lift = True
                        else:
                            with self._state_lock:
                                self._source_return_state['lift_status'] = 'already_high'
                adjusted, height_record = proposed, None
                if (self.enable_runtime_source_return and return_active and action == 'move' and not forced_lift
                        and not preserve_student_endpoint):
                    adjusted, return_floor = self._source_return_grasp_floor(adjusted, task, slot_xy_mm)
                    if return_floor is not None:
                        row['source_return_grasp_height'] = return_floor
                if (self.enable_mock_nominal_trim and return_active and action == 'move'
                        and not forced_lift):
                    config = self.controller.config
                    geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                        tool_vertical_mm=config.tool_vertical_mm)
                    source_xy = slot_xy_mm[task['source_slot'] - 1]
                    source_height = (self.path_calibration.table_z_mm
                        + (task['source_level'] - 1) * self.path_calibration.cylinder_height_mm
                        + self.path_calibration.grip_offset_mm)
                    adjusted, source_floor = source_floor_target(
                        geometry, adjusted, source_xy, self.path_calibration.vertical_corridor_mm, source_height)
                    if source_floor is not None:
                        row['mock_source_nominal_floor'] = source_floor
                if height_guard is not None:
                    try:
                        adjusted, height_record = height_guard.apply(
                            action=action, proposed_target_deg=proposed, previous_target_deg=applied,
                            gripper_opening_m=0. if gripper == 'closed' else .04,
                            observed_tcp_mm=observation.get('tcp_mm'))
                    except Exception as exc:
                        row['result'] = 'support_guard_error_before_send'
                        raise _Abort('support_guard_error', type(exc).__name__) from None
                    if height_record is not None:
                        row['support_height_adjustment'] = height_record
                    if self._cancelled.is_set():
                        row['result'] = 'cancelled_before_send'
                        raise _Abort('cancelled', 'cancelled during support target preparation')
                if action == 'move':
                    requested = _joints(adjusted)
                    steps, effective = quantize_joints(requested)
                    effective = _joints(effective)
                    command = {'kind': 'move_joints', 'joints_deg': requested, 'speed_pct': self.speed_pct}
                    is_trajectory = False
                    row.update(proposed_target_deg=list(proposed), requested_target_deg=list(requested),
                               quantized_target_deg=list(effective))
                    if self.compare_virtual_descent or self.execute_mock_virtual_descent:
                        try:
                            path, comparison = self._virtual_descent_comparison(
                                before, task, slot_xy_mm, applied, requested, gripper,
                                start_source=('last_confirmed_applied_target' if confirmed_target
                                              else 'stationary_observation'))
                        except InterruptedError:
                            row['result'] = 'cancelled_before_send'
                            raise _Abort('cancelled', 'cancelled during virtual path comparison') from None
                        if self._cancelled.is_set():
                            row['result'] = 'cancelled_before_send'
                            raise _Abort('cancelled', 'cancelled after virtual path comparison')
                        if self.execute_mock_virtual_descent and path is not None:
                            is_trajectory = True
                            targets = [[math.degrees(value) for value in target] for target in path.targets_rad]
                            command = {'kind': 'mock_joint_trajectory', 'targets_deg': targets,
                                       'period_seconds': straight_descent.TIMESTEP}
                            effective, steps = list(requested), None
                            comparison.update(purpose='mock_execution_planned', applied_to_controller=False)
                        row['virtual_descent_comparison'] = comparison
                else:
                    is_trajectory = False
                    requested = effective = list(applied)
                    steps = None
                    command = {'kind': 'set_gripper', 'state': 'open' if action == 'open' else 'closed'}
                segments = [{'requested': requested, 'effective': effective}]
                split_reason, phase = None, None
                config = self.controller.config
                # The physical student chooses one endpoint per MOVE. The SDK
                # receives that goal directly; a geometric subdivision would
                # turn it into many different motor goals near an object.
                direct_student_move = action == 'move' and not is_trajectory and preserve_student_endpoint
                checked_move = (action == 'move' and not is_trajectory
                                and self.enable_experimental_segmented_path and not direct_student_move)
                if checked_move:
                    planning_started = time.monotonic()
                    try:
                        if lift_center_waypoint is not None:
                            _, center_effective = quantize_joints(lift_center_waypoint)
                            first, phase, first_reason = self._checked_move_segments(
                                observation, lift_center_waypoint, center_effective, task)
                            center_observation = {**observation, 'joints_deg': list(center_effective)}
                            second, second_phase, second_reason = self._checked_move_segments(
                                center_observation, requested, effective, task)
                            if phase != second_phase or phase.get('kind') != 'vertical':
                                raise _Abort('source_return_lift_unavailable',
                                             'the checked centering lift changed its vertical corridor')
                            segments, split_reason = first + second, first_reason or second_reason
                            if self.enable_blended_path and len(segments) > 64:
                                raise _Abort('source_return_lift_unavailable',
                                             'the checked centering lift exceeds the waypoint budget')
                        else:
                            segments, phase, split_reason = self._checked_move_segments(
                                observation, requested, effective, task)
                    except (ControlError, _Abort):
                        row['result'] = 'path_rejected_before_send'
                        if forced_lift:
                            with self._state_lock:
                                self._source_return_state['lift_status'] = 'path_rejected_before_send'
                        raise
                    row['path_phase'] = phase
                    row['path_preflight_seconds'] = time.monotonic() - planning_started
                    # Compaction can take longer than a sensor sample remains
                    # fresh. A stationary re-observation may refresh admission,
                    # while the first merged segment is checked again below.
                    if (self.path_calibration.compact_waypoints and
                            time.monotonic() - observation['observed_monotonic']
                            + observation['sensor_age_seconds'] >= config.limits.sensor_max_age * .5):
                        try:
                            observation = self.controller.observe(lease)
                        except Exception:
                            unknown = True
                            raise
                        instance, last_time, last_sequence = self._observation(
                            observation, instance, seen, last_time, last_sequence)
                        self._gripper(observation, gripper)
                        fresh_joints = _joints(observation['joints_deg'])
                        if not _close(fresh_joints, expected_joints, drift):
                            raise _Abort('state_changed', 'joints changed during path preparation')
                        expected_joints = fresh_joints
                        row['path_preflight_observation_refreshed'] = True
                # A single SDK position goal is safe only when the existing
                # independent-joint envelope already covers the entire
                # transport.  Vertical moves and any required subdivisions
                # retain their original waypoint execution.
                direct_transport = (checked_move and phase['kind'] == 'transport'
                                    and split_reason is None and len(segments) == 1
                                    and segments[0].get('path', {}).get('independent_joint_envelope', {})
                                        .get('independent_joint_progress') is True)
                allow_damped_compensation = False
                move_speed = (self.path_calibration.travel_speed_pct
                              if direct_transport and self.path_calibration.travel_speed_pct is not None
                              else self.speed_pct)
                if direct_student_move:
                    move_speed = self._direct_contact_speed_pct(
                        observation.get('joints_deg'), segments[-1]['effective'], move_speed)
                    command['speed_pct'] = move_speed
                    row['path_execution'] = {
                        'mode': 'single_joint_target', 'command_kind': 'move_joints',
                        'selection': 'original_student_target', 'speed_pct': move_speed,
                        'device_control': 'sdk_position_target',
                        'path_and_contact_verified': False,
                    }
                if checked_move:
                    mode = ('single_joint_target' if direct_transport else
                            'blended_C2' if self.enable_blended_path else 'stop_per_waypoint')
                    row['path_execution'] = {
                        'mode': mode,
                        'command_kind': 'blended_joint_path' if mode == 'blended_C2' else 'move_joints',
                        'selection': ('whole_transport_independent_joint_envelope' if direct_transport else
                                      'calibrated_waypoints'),
                        'speed_pct': move_speed,
                    }
                    if direct_transport:
                        row['path_execution']['preflight'] = copy.deepcopy(segments[0]['path'])
                        row['path_execution']['device_control'] = (
                            'sdk_position_target' if config.backend == 'sdk' else 'controller_position_target')
                if checked_move and self.enable_blended_path and not direct_transport:
                    joined_parts = segments
                    row['blended_path'] = {
                        'internal_waypoint_count': max(0, len(joined_parts) - 1),
                        'destination_waypoint_count': len(joined_parts),
                        'waypoints_deg': [list(part['effective']) for part in joined_parts],
                        'internal_stop_requested': False,
                        'model_endpoint_deg': list(effective),
                        'limits_are_virtual_scenario': True,
                        'max_velocity_deg_s': self.path_calibration.blended_max_velocity_deg_s,
                        'max_acceleration_deg_s2': 60.,
                        'max_jerk_deg_s3': 300., 'sample_period_seconds': .002}
                    segments = [{'requested': list(effective), 'effective': list(effective),
                                 'blend_parts': joined_parts,
                                 'path': {'joined_checks': [part['path'] for part in joined_parts]}}]
                if split_reason is not None:
                    row['segmentation'] = {
                        'reason': split_reason,
                        'planned_segment_count': len(segments),
                        'segments': [{'index': n, 'requested_target_deg': list(segment['requested']),
                                      'quantized_target_deg': list(segment['effective']),
                                      'preflight': segment['path'], 'result': 'planned'}
                                     for n, segment in enumerate(segments, 1)],
                    }
                applied_in_decision = 0
                damped_compensation_used = False
                damped_compensation_basis = None
                no_op_decision = False
                source_return_boundary = False
                interrupted_in_decision = False
                replan_deadline = None
                for segment_index, segment in enumerate(segments):
                    segment_observed_start = _joints(observation['joints_deg']) if action == 'move' else None
                    move_start_observation = copy.deepcopy(observation) if direct_student_move else None
                    move_attempts = []
                    previous_follow_up_remaining = None
                    follow_up_used = False
                    detail = (row['segmentation']['segments'][segment_index]
                              if split_reason is not None else None)
                    if cancelled():
                        row['result'] = 'cancelled_before_segment'
                        raise _Abort('cancelled', 'cancelled before sending the next segment')
                    if replan_deadline is not None and time.monotonic() >= replan_deadline:
                        raise _Abort('vertical_replan_timeout', 'mock replanned path exceeded its time budget')
                    if self._source_return_enabled and action == 'move' and not return_active:
                        with self._state_lock:
                            source_return_boundary = (self._source_return_state['pending']
                                                      or self._source_return_state['admission_waiting'])
                        if source_return_boundary:
                            break
                    segment_requested, segment_effective = segment['requested'], segment['effective']
                    nominal_segment = (list(segment_effective) if action == 'move' and
                                       (self.enable_mock_nominal_trim or self.enable_arrival_correction
                                        or config.backend == 'sdk') else None)
                    trim_record = None
                    if checked_move:
                        try:
                            if segment.get('blend_parts') is not None:
                                blended_command, blend_checks = self._blended_waypoint_command(
                                    observation, segment['blend_parts'], phase)
                                row['blended_path'].update(
                                    waypoints_deg=copy.deepcopy(blended_command['waypoints_deg']),
                                    destination_waypoint_count=len(blended_command['waypoints_deg']),
                                    internal_waypoint_count=max(0, len(blended_command['waypoints_deg']) - 1),
                                    **{key: blended_command[key] for key in (
                                        'max_velocity_deg_s', 'max_acceleration_deg_s2',
                                        'max_jerk_deg_s3', 'sample_period_seconds')})
                                actual_path = {'joined_checks': blend_checks, 'internal_stop_requested': False}
                            else:
                                actual_path = self._checked_joint_segment(
                                    _joints(observation['joints_deg']), segment_effective, phase, observed_start=True)
                                if direct_transport:
                                    if (actual_path.get('independent_joint_envelope', {})
                                            .get('independent_joint_progress') is not True):
                                        raise _Abort('path_evidence_unavailable',
                                                     'independent joint transport evidence is unavailable')
                                    row['path_execution']['observed_start_check'] = copy.deepcopy(actual_path)
                        except (ControlError, _Abort) as exc:
                            may_split = (self.enable_mock_nominal_trim and return_active
                                         and phase.get('kind') == 'vertical'
                                         and isinstance(exc, ControlError)
                                         and exc.code in {'ASYNC_ENVELOPE', 'VERTICAL_CORRIDOR'})
                            if not may_split:
                                row['result'] = 'segment_rejected_before_send'
                                if forced_lift:
                                    with self._state_lock:
                                        self._source_return_state['lift_status'] = ('path_rejected_after_partial_lift'
                                            if applied_in_decision else 'path_rejected_before_send')
                                if detail is not None:
                                    detail['result'] = 'rejected_before_send'
                                raise
                            actual_start = _joints(observation['joints_deg'])
                            geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                                tool_vertical_mm=config.tool_vertical_mm)
                            final_xyz = geometry.forward(segment_effective)
                            endpoint_geometry = _EndpointPreservingGeometry(
                                geometry, final_xyz, segment_requested, actual_start)
                            replacements = vertical_segments(
                                endpoint_geometry, actual_start, final_xyz,
                                lambda start, end: self._checked_joint_segment(start, end, phase, observed_start=False),
                                allow_async_subdivision=True, compact=False)
                            if (len(replacements) < 2 or len(segments) - 1 + len(replacements) > 64
                                    or len(row.get('mock_observed_resplits', [])) >= 8):
                                raise _Abort('observed_resplit_unavailable', 'bounded checked resplitting is unavailable')
                            segments[segment_index:segment_index + 1] = replacements
                            entries = [{'index': 0, 'requested_target_deg': list(part['requested']),
                                        'quantized_target_deg': list(part['effective']),
                                        'preflight': part['path'], 'result': 'planned'} for part in replacements]
                            if split_reason is None:
                                split_reason = exc.as_dict()
                                row['segmentation'] = {'reason': split_reason, 'segments': entries}
                            else:
                                row['segmentation']['segments'][segment_index:segment_index + 1] = entries
                            for number, entry in enumerate(row['segmentation']['segments'], 1):
                                entry['index'] = number
                            row['segmentation']['planned_segment_count'] = len(segments)
                            row.setdefault('mock_observed_resplits', []).append({
                                'observed_start_deg': actual_start, 'original_target_deg': list(segment_effective),
                                'reason': exc.as_dict(), 'replacement_count': len(replacements)})
                            segment = replacements[0]
                            segment_requested, segment_effective = segment['requested'], segment['effective']
                            nominal_segment = list(segment_effective)
                            detail = row['segmentation']['segments'][segment_index]
                            actual_path = self._checked_joint_segment(
                                actual_start, segment_effective, phase, observed_start=True)
                        if detail is not None:
                            detail['observed_start_deg'] = list(observation['joints_deg'])
                            detail['observed_start_check'] = actual_path
                        command = {'kind': 'move_joints', 'joints_deg': list(segment_requested),
                                   'speed_pct': move_speed,
                                   'path_constraints': {key: value for key, value in phase.items() if key != 'kind'}}
                        if segment.get('blend_parts') is not None:
                            command = blended_command
                            if config.backend == 'sdk':
                                command = self._stationary_blend_as_single(command, observation)
                                if command['kind'] == 'move_joints':
                                    row['path_execution'].update(
                                        mode='single_joint_target', command_kind='move_joints',
                                        selection='stationary_blend_single_target')
                        student_progress = not forced_lift and not manual_early_action
                        if (self.path_calibration.waypoint_tolerance_mm is not None
                                and (student_progress or (forced_lift and self.enable_runtime_source_return)
                                     or segment_index + 1 < len(segments))):
                            command['arrival_tolerance_mm'] = self.path_calibration.waypoint_tolerance_mm
                        row['path_execution']['arrival_tolerance_mm'] = command.get(
                            'arrival_tolerance_mm', config.limits.tcp_tolerance_mm)
                        if self.enable_arrival_correction:
                            command.update(nominal_arrival_joints_deg=list(nominal_segment),
                                           allow_nominal_correction=True)
                        if self.enable_mock_nominal_trim:
                            measured_joints = _joints(observation['joints_deg'])
                            bias_reference = list(applied) if confirmed_target else measured_joints
                            try:
                                proposed_trim, fallback = validated_bias_target(
                                    nominal_segment, bias_reference, measured_joints,
                                    lambda start, end: self._checked_joint_segment(start, end, phase, observed_start=True))
                                if (return_active and not self._mock_source_return_target_allowed(
                                        nominal_segment, proposed_trim, task, slot_xy_mm)):
                                    if not self._mock_source_return_target_allowed(
                                            nominal_segment, nominal_segment, task, slot_xy_mm):
                                        raise _Abort('source_return_target_out_of_bounds',
                                                     'nominal source target is outside the existing release bounds')
                                    self._checked_joint_segment(measured_joints, nominal_segment, phase, observed_start=True)
                                    proposed_trim, fallback = tuple(nominal_segment), 'source_return_release_bounds'
                            except (ControlError, ValueError, _Abort):
                                row['result'] = 'nominal_trim_rejected_before_send'
                                raise
                            segment_requested = _joints(proposed_trim)
                            _, segment_effective = quantize_joints(segment_requested)
                            segment_effective = list(segment_effective)
                            command.update(joints_deg=list(segment_requested),
                                           nominal_arrival_joints_deg=list(nominal_segment),
                                           allow_mock_nominal_retry=not segment.get('mock_replanned_no_retry', False))
                            trim_record = {'nominal_target_deg': list(nominal_segment),
                                           'previous_sent_target_deg': list(applied) if confirmed_target else None,
                                           'previous_reference_source': reference_source,
                                           'observed_before_deg': measured_joints,
                                           'first_requested_target_deg': list(segment_requested),
                                           'first_quantized_target_deg': list(segment_effective),
                                           'move_release_upper_separation': self.enable_mock_move_release_upper_separation,
                                           'fallback_reason': fallback, 'retry': None}
                            row.setdefault('mock_nominal_waypoints', []).append(trim_record)
                    if action == 'move' and not is_trajectory:
                        segment_steps, encoded = quantize_joints(segment_requested)
                        if not _close(encoded, segment_effective):
                            raise _Abort('segment_endpoint_mismatch', 'segment quantization changed')
                        if preserve_student_endpoint and segment_index + 1 == len(segments):
                            sent_goal = (command['waypoints_deg'][-1]
                                         if command.get('kind') == 'blended_joint_path'
                                         else command.get('joints_deg'))
                            _, sent_effective = quantize_joints(_joints(sent_goal))
                            if not _close(sent_effective, effective):
                                raise _Abort('segment_endpoint_mismatch',
                                             'the final physical command changed the student endpoint')
                    else:
                        segment_steps = None
                    retain_move_requested = action == 'move' and config.backend == 'sdk'
                    if retain_move_requested:
                        command['completion_mode'] = 'retain_position_target'
                        command['nominal_arrival_joints_deg'] = list(nominal_segment or segment_effective)
                        if direct_student_move:
                            command['direct_student_target'] = True
                        # Hardware moves, including a student-led source
                        # return, retain the encoded endpoint without a
                        # follow-up correction.
                        nominal_steps, _ = quantize_joints(nominal_segment or segment_effective)
                        allow_damped_compensation = (not preserve_student_endpoint
                            and segment_index + 1 == len(segments)
                            and list(segment_steps) == list(nominal_steps)
                            and command.get('correction_of_command_id') is None)
                        if allow_damped_compensation:
                            command['allow_nominal_correction'] = True
                            command['allow_damped_compensation'] = True
                        if preserve_student_endpoint:
                            # The common and blended-path builders can opt in
                            # to corrections. Remove both opts for every
                            # SDK segment, including a later gripper hold.
                            command.pop('allow_nominal_correction', None)
                            command.pop('allow_damped_compensation', None)
                            command.pop('allow_target_follow_up', None)
                    elif action != 'move' and config.backend == 'sdk' and retained_move is not None:
                        command['retain_position_from_command_id'] = retained_move['command_id']
                    # The controller validates this exact observation and revision.
                    observation_age = time.monotonic() - observation['observed_monotonic']
                    if detail is not None:
                        detail['observation_age_before_send_seconds'] = observation_age
                    if (observation_age > config.limits.observation_max_age
                            or (checked_move and observation['sensor_age_seconds'] + observation_age
                                > config.limits.sensor_max_age)):
                        row['result'] = 'observation_expired_before_segment'
                        if detail is not None:
                            detail['result'] = 'observation_expired_before_send'
                        raise _Abort('observation_expired', 'observation expired during path preparation')
                    if cancelled():
                        row['result'] = 'cancelled_before_segment'
                        raise _Abort('cancelled', 'cancelled before sending the next segment')
                    if self._source_return_enabled and action == 'move' and not return_active:
                        with self._state_lock:
                            source_return_boundary = (self._source_return_state['pending']
                                                      or self._source_return_state['admission_waiting'])
                        if source_return_boundary:
                            break
                    command_id = uuid.uuid4().hex
                    row.update(command_id=command_id, result='command_outcome_unknown')
                    if detail is not None:
                        detail.update(command_id=command_id, observation_id=observation['observation_id'],
                                      result='command_outcome_unknown')
                    request = {'lease_id': lease, 'command_id': command_id,
                               'observation_id': observation['observation_id'], 'action': command}
                    if (checked_move and len(segments) > 1 and segment.get('blend_parts') is None
                            and not self.enable_mock_nominal_trim):
                        self._publish_segment_plan(command_id, segment_index, segments)
                    before_command_sequence = observation['sensor_sequence']
                    if self._source_return_enabled and action == 'move' and not return_active:
                        def before_segment_send():
                            nonlocal in_flight
                            if cancelled():
                                row['result'] = 'cancelled_before_segment'
                                raise _Abort('cancelled', 'cancelled before sending the next segment')
                            in_flight = True

                        source_return_boundary, response = self._dispatch_guarded_action(
                            request, command_id=command_id, return_active=return_active,
                            on_before_send=before_segment_send)
                        if source_return_boundary:
                            break
                    elif action == 'open' and preserve_student_endpoint:
                        if (last_student_move is None or retained_move is None
                                or last_student_move['task'] != task
                                or last_student_move['direct_active'] != direct_active
                                or last_student_move['return_active'] != return_active
                                or last_student_move['target_steps'] != retained_move['target_steps']):
                            row['result'] = 'open_without_confirmed_student_goal'
                            raise _Abort('target_unconfirmed', 'OPEN has no confirmed current student MOVE')
                        repeat_target = list(last_student_move['requested_target_deg'])
                        repeat_steps, repeat_effective = quantize_joints(repeat_target)
                        repeat_nominal = list(retained_move['nominal_target_deg'])
                        if (list(repeat_steps) != retained_move['target_steps']
                                or list(quantize_joints(repeat_nominal)[0]) != list(repeat_steps)):
                            raise _Abort('target_unconfirmed', 'the last student target is no longer retained')
                        repeat_id = uuid.uuid4().hex
                        repeat_action = {'kind': 'move_joints', 'joints_deg': repeat_target,
                            'nominal_arrival_joints_deg': repeat_nominal,
                            'speed_pct': retained_move['speed_pct'], 'completion_mode': 'retain_position_target',
                            'allow_target_follow_up': True, 'direct_student_target': True}
                        if retained_move.get('arrival_tolerance_mm') is not None:
                            repeat_action['arrival_tolerance_mm'] = retained_move['arrival_tolerance_mm']
                        repeat_record = row['pre_open_same_goal'] = {
                            'command_id': repeat_id, 'student_move_command_id': retained_move['command_id'],
                            'requested_target_deg': list(repeat_target), 'result': 'not_sent',
                            'physical_success': None}

                        def before_pre_open_send():
                            nonlocal in_flight
                            if cancelled():
                                raise _Abort('cancelled', 'cancelled before the pre-OPEN target')
                            repeat_record['result'] = 'outcome_unknown'
                            in_flight = True

                        source_return_boundary, repeat_response = self._dispatch_guarded_action(
                            {'lease_id': lease, 'command_id': repeat_id,
                             'observation_id': observation['observation_id'], 'action': repeat_action},
                            command_id=repeat_id, return_active=return_active,
                            on_before_send=before_pre_open_send)
                        if source_return_boundary:
                            break
                        if cancelled():
                            raise _Abort('cancelled', 'cancelled during the pre-OPEN target')
                        if (not isinstance(repeat_response, dict)
                                or repeat_response.get('command_id') != repeat_id
                                or repeat_response.get('replayed') is not False
                                or repeat_response.get('applied') is not True
                                or repeat_response.get('status') not in ('completed', 'settled')):
                            raise _Abort('command_unconfirmed', 'the pre-OPEN target outcome is unconfirmed')
                        self._mock_nominal_response(repeat_response, repeat_target, repeat_effective,
                                                    repeat_steps, repeat_nominal)
                        repeat_after = repeat_response.get('observation')
                        instance, last_time, last_sequence = self._observation(
                            repeat_after, instance, seen, last_time, last_sequence, require_new_sample=True)
                        self._gripper(repeat_after, gripper)
                        repeat_retained = self._retained_move_response(
                            repeat_response, repeat_steps, repeat_action['speed_pct'],
                            observation['sensor_sequence'], repeat_after)
                        remember_student_host_frame(repeat_steps, repeat_id,
                            repeat_response['completion']['host_frame'], repeat_response.get('receipt'))
                        repeat_retained.update(nominal_target_deg=repeat_nominal,
                            speed_pct=repeat_action['speed_pct'], path_constraints={},
                            arrival_tolerance_mm=retained_move.get('arrival_tolerance_mm'),
                            compensation_allowed=False, compensation_used=False, checked_phase=None)
                        retained_move = repeat_retained
                        expected_joints = _joints(repeat_after['joints_deg'])
                        observation = repeat_after
                        sent_reached = self._target_reached_flag(
                            repeat_response.get('sent_target_reached'), expected_joints, repeat_effective,
                            allowed_tcp_mm=repeat_action.get('arrival_tolerance_mm'))
                        confirmed_target = (repeat_response.get('nominal_arrival_reached') is True
                                            and sent_reached is True)
                        applied = list(repeat_effective if confirmed_target else expected_joints)
                        segment_effective = list(applied)
                        reference_source = ('controller_confirmed_move' if confirmed_target
                                            else 'stationary_observation_unreached_target')
                        result.update(current_target_known=confirmed_target,
                            last_confirmed_target_deg=list(applied) if confirmed_target else None,
                            current_control_reference_deg=list(applied),
                            current_control_reference_source=reference_source)
                        result['applied_command_count'] += 1
                        repeat_record.update(result=repeat_response['status'],
                            nominal_arrival_reached=repeat_response['nominal_arrival_reached'],
                            observed_joints_deg=list(expected_joints),
                            completion=copy.deepcopy(repeat_response['completion']))
                        in_flight, sent_or_unknown = False, True
                        with self._state_lock:
                            pending_return = self._has_pending_retarget_locked(return_active=return_active)
                            if self._source_return_enabled and not pending_return:
                                self._source_return_state['inflight_action'] = 'open'
                        if cancelled():
                            raise _Abort('cancelled', 'cancelled after the pre-OPEN target')
                        if pending_return:
                            source_return_boundary = True
                            break
                        command['retain_position_from_command_id'] = repeat_id
                        request['observation_id'] = observation['observation_id']
                        before_command_sequence = observation['sensor_sequence']
                        in_flight = True
                        response = self.controller.apply_action(request)
                    elif action == 'close' and self._source_return_enabled:
                        with self._mock_move_dispatch_lock:
                            with self._state_lock:
                                source_return_boundary = (self._source_return_state['pregrasp_cancel_requested']
                                                          and not self._cancelled.is_set())
                                if self._cancelled.is_set():
                                    raise _Abort('cancelled', 'pregrasp was cancelled before CLOSE')
                                if not source_return_boundary:
                                    self._source_return_state['pregrasp_lowering'] = False
                                    self._source_return_state['close_dispatch_started'] = True
                                    self._source_return_state['preclose_observed_joints_deg'] = list(observation['joints_deg'])
                            if source_return_boundary:
                                break
                            in_flight = True
                            response = self.controller.apply_action(request)
                    else:
                        in_flight = True
                        response = self.controller.apply_action(request)
                    if (not isinstance(response, dict) or response.get('command_id') != command_id
                            or response.get('replayed') is not False):
                        raise _Abort('reply_mismatch', 'the command response cannot be confirmed')
                    if direct_student_move:
                        move_attempts.append(copy.deepcopy(response))
                    status = response.get('status')
                    if action != 'move' and response.get('applied') is True:
                        # A later arm failure must not erase the fact that the
                        # gripper command was already transmitted or reported.
                        row['gripper_delivery'] = {
                            'command_id': command_id, 'state': command['state'], 'transmitted': True,
                            'reported_confirmed': response.get('gripper_reported_confirmed') is True,
                            'response_status': status}
                        if status == 'failed':
                            row['gripper_delivery']['controller_error'] = copy.deepcopy(response.get('error'))
                            failed_observation = response.get('observation')
                            if isinstance(failed_observation, dict):
                                row['gripper_delivery']['failure_observation_unverified'] = {
                                    key: copy.deepcopy(failed_observation.get(key)) for key in
                                    ('observation_id', 'sensor_sequence', 'sensor_valid', 'moving',
                                     'gripper_command', 'gripper_reported_state', 'gripper_reported_valid',
                                     'gripper_feedback_fresh', 'joints_deg', 'tcp_mm')}
                                failed_state = failed_observation.get('state')
                                if (response.get('gripper_command') == command['state']
                                        and response.get('gripper_reported_confirmed') is True
                                        and failed_state in ('armed', 'stopped', 'fault')):
                                    try:
                                        verified = self._observation(
                                            failed_observation, instance, seen, last_time, last_sequence,
                                            require_new_sample=True, expected_state=failed_state)
                                        self._gripper(failed_observation, command['state'])
                                    except _Abort:
                                        row['gripper_delivery']['observation_verified'] = False
                                    else:
                                        instance, last_time, last_sequence = verified
                                        gripper = command['state']
                                        result['last_confirmed_gripper_command'] = gripper
                                        row['gripper_delivery']['observation_verified'] = True
                    if command.get('kind') == 'blended_joint_path' and response.get('applied') is True:
                        if status in ('completed', 'settled_needs_nominal_correction',
                                      'settled_needs_target_follow_up', 'settled', 'cancelled'):
                            row['blended_path']['execution'] = self._blended_response(response, segment_effective)
                    while status in ('settled', 'settled_no_motion') and preserve_student_endpoint and not follow_up_used:
                        follow_up_used = True
                        # The first command has a complete reply and a fresh,
                        # stationary observation. Reissue exactly that student
                        # target once; never infer a reply after a timeout.
                        if (response.get('applied') is not True or response.get('nominal_arrival_reached') is not False
                                or response.get('stop', {}).get('requested') is not False):
                            raise _Abort('command_unconfirmed', 'target follow-up lacks a confirmed completed command')
                        self._mock_nominal_response(response, segment_requested, segment_effective,
                                                    segment_steps, nominal_segment)
                        after = response.get('observation')
                        instance, last_time, last_sequence = self._observation(
                            after, instance, seen, last_time, last_sequence, require_new_sample=True)
                        self._gripper(after, gripper)
                        # A settled receipt alone is not evidence of an SDK
                        # frame if a retarget interrupts before the retry.
                        self._retained_move_response(
                            response, segment_steps, command['speed_pct'], before_command_sequence, after)
                        if direct_student_move:
                            remember_student_host_frame(segment_steps, command_id,
                                response['completion']['host_frame'], response.get('receipt'))
                        retained_move = None
                        applied_in_decision += 1
                        result['applied_command_count'] += 1
                        sent_or_unknown, in_flight = True, False
                        # The SDK encoded the student's target, although its
                        # arrival is not confirmed. Preserve that history if a
                        # later retarget interrupts the follow-up.
                        applied, confirmed_target = list(segment_effective), False
                        observation = after
                        expected_joints = _joints(after['joints_deg'])
                        reference_source = 'stationary_observation_unreached_target'
                        result.update(current_target_known=False, last_confirmed_target_deg=None,
                                      current_control_reference_deg=list(expected_joints),
                                      current_control_reference_source=reference_source)
                        target_steps, _ = quantize_joints(nominal_segment)
                        observed_steps, _ = quantize_joints(expected_joints)
                        remaining = sum(abs(target - observed) for target, observed
                                        in zip(target_steps, observed_steps))
                        stalled = (previous_follow_up_remaining is not None
                                   and remaining >= previous_follow_up_remaining)
                        previous_follow_up_remaining = remaining
                        original_command_id = command_id
                        follow_id = uuid.uuid4().hex
                        follow_command = {'kind': 'move_joints', 'joints_deg': list(segment_requested),
                                          'nominal_arrival_joints_deg': list(nominal_segment),
                                          'speed_pct': command['speed_pct'],
                                          'path_constraints': copy.deepcopy(command.get('path_constraints', {})),
                                          'completion_mode': 'retain_position_target',
                                          'allow_target_follow_up': True}
                        if direct_student_move:
                            follow_command['direct_student_target'] = True
                        if 'arrival_tolerance_mm' in command:
                            follow_command['arrival_tolerance_mm'] = command['arrival_tolerance_mm']
                        row.setdefault('target_follow_ups', []).append({
                            'original_command_id': original_command_id, 'command_id': follow_id,
                            'target_deg': list(segment_effective), 'result': 'not_sent',
                            'first_observed_joints_deg': list(expected_joints),
                            'first_nominal_arrival_reached': False,
                            'first_arrival_joint_error_deg': response.get('arrival_joint_error_deg'),
                            'first_arrival_tcp_error_mm': response.get('arrival_tcp_error_mm')})
                        before_command_sequence = after['sensor_sequence']
                        request = {'lease_id': lease, 'command_id': follow_id,
                                   'observation_id': after['observation_id'], 'action': follow_command}
                        def cancel_before_follow_pending():
                            if cancelled():
                                raise _Abort('cancelled', 'cancelled before target follow-up')

                        def on_pending_follow():
                            nonlocal observation, expected_joints, instance, last_time, last_sequence
                            nonlocal applied, confirmed_target, reference_source, interrupted_in_decision
                            # The old target is still retained. Use the
                            # existing interrupt/re-arm path before handing
                            # the changed task back to the policy loop.
                            stop, measured, observation, expected_joints, instance, last_time, last_sequence = (
                                self._preempt_retained_before_send(
                                    lease, follow_id, instance=instance, seen=seen,
                                    last_time=last_time, last_sequence=last_sequence,
                                    gripper=gripper, label='follow-up'))
                            self.model.on_control_applied('move', list(segment_effective),
                                                          list(segment_effective), None)
                            row['student_history_after_interruption'] = {
                                'disposition': 'sdk_host_encoded_not_physical_arrival',
                                'previous_target_deg': list(self.model.policy.current_target),
                                'previous_action': self.model.policy.previous_action,
                                'interrupted_target_committed': True,
                                'target_arrival_confirmed': False,
                                'stopped_observation_is_applied_target': False}
                            applied, confirmed_target = list(measured), False
                            reference_source = 'stationary_observation_before_retarget'
                            row['target_follow_ups'][-1].update(result='superseded_before_send',
                                                                preemption_stop=copy.deepcopy(stop))
                            row.update(result='retarget_before_target_follow_up', transmitted=True)
                            result.update(current_target_known=False, last_confirmed_target_deg=None,
                                          current_control_reference_deg=list(measured),
                                          current_control_reference_source=reference_source)
                            interrupted_in_decision = True

                        def before_follow_send():
                            nonlocal in_flight
                            if stalled:
                                row['target_follow_ups'][-1]['result'] = 'not_sent_no_progress'
                                raise _Abort('target_unconfirmed', 'the target follow-up stopped making progress')
                            if cancelled():
                                raise _Abort('cancelled', 'cancelled before target follow-up dispatch')
                            in_flight = True
                            row['target_follow_ups'][-1]['result'] = 'outcome_unknown'

                        source_return_boundary, response = self._dispatch_guarded_action(
                            request, command_id=follow_id, return_active=return_active,
                            include_pregrasp=True, on_before_pending=cancel_before_follow_pending,
                            on_pending=on_pending_follow, on_before_send=before_follow_send)
                        if source_return_boundary:
                            break
                        command_id, command = follow_id, follow_command
                        row.update(command_id=command_id, result='target_follow_up_outcome_unknown')
                        if (not isinstance(response, dict) or response.get('command_id') != command_id
                                or response.get('replayed') is not False):
                            raise _Abort('reply_mismatch', 'target follow-up response cannot be confirmed')
                        if direct_student_move:
                            move_attempts.append(copy.deepcopy(response))
                        status = response.get('status')
                        row['target_follow_ups'][-1]['result'] = status
                    if source_return_boundary:
                        break
                    if follow_up_used and cancelled():
                        raise _Abort('cancelled', 'cancelled after target follow-up')
                    if (direct_student_move and follow_up_used and status in ('settled', 'settled_no_motion')
                            and response.get('applied') is True):
                        after_retry = response.get('observation')
                        observed_retry = _joints(after_retry.get('joints_deg') if isinstance(after_retry, dict) else None)
                        no_progress = (max(abs(a-b) for a, b in zip(observed_retry, segment_observed_start))
                                       <= config.limits.joint_tolerance_deg)
                        exactly_at_target = tuple(quantize_joints(observed_retry)[0]) == tuple(segment_steps)
                        if no_progress and not exactly_at_target:
                            self._mock_nominal_response(response, segment_requested, segment_effective,
                                                        segment_steps, nominal_segment)
                            instance, last_time, last_sequence = self._observation(
                                after_retry, instance, seen, last_time, last_sequence, require_new_sample=True)
                            self._gripper(after_retry, gripper)
                            self._retained_move_response(response, segment_steps, command['speed_pct'],
                                                         before_command_sequence, after_retry)
                            remember_student_host_frame(segment_steps, command_id,
                                response['completion']['host_frame'], response.get('receipt'))
                            result['applied_command_count'] += 1
                            result.update(current_target_known=False, last_confirmed_target_deg=None,
                                          current_control_reference_deg=list(observed_retry),
                                          current_control_reference_source='stationary_observation_unreached_target')
                            row.update(result='motion_not_observed_after_follow_up', transmitted=True,
                                       nominal_arrival_confirmed=False,
                                       observed_start_deg=list(segment_observed_start),
                                       observed_after_retry_deg=list(observed_retry),
                                       sdk_host_encoded_target_deg=list(segment_effective))
                            remember_no_motion(move_start_observation, move_attempts)
                            in_flight, sent_or_unknown = False, True
                            raise _Abort('motion_not_observed',
                                         'no meaningful joint movement was observed after the same target was sent twice')
                    if (self.enable_arrival_correction or allow_damped_compensation) and status == 'settled_needs_nominal_correction':
                        if retain_move_requested:
                            retained_move = None
                        self._mock_nominal_response(response, segment_requested, segment_effective,
                                                    segment_steps, nominal_segment)
                        if (response.get('applied') is not True
                                or response.get('nominal_correction_permitted') is not True
                                or response.get('nominal_arrival_reached') is not False
                                or (not allow_damped_compensation and response.get('stop', {}).get('confirmed') is not True)
                                or (allow_damped_compensation and (response.get('stop', {}).get('requested') is not False
                                                                   or response.get('stop', {}).get('confirmed') is not False))):
                            raise _Abort('command_unconfirmed', 'arrival correction lacks the required feedback')
                        correction_observation = response.get('observation')
                        instance, last_time, last_sequence = self._observation(
                            correction_observation, instance, seen, last_time, last_sequence, require_new_sample=True)
                        self._gripper(correction_observation, gripper)
                        measured = _joints(correction_observation['joints_deg'])
                        applied, confirmed_target = list(segment_effective), True
                        reference_source = 'controller_confirmed_sent_target'
                        result.update(current_target_known=True, last_confirmed_target_deg=list(applied),
                                      current_control_reference_deg=list(applied),
                                      current_control_reference_source=reference_source)
                        row['nominal_arrival_confirmed'] = False
                        applied_in_decision += 1
                        result['applied_command_count'] += 1
                        in_flight = False
                        sent_or_unknown = True
                        observation = correction_observation
                        expected_joints = measured
                        if preserve_student_endpoint:
                            geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                                tool_vertical_mm=config.tool_vertical_mm)
                            row.setdefault('arrival_observations', []).append({
                                'command_id': command_id, 'observation_id': correction_observation['observation_id'],
                                'observed_joints_deg': list(measured),
                                'nominal_target_deg': list(nominal_segment),
                                'nominal_joint_error_deg': max(abs(a-b) for a, b in zip(measured, nominal_segment)),
                                'nominal_tcp_error_mm': math.dist(geometry.forward(measured),
                                                                  geometry.forward(nominal_segment)),
                                'arrival_tolerance_mm': command.get('arrival_tolerance_mm', config.limits.tcp_tolerance_mm),
                                'correction': False})
                            if not allow_damped_compensation:
                                row.update(result='nominal_target_unconfirmed', transmitted=True,
                                           applied_target_deg=list(applied), alternate_correction_sent=False,
                                           control_reference_deg=list(applied), control_reference_source=reference_source)
                                raise _Abort('target_unconfirmed',
                                             'the student endpoint was not reached; no replacement goal was sent')
                        correction = response.get('correction_action')
                        if (not isinstance(correction, dict) or correction.get('kind') != 'move_joints'
                                or correction.get('correction_of_command_id') != command_id
                                or correction.get('allow_nominal_correction', False) is not False
                                or not _close(_joints(correction.get('nominal_arrival_joints_deg')), nominal_segment)
                                or correction.get('speed_pct') != command['speed_pct']
                                or correction.get('path_constraints') != {
                                    key: list(value) if isinstance(value, tuple) else value
                                    for key, value in command.get('path_constraints', {}).items()}
                                or correction.get('arrival_tolerance_mm') != command.get('arrival_tolerance_mm')):
                            raise _Abort('reply_mismatch', 'arrival correction differs from the original path')
                        correction_requested = _joints(correction.get('joints_deg'))
                        correction_steps, correction_effective = quantize_joints(correction_requested)
                        if max(abs(a-b) for a,b in zip(correction_effective, nominal_segment)) > 1.0:
                            raise _Abort('nominal_correction_out_of_range', 'arrival correction exceeds one degree')
                        if checked_move:
                            self._checked_joint_segment(measured, correction_effective, phase, observed_start=True)
                        if allow_damped_compensation:
                            damped_compensation_basis = self._damped_correction_basis(
                                response, correction, command_id, nominal_segment, measured, correction_observation,
                                command['speed_pct'], before_command_sequence, command.get('path_constraints', {}))
                            damped_compensation_used = True
                        correction_id = uuid.uuid4().hex
                        if retain_move_requested:
                            correction['completion_mode'] = 'retain_position_target'
                        correction_geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                                       tool_vertical_mm=config.tool_vertical_mm)
                        row.setdefault('arrival_corrections', []).append({
                            'original_command_id': command_id, 'command_id': correction_id,
                            'nominal_target_deg': list(nominal_segment),
                            'first_sent_target_deg': list(applied),
                            ('basis_observed_joints_deg' if damped_compensation_used else 'stopped_joints_deg'): measured,
                            'first_nominal_tcp_error_mm': math.dist(correction_geometry.forward(measured),
                                                                    correction_geometry.forward(nominal_segment)),
                            'requested_target_deg': correction_requested,
                            'effective_target_deg': list(correction_effective), 'result': 'outcome_unknown'})
                        if damped_compensation_used:
                            row['arrival_corrections'][-1]['damped_compensation_basis'] = copy.deepcopy(
                                damped_compensation_basis)
                            row['arrival_corrections'][-1]['retained_parent'] = copy.deepcopy(
                                response['retained_parent'])
                        request = {'lease_id': lease, 'command_id': correction_id,
                                   'observation_id': correction_observation['observation_id'], 'action': correction}
                        before_command_sequence = correction_observation['sensor_sequence']
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'cancelled before arrival correction')
                        def on_pending_correction():
                            nonlocal observation, expected_joints, instance, last_time, last_sequence, measured
                            nonlocal applied, confirmed_target, reference_source, interrupted_in_decision
                            if damped_compensation_used and self.enable_runtime_source_return:
                                # The nominal target is still held here.
                                # Reserve the unsent correction ID so the
                                # existing interrupt/resume path performs
                                # a real stop before the return is planned.
                                stop, measured, observation, expected_joints, instance, last_time, last_sequence = (
                                    self._preempt_retained_before_send(
                                        lease, correction_id, instance=instance, seen=seen,
                                        last_time=last_time, last_sequence=last_sequence,
                                        gripper=gripper, label='compensation'))
                                row['arrival_corrections'][-1]['preemption_stop'] = copy.deepcopy(stop)
                            self.model.policy.on_move_interrupted(measured)
                            applied, confirmed_target = list(measured), False
                            reference_source = 'stationary_observation_before_retarget'
                            row['arrival_corrections'][-1]['result'] = 'superseded_before_send'
                            row.update(result='retarget_before_arrival_correction', transmitted=True)
                            result.update(current_target_known=False, last_confirmed_target_deg=None,
                                          current_control_reference_deg=list(measured),
                                          current_control_reference_source=reference_source)
                            interrupted_in_decision = True

                        def before_correction_send():
                            nonlocal in_flight
                            if self._cancelled.is_set():
                                raise _Abort('cancelled', 'cancelled before arrival correction dispatch')
                            in_flight = True

                        source_return_boundary, response = self._dispatch_guarded_action(
                            request, command_id=correction_id, return_active=return_active,
                            include_pregrasp=True, on_pending=on_pending_correction,
                            on_before_send=before_correction_send)
                        if source_return_boundary:
                            break
                        command_id, command = correction_id, correction
                        segment_requested = correction_requested
                        segment_effective, segment_steps = list(correction_effective), correction_steps
                        row.update(command_id=command_id, result='arrival_correction_outcome_unknown')
                        if (not isinstance(response, dict) or response.get('command_id') != command_id
                                or response.get('replayed') is not False):
                            raise _Abort('reply_mismatch', 'arrival correction response cannot be confirmed')
                        status = response.get('status')
                        row['arrival_corrections'][-1]['result'] = status
                        if status == 'failed':
                            row['arrival_corrections'][-1]['controller_error'] = copy.deepcopy(response.get('error'))
                            failure_observation = response.get('observation')
                            if isinstance(failure_observation, dict):
                                row['arrival_corrections'][-1]['failure_observation_unverified'] = {
                                    key: copy.deepcopy(failure_observation.get(key)) for key in
                                    ('observation_id', 'sensor_sequence', 'sensor_valid', 'moving',
                                     'joints_deg', 'tcp_mm')}
                    interruption = None
                    if self._interrupt_retarget_enabled and action == 'move' and not return_active:
                        # A request can be stopping this very command while the
                        # synchronous response is returning. Do not plan or
                        # send another segment until that request has resolved.
                        while not self._source_return_admission_done.wait(.02):
                            if self._cancelled.is_set():
                                raise _Abort('cancelled', 'transfer cancelled during retarget admission')
                        with self._state_lock:
                            candidate = self._source_return_state['interruption']
                            pending_cancel = self._source_return_state['pending_cancel_event']
                            if isinstance(candidate, dict) and candidate.get('command_id') == command_id:
                                interruption = candidate
                    if interruption is not None:
                        sent, stopped, stopped_joints, instance, last_time, last_sequence = (
                            self._observe_interrupted_stop(
                                lease, response, segment_requested, segment_effective, segment_steps,
                                instance=instance, seen=seen, last_time=last_time, last_sequence=last_sequence,
                                gripper=gripper, expected_joints=expected_joints, pending_cancel=pending_cancel,
                                subject='interrupted command', cancel_stage='interrupted retarget'))
                        # The controller has stopped: any earlier SDK hold is
                        # no longer evidence for a later gripper command.
                        retained_move = None
                        in_flight = False
                        sent_or_unknown = sent_or_unknown or sent
                        interrupted_target = None
                        interrupted_basis = None
                        if sent:
                            result['interrupted_command_count'] = result.get('interrupted_command_count', 0) + 1
                            if command.get('kind') == 'blended_joint_path':
                                trace = row['blended_path']['execution']
                                if trace.get('scope') == 'sdk_host_stream_unverified':
                                    submitted = trace['sdk_submitted_count']
                                    encoded = trace['sdk_encoded_count']
                                    if submitted != encoded:
                                        row.update(result='interrupted_sdk_frame_unconfirmed',
                                            interrupted_command_transmitted=True,
                                            interrupted_sdk_submitted_count=submitted,
                                            interrupted_sdk_encoded_count=encoded,
                                            interrupted_sdk_submitted_target_deg=trace.get('last_sdk_submitted_target_deg'),
                                            interrupted_sdk_encoded_target_deg=trace.get('last_sdk_encoded_target_deg'),
                                            stopped_joints_deg=stopped_joints,
                                            stopped_reference_source='stationary_observation',
                                            interrupted_target_arrival_confirmed=False,
                                            physical_success=None)
                                        raise _Abort('interrupted_sdk_frame_unconfirmed',
                                            'the last SDK submission has no confirmed encoded frame')
                                    if trace.get('sdk_encoded_count', 0):
                                        interrupted_target = _joints(trace['last_sdk_encoded_target_deg'])
                                        interrupted_basis = 'sdk_host_encoded_not_physical_arrival'
                                elif trace.get('applied_count', 0):
                                    interrupted_target = _joints(trace['last_applied_joints_deg'])
                                    interrupted_basis = 'virtual_worker_applied_frame_not_arrival'
                            else:
                                receipt = response.get('receipt')
                                if (not isinstance(receipt, dict)
                                        or list(receipt.get('steps', ())) != list(segment_steps)
                                        or not _close(_joints(receipt.get('effective_target_deg')), segment_effective)):
                                    raise _Abort('target_unconfirmed', 'interrupted MOVE receipt is unconfirmed')
                                if direct_student_move:
                                    frame = response.get('interrupted_host_frame')
                                    after_frame = receipt.get('encoding_after_frame_sequence')
                                    submitted = receipt.get('submitted_monotonic')
                                    encoded = (isinstance(frame, dict) and type(after_frame) is int
                                        and _number(submitted) and frame.get('mode_hex') == '11'
                                        and type(frame.get('frame_sequence')) is int
                                        and frame['frame_sequence'] > after_frame
                                        and type(frame.get('target_segment_start_sequence')) is int
                                        and 1 <= frame['target_segment_start_sequence'] <= frame['frame_sequence']
                                        and type(frame.get('command_sequence')) is int
                                        and 0 <= frame['command_sequence'] <= 255
                                        and frame.get('joint_steps') == list(segment_steps)
                                        and frame.get('speed_pct') == int(.5 + command['speed_pct'])
                                        and _number(frame.get('monotonic')) and frame['monotonic'] >= submitted)
                                    if encoded:
                                        interrupted_target = list(segment_effective)
                                        interrupted_basis = 'sdk_host_encoded_not_physical_arrival'
                                        row['interrupted_sdk_host_frame'] = copy.deepcopy(frame)
                                        remember_student_host_frame(segment_steps, command_id, frame, receipt)
                                    else:
                                        interrupted_basis = 'sdk_encoded_frame_unconfirmed'
                                else:
                                    interrupted_target = list(segment_effective)
                                    interrupted_basis = 'worker_dispatch_receipt_not_arrival'
                        if interrupted_target is not None:
                            # Record the target actually sent/encoded, never
                            # the unexecuted final endpoint of a partial stream.
                            self.model.on_control_applied('move', interrupted_target, interrupted_target, None)
                            history_basis = interrupted_basis
                        elif applied_in_decision:
                            self.model.on_control_applied('move', list(applied), list(applied), None)
                            history_basis = 'confirmed_previous_segment'
                        else:
                            # No worker target was sent or no SDK frame was
                            # encoded. Keep the previous confirmed history.
                            self.model.policy.discard_unconfirmed_move()
                            history.pop()
                            history_basis = 'last_confirmed_command_unchanged'
                        row['student_history_after_interruption'] = {
                            'disposition': history_basis,
                            'previous_target_deg': list(self.model.policy.current_target),
                            'previous_action': self.model.policy.previous_action,
                            'interrupted_target_committed': interrupted_target is not None,
                            'target_arrival_confirmed': False,
                            'stopped_observation_is_applied_target': False}
                        applied = list(stopped_joints)
                        confirmed_target = False
                        reference_source = 'stationary_observation_after_interruption'
                        observation, expected_joints, instance, last_time, last_sequence = (
                            self._rearm_stopped_controller(
                                lease, interruption, stopped, stopped_joints,
                                instance=instance, seen=seen, last_time=last_time, last_sequence=last_sequence,
                                gripper=gripper, pending_cancel=pending_cancel, cancel_stage='interrupted retarget',
                                rearm_error='controller did not re-arm from the stopped observation',
                                post_drift_error='joints changed after interrupted stop'))
                        result.update(current_target_known=confirmed_target,
                                      last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                      current_control_reference_deg=list(applied),
                                      current_control_reference_source=reference_source)
                        row.update(result='interrupted_after_send' if sent else 'interrupted_before_send',
                                   transmitted=bool(sent or applied_in_decision),
                                   interrupted_command_transmitted=sent,
                                   interrupted_requested_target_deg=list(segment_requested),
                                   interrupted_effective_target_deg=(list(interrupted_target)
                                       if direct_student_move and interrupted_target is not None else
                                       None if direct_student_move else list(segment_effective) if sent else None),
                                   interrupted_target_evidence=interrupted_basis,
                                   interrupted_target_arrival_confirmed=False,
                                   stopped_joints_deg=stopped_joints,
                                   stopped_reference_source='stationary_observation',
                                   grasp_verified=False, physical_success=None)
                        if sent and response.get('receipt') is not None:
                            row['interrupted_controller_receipt'] = copy.deepcopy(response['receipt'])
                        if command.get('kind') == 'blended_joint_path' and sent:
                            row['interrupted_final_endpoint_deg'] = list(segment_effective)
                            row['interrupted_effective_target_deg'] = response['blended_path'].get('last_applied_joints_deg')
                            if response['blended_path'].get('scope') == 'sdk_host_stream_unverified':
                                row['interrupted_target_evidence'] = ('sdk_host_encoded_not_physical_arrival'
                                    if response['blended_path'].get('sdk_encoded_count') else
                                    'sdk_no_encoded_frame')
                                row['interrupted_sdk_encoded_target_deg'] = response['blended_path'].get('last_sdk_encoded_target_deg')
                                row['interrupted_physical_target_known'] = False
                        if detail is not None:
                            detail.update(result=row['result'], transmitted=sent,
                                          stopped_joints_deg=stopped_joints)
                        source_return_boundary = True
                        interrupted_in_decision = True
                        break
                    if status == 'cancelled' and response.get('applied') is False:
                        in_flight = False
                        row.update(result='cancelled_not_sent', transmitted=bool(applied_in_decision))
                        if detail is not None:
                            detail.update(result='cancelled_not_sent', transmitted=False)
                        raise _Abort('cancelled', 'controller cancelled before transmission')
                    if self.enable_mock_nominal_trim and status == 'settled_needs_nominal_retry':
                        if command.get('allow_mock_nominal_retry') is not True:
                            raise _Abort('command_unconfirmed', 'mock retry was not permitted for this segment')
                        self._mock_nominal_response(response, segment_requested, segment_effective,
                                                    segment_steps, nominal_segment)
                        if (response.get('applied') is not True or response.get('mock_retry_permitted') is not True
                                or response.get('nominal_arrival_reached') is not False
                                or not isinstance(response.get('stop'), dict)
                                or response['stop'].get('confirmed') is not True):
                            raise _Abort('command_unconfirmed', 'mock retry lacks confirmed stationary feedback')
                        after_first = response.get('observation')
                        instance, last_time, last_sequence = self._observation(
                            after_first, instance, seen, last_time, last_sequence, require_new_sample=True)
                        self._gripper(after_first, gripper)
                        measured = _joints(after_first['joints_deg'])
                        geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                                            tool_vertical_mm=config.tool_vertical_mm)
                        nominal_error = max(abs(a-b) for a, b in zip(nominal_segment, measured))
                        nominal_tcp_error = math.dist(geometry.forward(nominal_segment), geometry.forward(measured))
                        if nominal_error > 1.0 or nominal_tcp_error > 5.0:
                            raise _Abort('nominal_retry_out_of_range', 'mock retry residual exceeds its bound')
                        in_flight = False
                        sent_or_unknown = True
                        applied = list(segment_effective)
                        confirmed_target = True
                        reference_source = 'controller_confirmed_sent_target'
                        applied_in_decision += 1
                        result['applied_command_count'] += 1
                        result.update(current_target_known=True, last_confirmed_target_deg=list(applied),
                                      current_control_reference_deg=list(applied),
                                      current_control_reference_source=reference_source)
                        final_xyz = geometry.forward(_joints(segments[-1]['effective']))
                        measured_xyz, old_xyz = geometry.forward(measured), geometry.forward(nominal_segment)
                        replan_descent = (self.enable_mock_vertical_replan and return_active and gripper == 'closed'
                            and phase.get('kind') == 'vertical'
                            and final_xyz[2] < measured_xyz[2] < old_xyz[2])
                        retry_fraction = None
                        if replan_descent:
                            if mock_vertical_replans_used >= 1:
                                raise _Abort('vertical_replan_limit', 'only one mock vertical replan is permitted per transfer')
                            mock_vertical_replans_used += 1
                            replan_deadline = time.monotonic() + _MOCK_VERTICAL_REPLAN_WALL_SECONDS
                            counts = self._mock_quiet_diagnostics(observation, after_first)
                            previous_nominal = list(nominal_segment)
                            replacements, retry_requested, retry_effective, retry_steps, fallback = (
                                self._mock_replan_source_descent(measured, applied, nominal_segment, segments[-1],
                                    phase, task, slot_xy_mm, remaining_capacity=64-segment_index,
                                    deadline=replan_deadline))
                            segments[segment_index:] = replacements
                            entries = [{'index': 0, 'requested_target_deg': list(part['requested']),
                                        'quantized_target_deg': list(part['effective']),
                                        'preflight': part['path'], 'result': 'planned'} for part in replacements]
                            if split_reason is None:
                                split_reason = {'kind': 'mock_vertical_replan_after_unreached_nominal'}
                                row['segmentation'] = {'reason': split_reason, 'segments': entries}
                            else:
                                row['segmentation']['segments'][segment_index:] = entries
                            for number, entry in enumerate(row['segmentation']['segments'], 1):
                                entry['index'] = number
                            row['segmentation']['planned_segment_count'] = len(segments)
                            detail = row['segmentation']['segments'][segment_index]
                            nominal_segment = list(replacements[0]['effective'])
                            row['mock_vertical_replan'] = {
                                'superseded_nominal_deg': previous_nominal,
                                'superseded_nominal_arrival_confirmed': False,
                                'stopped_observation_deg': list(measured),
                                'last_sent_target_deg': list(applied),
                                'final_nominal_deg': list(replacements[-1]['effective']),
                                'replacement_count': len(replacements), 'maximum_total_segments': 64,
                                'maximum_replans_per_transfer': 1,
                                'time_budget_seconds': _MOCK_VERTICAL_REPLAN_WALL_SECONDS, 'warning_raw': 0,
                                'collision_event_counts': counts, 'fallback_reason': fallback,
                                'subsequent_retries_allowed': False,
                                'continuous_contact_assessed': False, 'physical_success': None}
                            trim_record.update(superseded_nominal_target_deg=previous_nominal,
                                               superseded_nominal_arrival_confirmed=False,
                                               nominal_target_deg=list(nominal_segment))
                        else:
                            retry_requested, retry_effective, retry_steps, retry_fraction = self._mock_bounded_nominal_retry(
                                nominal_segment, measured, applied, phase, task, slot_xy_mm, return_active=return_active)
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'cancelled before mock nominal retry')
                        if replan_deadline is not None and time.monotonic() >= replan_deadline:
                            raise _Abort('vertical_replan_timeout', 'mock replanning exceeded its time budget before retry')
                        age = time.monotonic() - after_first['observed_monotonic']
                        if (age > config.limits.observation_max_age
                                or after_first['sensor_age_seconds'] + age > config.limits.sensor_max_age):
                            raise _Abort('observation_expired', 'mock retry observation expired')
                        retry_id = uuid.uuid4().hex
                        trim_record['retry'] = {'first_command_id': command_id,
                            'first_applied_target_deg': list(applied), 'first_observed_deg': measured,
                            'first_reference_source': reference_source, 'first_nominal_arrival_confirmed': False,
                            'requested_target_deg': list(retry_requested),
                            'quantized_target_deg': list(retry_effective), 'correction_fraction': retry_fraction,
                            'command_id': retry_id}
                        segment_requested, segment_effective, segment_steps = retry_requested, retry_effective, retry_steps
                        command = {**command, 'joints_deg': list(retry_requested),
                                   'nominal_arrival_joints_deg': list(nominal_segment),
                                   'allow_mock_nominal_retry': False}
                        observation = after_first
                        expected_joints = measured
                        request = {'lease_id': lease, 'command_id': retry_id,
                                   'observation_id': observation['observation_id'], 'action': command}
                        row.update(command_id=retry_id, result='mock_retry_outcome_unknown')
                        if self._source_return_enabled and not return_active:
                            with self._mock_move_dispatch_lock:
                                with self._state_lock:
                                    source_return_boundary = (self._source_return_state['pending']
                                                              or self._source_return_state['admission_waiting'])
                                    if not source_return_boundary:
                                        self._mock_move_dispatch_owner = threading.get_ident()
                                        self._source_return_state['active_command_id'] = retry_id
                                if not source_return_boundary:
                                    try:
                                        if self._cancelled.is_set():
                                            raise _Abort('cancelled', 'cancelled before mock nominal retry')
                                        in_flight = True
                                        response = self.controller.apply_action(request)
                                    finally:
                                        with self._state_lock:
                                            self._mock_move_dispatch_owner = None
                                            self._source_return_state['active_command_id'] = None
                            if source_return_boundary:
                                trim_record['retry']['result'] = 'superseded_before_send'
                                break
                        else:
                            in_flight = True
                            response = self.controller.apply_action(request)
                        if (not isinstance(response, dict) or response.get('command_id') != retry_id
                                or response.get('replayed') is not False):
                            raise _Abort('reply_mismatch', 'mock retry response cannot be confirmed')
                        status = response.get('status')
                        retry_interruption = None
                        pending_cancel = None
                        if self._interrupt_retarget_enabled and not return_active:
                            while not self._source_return_admission_done.wait(.02):
                                if self._cancelled.is_set():
                                    raise _Abort('cancelled', 'transfer cancelled during mock retry retarget')
                            with self._state_lock:
                                candidate = self._source_return_state['interruption']
                                pending_cancel = self._source_return_state['pending_cancel_event']
                                if isinstance(candidate, dict) and candidate.get('command_id') == retry_id:
                                    retry_interruption = candidate
                        if retry_interruption is not None:
                            sent, stopped, stopped_joints, instance, last_time, last_sequence = (
                                self._observe_interrupted_stop(
                                    lease, response, segment_requested, segment_effective, segment_steps,
                                    instance=instance, seen=seen, last_time=last_time, last_sequence=last_sequence,
                                    gripper=gripper, expected_joints=expected_joints, pending_cancel=pending_cancel,
                                    subject='mock retry', cancel_stage='mock retry retarget'))
                            in_flight = False
                            sent_or_unknown = True
                            if sent:
                                self.model.policy.on_move_interrupted(stopped_joints)
                                applied = list(stopped_joints)
                                confirmed_target = False
                                reference_source = 'stationary_observation_after_interruption'
                                result['interrupted_command_count'] = result.get('interrupted_command_count', 0) + 1
                            else:
                                self.model.on_control_applied('move', list(applied), list(applied), None)
                            observation, expected_joints, instance, last_time, last_sequence = (
                                self._rearm_stopped_controller(
                                    lease, retry_interruption, stopped, stopped_joints,
                                    instance=instance, seen=seen, last_time=last_time, last_sequence=last_sequence,
                                    gripper=gripper, pending_cancel=pending_cancel,
                                    cancel_stage='mock retry retarget',
                                    rearm_error='controller did not re-arm after mock retry interruption',
                                    post_drift_error='joints changed after the mock retry stop',
                                    force_mock=True))
                            result.update(current_target_known=confirmed_target,
                                          last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                          current_control_reference_deg=list(applied),
                                          current_control_reference_source=reference_source)
                            row.update(result='interrupted_after_send' if sent else 'interrupted_before_send',
                                       transmitted=True, interrupted_command_transmitted=sent,
                                       interrupted_requested_target_deg=list(segment_requested),
                                       stopped_joints_deg=stopped_joints,
                                       stopped_reference_source='stationary_observation',
                                       grasp_verified=False, physical_success=None)
                            trim_record['retry'].update(result=row['result'], transmitted=sent,
                                                        stopped_joints_deg=stopped_joints)
                            if detail is not None:
                                detail.update(result=row['result'], transmitted=sent,
                                              stopped_joints_deg=stopped_joints)
                            source_return_boundary = True
                            interrupted_in_decision = True
                            break
                        if status == 'cancelled' and response.get('applied') is False:
                            in_flight = False
                            raise _Abort('cancelled', 'mock retry was cancelled before transmission')
                    gripper_needs_compensation = (action == 'open' and config.backend == 'sdk'
                        and status == 'gripper_reported_arm_needs_nominal_correction')
                    is_noop = (action == 'move' and status in ('already_at_target', 'position_target_unchanged')
                               and response.get('applied') is False)
                    is_applied = ((action == 'move' and
                                   (status == 'completed' or (preserve_student_endpoint and
                                    status in ('settled', 'settled_no_motion')))
                                   and response.get('applied') is True
                                   and (retain_move_requested or (isinstance(response.get('stop'), dict)
                                                                 and response['stop'].get('confirmed') is True)))
                                  or (action != 'move' and (status == 'gripper_reported' or gripper_needs_compensation)
                                      and response.get('applied') is True
                                      and response.get('gripper_command') == command['state']
                                      and response.get('gripper_reported_confirmed') is True))
                    if not (is_noop or is_applied):
                        raise _Abort('command_unconfirmed', 'the controller did not confirm command completion')
                    if action == 'move':
                        if is_trajectory:
                            target_reply, trace = response.get('target'), response.get('mock_trajectory')
                            if (not isinstance(target_reply, dict)
                                    or target_reply.get('kind') != 'mock_joint_trajectory'
                                    or not _close(_joints(target_reply.get('effective_target_deg')), requested)
                                    or not isinstance(trace, dict) or trace.get('completed') is not True
                                    or trace.get('applied_tick_count') != straight_descent.TICKS
                                    or trace.get('period_seconds') != straight_descent.TIMESTEP):
                                raise _Abort('target_unconfirmed', 'mock trajectory application was not confirmed')
                            row['virtual_descent_comparison'].update(
                                purpose='mock_execution_confirmed', applied_to_controller=True,
                                applied_tick_count=trace['applied_tick_count'], timing=trace.get('timing'))
                        else:
                            if self.enable_mock_nominal_trim or self.enable_arrival_correction or allow_damped_compensation:
                                self._mock_nominal_response(response, segment_requested, segment_effective,
                                                            segment_steps, nominal_segment)
                                if not preserve_student_endpoint and response.get('nominal_arrival_reached') is not True:
                                    raise _Abort('target_unconfirmed', 'mock nominal arrival was not confirmed')
                            else:
                                self._target_response(response, segment_requested, segment_effective, segment_steps)
                        valid_noops = ('encoded_target_unchanged', 'already_within_tolerance')
                        if retain_move_requested:
                            valid_noops += ('retained_target_unchanged',)
                        if self.enable_mock_nominal_trim or self.enable_arrival_correction:
                            valid_noops += ('already_at_nominal_target',)
                        if is_noop and response.get('reason') not in valid_noops:
                            raise _Abort('reply_mismatch', 'the no-transmission reason is missing')
                    after = response.get('observation')
                    instance, last_time, last_sequence = self._observation(
                        after, instance, seen, last_time, last_sequence,
                        # Arm and gripper feedback are sampled independently.
                        require_new_sample=(is_noop and retain_move_requested) or (is_applied and
                            (action == 'move' or (action == 'open'
                                and (return_active or self.enable_mock_open_drift_reconcile or self.enable_arrival_correction))
                                or (action != 'move' and retained_move is not None and config.backend == 'sdk'))))
                    next_gripper = gripper if action == 'move' else command['state']
                    self._gripper(after, next_gripper)
                    if action != 'move' and isinstance(row.get('gripper_delivery'), dict):
                        row['gripper_delivery']['observation_verified'] = True
                    if (direct_student_move and is_noop and response.get('nominal_arrival_reached') is False):
                        self._retained_noop_response(response, retained_move, segment_steps,
                                                     before_command_sequence, after)
                        row.update(result='retained_target_not_reached', transmitted=False,
                                   nominal_arrival_confirmed=False, observed_joints_deg=list(after['joints_deg']))
                        remember_no_motion(move_start_observation, move_attempts)
                        raise _Abort('motion_not_observed', 'the retained target is still not reached')
                    if is_applied and retain_move_requested:
                        retained_move = self._retained_move_response(
                            response, segment_steps, command['speed_pct'], before_command_sequence, after)
                        if direct_student_move:
                            remember_student_host_frame(segment_steps, command_id,
                                response['completion']['host_frame'], response.get('receipt'))
                        retained_move.update(
                            nominal_target_deg=list(nominal_segment or segment_effective),
                            speed_pct=command['speed_pct'],
                            path_constraints=copy.deepcopy(command.get('path_constraints', {})),
                            arrival_tolerance_mm=command.get('arrival_tolerance_mm'),
                            compensation_allowed=bool(allow_damped_compensation),
                            compensation_used=bool(damped_compensation_used),
                            checked_phase=copy.deepcopy(phase) if checked_move else None)
                        # The host frame, not the requested floating-point value,
                        # is the target recorded in the student's action history.
                        segment_effective = [step / STEPS_PER_DEGREE for step in retained_move['target_steps']]
                        row['completion'] = copy.deepcopy(response['completion'])
                        row['sdk_host_encoded_target_deg'] = list(segment_effective)
                    elif is_noop and retain_move_requested:
                        retained_move = self._retained_noop_response(
                            response, retained_move, segment_steps, before_command_sequence, after)
                        row['retained_position_observation'] = copy.deepcopy(response['retained_position_observation'])
                    elif is_applied and action != 'move' and config.backend == 'sdk' and retained_move is not None:
                        retained_move = self._retained_gripper_response(
                            response, retained_move, before_command_sequence, after)
                        row['arm_hold'] = copy.deepcopy(response['arm_hold'])
                        if 'arrival_diagnostics_before_gripper' in response:
                            row['arrival_diagnostics_before_gripper'] = copy.deepcopy(response['arrival_diagnostics_before_gripper'])
                        if 'arm_arrival_observation' in response:
                            row['arm_arrival_observation'] = copy.deepcopy(response['arm_arrival_observation'])
                    gripper_already_counted = False
                    if gripper_needs_compensation:
                        # OPEN has already been sent and freshly reported.  The
                        # following MOVE is a single correction of the last
                        # student's nominal MOVE, never a second OPEN.
                        sent_or_unknown, in_flight = True, False
                        gripper = 'open'
                        result['last_confirmed_gripper_command'] = 'open'
                        result['applied_command_count'] += 1
                        result['current_target_known'] = False
                        gripper_already_counted = True
                        row.update(result='gripper_reported_arm_correction_pending', transmitted=True)
                        correction = response.get('correction_action')
                        basis_reply = response.get('correction_basis')
                        if (retained_move is None or not retained_move.get('compensation_allowed')
                                or retained_move.get('compensation_used') is not False
                                or response.get('arm_nominal_arrival_reached') is not False
                                or response.get('nominal_correction_permitted') is not True
                                or not isinstance(basis_reply, dict)
                                or basis_reply.get('after_gripper_command_id') != command_id
                                or not isinstance(correction, dict) or correction.get('kind') != 'move_joints'
                                or correction.get('correction_of_command_id') != retained_move['command_id']
                                or correction.get('allow_nominal_correction', False) is not False):
                            raise _Abort('target_unconfirmed', 'OPEN was reported but its arm correction is unavailable')
                        nominal = _joints(retained_move['nominal_target_deg'])
                        constraints = retained_move['path_constraints']
                        if (not _close(_joints(correction.get('nominal_arrival_joints_deg')), nominal)
                                or correction.get('speed_pct') != retained_move['speed_pct']
                                or correction.get('path_constraints') != {
                                    key: list(value) if isinstance(value, tuple) else value
                                    for key, value in constraints.items()}
                                or correction.get('arrival_tolerance_mm') != retained_move['arrival_tolerance_mm']):
                            raise _Abort('reply_mismatch', 'post-OPEN correction differs from the last MOVE')
                        correction_requested = _joints(correction.get('joints_deg'))
                        correction_steps, correction_effective = quantize_joints(correction_requested)
                        measured = _joints(after['joints_deg'])
                        if max(abs(a-b) for a, b in zip(correction_effective, nominal)) > 1.0:
                            raise _Abort('nominal_correction_out_of_range', 'post-OPEN correction exceeds one degree')
                        basis = self._damped_correction_basis(
                            response, correction, retained_move['command_id'], nominal, measured, after,
                            retained_move['speed_pct'], before_command_sequence, constraints)
                        latest = basis['latest_host_frame']
                        if (latest['target_segment_start_sequence'] != retained_move['target_segment_start_sequence']
                                or latest['frame_sequence'] < retained_move['frame_sequence']
                                or basis['sensor_sequence'] < retained_move['sensor_sequence']):
                            raise _Abort('target_unconfirmed', 'post-OPEN correction predates the retained hold')
                        if retained_move['checked_phase'] is not None:
                            self._checked_joint_segment(measured, correction_effective,
                                                        retained_move['checked_phase'], observed_start=True)
                        correction_id = uuid.uuid4().hex
                        correction = copy.deepcopy(correction)
                        correction['completion_mode'] = 'retain_position_target'
                        row['post_gripper_correction'] = {
                            'command_id': correction_id, 'correction_of_command_id': retained_move['command_id'],
                            'after_gripper_command_id': command_id, 'nominal_target_deg': list(nominal),
                            'requested_target_deg': list(correction_requested),
                            'effective_target_deg': list(correction_effective),
                            'basis': copy.deepcopy(basis), 'result': 'not_sent'}
                        retained_move['compensation_used'] = True
                        correction_request = {'lease_id': lease, 'command_id': correction_id,
                                              'observation_id': after['observation_id'], 'action': correction}
                        correction_before_sequence = after['sensor_sequence']
                        if self._cancelled.is_set():
                            raise _Abort('cancelled', 'cancelled after OPEN and before arm correction')
                        def on_pending_post_open():
                            raise _Abort('cancelled', 'retarget requested after OPEN and before arm correction')

                        def before_post_open_send():
                            nonlocal in_flight
                            if self._cancelled.is_set():
                                raise _Abort('cancelled', 'cancelled before post-OPEN correction dispatch')
                            in_flight = True
                            row['post_gripper_correction']['result'] = 'outcome_unknown'

                        _, correction_response = self._dispatch_guarded_action(
                            correction_request, command_id=correction_id, return_active=False,
                            include_pregrasp=True, on_pending=on_pending_post_open,
                            on_before_send=before_post_open_send)
                        if (not isinstance(correction_response, dict)
                                or correction_response.get('command_id') != correction_id
                                or correction_response.get('replayed') is not False):
                            raise _Abort('reply_mismatch', 'post-OPEN correction response cannot be confirmed')
                        in_flight = False
                        row['post_gripper_correction']['result'] = correction_response.get('status')
                        if (correction_response.get('status') != 'completed'
                                or correction_response.get('applied') is not True):
                            unknown = True
                            row['post_gripper_correction']['controller_error'] = copy.deepcopy(
                                correction_response.get('error'))
                            raise _Abort('command_unconfirmed', 'OPEN was reported but its arm correction failed')
                        self._mock_nominal_response(correction_response, correction_requested,
                                                    correction_effective, correction_steps, nominal)
                        correction_after = correction_response.get('observation')
                        instance, last_time, last_sequence = self._observation(
                            correction_after, instance, seen, last_time, last_sequence, require_new_sample=True)
                        self._gripper(correction_after, 'open')
                        new_retained = self._retained_move_response(
                            correction_response, correction_steps, correction['speed_pct'],
                            correction_before_sequence, correction_after)
                        allowed_tcp = (retained_move['arrival_tolerance_mm'] if
                                       retained_move['arrival_tolerance_mm'] is not None
                                       else config.limits.tcp_tolerance_mm)
                        proof = self._damped_compensation_completed(
                            correction_response, basis, nominal, correction_effective, correction_after, allowed_tcp)
                        new_retained.update(
                            nominal_target_deg=list(nominal), speed_pct=retained_move['speed_pct'],
                            path_constraints=copy.deepcopy(constraints),
                            arrival_tolerance_mm=retained_move['arrival_tolerance_mm'],
                            compensation_allowed=True, compensation_used=True,
                            checked_phase=copy.deepcopy(retained_move['checked_phase']))
                        retained_move = new_retained
                        segment_effective = [step / STEPS_PER_DEGREE for step in retained_move['target_steps']]
                        expected_joints = _joints(correction_after['joints_deg'])
                        after = correction_after
                        confirmed_target, reference_source = True, 'controller_confirmed_move'
                        result['applied_command_count'] += 1
                        result['last_confirmed_nominal_target_deg'] = list(nominal)
                        row['post_gripper_correction'].update(
                            result='completed', completion=copy.deepcopy(correction_response['completion']),
                            damped_compensation=proof, observed_joints_deg=list(expected_joints))
                        row.update(arm_nominal_target_deg=list(nominal), nominal_arrival_confirmed=True,
                                   sdk_host_encoded_target_deg=list(segment_effective))
                        if preserve_student_endpoint:
                            row['student_nominal_target_deg'] = list(nominal)
                    after_joints = _joints(after['joints_deg'])
                    if damped_compensation_used:
                        row['damped_compensation'] = self._damped_compensation_completed(
                            response, damped_compensation_basis, nominal_segment, segment_effective, after,
                            command.get('arrival_tolerance_mm', config.limits.tcp_tolerance_mm))
                    if action == 'move':
                        arrival_target = nominal_segment if (self.enable_mock_nominal_trim or self.enable_arrival_correction
                                                             or allow_damped_compensation) else segment_effective
                        if checked_move or direct_student_move:
                            arrival_geometry = self._geometry()
                            arrival_tcp_error = math.dist(arrival_geometry.forward(after_joints),
                                                          arrival_geometry.forward(arrival_target))
                            row.setdefault('arrival_observations', []).append({
                                'command_id': command_id, 'observation_id': after['observation_id'],
                                'observed_joints_deg': list(after_joints),
                                'nominal_target_deg': list(arrival_target),
                                'nominal_joint_error_deg': max(abs(a-b) for a, b in zip(after_joints, arrival_target)),
                                'nominal_tcp_error_mm': arrival_tcp_error,
                                'arrival_tolerance_mm': command.get('arrival_tolerance_mm', config.limits.tcp_tolerance_mm),
                                'correction': bool(command.get('correction_of_command_id'))})
                            if command.get('correction_of_command_id') and row.get('arrival_corrections'):
                                row['arrival_corrections'][-1].update(
                                    final_observed_joints_deg=list(after_joints),
                                    final_nominal_tcp_error_mm=arrival_tcp_error)
                        if not preserve_student_endpoint and not _close(after_joints, arrival_target, tolerance):
                            raise _Abort('target_unconfirmed', 'observed joints do not agree with the arrival target')
                        if not preserve_student_endpoint and (self.enable_mock_nominal_trim or self.enable_arrival_correction or allow_damped_compensation):
                            allowed_tcp = command.get('arrival_tolerance_mm', config.limits.tcp_tolerance_mm)
                            if math.dist(self._geometry().forward(after_joints), self._geometry().forward(arrival_target)) > allowed_tcp:
                                raise _Abort('target_unconfirmed', 'observed TCP does not agree with the nominal target')
                            nominal_applied = list(nominal_segment)
                            result['last_confirmed_nominal_target_deg'] = list(nominal_applied)
                            if confirmed_target and self.enable_mock_nominal_trim:
                                reference_source = 'controller_confirmed_nominal_move'
                                result['current_control_reference_source'] = reference_source
                            if trim_record is not None:
                                trim_record.update(final_observed_deg=list(after_joints),
                                                   nominal_arrival_confirmed=True,
                                                   final_sent_target_deg=(list(applied) if is_noop
                                                                          else list(segment_effective)))
                        if (retain_move_requested and not preserve_student_endpoint and not damped_compensation_used
                                and not self._within_target_tolerances(
                                    after_joints, segment_effective,
                                    allowed_tcp_mm=command.get('arrival_tolerance_mm'))):
                            raise _Abort('target_unconfirmed', 'observed joints or TCP do not agree with the sent target')
                    if action != 'move' and not _close(after_joints, expected_joints, drift):
                        if preserve_student_endpoint:
                            row['gripper_observed_joint_change_deg'] = [a-b for a, b in zip(after_joints, expected_joints)]
                        else:
                            if not (action == 'open' and (self.enable_mock_open_drift_reconcile or self.enable_arrival_correction)):
                                raise _Abort('state_changed', 'joints changed during the gripper command')
                            if self._cancelled.is_set():
                                raise _Abort('cancelled', 'cancelled before mock OPEN drift reconciliation')
                            key = ('open_feedback_reconciliation' if self.enable_arrival_correction
                                   else 'mock_open_drift_reconciliation')
                            row[key] = self._reconcile_open_drift(
                                observation, after, task, slot_xy_mm, applied,
                                runtime=self.enable_arrival_correction)
                    expected_joints = after_joints
                    observation = after
                    in_flight = False
                    after_open = None
                    if return_active and action == 'open':
                        after_open = self._source_return_open_pose(after, applied, task, slot_xy_mm)
                        row['source_return_open_pose']['after'] = after_open
                        with self._state_lock:
                            self._source_return_state['open_pose_after'] = after_open
                    if (self.require_retarget_release_check or direct_active) and action == 'open' and gripper == 'closed':
                        checked_after = self._retarget_grip_pose(after, applied, task, slot_xy_mm, release=True)
                        row['retarget_release_pose']['after'] = checked_after
                        if not checked_after['matches'] and not preserve_student_endpoint:
                            raise _Abort('retarget_release_unconfirmed',
                                         'redirected OPEN feedback is outside the registered nominal destination')
                        result['retarget_release_pose_confirmed'] = bool(checked_after['matches'])
                    if self.require_retarget_release_check and action == 'close' and gripper == 'open':
                        checked_after = self._retarget_grip_pose(after, applied, task, slot_xy_mm, release=False)
                        row['retarget_grasp_pose']['after'] = checked_after
                        if not checked_after['matches']:
                            raise _Abort('retarget_grasp_unconfirmed',
                                         'redirected CLOSE feedback is outside the registered nominal source')
                        result['retarget_grasp_pose_confirmed'] = True
                    if is_noop:
                        if preserve_student_endpoint:
                            sent_reached = self._target_reached_flag(
                                response.get('sent_target_reached'), after_joints, segment_effective,
                                allowed_tcp_mm=command.get('arrival_tolerance_mm'))
                            confirmed_target = (response.get('nominal_arrival_reached') is True
                                                and sent_reached is True)
                            row['nominal_arrival_confirmed'] = confirmed_target
                            applied = list(segment_effective if confirmed_target else after_joints)
                            reference_source = ('controller_confirmed_move' if confirmed_target
                                                else 'stationary_observation_unreached_target')
                            result.update(current_target_known=confirmed_target,
                                          last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                          last_confirmed_nominal_target_deg=(list(nominal_segment) if confirmed_target else None),
                                          current_control_reference_deg=list(applied),
                                          current_control_reference_source=reference_source)
                        if forced_lift and segment_index + 1 == len(segments) and not applied_in_decision:
                            with self._state_lock:
                                self._source_return_state['lift_status'] = 'not_applied'
                            raise _Abort('source_return_lift_unavailable', 'the mock lift was not applied')
                        if detail is not None:
                            detail.update(result='already_at_target', transmitted=False,
                                          retained_applied_target_deg=list(applied) if confirmed_target else None,
                                          retained_control_reference_deg=list(applied))
                        if manual_early_action and segment_index + 1 == len(segments):
                            with self._state_lock:
                                self._source_return_state['return_move_confirmed'] = True
                                self._source_return_state['early_return_phase'] = 1
                        if segment_index + 1 < len(segments):
                            continue
                        if not applied_in_decision:
                            if retain_move_requested:
                                # This is a new policy time step, but no new
                                # host target or physical transport was sent.
                                self.model.policy.on_move_retained(segment_effective)
                                row.update(result='position_target_reused', transmitted=False,
                                           retained_position_from_command_id=retained_move['command_id'],
                                           retained_applied_target_deg=list(segment_effective) if confirmed_target else None,
                                           retained_control_reference_deg=list(applied))
                                if return_active and not forced_lift:
                                    with self._state_lock:
                                        self._source_return_state['return_move_confirmed'] = True
                            else:
                                self.model.policy.on_move_not_sent()
                                history.pop()
                                row.update(result='already_at_target', transmitted=False,
                                           retained_applied_target_deg=list(applied) if confirmed_target else None,
                                           retained_control_reference_deg=list(applied))
                            no_op_decision = True
                            break
                        if (not _close(applied, segment_effective)
                                and not ((self.enable_mock_nominal_trim or self.enable_arrival_correction)
                                         and response.get('reason') == 'already_at_nominal_target')):
                            row.update(result='partial_goal_not_applied', transmitted=True)
                            raise _Abort('segment_target_not_applied', 'final student target was not transmitted')
                    else:
                        sent_or_unknown = True
                        applied, gripper = list(segment_effective), next_gripper
                        applied_in_decision += 1
                        if not gripper_already_counted:
                            result['applied_command_count'] += 1
                        if action == 'move':
                            sent_reached = self._target_reached_flag(
                                response.get('sent_target_reached'), after_joints, segment_effective,
                                allowed_tcp_mm=command.get('arrival_tolerance_mm'))
                            reached = (not preserve_student_endpoint or
                                       (response.get('nominal_arrival_reached') is True and sent_reached is True))
                            row['nominal_arrival_confirmed'] = reached
                            if preserve_student_endpoint:
                                result['last_confirmed_nominal_target_deg'] = (list(nominal_segment) if reached else None)
                            confirmed_target = reached
                            if preserve_student_endpoint and not reached:
                                applied = list(after_joints)
                                reference_source = 'stationary_observation_unreached_target'
                            else:
                                reference_source = ('controller_confirmed_nominal_move' if self.enable_mock_nominal_trim
                                                    else 'controller_confirmed_move')
                        elif preserve_student_endpoint and retained_move is not None:
                            row['arm_nominal_arrival_reached'] = response.get('arm_nominal_arrival_reached')
                            arm_reached = self._target_reached_flag(
                                response.get('arm_nominal_arrival_reached'), after_joints,
                                retained_move['nominal_target_deg'],
                                allowed_tcp_mm=retained_move.get('arrival_tolerance_mm'))
                            if arm_reached is not True:
                                confirmed_target = False
                                applied = list(after_joints)
                                reference_source = 'stationary_observation_unreached_target'
                        if action == 'close' and self.enable_release_setdown:
                            grasp_geometry = self._geometry()
                            grasp_reference = {'joints_deg': _joints(closing_reference),
                                'nominal_tcp_mm': list(grasp_geometry.forward(closing_reference)),
                                'source': 'stationary_observation_before_confirmed_close',
                                'object_pose_observed': False}
                            row['recorded_grasp_reference'] = copy.deepcopy(grasp_reference)
                        elif action == 'open' and self.enable_release_setdown:
                            grasp_reference = None
                        if self._source_return_enabled:
                            with self._state_lock:
                                state = self._source_return_state
                                if action == 'close':
                                    if self.enable_runtime_source_return and not state['ever_closed']:
                                        grasp_joints = _joints(state['preclose_observed_joints_deg'])
                                        grasp_geometry = self._geometry()
                                        state['grasp_reference'] = {
                                            'joints_deg': grasp_joints,
                                            'nominal_tcp_mm': list(grasp_geometry.forward(grasp_joints)),
                                            'source': 'stationary_observation_before_confirmed_close',
                                            'object_pose_observed': False}
                                        row['recorded_grasp_reference'] = copy.deepcopy(state['grasp_reference'])
                                    state['gripper'], state['ever_closed'] = 'closed', True
                                elif action == 'open':
                                    state['gripper'] = 'open'
                                    state['opened_after_close'] = state['ever_closed']
                                    if state['switched'] and state.get('direct_task') is None:
                                        state['open_after_switch'] = True
                                    if manual_early_action:
                                        state['early_return_phase'] = 2
                                elif (action == 'move' and state['switched'] and state.get('direct_task') is None
                                      and segment_index + 1 == len(segments)):
                                    if forced_lift:
                                        state['lift_status'] = 'applied'
                                    else:
                                        state['return_move_confirmed'] = True
                                    if manual_early_action:
                                        state['early_return_phase'] = 1
                        result.update(current_target_known=confirmed_target,
                                      last_confirmed_target_deg=list(applied) if confirmed_target else None,
                                      current_control_reference_deg=list(applied),
                                      current_control_reference_source=reference_source,
                                      last_confirmed_gripper_command=gripper)
                        row.update(result='partial_path_applied', transmitted=True,
                                   last_confirmed_segment_target_deg=list(applied) if confirmed_target else None)
                        if detail is not None:
                            detail.update(result='command_applied', transmitted=True,
                                          applied_target_deg=list(applied) if confirmed_target else None)
                    if after_open is not None and not after_open['matches'] and not preserve_student_endpoint:
                        row['result'] = 'source_return_open_observation_not_at_source'
                        raise _Abort('source_return_not_at_source', 'post-OPEN observation is outside the nominal source')
                    if cancelled() and action != 'open':
                        row.update(result='cancelled_after_segment', transmitted=bool(applied_in_decision))
                        raise _Abort('cancelled', 'cancelled after a confirmed segment')
                if source_return_boundary:
                    if split_reason is not None:
                        for skipped in row['segmentation']['segments'][segment_index:]:
                            if skipped['result'] == 'planned':
                                skipped['result'] = 'skipped_for_source_return'
                    if interrupted_in_decision:
                        pass  # History and the stopped reference were handled above.
                    elif applied_in_decision:
                        policy_target = (list(nominal_segment) if damped_compensation_used
                                         else list(applied))
                        self.model.on_control_applied('move', policy_target, policy_target, None)
                        row.update(result='source_return_after_confirmed_segment', transmitted=True,
                                   applied_target_deg=list(applied), original_goal_reached=False,
                                   control_reference_deg=list(applied), control_reference_source=reference_source)
                    else:
                        if action == 'move':
                            self.model.policy.on_move_not_sent()
                        else:
                            self.model.policy.on_decision_not_sent()
                        history.pop()
                        row.update(result='superseded_before_send', transmitted=False)
                    with self._state_lock:
                        self._source_return_state['inflight_action'] = None
                    continue
                if no_op_decision:
                    if direct_student_move and retained_move is not None:
                        last_student_move = {'requested_target_deg': list(row['requested_target_deg']),
                            'target_steps': list(retained_move['target_steps']), 'task': copy.deepcopy(task),
                            'direct_active': direct_active, 'return_active': return_active}
                    if self._source_return_enabled:
                        with self._state_lock:
                            self._source_return_state['inflight_action'] = None
                    continue
                if return_active and action == 'move' and applied_in_decision and _close(applied, effective):
                    with self._state_lock:
                        if forced_lift:
                            self._source_return_state['lift_status'] = 'applied'
                        else:
                            self._source_return_state['return_move_confirmed'] = True
                # Internal waypoints are logged individually. The policy sees
                # one decision only after its original effective goal is applied.
                policy_target = (list(effective) if action == 'move' and
                                 (preserve_student_endpoint or damped_compensation_used)
                                 else list(self.model.policy.current_target) if config.backend == 'sdk' and action != 'move'
                                 else list(applied))
                self.model.on_control_applied(action, policy_target, policy_target, None)
                if (direct_student_move and gripper == 'closed' and not return_active and not direct_active
                        and self._is_unstack_destination_descent(
                            segment_observed_start, segment_effective, task, slot_xy_mm)):
                    preceding_move_attempts = move_attempts
                row.update(result='command_applied', transmitted=True,
                           applied_target_deg=list(applied) if confirmed_target else None,
                           student_policy_target_deg=list(policy_target),
                           control_reference_deg=list(applied), control_reference_source=reference_source)
                if direct_student_move and retained_move is not None:
                    last_student_move = {'requested_target_deg': list(row['requested_target_deg']),
                        'target_steps': list(retained_move['target_steps']), 'task': copy.deepcopy(task),
                        'direct_active': direct_active, 'return_active': return_active}
                if damped_compensation_used:
                    row.update(student_nominal_target_deg=list(effective), nominal_arrival_confirmed=True)
                if action == 'open' and on_release_confirmed is not None:
                    # The snapshot records controller feedback, not the object's pose.
                    # Deliver it before the student is asked for the next action.
                    on_release_confirmed(copy.deepcopy(result))
                if action == 'open' and cancelled():
                    raise _Abort('cancelled', 'cancelled after a confirmed OPEN')
                if self._source_return_enabled:
                    with self._state_lock:
                        self._source_return_state['inflight_action'] = None
            else:
                result['termination'] = 'decision_limit'
        except _Abort as exc:
            result.update(termination=exc.code, error=str(exc))
            if exc.code in {'invalid_observation', 'controller_changed', 'device_warning', 'warning_unavailable',
                            'gripper_unconfirmed', 'state_changed', 'initial_target_mismatch',
                            'initial_reference_changed', 'interrupted_sdk_frame_unconfirmed'}:
                unknown = True
        except ControlError as exc:
            result.update(termination='controller_error', error=exc.as_dict())
        except Exception as exc:
            result.update(termination='error', error=type(exc).__name__)
        finally:
            if self._source_return_enabled:
                with self._state_lock:
                    if self._source_return_state is not None:
                        self._source_return_state['phase'] = 'ending'
            if in_flight:
                unknown = True
                sent_or_unknown = True
                result['last_encoded_student_target'] = None
            if lease is not None:
                # End-of-episode I/O can block. Keep the lease visible so a
                # concurrent cancel can still send its own stop immediately.
                try:
                    end = self.controller.end_episode(lease)
                    if 'no_motion_evidence' in result:
                        result['no_motion_evidence']['end_episode'] = copy.deepcopy(end)
                    if (not isinstance(end, dict) or not isinstance(end.get('stop'), dict)
                            or end['stop'].get('confirmed') is not True or end.get('fault') is not None):
                        raise _Abort('end_unconfirmed', 'controller did not confirm a fault-free stop')
                    result['end_confirmed'] = True
                except Exception as exc:
                    result['end_confirmed'] = False
                    result['previous_termination'] = result['termination']
                    result['termination'] = 'end_unconfirmed'
                    result['end_error'] = str(exc) if isinstance(exc, _Abort) else type(exc).__name__
                    unknown = True
                pregrasp_reference = result['termination'] == 'pregrasp_source_return'
                sdk_next_reference = (self.controller.config.backend == 'sdk'
                                      and result['termination'] == 'model_done')
                no_motion_reference = (self.controller.config.backend == 'sdk'
                                       and result['termination'] == 'motion_not_observed'
                                       and 'no_motion_evidence' in result)
                if result.get('end_confirmed') is True and (pregrasp_reference or sdk_next_reference or no_motion_reference):
                    try:
                        final_observation = self.controller.observe()
                        if no_motion_reference:
                            result['no_motion_evidence']['post_stop_observation'] = copy.deepcopy(final_observation)
                        instance, last_time, last_sequence = self._observation(
                            final_observation, instance, seen, last_time, last_sequence,
                            require_new_sample=True, expected_state='stopped')
                        self._gripper(final_observation, 'open' if pregrasp_reference
                                      else result['last_confirmed_gripper_command'])
                        if pregrasp_reference and self.enable_runtime_source_return:
                            self._mock_quiet_diagnostics(observation, final_observation)
                        final_joints = _joints(final_observation['joints_deg'])
                        stop = end['stop']
                        if (self._cancelled.is_set()
                                or (no_motion_reference and direct_cancel is not None and direct_cancel.is_set())
                                or final_observation['sensor_sequence'] < stop.get('confirmed_sensor_sequence', float('inf'))
                                or final_observation['observed_monotonic'] < stop.get('confirmed_sensor_monotonic', float('inf'))
                                or (self.controller.config.backend != 'sdk'
                                    and not _close(final_joints, expected_joints, drift))
                                or type(final_observation.get('revision')) is not int):
                            raise _Abort('next_reference_unconfirmed', 'fresh post-stop reference was not confirmed')
                        reference = {'joints_deg': list(final_joints),
                            'source': 'stationary_observation', 'instance_id': instance,
                            'revision': final_observation['revision'], 'physical_success': None}
                        if pregrasp_reference:
                            result['pregrasp_next_reference'] = reference
                        else:
                            result['next_control_reference'] = reference
                            if no_motion_reference:
                                result['no_motion_evidence']['post_stop_observation_validated'] = True
                            # The previous SDK target remains historical after
                            # end_episode has actually stopped and released it.
                            result.update(current_target_known=False, current_control_reference_deg=list(final_joints),
                                          current_control_reference_source='stationary_observation')
                    except Exception as exc:
                        if no_motion_reference:
                            result['no_motion_evidence']['post_stop_observation_error'] = (
                                str(exc) if isinstance(exc, _Abort) else type(exc).__name__)
                        else:
                            result['previous_termination'] = result['termination']
                            result['termination'] = ('pregrasp_reference_unconfirmed' if pregrasp_reference
                                                     else 'next_reference_unconfirmed')
                            error_key = 'pregrasp_reference_error' if pregrasp_reference else 'next_reference_error'
                            result[error_key] = str(exc) if isinstance(exc, _Abort) else type(exc).__name__
                        unknown = True
                with self._state_lock:
                    self._lease = None
                    if self._stop_error:
                        result['stop_error'] = self._stop_error
                        if result['termination'] != 'end_unconfirmed':
                            result['previous_termination'] = result['termination']
                            result['termination'] = 'stop_unconfirmed'
                        unknown = True
                    elif self._cancelled.is_set() and result['termination'] == 'model_done':
                        result['previous_termination'] = 'model_done'
                        result['termination'] = 'cancelled'
            result['layout_uncertain'] = bool(sent_or_unknown or unknown)
            if unknown:
                result['current_target_known'] = False
            if self._source_return_enabled:
                with self._state_lock:
                    state = self._source_return_state
                    if state is not None:
                        pregrasp_expected = bool(state['pregrasp_return_applied'] and not state['ever_closed']
                                        and not state['close_dispatch_started']
                                        and result['termination'] == 'pregrasp_source_return'
                                        and isinstance(result.get('pregrasp_next_reference'), dict)
                                        and result.get('end_confirmed') is True and not unknown
                                        and not self._cancelled.is_set())
                        direct_task = state.get('direct_task')
                        expected = pregrasp_expected or bool(direct_task is None and state['switched']
                                        and state['return_move_confirmed']
                                        and state['open_after_switch'] and result['model_done']
                                        and state['open_pose_before'] is not None
                                        and state['open_pose_before']['matches']
                                        and state['open_pose_after'] is not None
                                        and state['open_pose_after']['matches']
                                        and state['gripper'] == 'open'
                                        and result['last_confirmed_gripper_command'] == 'open'
                                        and result.get('end_confirmed') is True and not unknown
                                        and not self._cancelled.is_set())
                        result['pregrasp_return_expected'] = pregrasp_expected
                        result['source_return_expected'] = expected
                        result['source_return'] = {'status': ('not_requested' if direct_task is not None else
                            'pregrasp_stopped' if pregrasp_expected else 'model_done' if expected else
                            'incomplete' if state['switched'] or state['pending'] else 'not_requested'),
                            'source_slot': state['source_slot'],
                            'grasp_reference': copy.deepcopy(state.get('grasp_reference')),
                            'activation_decision_index': (state['activation_decision_index']
                                                          if direct_task is None else None),
                            'lift_status': state['lift_status'],
                            'return_move_confirmed': state['return_move_confirmed'],
                            'open_after_switch': state['open_after_switch'],
                            'open_pose_before': state['open_pose_before'],
                            'open_pose_after': state['open_pose_after'],
                            # The allowed nominal Z windows overlap between
                            # layers; no object or actual layer is observed.
                            'layer_verified': False, 'physical_success': None}
                        if direct_task is not None:
                            result['retarget'] = {
                                'mode': 'direct_held_transfer',
                                'source_slot': state['source_slot'],
                                'destination_slot': direct_task['destination_slot'],
                                'activation_decision_index': state['activation_decision_index'],
                                'source_return_expected': False,
                                'before_grasp': False,
                                'object_position_observed': False,
                                'physical_success': None}
                    self._source_return_state = None
            self._direct_previous_destination_slot = None
            self._run_lock.release()
        return result
