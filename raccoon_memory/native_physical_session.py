"""Physical-preparation binding for the shared v2 scene coordinator.

The caller supplies an already owned controller, slot registration and last
confirmed applied target. Construction, registration, confirmation and
recovery never connect to, query, start or move a device. Execution delegates
one transfer at a time to NativePhysicalStudent. A completed transfer with a
minimum confirmed close, arm-move, open sequence advances only the expected
layout, not a measured layout or a physical-success claim.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
import hashlib
import io
import math
from pathlib import Path
import threading
import time
import weakref

from raccoon_control.kinematics import JOINT_LIMITS, quantize_joints
from . import native_close_evidence_session as selected
from . import scene_demo_data, scene_features, scene_policy, scene_recall, task_catalog, task_program
from . import terra_native_close_evidence
from .native_physical_student import NativePhysicalStudent
from .scene_model_session import DuplicateRequest, SceneModelSession, UnsupportedStep, SPLIT_PLAN, plan_task
from .scene_program_coordinator import Busy, NoMotionObserved, SceneProgramCoordinator, StepUnavailable


_OWNERS_LOCK = threading.Lock()
_OWNERS = weakref.WeakValueDictionary()
# The selected student's app uses these two previously exposed structures in
# addition to the public train/validation groups. They are execution scope,
# not a claim about the checkpoint's training or an independent evaluation.
_EXPOSED_GROUPS = frozenset({'n4_s1_d2_other11', 'n4_s2_d1_other11'})
PHYSICAL_INITIAL_PROFILE = 'physical_initial'
SIMULATION_FOUR_SLOT_PROFILE = 'simulation_four_slot'


class _ScopedPathUnavailable(UnsupportedStep):
    """No complete symbolic path exists inside the selected slot scope."""


def _execution_scope(controller, profile, active_slots):
    if profile == PHYSICAL_INITIAL_PROFILE:
        # Product callers pass the validated app profile's enabled slots.
        # Keep the older default only for callers that did not supply a scope.
        permitted = (1, 2, 3, 4)
        if active_slots is None:
            active_slots = (1, 3)
    elif profile == SIMULATION_FOUR_SLOT_PROFILE:
        config = getattr(controller, 'config', None)
        if getattr(config, 'backend', None) != 'mock' or getattr(config, 'hardware_enabled', None) is not False:
            raise ValueError('the four-slot simulation profile requires a mock-only controller configuration')
        permitted = (1, 2, 3, 4)
    else:
        raise ValueError('unknown physical-preparation execution profile')
    if active_slots is None:
        return permitted
    if (not isinstance(active_slots, (tuple, list)) or not active_slots
            or any(type(slot) is not int or slot not in permitted for slot in active_slots)
            or len(set(active_slots)) != len(active_slots)):
        raise ValueError('active slots must be a nonempty, unique subset of the execution profile')
    return tuple(sorted(active_slots))


def _scoped_path_to(start, goal_test, train_groups, active_slots):
    """Search entire symbolic paths while keeping every transfer in scope."""
    start = scene_recall.scene(start)
    queue, previous = deque([start]), {start: None}
    while queue:
        before = queue.popleft()
        if goal_test(before):
            path = []
            while previous[before] is not None:
                parent, action = previous[before]
                path.append(action)
                before = parent
            return list(reversed(path))
        for source in active_slots:
            for destination in active_slots:
                try:
                    after = task_catalog.transfer(before, source, destination)
                except ValueError:
                    continue
                action = {'source_slot': source, 'destination_slot': destination}
                if after not in previous and scene_policy.split_group_for(before, action) in train_groups:
                    previous[after] = (before, action)
                    queue.append(after)
    raise _ScopedPathUnavailable('no complete path inside the active slot scope')


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _has_applied_transfer_commands(result):
    """Check command evidence only; the object is not observed by this session."""
    decisions = result.get('decisions') if isinstance(result, dict) else None
    if not isinstance(decisions, list):
        return False
    profile = result.get('control_profile')
    physical_position = (isinstance(profile, dict) and profile.get('student_endpoint') in {
        'normal_original_quantized_target_no_alternate_correction',
        'normal_original_nominal_with_bounded_damped_compensation'})
    closed = moved_while_closed = False
    for index, decision in enumerate(decisions):
        if (not isinstance(decision, dict) or decision.get('result') != 'command_applied'
                or decision.get('transmitted') is not True):
            continue
        action = decision.get('action')
        if action == 'close':
            if not closed:
                moved_while_closed = False
            closed = True
        elif action == 'move' and closed:
            if physical_position:
                # Encoding a new, in-tolerance SDK target can establish hold
                # with no arm motion. Use the actual observations already in
                # the student record, not the difference between two goals.
                completion = decision.get('completion')
                features = decision.get('policy_features')
                arrivals = decision.get('arrival_observations')
                start = ([value * 180. for value in features[:4]]
                         if isinstance(features, (list, tuple)) and len(features) >= 4
                         and all(_finite_number(value) for value in features[:4]) else None)
                end = (arrivals[-1].get('observed_joints_deg') if isinstance(arrivals, list) and arrivals
                       and isinstance(arrivals[-1], dict) else None)
                if end is None and index + 1 < len(decisions) and isinstance(decisions[index + 1], dict):
                    following = decisions[index + 1].get('policy_features')
                    if (isinstance(following, (list, tuple)) and len(following) >= 4
                            and all(_finite_number(value) for value in following[:4])):
                        end = [value * 180. for value in following[:4]]
                if (isinstance(completion, dict) and completion.get('mode') == 'position_target_retained'
                        and start is not None and isinstance(end, (list, tuple)) and len(end) == 4
                        and all(_finite_number(value) for value in end)
                        and any(abs(a - b) > 1e-7 for a, b in zip(start, end))):
                    moved_while_closed = True
                continue
            previous = decision.get('previous_control_reference_deg')
            if previous is None:
                previous = decision.get('previous_applied_target_deg')
            target = decision.get('applied_target_deg')
            if (isinstance(previous, (list, tuple)) and isinstance(target, (list, tuple))
                    and len(previous) == len(target) == 4
                    and all(_finite_number(value) for value in (*previous, *target))
                    and any(abs(a - b) > 1e-7 for a, b in zip(previous, target))):
                moved_while_closed = True
        elif action == 'open':
            if closed and moved_while_closed:
                return True
            closed = moved_while_closed = False
    return False


def _confirmed_no_motion(result, controller):
    """Use only a complete, fresh SDK record for the first MOVE with open fingers."""
    if (getattr(controller.config, 'backend', None) != 'sdk' or not isinstance(result, dict)
            or result.get('termination') != 'motion_not_observed' or result.get('model_done') is not False
            or result.get('physical_success') is not None or result.get('end_confirmed') is not True
            or result.get('last_confirmed_gripper_command') != 'open'
            or result.get('stop_error') or result.get('lease_state_uncertain')):
        return None
    evidence, decisions = result.get('no_motion_evidence'), result.get('decisions')
    if (not isinstance(evidence, dict) or evidence.get('post_stop_observation_validated') is not True
            or evidence.get('gripper_commands') != [] or not isinstance(decisions, list)
            or len(decisions) != 1 or not isinstance(decisions[0], dict)
            or decisions[0].get('action') != 'move'):
        return None
    attempts, end = evidence.get('attempts'), evidence.get('end_episode')
    initial, start, final = (evidence.get(key) for key in
                              ('initial_observation', 'move_start_observation', 'post_stop_observation'))
    if (not isinstance(attempts, list) or len(attempts) != 2
            or not isinstance(end, dict) or end.get('fault') is not None
            or not isinstance(end.get('stop'), dict) or end['stop'].get('confirmed') is not True
            or not all(isinstance(item, dict) for item in (initial, start, final))
            or start.get('observation_id') != initial.get('observation_id')):
        return None
    observations, targets = [initial], []
    for attempt in attempts:
        if (not isinstance(attempt, dict) or attempt.get('status') != 'settled_no_motion'
                or attempt.get('applied') is not True or attempt.get('replayed') is not False
                or not isinstance(attempt.get('completion'), dict)
                or attempt['completion'].get('motion_observed') is not False
                or attempt.get('nominal_arrival_reached') is not False
                or not isinstance(attempt.get('target'), dict)):
            return None
        targets.append(attempt['target'].get('steps'))
        observations.append(attempt.get('observation'))
    observations.append(final)
    if (not isinstance(targets[0], (list, tuple)) or len(targets[0]) != 4
            or targets[0] != targets[1]):
        return None
    identity, previous_sequence, previous_time, initial_steps = None, 0, -1, None
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            return None
        joints, sequence = observation.get('joints_deg'), observation.get('sensor_sequence')
        observed, diagnostics = observation.get('observed_monotonic'), observation.get('device_diagnostics')
        if (observation.get('sensor_valid') is not True or observation.get('moving') is not False
                or observation.get('state') != ('stopped' if index == len(observations)-1 else 'armed')
                or observation.get('gripper_command') != 'open'
                or observation.get('gripper_reported_state') != 'open'
                or observation.get('gripper_reported_valid') is not True
                or observation.get('gripper_feedback_fresh') is not True
                or not isinstance(diagnostics, dict) or diagnostics.get('firmware_warning_raw') != 0
                or type(sequence) is not int or sequence <= previous_sequence
                or not _finite_number(observed) or observed < previous_time
                or not isinstance(joints, (list, tuple)) or len(joints) != 4
                or not all(_finite_number(value) for value in joints)):
            return None
        if identity is None:
            identity = observation.get('instance_id')
            if not isinstance(identity, str) or not identity:
                return None
            initial_steps = quantize_joints(joints)[0]
        elif observation.get('instance_id') != identity:
            return None
        if any(abs(a-b) > 1 for a, b in zip(quantize_joints(joints)[0], initial_steps)):
            return None
        previous_sequence, previous_time = sequence, observed
    reference = result.get('next_control_reference')
    if (not isinstance(reference, dict) or reference.get('source') != 'stationary_observation'
            or reference.get('instance_id') != identity or reference.get('revision') != final.get('revision')
            or reference.get('joints_deg') != final.get('joints_deg')):
        return None
    try:
        _initial_reference(reference['joints_deg'], 'stationary_observation', identity, reference['revision'])
    except (TypeError, ValueError):
        return None
    return reference


def _slot_coordinates(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('four explicitly registered XY coordinates are required')
    result = []
    for xy in value:
        if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                or any(not _finite_number(x) for x in xy)):
            raise ValueError('each registered XY coordinate must contain two finite numbers')
        result.append(tuple(float(x) for x in xy))
    return tuple(result)


def _applied_target(value, *, require_quantized=True):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(not _finite_number(x) for x in value)
            or any(not low <= x <= high for x, (low, high) in zip(value, JOINT_LIMITS))):
        raise ValueError('four finite, in-range applied joint targets are required')
    target = [float(x) for x in value]
    _, quantized = quantize_joints(target)
    if require_quantized and any(abs(a - b) > 1e-7 for a, b in zip(target, quantized)):
        raise ValueError('the caller-confirmed applied target must already be quantized')
    return target


def _initial_reference(value, source, instance_id, revision):
    if source not in ('caller_confirmed', 'controller_confirmed_move', 'stationary_observation'):
        raise ValueError('invalid initial control reference source')
    observed = source == 'stationary_observation'
    if observed and (not isinstance(instance_id, str) or not instance_id.strip()
                     or type(revision) is not int or revision < 0):
        raise ValueError('stationary observation identity and revision are required')
    if not observed and (instance_id is not None or revision is not None):
        raise ValueError('observation identity is only valid for a stationary observation')
    return _applied_target(value, require_quantized=not observed)


def _encoded_student_reference(value, stopped_reference):
    """Keep a historical SDK target only when its host frame is identified."""
    if (not isinstance(value, dict) or value.get('source') != 'sdk_host_frame'
            or value.get('instance_id') != stopped_reference.get('instance_id')
            or not isinstance(value.get('command_id'), str) or not value['command_id'].strip()):
        return None
    frame, submission = value.get('host_frame'), value.get('submission')
    try:
        joints = _applied_target(value.get('joints_deg'))
        steps, _ = quantize_joints(joints)
    except (TypeError, ValueError):
        return None
    if (not isinstance(frame, dict) or frame.get('mode_hex') != '11'
            or not isinstance(frame.get('joint_steps'), (list, tuple))
            or list(frame['joint_steps']) != list(steps)
            or type(frame.get('frame_sequence')) is not int or frame['frame_sequence'] < 1
            or type(frame.get('command_sequence')) is not int or not 0 <= frame['command_sequence'] <= 255
            or type(frame.get('target_segment_start_sequence')) is not int
            or not 1 <= frame['target_segment_start_sequence'] <= frame['frame_sequence']
            or not _finite_number(frame.get('monotonic'))
            or type(frame.get('speed_pct')) is not int
            or not isinstance(submission, dict)
            or type(submission.get('encoding_after_frame_sequence')) is not int
            or submission['encoding_after_frame_sequence'] >= frame['frame_sequence']
            or not _finite_number(submission.get('submitted_monotonic'))
            or submission['submitted_monotonic'] > frame['monotonic']):
        return None
    return {**copy.deepcopy(value), 'joints_deg': joints,
            'stopped_revision': stopped_reference['revision']}


class _ForwardingCancel(threading.Event):
    """Forward coordinator cancellation once to the currently bound adapter."""

    def __init__(self):
        super().__init__()
        self._adapter_lock = threading.Lock()
        self._adapter = None
        self._forwarded = False

    def _take_adapter(self):
        with self._adapter_lock:
            if self._adapter is None or self._forwarded or not self.is_set():
                return None
            self._forwarded = True
            return self._adapter

    @staticmethod
    def _forward(adapter):
        if adapter is not None:
            # NativePhysicalStudent.cancel records an unconfirmed stop itself.
            adapter.cancel()

    def set(self):
        super().set()
        self._forward(self._take_adapter())

    def bind(self, adapter):
        with self._adapter_lock:
            if self._adapter is not None:
                raise RuntimeError('a physical transfer is already bound')
            self._adapter = adapter
            self._forwarded = False
        self._forward(self._take_adapter())

    def unbind(self, adapter):
        with self._adapter_lock:
            if self._adapter is adapter:
                self._adapter = None
                self._forwarded = False

    def request_mock_source_return(self, source_slot, *, cancel_event=None, replacement_task=None):
        with self._adapter_lock:
            if self.is_set() or self._adapter is None or (cancel_event is not None and cancel_event.is_set()):
                return {'status': 'retarget_rejected', 'reason': 'transfer_not_active'}
            adapter = self._adapter
        # Do not hold the binding lock while a synchronous controller command
        # reaches its boundary; cancellation must still reach this adapter.
        if replacement_task is not None:
            return adapter.request_mock_source_return(source_slot=source_slot, cancel_event=cancel_event,
                                                      replacement_task=replacement_task)
        return adapter.request_mock_source_return(source_slot=source_slot, cancel_event=cancel_event)

    def mock_source_return_status(self):
        with self._adapter_lock:
            adapter = self._adapter
        inactive = {'status': 'inactive', 'source_slot': None,
                    'activation_decision_index': None, 'physical_success': None}
        if adapter is None:
            return inactive
        # Query outside the binding lock: observing progress must not delay a
        # cancellation or hold the adapter's lifetime across a controller call.
        status = adapter.mock_source_return_status()
        with self._adapter_lock:
            return status if self._adapter is adapter else inactive


class NativePhysicalSession(SceneModelSession):
    """Reuse the symbolic planner and memory with an injected physical controller.

    The caller must establish its workspace and controller independently. The
    initial control reference is either a confirmed applied target or an
    explicitly identified stationary observation. The caller validates that
    evidence. This class owns no device life
    cycle and must not be used as an automatic safety or success assessment.
    """

    def __init__(self, controller, memory, student, *, slot_xy_mm,
                 initial_applied_target_deg, speed_pct, max_decisions=60,
                 enable_arrival_correction=False, enable_blended_path=False, enable_release_setdown=False,
                 initial_target_source='caller_confirmed', initial_observation_instance_id=None,
                 initial_observation_revision=None,
                 execution_profile=PHYSICAL_INITIAL_PROFILE, active_slots=None,
                 compare_virtual_descent=False, execute_mock_virtual_descent=False,
                 path_calibration=None, enable_experimental_segmented_path=False,
                 enable_mock_source_return=False, enable_mock_interrupt_retarget=False,
                 enable_runtime_source_return=False, enable_mock_nominal_trim=False,
                 enable_staged_destination_retarget=False,
                 enable_mock_layered_grasp_lowering=False, enable_mock_early_source_return=False,
                 enable_mock_open_drift_reconcile=False, enable_mock_vertical_replan=False,
                 enable_mock_move_release_upper_separation=False):
        slots = _slot_coordinates(slot_xy_mm)
        target = _initial_reference(initial_applied_target_deg, initial_target_source,
                                    initial_observation_instance_id, initial_observation_revision)
        active_slots = _execution_scope(controller, execution_profile, active_slots)
        config = getattr(controller, 'config', None)
        if type(enable_mock_source_return) is not bool:
            raise ValueError('mock source return flag must be boolean')
        if type(enable_mock_interrupt_retarget) is not bool:
            raise ValueError('mock interruption flag must be boolean')
        if type(enable_runtime_source_return) is not bool:
            raise ValueError('runtime source return flag must be boolean')
        if type(enable_staged_destination_retarget) is not bool:
            raise ValueError('staged destination retarget flag must be boolean')
        if type(enable_mock_nominal_trim) is not bool:
            raise ValueError('mock nominal trim flag must be boolean')
        if (type(enable_mock_layered_grasp_lowering) is not bool or type(enable_mock_early_source_return) is not bool
                or type(enable_mock_open_drift_reconcile) is not bool or type(enable_mock_vertical_replan) is not bool
                or type(enable_mock_move_release_upper_separation) is not bool):
            raise ValueError('mock grasp and early return flags must be boolean')
        if enable_runtime_source_return and execution_profile != PHYSICAL_INITIAL_PROFILE:
            raise ValueError('runtime source return requires the physical execution profile')
        if enable_mock_interrupt_retarget and not enable_mock_source_return:
            raise ValueError('mock interruption requires mock source return')
        if enable_staged_destination_retarget and not (
                (enable_mock_source_return and enable_mock_interrupt_retarget or enable_runtime_source_return)
                and path_calibration is not None and enable_experimental_segmented_path):
            raise ValueError('staged destination retarget requires interruption and calibrated paths')
        if enable_mock_source_return and (execution_profile != SIMULATION_FOUR_SLOT_PROFILE
                or getattr(config, 'backend', None) != 'mock'
                or getattr(config, 'hardware_enabled', None) is not False
                or getattr(config, 'port', None) is not None):
            raise ValueError('source return research requires an isolated mock controller')
        self.supports_mock_source_return = enable_mock_source_return
        self.supports_mock_interrupt_retarget = enable_mock_interrupt_retarget
        self.supports_runtime_source_return = enable_runtime_source_return
        self.supports_staged_destination_retarget = enable_staged_destination_retarget
        self.enable_mock_nominal_trim = enable_mock_nominal_trim
        self.enable_mock_layered_grasp_lowering = enable_mock_layered_grasp_lowering
        self.enable_mock_early_source_return = enable_mock_early_source_return
        self.enable_mock_open_drift_reconcile = enable_mock_open_drift_reconcile
        self.enable_mock_vertical_replan = enable_mock_vertical_replan
        self.enable_mock_move_release_upper_separation = enable_mock_move_release_upper_separation
        if type(max_decisions) is not int or not 1 <= max_decisions <= 60:
            raise ValueError('max_decisions must be between 1 and 60')
        # Validate the adapter configuration without acquiring or querying it.
        NativePhysicalStudent(controller, student, speed_pct=speed_pct, max_decisions=max_decisions,
                              **({'enable_arrival_correction': True} if enable_arrival_correction else {}),
                              **({'enable_blended_path': True} if enable_blended_path else {}),
                              **({'enable_release_setdown': True} if enable_release_setdown else {}),
                              compare_virtual_descent=compare_virtual_descent,
                              execute_mock_virtual_descent=execute_mock_virtual_descent,
                              path_calibration=path_calibration,
                              enable_experimental_segmented_path=enable_experimental_segmented_path,
                              **({'enable_mock_source_return': True} if enable_mock_source_return else {}),
                              **({'enable_mock_interrupt_retarget': True} if enable_mock_interrupt_retarget else {}),
                              **({'enable_runtime_source_return': True} if enable_runtime_source_return else {}),
                              **({'enable_mock_nominal_trim': True} if enable_mock_nominal_trim else {}),
                              **({'enable_mock_layered_grasp_lowering': True} if enable_mock_layered_grasp_lowering else {}),
                              **({'enable_mock_early_source_return': True} if enable_mock_early_source_return else {}),
                              **({'enable_mock_open_drift_reconcile': True} if enable_mock_open_drift_reconcile else {}),
                              **({'enable_mock_vertical_replan': True} if enable_mock_vertical_replan else {}),
                              **({'enable_mock_move_release_upper_separation': True}
                                 if enable_mock_move_release_upper_separation else {}))
        super().__init__(None, memory, student, device='cpu', max_decisions=max_decisions,
                         cancel=_ForwardingCancel())
        split, _, _ = scene_demo_data.read_split_plan(SPLIT_PLAN)
        self.train_groups = frozenset(split['train_groups']) | frozenset(split['validation_groups']) | _EXPOSED_GROUPS
        self.checkpoint_train_groups = None
        self.controller, self.speed_pct = controller, speed_pct
        self.enable_arrival_correction = enable_arrival_correction
        self.enable_blended_path = enable_blended_path
        self.enable_release_setdown = enable_release_setdown
        self.compare_virtual_descent = compare_virtual_descent
        self.execute_mock_virtual_descent = execute_mock_virtual_descent
        self.path_calibration = path_calibration
        self.enable_experimental_segmented_path = enable_experimental_segmented_path
        self._execution_profile, self._active_slots = execution_profile, active_slots
        self._registered_slot_xy_mm = slots
        self._current_target = target
        self._current_target_source = initial_target_source
        self._current_nominal_target = None
        self._last_encoded_student_target = None
        self._initial_observation_instance_id = initial_observation_instance_id
        self._initial_observation_revision = initial_observation_revision
        self._last_registered_scene = self._current_registered_scene = None
        self._last_confirmed_scene = self._confirmed_current_scene = None
        self._unresolved_transfer = None
        self._staged_redirect = None
        self._pending_release_recovery = None
        self._connection_recovery_revision = 0
        self._scene_view_lock = threading.RLock()
        with _OWNERS_LOCK:
            if _OWNERS.get(id(controller)) is not None:
                raise RuntimeError('this controller already has a live physical session')
            _OWNERS[id(controller)] = self

    def _registered(self, expected):
        if self.status != 'registered' or self.current != expected or self._current_target is None:
            raise RuntimeError('the expected scene or initial control reference is unavailable')
        return {'expected_scene': expected, 'physical_confirmation': 'not_assessed_by_session'}

    def retarget_capabilities(self):
        """Advertise the executor's current scope without asserting success."""
        enabled = self.supports_mock_source_return or self.supports_runtime_source_return
        progress = self.cancel.mock_source_return_status() if enabled else None
        with self._scene_view_lock:
            transfer = self._unresolved_transfer
            source = transfer.get('source_slot') if isinstance(transfer, dict) else None
            redirect_available = (self.supports_staged_destination_retarget
                                  and self._staged_redirect is None
                                  and isinstance(progress, dict)
                                  and progress.get('staged_redirect_available') is True)
        return {'available': enabled, 'return_to_source': enabled,
                'destinations': (list(self._active_slots) if redirect_available else
                                 [source] if enabled and source in self._active_slots else [])}

    def _register(self, value):
        if self.status == 'uncertain':
            raise RuntimeError('uncertain sessions require explicit recovery')
        if self.cancel.is_set():
            raise InterruptedError('registration cancelled')
        registered = scene_recall.scene(value)
        self.memory.clear()
        self.memory.observe(registered, write=False)
        if self.cancel.is_set():
            raise InterruptedError('registration cancelled')
        with self._scene_view_lock:
            self.current = registered
            self.status = 'registered'
            self._last_registered_scene = self._current_registered_scene = registered
            self._last_confirmed_scene = self._confirmed_current_scene = None
            self._unresolved_transfer = None
            self._pending_release_recovery = None
            self._last_encoded_student_target = None
            self.events.append({'event': 'registered', 'scene': registered, 'memory_cleared': True,
                                'scene_source': 'user_declared', 'physical_assessment_performed': False,
                                'control_target_source': self._current_target_source})

    def _halt(self, reason):
        with self._scene_view_lock:
            self.status = 'uncertain'
            self._current_registered_scene = None
            self._current_target = None
            self._current_target_source = None
            self._current_nominal_target = None
            self._last_encoded_student_target = None
            self._initial_observation_instance_id = self._initial_observation_revision = None
            if self._unresolved_transfer is not None:
                self._confirmed_current_scene = None

    def note_stationary_observation(self, observation):
        """Bind a caller-confirmed stopped observation, without moving the arm."""
        target = _initial_reference(observation.get('joints_deg'), 'stationary_observation',
                                    observation.get('instance_id'), observation.get('revision'))
        with self._scene_view_lock:
            # A stopped transfer remains uncertain until its object outcome is
            # resolved; a stationary robot observation does not resolve it.
            if self.status == 'registered' and not self.cancel.is_set():
                self._confirmed_current_scene = None
                self._current_target = target
                self._current_target_source = 'stationary_observation'
                self._current_nominal_target = None
                self._last_encoded_student_target = None
                self._initial_observation_instance_id = observation['instance_id']
                self._initial_observation_revision = observation['revision']
                return True
            return False

    def note_arm_home(self, observation):
        """Update only a known control reference; never infer an object outcome."""
        self.note_stationary_observation(observation)

    def rebind_controller(self, controller, observation):
        """Bind a replacement while idle, retaining scene and memory unchanged.

        The caller owns the session execution lock and has already obtained the
        new controller's stationary observation. This never commands the arm.
        An unresolved transfer remains uncertain after rebinding.
        """
        if not self._lock.locked():
            raise RuntimeError('controller replacement requires the idle session lock')
        target = _initial_reference(observation.get('joints_deg'), 'stationary_observation',
                                    observation.get('instance_id'), observation.get('revision'))
        if (observation.get('instance_id') != getattr(controller, 'instance_id', None)
                or observation.get('sensor_valid') is not True or observation.get('moving') is not False):
            raise ValueError('replacement requires a stopped controller observation')
        previous = self.controller
        with _OWNERS_LOCK, self._scene_view_lock:
            if _OWNERS.get(id(previous)) is not self:
                raise RuntimeError('previous controller is no longer owned by this session')
            owner = _OWNERS.get(id(controller))
            if owner is not None and owner is not self:
                raise RuntimeError('replacement controller already has a live session')
            _OWNERS.pop(id(previous), None)
            _OWNERS[id(controller)] = self
            self.controller = controller
            self._current_target = target if self.status == 'registered' else None
            self._current_target_source = 'stationary_observation' if self.status == 'registered' else None
            self._current_nominal_target = None
            self._last_encoded_student_target = None
            self._initial_observation_instance_id = observation['instance_id']
            self._initial_observation_revision = observation['revision']
            self._confirmed_current_scene = None
            pending = self._pending_release_recovery
            if (isinstance(pending, dict)
                    and pending.get('instance_id') == getattr(previous, 'instance_id', None)):
                # The evidence about which scene is expected belongs to the
                # same interrupted session, even though the current stopped
                # observation now belongs to a new controller instance.
                pending['instance_id'] = observation['instance_id']
                pending['replacement'] = True
            if self.status == 'registered':
                self._connection_recovery_revision += 1
            # A prior incomplete transfer remains unresolved. In particular,
            # reading the joints cannot determine where an object ended up.
            self.events.append({'event': 'controller_rebound', 'replayed_commands': 0,
                                'scene_status': self.status, 'physical_success': None})

    def scene_state(self):
        retarget = self.cancel.mock_source_return_status() if (self.supports_mock_source_return
                                                               or self.supports_runtime_source_return) else None
        with self._scene_view_lock:
            unresolved = copy.deepcopy(self._unresolved_transfer)
            redirect = self._staged_redirect
            if redirect is not None:
                direct_applied = (redirect.get('accepted') is True and isinstance(retarget, dict)
                                  and retarget.get('mode') == 'direct_held_transfer'
                                  and retarget.get('destination_slot') == redirect['destination_slot']
                                  and retarget.get('status') == 'applied')
                retarget = {'status': ('applied' if direct_applied else
                                       'requested' if redirect.get('accepted') is True else 'waiting'),
                            'source_slot': redirect['source_slot'],
                            'destination_slot': redirect['destination_slot'],
                            'activation_decision_index': (retarget.get('activation_decision_index')
                                                          if direct_applied else None),
                            'physical_success': None}
            if unresolved is not None:
                unresolved['status'] = 'uncertain' if self.status == 'uncertain' else 'in_progress'
            registered = copy.deepcopy(self._current_registered_scene)
            confirmed = copy.deepcopy(self._confirmed_current_scene)
            return {'mode': ('simulation_expected_scene' if self._execution_profile == SIMULATION_FOUR_SLOT_PROFILE
                             else 'physical_preparation_expected_scene'),
                    'execution_profile': self._execution_profile, 'active_slots': list(self._active_slots),
                    'maximum_stack_height': scene_recall.MAX_HEIGHT, 'expected_scene': copy.deepcopy(self.current),
                    'current_registered_scene': registered,
                    'last_registered_scene': copy.deepcopy(self._last_registered_scene),
                    'registration_source': 'user_declared' if self._last_registered_scene is not None else None,
                    'confirmed_current_scene': confirmed, 'last_confirmed_scene': copy.deepcopy(self._last_confirmed_scene),
                    'confirmation': ('explicit_registration' if registered is not None
                                     else 'caller_confirmed' if confirmed is not None
                                     else 'unconfirmed_after_motion' if self._last_registered_scene is not None
                                     else 'not_registered'),
                    'physical_confirmation': 'caller_confirmed' if confirmed is not None else 'not_inspected',
                    'unresolved_transfer': unresolved, 'retarget': retarget, 'physical_success': None,
                    'current_applied_target_deg': (copy.deepcopy(self._current_target)
                        if self._current_target_source in ('caller_confirmed', 'controller_confirmed_move',
                                                           'controller_confirmed_nominal_move') else None),
                    'current_nominal_target_deg': copy.deepcopy(self._current_nominal_target),
                    'current_control_reference_deg': copy.deepcopy(self._current_target),
                    'control_target_source': self._current_target_source,
                    'connection_recovery_revision': self._connection_recovery_revision,
                    'connection_recovery_pending': (self._pending_release_recovery or {}).get('basis')
                        or ('expected_after_release' if self._pending_release_recovery else None)}

    def recover_expected_after_disconnect(self, observation, *, replacement=False):
        """Restore a known expected scene from a fresh stopped controller reference."""
        with self._scene_view_lock:
            pending = self._pending_release_recovery
            basis = pending.get('basis', 'expected_after_release') if isinstance(pending, dict) else None
            if (self.status != 'uncertain' or self.cancel.is_set() or not isinstance(pending, dict)
                    or self.current != pending['expected_scene']
                    or basis not in ('expected_after_release', 'before_grasp', 'arm_only')
                    or (basis == 'expected_after_release' and
                        (self._unresolved_transfer is None
                         or self._unresolved_transfer.get('release_expected') is not True))
                    or (basis == 'before_grasp' and
                        (self._unresolved_transfer is None
                         or self._unresolved_transfer.get('release_expected') is True))
                    or (basis == 'arm_only' and self._unresolved_transfer is not None)):
                return False
            states = ('idle', 'stopped') if replacement else ('stopped',)
            gripper_commands = ('open', None) if replacement else ('open',)
            if (not isinstance(observation, dict) or observation.get('state') not in states
                    or observation.get('instance_id') != pending['instance_id']
                    or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                    or observation.get('gripper_command') not in gripper_commands
                    or observation.get('gripper_reported_state') != 'open'
                    or observation.get('gripper_reported_valid') is not True
                    or observation.get('gripper_feedback_fresh') is not True
                    or (observation.get('device_diagnostics') or {}).get('firmware_warning_raw') != 0):
                return False
            observed, age, sequence = (observation.get('observed_monotonic'),
                                       observation.get('sensor_age_seconds'), observation.get('sensor_sequence'))
            now = time.monotonic()
            if (not _finite_number(observed) or observed < 0 or observed > now
                    or now - observed > self.controller.config.limits.observation_max_age
                    or not _finite_number(age) or age < 0
                    or age + now - observed > self.controller.config.limits.sensor_max_age
                    or type(sequence) is not int or sequence < 1):
                return False
            try:
                target = _initial_reference(observation.get('joints_deg'), 'stationary_observation',
                                            observation.get('instance_id'), observation.get('revision'))
            except (TypeError, ValueError):
                return False
            self.status = 'registered'
            self._current_target, self._current_target_source = target, 'stationary_observation'
            self._current_nominal_target = None
            self._last_encoded_student_target = None
            self._initial_observation_instance_id = observation['instance_id']
            self._initial_observation_revision = observation['revision']
            self._current_registered_scene = self._confirmed_current_scene = None
            self._unresolved_transfer = None
            self._pending_release_recovery = None
            self._connection_recovery_revision += 1
            self.events.append({'event': 'connection_recovered', 'scene_source': basis,
                                'physical_success': None, 'replayed_commands': 0})
            return True

    def _scoped_plan_task(self, start, step, *, validation_source=None, train_groups=None):
        # Use the shared planner for task semantics and all its validation.
        # Its first unrestricted goal for an arrangement is only one possible
        # count-matching scene, so search the scoped state space for any match.
        groups = self.train_groups if train_groups is None else train_groups
        goal, _ = plan_task(start, step, groups, validation_source=validation_source)
        if step['kind'] == 'arrange':
            counts = {entry['slot'] - 1: entry['count'] for entry in step['layout']}
            predicate = lambda scene: all(len(scene[slot]) == count for slot, count in counts.items())
        else:
            predicate = lambda scene: scene == goal
        path = _scoped_path_to(start, predicate, groups, self._active_slots)
        scoped_goal = start
        for action in path:
            scoped_goal = task_catalog.transfer(scoped_goal, action['source_slot'], action['destination_slot'])
        return scoped_goal, path

    def _scoped_goal_path(self, start, goal, *, train_groups=None):
        goal = scene_recall.scene(goal, scene_recall.identities(start))
        groups = self.train_groups if train_groups is None else train_groups
        return _scoped_path_to(start, lambda scene: scene == goal, groups, self._active_slots)

    def _execute_validated_step(self, step, *, registration_scene=None, registered_initial=None,
                                task_start=None, commit_lock=None, is_current=None,
                                before_transfer=None, after_transfer=None, on_transfer_commit=None,
                                planned_tasks=False, validation_source=None):
        # In particular, an arrangement must select its goal within the slot
        # scope, rather than reject a different goal chosen by the shared BFS.
        with self._memory_binding_lock:
            if step['op'] == 'task' and step['kind'] not in ('register', 'remember') and planned_tasks:
                self._ready()
                goal, path = self._scoped_plan_task(self.current, step, validation_source=validation_source)
                self.events.append({'event': 'symbolic_plan', 'steps': len(path),
                                    'planner': 'scoped_breadth_first_search_not_learned',
                                    'goal_source': 'count_constraints' if step['kind'] == 'arrange' else 'explicit_task',
                                    'planned_goal': goal, 'active_slots': list(self._active_slots)})
                self._execute_path(path, commit_lock, is_current, before_transfer, after_transfer,
                                   on_transfer_commit)
                return
            return super()._execute_validated_step(
                step, registration_scene=registration_scene, registered_initial=registered_initial,
                task_start=task_start, commit_lock=commit_lock, is_current=is_current,
                before_transfer=before_transfer, after_transfer=after_transfer,
                on_transfer_commit=on_transfer_commit, planned_tasks=planned_tasks,
                validation_source=validation_source)

    def _execute_path(self, path, commit_lock, is_current, before_transfer, after_transfer,
                      on_transfer_commit=None):
        # Returns, including decoded GRU goals unavailable at admission, are
        # checked and, when possible, rerouted before their first transfer.
        goal = self.current
        for action in path:
            goal = task_catalog.transfer(goal, action['source_slot'], action['destination_slot'])
        checked = self._scoped_goal_path(self.current, goal)
        return super()._execute_path(checked, commit_lock, is_current, before_transfer,
                                     after_transfer, on_transfer_commit)

    def execute(self, *args, **kwargs):
        # The older direct v1 entry point does not preflight a whole compound
        # program. Require the coordinator, which owns that admission boundary.
        raise UnsupportedStep('physical preparation requires NativePhysicalCoordinator')

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        self._ready()
        if (action.get('source_slot') not in self._active_slots
                or action.get('destination_slot') not in self._active_slots):
            raise _ScopedPathUnavailable('transfer is outside the active slot scope')
        before, task = scene_features.transfer_task(self.current, action)
        after = task_catalog.transfer(before, task['source_slot'], task['destination_slot'])
        group = scene_policy.split_group_for(before, task)
        if group not in self.train_groups:
            raise UnsupportedStep(f'transfer is outside allowed groups: {group}')
        event = {'event': 'transfer', 'before': before, 'planned_after': after, 'task': task,
                 'split_group': group, 'result': None, 'committed': False,
                 'commit_kind': 'expected_only', 'physical_success': None}
        self.events.append(event)
        def make_adapter(*, redirected=False):
            return NativePhysicalStudent(self.controller, self.student, speed_pct=self.speed_pct,
                                        max_decisions=self.max_decisions,
                                        require_retarget_release_check=redirected,
                                        **({'enable_arrival_correction': True} if self.enable_arrival_correction else {}),
                                        **({'enable_blended_path': True} if self.enable_blended_path else {}),
                                        **({'enable_release_setdown': True} if self.enable_release_setdown else {}),
                                        compare_virtual_descent=self.compare_virtual_descent,
                                        execute_mock_virtual_descent=self.execute_mock_virtual_descent,
                                        path_calibration=self.path_calibration,
                                        enable_experimental_segmented_path=self.enable_experimental_segmented_path,
                                        **({'enable_mock_source_return': True} if self.supports_mock_source_return else {}),
                                        **({'enable_mock_interrupt_retarget': True} if self.supports_mock_interrupt_retarget else {}),
                                        **({'enable_runtime_source_return': True} if self.supports_runtime_source_return else {}),
                                        **({'enable_mock_nominal_trim': True} if self.enable_mock_nominal_trim else {}),
                                        **({'enable_mock_layered_grasp_lowering': True} if self.enable_mock_layered_grasp_lowering else {}),
                                        **({'enable_mock_early_source_return': True} if self.enable_mock_early_source_return else {}),
                                        **({'enable_mock_open_drift_reconcile': True} if self.enable_mock_open_drift_reconcile else {}),
                                        **({'enable_mock_vertical_replan': True} if self.enable_mock_vertical_replan else {}),
                                        **({'enable_mock_move_release_upper_separation': True}
                                           if self.enable_mock_move_release_upper_separation else {}))
        adapter = make_adapter()
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError('request invalidated before transfer')
            with self._scene_view_lock:
                initial_target = self._current_target
                initial_source = self._current_target_source
                initial_nominal = self._current_nominal_target
                initial_instance = self._initial_observation_instance_id
                initial_revision = self._initial_observation_revision
                encoded = self._last_encoded_student_target
                previous_student_target = (list(encoded['joints_deg']) if isinstance(encoded, dict)
                    and initial_source == 'stationary_observation'
                    and encoded.get('instance_id') == initial_instance
                    and encoded.get('stopped_revision') == initial_revision else None)
                initial_registered_scene = self._current_registered_scene
                initial_confirmed_scene = self._confirmed_current_scene
                self._current_target = None
                self._current_target_source = None
                self._current_nominal_target = None
                # Do not reuse an older target if this transfer's outcome is unknown.
                self._last_encoded_student_target = None
                self._initial_observation_instance_id = self._initial_observation_revision = None
                self._current_registered_scene = self._confirmed_current_scene = None
                self._unresolved_transfer = {'expected_object_id': task['object_id'],
                    'object_id_source': 'expected_scene_source_top', 'physical_object_id': None,
                    'source_slot': task['source_slot'], 'destination_slot': task['destination_slot']}
                self._staged_redirect = None
        def on_release_confirmed(partial):
            # OPEN is a controller report, not an object observation. Keep an
            # expected scene as soon as the existing transport evidence exists.
            rows = partial.get('decisions') if isinstance(partial, dict) else None
            last = rows[-1] if isinstance(rows, list) and rows else None
            delivery = last.get('gripper_delivery') if isinstance(last, dict) else None
            if (getattr(self.controller.config, 'backend', None) != 'sdk'
                    or partial.get('last_confirmed_gripper_command') != 'open'
                    or not isinstance(delivery, dict) or delivery.get('state') != 'open'
                    or delivery.get('reported_confirmed') is not True
                    or delivery.get('observation_verified') is not True
                    or not _has_applied_transfer_commands(partial)):
                return
            with commit_lock if commit_lock is not None else nullcontext():
                if self.cancel.is_set() or (is_current is not None and not is_current()):
                    return
                with self._scene_view_lock:
                    if self._staged_redirect is not None or event['committed'] or self._unresolved_transfer is None:
                        return
                    self.memory.observe(after, write=False)
                    self.current = after
                    self._unresolved_transfer['release_expected'] = True
                    event['committed'] = True
                    event['release_expected'] = True
                    event['result'] = copy.deepcopy(partial)
                    if on_transfer_commit is not None:
                        on_transfer_commit()
        self.cancel.bind(adapter)
        try:
            result = adapter.run(before, task, self._registered_slot_xy_mm,
                                 initial_applied_target_deg=initial_target,
                                 initial_target_source=initial_source,
                                 initial_observation_instance_id=initial_instance,
                                 initial_observation_revision=initial_revision,
                                 on_release_confirmed=on_release_confirmed,
                                 **({'initial_previous_student_target_deg': previous_student_target}
                                    if previous_student_target is not None else {}),
                                 **({'initial_nominal_target_deg': initial_nominal}
                                    if initial_nominal is not None else {}))
        except BaseException as exc:
            event['result'] = {'termination': 'interrupted' if isinstance(exc, InterruptedError) else 'error',
                               'error_type': type(exc).__name__, 'physical_success': None}
            raise
        finally:
            self.cancel.unbind(adapter)
        # An explicit acquire refusal happens before this adapter can send a
        # device command. Preserve the expected scene and memory; a lost or
        # ambiguous acquire response must still take the uncertainty path.
        error = result.get('error') if isinstance(result, dict) else None
        if (isinstance(result, dict) and result.get('termination') == 'controller_error'
                and result.get('acquire_rejected') == 'not_ready'
                and result.get('lease_state_uncertain') is False
                and result.get('layout_uncertain') is False
                and result.get('applied_command_count') == 0 and result.get('decisions') == []
                and isinstance(error, dict) and error.get('code') == 'NOT_READY'):
            with commit_lock if commit_lock is not None else nullcontext():
                if self.cancel.is_set() or (is_current is not None and not is_current()):
                    raise InterruptedError('request invalidated before retaining the prior scene')
                with self._scene_view_lock:
                    if self._staged_redirect is not None or event['committed']:
                        raise RuntimeError('transfer ownership changed before acquire refusal')
                    self._current_target = initial_target
                    self._current_target_source = initial_source
                    self._current_nominal_target = initial_nominal
                    self._last_encoded_student_target = encoded
                    self._initial_observation_instance_id = initial_instance
                    self._initial_observation_revision = initial_revision
                    self._current_registered_scene = initial_registered_scene
                    self._confirmed_current_scene = initial_confirmed_scene
                    self._unresolved_transfer = None
                    event['result'] = result
                    event['not_started'] = True
                    event['not_started_reason'] = 'controller_not_ready'
            raise StepUnavailable('제어기 수신이 준비되지 않아 이동을 시작하지 않았습니다.',
                                  not_started_reason='controller_not_ready')
        # Before grasp a redirect stops and restarts from a fresh reference.
        # A held runtime redirect may instead continue in this same adapter;
        # neither path establishes an observed object location.
        with self._scene_view_lock:
            redirect = self._staged_redirect
        direct = False
        if redirect is not None:
            redirect['done'].wait()
            redirect_result = result.get('retarget') if isinstance(result, dict) else None
            direct = (redirect.get('accepted') is True and isinstance(redirect_result, dict)
                      and redirect_result.get('mode') == 'direct_held_transfer'
                      and redirect_result.get('source_slot') == task['source_slot']
                      and redirect_result.get('destination_slot') == redirect['destination_slot'])
            if redirect.get('unknown') or (redirect.get('accepted') is True and
                    (not isinstance(result, dict) or (not direct and result.get('source_return_expected') is not True)
                     or result.get('end_confirmed') is not True
                     or result.get('last_confirmed_gripper_command') != 'open'
                     or result.get('physical_success') is not None
                     or result.get('lease_state_uncertain') or result.get('stop_error'))):
                event['result'] = result
                raise RuntimeError('redirect execution or control state is unconfirmed')
        if direct:
            replacement_before, replacement_task = scene_features.transfer_task(before, redirect['replacement_task'])
            replacement_after = task_catalog.transfer(
                replacement_before, replacement_task['source_slot'], replacement_task['destination_slot'])
            if scene_policy.split_group_for(replacement_before, replacement_task) not in self.train_groups:
                raise UnsupportedStep('redirected transfer is outside allowed groups')
            redirect['direct'] = True
            event['original_task'] = task
            event['task'] = replacement_task
            event['planned_after'] = replacement_after
            event['split_group'] = scene_policy.split_group_for(replacement_before, replacement_task)
            after = replacement_after
        elif (redirect is not None and redirect.get('accepted') is True
                and isinstance(result, dict) and result.get('source_return_expected') is True):
            returned = result
            event['staged_return'] = returned
            before_grasp = returned.get('pregrasp_return_expected') is True
            reference = (returned.get('pregrasp_next_reference') if before_grasp else
                         returned.get('next_control_reference') if self.controller.config.backend == 'sdk' else None)
            restart_kwargs = {}
            if before_grasp or self.controller.config.backend == 'sdk':
                if not isinstance(reference, dict) or reference.get('source') != 'stationary_observation':
                    raise RuntimeError('staged redirect has no fresh stopped reference')
                target = _initial_reference(reference.get('joints_deg'), 'stationary_observation',
                                            reference.get('instance_id'), reference.get('revision'))
                restart_kwargs.update(initial_target_source='stationary_observation',
                                      initial_observation_instance_id=reference['instance_id'],
                                      initial_observation_revision=reference['revision'])
            else:
                if returned.get('current_target_known') is not True:
                    raise RuntimeError('staged redirect has no confirmed control target')
                target = _applied_target(returned.get('last_confirmed_target_deg'))
                if self.enable_mock_nominal_trim:
                    restart_kwargs.update(initial_target_source='controller_confirmed_nominal_move',
                        initial_nominal_target_deg=_applied_target(returned.get('last_confirmed_nominal_target_deg')))
                else:
                    restart_kwargs['initial_target_source'] = 'controller_confirmed_move'
            replacement_before, replacement_task = scene_features.transfer_task(
                before, {'source_slot': task['source_slot'], 'destination_slot': redirect['destination_slot']})
            replacement_after = task_catalog.transfer(
                replacement_before, replacement_task['source_slot'], replacement_task['destination_slot'])
            if scene_policy.split_group_for(replacement_before, replacement_task) not in self.train_groups:
                raise UnsupportedStep('redirected transfer is outside allowed groups')
            with commit_lock if commit_lock is not None else nullcontext():
                if (self.cancel.is_set() or redirect['cancel_event'].is_set()
                        or (is_current is not None and not is_current())):
                    raise InterruptedError('staged redirect invalidated before restart')
                with self._scene_view_lock:
                    if self._staged_redirect is not redirect or self._unresolved_transfer is None:
                        raise InterruptedError('staged redirect no longer owns the transfer')
                    redirect['restarting'] = True
                    self._unresolved_transfer['redirect_destination_slot'] = redirect['destination_slot']
            event['original_task'] = task
            event['task'] = replacement_task
            event['planned_after'] = replacement_after
            event['split_group'] = scene_policy.split_group_for(replacement_before, replacement_task)
            adapter = make_adapter(redirected=True)
            self.cancel.bind(adapter)
            try:
                result = adapter.run(replacement_before, replacement_task, self._registered_slot_xy_mm,
                                     initial_applied_target_deg=target,
                                     **restart_kwargs)
            except BaseException as exc:
                event['result'] = {'termination': 'interrupted' if isinstance(exc, InterruptedError) else 'error',
                                   'error_type': type(exc).__name__, 'physical_success': None}
                raise
            finally:
                self.cancel.unbind(adapter)
            if isinstance(result, dict):
                result['retarget'] = {'source_slot': task['source_slot'],
                    'destination_slot': redirect['destination_slot'],
                    'activation_decision_index': 0 if result.get('decisions') else None,
                    'source_return_expected': True, 'before_grasp': before_grasp,
                    'object_position_observed': False, 'physical_success': None}
            after = replacement_after
        event['result'] = result
        event['minimum_command_sequence_confirmed'] = _has_applied_transfer_commands(result)
        if event.get('release_expected') and isinstance(result, dict):
            rows = result.get('decisions') or []
            releases = [i for i, row in enumerate(rows) if isinstance(row, dict)
                        and row.get('action') == 'open' and row.get('result') == 'command_applied']
            termination = result.get('termination')
            cause = result.get('previous_termination') if termination == 'end_unconfirmed' else termination
            after_release = rows[releases[0] + 1:] if len(releases) == 1 else []
            if (len(releases) == 1 and all(isinstance(row, dict) and
                    (row.get('action') == 'move' or
                     (index == len(after_release) - 1 and row.get('action') == 'done'
                      and row.get('result') == 'model_done_no_command'))
                    for index, row in enumerate(after_release))
                    and cause != 'cancelled' and not self.cancel.is_set()
                    and not result.get('stop_error') and not result.get('lease_state_uncertain')):
                with self._scene_view_lock:
                    self._pending_release_recovery = {'expected_scene': after,
                                                      'instance_id': getattr(self.controller, 'instance_id', None)}
        reference = _confirmed_no_motion(result, self.controller)
        if reference is not None and redirect is None and not event['committed']:
            with commit_lock if commit_lock is not None else nullcontext():
                if self.cancel.is_set() or (is_current is not None and not is_current()):
                    raise InterruptedError('request invalidated before retaining the prior scene')
                with self._scene_view_lock:
                    self._current_target = _initial_reference(reference['joints_deg'], 'stationary_observation',
                                                             reference['instance_id'], reference['revision'])
                    self._current_target_source = 'stationary_observation'
                    self._initial_observation_instance_id = reference['instance_id']
                    self._initial_observation_revision = reference['revision']
                    self._current_registered_scene = initial_registered_scene
                    self._confirmed_current_scene = initial_confirmed_scene
                    self._unresolved_transfer = None
                    self._staged_redirect = None
                    event['no_motion_confirmed'] = True
            raise NoMotionObserved('no arm movement or gripper command was observed')
        # If no CLOSE or OPEN decision was issued, an interrupted approach
        # does not advance the expected object arrangement. Keep its previous
        # scene available for the same-resident connection recovery; do not
        # claim the interrupted arm command completed or replay it.
        if (isinstance(result, dict) and self.controller.config.backend == 'sdk'
                and not event['committed'] and redirect is None and not self.cancel.is_set()
                and not result.get('stop_error') and result.get('termination') != 'model_done'
                and isinstance(result.get('decisions'), list)
                and all(isinstance(row, dict) and row.get('action') == 'move'
                        for row in result['decisions'])
                and result.get('last_confirmed_gripper_command') in (None, 'open')
                and isinstance(getattr(self.controller, 'instance_id', None), str)):
            with self._scene_view_lock:
                self._pending_release_recovery = {'expected_scene': before,
                    'instance_id': self.controller.instance_id, 'basis': 'before_grasp'}
        # layout_uncertain means the adapter did not observe the object. That
        # alone does not make a normally completed command sequence a failure.
        # Confirmed commands are necessary evidence, not proof of object motion.
        pregrasp = isinstance(result, dict) and result.get('pregrasp_return_expected') is True
        sdk_stopped_reference = self.controller.config.backend == 'sdk' and not pregrasp
        if (not isinstance(result, dict) or result.get('physical_success') is not None
                or result.get('end_confirmed') is not True
                or result.get('last_confirmed_gripper_command') != 'open'
                or result.get('lease_state_uncertain') or result.get('stop_error')
                or (redirect is not None and redirect.get('restarting') is True
                    and (result.get('retarget_grasp_pose_confirmed') is not True
                         or result.get('retarget_release_pose_confirmed') is not True))
                or (direct and result.get('retarget_release_pose_confirmed') is not True)
                or (pregrasp and (result.get('termination') != 'pregrasp_source_return'
                                  or result.get('model_done') is not False
                                  or result.get('source_return_expected') is not True))
                or (not pregrasp and (result.get('termination') != 'model_done'
                    or result.get('model_done') is not True
                    or type(result.get('applied_command_count')) is not int or result['applied_command_count'] < 1
                    or event['minimum_command_sequence_confirmed'] is not True
                     or (not sdk_stopped_reference and result.get('current_target_known') is not True)
                     or (sdk_stopped_reference and result.get('current_target_known') is not False)))):
            raise RuntimeError('physical transfer did not complete with a confirmed control state')
        if result.get('source_return_expected') is True:
            if not (self.supports_mock_source_return or self.supports_runtime_source_return):
                raise RuntimeError('source return is not enabled for this session')
            after = before
            event.update(planned_after=before, source_return_expected=True)
        if pregrasp or sdk_stopped_reference:
            reference = result.get('pregrasp_next_reference' if pregrasp else 'next_control_reference')
            if not isinstance(reference, dict) or reference.get('source') != 'stationary_observation':
                raise RuntimeError('confirmed stop has no fresh reference')
            target = _initial_reference(reference.get('joints_deg'), 'stationary_observation',
                                        reference.get('instance_id'), reference.get('revision'))
            nominal = None
            encoded = (_encoded_student_reference(result.get('last_encoded_student_target'), reference)
                       if sdk_stopped_reference and not pregrasp else None)
        else:
            target = _applied_target(result.get('last_confirmed_target_deg'),
                                     require_quantized=not self.execute_mock_virtual_descent)
            nominal = (_applied_target(result.get('last_confirmed_nominal_target_deg'))
                       if self.enable_mock_nominal_trim else None)
            encoded = None
        with commit_lock if commit_lock is not None else nullcontext():
            if (self.cancel.is_set() or (redirect is not None and redirect['cancel_event'].is_set())
                    or (is_current is not None and not is_current())):
                raise InterruptedError('request invalidated before expected-scene update')
            if not event['committed']:
                self.memory.observe(after, write=False)
            with self._scene_view_lock:
                self.current, self._current_target = after, target
                self._current_target_source = ('stationary_observation' if pregrasp or sdk_stopped_reference else
                    'controller_confirmed_nominal_move' if nominal is not None else 'controller_confirmed_move')
                self._current_nominal_target = nominal
                self._last_encoded_student_target = encoded
                if pregrasp or sdk_stopped_reference:
                    self._initial_observation_instance_id = reference['instance_id']
                    self._initial_observation_revision = reference['revision']
                self._unresolved_transfer = None
                self._staged_redirect = None
                self._pending_release_recovery = None
                first_commit = not event['committed']
                event['committed'] = True
                if redirect is not None and (redirect.get('restarting') is True or direct):
                    event['retarget_expected'] = True
            if first_commit and on_transfer_commit is not None:
                on_transfer_commit()

    def request_transfer_retarget(self, destination_slot, *, active_request_id, program_id,
                                  task_id, transfer_id, transfer_sequence, cancel_event=None):
        # Coordinator has already checked the caller's ownership and generation.
        if cancel_event is not None and cancel_event.is_set():
            return {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
        if not (self.supports_mock_source_return or self.supports_runtime_source_return):
            return {'status': 'retarget_unsupported', 'reason': 'executor_not_available'}
        with self._scene_view_lock:
            transfer = self._unresolved_transfer
            if not isinstance(transfer, dict) or self.status != 'registered':
                return {'status': 'retarget_rejected', 'reason': 'transfer_not_active'}
            if self._staged_redirect is not None:
                return {'status': 'retarget_rejected', 'reason': 'retarget_already_requested'}
            source = transfer.get('source_slot')
            if destination_slot == source:
                redirect = None
            else:
                if not self.supports_staged_destination_retarget:
                    return {'status': 'retarget_unsupported', 'reason': 'destination_not_supported'}
                if destination_slot not in self._active_slots or destination_slot == transfer.get('destination_slot'):
                    return {'status': 'retarget_rejected', 'reason': 'destination_infeasible'}
                try:
                    before, replacement = scene_features.transfer_task(self.current, {
                        'source_slot': source, 'destination_slot': destination_slot})
                    calibration = self.path_calibration
                    if (destination_slot not in calibration.enabled_slots
                            or replacement['destination_level'] > calibration.max_height
                            or scene_policy.split_group_for(before, replacement) not in self.train_groups):
                        return {'status': 'retarget_rejected', 'reason': 'destination_infeasible'}
                except (ValueError, TypeError, IndexError, KeyError, AttributeError):
                    return {'status': 'retarget_rejected', 'reason': 'destination_infeasible'}
                redirect = {'destination_slot': destination_slot, 'source_slot': source,
                            'replacement_task': replacement,
                            'cancel_event': cancel_event if cancel_event is not None else threading.Event(),
                            'done': threading.Event(), 'accepted': False, 'unknown': False}
                self._staged_redirect = redirect
        if redirect is None:
            return self.cancel.request_mock_source_return(source, cancel_event=cancel_event)
        try:
            kwargs = ({'replacement_task': replacement} if self.supports_runtime_source_return else {})
            answer = self.cancel.request_mock_source_return(source, cancel_event=redirect['cancel_event'], **kwargs)
            if not isinstance(answer, dict) or answer.get('status') not in (
                    'retarget_requested', 'retarget_rejected', 'retarget_unsupported'):
                redirect['unknown'] = True
                return answer
            redirect['accepted'] = answer['status'] == 'retarget_requested'
            if not redirect['accepted']:
                with self._scene_view_lock:
                    if self._staged_redirect is redirect:
                        self._staged_redirect = None
            return {**answer, 'destination_slot': destination_slot}
        except BaseException:
            redirect['unknown'] = True
            raise
        finally:
            redirect['done'].set()


class NativePhysicalCoordinator(SceneProgramCoordinator):
    """Shared v2 interpretation and execution plus explicit, device-free recovery."""

    def __init__(self, session):
        if not isinstance(session, NativePhysicalSession):
            raise TypeError('a physical-preparation session is required')
        super().__init__(session)

    def _preview_scene(self, step, scene, registration_scene=None, *, registered_initial=None,
                       task_start=None, train_groups, validation_source=None):
        if (scene is not None and step['op'] == 'task'
                and step['kind'] in ('move', 'stack', 'unstack', 'swap', 'arrange')):
            return self.session._scoped_plan_task(scene, step, validation_source=validation_source,
                                                  train_groups=train_groups)[0]
        return super()._preview_scene(step, scene, registration_scene, registered_initial=registered_initial,
                                      task_start=task_start, train_groups=train_groups,
                                      validation_source=validation_source)

    def _preflight(self, program, registration_scene, state, *, validation_source=None):
        plan = super()._preflight(program, registration_scene, state, validation_source=validation_source)
        if plan['kind'] == 'control':
            return plan
        steps = program['steps']
        registered_initial = (scene_recall.scene(registration_scene) if registration_scene is not None
                              else state['registered_initial'])
        known_scene = state['scene'] if state['session_status'] == 'registered' else None
        previous = state['previous_task'] if registration_scene is None else None
        known_task_start = previous['start_scene'] if previous is not None else None

        def preview(step, scene, task_start, source):
            if step['op'] == 'task' and step['kind'] == 'register':
                return scene_recall.scene(registration_scene)
            if step['op'] == 'task' and step['kind'] == 'remember':
                return scene
            if step['op'] == 'task':
                if step['kind'] in ('move', 'stack', 'unstack', 'swap'):
                    for key in ('source_slot', 'destination_slot'):
                        slot = step.get(key)
                        if slot is not None and slot not in self.session._active_slots:
                            raise _ScopedPathUnavailable('requested slot is outside the active slot scope')
                if scene is None:
                    # A previous learned-memory target is not known until its
                    # execution. Explicit endpoints can still be checked now;
                    # the complete path is checked once its scene is known.
                    return None
                return self.session._scoped_plan_task(scene, step, validation_source=source,
                                                       train_groups=state['train_groups'])[0]
            if step['op'] == 'return':
                kind = step['target']['kind']
                target = registered_initial if kind == 'registered_initial' else task_start if kind == 'task_start' else None
                if scene is None or target is None:
                    return None
                self.session._scoped_goal_path(scene, target, train_groups=state['train_groups'])
                return scene_recall.scene(target, scene_recall.identities(scene))
            return scene

        initial = steps if plan['repeat_index'] is None else steps[:plan['repeat_index']]
        for step in initial:
            before = known_scene
            known_scene = preview(step, known_scene, known_task_start, validation_source)
            if step['op'] in ('task', 'return') and not (step['op'] == 'task' and step['kind'] == 'register'):
                known_task_start = before

        if plan['repeat_index'] is not None:
            # The shared coordinator stops an infeasible repetition without
            # starting it. For predictable repetitions inspect every reachable
            # round (a repeated scene proves the remaining cycle); a decoded
            # memory goal remains unknown and gets its full check at execution.
            seen = set()
            rounds = 0
            count = plan['count']
            while count['mode'] == 'continuous' or rounds < count['value']:
                marker = (known_scene, known_task_start)
                if marker in seen:
                    break
                seen.add(marker)
                for step in plan['body']:
                    before = known_scene
                    try:
                        known_scene = preview(step, known_scene, known_task_start, plan['body_validation_source'])
                    except _ScopedPathUnavailable:
                        raise
                    except UnsupportedStep:
                        return plan  # The existing repeat-unavailable boundary handles this round.
                    known_task_start = before
                rounds += 1
                if known_scene is None:
                    break
        return plan

    def confirm_scene(self, request_id, confirmed_scene):
        """Record a caller's explicit observation; a mismatch requires recovery."""
        if not task_program._valid_id(request_id):
            raise ValueError('invalid confirmation request ID')
        with self._lock:
            self._refresh_external_state()
            if self._active is not None or not self.session._lock.acquire(blocking=False):
                raise Busy('confirmation requires an idle session')
            try:
                if self.session.status != 'registered' or self.session.cancel.is_set() or self._stop_unconfirmed:
                    raise RuntimeError('confirmation requires a ready session')
                confirmed = scene_recall.scene(confirmed_scene, scene_recall.identities(self.session.current))
                with self.session._request_lock:
                    if request_id in self.session._requests:
                        raise DuplicateRequest('request ID was already used')
                    self.session._requests.add(request_id)
                with self.session._scene_view_lock:
                    self.session._confirmed_current_scene = self.session._last_confirmed_scene = confirmed
                    matches = confirmed == self.session.current
                    if not matches:
                        self.session.status = 'uncertain'
                        self.session._current_target = None
                        self.session._current_target_source = None
                        self.session._current_nominal_target = None
                        self.session._last_encoded_student_target = None
                        self.session._initial_observation_instance_id = None
                        self.session._initial_observation_revision = None
                        self.session._current_registered_scene = None
                    self.session.events.append({'event': 'caller_confirmed_scene', 'request_id': request_id,
                        'scene': confirmed, 'matches_expected': matches, 'physical_assessment_performed': False})
                self._previous_task = self._previous_program = None
                if not matches:
                    self._memory_revision = None
                self._known_session_stamp = self._session_stamp()
                self._bump()
                return {'request_id': request_id, 'status': 'confirmed' if matches else 'uncertain',
                        'matches_expected': matches, 'scene_state': self.session.scene_state(),
                        'physical_success': None, 'physical_assessment_performed': False}
            finally:
                self.session._lock.release()

    def recover_registration(self, request_id, registration_scene, *, initial_applied_target_deg,
                             controller_ready_confirmed, scene_confirmed=False,
                             initial_target_source='caller_confirmed', initial_observation_instance_id=None,
                             initial_observation_revision=None, preserve_memory=False):
        """Re-register after uncertainty without touching the controller.

        The caller must independently confirm a stopped, fault-free controller,
        open gripper and the supplied control-reference source. ``scene_confirmed`` is a
        caller assertion, not an assessment performed by this session. The
        caller may preserve the existing memory while correcting the current
        scene; recovery never resumes a partially executed move.
        """
        if not task_program._valid_id(request_id):
            raise ValueError('invalid recovery request ID')
        if (controller_ready_confirmed is not True or type(scene_confirmed) is not bool
                or type(preserve_memory) is not bool):
            raise ValueError('explicit controller readiness confirmation is required')
        target = _initial_reference(initial_applied_target_deg, initial_target_source,
                                    initial_observation_instance_id, initial_observation_revision)
        with self._lock:
            self._refresh_external_state()
            if self._active is not None or not self.session._lock.acquire(blocking=False):
                raise Busy('recovery requires an idle session')
            try:
                if self.session.status != 'uncertain':
                    raise RuntimeError('recovery requires an uncertain session')
                ids = scene_recall.identities(self.session.current) if self.session.current is not None else None
                registered = scene_recall.scene(registration_scene, ids)
                with self.session._request_lock:
                    if request_id in self.session._requests:
                        raise DuplicateRequest('request ID was already used')
                    self.session._requests.add(request_id)
                self._generation += 1
                generation = self._generation
                previous_memory_revision = self._memory_revision
                previous_registered_initial = self._registered_initial
                run = {'kind': 'recovery_registration', 'request_id': request_id,
                       'program_id': self._new_id('program'), 'program_revision': self._new_id('revision'),
                       'generation': generation, 'round': 1, 'step_index': None, 'task_id': None,
                       'task_revision': None, 'task_sequence': 0, 'start_scene': copy.deepcopy(self.session.current),
                       'task_start_scene': None, 'repeat_active': False, 'stop_repeat': None,
                       'stop_requested': False, 'stop_requests': [], 'last_completed': None,
                       'stop_boundary_missed': False, 'atomic_inflight': False, 'transfer_id': None,
                       'transfer_sequence': 0, 'transfer_phase': 'between', 'last_committed_transfer_id': None,
                       'last_committed_transfer_sequence': 0, 'stop_transfer_id': None}
                self._active = run
                self.session.cancel.clear()
                self._bump()
            except BaseException:
                self.session._lock.release()
                raise
        try:
            with self.session._memory_binding_lock:
                memory = self.session.memory
                guard = getattr(memory, 'synchronized', None)
                with guard() if callable(guard) else nullcontext():
                    if self.session.cancel.is_set():
                        raise InterruptedError('recovery cancelled')
                    if not preserve_memory:
                        memory.clear()
                        if self.session.cancel.is_set():
                            raise InterruptedError('recovery cancelled')
                        memory.observe(registered, write=False)
                    prepared_stamp = self._memory_stamp()
                    with self._lock:
                        if (self._generation != generation or self._active is not run
                                or self.session.cancel.is_set()):
                            raise InterruptedError('recovery cancelled before registration')
                        if self._memory_stamp() != prepared_stamp:
                            raise RuntimeError('memory changed during recovery')
                        with self.session._scene_view_lock:
                            self.session.current, self.session.status = registered, 'registered'
                            self.session._current_target = target
                            self.session._current_target_source = initial_target_source
                            self.session._current_nominal_target = None
                            self.session._last_encoded_student_target = None
                            self.session._initial_observation_instance_id = initial_observation_instance_id
                            self.session._initial_observation_revision = initial_observation_revision
                            self.session._last_registered_scene = self.session._current_registered_scene = registered
                            self.session._confirmed_current_scene = registered if scene_confirmed else None
                            if scene_confirmed:
                                self.session._last_confirmed_scene = registered
                            self.session._unresolved_transfer = None
                            self.session._staged_redirect = None
                            self.session._pending_release_recovery = None
                            self.session.events.append({'event': 'registered', 'scene': registered,
                                'memory_cleared': not preserve_memory, 'scene_source': 'explicit_recovery_registration',
                                'request_id': request_id, 'caller_confirmed_scene': scene_confirmed,
                                'control_target_source': initial_target_source, 'physical_assessment_performed': False})
                        self._generation += 1
                        self._stop_unconfirmed = False
                        self._active = None
                        self._previous_task = self._previous_program = None
                        self._registered_initial = (previous_registered_initial
                            if preserve_memory and previous_registered_initial is not None else registered)
                        self._registration_revision = self._new_id('registration')
                        self._memory_revision = previous_memory_revision if preserve_memory else None
                        self._known_session_stamp = self._session_stamp()
                        self._expected_memory_stamp = prepared_stamp
                        self.events.append({'event': 'recovery_registered', 'request_id': request_id})
                        self._bump()
                        return {'request_id': request_id, 'status': 'registered',
                                'scene_state': self.session.scene_state(), 'physical_success': None,
                                'physical_assessment_performed': False, 'control_target_source': initial_target_source}
        except BaseException as exc:
            with self._lock:
                self.session.events.append({'event': 'recovery_registration_failed', 'request_id': request_id,
                                            'error_type': type(exc).__name__})
                if self._active is run:
                    self._active = None
                if not preserve_memory or self._memory_stamp() != self._expected_memory_stamp:
                    self._memory_revision = None
                self._known_session_stamp = self._session_stamp()
                self._expected_memory_stamp = self._memory_stamp()
                self._bump()
            raise
        finally:
            self.session._lock.release()


def load_session(controller, *, slot_xy_mm, initial_applied_target_deg, speed_pct,
                 enable_arrival_correction=False, enable_blended_path=False, enable_release_setdown=False,
                 initial_target_source='caller_confirmed', initial_observation_instance_id=None,
                 initial_observation_revision=None,
                 student_path=selected.BEST_DEVELOPMENT_STUDENT, memory_path=selected.MEMORY,
                 max_decisions=60, execution_profile=PHYSICAL_INITIAL_PROFILE, active_slots=None,
                 compare_virtual_descent=False, execute_mock_virtual_descent=False,
                 path_calibration=None, enable_experimental_segmented_path=False,
                 enable_mock_source_return=False, enable_mock_interrupt_retarget=False,
                 enable_runtime_source_return=False, enable_mock_nominal_trim=False,
                 enable_staged_destination_retarget=False,
                 enable_mock_layered_grasp_lowering=False, enable_mock_early_source_return=False,
                 enable_mock_open_drift_reconcile=False, enable_mock_vertical_replan=False,
                 enable_mock_move_release_upper_separation=False):
    """Load the selected 20261030 student and GRU128 into the shared v2 API."""
    student_bytes, memory_bytes = Path(student_path).read_bytes(), Path(memory_path).read_bytes()
    if hashlib.sha256(student_bytes).hexdigest() != selected.BEST_DEVELOPMENT_STUDENT_SHA256:
        raise ValueError('selected student checkpoint differs')
    if hashlib.sha256(memory_bytes).hexdigest() != selected.MEMORY_SHA256:
        raise ValueError('GRU checkpoint differs')
    student = terra_native_close_evidence.load_student_checkpoint(io.BytesIO(student_bytes), device='cpu').requires_grad_(False)
    memory_model = scene_recall.load_scene_memory(io.BytesIO(memory_bytes), device='cpu').requires_grad_(False)
    if memory_model.hidden_size != 128:
        raise ValueError('expected the GRU128 memory')
    session = NativePhysicalSession(controller, scene_recall.SceneRecall(memory_model), student,
        slot_xy_mm=slot_xy_mm, initial_applied_target_deg=initial_applied_target_deg,
        initial_target_source=initial_target_source,
        initial_observation_instance_id=initial_observation_instance_id,
        initial_observation_revision=initial_observation_revision,
        speed_pct=speed_pct, max_decisions=max_decisions, execution_profile=execution_profile,
        enable_arrival_correction=enable_arrival_correction,
        enable_blended_path=enable_blended_path,
        enable_release_setdown=enable_release_setdown,
        active_slots=active_slots, compare_virtual_descent=compare_virtual_descent,
        execute_mock_virtual_descent=execute_mock_virtual_descent,
        path_calibration=path_calibration,
        enable_experimental_segmented_path=enable_experimental_segmented_path,
        enable_mock_source_return=enable_mock_source_return,
        enable_mock_interrupt_retarget=enable_mock_interrupt_retarget,
        enable_runtime_source_return=enable_runtime_source_return,
        enable_staged_destination_retarget=enable_staged_destination_retarget,
        enable_mock_nominal_trim=enable_mock_nominal_trim,
        enable_mock_layered_grasp_lowering=enable_mock_layered_grasp_lowering,
        enable_mock_early_source_return=enable_mock_early_source_return,
        enable_mock_open_drift_reconcile=enable_mock_open_drift_reconcile,
        enable_mock_vertical_replan=enable_mock_vertical_replan,
        enable_mock_move_release_upper_separation=enable_mock_move_release_upper_separation)
    return NativePhysicalCoordinator(session)
