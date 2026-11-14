from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from .device import Snapshot
from .calibration_moves import prepare_step
from . import __version__
from .errors import ControlError
from .journal import Journal
from .kinematics import ENCODER_STEP_DEG, JOINT_COUNT, Geometry, check_joints, distance, quantize_joints
from .motion_envelope import bounds as motion_bounds, check as check_envelope
from .locking import FileLock
from .supervisor import DeviceProcess
from .validation import identifier, keys, number, vector
from .timing import measured,span
from .blended_execution import (prepare_virtual_blend, planned_sample_digest, update_sample_digest,
                                virtual_blend_time_budget, sdk_blended_commands, sdk_blended_command_digest,
                                SDK_BLENDED_COMMAND_PERIOD_S)


# A position-servo trim is intentionally small and does not enlarge any
# existing joint, workspace, speed, path, or arrival-tolerance limit.
MAX_NOMINAL_TRIM_DEG = 1.0
MAX_MOCK_NOMINAL_RETRY_TCP_MM = 5.0
# The SDK transport tolerates a brief receive gap. Samples older than the
# normal sensor-age limit are still invalid for dispatch and completion.
SDK_FEEDBACK_GAP_GRACE_SECONDS = 1.0
OBSERVATION_HISTORY_LIMIT = 64
MAX_BLENDED_WAYPOINTS = 64
STALL_PROGRESS_TIMEOUT_SECONDS = 2.0
GRIPPER_TIMEOUT_CAP_SECONDS = 1.0
RECOVERABLE_FAULT_CODES = frozenset({
    'TARGET_NOT_REACHED', 'MOTION_TIMEOUT', 'NO_MOTION_CONFIRMATION',
    'DEVICE_WARNING', 'DISCONNECTED', 'SENSOR_STALE', 'STOP_UNCONFIRMED',
    'DAMPED_COMPENSATION_DIVERGED',
    # Only the measured tracking outcomes use the same
    # stopped/confirmed-layout recovery as target failure.
    # Unknown send, path, and infrastructure faults still
    # require the existing separate review.
    'SDK_TRACKING_DIVERGED', 'SDK_TRACKING_STALLED',
})
MOTION_KINDS = frozenset({
    'move_joints', 'correct_xy', 'jog_joint', 'jog_tcp', 'mock_joint_trajectory', 'blended_joint_path',
})
RETAINED_POSITION_SUCCESS_STATUSES = frozenset({'completed', 'settled', 'settled_no_motion'})
NOMINAL_CORRECTION_STATUSES = frozenset({
    'settled_needs_nominal_correction', 'gripper_reported_arm_needs_nominal_correction',
})
SUCCESS_LIKE_STATUSES = frozenset({
    'completed', 'settled_no_motion', 'settled_needs_observation', 'settled_needs_nominal_retry',
    'settled_needs_nominal_correction', 'settled_needs_target_follow_up', 'already_at_target',
    'settled', 'position_target_unchanged',
    'gripper_reported', 'gripper_reported_arm_needs_nominal_correction',
    'gripper_commanded_unverified', 'observed',
})
ACTION_FIELDS = {
    "move_joints": {"kind", "joints_deg", "nominal_arrival_joints_deg", "allow_mock_nominal_retry",
                    "allow_nominal_correction", "correction_of_command_id", "speed_pct", "allow_visual_settle",
                    "path_constraints", "arrival_tolerance_mm", "completion_mode", "allow_damped_compensation",
                    "allow_target_follow_up", "direct_student_target"},
    "mock_joint_trajectory": {"kind", "targets_deg", "period_seconds"},
    "blended_joint_path": {"kind", "waypoints_deg", "segment_durations_s", "max_velocity_deg_s",
                           "max_acceleration_deg_s2", "max_jerk_deg_s3", "sample_period_seconds",
                           "speed_pct", "nominal_arrival_joints_deg", "allow_nominal_correction",
                           "path_constraints", "arrival_tolerance_mm", "completion_mode", "allow_damped_compensation",
                           "allow_target_follow_up"},
    "correct_xy": {"kind", "delta_xy_mm", "reference_id", "speed_pct"},
    "set_gripper": {"kind", "state", "retain_position_from_command_id"},
    "reobserve": {"kind"},
    "jog_joint": {"kind", "joint", "delta_deg", "speed_pct"},
    "jog_tcp": {"kind", "delta_mm", "speed_pct"},
}


class _DispatchCancelled(Exception):
    """An action was cancelled before its worker request was sent."""


class Controller:
    def __init__(self, config, runtime):
        self.config = config
        self.limits = config.limits
        self.runtime = Path(runtime).resolve()
        self.resource = FileLock(self.runtime / "controller.lock").acquire()
        try:
            self.journal = Journal(self.runtime, config.backend)
        except Exception:
            self.resource.close()
            raise
        self.worker = DeviceProcess(config)
        self.geometry = Geometry(tool_forward_mm=config.tool_forward_mm, tool_vertical_mm=config.tool_vertical_mm)
        self.instance_id = uuid.uuid4().hex
        self._condition = threading.Condition(threading.RLock())
        self._stop_lock = threading.Lock()
        # A stop must be visible even while another thread holds _condition
        # validating a long path.  The counter also prevents a new lease from
        # clearing cancellation before all concurrent stops have completed.
        self._stop_intent_lock = threading.Lock()
        self._pending_stops = 0
        self._stop_epoch = 0
        self._dispatch_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closing = threading.Event()
        self._cancel = threading.Event()
        self._cancel_reason = "STOP_REQUESTED"
        self._sampler = None
        self._snapshot = None
        self._fault = None
        self._restored_fault_instance_id = None
        self._state = "starting"
        self._lease = None
        self._active = None
        self._active_kind = None
        # Interruption reservations survive re-arming: a delayed call for the
        # old command must never be transmitted later.
        self._motion_interruption = None
        self._retarget_preempted_commands = {}
        self._active_collision_baseline = None
        self._active_warning_baseline = None
        self._lease_finished_commands = set()
        # Requests register before taking _condition so an authorized
        # interruption can identify the old MOVE during long admission work.
        self._command_intentions = {}
        self._preparing = None
        self._revision = 0
        self._observations = OrderedDict()
        self._history = []
        self._corrections = 0
        self._gripper_command = None
        self._gripper_evidence = None
        # This cache belongs only to this controller instance; prior journal
        # entries or an encoder reading alone do not prove an applied target.
        self._last_confirmed_applied_target = None
        # A host-encoded position target is separate from applied/received
        # evidence. It is valid only in this lease and stop generation.
        self._retained_position_target = None
        # Instance-local, lease-bound and single-use; never reconstructed from
        # a journal after restart or inferred from an encoder position.
        self._pending_nominal_correction = None
        self._last_stop = {"requested": False, "confirmed": False}
        self._closed = False
        self._started = False

    @property
    def supports_mock_joint_trajectory(self):
        """Static capability; querying it does not start or contact a device."""
        return self.config.backend == "mock" and self.config.hardware_enabled is False

    @property
    def supports_mock_blended_joint_path(self):
        # Only a deliberately injected virtual worker or a mock DeviceProcess
        # may advertise this. An SDK DeviceProcess always returns False.
        return getattr(self.worker, 'supports_mock_blended_joint_path', False) is True

    @property
    def blended_path_scope(self):
        if self.supports_mock_blended_joint_path:
            return 'virtual_worker_only'
        if getattr(self.worker, 'supports_sdk_blended_joint_path', False) is True:
            return 'sdk_host_stream_unverified'
        return 'unavailable'

    def motion_interruption_capability(self):
        """Report API availability, not verified physical performance."""
        return {'instance_id': self.instance_id, 'supported': True,
                'nominal_correction_supported': True,
                'blended_joint_path_supported': self.blended_path_scope != 'unavailable',
                'blended_joint_path_scope': self.blended_path_scope,
                'backend': self.config.backend, 'physical_execution_verified': False,
                'stop_delay_bounded': False}

    def start(self):
        if self._started or self._closed:
            raise ControlError("ALREADY_STARTED", "A controller instance cannot be restarted")
        self._started = True
        if self.journal.pending:
            raise ControlError("RECOVERY_REQUIRED", "An admitted command has no recorded outcome; it will not be replayed", commands=self.journal.pending)
        restored = self._unrecovered_restart_fault()
        if restored is not None:
            error, self._restored_fault_instance_id = restored
            self._fault = error
            self._cancel_reason = error.code
            self._cancel.set()
        began = time.monotonic()
        metadata = self.worker.start()
        self.journal.append("service_started", instance_id=self.instance_id, worker=metadata,
                            controller_version=__version__, config=self.config.as_dict(),
                            restart_fault_protocol=1)
        if restored is not None:
            self.journal.append("fault_restored", instance_id=self.instance_id,
                                original_instance_id=self._restored_fault_instance_id,
                                error=self._fault.as_dict())
        self._sampler = threading.Thread(target=self._sample_loop, name="raccoon-telemetry", daemon=True)
        self._sampler.start()
        with self._condition:
            while not self._fresh(self._snapshot):
                if self._fault and restored is None:
                    raise self._fault
                left = began + self.limits.connect_timeout - time.monotonic()
                if left <= 0:
                    raise ControlError("SENSOR_START_TIMEOUT", "No fresh sensor state arrived before the connection deadline")
                self._condition.wait(min(left, 0.05))
            self._state = "fault" if self._fault is not None else "idle"
        return self.status()

    def _unrecovered_restart_fault(self):
        """Retain unresolved fault and stop evidence across a controller restart.

        Legacy service starts did not preserve connection faults. Limit those
        older records to the last legacy instance, without asserting that an
        earlier fault was recovered. Starts using this protocol preserve the
        outstanding lineage even if they stop before writing fault_restored.
        Compensation failures retain their previous cross-instance treatment.
        """
        damped_pending = []
        current_pending = []
        current_instance = None
        protocol_seen = False
        try:
            with self.journal.path.open(encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    event = item.get('event')
                    error = item.get('error')
                    if event == 'service_started':
                        new_protocol = item.get('restart_fault_protocol') == 1
                        # Old binaries recorded no lineage for connection faults;
                        # do not treat their old starts as proof of recovery.
                        # Once the protocol is in use, no later start clears an
                        # unresolved connection/stop fault implicitly.
                        if ((not protocol_seen and not new_protocol)
                                or not any(fault['connection_related'] for fault in current_pending)):
                            current_pending = []
                        protocol_seen = protocol_seen or new_protocol
                        current_instance = item.get('instance_id')
                        # A protocol start can only be recorded after this
                        # implementation selected its one restored fault. If
                        # it stops before fault_restored is written, keep a
                        # later unconfirmed stop bound to that fault as well.
                        if (new_protocol and len(damped_pending) == 1 and not any(
                                damped_pending[0] is fault for fault in current_pending)):
                            current_pending.append(damped_pending[0])
                    elif event == 'fault_restored' and item.get('instance_id') == current_instance:
                        # A stop in this instance belongs to the original
                        # latched fault, whose ID is used by fault_recovered.
                        # This also handles legacy compensation restorations.
                        origin = item.get('original_instance_id')
                        for fault in (*current_pending, *damped_pending):
                            if fault['instance_id'] == origin and fault['error'] == error:
                                if not any(fault is current for current in current_pending):
                                    current_pending.append(fault)
                                break
                    elif event == 'fault':
                        code = error.get('code') if isinstance(error, dict) else None
                        entry = {'instance_id': item.get('instance_id'), 'error': error,
                                 'connection_related': code in {'DISCONNECTED', 'SENSOR_STALE',
                                                                'STOP_UNCONFIRMED'}}
                        if damped_pending or code == 'DAMPED_COMPENSATION_DIVERGED':
                            damped_pending.append(entry)
                        if entry['instance_id'] == current_instance and current_instance is not None:
                            current_pending.append(entry)
                    elif event in {'stop_requested', 'service_closed'} and current_instance is not None:
                        if event == 'service_closed' and item.get('instance_id') != current_instance:
                            continue
                        stop = item.get('result') if event == 'stop_requested' else item.get('stop')
                        if isinstance(stop, dict) and stop.get('confirmed') is not True:
                            if current_pending:
                                # The latched fault already carries this stop
                                # uncertainty, even if its code is unrelated.
                                current_pending[-1]['connection_related'] = True
                            else:
                                current_pending.append({
                                    'instance_id': current_instance,
                                    'error': {'code': 'STOP_UNCONFIRMED',
                                              'message': 'Recorded stop was not confirmed before restart',
                                              'details': {}},
                                    'connection_related': True,
                                })
                    elif event == 'fault_recovered':
                        origin = item.get('restored_fault_instance_id', item.get('instance_id'))
                        previous = item.get('previous_fault')
                        for pending in (damped_pending, current_pending):
                            for index, fault in enumerate(pending):
                                if fault['instance_id'] == origin and fault['error'] == previous:
                                    pending.pop(index)
                                    break
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ControlError('JOURNAL_CORRUPT', 'Inspect the existing journal before starting another controller') from exc
        pending = list(damped_pending)
        if any(fault['connection_related'] for fault in current_pending):
            # The same record can also be tracked by the compensation rule.
            pending.extend(fault for fault in current_pending if not any(fault is other for other in pending))
        for entry in pending:
            fault = entry['error']
            if (not isinstance(fault, dict) or not isinstance(fault.get('code'), str)
                    or not isinstance(fault.get('message'), str)
                    or not isinstance(fault.get('details'), dict)
                    or not isinstance(entry['instance_id'], str)):
                raise ControlError('JOURNAL_CORRUPT', 'Inspect the existing journal before starting another controller')
        if not pending:
            return None
        if (len(pending) != 1 or
                (any(pending[0] is fault for fault in damped_pending) and
                 pending[0]['error']['code'] != 'DAMPED_COMPENSATION_DIVERGED')):
            raise ControlError('RECOVERY_REVIEW_REQUIRED',
                               '기록된 다른 오류도 먼저 점검해야 합니다. 자동 초기화하지 않습니다.',
                               recorded_faults=[entry['error'] for entry in pending])
        entry = pending[0]
        fault = entry['error']
        return ControlError(fault['code'], fault['message'], **fault['details']), entry['instance_id']

    def _fresh(self, snapshot):
        if snapshot is None or not snapshot.connected or snapshot.sensor_time is None or snapshot.sensor_sequence < 1:
            return False
        if not 0 <= time.monotonic() - snapshot.sensor_time <= self.limits.sensor_max_age:
            return False
        try:
            vector(snapshot.joints_deg, JOINT_COUNT, "encoder")
        except ControlError:
            return False
        return True

    def _sdk_feedback_gap_in_grace(self, snapshot):
        if (self.config.backend != 'sdk' or snapshot is None
                or not snapshot.connected or snapshot.sensor_time is None
                or snapshot.sensor_sequence < 1):
            return False
        age = time.monotonic() - snapshot.sensor_time
        try:
            vector(snapshot.joints_deg, JOINT_COUNT, 'encoder')
        except ControlError:
            return False
        return 0 <= age <= max(self.limits.sensor_max_age, SDK_FEEDBACK_GAP_GRACE_SECONDS)

    def _publish(self, snapshot):
        with self._condition:
            if self._snapshot is None or snapshot.sensor_sequence >= self._snapshot.sensor_sequence:
                self._snapshot = snapshot
            self._condition.notify_all()

    def _sample_loop(self):
        idle_disconnect_deferred = False
        while not self._closing.is_set():
            try:
                snapshot = Snapshot.from_dict(self.worker.call("snapshot"))
                with self._condition:
                    # Publish the sample and decide whether it is the one
                    # idle poll to defer under the same admission boundary.
                    self._publish(snapshot)
                    state, fault = self._state, self._fault
                    latest = self._snapshot
                    expired = self._lease is not None and time.monotonic() >= self._lease["deadline"]
                    with self._stop_intent_lock:
                        idle = (state in ('idle', 'stopped') and self._lease is None
                                and self._active is None and self._preparing is None
                                and self._retained_position_target is None and not self._pending_stops)
                        if self.config.backend == 'sdk':
                            # An idle feedback gap does not create a persistent
                            # controller fault. While data is unavailable the
                            # UI remains disconnected and no result is inferred.
                            # An active episode keeps its existing fault path.
                            defer_disconnect = (not fault and state != 'starting' and not self._fresh(latest)
                                                and (idle or self._sdk_feedback_gap_in_grace(latest)))
                        else:
                            defer_disconnect = (not fault and state != 'starting' and not latest.connected
                                                and idle and not idle_disconnect_deferred)
                            if defer_disconnect:
                                idle_disconnect_deferred = True
                            elif latest.connected:
                                idle_disconnect_deferred = False
                self.journal.sample(self.instance_id, snapshot.as_dict())
                if not fault and state != "starting" and not self._fresh(latest):
                    code = "DISCONNECTED" if not latest.connected else "SENSOR_STALE"
                    if not defer_disconnect:
                        self._trip_fault(ControlError(code, "Fresh device feedback was lost"))
                elif not fault and state != 'starting' and latest.firmware_warning_raw==1:
                    self._trip_fault(ControlError('DEVICE_WARNING','The device warning bit became active'))
                if expired:
                    with self._condition:
                        if self._lease is None or time.monotonic() < self._lease["deadline"]:
                            continue
                        with self._stop_intent_lock:
                            expired_lease = self._lease
                            self._lease = None
                            self._stop_epoch += 1
                            self._cancel_reason = "LEASE_EXPIRED"
                            self._cancel.set()
                        self._revision += 1
                        self._last_stop = {"requested": False, "confirmed": False}
                        if self._fault is None:
                            self._state = "stopped"
                    self.journal.append("lease_expired", instance_id=self.instance_id, episode_id=expired_lease["episode_id"])
                    stopped = self._stop_device()
                    self.journal.append("episode_ended", episode_id=expired_lease["episode_id"], reason="LEASE_EXPIRED", verified_success=None, stop=stopped)
            except ControlError as exc:
                if not self._closing.is_set():
                    self._trip_fault(exc)
            except Exception as exc:
                if not self._closing.is_set():
                    self._trip_fault(ControlError("SAMPLER_ERROR", type(exc).__name__))
            self._closing.wait(self.limits.sample_period)

    def _trip_fault(self, error):
        if self._latch_fault(error):
            self._stop_device()

    def _latch_fault_state(self, error, *, clear_stop=True):
        """Close command admission while holding only the state lock."""
        with self._condition:
            if self._fault is not None:
                return False
            with self._stop_intent_lock:
                self._fault = error
                self._retained_position_target = None
                self._state = "fault"
                self._revision += 1
                self._cancel_reason = error.code
                self._cancel.set()
            if clear_stop:
                self._last_stop = {"requested": False, "confirmed": False}
            self._condition.notify_all()
        return True

    def _latch_fault(self, error, *, clear_stop=True):
        if not self._latch_fault_state(error, clear_stop=clear_stop):
            return False
        try:
            self.journal.append("fault", instance_id=self.instance_id, error=error.as_dict())
        except ControlError:
            pass
        return True

    def _stop_device(self):
        with span('stop_confirmation'):
            return self._stop_device_impl()

    def _fault_stop_result(self):
        """Wait for a concurrently faulting sampler's already-started stop."""
        deadline = time.monotonic() + self.limits.stop_timeout + self.limits.rpc_timeout
        with self._condition:
            while (self._fault is not None and self._last_stop.get('requested') is False
                   and 'error' not in self._last_stop and time.monotonic() < deadline):
                self._condition.wait(min(self.limits.sample_period, max(0., deadline-time.monotonic())))
            return dict(self._last_stop)

    def _stop_device_impl(self):
        with self._stop_lock:
            result = {"requested": False, "confirmed": False, "scope": "arm_joints"}
            requested_at = time.monotonic()
            deadline = requested_at + self.limits.stop_timeout
            with self._condition:
                before = self._snapshot.sensor_sequence if self._snapshot else -1
                self._retained_position_target = None
            try:
                with self._dispatch_lock:
                    self.worker.call("stop", timeout=min(self.limits.rpc_timeout, self.limits.stop_timeout))
                result["requested"] = True
                stable_since, stable_pose = None, None
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    snapshot = Snapshot.from_dict(self.worker.call("snapshot", timeout=min(self.limits.rpc_timeout, remaining)))
                    self._publish(snapshot)
                    if (self._fresh(snapshot) and snapshot.sensor_time >= requested_at and snapshot.moving is False):
                        if snapshot.sensor_sequence > before:
                            if stable_pose is None or max(abs(a - b) for a, b in zip(snapshot.joints_deg, stable_pose)) > self.limits.joint_tolerance_deg / 2:
                                stable_pose, stable_since = snapshot.joints_deg, snapshot.sensor_time
                            if snapshot.sensor_time - stable_since >= self.limits.stable_seconds:
                                result["confirmed"] = True
                                result["confirmed_sensor_sequence"] = snapshot.sensor_sequence
                                result["confirmed_sensor_monotonic"] = snapshot.sensor_time
                                break
                            before = snapshot.sensor_sequence
                    else:
                        stable_since, stable_pose = None, None
                    time.sleep(min(0.01, max(0, deadline - time.monotonic())))
            except ControlError as exc:
                result["error"] = exc.as_dict()
            with self._condition:
                self._last_stop = result
            if not result["confirmed"]:
                self._latch_fault(ControlError("STOP_UNCONFIRMED", "Fresh stationary arm feedback was not confirmed after stop"), clear_stop=False)
            return result

    def status(self):
        with self._condition:
            sample = self._snapshot
            now = time.monotonic()
            fresh = self._fresh(sample)
            grip_fresh = bool(sample and sample.connected and sample.gripper_time is not None
                              and 0 <= now - sample.gripper_time <= self.limits.sensor_max_age)
            joints = list(sample.joints_deg) if fresh else None
            # Display only existing sampler data.  In particular this must not
            # call observe() or create an observation ID while a student runs.
            display_observation = {
                "source": "mock_controller_sample" if self.config.backend == "mock" else "controller_sample",
                "simulated": self.config.backend == "mock",
                "joints_deg": joints,
                "tcp_mm": list(self.geometry.forward(joints)) if joints is not None else None,
                "tcp_source": "nominal_fk_from_sampled_joints",
                "sensor_valid": fresh,
                "sensor_sequence": sample.sensor_sequence if sample else None,
                "sensor_age_seconds": now - sample.sensor_time if sample and sample.sensor_time is not None else None,
                "moving": sample.moving if sample and fresh else None,
                "gripper_command": self._gripper_command,
                "gripper_reported_state": sample.gripper_reported_state if grip_fresh else None,
                "gripper_reported_valid": bool(grip_fresh and sample.gripper_reported_state is not None),
                "gripper_feedback_age_seconds": now - sample.gripper_time if sample and sample.gripper_time is not None else None,
            }
            candidate = self._last_confirmed_applied_target
            with self._stop_intent_lock:
                stop_pending = bool(self._pending_stops)
            target_confirmed = (candidate is not None and fresh and sample.moving is False
                                and self._active is None and self._fault is None and not stop_pending
                                and all(abs(a - b) <= self.limits.joint_tolerance_deg + self.limits.observation_joint_drift_deg
                                        for a, b in zip(sample.joints_deg, candidate['joints_deg'])))
            return {
                "instance_id": self.instance_id,
                "resident_api_revision": 2,
                "configuration_sha256": self.config.fingerprint(),
                "backend": self.config.backend,
                "simulated": self.config.backend == "mock",
                "mode": self.config.mode,
                "state": self._state,
                "worker_alive": self.worker.alive,
                "worker_pid": self.worker.process.pid if self.worker.process else None,
                "fresh_feedback": self._fresh(self._snapshot),
                "transport_connected": bool(sample and sample.connected),
                "active_command_id": self._active,
                "lease_owner": self._lease["owner"] if self._lease else None,
                "fault": self._fault.as_dict() if self._fault else None,
                "restored_connection_fault": bool(
                    self._restored_fault_instance_id is not None and self._fault is not None
                    and self._fault.code in {"DISCONNECTED", "SENSOR_STALE"}),
                "last_stop": ({"requested": None, "confirmed": False, "scope": "arm_joints",
                               "reason": "stop_in_progress"} if stop_pending else dict(self._last_stop)),
                "display_observation": display_observation,
                "last_confirmed_applied_target": deepcopy(candidate) if target_confirmed else None,
                "workspace": self.config.workspace_status(),
                "capabilities": {"hardware_precision_validated": self.config.precision_validated if self.config.backend == "sdk" else None, "calibration_jog": self.config.mode == "calibration", "camera": False, "full_collision_check": False, "task_success_verification": False,
                                 "expected_layout_recovery": True,
                                 "mock_joint_trajectory": self.config.backend == "mock" and self.config.hardware_enabled is False,
                                 "blended_joint_path_supported": self.blended_path_scope != 'unavailable',
                                 "blended_joint_path_scope": self.blended_path_scope},
            }

    def acquire(self, owner, *, expected_revision=None, expected_instance_id=None):
        owner = identifier(owner, "owner")
        with self._condition:
            if expected_revision is not None or expected_instance_id is not None:
                if (type(expected_revision) is not int or expected_revision < 0
                        or expected_instance_id != self.instance_id or self._revision != expected_revision):
                    raise ControlError('CONTROLLER_CHANGED', 'Controller state changed before control was acquired')
            if self._fault:
                raise self._fault
            if self._closed or self._closing.is_set():
                raise ControlError("CLOSED", "Controller is shutting down")
            if self._active or self._lease is not None:
                raise ControlError("LEASE_BUSY", "A control lease is already active")
            if not self._fresh(self._snapshot) or self._snapshot.moving is not False:
                raise ControlError("NOT_READY", "Fresh stationary feedback is required before acquiring control")
            if self._state == "stopped" and not self._last_stop["confirmed"]:
                raise ControlError("STOP_UNCONFIRMED", "The last stop has not been confirmed")
            lease_id = uuid.uuid4().hex
            with self._stop_intent_lock:
                if self._pending_stops:
                    raise ControlError("STOP_IN_PROGRESS", "Wait for the stop request to finish")
                self._cancel.clear()
                self._lease = {"lease_id": lease_id, "owner": owner,
                               "deadline": time.monotonic() + self.limits.lease_seconds,
                               "episode_id": uuid.uuid4().hex}
                self._history = []
                self._lease_finished_commands.clear()
                self._state = "armed"
            self._revision += 1
            self._corrections = 0
            self.journal.append("lease_acquired", instance_id=self.instance_id, owner=owner, episode_id=self._lease["episode_id"])
            return {"lease_id": lease_id, "episode_id": self._lease["episode_id"], "expires_in_seconds": self.limits.lease_seconds}

    def recover(self,observation_id,layout_confirmed,*,layout_basis=None,expected_instance_id=None):
        """Acknowledge a resolved fault without device I/O or command replay.

        Requires an ended episode, confirmed stop and fresh feedback. An unknown
        command or infrastructure failure must be inspected, never erased here.
        The caller may retain its already recorded expected layout after a
        connection fault in this instance; this is not a user confirmation.
        """
        expected_layout = layout_confirmed is False and layout_basis == 'expected'
        unregistered = layout_confirmed is False and layout_basis == 'unregistered'
        if not expected_layout and not unregistered and not (layout_confirmed is True and layout_basis is None):
            raise ControlError('LAYOUT_CONFIRMATION_REQUIRED','현재 물체 배치를 다시 등록해야 합니다.')
        identifier(observation_id,'observation_id')
        if expected_instance_id is not None:
            identifier(expected_instance_id,'expected_instance_id')
        with self._condition:
            if expected_instance_id is not None and expected_instance_id != self.instance_id:
                raise ControlError('CONTROLLER_CHANGED','복구할 상주 제어기가 바뀌었습니다.')
            if self._closed or self._closing.is_set():
                raise ControlError('CLOSED','Controller is shutting down')
            if self._active is not None or self._lease is not None:
                raise ControlError('RECOVERY_BUSY','진행 중인 작업 또는 제어 소유권이 남아 있습니다.')
            self._validate_observation(observation_id)
            if not self.worker.alive or self._snapshot.moving is not False:
                raise ControlError('NOT_READY','새 정지 관측과 살아 있는 제어기가 필요합니다.')
            if self._fault is None:
                return {'recovered':False,'reason':'no_latched_fault','instance_id':self.instance_id}
            if self._fault.code not in RECOVERABLE_FAULT_CODES or self.journal.pending:
                raise ControlError('RECOVERY_REVIEW_REQUIRED',
                    '명령 기록 또는 제어기 내부 오류를 먼저 점검해야 합니다. 자동 초기화하지 않습니다.',
                    fault=self._fault.as_dict(),pending_commands=self.journal.pending)
            if expected_layout and (self._fault.code not in {'DISCONNECTED','SENSOR_STALE'}
                                    or self._restored_fault_instance_id is not None):
                raise ControlError('RECOVERY_REVIEW_REQUIRED',
                    '기존 예상 배치를 사용한 복구는 같은 제어기의 통신 오류에만 적용됩니다.',
                    fault=self._fault.as_dict())
            if unregistered and (self._fault.code not in {'DISCONNECTED','SENSOR_STALE'}
                                 or self._restored_fault_instance_id is None):
                raise ControlError('RECOVERY_REVIEW_REQUIRED',
                    '새 연결에 복원된 통신 오류만 배치 확인 없이 해제할 수 있습니다.',
                    fault=self._fault.as_dict())
            if self._last_stop.get('confirmed') is not True:
                raise ControlError('STOP_UNCONFIRMED','정지 버튼으로 정지를 확인한 뒤 다시 등록해 주세요.')
            warning=self._snapshot.firmware_warning_raw
            if warning==1 or ((self.config.backend=='sdk' or self._fault.code=='DEVICE_WARNING') and warning!=0):
                raise ControlError('DEVICE_WARNING','장치 경고가 해소됐다는 새 수신이 필요합니다.')
            if expected_layout or unregistered:
                with self._stop_intent_lock:
                    if self._pending_stops:
                        raise ControlError('STOP_IN_PROGRESS','정지 요청이 진행 중입니다.')
            previous=self._fault.as_dict()
            # Persist the acknowledgement before changing admission state.
            restored = ({'restored_fault_instance_id': self._restored_fault_instance_id}
                        if self._restored_fault_instance_id is not None else {})
            self.journal.append('fault_recovered',instance_id=self.instance_id,previous_fault=previous,
                observation_id=observation_id,layout_confirmed=layout_confirmed,
                layout_basis='expected' if expected_layout else 'unregistered' if unregistered else 'confirmed',replayed_commands=0,
                device_commands_sent=0,physical_success=None,**restored)
            self._fault=None;self._state='stopped';self._revision+=1
            self._restored_fault_instance_id=None
            self._observations.clear()
            self._condition.notify_all()
            return {'recovered':True,'previous_fault':previous,'instance_id':self.instance_id,
                    'layout_confirmed':layout_confirmed,
                    'layout_basis':'expected' if expected_layout else 'unregistered' if unregistered else 'confirmed',
                    'replayed_commands':0,'device_commands_sent':0}

    def _require_lease(self, lease_id, *, renew=True):
        if self._closed or self._closing.is_set():
            raise ControlError("CLOSED", "Controller is shutting down")
        if self._fault:
            raise self._fault
        with self._stop_intent_lock:
            if self._lease is None or lease_id != self._lease["lease_id"]:
                raise ControlError("LEASE_REQUIRED", "A matching control lease is required")
            if time.monotonic() >= self._lease["deadline"]:
                raise ControlError("LEASE_EXPIRED", "The control lease has expired")
            # Observation calls do not extend a running command. SDK motion
            # may renew its lease only on observed directional progress.
            if renew and self._active is None:
                self._lease["deadline"] = time.monotonic() + self.limits.lease_seconds

    def observe(self, lease_id=None):
        with self._condition:
            if lease_id is not None:
                self._require_lease(lease_id)
            snapshot = self._snapshot
            fresh = self._fresh(snapshot)
            now = time.monotonic()
            observation_id = uuid.uuid4().hex
            joints = snapshot.joints_deg if snapshot and fresh else None
            tcp = self.geometry.forward(joints) if joints is not None else None
            feedback_valid = (snapshot is not None and snapshot.gripper_time is not None
                              and 0 <= now - snapshot.gripper_time <= self.limits.sensor_max_age
                              and snapshot.connected)
            if self._state == "awaiting_mock_observation" and lease_id is not None:
                self._state = "armed"
            value = {
                "schema_version": "PolicyObservation.phase1",
                "observation_id": observation_id,
                "instance_id": self.instance_id,
                "revision": self._revision,
                "observed_monotonic": now,
                "simulated": self.config.backend == "mock",
                "state": self._state,
                "joints_deg": joints,
                "tcp_mm": tcp,
                "sensor_sequence": snapshot.sensor_sequence if snapshot else None,
                "sensor_age_seconds": now - snapshot.sensor_time if snapshot and snapshot.sensor_time is not None else None,
                "sensor_valid": fresh,
                "moving": snapshot.moving if snapshot and fresh else None,
                "gripper_command": self._gripper_command,
                "gripper_reported_state": snapshot.gripper_reported_state if feedback_valid else None,
                "gripper_reported_raw": snapshot.gripper_raw if feedback_valid else None,
                "gripper_module_type": snapshot.gripper_module_type if feedback_valid else None,
                "gripper_feedback_age_seconds": now - snapshot.gripper_time if snapshot and snapshot.gripper_time is not None else None,
                "gripper_reported_valid": feedback_valid and snapshot.gripper_reported_state is not None,
                "gripper_feedback_fresh": feedback_valid,
                "device_diagnostics": {
                    "firmware_warning_raw": snapshot.firmware_warning_raw if snapshot and fresh else None,
                    "collision_event_counts": snapshot.collision_event_counts if snapshot and fresh else None,
                    "battery_state_raw": snapshot.battery_state_raw if snapshot and fresh else None,
                    "signal_strength_raw": snapshot.signal_strength_raw if snapshot and fresh else None,
                },
                "object_held_visual": None,
                "perception": {"available": False, "source": "not_implemented", "objects": [], "alignment_error_px": None},
                "references": {"tcp": tcp} if tcp else {},
                "recent_actions": list(self._history[-2:]),
                "move_count": snapshot.move_count if snapshot else 0,
                "stop_count": snapshot.stop_count if snapshot else 0,
            }
            self._observations[observation_id] = value
            while len(self._observations) > OBSERVATION_HISTORY_LIMIT:
                self._observations.popitem(last=False)
            return deepcopy(value)

    def diagnostics(self):
        """Read raw SDK diagnostics even while a stopped controller is faulted."""
        if self.config.backend!="sdk":
            raise ControlError("NOT_AVAILABLE","Raw device diagnostics require the SDK backend")
        return self.worker.call("diagnostics")

    def _validate_observation(self, observation_id):
        observation = self._observations.get(observation_id)
        if observation is None or observation["revision"] != self._revision:
            raise ControlError("OBSERVATION_STALE", "Observe again before issuing a new command")
        if time.monotonic() - observation["observed_monotonic"] > self.limits.observation_max_age:
            raise ControlError("OBSERVATION_STALE", "The observation has expired")
        if not self._fresh(self._snapshot) or not observation["sensor_valid"]:
            raise ControlError("SENSOR_STALE", "Fresh device feedback is required")
        if max(abs(a - b) for a, b in zip(self._snapshot.joints_deg, observation["joints_deg"])) > self.limits.observation_joint_drift_deg:
            raise ControlError("STATE_CHANGED", "The arm changed after the supplied observation")

    def record_calibration(self, lease_id, label, reference_mm=None, note="", gripper_seen=None):
        """Attach an operator measurement to fresh feedback, without approving a workspace."""
        label = identifier(label, "label")
        reference = vector(reference_mm, 3, "reference_mm") if reference_mm is not None else None
        if not isinstance(note, str) or len(note) > 2000:
            raise ControlError("INVALID_REQUEST", "note must be text of at most 2000 characters")
        if gripper_seen is not None and gripper_seen not in ("open", "closed"):
            raise ControlError("INVALID_GRIPPER", "gripper_seen must be open or closed")
        with self._condition:
            self._require_lease(lease_id)
            if self.config.mode != "calibration":
                raise ControlError("MODE_RESTRICTED", "Record measurements in calibration mode")
            if self._active or self._state != "armed":
                raise ControlError("NOT_ARMED", "Finish or stop the active movement before recording")
            if not self._fresh(self._snapshot) or self._snapshot.moving is not False:
                raise ControlError("NOT_READY", "Recording requires fresh stationary arm feedback")
            observation = self.observe(lease_id)
            if gripper_seen is not None and (not observation["gripper_feedback_fresh"] or observation["gripper_reported_raw"] is None):
                raise ControlError("GRIPPER_FEEDBACK_UNAVAILABLE", "No fresh gripper byte is available for this observation")
            sample = {
                "schema": "raccoon-calibration-sample.v1", "at": datetime.now(timezone.utc).isoformat(),
                "label": label, "note": note, "backend": self.config.backend, "port": self.config.port,
                "tool_forward_mm": self.config.tool_forward_mm, "tool_vertical_mm": self.config.tool_vertical_mm,
                "observation": observation, "reference_mm": reference,
                "reference_source": "operator_entry" if reference is not None else None,
                "nominal_error_mm": tuple(a - b for a, b in zip(observation["tcp_mm"], reference)) if reference is not None else None,
                "gripper_seen": gripper_seen, "gripper_seen_source": "operator_entry" if gripper_seen is not None else None,
                "workspace_verified": self.config.workspace_verified, "calibration_complete": False,
            }
            self.journal.append("calibration_sample_recorded", instance_id=self.instance_id, sample=sample)
            return deepcopy(sample)

    @contextmanager
    def _record_command_intention(self, request):
        """Publish only the identity of an authorized request before state-lock I/O."""
        token = object()
        registered = False
        with self._stop_intent_lock:
            lease = self._lease
            if (lease is not None and lease['lease_id'] == request['lease_id']
                    and time.monotonic() < lease['deadline']):
                action = request.get('action')
                self._command_intentions[token] = {
                    'lease_id': request['lease_id'], 'command_id': request['command_id'],
                    'kind': action.get('kind') if isinstance(action, dict) else None}
                registered = True
        try:
            yield
        finally:
            if registered:
                with self._stop_intent_lock:
                    self._command_intentions.pop(token, None)

    def _unsent_retarget_result(self, command_id):
        return {"command_id": command_id, "status": "cancelled", "applied": False,
                "reason": "RETARGET_BEFORE_SEND", "simulated": self.config.backend == 'mock',
                "task_success": None, "replayed": False}

    def _trip_damped_correction_fault(self, request, error):
        """Trip controller fault when a bound damped nominal correction becomes invalid."""
        action = request.get('action')
        pending = self._pending_nominal_correction
        if (isinstance(action, dict) and pending is not None
                and action.get('correction_of_command_id') == pending.get('parent_command_id')
                and pending.get('basis', {}).get('kind') == 'damped_nominal_half'):
            self._trip_fault(error)

    @measured('controller')
    def apply_action(self, request):
        keys(request, allowed={"lease_id", "command_id", "observation_id", "action"}, required={"lease_id", "command_id", "observation_id", "action"})
        command_id = identifier(request["command_id"], "command_id")
        identifier(request["lease_id"], "lease_id")
        identifier(request["observation_id"], "observation_id")
        try:
            encoded_request = json.dumps(request, sort_keys=True, allow_nan=False)
            fingerprint = hashlib.sha256(encoded_request.encode()).hexdigest()
            request = json.loads(encoded_request)
        except (ValueError, TypeError) as exc:
            raise ControlError("INVALID_REQUEST", "Action must be finite JSON") from exc
        with self._record_command_intention(request), self._condition:
            existing = self.journal.records.get(command_id)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ControlError("COMMAND_ID_CONFLICT", "This command ID has already been used for another request")
                if existing["result"] is None:
                    return {"command_id": command_id, "status": "in_progress", "replayed": True}
                return {**deepcopy(existing["result"]), "replayed": True}
            self._require_lease(request["lease_id"])
            with self._stop_intent_lock:
                preempted = self._retarget_preempted_commands.get(command_id) == request['lease_id']
            if preempted:
                # The reservation is checked even after a later re-arm, so a
                # delayed call cannot send an obsolete target.
                return self._unsent_retarget_result(command_id)
            if self._active is not None or self._preparing is not None:
                raise ControlError("BUSY", "Wait for the current command to finish")
            if self._state != "armed":
                raise ControlError("NOT_ARMED", "Control is stopped or awaiting a new observation")
            try:
                self._validate_observation(request["observation_id"])
            except ControlError as exc:
                self._trip_damped_correction_fault(request, exc)
                raise
            preparing = None
            if isinstance(request['action'], dict) and request['action'].get('kind') == 'blended_joint_path':
                preparing = {'lease_id': request['lease_id'], 'command_id': command_id, 'kind': 'blended_joint_path'}
                with self._stop_intent_lock:
                    self._preparing = preparing
            try:
                with span('action_validation'):
                    prepared = self._prepare(request["action"])
            except ControlError as exc:
                with self._stop_intent_lock:
                    preempted = self._retarget_preempted_commands.get(command_id) == request['lease_id']
                if self._cancel.is_set() and preempted:
                    return self._unsent_retarget_result(command_id)
                self._trip_damped_correction_fault(request, exc)
                raise
            finally:
                if preparing is not None:
                    with self._stop_intent_lock:
                        if self._preparing is preparing:
                            self._preparing = None
            with self._stop_intent_lock:
                cancelled = self._cancel.is_set()
                preempted = self._retarget_preempted_commands.get(command_id) == request['lease_id']
            if preempted:
                return self._unsent_retarget_result(command_id)
            if cancelled:
                raise ControlError("STOP_REQUESTED", "Stop was requested while the action was being checked")
            if preparing is not None:
                # The state lock was released for bounded path hashing.
                # Recheck ownership and the original observation before send.
                self._require_lease(request['lease_id'])
                self._validate_observation(request['observation_id'])
            try:
                with span('journal_admission'):
                    self.journal.admit(command_id, fingerprint, request, self._observations[request["observation_id"]])
            except ControlError as exc:
                self._trip_fault(exc)
                raise
            self._revision += 1
            before = self._snapshot
            with self._stop_intent_lock:
                self._active = command_id
                self._active_kind = prepared['kind']
                if prepared['kind'] == 'set_gripper':
                    self._gripper_evidence = None
                # Start with the existing admission budget. Physical SDK
                # motion renews this only when encoder progress is observed;
                # polling and repeated calls never extend an active command.
                if prepared['kind'] == 'blended_joint_path':
                    command_budget = prepared['blended_spec']['motion_timeout_seconds']
                elif prepared['kind'] == 'set_gripper':
                    command_budget = min(GRIPPER_TIMEOUT_CAP_SECONDS, self.limits.motion_timeout)
                else:
                    command_budget = self.limits.motion_timeout
                self._lease['deadline'] = max(self._lease['deadline'],
                    time.monotonic() + command_budget + self.limits.rpc_timeout)
                self._active_collision_baseline = (tuple(before.collision_event_counts)
                    if before is not None and before.collision_event_counts is not None else None)
                self._active_warning_baseline = before.firmware_warning_raw if before is not None else None
                self._state = "stopped" if self._cancel.is_set() else "executing"
                admitted_stop_epoch = self._stop_epoch
        try:
            if prepared["kind"] == "reobserve":
                value = {"status": "observed", "applied": False}
            elif prepared["kind"] == "set_gripper":
                value = self._execute_gripper(prepared)
            else:
                value = self._execute_motion(prepared, before)
        except ControlError as exc:
            self._trip_fault(exc)
            retain_request = (prepared.get('completion_mode') == 'retain_position_target'
                              or 'retain_position_from_command_id' in prepared)
            stop = self._fault_stop_result() if retain_request else dict(self._last_stop)
            value = {"status": "failed", "applied": None, "error": exc.as_dict(), "stop": stop}
            if prepared['kind'] == 'set_gripper' and self._gripper_evidence is not None:
                value.update(deepcopy(self._gripper_evidence))
        except Exception as exc:
            error = ControlError("CONTROLLER_ERROR", type(exc).__name__)
            self._trip_fault(error)
            retain_request = (prepared.get('completion_mode') == 'retain_position_target'
                              or 'retain_position_from_command_id' in prepared)
            stop = self._fault_stop_result() if retain_request else dict(self._last_stop)
            value = {"status": "failed", "applied": None, "error": error.as_dict(), "stop": stop}
            if prepared['kind'] == 'set_gripper' and self._gripper_evidence is not None:
                value.update(deepcopy(self._gripper_evidence))
        finish_error = None
        finish_fault_new = False
        with self._condition:
            # Finalization and stop intent share one short ordering boundary.
            # Never hold this lock during the observation or journal I/O below:
            # stop intent must not wait for those operations.
            with self._stop_intent_lock:
                if self._cancel.is_set() and value.get('status') in SUCCESS_LIKE_STATUSES:
                    if prepared.get('direct_student_target') is True:
                        completion = value.get('completion')
                        if isinstance(completion, dict) and isinstance(completion.get('host_frame'), dict):
                            # Stop invalidates a hold, but not the historical
                            # fact that this target appeared in an SDK frame.
                            value['interrupted_host_frame'] = deepcopy(completion['host_frame'])
                    if self._fault is not None:
                        value = {**value, 'status': 'failed', 'error': self._fault.as_dict()}
                    else:
                        value = {**value, 'status': 'cancelled', 'reason': self._cancel_reason}
                    # A previous waypoint's stop confirmation is not evidence
                    # for the stop that won this race.
                    value['stop'] = {'requested': None, 'confirmed': False, 'scope': 'arm_joints',
                                     'reason': 'stop_evidence_not_bound_to_action'}
                    # A stop that wins finalization invalidates position-mode
                    # and gripper hold evidence, even if arrival was observed
                    # just before the stop request.
                    for field in ('completion', 'arm_hold', 'retained_parent', 'retained_position_from_command_id',
                                  'retained_position_observation', 'correction_action', 'correction_basis'):
                        value.pop(field, None)
                elif (value.get('status') in {'cancelled','failed'}
                      and isinstance(value.get('stop'),dict) and value['stop'].get('confirmed') is True
                      and (self._pending_stops or self._stop_epoch != admitted_stop_epoch)):
                    value['earlier_stop_evidence'] = deepcopy(value['stop'])
                    value['stop'] = {'requested':None, 'confirmed':False, 'scope':'arm_joints',
                                     'reason':'stop_evidence_not_bound_to_current_request'}
                if prepared['kind'] in MOTION_KINDS:
                    completion = value.get('completion')
                    if (value.get('status') in RETAINED_POSITION_SUCCESS_STATUSES and value.get('applied') is True
                            and isinstance(completion, dict)
                            and completion.get('mode') == 'position_target_retained'):
                        self._retained_position_target = {
                            'instance_id': self.instance_id, 'lease_id': request['lease_id'],
                            'stop_epoch': admitted_stop_epoch, 'command_id': command_id,
                            'action': deepcopy(prepared), 'before': before,
                            'completion': deepcopy(completion),
                        }
                    if (value.get('status') == 'completed' and value.get('applied') is True
                            and value.get('stop', {}).get('confirmed') is True
                            and ((self.config.backend != 'sdk' and 'nominal_arrival' not in prepared)
                                 or value.get('sent_target_reached') is True)):
                        self._last_confirmed_applied_target = {
                            'instance_id': self.instance_id, 'command_id': command_id,
                            'joints_deg': list(prepared['effective_target_deg']),
                        }
                    elif value.get('applied') is not False:
                        self._last_confirmed_applied_target = None
                if self._fault is None:
                    if self._cancel.is_set() or self._lease is None:
                        self._state = "stopped"
                    elif value["status"] == "settled_needs_observation":
                        self._state = "awaiting_mock_observation"
                    else:
                        self._state = "armed"
                self._active = None
                self._active_kind = None
                self._active_collision_baseline = None
                self._active_warning_baseline = None
                if self._lease is not None and self._lease['lease_id'] == request['lease_id']:
                    # Once this command ends, restore the ordinary idle lease
                    # interval instead of retaining unused motion budget.
                    self._lease['deadline'] = time.monotonic() + self.limits.lease_seconds
                self._revision += 1
                self._history.append({"command_id": command_id, "action": request["action"], "status": value["status"]})
                self._lease_finished_commands.add(command_id)
            value = {"command_id": command_id, "simulated": self.config.backend == "mock", "task_success": None, "replayed": False, **value}
            try:
                value["observation"] = self.observe()
                with span('journal_result'):
                    self.journal.finish(command_id, value)
                if value.get('status') in NOMINAL_CORRECTION_STATUSES:
                    self._pending_nominal_correction = {
                        'lease_id': request['lease_id'],
                        'parent_command_id': value['correction_basis'].get('parent_command_id', command_id),
                        'binding_command_id': command_id,
                        'action': deepcopy(value['correction_action']),
                        'basis': deepcopy(value['correction_basis']),
                        'warning_baseline': before.firmware_warning_raw,
                        'collision_baseline': before.collision_event_counts,
                        'stop_epoch': admitted_stop_epoch,
                    }
                else:
                    self._pending_nominal_correction = None
            except Exception as exc:
                finish_error = (exc if isinstance(exc,ControlError) else
                    ControlError('RESULT_COMMIT_FAILED','The command result could not be recorded',
                                 error_type=type(exc).__name__))
                # Close admission before releasing _condition. A successful
                # motion with an unfinished journal must never be replayed.
                finish_fault_new = self._latch_fault_state(finish_error)
            self._condition.notify_all()
        if finish_error:
            if finish_fault_new:
                try:
                    self._stop_device()
                except Exception as exc:
                    with self._condition:
                        self._last_stop = {'requested':None,'confirmed':False,
                                           'error':{'code':'STOP_ATTEMPT_FAILED','type':type(exc).__name__}}
                try:
                    self.journal.append('fault',instance_id=self.instance_id,error=finish_error.as_dict())
                except Exception:
                    pass
            raise finish_error
        return deepcopy(value)

    @staticmethod
    def _valid_collision_counts(value):
        return (isinstance(value, (tuple, list)) and len(value) == JOINT_COUNT
                and all(type(item) is int and item >= 0 for item in value))

    @staticmethod
    def _require_quiet_motion_diagnostics(baseline_warning, baseline_counts, snapshot):
        """Evidence for a bounded correction; missing/reset data are not quiet."""
        warning, counts = snapshot.firmware_warning_raw, snapshot.collision_event_counts
        if baseline_warning == 1 or warning == 1:
            raise ControlError('DEVICE_WARNING', 'Device warning blocks nominal correction')
        if (type(baseline_warning) is not int or baseline_warning != 0
                or type(warning) is not int or warning != 0
                or not Controller._valid_collision_counts(baseline_counts)
                or not Controller._valid_collision_counts(counts)):
            raise ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE',
                               'Valid warning and collision counters are required before and after motion')
        if any(new < old for old, new in zip(baseline_counts, counts)):
            raise ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE', 'Collision counters were reset during motion')
        if any(new > old for old, new in zip(baseline_counts, counts)):
            raise ControlError('DEVICE_COLLISION_EVENT', 'A collision event blocks nominal correction',
                               physical_contact_cause_confirmed=False)

    def _require_retained_position_capability(self):
        try:
            capability = self.worker.call('position_target_capability')
        except ControlError as exc:
            raise ControlError('UNSUPPORTED_ACTION', 'This worker does not support retained position targets') from exc
        if not isinstance(capability, dict) or capability.get('version') != 'retained-position-target.v3':
            raise ControlError('UNSUPPORTED_ACTION', 'This worker does not support retained position targets')

    @staticmethod
    def _position_frame_matches(frame, steps, speed, *, command_sequence=None, after_sequence=-1,
                                target_segment_start_sequence=None):
        return (isinstance(frame, dict) and frame.get('mode_hex') == '11'
                and type(frame.get('frame_sequence')) is int and frame['frame_sequence'] > after_sequence
                and type(frame.get('target_segment_start_sequence')) is int
                and 1 <= frame['target_segment_start_sequence'] <= frame['frame_sequence']
                and (target_segment_start_sequence is None
                     or frame['target_segment_start_sequence'] == target_segment_start_sequence)
                and type(frame.get('command_sequence')) is int
                and 0 <= frame['command_sequence'] <= 255
                and frame.get('joint_steps') == list(steps)
                and frame.get('speed_pct') == int(.5 + speed)
                and type(frame.get('monotonic')) in (int, float)
                and (command_sequence is None or frame['command_sequence'] == command_sequence))

    def _retained_arrival_details(self, action, snapshot):
        """Describe the exact encoder sample used for an arrival decision."""
        observed_tcp = self.geometry.forward(snapshot.joints_deg)
        nominal = action['nominal_arrival']
        return {'observed_joints_deg': list(snapshot.joints_deg),
                'joint_error_deg': max(abs(a-b) for a, b in zip(snapshot.joints_deg, action['effective_target_deg'])),
                'tcp_error_mm': distance(observed_tcp, action['target_tcp_mm']),
                'arrival_joint_error_deg': max(abs(a-b) for a, b in zip(snapshot.joints_deg, nominal['effective_target_deg'])),
                'arrival_tcp_error_mm': distance(observed_tcp, nominal['target_tcp_mm']),
                'sensor_sequence': snapshot.sensor_sequence, 'sensor_monotonic': snapshot.sensor_time}

    def _checked_retained_position(self, command_id, snapshot, *, require_arrival=True, require_stationary=True):
        """Validate instance-local host evidence and current measured arrival."""
        with self._condition:
            retained = self._retained_position_target
            with self._stop_intent_lock:
                valid = (retained is not None and retained['instance_id'] == self.instance_id
                         and self._lease is not None and retained['lease_id'] == self._lease['lease_id']
                         and retained['stop_epoch'] == self._stop_epoch and not self._pending_stops
                         and not self._cancel.is_set() and self._fault is None
                         and retained['command_id'] == command_id)
            if not valid:
                raise ControlError('POSITION_TARGET_UNAVAILABLE', 'No matching active retained position target exists')
            action, before = retained['action'], retained['before']
            completion = retained['completion']
            if (not self._fresh(snapshot) or (require_stationary and snapshot.moving is not False)
                    or snapshot.sensor_sequence < completion['sensor_sequence']
                    or snapshot.sensor_time < completion['sensor_monotonic']):
                raise ControlError('SENSOR_STALE', 'Fresh stationary feedback is required for the retained target')
            self._require_quiet_motion_diagnostics(before.firmware_warning_raw, before.collision_event_counts, snapshot)
            self._validate_motion_feedback(action, before, snapshot)
            target, nominal = action['effective_target_deg'], action['nominal_arrival']
            nominal_only = 'damped_compensation' in action
            if require_arrival and ((not nominal_only and (max(abs(a-b) for a, b in zip(snapshot.joints_deg, target)) > self.limits.joint_tolerance_deg
                    or distance(self.geometry.forward(snapshot.joints_deg), action['target_tcp_mm']) > action['arrival_tolerance_mm']))
                    or max(abs(a-b) for a, b in zip(snapshot.joints_deg, nominal['effective_target_deg'])) > self.limits.joint_tolerance_deg
                    or distance(self.geometry.forward(snapshot.joints_deg), nominal['target_tcp_mm']) > action['arrival_tolerance_mm']):
                raise ControlError('TARGET_NOT_REACHED', 'The retained target is outside its original arrival bounds',
                                   **self._retained_arrival_details(action, snapshot))
            frame = self.worker.call('position_target_frame')
            original_frame = completion['host_frame']
            if (not self._position_frame_matches(frame, action['steps'], action['speed_pct'],
                                                 command_sequence=original_frame['command_sequence'],
                                                 target_segment_start_sequence=original_frame['target_segment_start_sequence'],
                                                 after_sequence=original_frame['frame_sequence']-1)
                    or not 0 <= time.monotonic()-frame['monotonic'] <= self.limits.sensor_max_age):
                raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The last SDK host position frame is unavailable or changed')
            return retained, frame

    def _check_damped_compensation_error(self, action, snapshot, *, allow_transient_worsening=False):
        """Compare actual observations with the original goal and stopped basis."""
        basis = action['damped_compensation']
        nominal = action['nominal_arrival']['effective_target_deg']
        before_error = [target-observed for target, observed in zip(nominal, basis['joints_deg'])]
        after_error = [target-observed for target, observed in zip(nominal, snapshot.joints_deg)]
        joint_error = max(map(abs, after_error))
        nominal_tcp = action['nominal_arrival']['target_tcp_mm']
        before_tcp = self.geometry.forward(basis['joints_deg'])
        after_tcp = self.geometry.forward(snapshot.joints_deg)
        tcp_error = distance(after_tcp, nominal_tcp)
        reverse_outside_joint = any(a*b < 0 and abs(b) > self.limits.joint_tolerance_deg
                                    for a, b in zip(before_error, after_error))
        reverse_outside_tcp = (sum((target-old)*(target-new) for target, old, new
                                   in zip(nominal_tcp, before_tcp, after_tcp)) < 0
                               and tcp_error > action['arrival_tolerance_mm'])
        if (reverse_outside_joint or reverse_outside_tcp
                or (not allow_transient_worsening and
                    (joint_error > basis['baseline_joint_error_deg'] or tcp_error > basis['baseline_tcp_error_mm']))):
            raise ControlError('DAMPED_COMPENSATION_DIVERGED', 'The bounded compensation worsened the original target error',
                               before_joint_error_deg=basis['baseline_joint_error_deg'],
                               after_joint_error_deg=joint_error, before_tcp_error_mm=basis['baseline_tcp_error_mm'],
                               after_tcp_error_mm=tcp_error)
        return joint_error, tcp_error

    def _require_damped_parent_frame(self, basis, snapshot):
        with self._stop_intent_lock:
            if self._cancel.is_set() or self._pending_stops or self._stop_epoch != basis['stop_epoch']:
                raise ControlError('STOP_IN_PROGRESS', 'A stop superseded the retained parent target')
        if (not self._fresh(snapshot) or snapshot.moving is not False
                or snapshot.sensor_sequence < basis['sensor_sequence']
                or snapshot.sensor_time < basis['sensor_monotonic']):
            raise ControlError('SENSOR_STALE', 'Fresh stationary feedback is required for compensation')
        parent = basis['original_host_frame']
        frame = self.worker.call('position_target_frame')
        if (not self._position_frame_matches(frame, basis['nominal_steps'], parent['speed_pct'],
                command_sequence=parent['command_sequence'],
                target_segment_start_sequence=parent['target_segment_start_sequence'],
                after_sequence=parent['frame_sequence']-1)
                or not 0 <= time.monotonic()-frame['monotonic'] <= self.limits.sensor_max_age):
            raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The original host position target did not remain continuous')
        return frame

    @staticmethod
    def _is_independent_single_transport(action):
        """Only an admitted single transport uses aggregate progress timing."""
        constraints = action.get('path_constraints', {})
        envelope = action.get('path', {}).get('independent_joint_envelope', {})
        nominal = action.get('nominal_arrival')
        return (action.get('kind') == 'move_joints'
                and action.get('correction_of_command_id') is None
                and constraints.get('minimum_z_mm') is not None
                and constraints.get('vertical_xy_mm') is None
                and envelope.get('independent_joint_progress') is True
                and (nominal is None or tuple(nominal['steps']) == tuple(action['steps'])))

    @staticmethod
    def _remaining_joint_steps(target_steps, observed_joints):
        observed_steps, _ = quantize_joints(observed_joints)
        return sum(abs(target-observed) for target, observed in zip(target_steps, observed_steps))

    def _check_motion_envelope(self, start, end, constraints, *, start_tolerance_deg=0):
        """Evaluate the independent-joint envelope between start and end under task constraints."""
        envelope = motion_bounds(
            self.geometry, start, end, self.limits.joint_tolerance_deg,
            start_tolerance_deg=start_tolerance_deg)
        return check_envelope(
            envelope, self.limits,
            vertical_xy=constraints.get('vertical_xy_mm'),
            corridor_mm=constraints.get('corridor_mm'),
            minimum_z_mm=constraints.get('minimum_z_mm'),
            maximum_z_mm=constraints.get('maximum_z_mm'))

    def _target_error_metrics(self, observed_joints, target_joints, target_tcp, arrival_tolerance):
        """Return (joint_error_deg, tcp_error_mm, reached) for observed joints against a target."""
        joint_error = max(abs(a - b) for a, b in zip(observed_joints, target_joints))
        tcp_error = distance(self.geometry.forward(observed_joints), target_tcp)
        reached = (joint_error <= self.limits.joint_tolerance_deg and tcp_error <= arrival_tolerance)
        return joint_error, tcp_error, reached

    def _validate_retained_and_correction_flags(self, action, kind):
        """Validate retained-position, follow-up, direct-student, and nominal-correction flags."""
        completion_mode = action.get('completion_mode')
        if 'completion_mode' in action:
            if completion_mode != 'retain_position_target':
                raise ControlError('INVALID_REQUEST', 'Unknown motion completion mode')
            if (self.config.backend != 'sdk' or self.config.hardware_enabled is not True
                    or 'nominal_arrival_joints_deg' not in action):
                raise ControlError('UNSUPPORTED_ACTION', 'Retained position requires an SDK motion with a nominal target')
            self._require_retained_position_capability()
        damped_opt_in = action.get('allow_damped_compensation', False)
        if type(damped_opt_in) is not bool:
            raise ControlError('INVALID_REQUEST', 'Damped compensation opt-in must be boolean')
        if damped_opt_in and (completion_mode != 'retain_position_target'
                                or action.get('allow_nominal_correction') is not True
                                or 'correction_of_command_id' in action):
            raise ControlError('DAMPED_COMPENSATION_UNAVAILABLE', 'A single initial retained SDK motion is required')
        follow_up = action.get('allow_target_follow_up', False)
        if type(follow_up) is not bool or (follow_up and completion_mode != 'retain_position_target'):
            raise ControlError('INVALID_REQUEST', 'Target follow-up requires a retained SDK motion')
        direct_student_target = action.get('direct_student_target', False)
        if (type(direct_student_target) is not bool
                or (direct_student_target and (kind != 'move_joints'
                    or completion_mode != 'retain_position_target'
                    or action.get('path_constraints', {})
                    or action.get('allow_nominal_correction', False)
                    or action.get('allow_damped_compensation', False)
                    or action.get('correction_of_command_id') is not None))):
            raise ControlError('INVALID_REQUEST', 'Direct student targets require an unmodified retained SDK move')
        correction_parent = action.get('correction_of_command_id')
        pending = None
        if correction_parent is not None:
            identifier(correction_parent, 'correction_of_command_id')
            pending = self._pending_nominal_correction
            if (pending is None or pending['parent_command_id'] != correction_parent
                    or self._lease is None or pending['lease_id'] != self._lease['lease_id']
                    or not self._history or self._history[-1]['command_id'] != pending.get('binding_command_id', correction_parent)):
                raise ControlError('NOMINAL_CORRECTION_UNAVAILABLE', 'No matching pending nominal correction exists')
            compared_action = dict(action)
            # The previously admitted correction is still identical; the
            # optional completion contract does not alter its motor target.
            compared_action.pop('completion_mode', None)
            if compared_action != pending['action']:
                raise ControlError('NOMINAL_CORRECTION_MISMATCH', 'The proposed correction differs from the bound action')
            with self._stop_intent_lock:
                if self._pending_stops or self._cancel.is_set() or self._stop_epoch != pending['stop_epoch']:
                    raise ControlError('STOP_IN_PROGRESS', 'A stop superseded nominal correction')
            basis = pending['basis']
            if (not self._fresh(self._snapshot) or self._snapshot.moving is not False
                    or self._snapshot.sensor_sequence < basis['sensor_sequence']
                    or self._snapshot.sensor_time < basis['sensor_monotonic']):
                raise ControlError('STOP_UNCONFIRMED', 'A fresh stationary correction observation is required')
            if max(abs(a - b) for a, b in zip(self._snapshot.joints_deg, basis['joints_deg'])) > self.limits.joint_tolerance_deg:
                raise ControlError('STATE_CHANGED', 'The arm moved since the correction was calculated')
            self._require_quiet_motion_diagnostics(pending['warning_baseline'], pending['collision_baseline'], self._snapshot)
            if basis.get('kind') == 'damped_nominal_half':
                if completion_mode != 'retain_position_target':
                    raise ControlError('DAMPED_COMPENSATION_UNAVAILABLE', 'The compensated motion must retain its encoded target')
                self._require_damped_parent_frame(basis, self._snapshot)
        return completion_mode, damped_opt_in, follow_up, direct_student_target, correction_parent, pending

    def _parse_path_constraints(self, action):
        """Validate optional vertical corridor and clearance height constraints."""
        constraints = action.get("path_constraints", {})
        keys(constraints, allowed={"vertical_xy_mm", "corridor_mm", "minimum_z_mm", "maximum_z_mm"})
        constraints = dict(constraints)
        if constraints.get("vertical_xy_mm") is not None:
            constraints["vertical_xy_mm"] = vector(constraints["vertical_xy_mm"], 2, "vertical_xy_mm")
            constraints["corridor_mm"] = number(constraints.get("corridor_mm"), "corridor_mm", minimum=0.01)
        elif constraints.get("corridor_mm") is not None:
            raise ControlError("INVALID_REQUEST", "A corridor radius requires an XY center")
        if constraints.get("minimum_z_mm") is not None:
            constraints["minimum_z_mm"] = number(constraints["minimum_z_mm"], "minimum_z_mm")
        if constraints.get("maximum_z_mm") is not None:
            constraints["maximum_z_mm"] = number(constraints["maximum_z_mm"], "maximum_z_mm",
                                                   maximum=self.limits.workspace_max_mm[2])
            if constraints.get("minimum_z_mm") is not None and constraints['maximum_z_mm'] < constraints['minimum_z_mm']:
                raise ControlError("INVALID_REQUEST", "Path ceiling must not be below its clearance floor")
        return constraints

    def _prepare_mock_joint_trajectory(self, action):
        """Validate a 1000-step mock-only timed trajectory."""
        if self.config.backend != "mock" or self.config.hardware_enabled is not False:
            raise ControlError("UNSUPPORTED_ACTION", "Timed trajectory execution is mock-only")
        if self.config.mode != "calibration" or not self.config.workspace_ready:
            raise ControlError("WORKSPACE_UNVERIFIED", "Mock trajectory requires a ready calibration workspace")
        if self._snapshot.moving is not False:
            raise ControlError("NOT_STATIONARY", "The arm must be stationary before accepting a trajectory")
        targets = action.get("targets_deg")
        period = action.get("period_seconds")
        if not isinstance(targets, list) or len(targets) != 1000 or type(period) not in (int, float) or period != .002:
            raise ControlError("INVALID_REQUEST", "Mock trajectory requires exactly 1000 targets at a 2 ms period")
        targets = [check_joints(row) for row in targets]
        previous = self._snapshot.joints_deg
        minimum_z = float('inf')
        for target in targets:
            if self._cancel.is_set():
                raise ControlError("STOP_REQUESTED", "Stop was requested while the trajectory was being checked")
            path = self.geometry.check_path(previous, target, self.limits)
            minimum_z = min(minimum_z, path['minimum_tcp_z_mm'])
            previous = target
        return {"kind": "mock_joint_trajectory", "targets_deg": targets, "period_seconds": .002,
                "requested_joints_deg": targets[-1], "effective_target_deg": targets[-1],
                "target_tcp_mm": self.geometry.forward(targets[-1]),
                "arrival_tolerance_mm": self.limits.tcp_tolerance_mm,
                "path": {"waypoints": len(targets), "minimum_tcp_z_mm": minimum_z,
                         "full_collision_check": False}, "path_constraints": {}}

    def _prepare_blended_path(self, action, waypoints, effective, speed, constraints):
        """Plan, subdivide, hash, and refresh-verify a blended multi-waypoint path."""
        blend_scope = self.blended_path_scope
        admission_snapshot = self._snapshot
        blend_waypoints = [list(check_joints(row)) for row in waypoints]
        blend_waypoints[-1] = list(effective)
        blend_budget = virtual_blend_time_budget(self.limits.motion_timeout, len(blend_waypoints))
        observed_start = list(check_joints(self._snapshot.joints_deg))
        reference = self.worker.call('blended_start_reference')
        expected_source = ('virtual_servo_target' if blend_scope == 'virtual_worker_only'
                           else 'sdk_stopped_encoder')
        if (not isinstance(reference, dict) or reference.get('available') is not True
                or reference.get('source') != expected_source):
            raise ControlError('BLENDED_REFERENCE_UNAVAILABLE', 'The path start reference is not known')
        control_start = check_joints(reference.get('joints_deg'))
        # The current encoder position and current servo command need not
        # coincide under load. Validate the complete bridge separately;
        # neither is treated as evidence about an object.
        bridge = self.geometry.check_path(observed_start, control_start, self.limits)
        bridge['independent_joint_envelope'] = self._check_motion_envelope(
            observed_start, control_start, constraints,
            start_tolerance_deg=self.limits.joint_tolerance_deg)
        original_count = len(blend_waypoints)
        durations = action.get('segment_durations_s')
        if not isinstance(durations, list) or len(durations) != original_count:
            raise ControlError('INVALID_REQUEST', 'A blended path needs one duration per waypoint')
        period = number(action.get('sample_period_seconds'), 'sample_period_seconds', minimum=.002, maximum=.2)
        durations = [number(value, 'segment_durations_s', minimum=period, maximum=self.limits.motion_timeout)
                     for value in durations]
        # Long path checking and hashing occur outside the state lock.
        # The admission reservation keeps all other actions out while
        # sensor publication and a stop remain available.
        lock_owned = self._condition._is_owned()
        if lock_owned:
            self._condition.release()
        try:
            def checked_section(left, right, is_first):
                if self._cancel.is_set():
                    raise ControlError('STOP_REQUESTED', 'Stop was requested during path checking')
                section = self.geometry.check_path(left, right, self.limits)
                start_tol = self.limits.joint_tolerance_deg if is_first else 0
                section['independent_joint_envelope'] = self._check_motion_envelope(
                    left, right, constraints, start_tolerance_deg=start_tol)
                return section

            def split_initial(left, right, *, is_first, depth):
                try:
                    return [(list(right), checked_section(left, right, is_first))]
                except ControlError as exc:
                    if (exc.code != 'ASYNC_ENVELOPE' or depth >= 3
                            or max(abs(a-b) for a, b in zip(left, right)) < 1e-6):
                        raise
                    midpoint = [(a+b)/2. for a, b in zip(left, right)]
                    return (split_initial(left, midpoint, is_first=is_first, depth=depth+1)
                            + split_initial(midpoint, right, is_first=False, depth=depth+1))

            pieces = split_initial(control_start, blend_waypoints[0], is_first=True, depth=0)
            if len(pieces) + original_count - 1 > MAX_BLENDED_WAYPOINTS:
                raise ControlError('INVALID_REQUEST', 'Initial subdivision exceeds the waypoint budget')
            point_groups = [[point for point, _ in pieces]] + [[point] for point in blend_waypoints[1:]]
            # At most eight time-driven midpoint insertions are attempted.
            # They keep all original waypoints and do not add budget.
            for split_count in range(9):
                effective_waypoints = [point for group in point_groups for point in group]
                if len(effective_waypoints) > MAX_BLENDED_WAYPOINTS:
                    raise ControlError('INVALID_REQUEST', 'Subdivision exceeds the waypoint budget')
                groups = [len(group) for group in point_groups]
                path_sections, all_durations = [], []
                previous = list(control_start)
                for group_index, group in enumerate(point_groups):
                    group_start = previous
                    group_end = group[-1]
                    group_size = max(abs(a-b) for a, b in zip(group_start, group_end))
                    if len(group) > 1 and group_size <= 0:
                        raise ControlError('INVALID_BLENDED_PATH', 'A zero-length segment cannot be subdivided')
                    for point in group:
                        path_sections.append(checked_section(previous, point, not path_sections))
                        part_size = max(abs(a-b) for a, b in zip(previous, point))
                        all_durations.append(durations[group_index] * (part_size / group_size if group_size else 1.))
                        previous = point
                blend_spec = {
                    'start_joints_deg': list(control_start),
                    'admission_observed_joints_deg': observed_start,
                    'segment_timeout_seconds': self.limits.motion_timeout,
                    'motion_timeout_seconds': blend_budget,
                    'original_waypoint_count': original_count,
                    'segment_groups': groups,
                    'waypoints_deg': effective_waypoints,
                    'segment_durations_s': all_durations,
                    'max_velocity_deg_s': action.get('max_velocity_deg_s'),
                    'max_acceleration_deg_s2': action.get('max_acceleration_deg_s2'),
                    'max_jerk_deg_s3': action.get('max_jerk_deg_s3'),
                    'sample_period_seconds': period,
                    'speed_pct': speed,
                    # The SDK must recheck each next target from the
                    # current encoder observation, not just from the
                    # previously planned target. Preserve exactly the
                    # same task constraints used by this admission.
                    'path_constraints': deepcopy(constraints),
                }
                try:
                    virtual_plan = prepare_virtual_blend(
                        blend_spec, motion_timeout=max(.001, blend_budget - self.limits.stable_seconds - .1))
                    break
                except ControlError as exc:
                    group_index = exc.details.get('group_index')
                    if (exc.code != 'MOTION_TIMEOUT' or type(group_index) is not int
                            or not 0 <= group_index < original_count or split_count == 8
                            or len(point_groups[group_index]) >= 8 or len(effective_waypoints) >= MAX_BLENDED_WAYPOINTS):
                        raise
                    group = point_groups[group_index]
                    start = control_start if group_index == 0 else point_groups[group_index-1][-1]
                    bounds = [start, *group]
                    sizes = [max(abs(a-b) for a, b in zip(left, right))
                             for left, right in zip(bounds, bounds[1:])]
                    index = max(range(len(sizes)), key=sizes.__getitem__)
                    if sizes[index] < 1e-6:
                        raise
                    midpoint = [(a+b)/2. for a, b in zip(bounds[index], bounds[index+1])]
                    group.insert(index, midpoint)
            digest = planned_sample_digest(virtual_plan)
            sdk_plan = None
            if blend_scope == 'sdk_host_stream_unverified':
                sdk_commands = sdk_blended_commands(virtual_plan)
                sdk_sha = sdk_blended_command_digest(sdk_commands)
                if tuple(sdk_commands[-1][1]) != tuple(effective):
                    raise ControlError('BLENDED_PLAN_MISMATCH', 'The last SDK target differs from the endpoint')
                # A joint box spanning a run of quantized SDK targets,
                # expanded by the existing tolerance on *both* ends,
                # encloses every independently progressing transition in
                # that run. If this stronger box fails, split down to the
                # unchanged individual transition check; never admit a
                # transition solely because a batch was too conservative.
                points = [tuple(control_start), *(check_joints(target) for _, target in sdk_commands)]
                minimum_z, box_checks = float('inf'), 0

                def check_sdk_box(first, last):
                    nonlocal minimum_z, box_checks
                    if self._cancel.is_set():
                        raise ControlError('STOP_REQUESTED', 'Stop was requested during SDK path checking')
                    box_checks += 1
                    left = [min(point[axis] for point in points[first:last+1]) for axis in range(JOINT_COUNT)]
                    right = [max(point[axis] for point in points[first:last+1]) for axis in range(JOINT_COUNT)]
                    try:
                        checked = self._check_motion_envelope(
                            left, right, constraints,
                            start_tolerance_deg=self.limits.joint_tolerance_deg)
                        minimum_z = min(minimum_z, checked['minimum_mm'][2])
                    except ControlError as exc:
                        if exc.code != 'ASYNC_ENVELOPE':
                            raise
                        if last - first == 1:
                            section = checked_section(points[first], points[last], first == 0)
                            minimum_z = min(minimum_z, section['minimum_tcp_z_mm'])
                        else:
                            middle = (first + last) // 2
                            check_sdk_box(first, middle)
                            check_sdk_box(middle, last)

                for first in range(0, len(sdk_commands), 32):
                    check_sdk_box(first, min(first + 32, len(sdk_commands)))
                sdk_plan = {'scope': 'sdk_host_stream_unverified',
                            'command_period_seconds': SDK_BLENDED_COMMAND_PERIOD_S,
                            'command_count': len(sdk_commands), 'planned_commands_sha256': sdk_sha,
                            'digest_encoding': 'be-u64-tick-and-5-f64-v1',
                            'minimum_tcp_z_mm': minimum_z, 'envelope_box_checks': box_checks,
                            'envelope_covers_all_command_transitions': True, 'full_collision_check': False}
                blend_spec['sdk_command_plan'] = sdk_plan
            if self._cancel.is_set():
                raise ControlError('STOP_REQUESTED', 'Stop was requested during path hashing')
            # Do not promote an old cached sample merely because planning
            # finished. Obtain a device snapshot outside the state lock,
            # retain its real sensor timestamp, and then publish it.
            direct_snapshot = Snapshot.from_dict(self.worker.call('snapshot'))
            if (not self._fresh(direct_snapshot)
                    or direct_snapshot.sensor_sequence < admission_snapshot.sensor_sequence):
                raise ControlError('SENSOR_STALE', 'Fresh sensor feedback is required after path hashing')
            self._publish(direct_snapshot)
        finally:
            if lock_owned:
                self._condition.acquire()
        if self._cancel.is_set():
            raise ControlError('STOP_REQUESTED', 'Stop was requested during path hashing')
        if self._fault is not None:
            raise self._fault
        latest = self._snapshot
        if not self._fresh(latest) or latest.moving is not False:
            raise ControlError('SENSOR_STALE', 'Fresh stationary feedback is required after path hashing')
        if max(abs(a-b) for a, b in zip(latest.joints_deg, admission_snapshot.joints_deg)) > self.limits.joint_tolerance_deg:
            raise ControlError('STATE_CHANGED', 'The observed joints changed while checking the path')
        self._validate_motion_feedback({'path_constraints': constraints}, admission_snapshot, latest)
        path = {'segments': path_sections, 'duration_s': virtual_plan.duration_s,
                'time_scale': virtual_plan.scale, 'sample_count': digest['sample_count'],
                'planned_sha256': digest['sha256'], 'digest_encoding': digest['encoding'],
                'original_waypoint_count': original_count, 'segment_groups': groups,
                'observation_to_control_start': bridge,
                'full_collision_check': False}
        if sdk_plan is not None:
            path['sdk_command_plan'] = sdk_plan
        return path, blend_spec, blend_scope, reference, observed_start, control_start

    def _prepare_nominal_arrival(self, action, kind, steps, target, effective, path, constraints,
                                 blended, direct_student_target, completion_mode, bounded_correction,
                                 correction_parent):
        """Validate nominal_arrival_joints_deg and return its checked arrival descriptor."""
        if kind not in ('move_joints', 'blended_joint_path') or 'nominal_arrival_joints_deg' not in action:
            return None
        nominal_requested = check_joints(action['nominal_arrival_joints_deg'])
        nominal_steps, nominal_effective = quantize_joints(nominal_requested)
        if (completion_mode == 'retain_position_target' and tuple(steps) != tuple(nominal_steps)
                and correction_parent is None):
            raise ControlError('INVALID_REQUEST', 'A retained non-correction move must send its nominal target')
        if (bounded_correction or blended) and tuple(steps) != tuple(nominal_steps):
            raise ControlError('INVALID_REQUEST', 'The first correction-enabled move must send its nominal target')
        raw_trim = [a - b for a, b in zip(target, nominal_requested)]
        quantized_trim = [a - b for a, b in zip(effective, nominal_effective)]
        if (max(map(abs, raw_trim)) > MAX_NOMINAL_TRIM_DEG
                or max(map(abs, quantized_trim)) > MAX_NOMINAL_TRIM_DEG):
            raise ControlError('NOMINAL_TRIM_LIMIT', 'Nominal and sent targets differ by more than one degree')
        nominal_path = (path if (blended or direct_student_target)
                        else self.geometry.check_path(self._snapshot.joints_deg, nominal_effective, self.limits))
        if not blended and not direct_student_target and (self.config.backend == 'sdk' or self.config.workspace_source == 'fixed_layout_photo'):
            nominal_path['independent_joint_envelope'] = self._check_motion_envelope(
                self._snapshot.joints_deg, nominal_effective, constraints)
        return {'requested_joints_deg': nominal_requested, 'effective_target_deg': nominal_effective,
                'steps': nominal_steps, 'target_tcp_mm': self.geometry.forward(nominal_effective),
                'quantized_trim_deg': quantized_trim, 'maximum_trim_deg': MAX_NOMINAL_TRIM_DEG,
                'path': nominal_path}

    def _prepare(self, action):
        keys(action, allowed={"kind", "joints_deg", "nominal_arrival_joints_deg", "allow_mock_nominal_retry",
                              "allow_nominal_correction", "correction_of_command_id", "speed_pct", "delta_xy_mm",
                              "reference_id", "state", "allow_visual_settle", "joint", "delta_deg", "delta_mm",
                              "path_constraints", "arrival_tolerance_mm", "targets_deg", "period_seconds",
                              "waypoints_deg", "segment_durations_s", "max_velocity_deg_s",
                              "max_acceleration_deg_s2", "max_jerk_deg_s3", "sample_period_seconds",
                              "completion_mode", "retain_position_from_command_id",
                              "allow_damped_compensation", "allow_target_follow_up",
                              "direct_student_target"}, required={"kind"})
        kind = action["kind"]
        if not isinstance(kind, str) or kind not in ACTION_FIELDS:
            raise ControlError("UNSUPPORTED_ACTION", "This phase implements calibration, mock XY correction, gripper and reobserve only")
        keys(action, allowed=ACTION_FIELDS[kind])
        completion_mode, damped_opt_in, follow_up, direct_student_target, correction_parent, pending = (
            self._validate_retained_and_correction_flags(action, kind))
        if kind == "reobserve":
            return {"kind": kind}
        if self._snapshot.firmware_warning_raw == 1:
            raise ControlError('DEVICE_WARNING', 'Device warning blocks new arm and gripper commands')
        if kind == "set_gripper":
            if self.config.mode != "calibration":
                raise ControlError("MODE_RESTRICTED", "Raw gripper control is restricted to calibration")
            if not isinstance(action.get("state"), str) or action.get("state") not in {"open", "closed"}:
                raise ControlError("INVALID_GRIPPER", "Expected open or closed")
            if self._snapshot.moving is not False:
                raise ControlError("NOT_STATIONARY", "Stop the arm before testing the gripper")
            result = {"kind": kind, "state": action["state"]}
            if 'retain_position_from_command_id' in action:
                identifier(action['retain_position_from_command_id'], 'retain_position_from_command_id')
                if self.config.backend != 'sdk' or self.config.hardware_enabled is not True:
                    raise ControlError('UNSUPPORTED_ACTION', 'Retained gripper position requires the SDK worker')
                self._require_retained_position_capability()
                result['retain_position_from_command_id'] = action['retain_position_from_command_id']
            return result
        if kind in {"jog_joint", "jog_tcp"}:
            if self.config.mode != "calibration":
                raise ControlError("MODE_RESTRICTED", "Incremental jogging is available in calibration mode only")
            if self._snapshot.moving is not False:
                raise ControlError("NOT_STATIONARY", "The arm must be stationary before a calibration step")
            return prepare_step(self.geometry, self._snapshot.joints_deg, action, self.limits)
        if kind == "mock_joint_trajectory":
            return self._prepare_mock_joint_trajectory(action)
        if not self.config.workspace_ready:
            raise ControlError("WORKSPACE_UNVERIFIED", "Provide measured coordinates or an explicit fixed-layout trial reference before absolute moves", workspace=self.config.workspace_status())
        speed = number(action.get("speed_pct", 12), "speed_pct", minimum=1, maximum=self.limits.max_speed_pct)
        arrival_tolerance = number(action.get("arrival_tolerance_mm", self.limits.tcp_tolerance_mm),
                                   "arrival_tolerance_mm", minimum=.01,
                                   maximum=max(3.0, self.limits.tcp_tolerance_mm))
        constraints = self._parse_path_constraints(action)
        if self._snapshot.moving is not False:
            raise ControlError("NOT_STATIONARY", "The arm must be stationary before accepting another target")
        blended = kind == 'blended_joint_path'
        mock_retry = bounded_correction = False
        waypoints = None
        if kind in ("move_joints", "blended_joint_path"):
            if self.config.mode != "calibration":
                raise ControlError("MODE_RESTRICTED", "Joint commands are not exposed in collection mode")
            if blended:
                if self.blended_path_scope == 'unavailable':
                    raise ControlError('UNSUPPORTED_ACTION', 'This worker does not support a blended path')
                waypoints = action.get('waypoints_deg')
                if not isinstance(waypoints, list) or not waypoints:
                    raise ControlError('INVALID_REQUEST', 'At least one waypoint is required')
                target = check_joints(waypoints[-1])
            else:
                target = check_joints(action.get("joints_deg"))
            allow_coarse = action.get("allow_visual_settle", False)
            if not isinstance(allow_coarse, bool):
                raise ControlError("INVALID_REQUEST", "allow_visual_settle must be boolean")
            if allow_coarse and self.config.backend != "mock":
                raise ControlError("PERCEPTION_UNAVAILABLE", "Real coarse-to-vision handoff requires the later perception integration")
            if 'nominal_arrival_joints_deg' in action and allow_coarse:
                raise ControlError('INVALID_REQUEST', 'Nominal arrival cannot use coarse visual settling')
            mock_retry = action.get('allow_mock_nominal_retry', False)
            if type(mock_retry) is not bool:
                raise ControlError('INVALID_REQUEST', 'Mock nominal retry must be boolean')
            if mock_retry and (self.config.backend != 'mock' or self.config.hardware_enabled is not False
                    or self.config.port is not None or 'nominal_arrival_joints_deg' not in action):
                raise ControlError('UNSUPPORTED_ACTION', 'Nominal retry is an explicit mock-only experiment')
            bounded_correction = action.get('allow_nominal_correction', False)
            if type(bounded_correction) is not bool:
                raise ControlError('INVALID_REQUEST', 'allow_nominal_correction must be boolean')
            if bounded_correction and ('nominal_arrival_joints_deg' not in action or mock_retry
                                       or correction_parent is not None):
                raise ControlError('INVALID_REQUEST', 'Bounded correction requires one initial nominal move')
        else:
            if self.config.backend != "mock":
                raise ControlError("PERCEPTION_UNAVAILABLE", "Phase-one XY corrections use simulated robot-state references only")
            if action.get("reference_id") != "tcp":
                raise ControlError("REFERENCE_UNAVAILABLE", "Phase one provides only the current TCP reference")
            delta = vector(action.get("delta_xy_mm"), 2, "delta_xy_mm")
            if distance(delta, (0, 0)) > self.limits.max_xy_correction_mm:
                raise ControlError("CORRECTION_LIMIT", "The XY correction is larger than 5 mm")
            if self._corrections >= self.limits.max_corrections:
                raise ControlError("CORRECTION_BUDGET", "The episode's correction budget is exhausted")
            tcp = self.geometry.forward(self._snapshot.joints_deg)
            target = self.geometry.inverse((tcp[0] + delta[0], tcp[1] + delta[1], tcp[2]), pitch_deg=0)
            allow_coarse = False
        steps, effective = quantize_joints(target)
        blend_spec = blend_scope = reference = observed_start = control_start = None
        if blended:
            path, blend_spec, blend_scope, reference, observed_start, control_start = self._prepare_blended_path(
                action, waypoints, effective, speed, constraints)
        elif direct_student_target:
            # The SDK's independent axis motion is not the sampled joint
            # interpolation or its hypothetical independent-axis envelope.
            # Preserve the existing workspace bound on the destination alone;
            # this does not establish a safe path or detect object contact.
            path = self.geometry.check_path(effective, effective, self.limits)
            path.update(source='direct_student_sdk_target', path_preflight_checked=False,
                        workspace_endpoint_checked=True)
        else:
            path = self.geometry.check_path(self._snapshot.joints_deg, effective, self.limits)
        if not blended and not direct_student_target and (self.config.backend == "sdk" or self.config.workspace_source == "fixed_layout_photo"):
            path["independent_joint_envelope"] = self._check_motion_envelope(
                self._snapshot.joints_deg, effective, constraints)
        nominal = self._prepare_nominal_arrival(
            action, kind, steps, target, effective, path, constraints, blended,
            direct_student_target, completion_mode, bounded_correction, correction_parent)
        if kind == "correct_xy":
            start_tcp, end_tcp = self.geometry.forward(self._snapshot.joints_deg), self.geometry.forward(effective)
            if distance(start_tcp[:2], end_tcp[:2]) > self.limits.max_xy_correction_mm or abs(start_tcp[2] - end_tcp[2]) > self.limits.tcp_tolerance_mm:
                raise ControlError("QUANTIZED_CORRECTION_LIMIT", "Quantization would exceed the correction limits")
        result = {"kind": kind, "requested_joints_deg": target, "effective_target_deg": effective, "steps": steps, "target_tcp_mm": self.geometry.forward(effective), "speed_pct": speed, "allow_visual_settle": allow_coarse, "path": path, "path_constraints": constraints, "arrival_tolerance_mm": arrival_tolerance}
        if direct_student_target:
            result['direct_student_target'] = True
        if completion_mode is not None:
            result['completion_mode'] = completion_mode
        if damped_opt_in:
            result['allow_damped_compensation'] = True
        if follow_up:
            result['allow_target_follow_up'] = True
        if blended:
            # A finished SDK axis may have a stable residual that the existing
            # nominal correction contract can evaluate at the final stop.
            # It is not an arrival or a correction command at this stage.
            blend_spec['allow_nominal_correction'] = bool(bounded_correction and nominal is not None)
            result['blended_spec'] = blend_spec
            result['blended_scope'] = blend_scope
            result['blended_start_reference'] = {'source': reference['source'],
                                                 'observed_joints_deg': observed_start,
                                                 'control_target_joints_deg': list(control_start)}
        if nominal is not None:
            result['nominal_arrival'] = nominal
            if mock_retry:
                result['allow_mock_nominal_retry'] = True
            if bounded_correction:
                result['allow_nominal_correction'] = True
                result['correction_arrival_tolerance_explicit'] = 'arrival_tolerance_mm' in action
            if correction_parent is not None:
                result['correction_of_command_id'] = correction_parent
                if pending['basis'].get('kind') == 'damped_nominal_half':
                    result['damped_compensation'] = deepcopy(pending['basis'])
        return result

    def _execute_motion(self, action, before):
        with span('motion_dispatch_feedback_and_stop'):
            if action['kind'] == 'blended_joint_path':
                try:
                    result = self._execute_motion_impl(action, before)
                except Exception:
                    self._stop_device()
                    # An unavailable trace is not evidence that no sample was
                    # transmitted. Preserve it without hiding the first error.
                    try:
                        self._record_blended_trace(action, require_complete=False)
                    except Exception:
                        pass
                    raise
                if result.get('applied') is not False:
                    trace = self._record_blended_trace(action, require_complete=result.get('status') in {
                        'completed', 'settled_needs_nominal_correction', 'settled_needs_nominal_retry',
                        'settled_needs_target_follow_up', 'settled'})
                    result['blended_path'] = trace
                return result
            if action["kind"] != "mock_joint_trajectory":
                return self._execute_motion_impl(action,before)
            try:
                result = self._execute_motion_impl(action,before)
            except Exception:
                self._stop_device()
                self._record_mock_trajectory(action)
                raise
            trace = self._record_mock_trajectory(action)
            samples = trace.get("samples", [])
            if result.get("status") == "completed":
                if (trace.get("completed") is not True or len(samples) != len(action["targets_deg"])
                        or any(row.get("tick") != i or row.get("joints_deg") != list(target)
                               for i, (row, target) in enumerate(zip(samples, action["targets_deg"]), 1))):
                    raise ControlError("MOCK_TRAJECTORY_UNCONFIRMED", "Mock did not apply every requested target in order")
                times = [row.get('applied_monotonic') for row in samples]
                if (not all(type(value) in (int, float) and value >= trace['started_monotonic'] for value in times)
                        or any(left > right for left, right in zip(times, times[1:]))
                        or any(abs(row.get('scheduled_monotonic', 0.) -
                                   (trace['started_monotonic'] + i * action['period_seconds'])) > 1e-7
                               for i, row in enumerate(samples, 1))):
                    raise ControlError("MOCK_TRAJECTORY_UNCONFIRMED", "Mock trajectory timestamps are invalid")
            times = [row['applied_monotonic'] for row in samples]
            intervals = [right - left for left, right in zip(times, times[1:])]
            result["mock_trajectory"] = {"applied_tick_count": len(trace.get("samples", [])),
                                         "completed": trace.get("completed") is True,
                                         "period_seconds": action["period_seconds"],
                                         "timing": "best_effort_wall_clock",
                                         "trace_event": "mock_trajectory_trace",
                                         "started_monotonic": trace.get("started_monotonic"),
                                         "first_applied_monotonic": times[0] if times else None,
                                         "last_applied_monotonic": times[-1] if times else None,
                                         "min_interval_seconds": min(intervals) if intervals else None,
                                         "max_interval_seconds": max(intervals) if intervals else None,
                                         "intervals_below_half_period": sum(
                                             value < action['period_seconds'] / 2. for value in intervals),
                                         "max_lateness_seconds": max(
                                             (row['applied_monotonic'] - row['scheduled_monotonic']
                                              for row in samples), default=None)}
            return result

    def _record_mock_trajectory(self, action):
        trace = self.worker.call("trajectory_trace", {"command_id": self._active})
        self.journal.append("mock_trajectory_trace", command_id=self._active, trace=trace)
        return trace

    def _record_blended_trace(self, action, *, require_complete):
        trace = self.worker.call('blended_path_trace', {'command_id': self._active})
        plan = action['path']
        if (not isinstance(trace, dict) or trace.get('available') is not True
                or trace.get('command_id') != self._active
                or trace.get('planned_sha256') != plan['planned_sha256']
                or trace.get('scheduled_count') != plan['sample_count']
                or trace.get('encoding') != plan['digest_encoding']):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The worker did not confirm the admitted path')
        if action.get('blended_scope') == 'sdk_host_stream_unverified':
            return self._record_sdk_blended_trace(action, trace, require_complete=require_complete)
        if (type(trace.get('applied_count')) is not int
                or not 0 <= trace['applied_count'] <= plan['sample_count']):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The virtual worker did not confirm the admitted path')
        # Check the actual applied prefix against the same planned samples.
        # Completion is not inferred from a worker receipt or a final encoder.
        prefix = hashlib.sha256()
        last_sample = None
        reconstructed = prepare_virtual_blend(action['blended_spec'],
                                              motion_timeout=action['blended_spec']['motion_timeout_seconds'])
        for tick, sample in enumerate(reconstructed.sample_ticks()):
            if not tick:
                continue
            if tick > trace['applied_count']:
                break
            update_sample_digest(prefix, tick, sample.time_s, sample.position_deg)
            last_sample = sample
        if trace.get('applied_sha256') != prefix.hexdigest():
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The applied virtual sample prefix did not match the plan')
        if (trace.get('last_applied_joints_deg') != (list(last_sample.position_deg) if last_sample else None)
                or trace.get('last_applied_time_s') != (last_sample.time_s if last_sample else None)):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The last applied virtual target did not match the prefix')
        if require_complete and (trace.get('completed') is not True
                or trace['applied_count'] != plan['sample_count']
                or trace.get('intermediate_stop_count') != 0
                or trace.get('stop_requested_during_path') is not False
                or trace.get('plan_verified') is not True or trace.get('worker_error') is not None):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The complete path or continuous execution was not confirmed')
        self.journal.append('blended_path_trace', command_id=self._active, trace=trace)
        return trace

    def _record_sdk_blended_trace(self, action, trace, *, require_complete):
        expected = action['path']['sdk_command_plan']
        submitted, encoded = trace.get('sdk_submitted_count'), trace.get('sdk_encoded_count')
        if (trace.get('scope') != 'sdk_host_stream_unverified'
                or trace.get('sdk_planned_sha256') != expected['planned_commands_sha256']
                or trace.get('sdk_planned_count') != expected['command_count']
                or trace.get('sdk_command_period_seconds') != expected['command_period_seconds']
                or type(submitted) is not int or type(encoded) is not int
                or not 0 <= encoded <= submitted <= expected['command_count']
                or submitted - encoded > 1 or trace.get('last_applied_joints_deg') is not None):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The SDK command evidence is invalid')
        path = prepare_virtual_blend(action['blended_spec'], motion_timeout=action['blended_spec']['motion_timeout_seconds'])
        commands = sdk_blended_commands(path)
        if len(commands) != expected['command_count'] or sdk_blended_command_digest(commands) != expected['planned_commands_sha256']:
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The SDK command plan changed')
        empty_digest = hashlib.sha256().hexdigest()
        for name, count in (('submitted', submitted), ('encoded', encoded)):
            prefix = commands[:count]
            expected_digest = sdk_blended_command_digest(prefix) if prefix else empty_digest
            if (trace.get(f'sdk_{name}_sha256') != expected_digest
                    or trace.get(f'last_sdk_{name}_target_deg') != (list(prefix[-1][1]) if prefix else None)
                    or trace.get(f'last_sdk_{name}_time_s') != (prefix[-1][0] if prefix else None)):
                raise ControlError('BLENDED_TRACE_UNCONFIRMED', f'The SDK {name} prefix differs from the checked plan')
        if require_complete and (trace.get('completed') is not True or not trace.get('plan_verified')
                or trace.get('worker_error') is not None or trace.get('stop_requested_during_path') is not False
                or trace.get('intermediate_stop_count') != 0
                or submitted != len(commands) or encoded != len(commands)):
            raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The SDK target sequence was not completely encoded')
        self.journal.append('blended_path_trace', command_id=self._active, trace=trace)
        return trace

    @contextmanager
    def _action_send_guard(self):
        # Ordered with stop intent after the worker RPC lock is acquired. Do
        # not hold this across RPC response waits or device execution.
        with self._stop_intent_lock:
            if self._fault:
                raise self._fault
            if self._cancel.is_set():
                raise _DispatchCancelled(self._cancel_reason)
            yield

    def _validate_motion_feedback(self, action, before, snapshot):
        """Check the same observed limits during motion and after its stop."""
        if snapshot.collision_event_counts is not None and before.collision_event_counts is not None:
            changed = [i + 1 for i, (old, new) in enumerate(
                zip(before.collision_event_counts, snapshot.collision_event_counts)) if new > old]
            if changed:
                raise ControlError("DEVICE_COLLISION_EVENT", "The SDK reported a new joint collision event during motion",
                                   joints=changed, counts=snapshot.collision_event_counts,
                                   physical_contact_cause_confirmed=False)
        if snapshot.firmware_warning_raw == 1:
            raise ControlError("DEVICE_WARNING", "The device warning bit became active during motion")
        check_joints(snapshot.joints_deg)
        observed_tcp = self.geometry.forward(snapshot.joints_deg)
        if self.limits.workspace_configured and any(
                p < lo or p > hi for p, lo, hi in
                zip(observed_tcp, self.limits.workspace_min_mm, self.limits.workspace_max_mm)):
            raise ControlError("PATH_OUTSIDE_WORKSPACE", "Observed nominal TCP left the configured workspace", tcp_mm=observed_tcp)
        constraints = action.get("path_constraints", {})
        if constraints.get("minimum_z_mm") is not None and observed_tcp[2] < constraints["minimum_z_mm"]:
            raise ControlError("TRANSFER_CLEARANCE", "Observed nominal TCP fell below the transport height", tcp_mm=observed_tcp)
        if constraints.get("maximum_z_mm") is not None and observed_tcp[2] > constraints["maximum_z_mm"]:
            raise ControlError("PATH_CEILING", "Observed nominal TCP rose above this path's height ceiling", tcp_mm=observed_tcp)
        if (constraints.get("vertical_xy_mm") is not None
                and distance(observed_tcp[:2], constraints["vertical_xy_mm"]) > constraints["corridor_mm"]):
            raise ControlError("VERTICAL_CORRIDOR", "Observed nominal TCP left the vertical corridor", tcp_mm=observed_tcp)
        return observed_tcp

    def _make_nominal_correction(self, action, settled, *, original_host_frame=None, latest_host_frame=None,
                                 parent_command_id=None):
        """Build one candidate from stopped encoders; this does not dispatch it."""
        nominal = action['nominal_arrival']['effective_target_deg']
        sent = action['effective_target_deg']
        damped = action.get('allow_damped_compensation') is True
        if damped:
            if not self._position_frame_matches(original_host_frame, action['steps'], action['speed_pct']):
                raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The original motion target was not host-encoded')
            if not self._position_frame_matches(latest_host_frame, action['steps'], action['speed_pct'],
                    command_sequence=original_host_frame['command_sequence'],
                    target_segment_start_sequence=original_host_frame['target_segment_start_sequence'],
                    after_sequence=original_host_frame['frame_sequence']-1):
                raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The original motion target was not retained')
        gain = .5 if damped else 1.
        requested = check_joints([target + gain * (desired - observed) for target, desired, observed
                                  in zip(sent, nominal, settled.joints_deg)])
        steps, effective = quantize_joints(requested)
        if tuple(steps) == tuple(action['steps']):
            raise ControlError('NOMINAL_CORRECTION_UNAVAILABLE', 'The bounded correction would not change the sent target')
        candidate = {'kind': 'move_joints', 'joints_deg': list(requested),
                     'nominal_arrival_joints_deg': list(nominal), 'speed_pct': action['speed_pct'],
                     'path_constraints': deepcopy(action['path_constraints'])}
        if action.get('correction_arrival_tolerance_explicit') is True:
            candidate['arrival_tolerance_mm'] = action['arrival_tolerance_mm']
        # Preflight at the current observation and repeat it at actual
        # admission. Both nominal and transmitted paths must pass unchanged.
        with self._condition:
            self._prepare(candidate)
        parent_command_id = parent_command_id or self._active
        candidate['correction_of_command_id'] = parent_command_id
        candidate = json.loads(json.dumps(candidate, allow_nan=False))
        basis = {'sensor_sequence': settled.sensor_sequence,
                 'sensor_monotonic': settled.sensor_time,
                 'joints_deg': list(settled.joints_deg),
                 'source': 'retained_position_encoder_feedback' if damped else 'stop_confirmed_encoder_feedback'}
        if damped:
            basis.update(kind='damped_nominal_half', gain=gain, parent_command_id=parent_command_id,
                         stop_epoch=self._stop_epoch,
                         nominal_target_deg=list(nominal), nominal_steps=list(action['nominal_arrival']['steps']),
                         compensated_target_deg=list(effective), compensated_steps=list(steps),
                         unquantized_target_deg=list(requested), original_host_frame=deepcopy(original_host_frame),
                         latest_host_frame=deepcopy(latest_host_frame),
                         baseline_joint_error_deg=max(abs(a-b) for a,b in zip(settled.joints_deg, nominal)),
                         baseline_tcp_error_mm=distance(self.geometry.forward(settled.joints_deg),
                                                        action['nominal_arrival']['target_tcp_mm']),
                         minimum_z_mm=action['path_constraints'].get('minimum_z_mm'))
        return candidate, basis

    def _reuse_retained_position(self, action, before, effective, target_tcp, arrival_joints, arrival_tcp, arrival_tolerance):
        """Confirm unchanged retained SDK position target against a fresh stationary sample."""
        deadline = time.monotonic() + self.limits.stop_timeout
        with self._condition:
            while (not self._cancel.is_set() and time.monotonic() < deadline
                   and self._snapshot.sensor_sequence <= before.sensor_sequence):
                self._condition.wait(min(self.limits.sample_period, max(0., deadline - time.monotonic())))
            settled = self._snapshot
        if self._cancel.is_set():
            return {'status': 'cancelled', 'applied': False, 'reason': self._cancel_reason}
        if settled.sensor_sequence <= before.sensor_sequence:
            raise ControlError('SENSOR_STALE', 'A new encoder sample is required to reuse a position target')
        retained, _ = self._checked_retained_position(self._retained_position_target['command_id'], settled,
                                                      require_arrival=False)
        self._validate_motion_feedback(action, before, settled)
        nominal_error, nominal_tcp_error, nominal_reached = self._target_error_metrics(
            settled.joints_deg, arrival_joints, arrival_tcp, arrival_tolerance)
        sent_error, sent_tcp_error, sent_reached = self._target_error_metrics(
            settled.joints_deg, effective, target_tcp, arrival_tolerance)
        return {'status': 'position_target_unchanged', 'applied': False, 'reason': 'retained_target_unchanged',
                'target': action, 'arrival_reference': 'nominal', 'nominal_arrival_reached': nominal_reached,
                'sent_target_reached': sent_reached, 'arrival_joint_error_deg': nominal_error,
                'arrival_tcp_error_mm': nominal_tcp_error, 'joint_error_deg': sent_error,
                'tcp_error_mm': sent_tcp_error,
                'retained_position_from_command_id': retained['command_id'],
                'retained_position_observation': {'sensor_sequence': settled.sensor_sequence,
                                                  'sensor_monotonic': settled.sensor_time}}

    def _verify_post_stop_arrival(self, action, before, effective, target_tcp, arrival_joints, arrival_tcp,
                                  arrival_tolerance, nominal, sdk_motion, status, pre_stop_joint_error,
                                  pre_stop_tcp_error, retained_host_frame):
        """Stop the device after motion settles and verify arrival residuals against post-stop feedback."""
        stop = self._stop_device()
        if not stop["confirmed"]:
            raise ControlError("STOP_UNCONFIRMED", "A fresh stationary state was not confirmed after stop")
        # Completion and returned residuals must refer to the
        # fresh stationary observation bound to this stop, not
        # the earlier in-flight sample. The device's physical
        # behaviour after its stop command is not assumed here.
        with self._condition:
            settled = self._snapshot
            fault = self._fault
        if fault is not None:
            raise fault
        if (not self._fresh(settled) or settled.moving is not False
                or settled.sensor_sequence < stop['confirmed_sensor_sequence']
                or settled.sensor_time < stop['confirmed_sensor_monotonic']):
            raise ControlError('STOP_UNCONFIRMED', 'Fresh stationary feedback after stop is required')
        self._validate_motion_feedback(action, before, settled)
        joint_error, tcp_error, sent_reached = self._target_error_metrics(
            settled.joints_deg, effective, target_tcp, arrival_tolerance)
        arrival_joint_error, arrival_tcp_error, nominal_reached = self._target_error_metrics(
            settled.joints_deg, arrival_joints, arrival_tcp, arrival_tolerance)
        if sdk_motion and 'damped_compensation' not in action:
            status = 'completed' if nominal_reached and sent_reached else 'settled'
        elif nominal is not None:
            retryable = (status == 'settled_needs_nominal_retry'
                         and arrival_joint_error <= MAX_NOMINAL_TRIM_DEG
                         and arrival_tcp_error <= MAX_MOCK_NOMINAL_RETRY_TCP_MM)
            correction_eligible = (action.get('allow_nominal_correction') is True
                                   and arrival_joint_error <= MAX_NOMINAL_TRIM_DEG
                                   and arrival_tcp_error <= MAX_MOCK_NOMINAL_RETRY_TCP_MM)
            if nominal_reached:
                status = 'completed'
            elif correction_eligible:
                self._require_quiet_motion_diagnostics(before.firmware_warning_raw,
                                                       before.collision_event_counts, settled)
                correction_action, correction_basis = self._make_nominal_correction(
                    action, settled, original_host_frame=retained_host_frame)
                status = 'settled_needs_nominal_correction'
            elif not retryable:
                raise ControlError('TARGET_NOT_REACHED', 'The nominal waypoint changed after stop',
                                   phase='after_stop', arrival_reference='nominal',
                                   joint_error_deg=joint_error, tcp_error_mm=tcp_error,
                                   arrival_joint_error_deg=arrival_joint_error,
                                   arrival_tcp_error_mm=arrival_tcp_error)
        else:
            coarse_allowed = (action.get('allow_visual_settle', False)
                              and tcp_error <= self.limits.coarse_residual_mm)
            if status == 'completed' and not sent_reached and coarse_allowed:
                status = 'settled_needs_observation'
            elif (status == 'completed' and not sent_reached
                  or status == 'settled_needs_observation' and not coarse_allowed):
                raise ControlError('TARGET_NOT_REACHED', 'The target waypoint changed after stop',
                                   phase='after_stop', joint_error_deg=joint_error,
                                   tcp_error_mm=tcp_error,
                                   pre_stop_joint_error_deg=pre_stop_joint_error,
                                   pre_stop_tcp_error_mm=pre_stop_tcp_error,
                                   confirmed_sensor_sequence=stop['confirmed_sensor_sequence'],
                                   confirmed_sensor_monotonic=stop['confirmed_sensor_monotonic'])
        result = {"status": status, "applied": True, "target": action, "receipt": None,
                  "joint_error_deg": joint_error, "tcp_error_mm": tcp_error, "stop": stop}
        if sdk_motion:
            result['sent_target_reached'] = sent_reached
        if nominal is not None:
            result.update(arrival_reference='nominal',
                          nominal_arrival_reached=nominal_reached,
                          arrival_joint_error_deg=arrival_joint_error, arrival_tcp_error_mm=arrival_tcp_error,
                          sent_target_reached=sent_reached,
                          sent_target_source='worker_dispatch_receipt')
            if status == 'settled_needs_nominal_retry':
                result['mock_retry_permitted'] = True
            if status == 'settled_needs_nominal_correction':
                result.update(nominal_correction_permitted=True,
                              correction_action=correction_action,
                              correction_basis=correction_basis)
        return result

    def _execute_motion_impl(self, action, before):
        effective = action["effective_target_deg"]
        target_tcp = action["target_tcp_mm"]
        nominal = action.get('nominal_arrival')
        arrival_joints = nominal['effective_target_deg'] if nominal is not None else effective
        arrival_tcp = nominal['target_tcp_mm'] if nominal is not None else target_tcp
        arrival_tolerance = action.get("arrival_tolerance_mm", self.limits.tcp_tolerance_mm)
        initial_joint_error, initial_tcp_error, initial_nominal_reached = self._target_error_metrics(
            before.joints_deg, arrival_joints, arrival_tcp, arrival_tolerance)
        _, _, initial_sent_reached = self._target_error_metrics(
            before.joints_deg, effective, target_tcp, arrival_tolerance)
        retain = action.get('completion_mode') == 'retain_position_target'
        initially_at_target = initial_nominal_reached and initial_sent_reached
        previous_same_target = None
        if (retain and action.get('direct_student_target') is True
                and action.get('allow_target_follow_up') is True):
            prior = self._retained_position_target
            old_action = prior.get('action', {}) if isinstance(prior, dict) else {}
            old_nominal = old_action.get('nominal_arrival', {})
            if (old_action.get('direct_student_target') is not True
                    or tuple(old_action.get('steps', ())) != tuple(action['steps'])
                    or tuple(old_nominal.get('steps', ())) != tuple(nominal['steps'])
                    or old_action.get('speed_pct') != action['speed_pct']):
                raise ControlError('POSITION_TARGET_UNAVAILABLE',
                                   'A direct target follow-up requires the same retained student goal')
            prior, _ = self._checked_retained_position(prior['command_id'], before, require_arrival=False)
            prior_completion = prior['completion']
            basis = prior_completion.get('observation_basis')
            if basis in ('encoder_displacement_then_stationary', 'target_encoder_steps_observed',
                         'within_arrival_tolerance', 'same_target_prior_evidence_then_stationary'):
                earlier = prior_completion.get('previous_same_target_evidence') or {}
                previous_same_target = {
                    'command_id': prior['command_id'], 'observation_basis': basis,
                    'sensor_sequence': prior_completion['sensor_sequence'],
                    'initial_observation_basis': earlier.get('initial_observation_basis', basis)}
        if (retain and action['kind'] == 'move_joints'
                and action.get('allow_target_follow_up') is not True
                and self._retained_position_target is not None
                and tuple(self._retained_position_target['action']['steps']) == tuple(action['steps'])
                and tuple(self._retained_position_target['action']['nominal_arrival']['steps'])
                    == tuple(action['nominal_arrival']['steps'])
                and self._retained_position_target['action']['speed_pct'] == action['speed_pct']):
            # Reuse is not a new command. Bind it to a subsequent distinct
            # sensor sample and the still-current original host position frame.
            return self._reuse_retained_position(
                action, before, effective, target_tcp, arrival_joints, arrival_tcp, arrival_tolerance)
        if (not retain and action["kind"] not in ("mock_joint_trajectory", "blended_joint_path")
                and initial_nominal_reached):
            if nominal is not None:
                sent_joint_error, sent_tcp_error, sent_reached = self._target_error_metrics(
                    before.joints_deg, effective, target_tcp, arrival_tolerance)
                return {"status": "already_at_target", "applied": False,
                        "reason": "already_at_nominal_target", "target": action,
                        "arrival_reference": "nominal", "nominal_arrival_reached": True,
                        "arrival_joint_error_deg": initial_joint_error, "arrival_tcp_error_mm": initial_tcp_error,
                        "sent_target_reached": sent_reached,
                        "joint_error_deg": sent_joint_error, "tcp_error_mm": sent_tcp_error}
            same_steps = quantize_joints(before.joints_deg)[0] == action["steps"]
            return {"status": "already_at_target", "applied": False, "reason": "encoded_target_unchanged" if same_steps else "already_within_tolerance", "target": action}
        with self._dispatch_lock:
            if self._fault:
                raise self._fault
            if self._cancel.is_set():
                return {"status": "cancelled", "applied": False, "reason": self._cancel_reason}
            if 'damped_compensation' in action:
                with self._condition:
                    checked = self._snapshot
                parent_frame = self._require_damped_parent_frame(action['damped_compensation'], checked)
                self._require_quiet_motion_diagnostics(before.firmware_warning_raw, before.collision_event_counts, checked)
                self._check_damped_compensation_error(action, checked)
                self.geometry.check_path(checked.joints_deg, effective, self.limits)
                self._check_motion_envelope(checked.joints_deg, effective, action['path_constraints'])
            sent_at = time.monotonic()
            with self._condition:
                # A newly dispatched arm command invalidates the old target,
                # even if its result later proves uncertain.
                self._retained_position_target = None
            try:
                with span('device_dispatch_rpc'):
                    if action["kind"] == "blended_joint_path":
                        receipt = self.worker.call('move_blended_joint_path', {'command_id': self._active,
                            'spec': action['blended_spec'], 'expected_sha256': action['path']['planned_sha256'],
                            'expected_count': action['path']['sample_count']}, send_guard=self._action_send_guard)
                    elif action["kind"] == "mock_joint_trajectory":
                        receipt = self.worker.call("move_joint_trajectory", {"command_id": self._active,
                            "targets_deg": action["targets_deg"], "period_seconds": action["period_seconds"]},
                            send_guard=self._action_send_guard)
                    else:
                        arguments = {"joints_deg": action["requested_joints_deg"], "speed_pct": action["speed_pct"]}
                        if retain:
                            arguments['capture_host_frame'] = True
                        if 'damped_compensation' in action:
                            arguments['position_parent'] = {
                                'host_frame': parent_frame,
                                'basis': action['damped_compensation'],
                                'path_constraints': action['path_constraints'],
                                'arrival_tolerance_mm': arrival_tolerance,
                                'collision_event_counts': before.collision_event_counts,
                            }
                        receipt = self.worker.call("move_joints", arguments,
                            send_guard=self._action_send_guard)
            except _DispatchCancelled as exc:
                return {"status": "cancelled", "applied": False, "reason": str(exc)}
        if action["kind"] == "correct_xy":
            self._corrections += 1
        with span('journal_dispatch'):
            self.journal.append("command_dispatched", command_id=self._active, receipt=receipt, target=action)
        if action['kind'] == 'blended_joint_path':
            expected_simulated = action.get('blended_scope') != 'sdk_host_stream_unverified'
            if (receipt.get('command_id') != self._active or receipt.get('simulated') is not expected_simulated
                    or receipt.get('planned_sha256') != action['path']['planned_sha256']
                    or receipt.get('scheduled_count') != action['path']['sample_count']
                    or receipt.get('encoding') != action['path']['digest_encoding']):
                raise ControlError('BLENDED_RECEIPT_MISMATCH', 'The virtual worker did not accept the checked path')
        sdk_motion = self.config.backend == 'sdk'
        if sdk_motion or nominal is not None or action['kind'] == 'blended_joint_path':
            try:
                receipt_steps = receipt['steps']
                receipt_effective = check_joints(receipt['effective_target_deg'])
                if (not isinstance(receipt_steps, (list, tuple)) or len(receipt_steps) != JOINT_COUNT
                        or any(type(value) is not int for value in receipt_steps)
                        or tuple(receipt_steps) != tuple(action['steps'])
                        or any(abs(a - b) > 1e-7 for a, b in zip(receipt_effective, effective))):
                    raise ValueError('receipt target differs')
            except (KeyError, TypeError, ValueError, ControlError):
                raise ControlError('TARGET_RECEIPT_MISMATCH', 'Device receipt did not match the sent target') from None
        sdk_stream = action.get('blended_scope') == 'sdk_host_stream_unverified'
        independent_transport = sdk_motion and self._is_independent_single_transport(action)
        total_goal_steps = ((nominal['steps'] if nominal is not None else action['steps'])
                            if independent_transport else None)
        best_total_remaining = (self._remaining_joint_steps(total_goal_steps, before.joints_deg)
                                if independent_transport else None)
        deadline = (None if sdk_motion else time.monotonic() +
                    (action['blended_spec']['motion_timeout_seconds']
                     if action['kind'] == 'blended_joint_path' else self.limits.motion_timeout))
        seen_motion = False
        seen_encoder_step = False
        last_progress, progress_pose = sent_at, before.joints_deg
        previous_observed = before.joints_deg
        progress_best = list(before.joints_deg)
        progress_reference = list(before.joints_deg if sdk_stream else effective)
        progress_direction = ([0] * JOINT_COUNT if sdk_stream else
                              [(b > a) - (b < a) for a, b in zip(before.joints_deg, effective)])
        stable_since, stable_pose, stable_sequence, stable_in_bounds = None, None, None, None
        retained_host_frame, retained_latest_host_frame = None, None
        # A loaded MOVE in the student's simulator is observed at the end of
        # its interpolation. Keep the last actual encoder change for policy
        # inference. Completion and the next command still require rest.
        last_motion_feedback = None
        preceding_sample_joints = before.joints_deg
        last_sequence = before.sensor_sequence
        while deadline is None or time.monotonic() < deadline:
            if self._cancel.is_set():
                if self._fault:
                    raise self._fault
                stop = self._stop_device()
                if self._fault:
                    raise self._fault
                result = {"status": "cancelled", "applied": True, "reason": self._cancel_reason,
                          "target": action, "receipt": receipt, "stop": stop}
                if action.get('direct_student_target') is True:
                    # Stop first. Afterwards, recover only a matching host
                    # frame later than this submission boundary; a receipt by
                    # itself does not prove that the target was encoded.
                    frame = retained_host_frame
                    if frame is None:
                        try:
                            diagnostic = self.worker.call('diagnostics')
                            frames = diagnostic.get('recent_motor_frames', [])
                            after = receipt.get('encoding_after_frame_sequence')
                            submitted = receipt.get('submitted_monotonic')
                            if type(after) is int and type(submitted) in (int, float):
                                frame = next((item for item in reversed(frames)
                                    if self._position_frame_matches(item, action['steps'], action['speed_pct'],
                                                                    after_sequence=after)
                                    and item['monotonic'] >= submitted), None)
                        except Exception:
                            frame = None
                    result['interrupted_host_frame'] = deepcopy(frame)
                return result
            with self._condition:
                snapshot = self._snapshot
                if snapshot is None or snapshot.sensor_sequence == last_sequence or snapshot.sensor_time is None or snapshot.sensor_time < sent_at:
                    wait = self.limits.sample_period if deadline is None else min(
                        self.limits.sample_period, max(0, deadline - time.monotonic()))
                    self._condition.wait(wait)
                    continue
            if not self._fresh(snapshot):
                if self._sdk_feedback_gap_in_grace(snapshot):
                    # Do not accept stale observations or send another
                    # target; wait for the next fresh encoder sample.
                    with self._condition:
                        self._condition.wait(self.limits.sample_period)
                    continue
                raise ControlError("DISCONNECTED" if not snapshot.connected else "SENSOR_STALE", "Motion lost fresh feedback")
            self._validate_motion_feedback(action, before, snapshot)
            sdk_trace = None
            if sdk_stream:
                sdk_trace = self.worker.call('blended_path_trace', {'command_id': self._active})
                if not isinstance(sdk_trace, dict) or sdk_trace.get('available') is not True:
                    raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'SDK stream progress is unavailable')
                if sdk_trace.get('worker_error') is not None:
                    error = sdk_trace['worker_error']
                    raise ControlError(error.get('code', 'SDK_STREAM_FAILED'),
                                       error.get('message', 'The SDK stream failed'), **error.get('details', {}))
                if not sdk_trace.get('running') and sdk_trace.get('completed') is not True:
                    raise ControlError('SDK_STREAM_INCOMPLETE', 'The SDK stream stopped before all targets were encoded')
            if retain and (not sdk_stream or sdk_trace.get('completed') is True):
                frame = self.worker.call('position_target_frame')
                if retained_host_frame is None:
                    candidate = sdk_trace.get('last_sdk_encoded_frame') if sdk_stream else frame
                    after_frame = -1 if sdk_stream else receipt.get('encoding_after_frame_sequence')
                    submitted = sent_at if sdk_stream else receipt.get('submitted_monotonic')
                    if (type(after_frame) is not int or type(submitted) not in (int, float)):
                        raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The SDK submission has no host-frame boundary')
                    if (self._position_frame_matches(candidate, action['steps'], action['speed_pct'], after_sequence=after_frame)
                            and candidate['monotonic'] >= submitted):
                        retained_host_frame = dict(candidate)
                if retained_host_frame is not None:
                    if (not self._position_frame_matches(frame, action['steps'], action['speed_pct'],
                            command_sequence=retained_host_frame['command_sequence'],
                            target_segment_start_sequence=retained_host_frame['target_segment_start_sequence'],
                            after_sequence=retained_host_frame['frame_sequence']-1)
                            or not 0 <= time.monotonic()-frame['monotonic'] <= self.limits.sensor_max_age):
                        raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The SDK host position frame changed during arrival')
                    retained_latest_host_frame = dict(frame)
                    if ('damped_compensation' in action
                            and snapshot.sensor_time > retained_host_frame['monotonic']):
                        self._check_damped_compensation_error(action, snapshot, allow_transient_worsening=True)
            now = time.monotonic()
            if sdk_motion:
                reference = sdk_trace.get('last_sdk_encoded_target_deg') if sdk_stream else None
                if reference is not None:
                    reference = check_joints(reference)
                    for axis, (old, new) in enumerate(zip(progress_reference, reference)):
                        direction = (new > old) - (new < old)
                        if direction:
                            if direction != progress_direction[axis]:
                                progress_best[axis] = previous_observed[axis]
                            progress_direction[axis] = direction
                    progress_reference = list(reference)
                advanced = False
                if independent_transport:
                    remaining = self._remaining_joint_steps(total_goal_steps, snapshot.joints_deg)
                    if remaining < best_total_remaining:
                        best_total_remaining, advanced = remaining, True
                else:
                    for axis, actual in enumerate(snapshot.joints_deg):
                        direction = progress_direction[axis]
                        if direction and (actual - progress_best[axis]) * direction >= ENCODER_STEP_DEG / 2:
                            progress_best[axis] = actual
                            advanced = True
                previous_observed = snapshot.joints_deg
                if sdk_stream and sdk_trace.get('completed') is True:
                    # A host receipt does not prove radio acknowledgement;
                    # retain the independent check of the final encoded goal.
                    encoded_target = check_joints(sdk_trace.get('last_sdk_encoded_target_deg'))
                    if any(abs(a-b) > 1e-7 for a, b in zip(encoded_target, receipt_effective)):
                        raise ControlError('BLENDED_TRACE_UNCONFIRMED', 'The last encoded target differs from the admitted goal')
                if advanced:
                    with self._condition:
                        with self._stop_intent_lock:
                            if self._lease is not None and self._active is not None and not self._cancel.is_set():
                                self._lease['deadline'] = max(self._lease['deadline'], now + self.limits.lease_seconds)
            elif max(abs(a-b) for a,b in zip(snapshot.joints_deg,progress_pose))>=ENCODER_STEP_DEG / 2:
                last_progress, progress_pose = now, snapshot.joints_deg
            elif now-last_progress>STALL_PROGRESS_TIMEOUT_SECONDS:
                raise ControlError("MOTION_STALLED", "No encoder progress for two seconds during an active command")
            last_sequence = snapshot.sensor_sequence
            seen_motion = seen_motion or snapshot.moving is True
            direct_no_motion = False
            direct_displacement = None
            direct_exact_target = False
            direct_within_arrival = False
            if (action.get('direct_student_target') is True and retained_host_frame is not None
                    and snapshot.sensor_time > retained_host_frame['monotonic']):
                # The frozen loaded MOVE gives the model its next observation
                # after a two-second ramp. The SDK can finish a small MOVE or
                # same-target retry much earlier. Its transient motion at the
                # start is not the measured state at that policy boundary.
                if (self._gripper_command == 'closed' and snapshot.moving is True
                        and snapshot.sensor_time - retained_host_frame['monotonic'] >= 2.
                        and snapshot.joints_deg != preceding_sample_joints
                        and snapshot.gripper_reported_state == 'closed'
                        and snapshot.gripper_time is not None
                        and 0 <= now - snapshot.gripper_time <= self.limits.sensor_max_age
                        and snapshot.firmware_warning_raw == 0):
                    last_motion_feedback = snapshot
                seen_encoder_step = (seen_encoder_step or
                    tuple(quantize_joints(snapshot.joints_deg)[0]) != tuple(quantize_joints(before.joints_deg)[0]))
                direct_displacement = max(abs(a-b) for a, b in zip(snapshot.joints_deg, before.joints_deg))
                direct_exact_target = tuple(quantize_joints(snapshot.joints_deg)[0]) == tuple(action['steps'])
                direct_within_arrival = (max(abs(a-b) for a, b in zip(snapshot.joints_deg, arrival_joints))
                    <= self.limits.joint_tolerance_deg
                    and distance(self.geometry.forward(snapshot.joints_deg), arrival_tcp) <= arrival_tolerance)
                # Reuse the existing motion timeout. A fresh stationary arm
                # that has not measurably moved is not a completed motion;
                # it may request only the one existing same-target retry.
                direct_no_motion = (snapshot.moving is False and not direct_within_arrival
                    and direct_displacement <= self.limits.joint_tolerance_deg
                    and now - sent_at >= self.limits.motion_timeout)
            preceding_sample_joints = snapshot.joints_deg
            # A fresh stationary sample can still precede device motion. A
            # completion event or observed motion followed by rest is needed;
            # host stream completion alone only proves command submission.
            completed_signal = (snapshot.completion_sequence > before.completion_sequence
                                or (seen_motion and snapshot.moving is False))
            if action.get('direct_student_target') is True:
                # One encoder step can be noise. Require a displacement above
                # the existing joint tolerance or the existing arrival bounds; a
                # timeout with no displacement is explicitly not a completion.
                completed_signal = (snapshot.moving is False and direct_displacement is not None
                    and (direct_displacement > self.limits.joint_tolerance_deg
                         or direct_within_arrival or direct_no_motion or previous_same_target is not None))
            if sdk_stream:
                completed_signal = completed_signal and sdk_trace.get('completed') is True
            if retain:
                # An already-in-range target can establish position mode even
                # when there was no moving event. This never asserts movement.
                stationary_same_target = initially_at_target
                if action.get('direct_student_target') is True:
                    stationary_same_target = (initially_at_target
                        and tuple(quantize_joints(before.joints_deg)[0]) == tuple(action['steps']))
                completed_signal = (retained_host_frame is not None
                                    and snapshot.sensor_time > retained_host_frame['monotonic']
                                    and (completed_signal or (stationary_same_target
                                         and (not sdk_stream or sdk_trace.get('completed') is True))))
            joint_error, tcp_error, sent_reached = self._target_error_metrics(
                snapshot.joints_deg, effective, target_tcp, arrival_tolerance)
            arrival_joint_error, arrival_tcp_error, nominal_reached = self._target_error_metrics(
                snapshot.joints_deg, arrival_joints, arrival_tcp, arrival_tolerance)
            if completed_signal and snapshot.moving is False:
                if retain:
                    self._require_quiet_motion_diagnostics(before.firmware_warning_raw,
                                                           before.collision_event_counts, snapshot)
                    in_bounds = nominal_reached and ('damped_compensation' in action or sent_reached)
                    if stable_in_bounds is not None and in_bounds != stable_in_bounds:
                        stable_pose, stable_since, stable_sequence = None, None, None
                    stable_in_bounds = in_bounds
                changed_while_settling = (stable_pose is not None and (
                    tuple(quantize_joints(snapshot.joints_deg)[0]) != tuple(quantize_joints(stable_pose)[0])
                    if action.get('direct_student_target') is True and not direct_no_motion else
                    max(abs(a - b) for a, b in zip(snapshot.joints_deg, stable_pose))
                        > self.limits.joint_tolerance_deg / 2))
                if stable_pose is None or changed_while_settling:
                    stable_pose = snapshot.joints_deg
                    stable_since = snapshot.sensor_time if retain else time.monotonic()
                    stable_sequence = snapshot.sensor_sequence
                elapsed = snapshot.sensor_time - stable_since if retain else time.monotonic() - stable_since
                if elapsed >= self.limits.stable_seconds and (not retain or snapshot.sensor_sequence > stable_sequence):
                    if 'damped_compensation' in action:
                        self._check_damped_compensation_error(action, snapshot)
                    if sdk_motion and 'damped_compensation' not in action:
                        # SDK completion and encoder residual are separate facts.
                        status = ('settled_no_motion' if direct_no_motion else
                                  'completed' if nominal_reached and sent_reached else 'settled')
                        if direct_no_motion:
                            nominal_reached = sent_reached = False
                    elif nominal_reached:
                        status = "completed"
                    elif retain and action.get('allow_target_follow_up') is True:
                        status = 'settled_needs_target_follow_up'
                    elif (nominal is not None and action.get('allow_mock_nominal_retry') is True
                          and arrival_joint_error <= MAX_NOMINAL_TRIM_DEG
                          and arrival_tcp_error <= MAX_MOCK_NOMINAL_RETRY_TCP_MM):
                        # This is not arrival or success. Only an isolated mock
                        # caller can use the fresh stopped observation to try a
                        # separately validated correction.
                        status = 'settled_needs_nominal_retry'
                    elif (nominal is not None and action.get('allow_nominal_correction') is True
                          and arrival_joint_error <= MAX_NOMINAL_TRIM_DEG
                          and arrival_tcp_error <= MAX_MOCK_NOMINAL_RETRY_TCP_MM):
                        # Normal motion finished, but nominal arrival was not
                        # reached. A later, explicitly bound command may apply
                        # a separately checked correction; this is not success.
                        status = 'settled_needs_nominal_correction'
                    elif action.get("allow_visual_settle", False) and tcp_error <= self.limits.coarse_residual_mm:
                        status = "settled_needs_observation"
                    else:
                        details = {'joint_error_deg': joint_error, 'tcp_error_mm': tcp_error}
                        if nominal is not None:
                            details.update(arrival_reference='nominal', arrival_joint_error_deg=arrival_joint_error,
                                           arrival_tcp_error_mm=arrival_tcp_error)
                        raise ControlError("TARGET_NOT_REACHED", "Motion ended outside the permitted residual", **details)
                    if retain and status in RETAINED_POSITION_SUCCESS_STATUSES:
                        if not sent_reached and not sdk_motion and 'damped_compensation' not in action:
                            raise ControlError('TARGET_NOT_REACHED', 'The sent position target was not reached')
                        self._require_quiet_motion_diagnostics(before.firmware_warning_raw,
                                                               before.collision_event_counts, snapshot)
                        completion = {'mode': 'position_target_retained', 'sensor_sequence': snapshot.sensor_sequence,
                                      'sensor_monotonic': snapshot.sensor_time, 'target_steps': list(action['steps']),
                                      'host_frame': retained_host_frame, 'latest_host_frame': retained_latest_host_frame,
                                      'motion_observed': seen_motion,
                                      'encoder_step_observed': seen_encoder_step,
                                      'observation_basis': ('no_motion_before_existing_timeout' if direct_no_motion else
                                          'encoder_displacement_then_stationary' if direct_displacement is not None
                                              and direct_displacement > self.limits.joint_tolerance_deg else
                                          'target_encoder_steps_observed' if direct_exact_target else
                                          'within_arrival_tolerance' if direct_within_arrival else
                                          'same_target_prior_evidence_then_stationary' if previous_same_target is not None else
                                          'motion_reported_then_stationary' if seen_motion else
                                          'sdk_completion_event' if snapshot.completion_sequence > before.completion_sequence
                                          else 'already_within_target_tolerance'),
                                      # SDK event counters and moving flags do not carry our command UUID.
                                      'device_event_bound_to_command_id': False}
                        if previous_same_target is not None:
                            completion['previous_same_target_evidence'] = previous_same_target
                        if last_motion_feedback is not None and status in ('completed', 'settled'):
                            measured = last_motion_feedback
                            completion['end_motion_observation'] = {
                                'schema_version': 'PolicyObservation.phase1', 'instance_id': self.instance_id,
                                'sensor_sequence': measured.sensor_sequence, 'sensor_monotonic': measured.sensor_time,
                                'seconds_after_encoded_target': measured.sensor_time - retained_host_frame['monotonic'],
                                'joints_deg': list(measured.joints_deg), 'sensor_valid': True, 'moving': measured.moving,
                                'gripper_command': self._gripper_command,
                                'gripper_reported_state': measured.gripper_reported_state,
                                'gripper_reported_valid': True, 'gripper_feedback_fresh': True,
                            }
                        result = {'status': status, 'applied': True, 'target': action, 'receipt': receipt,
                                'joint_error_deg': joint_error, 'tcp_error_mm': tcp_error,
                                'arrival_reference': 'nominal', 'nominal_arrival_reached': nominal_reached,
                                'arrival_joint_error_deg': arrival_joint_error,
                                'arrival_tcp_error_mm': arrival_tcp_error, 'sent_target_reached': sent_reached,
                                'sent_target_source': 'host_encoded_target_and_encoder_observation',
                                'completion': completion, 'stop': {'requested': False, 'confirmed': False}}
                        if 'damped_compensation' in action:
                            result['damped_compensation'] = {**deepcopy(action['damped_compensation']),
                                'nominal_joint_error_after_deg': arrival_joint_error,
                                'nominal_tcp_error_after_mm': arrival_tcp_error,
                                'compensated_target_reached': sent_reached}
                        return result
                    if retain and status == 'settled_needs_target_follow_up':
                        return {'status': status, 'applied': True, 'target': action, 'receipt': receipt,
                                'joint_error_deg': joint_error, 'tcp_error_mm': tcp_error,
                                'arrival_reference': 'nominal', 'nominal_arrival_reached': False,
                                'arrival_joint_error_deg': arrival_joint_error,
                                'arrival_tcp_error_mm': arrival_tcp_error, 'sent_target_reached': False,
                                'stop': {'requested': False, 'confirmed': False}}
                    if action.get('allow_damped_compensation') is True and status == 'settled_needs_nominal_correction':
                        self._require_quiet_motion_diagnostics(before.firmware_warning_raw,
                                                               before.collision_event_counts, snapshot)
                        correction_action, correction_basis = self._make_nominal_correction(
                            action, snapshot, original_host_frame=retained_host_frame,
                            latest_host_frame=retained_latest_host_frame)
                        parent = {'mode': 'position_target_retained', 'command_id': self._active,
                                  'stop_epoch': correction_basis['stop_epoch'],
                                  'sensor_sequence': snapshot.sensor_sequence,
                                  'sensor_monotonic': snapshot.sensor_time,
                                  'target_steps': list(action['steps']),
                                  'host_frame': deepcopy(retained_host_frame),
                                  'latest_host_frame': deepcopy(retained_latest_host_frame)}
                        return {'status': status, 'applied': True, 'target': action, 'receipt': receipt,
                                'joint_error_deg': joint_error, 'tcp_error_mm': tcp_error,
                                'arrival_reference': 'nominal', 'nominal_arrival_reached': False,
                                'arrival_joint_error_deg': arrival_joint_error,
                                'arrival_tcp_error_mm': arrival_tcp_error, 'sent_target_reached': False,
                                'sent_target_source': 'host_encoded_target_and_encoder_observation',
                                'nominal_correction_permitted': True,
                                'correction_action': correction_action, 'correction_basis': correction_basis,
                                'retained_parent': parent, 'stop': {'requested': False, 'confirmed': False}}
                    result = self._verify_post_stop_arrival(
                        action, before, effective, target_tcp, arrival_joints, arrival_tcp,
                        arrival_tolerance, nominal, sdk_motion, status, joint_error, tcp_error,
                        retained_host_frame)
                    result['receipt'] = receipt
                    return result
            else:
                stable_pose, stable_since, stable_sequence, stable_in_bounds = None, None, None, None
        raise ControlError("MOTION_TIMEOUT" if seen_motion else "NO_MOTION_CONFIRMATION", "Motion did not finish before its deadline")

    def _execute_gripper(self, action):
        with span('gripper_dispatch_and_feedback'):
            return self._execute_gripper_impl(action)

    def _execute_gripper_impl(self, action):
        retain_id = action.get('retain_position_from_command_id')
        retained = None
        with self._dispatch_lock:
            if self._fault:
                raise self._fault
            if self._snapshot.firmware_warning_raw==1:
                raise ControlError('DEVICE_WARNING','Device warning blocks the gripper command')
            if self._cancel.is_set():
                return {"status": "cancelled", "applied": False, "reason": self._cancel_reason}
            arguments = {'state': action['state']}
            if retain_id is not None:
                with self._condition:
                    current = self._snapshot
                retained, frame = self._checked_retained_position(retain_id, current, require_arrival=False)
                held_action = retained['action']
                arguments['retain_arm'] = {
                    'command_sequence': frame['command_sequence'],
                    'target_segment_start_sequence': frame['target_segment_start_sequence'],
                    'target_steps': list(held_action['steps']),
                    'target_deg': list(held_action['effective_target_deg']),
                    'nominal_target_deg': list(held_action['nominal_arrival']['effective_target_deg']),
                    'nominal_only_arrival': 'damped_compensation' in held_action,
                    'speed_pct': frame['speed_pct'], 'sensor_sequence': current.sensor_sequence,
                    'arrival_tolerance_mm': held_action['arrival_tolerance_mm'],
                    'path_constraints': deepcopy(held_action['path_constraints']),
                    'collision_event_counts': retained['before'].collision_event_counts,
                }
            sent_at = time.monotonic()
            try:
                with span('device_dispatch_rpc'):
                    receipt = self.worker.call("set_gripper", arguments, send_guard=self._action_send_guard)
            except _DispatchCancelled as exc:
                return {"status": "cancelled", "applied": False, "reason": str(exc)}
            # The SLOT_W2 command has been submitted even if a later evidence
            # check fails. Preserve that fact across an actual emergency stop.
            self._gripper_command = action["state"]
            self._gripper_evidence = {'applied': True, 'gripper_command': action['state'],
                                      'gripper_reported_confirmed': False, 'object_held_visual': None}
            if retained is not None:
                self._gripper_evidence['arm_nominal_arrival_reached'] = False
            if retained is not None:
                if (not isinstance(receipt, dict) or type(receipt.get('after_frame_sequence')) is not int
                        or type(receipt.get('before_sensor_sequence')) is not int
                        or type(receipt.get('sent_monotonic')) not in (int, float)):
                    raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The gripper dispatch has no SDK host boundary')
                sent_at = receipt['sent_monotonic']
        with span('journal_dispatch'):
            self.journal.append("command_dispatched", command_id=self._active, gripper_command=action["state"], object_held_visual=None)
        confirmed = False
        arm_hold, post_gripper_frame = None, None
        stable_pose, stable_since, stable_sequence = None, None, None
        first_gripper_report = None
        deadline = sent_at + min(GRIPPER_TIMEOUT_CAP_SECONDS, self.limits.motion_timeout)
        while time.monotonic() < deadline and not self._cancel.is_set():
            with self._condition:
                snapshot = self._snapshot
            fresh = self._fresh(snapshot)
            if (fresh and snapshot.gripper_time is not None and snapshot.gripper_time >= sent_at
                    and snapshot.gripper_reported_state == action['state']):
                self._gripper_evidence['gripper_reported_confirmed'] = True
            if retained is not None and not fresh:
                if self._sdk_feedback_gap_in_grace(snapshot):
                    wait_started = time.monotonic()
                    with self._condition:
                        self._condition.wait(self.limits.sample_period)
                    deadline += time.monotonic() - wait_started
                    continue
                raise ControlError('SENSOR_STALE', 'Fresh feedback is required while the gripper changes state')
            if retained is not None and snapshot.sensor_sequence > receipt['before_sensor_sequence'] and snapshot.sensor_time > sent_at:
                # Do not confuse transient movement during SLOT_W2 with final
                # arrival. The same live position frame, diagnostics and the
                # original spatial constraints remain mandatory throughout.
                _, frame = self._checked_retained_position(retain_id, snapshot,
                                                          require_arrival=False, require_stationary=False)
                if (post_gripper_frame is None and frame['frame_sequence'] > receipt['after_frame_sequence']
                        and frame['monotonic'] >= sent_at):
                    post_gripper_frame = dict(frame)
                held_action = retained['action']
                nominal = held_action['nominal_arrival']
                joint_error = max(abs(a-b) for a, b in zip(snapshot.joints_deg, nominal['effective_target_deg']))
                tcp_error = distance(self.geometry.forward(snapshot.joints_deg), nominal['target_tcp_mm'])
                reported_after_frame = (post_gripper_frame is not None and snapshot.gripper_time is not None
                                        and snapshot.gripper_time > post_gripper_frame['monotonic']
                                        and snapshot.gripper_reported_state == action['state'])
                if reported_after_frame and first_gripper_report is None:
                    # Latch once. Later updates of the same gripper state must
                    # not keep pushing the stable encoder window forward.
                    first_gripper_report = snapshot.gripper_time
                elif not reported_after_frame:
                    first_gripper_report = None
                if (first_gripper_report is not None and snapshot.sensor_time > first_gripper_report
                        and snapshot.moving is False):
                    if (stable_pose is None or
                            max(abs(a-b) for a, b in zip(snapshot.joints_deg, stable_pose)) > self.limits.joint_tolerance_deg / 2):
                        stable_pose, stable_since, stable_sequence = (snapshot.joints_deg, snapshot.sensor_time,
                                                                     snapshot.sensor_sequence)
                else:
                    stable_pose, stable_since, stable_sequence = None, None, None
                stable = (stable_since is not None and snapshot.sensor_sequence > stable_sequence
                          and snapshot.sensor_time-stable_since >= self.limits.stable_seconds)
                if stable and reported_after_frame:
                    confirmed = True
                    arm_hold = {'motion_command_id': retain_id, 'target_steps': list(held_action['steps']),
                                'frame_sequence': post_gripper_frame['frame_sequence'],
                                'target_segment_start_sequence': post_gripper_frame['target_segment_start_sequence'],
                                'frame_monotonic': post_gripper_frame['monotonic'], 'sensor_sequence': snapshot.sensor_sequence,
                                'sensor_monotonic': snapshot.sensor_time,
                                'first_gripper_report_monotonic': first_gripper_report,
                                'stable_sensor_start_sequence': stable_sequence,
                                'stable_sensor_start_monotonic': stable_since}
                    arrived = (joint_error <= self.limits.joint_tolerance_deg
                               and tcp_error <= held_action['arrival_tolerance_mm'])
                    self._gripper_evidence['arm_nominal_arrival_reached'] = arrived
                    if arrived or self.config.backend == 'sdk':
                        break
                    if (action['state'] != 'open' or held_action.get('allow_damped_compensation') is not True
                            or 'damped_compensation' in held_action):
                        raise ControlError('TARGET_NOT_REACHED', 'The arm settled outside its original arrival bounds',
                                           **self._retained_arrival_details(held_action, snapshot))
                    correction_action, correction_basis = self._make_nominal_correction(
                        held_action, snapshot, original_host_frame=retained['completion']['host_frame'],
                        latest_host_frame=frame, parent_command_id=retain_id)
                    correction_basis['after_gripper_command_id'] = self._active
                    parent = {'mode': 'position_target_retained', 'command_id': retain_id,
                              'stop_epoch': correction_basis['stop_epoch'],
                              'sensor_sequence': snapshot.sensor_sequence, 'sensor_monotonic': snapshot.sensor_time,
                              'target_steps': list(held_action['steps']),
                              'host_frame': deepcopy(retained['completion']['host_frame']),
                              'latest_host_frame': deepcopy(frame)}
                    return {'status': 'gripper_reported_arm_needs_nominal_correction', 'applied': True,
                            'gripper_command': action['state'], 'gripper_reported_confirmed': True,
                            'object_held_visual': None, 'arm_nominal_arrival_reached': False, 'arm_hold': arm_hold,
                            'nominal_correction_permitted': True, 'correction_action': correction_action,
                            'correction_basis': correction_basis, 'retained_parent': parent,
                            'stop': {'requested': False, 'confirmed': False}}
            elif (retained is None and self._fresh(snapshot) and snapshot.gripper_time is not None
                  and snapshot.gripper_time >= sent_at and snapshot.gripper_reported_state == action['state']):
                confirmed = True
                break
            with self._condition:
                self._condition.wait(self.limits.sample_period)
        if self._fault:
            raise self._fault
        if self._cancel.is_set():
            return {"status": "cancelled", "reason": self._cancel_reason, **deepcopy(self._gripper_evidence)}
        if retained is not None and not confirmed:
            raise ControlError('POSITION_TARGET_UNCONFIRMED', 'Fresh gripper and held-position evidence was not confirmed')
        result = {"status": "gripper_reported" if confirmed else "gripper_commanded_unverified", "applied": True, "gripper_command": action["state"], "gripper_reported_confirmed": confirmed, "object_held_visual": None}
        if isinstance(receipt, dict) and 'arrival_diagnostics' in receipt:
            result['arrival_diagnostics_before_gripper'] = deepcopy(receipt['arrival_diagnostics'])
        if arm_hold is not None:
            result['arm_hold'] = arm_hold
            result['arm_nominal_arrival_reached'] = self._gripper_evidence['arm_nominal_arrival_reached']
            result['arm_arrival_observation'] = self._retained_arrival_details(retained['action'], snapshot)
        return result

    def confirm_idle_stop(self, observation_id):
        """Retry only stop confirmation after an idle fault, with no owner takeover."""
        with self._condition:
            if self._closed or self._closing.is_set():
                raise ControlError('CLOSED','Controller is shutting down')
            if self._active is not None or self._lease is not None:
                raise ControlError('RECOVERY_BUSY','기존 작업이 아직 종료되지 않았습니다.')
            if self._fault is None:
                raise ControlError('RECOVERY_NOT_NEEDED','고장 상태가 아니므로 추가 정지는 필요하지 않습니다.')
            self._validate_observation(observation_id)
            # Reserve the idle fault while waiting without holding the sensor lock.
            self._active='idle-stop-confirmation'
        try:
            return {**self.stop('RECOVERY_STOP_REQUESTED'),'instance_id':self.instance_id}
        finally:
            with self._condition:
                self._active=None
                self._condition.notify_all()

    def stop(self, reason="STOP_REQUESTED"):
        reason = identifier(reason, "reason")
        with self._stop_intent_lock:
            self._pending_stops += 1
            self._stop_epoch += 1
            self._cancel_reason = reason
            self._cancel.set()
        try:
            with self._condition:
                self._last_stop = {"requested": False, "confirmed": False}
                self._revision += 1
                if self._fault is None:
                    self._state = "stopped"
                self._condition.notify_all()
            result = self._stop_device()
            self.journal.append("stop_requested", reason=reason, result=result)
            return result
        finally:
            with self._stop_intent_lock:
                self._pending_stops -= 1

    def interrupt_motion(self, lease_id, command_id):
        """Stop one MOVE, retaining the lease for a separately checked replan.

        The intent is ordered with dispatch, not the worker's response or the
        device's stop. SDK command delivery and stop latency remain unverified;
        a closed gripper does not prove that an object is held.
        """
        identifier(lease_id, 'lease_id')
        identifier(command_id, 'command_id')
        token = uuid.uuid4().hex
        # This short lock also orders ownership, the currently active or
        # admitted request identity, cancellation and the final pipe send.
        # It is never held while acquiring _condition or performing I/O.
        with self._stop_intent_lock:
            if self._closed or self._closing.is_set():
                raise ControlError('CLOSED', 'Controller is shutting down')
            if self._fault:
                raise self._fault
            if self._lease is None or self._lease['lease_id'] != lease_id:
                raise ControlError('LEASE_REQUIRED', 'A matching control lease is required')
            now = time.monotonic()
            if now >= self._lease['deadline']:
                raise ControlError('LEASE_EXPIRED', 'The control lease has expired')
            if self._pending_stops or self._cancel.is_set():
                raise ControlError('STOP_IN_PROGRESS', 'Another stop already has priority')
            if self._active is not None and (self._active != command_id or self._active_kind not in ('move_joints', 'blended_joint_path')):
                raise ControlError('INTERRUPTION_MISMATCH', 'A different command owns the controller')
            intentions = [item for item in self._command_intentions.values() if item['lease_id'] == lease_id]
            if self._active is None and self._preparing is not None:
                if self._preparing['lease_id'] != lease_id or self._preparing['command_id'] != command_id:
                    raise ControlError('INTERRUPTION_MISMATCH', 'A different request owns admission')
            elif self._active is None and intentions and any(
                    item['command_id'] != command_id or item['kind'] not in ('move_joints', 'blended_joint_path') for item in intentions):
                raise ControlError('INTERRUPTION_MISMATCH', 'A different request owns admission')
            existing = self.journal.records.get(command_id)
            if (self._active is None and existing is not None
                    and (existing['result'] is not None or command_id in self._lease_finished_commands)):
                raise ControlError('COMMAND_ALREADY_FINISHED', 'The command has already finished')
            if self._state not in ('armed', 'executing'):
                raise ControlError('NOT_ARMED', 'The controller is not executing or armed')
            if self._active is None:
                self._lease['deadline'] = now + self.limits.lease_seconds
            self._pending_stops += 1
            self._stop_epoch += 1
            epoch = self._stop_epoch
            self._cancel_reason = 'RETARGET_INTERRUPT'
            self._cancel.set()
            self._retarget_preempted_commands[command_id] = lease_id
            latest_counts = (tuple(self._snapshot.collision_event_counts)
                if self._snapshot is not None and self._snapshot.collision_event_counts is not None else None)
            latest_warning = self._snapshot.firmware_warning_raw if self._snapshot is not None else None
            # For an already admitted move, do not replace missing pre-move
            # diagnostics with a later sample: that would erase the interval
            # in which an event could have happened.
            active_move = self._active == command_id
            record = {'id': token, 'lease_id': lease_id, 'command_id': command_id,
                      'stop_epoch': epoch, 'stop': None,
                      'collision_baseline': self._active_collision_baseline if active_move else latest_counts,
                      'warning_baseline': self._active_warning_baseline if active_move else latest_warning}
            self._motion_interruption = record
        try:
            with self._condition:
                if self._fault is None and not self._closing.is_set():
                    self._state = 'stopped'
                self._last_stop = {'requested': False, 'confirmed': False}
                self._revision += 1
                self._condition.notify_all()
            stopped = self._stop_device()
            try:
                self.journal.append('motion_interrupted', command_id=command_id,
                                    interruption_id=token, stop=stopped, physical_success=None)
            except ControlError as exc:
                self._latch_fault(exc)
                raise
            with self._condition:
                with self._stop_intent_lock:
                    record['stop'] = deepcopy(stopped)
                    current = (self._motion_interruption is record and self._stop_epoch == epoch
                               and self._fault is None and self._cancel_reason == 'RETARGET_INTERRUPT')
            return {'instance_id': self.instance_id, 'interruption_id': token, 'command_id': command_id,
                    'stop': stopped, 'superseded_by_stop': not current,
                    'grasp_verified': False, 'physical_success': None,
                    'stop_delay_bounded': False}
        finally:
            with self._stop_intent_lock:
                self._pending_stops -= 1

    def resume_interrupted_motion(self, lease_id, interruption_id, observation_id):
        """Re-arm only this lease after its own stop and a fresh observation.

        This performs no motion and provides no object-state evidence. The
        caller must separately construct and validate every subsequent action.
        """
        identifier(lease_id, 'lease_id')
        identifier(interruption_id, 'interruption_id')
        identifier(observation_id, 'observation_id')
        with self._condition:
            self._require_lease(lease_id)
            record = self._motion_interruption
            if (record is None or record['id'] != interruption_id or record['lease_id'] != lease_id):
                raise ControlError('INTERRUPTION_MISMATCH', 'Interruption does not belong to this lease')
            if self._active is not None or self._preparing is not None:
                raise ControlError('BUSY', 'The interrupted action has not returned')
            self._validate_observation(observation_id)
            observation = self._observations[observation_id]
            stopped = record['stop']
            if (not isinstance(stopped, dict) or stopped.get('confirmed') is not True
                    or observation['state'] != 'stopped' or observation['moving'] is not False
                    or observation['sensor_sequence'] < stopped.get('confirmed_sensor_sequence', float('inf'))
                    or observation['observed_monotonic'] < stopped.get('confirmed_sensor_monotonic', float('inf'))
                    or not self._fresh(self._snapshot) or self._snapshot.moving is not False):
                raise ControlError('STOP_UNCONFIRMED', 'Fresh stationary feedback after this stop is required')
            # An interrupt can arrive between two motion samples. Preserve the
            # original command's collision baseline across the stop and never
            # re-arm over a newly reported warning or collision event.
            if self._snapshot.firmware_warning_raw == 1:
                error = ControlError('DEVICE_WARNING', 'Device warning blocks replanning')
                self._latch_fault(error)
                raise error
            baseline, current = record['collision_baseline'], self._snapshot.collision_event_counts
            if self.config.backend == 'sdk':
                warning_before, warning_now = record['warning_baseline'], self._snapshot.firmware_warning_raw
                if (type(warning_before) is not int or type(warning_now) is not int
                        or warning_before != 0 or warning_now != 0
                        or not self._valid_collision_counts(baseline)
                        or not self._valid_collision_counts(current)):
                    error = ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE',
                        'Warning and collision-event feedback must be present before and after interruption')
                    self._latch_fault(error)
                    raise error
                if any(new < old for old, new in zip(baseline, current)):
                    error = ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE',
                        'Collision-event counters changed discontinuously during interruption')
                    self._latch_fault(error)
                    raise error
            if (baseline is not None and current is not None and len(baseline) == len(current)
                    and any(new > old for old, new in zip(baseline, current))):
                error = ControlError('DEVICE_COLLISION_EVENT', 'A new collision event blocks replanning',
                                     physical_contact_cause_confirmed=False)
                self._latch_fault(error)
                raise error
            with self._stop_intent_lock:
                if (self._pending_stops or self._stop_epoch != record['stop_epoch']
                        or self._cancel_reason != 'RETARGET_INTERRUPT'):
                    raise ControlError('STOP_IN_PROGRESS', 'Another stop superseded the interruption')
                self._cancel.clear()
                self._motion_interruption = None
                self._state = 'armed'
                self._revision += 1
                self._condition.notify_all()
            return {'instance_id': self.instance_id, 'resumed': True, 'interruption_id': interruption_id,
                    'command_id': record['command_id'], 'reference_source': 'stationary_observation',
                    'joints_deg': list(observation['joints_deg']),
                    'grasp_verified': False, 'physical_success': None}

    def interrupt_mock_motion(self, lease_id, command_id):
        """Compatibility API that remains explicitly mock-only."""
        if self.config.backend != 'mock' or self.config.hardware_enabled is not False or self.config.port is not None:
            raise ControlError('UNSUPPORTED_ACTION', 'Motion interruption is mock-only')
        return self.interrupt_motion(lease_id, command_id)

    def resume_mock_interrupted_motion(self, lease_id, interruption_id, observation_id):
        """Compatibility API that remains explicitly mock-only."""
        if self.config.backend != 'mock' or self.config.hardware_enabled is not False or self.config.port is not None:
            raise ControlError('UNSUPPORTED_ACTION', 'Motion interruption is mock-only')
        return self.resume_interrupted_motion(lease_id, interruption_id, observation_id)

    def end_episode(self, lease_id):
        with self._condition:
            # Faulted episodes still need an explicit outcome. Ending control
            # does not clear the fault or authorize another movement.
            identifier(lease_id, "lease_id")
            if self._closed or self._closing.is_set():
                raise ControlError("CLOSED", "Controller is shutting down")
            if self._lease is None or lease_id != self._lease["lease_id"]:
                raise ControlError("LEASE_REQUIRED", "A matching control lease is required")
            if self._active is not None or self._preparing is not None:
                raise ControlError("BUSY", "Stop or finish the active command before ending the episode")
            episode_id = self._lease["episode_id"]
        stopped = self.stop("EPISODE_ENDED")
        with self._condition:
            with self._stop_intent_lock:
                self._lease = None
            self._revision += 1
        fault = self._fault.as_dict() if self._fault else None
        self.journal.append("episode_ended", episode_id=episode_id, verified_success=None, fault=fault, stop=stopped)
        return {"episode_id": episode_id, "verified_success": None, "fault": fault, "stop": stopped}

    def result(self, command_id):
        identifier(command_id, "command_id")
        with self._condition:
            record = self.journal.records.get(command_id)
            if record is None:
                raise ControlError("COMMAND_UNKNOWN", "No command with that ID is recorded")
            return deepcopy(record["result"]) if record["result"] is not None else {"command_id": command_id, "status": "outcome_unknown_or_in_progress"}

    def close(self):
        with self._close_lock:
            return self._close()

    def _close(self):
        if self._closed:
            return {"closed": True, "worker_alive": self.worker.alive}
        with self._condition:
            self._closing.set()
            self._state = "closing"
            with self._stop_intent_lock:
                self._stop_epoch += 1
                self._cancel_reason = "SHUTDOWN"
                self._cancel.set()
            self._revision += 1
        stop = self._stop_device() if self.worker.alive else {"requested": False, "confirmed": False}
        if self._sampler is not None:
            self._sampler.join(self.limits.rpc_timeout + 1)
        with self._condition:
            self._condition.wait_for(lambda: self._active is None, timeout=self.limits.motion_timeout + self.limits.rpc_timeout)
        worker = self.worker.close()
        with self._condition:
            self._state = "closed"
            self._closed = True
        try:
            self.journal.append("service_closed", instance_id=self.instance_id, stop=stop, worker=worker)
        finally:
            self.journal.close()
            self.resource.close()
        return {"closed": True, "stop": stop, "worker": worker}
