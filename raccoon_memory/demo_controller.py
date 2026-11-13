"""Explicitly owned controller for the local demonstration service.

Only an explicit connection request starts or attaches a controller. The mock
is isolated from serial devices; hardware requires an existing validated
project profile, including an explicitly identified photo-based trial profile.
The display reads the controller's existing sampler, never an observation ID
that the student's next command might need.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import uuid

from raccoon_control.config import Config, Limits
from raccoon_control.controller import Controller
from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, quantize_joints


class PhysicalProfileValidationError(ValueError):
    """A profile check failed before any controller lookup or start."""


def mock_controller(runtime):
    """Start a separate mock controller without sending a movement command."""
    start = list(quantize_joints([0., -10., -110., 30.])[1])
    # This is a simulated envelope only.  The four displayed mock slots
    # include two around Y=0; no value here is a physical workspace claim.
    limits = Limits(workspace_min_mm=(-180., -180., 5.), workspace_max_mm=(180., 280., 290.))
    config = Config(backend='mock', hardware_enabled=False, port=None, workspace_verified=True,
                    limits=limits, mock={'initial_joints_deg': start, 'move_duration': .08})
    if config.backend != 'mock' or config.hardware_enabled is not False or config.port is not None:
        raise ValueError('모의 제어기 설정을 확인해 주세요.')
    controller = Controller(config, Path(runtime) / uuid.uuid4().hex)
    try:
        controller.start()
        return DisplayController(controller), start
    except BaseException:
        controller.close()
        raise


def physical_controller(profile_path, *, force_reconnect=False, previous_controller=None):
    """Connect only after explicit selection and project profile validation.

    Connection does not command the gripper or arm.  A caller must separately
    supply a confirmed applied target and a controller-confirmed open gripper
    before registering a physical scene.
    """
    from .app_runtime import AppProfile, attach_controller

    try:
        profile = AppProfile.load(profile_path)
        config, calibration = profile.settings()
        if config.backend != 'sdk' or config.hardware_enabled is not True or calibration.simulation_only is not False:
            raise ValueError('실물 장치 전용 작업 공간과 자리 교정 설정이 필요합니다.')
        if profile.max_height > 2:
            raise ValueError('실물 준비 범위는 최대 2층입니다.')
    except (ControlError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        # Only file loading and validation are inside this boundary. Errors
        # from resident lookup, process startup and control stay uncertain.
        raise PhysicalProfileValidationError(str(exc)) from exc
    # Reuse the existing resident when its configuration matches.  Its
    # connection belongs to the resident, not this UI, even if this explicit
    # request launched it.  A nonresponsive or differently configured owner is
    # never stopped or replaced here.
    try:
        previous_owner = None
        if previous_controller is not None:
            attachment = getattr(previous_controller, 'attachment', None)
            if isinstance(attachment, dict) and isinstance(attachment.get('runtime'), str):
                previous_owner = {'runtime': attachment['runtime'],
                                  'instance_id': getattr(previous_controller, 'instance_id', None)}
        controller, attachment = attach_controller(profile, allow_hardware=True, allow_start=True,
            _validated_config=config, force_reconnect=force_reconnect, previous_owner=previous_owner)
        wrapped = DisplayController(controller)
        wrapped.attachment = attachment
        return wrapped, calibration, profile
    except ControlError as exc:
        if getattr(exc, 'pending_start', None) is not None:
            # Retain the exact settings validated before this process started.
            exc.physical_context = (exc.pending_start, calibration, profile)
        raise
    except PhysicalProfileValidationError as exc:
        # Even this exception type must not certify a failure raised after the
        # lookup/start boundary as safe to retry.
        raise RuntimeError(str(exc)) from exc


def verify_pending_physical_profile(context, profile_path):
    """Do not ignore configuration or calibration edits made while waiting."""
    from .app_runtime import AppProfile

    start, calibration, original = context
    try:
        current = AppProfile.load(profile_path)
        config, current_calibration = current.settings()
        if (current != original or config.fingerprint() != start.config.fingerprint()
                or current_calibration != calibration):
            raise ValueError('연결을 기다리는 동안 설정 또는 교정이 변경됐습니다.')
    except (ControlError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ControlError('CONTROLLER_CONFIG_MISMATCH',
                           '연결을 기다리는 동안 설정 또는 교정을 확인할 수 없거나 내용이 바뀌었습니다.') from exc


def resume_pending_physical_controller(context, profile_path):
    """Recheck only the child belonging to the original connection request."""
    from .app_runtime import reattach_pending_controller

    start, calibration, profile = context
    verify_pending_physical_profile(context, profile_path)
    controller, attachment = reattach_pending_controller(start)
    verify_pending_physical_profile(context, profile_path)
    wrapped = DisplayController(controller)
    wrapped.attachment = attachment
    return wrapped, calibration, profile


class UnavailableControllerForecast:
    """Keep optional forecast failures separate from controller registration."""
    def begin_decision(self, *_args):
        pass

    def stage_segments(self, *_args):
        pass

    def invalidate(self, *_args):
        pass

    def command_started(self, *_args):
        pass

    def command_finished(self, *_args):
        pass

    def snapshot(self):
        return {'segments': [], 'complete': False, 'reason': 'unavailable',
                'scope': 'current_transfer', 'basis_command_id': None}

    def close(self, *, wait=False):
        return True


class DisplayController:
    """Forward student commands while publishing a read-only command preview."""

    def __init__(self, controller):
        self._controller = controller
        self._lock = threading.Lock()
        self._command = None
        self._trajectory = None
        self._forecast = None
        self._stopping = False
        self._stop_generation = 0
        self._lease_owner = None

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def acquire(self, owner):
        with self._lock:
            generation = self._stop_generation
        result = self._controller.acquire(owner)
        with self._lock:
            if self._stop_generation == generation:
                self._stopping = False
            self._lease_owner = owner
        return result

    def end_episode(self, lease_id):
        try:
            return self._controller.end_episode(lease_id)
        finally:
            with self._lock:
                self._lease_owner = None
                if self._forecast is not None:
                    try:
                        self._forecast.invalidate('no_active_command')
                    except Exception:
                        pass

    def observe(self, lease_id=None):
        return self._controller.observe(lease_id)

    def _student_decision_context(self, observation, before, task, slots, transfer, decision_index):
        """Receive a display-only copy of the inputs to one student decision."""
        with self._lock:
            if self._forecast is None:
                return
            try:
                if self._stopping or before is None or task is None:
                    self._forecast.invalidate('unavailable')
                else:
                    self._forecast.begin_decision(observation, before, task, slots, transfer, decision_index)
            except Exception:
                try:
                    self._forecast.invalidate('unavailable')
                except Exception:
                    pass

    def _student_segment_plan(self, command_id, segment_index, waypoints):
        """Stage the executor's own planned endpoints for display only."""
        with self._lock:
            if self._forecast is None:
                return
            try:
                if self._stopping:
                    self._forecast.invalidate('cancelled')
                else:
                    self._forecast.stage_segments(command_id, segment_index, waypoints)
            except Exception:
                try:
                    self._forecast.invalidate('unavailable')
                except Exception:
                    pass

    def _invalidate_student_forecast(self, reason):
        with self._lock:
            if self._forecast is not None:
                try:
                    self._forecast.invalidate(reason)
                except Exception:
                    pass

    def stop(self, reason='STOP_REQUESTED'):
        with self._lock:
            self._stopping = True
            self._stop_generation += 1
            if self._command is not None:
                self._command['status'] = 'stop_requested'
            if self._forecast is not None:
                try:
                    self._forecast.invalidate('cancelled')
                except Exception:
                    pass
        return self._controller.stop(reason)

    def _interrupt_motion(self, handler, lease_id, command_id):
        with self._lock:
            command = self._command
            matching = command is not None and command['command_id'] == command_id
            previous = command.get('status') if matching else None
            if matching:
                command['status'] = 'retarget_interrupt_requested'
            if matching and self._forecast is not None:
                try:
                    self._forecast.invalidate('retargeting')
                except Exception:
                    pass
        try:
            return handler(lease_id, command_id)
        except BaseException as exc:
            # A lost RPC response may mean that the stop reached the server.
            # Only a definite admission rejection may restore the old path.
            definitely_rejected = getattr(exc, 'code', None) in {
                'LEASE_REQUIRED', 'LEASE_EXPIRED', 'INTERRUPTION_MISMATCH',
                'COMMAND_ALREADY_FINISHED', 'CONTROLLER_CHANGED', 'METHOD_NOT_FOUND',
                'UNSUPPORTED_ACTION', 'NOT_ARMED', 'CLOSED'}
            with self._lock:
                if (matching and self._command is command and not self._stopping
                        and command.get('status') == 'retarget_interrupt_requested'):
                    command['status'] = previous if definitely_rejected else 'retarget_interrupt_unconfirmed'
            raise

    def interrupt_motion(self, lease_id, command_id):
        return self._interrupt_motion(self._controller.interrupt_motion, lease_id, command_id)

    def interrupt_mock_motion(self, lease_id, command_id):
        return self._interrupt_motion(self._controller.interrupt_mock_motion, lease_id, command_id)

    def apply_action(self, request):
        action = request['action']
        kind = action.get('kind')
        trajectory = (action.get('targets_deg') if kind == 'mock_joint_trajectory' else
                      action.get('waypoints_deg') if kind == 'blended_joint_path' else None)
        target = (trajectory[-1] if trajectory else action.get('joints_deg'))
        command = {'command_id': request['command_id'], 'kind': kind,
                   'target_joints_deg': list(target) if target is not None else None,
                   'gripper_state': action.get('state') if kind == 'set_gripper' else None,
                   # The controller still performs admission and safety checks.
                   'status': 'in_progress'}
        with self._lock:
            self._command = command
            self._trajectory = tuple(tuple(row) for row in trajectory) if trajectory else None
            student_command = self._lease_owner == 'native-student-transfer' and not self._stopping
            if self._forecast is not None and student_command:
                try:
                    self._forecast.command_started(command, trajectory)
                except Exception:
                    try:
                        self._forecast.invalidate('unavailable')
                    except Exception:
                        pass
        response = None
        try:
            response = self._controller.apply_action(request)
            return response
        finally:
            with self._lock:
                self._command = self._trajectory = None
            if self._forecast is not None:
                try:
                    self._forecast.command_finished(command['command_id'], response)
                except Exception:
                    pass

    def display_snapshot(self):
        """Read cached samples without creating or consuming an observation ID."""
        controller = self._controller
        # Resident status is read-only. Older residents without the cached
        # display field remain connected but cannot register or render an
        # invented pose; they are not restarted automatically.
        read_status = getattr(controller, 'display_status', controller.status)
        status = read_status()
        observation = deepcopy(status.get('display_observation'))
        if not isinstance(observation, dict):
            observation = None
        geometry = Geometry(tool_forward_mm=controller.config.tool_forward_mm,
                            tool_vertical_mm=controller.config.tool_vertical_mm)
        joints = observation.get('joints_deg') if observation else None
        with self._lock:
            command = deepcopy(self._command)
            trajectory = self._trajectory
            stopping = self._stopping
        path = []
        if (command and not stopping and command.get('status') not in
                ('retarget_interrupt_requested', 'retarget_interrupt_unconfirmed')
                and joints is not None and command['target_joints_deg'] is not None):
            if trajectory:
                # Select the nearest cached trajectory point, then show only its
                # nominal remainder. This is a preview, not measured motion.
                closest = min(range(len(trajectory)), key=lambda i: sum((a-b)**2 for a, b in zip(trajectory[i], joints)))
                remainder = trajectory[closest:]
                indexes = sorted({round(i * (len(remainder)-1) / 24) for i in range(25)})
                targets = [remainder[i] for i in indexes]
            else:
                target = command['target_joints_deg']
                targets = [[a+(b-a)*i/24 for a, b in zip(joints, target)] for i in range(25)]
            path = [list(geometry.forward(row)) for row in targets]
        source = ('nominal_waypoint_preview' if command and command.get('kind') == 'blended_joint_path'
                  else 'nominal_single_command_preview')
        forecast = self._forecast
        if forecast is not None:
            try:
                forecast = forecast.snapshot()
            except Exception:
                # A failed display-only copy must not make a healthy
                # controller look disconnected or require device recovery.
                try:
                    forecast.invalidate('unavailable')
                except Exception:
                    pass
                forecast = {'segments': [], 'complete': False, 'reason': 'unavailable',
                            'scope': 'current_transfer', 'basis_command_id': None}
        if (isinstance(forecast, dict) and forecast.get('reason') == 'unavailable'
                and forecast.get('basis_command_id') is None and command and not stopping
                and command.get('status') == 'in_progress'):
            # Bind a display failure to the command in this same snapshot so
            # the UI can distinguish unavailable from still computing.
            forecast = {**forecast, 'basis_command_id': command['command_id']}
        return {'observation': observation, 'controller': status, 'current_command': command,
                'remaining_target_path_mm': path,
                'path_source': source if path else None,
                'forecast': forecast}
