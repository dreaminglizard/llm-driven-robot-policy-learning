"""Deterministic virtual device. It never imports a vendor SDK or serial module."""
import time
import threading
import hashlib

from .device import Snapshot
from .errors import ControlError
from .kinematics import check_joints, quantize_joints
from .validation import number, vector


class MockDevice:
    def __init__(self, config):
        self.options = dict(config.mock)
        self._blended_motion_timeout = config.limits.motion_timeout
        self._blended_start_tolerance = config.limits.joint_tolerance_deg
        self._blended_plan_delay = number(self.options.get('blended_plan_delay', 0.),
                                          'blended_plan_delay', minimum=0., maximum=5.)
        if self.options.get("connect_hang"):
            while True:
                time.sleep(1)
        self.joints = check_joints(self.options.get("initial_joints_deg", [0, -10, -110, 30]))
        self._servo_target = check_joints(self.options.get('initial_virtual_servo_target_deg', self.joints))
        self.duration = number(self.options.get("move_duration", 0.15), "move_duration", minimum=0)
        self.sensor_period = number(self.options.get("sensor_period", 0), "sensor_period", minimum=0)
        self.residual = vector(self.options.get("residual_deg", [0, 0, 0, 0]), 4, "residual_deg")
        self.motion = None
        self.connected = True
        self.moving = False
        self.sequence = 0
        self.completion = 0
        self.received = time.monotonic()
        self.gripper = "open"
        self.moves = 0
        self.stops = 0
        self.cached_feedback = None
        self._motion_lock = threading.RLock()
        self._trajectory_stop = threading.Event()
        self._trajectory = None

    def _update(self):
        if self.motion is None:
            return
        start, target, begun = self.motion
        if self.options.get("disconnect_after_move"):
            self.connected = False
            return
        if self.options.get("no_start"):
            self.moving = False
            return
        if self.options.get("stall"):
            self.moving = True
            return
        fraction = 1.0 if self.duration == 0 else min(1.0, (time.monotonic() - begun) / self.duration)
        self.joints = tuple(a + (b - a) * fraction for a, b in zip(start, target))
        self.moving = fraction < 1.0
        if fraction == 1.0:
            self.joints = tuple(q + d for q, d in zip(target, self.residual))
            self.completion += 1
            self.motion = None

    def snapshot(self):
        if self.moves and self.options.get("snapshot_hang_after_move"):
            while True:
                time.sleep(1)
        with self._motion_lock:
            self._update()
            stale = self.options.get("stale") or self.moves and self.options.get("stale_after_move")
            if self.connected and not stale and (self.sequence == 0 or time.monotonic() - self.received >= self.sensor_period):
                self.sequence += 1
                self.received = time.monotonic()
                self.cached_feedback = (self.joints, self.moving, self.completion, self.gripper)
            joints, moving, completion, gripper = self.cached_feedback or (self.joints, self.moving, self.completion, self.gripper)
            reported = None if self.options.get("no_gripper_feedback") else gripper
            return Snapshot(
                self.connected, joints, self.sequence, self.received,
                moving, completion, reported,
                self.received if reported else None,
                {"open": 0, "closed": 1}.get(reported), self.stops, self.moves,
            )

    def move_joints(self, joints_deg, speed_pct):
        if self.options.get("move_rpc_hang"):
            while True:
                time.sleep(1)
        if not self.connected:
            raise ControlError("DISCONNECTED", "Mock device is disconnected")
        check_joints(joints_deg)
        steps, effective = quantize_joints(joints_deg)
        with self._motion_lock:
            self._update()
            self._trajectory = None
            self.moves += 1
            self.motion = (self.joints, effective, time.monotonic())
            self._servo_target = effective
            self.moving = not self.options.get("no_start", False)
        return {"steps": steps, "effective_target_deg": effective, "speed_pct": speed_pct}

    def move_joint_trajectory(self, command_id, targets_deg, period_seconds):
        """Apply unquantized simulated targets on a best-effort wall-clock schedule."""
        if not self.connected:
            raise ControlError("DISCONNECTED", "Mock device is disconnected")
        if any(self.options.get(name) for name in
               ("disconnect_after_move", "no_start", "stall", "move_rpc_hang", "residual_deg")):
            raise ControlError("MOCK_TRAJECTORY_FIXTURE_UNSUPPORTED", "This mock fault fixture has no trajectory implementation")
        targets = tuple(tuple(check_joints(row)) for row in targets_deg)
        with self._motion_lock:
            self._update()
            if self.moving or self.motion is not None:
                raise ControlError("BUSY", "Mock device is already moving")
            self._trajectory_stop = threading.Event()
            stop = self._trajectory_stop
            state = {"command_id": command_id, "started_monotonic": time.monotonic(),
                     "period_seconds": period_seconds, "samples": [], "completed": False}
            self._trajectory = state
            self.moves += 1
            self.moving = True

        def apply_targets():
            for tick, target in enumerate(targets, 1):
                deadline = state["started_monotonic"] + tick * period_seconds
                # Event.wait uses a coarse Windows wait on some versions. Use
                # short, interruptible sleep intervals and retain both the
                # planned and observed times; this is still not a real-time OS.
                while not stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.:
                        break
                    time.sleep(min(.001, remaining))
                if stop.is_set():
                    break
                with self._motion_lock:
                    if stop.is_set():
                        break
                    self.joints = target
                    self._servo_target = target
                    state["samples"].append({"tick": tick, "scheduled_monotonic": deadline,
                                             "applied_monotonic": time.monotonic(),
                                             "joints_deg": list(target)})
            with self._motion_lock:
                if self._trajectory is state:
                    if len(state["samples"]) == len(targets) and not stop.is_set():
                        state["completed"] = True
                        self.completion += 1
                    self.moving = False

        threading.Thread(target=apply_targets, name="mock-joint-trajectory", daemon=True).start()
        return {"command_id": command_id, "scheduled_ticks": len(targets),
                "period_seconds": period_seconds, "simulated": True}

    def trajectory_trace(self, command_id):
        with self._motion_lock:
            state = self._trajectory
            if state is None or state["command_id"] != command_id:
                return {"command_id": command_id, "available": False}
            return {"command_id": command_id, "available": True,
                    "started_monotonic": state["started_monotonic"],
                    "period_seconds": state["period_seconds"], "completed": state["completed"],
                    "samples": [dict(row, joints_deg=list(row["joints_deg"])) for row in state["samples"]]}

    def blended_start_reference(self):
        with self._motion_lock:
            self._update()
            if self.moving or self.motion is not None:
                raise ControlError('NOT_STATIONARY', 'A virtual blended start requires a stationary observation')
            return {'available': True, 'joints_deg': list(self._servo_target), 'source': 'virtual_servo_target'}

    def move_blended_joint_path(self, command_id, spec, expected_sha256, expected_count):
        """Apply one prechecked virtual path; no intermediate stop is issued."""
        from .blended_execution import (prepare_virtual_blend, planned_sample_digest, update_sample_digest,
                                        virtual_blend_time_budget)
        if not self.connected:
            raise ControlError('DISCONNECTED', 'Mock device is disconnected')
        if any(self.options.get(name) for name in
               ('disconnect_after_move', 'no_start', 'stall', 'move_rpc_hang', 'residual_deg')):
            raise ControlError('MOCK_TRAJECTORY_FIXTURE_UNSUPPORTED', 'This mock fault fixture has no blended path implementation')
        budget = virtual_blend_time_budget(self._blended_motion_timeout, spec.get('original_waypoint_count', 0))
        if (spec.get('segment_timeout_seconds') != self._blended_motion_timeout
                or spec.get('motion_timeout_seconds') != budget):
            raise ControlError('BLENDED_PLAN_MISMATCH', 'The virtual time budget differs from the worker configuration')
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)
                or type(expected_count) is not int or expected_count < 1
                or expected_count > int(budget / .002) + 1):
            raise ControlError('INVALID_REQUEST', 'Invalid planned virtual sample evidence')
        start_target = check_joints(spec.get('start_joints_deg'))
        admission = check_joints(spec.get('admission_observed_joints_deg'))
        final_target = check_joints(spec['waypoints_deg'][-1])
        steps, effective = quantize_joints(final_target)
        if any(abs(a - b) > 1e-10 for a, b in zip(effective, final_target)):
            raise ControlError('BLENDED_PLAN_MISMATCH', 'The final virtual target must be quantized')
        with self._motion_lock:
            self._update()
            if self.moving or self.motion is not None:
                raise ControlError('BUSY', 'Mock device is already moving')
            if max(abs(a-b) for a,b in zip(self._servo_target, start_target)) > 1e-7:
                raise ControlError('STATE_CHANGED', 'The virtual servo target changed since path admission')
            if max(abs(a-b) for a,b in zip(self.joints, admission)) > self._blended_start_tolerance:
                raise ControlError('STATE_CHANGED', 'The virtual observation changed since path admission')
            self._trajectory_stop = threading.Event()
            stop = self._trajectory_stop
            state = {'command_id': command_id, 'started_monotonic': None,
                     'sample_period_seconds': spec.get('sample_period_seconds'), 'duration_s': None,
                     'scheduled_count': expected_count, 'planned_sha256': expected_sha256,
                     'applied_count': 0, 'applied_sha256': hashlib.sha256().hexdigest(),
                     'completed': False, 'intermediate_stop_count': 0, 'stop_requested_during_path': False,
                     'plan_verified': False, 'worker_error': None,
                     'last_applied_joints_deg': None, 'last_applied_time_s': None,
                     'first_applied_monotonic': None, 'last_applied_monotonic': None,
                     'max_lateness_seconds': None, 'encoding': 'be-u64-tick-and-5-f64-v1'}
            self._trajectory = state
            self.moves += 1
            self.moving = True

        def apply_targets():
            # Planning may take longer than one sensor freshness window. It
            # runs outside both the worker RPC and the device state lock, so
            # snapshots and stop remain available while no target is sent.
            try:
                if self._blended_plan_delay and stop.wait(self._blended_plan_delay):
                    return
                path = prepare_virtual_blend(spec, motion_timeout=budget)
                planned = planned_sample_digest(path)
                if planned['sha256'] != expected_sha256 or planned['sample_count'] != expected_count:
                    raise ControlError('BLENDED_PLAN_MISMATCH', 'The worker reconstructed a different virtual path')
                with self._motion_lock:
                    if stop.is_set() or self._trajectory is not state:
                        return
                    if max(abs(a-b) for a,b in zip(self._servo_target, start_target)) > 1e-7:
                        raise ControlError('STATE_CHANGED', 'The virtual servo target changed during planning')
                    if max(abs(a-b) for a,b in zip(self.joints, admission)) > self._blended_start_tolerance:
                        raise ControlError('STATE_CHANGED', 'The virtual observation changed during planning')
                    state['plan_verified'] = True
                    state['duration_s'] = path.duration_s
                    state['started_monotonic'] = time.monotonic()
            except Exception as exc:
                with self._motion_lock:
                    if self._trajectory is state:
                        state['worker_error'] = exc.as_dict() if isinstance(exc, ControlError) else {
                            'code': 'DEVICE_ERROR', 'message': type(exc).__name__, 'details': {}}
                        self.moving = False
                return
            digest = hashlib.sha256()
            for tick, sample in enumerate(path.sample_ticks()):
                if tick == 0:
                    continue
                deadline = state['started_monotonic'] + sample.time_s
                while not stop.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    stop.wait(min(.01, remaining))
                with self._motion_lock:
                    if stop.is_set() or self._trajectory is not state:
                        break
                    applied_at = time.monotonic()
                    self.joints = sample.position_deg
                    self._servo_target = sample.position_deg
                    update_sample_digest(digest, tick, sample.time_s, self.joints)
                    state['applied_count'] = tick
                    state['applied_sha256'] = digest.hexdigest()
                    state['last_applied_joints_deg'] = list(self.joints)
                    state['last_applied_time_s'] = sample.time_s
                    if state['first_applied_monotonic'] is None:
                        state['first_applied_monotonic'] = applied_at
                    state['last_applied_monotonic'] = applied_at
                    lateness = applied_at - deadline
                    state['max_lateness_seconds'] = max(state['max_lateness_seconds'] or 0., lateness)
            with self._motion_lock:
                if self._trajectory is state:
                    if state['applied_count'] == expected_count and not stop.is_set():
                        state['completed'] = True
                        self.completion += 1
                    self.moving = False

        threading.Thread(target=apply_targets, name='mock-blended-joint-path', daemon=True).start()
        return {'command_id': command_id, 'scheduled_count': expected_count,
                'planned_sha256': expected_sha256, 'encoding': 'be-u64-tick-and-5-f64-v1',
                'steps': steps, 'effective_target_deg': effective, 'simulated': True,
                'plan_verification': 'pending_worker_reconstruction'}

    def blended_path_trace(self, command_id):
        with self._motion_lock:
            state = self._trajectory
            if state is None or state['command_id'] != command_id or 'planned_sha256' not in state:
                return {'command_id': command_id, 'available': False}
            return {'available': True, **state}

    def set_gripper(self, state):
        if state not in {"open", "closed"}:
            raise ControlError("INVALID_GRIPPER", "Expected open or closed")
        with self._motion_lock:
            self.gripper = state
        return {"command": state, "object_held_visual": None}

    def stop(self):
        with self._motion_lock:
            self._update()
            self.stops += 1
            if not self.options.get("stop_unconfirmed"):
                if (self._trajectory is not None and 'planned_sha256' in self._trajectory
                        and not self._trajectory['completed'] and self.moving):
                    self._trajectory['stop_requested_during_path'] = True
                self._trajectory_stop.set()
                self.motion = None
                self.moving = False
                self._servo_target = self.joints
        return {"requested": True, "connected": self.connected}

    def close(self):
        if self.options.get("close_hang"):
            while True:
                time.sleep(1)
        with self._motion_lock:
            self._trajectory_stop.set()
            self.connected = False
