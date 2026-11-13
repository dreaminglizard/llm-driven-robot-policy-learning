"""Version-guarded SDK access. Imported/constructed only inside the SDK worker.

The compatibility tweak is instance-local. No installed SDK files are edited.
"""
import hashlib
from collections import deque
import importlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import re
import tempfile
import threading
import time

from .blended_execution import (SDK_BLENDED_COMMAND_PERIOD_S, planned_sample_digest, prepare_virtual_blend,
                                sdk_blended_command_digest, sdk_blended_commands, update_sample_digest,
                                virtual_blend_time_budget)
from .device import Snapshot
from .errors import ControlError
from .hardware_policy import check_hardware_time
from .kinematics import Geometry, check_joints, distance, quantize_joints
from .locking import FileLock
from .motion_envelope import bounds as motion_bounds, check as check_envelope
from .validation import keys, number, vector

SDK_VERSION = "1.7.11"
SDK_STREAM_FRAME_TIMEOUT_S = .2
SDK_STREAM_MAX_LATENESS_S = .15
SDK_TRACKING_STALL_SECONDS = 2.
SDK_FEEDBACK_GAP_GRACE_SECONDS = 1.0
SDK_ENCODER_STEP_DEG = 360. / 4096.
SDK_FINISHED_AXIS_CORRECTION_LIMIT_DEG = 1.


class _TrackingDiagnostics:
    """Bounded host-only evidence; never used to decide or send a command."""
    def __init__(self):
        self._lock = threading.Lock()
        self._events = deque(maxlen=256)
        self._dropped = 0
        self._axes = [None] * 4
        self._delay_by_mask = {}
        self._wait_active = False

    def _append(self, event):
        if len(self._events) == self._events.maxlen:
            self._dropped += 1
        self._events.append(event)

    def record_axis(self, axis, detail, *, dynamic_reason, residual_reason):
        with self._lock:
            previous = self._axes[axis]
            self._axes[axis] = detail
            for name, reason in (('dynamic_eligible', dynamic_reason),
                                 ('dynamic_applied', dynamic_reason),
                                 ('residual_certificate', residual_reason),
                                 ('residual_applied', residual_reason)):
                before = bool(previous and previous[name])
                after = detail[name]
                if before != after:
                    if name == 'dynamic_applied':
                        reason = ('used_for_pacing' if after else
                                  'no_longer_needed' if detail['dynamic_eligible'] else dynamic_reason)
                    elif name == 'residual_applied':
                        reason = ('baseline_subtracted_for_pacing' if after else
                                  'additional_lag_above_tolerance' if detail['residual_certificate']
                                  else residual_reason)
                    self._append({'kind': 'certificate_transition', 'axis': axis + 1,
                                  'certificate': name, 'active': after, 'reason': reason,
                                  'monotonic': detail['monotonic'],
                                  'sensor_sequence': detail['sensor_sequence'],
                                  'sdk_encoder_callback_monotonic': detail['sdk_encoder_callback_monotonic'],
                                  'raw_lag_deg': detail['raw_lag_deg'],
                                  'residual_adjusted_lag_deg': detail['residual_adjusted_lag_deg'],
                                  'pacing_lag_deg': detail['pacing_lag_deg'],
                                  'watchdog_reference_before_update_deg': detail['watchdog_reference_before_update_deg'],
                                  'watchdog_best_before_update_deg': detail['watchdog_best_before_update_deg'],
                                  'watchdog_best_time_before_update_monotonic':
                                      detail['watchdog_best_time_before_update_monotonic']})

    def record_wait(self, start, end, delay, band_axes, tolerance):
        with self._lock:
            lag_axes = [axis + 1 for axis, detail in enumerate(self._axes)
                        if detail is not None and detail['pacing_lag_deg'] > tolerance]
            blocked = sorted(set(lag_axes).union(band_axes))
            mask = sum(1 << (axis - 1) for axis in blocked)
            self._delay_by_mask[mask] = self._delay_by_mask.get(mask, 0.) + delay
            detail = {'kind': 'feedback_wait', 'start_monotonic': start, 'end_monotonic': end,
                      'delay_seconds': delay, 'blocked_axes': blocked,
                      'lag_blocked_axes': lag_axes, 'remaining_band_blocked_axes': list(band_axes),
                      'wait_count': 1,
                      'axes': [dict(axis) if axis is not None else None for axis in self._axes]}
            # Keep the most recent observation for a continuous identical wait;
            # transitions still get their own entries in the bounded ring.
            previous = self._events[-1] if self._events else None
            if (self._wait_active and previous is not None and previous['kind'] == 'feedback_wait'
                    and previous['blocked_axes'] == detail['blocked_axes']
                    and previous['lag_blocked_axes'] == detail['lag_blocked_axes']
                    and previous['remaining_band_blocked_axes'] == detail['remaining_band_blocked_axes']):
                previous['end_monotonic'] = end
                previous['delay_seconds'] += delay
                previous['wait_count'] += 1
                previous['axes'] = detail['axes']
            else:
                self._append(detail)
            self._wait_active = True

    def end_wait(self):
        with self._lock:
            self._wait_active = False

    def record_error(self, error):
        with self._lock:
            self._append({'kind': 'error', 'error': error,
                          'axes': [dict(axis) if axis is not None else None for axis in self._axes]})

    def snapshot(self):
        with self._lock:
            # The worker never returns this mutable ring while it is changing.
            events = []
            for event in self._events:
                row = dict(event)
                if 'axes' in row:
                    row['axes'] = [dict(axis) if axis is not None else None for axis in row['axes']]
                events.append(row)
            groups = [{'blocked_axes': [axis + 1 for axis in range(4) if mask & (1 << axis)],
                       'delay_seconds': delay} for mask, delay in sorted(self._delay_by_mask.items())]
            return {'capacity': self._events.maxlen, 'dropped_events': self._dropped,
                    'feedback_delay_by_blocked_axes': groups, 'events': events,
                    'scope': 'host_tracking_diagnostics_not_device_application',
                    'sensor_time_source': 'host SDK encoder callback; not device acquisition time'}


class _SDKTrackingProgress:
    """Bound planned-time advance using measured joint tracking.

    This is a host-side pacing rule, not an estimate of radio delivery or a
    claim that the physical arm follows the planned trajectory.
    """
    def __init__(self, limits, *, allow_stable_finished_axis=False, stable_axis_mask=None, diagnostics=None):
        self.tolerance = limits.joint_tolerance_deg
        self.stale_seconds = limits.sensor_max_age
        self.stable_seconds = limits.stable_seconds
        self.allow_stable_finished_axis = allow_stable_finished_axis
        if stable_axis_mask is None:
            stable_axis_mask = (True,) * 4
        if (not isinstance(stable_axis_mask, (list, tuple)) or len(stable_axis_mask) != 4
                or any(type(value) is not bool for value in stable_axis_mask)):
            raise ValueError('stable axis mask needs four booleans')
        self.stable_axis_mask = tuple(stable_axis_mask)
        self.references = [None] * 4
        self.directions = [None] * 4
        self.best_errors = [None] * 4
        self.best_times = [None] * 4
        self.stable_windows = [deque() for _ in range(4)]
        self.stable_window_directions = [None] * 4
        self.stable_certificates = [None] * 4
        self.dynamic_windows = [deque() for _ in range(4)]
        self.dynamic_window_directions = [None] * 4
        self.dynamic_following_axes = [False] * 4
        self.used_dynamic_following_axes = [False] * 4
        self.stable_finished_axes = [False] * 4
        self.stable_residual_axes = [False] * 4
        self.used_stable_residual_axes = [False] * 4
        self.confirmed_residuals = [None] * 4
        self.max_additional_lags = [0.] * 4
        self.max_errors = [0.] * 4
        self.max_lag_errors = [0.] * 4
        self.diagnostics = diagnostics if diagnostics is not None else _TrackingDiagnostics()
        self._dynamic_reason = ['not_evaluated'] * 4
        self._residual_reason = ['not_evaluated'] * 4

    def _dynamic_following(self, axis, direction, error, actual, reference,
                           sensor_sequence, sensor_monotonic):
        """Recognize fresh, real joint motion following a moving reference."""
        window = self.dynamic_windows[axis]
        reset_reason = None
        if self.dynamic_window_directions[axis] != direction:
            window.clear()
            self.dynamic_window_directions[axis] = direction
            reset_reason = 'direction_changed'
        if direction == 0:
            window.clear()
            self._dynamic_reason[axis] = 'no_remaining_direction'
            return False
        if window:
            previous_sequence, previous_time, _, _, _ = window[-1]
            if (sensor_sequence < previous_sequence or sensor_monotonic < previous_time
                    or sensor_monotonic-previous_time > self.stale_seconds+1e-9
                    or (sensor_sequence == previous_sequence and sensor_monotonic != previous_time)):
                window.clear()
                reset_reason = 'sensor_order_or_age'
        sample = (sensor_sequence, sensor_monotonic, actual, reference, error)
        if window and sensor_sequence == window[-1][0]:
            window[-1] = sample
        else:
            window.append(sample)
        while len(window) > 2 and sensor_monotonic-window[1][1] >= self.stable_seconds:
            window.popleft()
        if len(window) < 2 or sensor_monotonic-window[0][1] < self.stable_seconds:
            self._dynamic_reason[axis] = reset_reason or 'observation_window_incomplete'
            return False
        joint_deltas = [(b[2]-a[2])*direction for a, b in zip(window, list(window)[1:])]
        if any(delta < -1e-9 for delta in joint_deltas):
            self._dynamic_reason[axis] = 'observed_reverse'
            return False
        if (window[-1][2]-window[0][2])*direction < SDK_ENCODER_STEP_DEG-1e-9:
            self._dynamic_reason[axis] = 'observed_motion_below_one_step'
            return False
        if (window[-1][3]-window[0][3])*direction <= 1e-9:
            self._dynamic_reason[axis] = 'reference_not_advanced'
            return False
        if error > SDK_FINISHED_AXIS_CORRECTION_LIMIT_DEG:
            self._dynamic_reason[axis] = 'lag_exceeds_existing_correction_limit'
            return False
        errors = [item[4] for item in window]
        changes = [b-a for a, b in zip(errors, errors[1:])]
        continuously_worsening = (sum(change > 1e-9 for change in changes) >= 2
                                  and not any(change < -1e-9 for change in changes))
        if continuously_worsening:
            self._dynamic_reason[axis] = 'lag_continuously_worsening'
            return False
        if error-min(errors) > self.tolerance+1e-9:
            self._dynamic_reason[axis] = 'lag_growth_exceeds_tolerance'
            return False
        self._dynamic_reason[axis] = 'fresh_motion_following'
        return True

    def _stable_residual(self, axis, direction, error, actual, reference,
                         sensor_sequence, sensor_monotonic):
        """Recognize a non-growing residual using distinct fresh encoder samples.

        Compare tracking error, rather than the joint position or individual
        reference, so a joint that follows a moving target with a stable load
        offset is not required to stop at every quantized target.  One encoder
        step of variation accounts for quantization, but a rising residual at
        the ends of the observation window cannot qualify as stable.
        """
        window = self.stable_windows[axis]
        certificate = self.stable_certificates[axis]
        reset_reason = None
        if certificate is not None:
            out_of_order = (sensor_sequence < certificate['sequence']
                            or sensor_monotonic < certificate['sensor_monotonic']
                            or sensor_monotonic-certificate['sensor_monotonic'] > self.stale_seconds+1e-9
                            or (sensor_sequence == certificate['sequence']
                                and sensor_monotonic != certificate['sensor_monotonic']))
            reversed_motion = (direction != 0
                               and (actual-certificate['actual'])*direction < -1e-9)
            worsening_finished_axis = (direction == 0 and reference == certificate['reference']
                                      and error > certificate['error']+1e-9)
            if (out_of_order or direction != certificate['direction']
                    or reversed_motion or worsening_finished_axis):
                self._residual_reason[axis] = ('sensor_order_or_age' if out_of_order else
                    'direction_changed' if direction != certificate['direction'] else
                    'observed_reverse' if reversed_motion else 'finished_axis_error_increased')
                reset_reason = self._residual_reason[axis]
                self.stable_certificates[axis] = None
                window.clear()
            else:
                certificate.update(actual=actual, reference=reference, error=error,
                                   sequence=sensor_sequence, sensor_monotonic=sensor_monotonic)
                self._residual_reason[axis] = 'certificate_retained'
                return True
        if self.stable_window_directions[axis] != direction:
            window.clear()
            self.stable_window_directions[axis] = direction
            reset_reason = reset_reason or 'direction_changed'
        if window:
            previous_sequence, previous_time, _ = window[-1]
            if (sensor_sequence < previous_sequence or sensor_monotonic < previous_time
                    or sensor_monotonic-previous_time > self.stale_seconds+1e-9
                    or (sensor_sequence == previous_sequence and sensor_monotonic != previous_time)):
                window.clear()
                reset_reason = reset_reason or 'sensor_order_or_age'
        sample = (sensor_sequence, sensor_monotonic, error)
        if window and sensor_sequence == window[-1][0]:
            # A new encoded reference can change the residual without a new
            # observation. Replace, but never count it as another sample.
            window[-1] = sample
        else:
            window.append(sample)
        while len(window) > 2 and sensor_monotonic-window[1][1] >= self.stable_seconds:
            window.popleft()
        if len(window) < 2 or sensor_monotonic-window[0][1] < self.stable_seconds:
            self._residual_reason[axis] = reset_reason or 'observation_window_incomplete'
            return False
        errors = [item[2] for item in window]
        epsilon = 1e-9
        spread = max(errors)-min(errors)
        stable = (spread <= SDK_ENCODER_STEP_DEG + epsilon and errors[-1] <= errors[0] + epsilon)
        if stable:
            self.stable_certificates[axis] = {
                'direction': direction, 'actual': actual, 'reference': reference,
                'error': error, 'baseline_error': error,
                'sequence': sensor_sequence, 'sensor_monotonic': sensor_monotonic}
            self._residual_reason[axis] = 'stable_observation_window'
        else:
            self._residual_reason[axis] = ('error_range_exceeds_one_encoder_step'
                if spread > SDK_ENCODER_STEP_DEG + epsilon else 'error_increased')
        return stable

    def rate(self, observed, encoded_target, now, directional_targets=None, *,
             sensor_sequence=None, sensor_monotonic=None, proposed_target=None,
             final_damped_target_encoded=False, defer_stall_to_total_progress=False):
        if encoded_target is None:
            return 1.
        observed, encoded_target = check_joints(observed), check_joints(encoded_target)
        if directional_targets is None:
            directional_targets = (None,) * 4
        if not isinstance(directional_targets, (list, tuple)) or len(directional_targets) != 4:
            raise ControlError('INVALID_VECTOR', 'Tracking direction needs four planned targets')
        if proposed_target is not None:
            proposed_target = check_joints(proposed_target)
        errors = []
        for axis, (actual, reference, upcoming) in enumerate(zip(observed, encoded_target, directional_targets)):
            absolute_error = abs(actual-reference)
            self.max_errors[axis] = max(self.max_errors[axis], absolute_error)
            if upcoming is None:
                direction, error = 0, absolute_error
            else:
                upcoming = number(upcoming, 'directional_target')
                direction = (upcoming > reference) - (upcoming < reference)
                error = max(0., (reference-actual)*direction) if direction else absolute_error
            final_damped_watch = final_damped_target_encoded is True and direction == 0
            self.max_lag_errors[axis] = max(self.max_lag_errors[axis], error)
            # Compare held-target progress in one coordinate system even if a
            # stable residual certificate is acquired or revoked.  Residual
            # subtraction is used below only to pace the next path sample.
            watchdog_error = error
            self.stable_finished_axes[axis] = False
            self.stable_residual_axes[axis] = False
            self.dynamic_following_axes[axis] = False
            self.confirmed_residuals[axis] = None
            valid_sample = (type(sensor_sequence) is int and type(sensor_monotonic) in (int, float)
                            and math.isfinite(sensor_monotonic))
            dynamic = (self._dynamic_following(axis, direction, error, actual, reference,
                                               sensor_sequence, sensor_monotonic)
                       if valid_sample else False)
            if not valid_sample:
                self.dynamic_windows[axis].clear()
                self.dynamic_window_directions[axis] = None
                self._dynamic_reason[axis] = 'invalid_sensor_sample'
            certificate = self.stable_certificates[axis]
            established = (certificate is not None and direction != 0)
            if (not final_damped_watch and self.allow_stable_finished_axis and self.stable_axis_mask[axis]
                    and self.tolerance < error
                    and (absolute_error <= SDK_FINISHED_AXIS_CORRECTION_LIMIT_DEG or established)
                    and valid_sample):
                stable = self._stable_residual(axis, direction, error, actual, reference,
                                               sensor_sequence, sensor_monotonic)
                if stable:
                    baseline = self.stable_certificates[axis]['baseline_error']
                    self.confirmed_residuals[axis] = baseline
                    additional_lag = max(0., error-baseline) if direction else 0.
                    self.max_additional_lags[axis] = max(self.max_additional_lags[axis], additional_lag)
                    # The correction limit bounds the independently observed
                    # baseline, not each following trajectory increment.
                    # Extra lag after that baseline uses the original joint
                    # tolerance. Final arrival/correction remain separate.
                    error = additional_lag
                    if error <= self.tolerance:
                        self.stable_residual_axes[axis] = True
                        self.used_stable_residual_axes[axis] = True
                        self.stable_finished_axes[axis] = direction == 0
            else:
                self.stable_windows[axis].clear()
                self.stable_window_directions[axis] = None
                self.stable_certificates[axis] = None
                self._residual_reason[axis] = ('correction_disabled' if not self.allow_stable_finished_axis else
                    'axis_not_stationary' if not self.stable_axis_mask[axis] else
                    'raw_lag_within_tolerance' if error <= self.tolerance else
                    'invalid_sensor_sample' if not valid_sample else 'outside_existing_correction_limit')
            residual_adjusted_error = error
            if self.tolerance < error and dynamic:
                # This permits continued path progress only; it does not
                # alter the Controller's final arrival checks.
                error = 0.
                self.dynamic_following_axes[axis] = True
                self.used_dynamic_following_axes[axis] = True
            errors.append(error)
            self.diagnostics.record_axis(axis, {
                'axis': axis + 1, 'monotonic': now, 'sensor_sequence': sensor_sequence,
                'sdk_encoder_callback_monotonic': sensor_monotonic if valid_sample else None,
                'observed_deg': actual, 'encoded_target_deg': reference, 'direction': direction,
                'absolute_error_deg': absolute_error, 'raw_lag_deg': watchdog_error,
                'residual_adjusted_lag_deg': residual_adjusted_error, 'pacing_lag_deg': error,
                'residual_baseline_deg': self.confirmed_residuals[axis],
                'watchdog_reference_before_update_deg': self.references[axis],
                'watchdog_direction_before_update': self.directions[axis],
                'watchdog_best_before_update_deg': self.best_errors[axis],
                'watchdog_best_time_before_update_monotonic': self.best_times[axis],
                'dynamic_eligible': dynamic, 'dynamic_applied': self.dynamic_following_axes[axis],
                'residual_certificate': self.stable_certificates[axis] is not None,
                'residual_applied': self.stable_residual_axes[axis],
                'dynamic_reason': self._dynamic_reason[axis],
                'residual_reason': self._residual_reason[axis],
            }, dynamic_reason=self._dynamic_reason[axis], residual_reason=self._residual_reason[axis])
            if error <= self.tolerance:
                self.references[axis] = reference
                self.directions[axis] = direction
                self.best_errors[axis] = None
                self.best_times[axis] = None
                continue
            if self.references[axis] != reference or self.directions[axis] != direction:
                self.references[axis] = reference
                self.directions[axis] = direction
                self.best_errors[axis] = watchdog_error
                self.best_times[axis] = now
                continue
            best, since = self.best_errors[axis], self.best_times[axis]
            if best is None:
                self.best_errors[axis], self.best_times[axis] = watchdog_error, now
            elif watchdog_error <= best - SDK_ENCODER_STEP_DEG:
                self.best_errors[axis], self.best_times[axis] = watchdog_error, now
            elif watchdog_error >= best + SDK_ENCODER_STEP_DEG and now-since >= self.stale_seconds:
                # A correction-enabled path may pause through a small
                # transient within the existing correction bound. No next
                # target is sent unless fresh feedback becomes stable again.
                one_step_at_final = (final_damped_watch and valid_sample
                    and watchdog_error == best + SDK_ENCODER_STEP_DEG
                    and absolute_error <= SDK_FINISHED_AXIS_CORRECTION_LIMIT_DEG)
                legacy_bounded_stationary = (not final_damped_watch and self.allow_stable_finished_axis
                    and self.stable_axis_mask[axis] and absolute_error <= SDK_FINISHED_AXIS_CORRECTION_LIMIT_DEG)
                if not (one_step_at_final or legacy_bounded_stationary):
                    raise ControlError('SDK_TRACKING_DIVERGED',
                        'An observed joint continued away from the held encoded target',
                        joint=axis+1, error_deg=watchdog_error, best_error_deg=best)
                if now-since >= SDK_TRACKING_STALL_SECONDS and not defer_stall_to_total_progress:
                    raise ControlError('SDK_TRACKING_STALLED',
                        'An observed joint did not settle within the original tracking deadline',
                        joint=axis+1, error_deg=watchdog_error)
            elif now-since >= SDK_TRACKING_STALL_SECONDS and not defer_stall_to_total_progress:
                raise ControlError('SDK_TRACKING_STALLED',
                    'An observed joint did not approach the held encoded target',
                    joint=axis+1, error_deg=watchdog_error)
        # Slow for lag behind the next planned direction. An axis ahead of its
        # old target need not reverse before the plan can approach it; an axis
        # with no remaining movement retains the absolute-error check.
        # The exact tolerance is accepted, as in the arrival check. All
        # measured-to-next paths still use the original envelope and deadline.
        return 0. if max(errors) > self.tolerance else 1.


def _next_directional_targets(commands, first_index, encoded_target):
    """Look past unchanged quantized axes to their next planned direction."""
    if encoded_target is None:
        return None
    found = [None] * 4
    for _, target in commands[first_index:]:
        for axis in range(4):
            if found[axis] is None and target[axis] != encoded_target[axis]:
                found[axis] = target[axis]
        if all(value is not None for value in found):
            break
    return found


def _remaining_planned_bounds(commands):
    """Bounds for each remaining suffix; these contain no observed object data."""
    remaining = [None] * len(commands)
    lower, upper = [math.inf] * 4, [-math.inf] * 4
    for index in range(len(commands)-1, -1, -1):
        target = commands[index][1]
        for axis in range(4):
            lower[axis] = min(lower[axis], target[axis])
            upper[axis] = max(upper[axis], target[axis])
        remaining[index] = (tuple(lower), tuple(upper))
    return remaining


def _check_remaining_tracking_bounds(observed, encoded_target, remaining, allowed_error,
                                     directional_targets=None, behind_allowances=None, blocked_axes=None):
    """Refuse another target outside the path band; distinguish lag from overshoot."""
    if encoded_target is None:
        return True
    lower, upper = remaining
    inside = True
    for axis, (actual, reference, low, high) in enumerate(zip(observed, encoded_target, lower, upper)):
        low, high = min(low, reference), max(high, reference)
        if actual < low-allowed_error-1e-9 or actual > high+allowed_error+1e-9:
            if directional_targets is not None:
                upcoming = directional_targets[axis]
                direction = 0 if upcoming is None else ((upcoming > reference) - (upcoming < reference))
                if not direction or (reference-actual)*direction > 0.:
                    if (behind_allowances is not None and direction
                            and low-behind_allowances[axis]-1e-9 <= actual <= high+behind_allowances[axis]+1e-9):
                        continue
                    # Behind the last target: hold it while the existing
                    # tracking watchdog allows feedback to catch up.  Never
                    # dispatch a further target while outside the band.
                    inside = False
                    if blocked_axes is not None:
                        blocked_axes.append(axis + 1)
                    continue
            raise ControlError('SDK_TRACKING_DIVERGED',
                'An observed joint left the remaining planned range', joint=axis+1,
                observed_deg=actual, remaining_min_deg=low, remaining_max_deg=high,
                allowed_error_deg=allowed_error)
    return inside

SDK_HASHES = {
    "__init__.py": "57bc07d10808b63b397299e0b81a3b155c0ff1ebd72824005ac2da3763bba1e5",
    "raccoon.py": "7f67994c2047623508db97f85ac7c67a10d2b7527f370a3ae020095c07a332be",
    "raccoon_roboid.py": "58e2abea6fa8fabcab87ddc057f6b78a41b64c6b8fd5bd5afa6108fb3312353d",
    "model.py": "6d2825c9c85fe6612ea0cd334661ed58918b8eb8653de114bc6d93cd9cbcece7",
    "runner.py": "0c8489ce5efbe12b77efb14ed412c30b1619df8558418661f6b8d2d425eb4232",
    "serial_connector.py": "9a087e0550575351e2b8cd206cadd778f0d0e1d6e3021022fdc2f922e149f622",
}


def inspect_sdk():
    """Read distribution metadata/source hashes without importing roboid."""
    try:
        version = importlib.metadata.version("roboid")
        spec = importlib.util.find_spec("roboid")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ControlError("SDK_MISSING", "roboid 1.7.11 is not installed") from exc
    if spec is None or spec.origin is None:
        raise ControlError("SDK_MISSING", "roboid module cannot be resolved")
    root = Path(spec.origin).resolve().parent
    hashes = {}
    for name, expected in SDK_HASHES.items():
        path = root / name
        if not path.is_file():
            raise ControlError("SDK_MISMATCH", "SDK source file is missing", file=name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = actual
        if actual != expected:
            raise ControlError("SDK_MISMATCH", "SDK source fingerprint differs from the inspected version", file=name)
    if version != SDK_VERSION:
        raise ControlError("SDK_MISMATCH", "Unsupported SDK version", actual=version, expected=SDK_VERSION)
    return {"version": version, "root": str(root), "hashes": hashes, "hardware_connected": False}


class _SDKPeerBoundary:
    """Bind this SDK instance to its first checked transport address.

    This validates addresses, not device authenticity. It must be installed
    before the vendor starts its receive thread, including at construction.
    """
    def __init__(self, core):
        self.core = core
        self._lock = threading.Lock()
        self._connector = None
        self._address = None
        self._validated = False
        # A vendor instance keeps the last motor and gripper output in memory.
        # Once its established connection is lost it must not resume sending
        # those old outputs on the first packet from a returning device. The
        # application replaces this instance with a fresh controller instead.
        self._retired_after_disconnect = False
        self._receive = core._receive
        core._receive = self.receive
        core.is_connected = self.is_connected

    @staticmethod
    def _address_or_none(value):
        if not isinstance(value, str) or re.fullmatch(r'[0-9a-fA-F]{12}', value) is None:
            return None
        address = value.upper()
        return None if address == '000000000000' else address

    def is_connected(self):
        with self._lock:
            connector = self._connector
            return bool(connector is not None and connector is self.core._connector
                        and not self._retired_after_disconnect
                        and self._validated and connector.is_connected())

    def receive(self, connector):
        if connector is None:
            return False
        with self._lock:
            if self._retired_after_disconnect:
                if getattr(self.core, '_running', True) is False:
                    self.core._releasing = 0
                return False
            if self._connector is None:
                self._connector = connector
            if connector is not self._connector:
                self._validated = False
                return False
            # In the pinned connector, a successful initial handshake sets
            # _found and _address before the serial handle becomes readable.
            if (self._address is None and connector._serial is not None
                    and connector._found and connector.is_connected()):
                self._address = self._address_or_none(connector.get_address())
            if not connector.is_connected():
                # read() can set connected before it returns any packets.
                if self._validated:
                    self._retired_after_disconnect = True
                self._validated = False
                if self._retired_after_disconnect:
                    return False

        packets = connector.read()
        with self._lock:
            if not packets:
                if not connector.is_connected():
                    if self._validated:
                        self._retired_after_disconnect = True
                    self._validated = False
                    # The vendor release thread waits for five received
                    # packets. A lost connection cannot drain those sends.
                    if getattr(self.core, '_running', True) is False:
                        self.core._releasing = 0
                return False
            # For a bridge initially found without a robot, read() performs
            # the first SDK handshake before returning its first packet batch.
            if self._address is None and connector._found and connector.is_connected():
                self._address = self._address_or_none(connector.get_address())
            expected = self._address
            # A malformed or unrelated packet does not establish a new
            # device and must not retire a previously healthy connection.
            # Only pass correctly framed packets from the bound address to
            # the decoder; the controller still checks the age of the last
            # accepted observation before using it.
            accepted = [packet for packet in packets
                        if (expected is not None and connector.is_connected()
                            and isinstance(packet, str) and len(packet) == 54
                            and packet[40] == '-' and packet[53] == '\r'
                            and packet[0] == '1'
                            and re.fullmatch(r'[0-9a-fA-F]{40}', packet[:40]) is not None
                            and self._address_or_none(packet[41:53]) == expected)]
            if not accepted:
                if expected is not None:
                    connector._address = expected
                if getattr(self.core, '_running', True) is False:
                    self.core._releasing = 0
                return False
            # The vendor replaces its address on reconnect; keep the first
            # handshake address for every future outgoing packet.
            connector._address = expected

        class CheckedPackets:
            def read(self):
                return accepted

        received = self._receive(CheckedPackets())
        with self._lock:
            # Do not expose a recovered connection until the vendor decoder
            # and encoder callbacks have delivered the new sample.
            self._validated = bool(received and connector.is_connected())
        return received and self._validated


def managed_class(base):
    class ManagedRaccoon(base):
        def __init__(self, index=0, port_name=None):
            if base._robots.get(index) is not None:
                raise ControlError("INDEX_ALREADY_OWNED", "Do not dispose an existing robot by reconstructing its index")
            self._suppress_auto_init_pose = True
            try:
                super().__init__(index, port_name=port_name)
            finally:
                self._suppress_auto_init_pose = False

        def init_pose(self, wait=True):
            if getattr(self, "_suppress_auto_init_pose", False):
                return
            raise ControlError("UNSAFE_SDK_OPERATION", "Use the controller's checked low-speed start motion")

        def reset(self):
            raise ControlError("UNSAFE_SDK_OPERATION", "SDK reset includes an implicit blocking motion")

        def _add_roboid(self, roboid):
            # Raccoon._init calls this before Runner starts the receive thread.
            self._peer_boundary = _SDKPeerBoundary(roboid)
            return super()._add_roboid(roboid)

        def dispose(self):
            # The pinned Roboid._dispose resets motor/peripheral buffers BEFORE
            # RaccoonRoboid._release sends its final five packets. Close the
            # transport first, retaining the stopped arm and gripper command;
            # only then let the SDK clear its in-memory state and registration.
            core = self._roboid
            if not core._is_disposed():
                core._release()
            super().dispose()

    return ManagedRaccoon


def enable_float_targets(robot, sdk_model, raccoon_type):
    device = getattr(robot._roboid, "_joint_angle_device", None)
    if (device is None or device.get_data_size() != 4
            or device.get_device_type() != sdk_model.DeviceType.COMMAND
            or device.get_data_type() != sdk_model.DataType.INTEGER
            or device.get_id() != (raccoon_type.JOINT_ANGLE & 0xfff00fff)
            or device._min_value != -180 or device._max_value != 179):
        raise ControlError("SDK_STRUCTURE_MISMATCH", "Unexpected JOINT_ANGLE device structure")
    device._data_type = sdk_model.DataType.FLOAT
    return device


class SDKDevice:
    def __init__(self, config):
        if config.backend != "sdk" or not config.hardware_enabled:
            raise ControlError("HARDWARE_DISABLED", "SDK construction requires explicit hardware configuration")
        check_hardware_time()
        self.metadata = inspect_sdk()
        self.lock = FileLock(Path(tempfile.gettempdir()) / "raccoon-control-port-locks" / f"{config.port}.lock").acquire()
        self.robot = None
        self._lock = threading.Lock()
        self.sequence = 0
        self.sensor_time = None
        self.joints = None
        self.completion = 0
        self.gripper_time = None
        self.gripper_raw = None
        self.gripper_module_type = None
        self.gripper_slot_raw = None
        self.collision_event_counts = [0,0,0,0]
        self.motor_frames = deque(maxlen=32)
        self.sensor_packets = deque(maxlen=256)
        self.serial_writes = deque(maxlen=256)
        self._serial_diagnostics_lock = threading.Lock()
        self._serial_diagnostics_port = None
        self._serial_diagnostics_status = 'not_observed'
        self._serial_send_context = threading.local()
        self._last_motor_header = None
        self._initialize_blended_stream()
        self.moves = 0
        self.gripper_commands = 0
        self.stops = 0
        self.config = config
        # Import only here: roboid changes SIGINT handling during import.
        sdk = importlib.import_module("roboid.raccoon")
        sdk_model = importlib.import_module("roboid.model")
        # roboid 1.7.11 declares a connection lost after only 100 ms without
        # serial input. Under normal host scheduling jitter this can recycle a
        # healthy controller repeatedly. This SDK runs in its own worker;
        # allow a short receive gap before treating it as a broken transport.
        # The separate, shorter sensor-age requirement for actual commands is
        # unchanged.
        sdk_connector = importlib.import_module("roboid.serial_connector")
        sdk_connector.TIMEOUT = max(float(sdk_connector.TIMEOUT), 1.0)
        if Path(sdk.__file__).resolve().parent != Path(self.metadata["root"]):
            raise ControlError("SDK_MISMATCH", "Loaded module does not match the inspected source path")
        self.sdk_type = sdk.Raccoon
        robot_type = managed_class(self.sdk_type)
        self.robot = robot_type(0, port_name=config.port)
        self._record_motor_frames()
        self._install_stream_fences()
        self._record_sensor_packets()
        enable_float_targets(self.robot, sdk_model, self.sdk_type)
        self.robot.unit("cm")
        self.robot.end_effector_to(config.tool_forward_mm / 10, config.tool_vertical_mm / 10)
        self.robot.precision_mode(False)
        self.robot.add_device_data_changed_listener(self._on_data)

    def _record_motor_frames(self):
        """Record the pinned encoder's output without modifying any packet."""
        core=self.robot._roboid
        original=core._encode_motoring_packet
        def encode(address):
            packet=original(address)
            self._record_encoded_motor_frame(packet)
            return packet
        core._encode_motoring_packet=encode

    def _initialize_blended_stream(self):
        """Initialize host-only bookkeeping; does not construct or query a robot."""
        self._stream_send_lock = threading.Lock()
        self._sdk_runner_lock = threading.RLock()
        self._sdk_wire_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._motor_frame_sequence = 0
        self._latest_motor_frame = None
        self._target_frame_signature = None
        self._target_frame_segment_start = None
        self._blended_state = None
        self._blended_cancel = threading.Event()
        self._blended_thread = None
        self._blended_diagnostics = None
        self._stop_epoch = 0

    def _install_stream_fences(self):
        """Fence the pinned SDK's instance-local Runner and send boundaries."""
        core = self.robot._roboid
        request_original = core._request_motoring_data
        send_original = core._send

        def request():
            # Includes _clear_written(), which is outside the vendor's own
            # _thread_lock and would otherwise race with four-angle writes.
            with self._sdk_runner_lock:
                return request_original()

        def send(connector):
            # Covers SDK encode + connector.write as one ordering unit. This
            # does not acknowledge delivery beyond the host connector.
            with self._sdk_wire_lock:
                self._instrument_serial_write(connector)
                self._serial_send_context.active = True
                try:
                    return send_original(connector)
                finally:
                    self._serial_send_context.active = False

        core._request_motoring_data = request
        core._send = send

    def _instrument_serial_write(self, connector):
        """Observe the instance's serial return/exception without changing it."""
        try:
            port = getattr(connector, '_serial', None)
            if port is None:
                return
            if port is self._serial_diagnostics_port:
                return
            original_write = port.write

            def record(data, written, error):
                if not getattr(self._serial_send_context, 'active', False):
                    return
                try:
                    frame = self._latest_motor_frame
                    event = {'monotonic': time.monotonic(),
                             'frame_sequence': frame.get('frame_sequence') if frame else None,
                             'attempted_bytes': len(data), 'written_bytes': written,
                             'exception_type': type(error).__name__ if error is not None else None,
                             'exception_message': str(error)[:512] if error is not None else None}
                    with self._serial_diagnostics_lock:
                        self.serial_writes.append(event)
                except Exception:
                    # Diagnostic bookkeeping never changes SDK I/O behavior.
                    pass

            def write(data, *args, **kwargs):
                try:
                    written = original_write(data, *args, **kwargs)
                except BaseException as error:
                    record(data, None, error)
                    raise
                record(data, written, None)
                return written

            try:
                port.write = write
            except Exception:
                with self._serial_diagnostics_lock:
                    self._serial_diagnostics_status = 'unavailable'
            else:
                with self._serial_diagnostics_lock:
                    self._serial_diagnostics_status = 'active'
            self._serial_diagnostics_port = port
        except Exception:
            # Failure to attach diagnostics does not block a vendor send.
            with self._serial_diagnostics_lock:
                self._serial_diagnostics_status = 'unavailable'

    def _record_encoded_motor_frame(self, packet):
        """Record SDK encoder output, not radio delivery or physical motion."""
        header = str(packet)[:24]
        frame = {'monotonic': time.monotonic(), 'header': header, 'mode_hex': header[:2]}
        if header[:2] in {'11', '1C', '1D'}:
            raw = [int(header[4+4*i:8+4*i], 16) for i in range(4)]
            frame.update(command_sequence=int(header[2:4], 16),
                         joint_steps=[v-65536 if v > 32767 else v for v in raw],
                         speed_pct=int(header[20:22], 16))
        with self._frame_condition:
            self._motor_frame_sequence += 1
            frame['frame_sequence'] = self._motor_frame_sequence
            signature = (frame['mode_hex'], frame.get('command_sequence'),
                         tuple(frame['joint_steps']) if 'joint_steps' in frame else None,
                         frame.get('speed_pct'))
            if signature != self._target_frame_signature:
                self._target_frame_signature = signature
                self._target_frame_segment_start = frame['frame_sequence']
            frame['target_segment_start_sequence'] = self._target_frame_segment_start
            self._latest_motor_frame = frame
            if header != self._last_motor_header:
                self._last_motor_header = header
                self.motor_frames.append(frame)
            self._frame_condition.notify_all()

    def position_target_capability(self):
        # A worker started from an older version must reject this method before
        # any command is sent. This capability is host-side only.
        return {'version': 'retained-position-target.v3', 'sdk_host_frame_only': True}

    def position_target_frame(self):
        with self._lock:
            frame = self._latest_motor_frame
            return dict(frame) if frame is not None else None

    def _record_sensor_packets(self):
        """Keep received SDK payloads for diagnosis without changing decoding."""
        core=self.robot._roboid
        original=core._decode_sensory_packet
        def decode(packet):
            received_at=time.monotonic()
            result=original(packet)
            with self._lock:
                self.sensor_packets.append({'received_monotonic':received_at,
                    'payload_hex':str(packet)[:40],'accepted_by_sdk':bool(result)})
            return result
        core._decode_sensory_packet=decode

    def _on_data(self, device, data):
        now = time.monotonic()
        with self._lock:
            device_id = device.get_id()
            if device_id == self.sdk_type.ENCODER and len(data) == 4:
                if all(isinstance(x, (int, float)) and math.isfinite(x) for x in data):
                    self.joints = tuple(float(x) for x in data)
                    self.sensor_time = now
                    self.sequence += 1
            elif device_id == self.sdk_type.JOINT_STATE:
                self.completion += 1
            elif device_id in (self.sdk_type.COLLISION_1,self.sdk_type.COLLISION_2,
                                self.sdk_type.COLLISION_3,self.sdk_type.COLLISION_4) and device._can_notify:
                # These are SDK-decoded event counts, not measured contact force.
                self.collision_event_counts[device_id-self.sdk_type.COLLISION_1] += 1
            elif device_id == self.sdk_type.SLOT_R2 and len(data) >= 3 and device._can_notify:
                # SLOT_R2 is an EVENT in the pinned SDK. The notification flag
                # is an additional guard; only this callback timestamps reception.
                self.gripper_raw = int(data[2]) & 0xff
                self.gripper_module_type = int(data[0]) & 0x0f
                self.gripper_slot_raw = tuple(int(value) & 0xff for value in data)
                self.gripper_time = now

    def diagnostics(self):
        """Host I/O evidence only; makes no writes or physical-success inference."""
        with self._lock:
            slot = tuple(self.robot.read(self.sdk_type.SLOT_R2, i) & 0xff for i in range(8))
            with self._serial_diagnostics_lock:
                serial_diagnostics = {'status': self._serial_diagnostics_status,
                                      'recent_writes': [dict(event) for event in self.serial_writes],
                                      'scope': 'host serial.write return; not radio delivery or device acknowledgement'}
            return {
                "connected": bool(self.robot.is_connected()),
                "slot2_cached_bytes": slot,
                "slot2_last_callback_bytes": self.gripper_slot_raw,
                "slot2_last_callback_monotonic": self.gripper_time,
                "gripper_module_type": slot[0] & 0x0f,
                "gripper_raw": self.gripper_raw,
                "move_commands": self.moves,
                "gripper_commands": self.gripper_commands,
                "automatic_init_pose_suppressed": True,
                "transport_closed_before_sdk_reset": True,
                "grasp_success": None,
                "gripper_load_api_available": False,
                "firmware_warning_raw": self.robot.read(self.sdk_type.WARNING),
                "collision_event_counts": tuple(self.collision_event_counts),
                "collision_event_ids_raw": tuple(getattr(self.robot._roboid,f"_event_collision_{i}_id") for i in range(1,5)),
                "battery_state_raw": self.robot.battery_state(),
                "signal_strength_raw": self.robot.signal_strength(),
                "recent_motor_frames": list(self.motor_frames),
                "recent_sensor_packets": list(self.sensor_packets),
                "serial_write_diagnostics": serial_diagnostics,
                "sensor_packet_source": "SDK decoder input; not an independent wire capture",
                "event_semantics": "SDK collision-event identifiers; physical contact and force are not measured",
            }

    def snapshot(self):
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self):
        # Caller holds _lock. Module 4's 0/1 is a device report, not proof
        # that an object has been grasped.
        reported = {0: "open", 1: "closed"}.get(self.gripper_raw) if self.gripper_module_type == 4 else None
        return Snapshot(
            bool(self.robot.is_connected()), self.joints, self.sequence,
            self.sensor_time, bool(self.robot.is_moving()), self.completion,
            reported, self.gripper_time,
            self.gripper_raw, self.stops, self.moves,
            gripper_module_type=self.gripper_module_type,
            firmware_warning_raw=self.robot.read(self.sdk_type.WARNING) if self.sequence else None,
            collision_event_counts=tuple(self.collision_event_counts),
            battery_state_raw=self.robot.battery_state() if self.sequence else None,
            signal_strength_raw=self.robot.signal_strength() if self.sequence else None,
        )

    def _stream_feedback(self, baseline_counts=None):
        return self._validate_stream_feedback(self.snapshot(), baseline_counts)

    def _transient_stream_feedback_gap(self, snapshot):
        if (not snapshot.connected or snapshot.sensor_time is None
                or snapshot.sensor_sequence < 1):
            return False
        age = time.monotonic() - snapshot.sensor_time
        try:
            check_joints(snapshot.joints_deg)
        except ControlError:
            return False
        return 0 <= age <= max(self.config.limits.sensor_max_age, SDK_FEEDBACK_GAP_GRACE_SECONDS)

    def _wait_for_stream_feedback(self, baseline_counts, cancel):
        # An old observation never authorizes the next stream target. Wait
        # briefly for a new one and shift the remaining schedule instead of
        # replaying commands that became due while feedback was absent.
        waiting_since = None
        while not cancel.is_set():
            try:
                result = self._stream_feedback(baseline_counts)
                return result, (time.monotonic() - waiting_since if waiting_since is not None else 0.)
            except ControlError as exc:
                if exc.code != 'SENSOR_STALE' or not self._transient_stream_feedback_gap(self.snapshot()):
                    raise
                if waiting_since is None:
                    waiting_since = time.monotonic()
                cancel.wait(.01)
        return None, 0.

    def _validate_stream_feedback(self, snapshot, baseline_counts=None):
        if (not snapshot.connected or snapshot.joints_deg is None or snapshot.sensor_time is None
                or snapshot.sensor_sequence < 1
                or not 0 <= time.monotonic() - snapshot.sensor_time <= self.config.limits.sensor_max_age):
            raise ControlError('SENSOR_STALE', 'The SDK stream needs fresh encoder feedback')
        check_joints(snapshot.joints_deg)
        if snapshot.firmware_warning_raw == 1:
            raise ControlError('DEVICE_WARNING', 'The SDK warning bit became active during the stream')
        if (snapshot.firmware_warning_raw != 0 or snapshot.collision_event_counts is None
                or len(snapshot.collision_event_counts) != 4
                or any(type(value) is not int or value < 0 for value in snapshot.collision_event_counts)):
            raise ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE', 'Quiet SDK warning and collision data are required')
        if baseline_counts is not None:
            counts = snapshot.collision_event_counts
            if len(counts) != len(baseline_counts) or any(a < b for a, b in zip(counts, baseline_counts)):
                raise ControlError('DEVICE_DIAGNOSTICS_UNAVAILABLE', 'SDK collision counters changed unexpectedly')
            if any(a > b for a, b in zip(counts, baseline_counts)):
                raise ControlError('DEVICE_COLLISION_EVENT', 'The SDK reported a collision event during the stream')
        return snapshot

    def blended_start_reference(self):
        # A stationary arm may be retaining a position target or may have been
        # explicitly stopped. The path reference remains the fresh encoder;
        # a host-encoded target is never promoted to a device observation.
        with self._lock:
            state = self._blended_state
            if state is not None and state.get('running'):
                raise ControlError('BUSY', 'An SDK blended path is already active')
        snapshot = self._stream_feedback()
        if snapshot.moving is not False:
            raise ControlError('NOT_STATIONARY', 'A stopped SDK encoder is required before a path')
        return {'available': True, 'source': 'sdk_stopped_encoder', 'joints_deg': list(snapshot.joints_deg),
                'sensor_sequence': snapshot.sensor_sequence, 'sensor_monotonic': snapshot.sensor_time,
                'physical_success': None}

    def _stream_constraints(self, spec):
        constraints = spec.get('path_constraints')
        keys(constraints, allowed={'vertical_xy_mm', 'corridor_mm', 'minimum_z_mm', 'maximum_z_mm'})
        constraints = dict(constraints)
        if constraints.get('vertical_xy_mm') is not None:
            constraints['vertical_xy_mm'] = vector(constraints['vertical_xy_mm'], 2, 'vertical_xy_mm')
            constraints['corridor_mm'] = number(constraints.get('corridor_mm'), 'corridor_mm', minimum=.01)
        elif constraints.get('corridor_mm') is not None:
            raise ControlError('INVALID_REQUEST', 'A corridor radius requires an XY center')
        if constraints.get('minimum_z_mm') is not None:
            constraints['minimum_z_mm'] = number(constraints['minimum_z_mm'], 'minimum_z_mm')
        if constraints.get('maximum_z_mm') is not None:
            maximum = (self.config.limits.workspace_max_mm[2]
                       if self.config.limits.workspace_configured else None)
            constraints['maximum_z_mm'] = number(constraints['maximum_z_mm'], 'maximum_z_mm', maximum=maximum)
            if (constraints.get('minimum_z_mm') is not None
                    and constraints['maximum_z_mm'] < constraints['minimum_z_mm']):
                raise ControlError('INVALID_REQUEST', 'Path ceiling must not be below its clearance floor')
        return constraints

    def _check_stream_transition(self, observed, target, constraints):
        """Recheck the measured-to-next-target path, with unchanged bounds."""
        geometry = Geometry(tool_forward_mm=self.config.tool_forward_mm,
                            tool_vertical_mm=self.config.tool_vertical_mm)
        geometry.check_path(observed, target, self.config.limits)
        envelope = motion_bounds(geometry, observed, target, self.config.limits.joint_tolerance_deg,
                                 start_tolerance_deg=self.config.limits.joint_tolerance_deg)
        return check_envelope(envelope, self.config.limits,
            vertical_xy=constraints.get('vertical_xy_mm'), corridor_mm=constraints.get('corridor_mm'),
            minimum_z_mm=constraints.get('minimum_z_mm'), maximum_z_mm=constraints.get('maximum_z_mm'))

    def _check_stream_observation(self, observed, constraints):
        """A held target never permits an observed point outside the path."""
        observed = check_joints(observed)
        geometry = Geometry(tool_forward_mm=self.config.tool_forward_mm,
                            tool_vertical_mm=self.config.tool_vertical_mm)
        point = geometry.forward(observed)
        limits = self.config.limits
        if not limits.workspace_configured:
            raise ControlError('WORKSPACE_UNSPECIFIED', 'The SDK stream requires workspace bounds')
        if any(p < lo or p > hi for p, lo, hi in
               zip(point, limits.workspace_min_mm, limits.workspace_max_mm)):
            raise ControlError('PATH_OUTSIDE_WORKSPACE', 'Observed nominal TCP left the workspace', tcp_mm=point)
        if constraints.get('minimum_z_mm') is not None and point[2] < constraints['minimum_z_mm']:
            raise ControlError('TRANSFER_CLEARANCE', 'Observed nominal TCP fell below the path', tcp_mm=point)
        if constraints.get('maximum_z_mm') is not None and point[2] > constraints['maximum_z_mm']:
            raise ControlError('PATH_CEILING', 'Observed nominal TCP rose above the path', tcp_mm=point)
        if (constraints.get('vertical_xy_mm') is not None
                and distance(point[:2], constraints['vertical_xy_mm']) > constraints['corridor_mm']):
            raise ControlError('VERTICAL_CORRIDOR', 'Observed nominal TCP left the vertical corridor', tcp_mm=point)
        return point

    def _wait_for_stream_frame(self, *, after_sequence, steps, speed_pct, cancel):
        deadline = time.monotonic() + SDK_STREAM_FRAME_TIMEOUT_S
        with self._frame_condition:
            initial_sensor_sequence = self.sequence
            extended_after_recovery = False
            while not cancel.is_set():
                frame = self._latest_motor_frame
                if (frame is not None and frame['frame_sequence'] > after_sequence
                        and frame.get('mode_hex') == '11' and frame.get('joint_steps') == list(steps)
                        and frame.get('speed_pct') == int(.5 + speed_pct)):
                    return dict(frame)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    snapshot = self._snapshot_locked()
                    if not self._transient_stream_feedback_gap(snapshot):
                        break
                    if snapshot.sensor_sequence == initial_sensor_sequence:
                        remaining = .01
                    elif not extended_after_recovery:
                        # Let the encoder callback finish after the first
                        # returning sample, without an unbounded extension.
                        extended_after_recovery = True
                        deadline = time.monotonic() + SDK_STREAM_FRAME_TIMEOUT_S
                        remaining = SDK_STREAM_FRAME_TIMEOUT_S
                    else:
                        break
                self._frame_condition.wait(min(.01, remaining))
        if cancel.is_set():
            return None
        raise ControlError('SDK_FRAME_UNCONFIRMED', 'The SDK encoder did not expose the submitted target in time')

    def move_blended_joint_path(self, command_id, spec, expected_sha256, expected_count):
        """Begin a checked host SDK stream; return before planning or movement.

        Frame acknowledgements are SDK encoder output, not radio receipt or
        evidence of an object's position. No pending target queue is replayed.
        """
        if self.config.backend != 'sdk' or self.config.hardware_enabled is not True:
            raise ControlError('HARDWARE_DISABLED', 'SDK streaming requires an enabled SDK worker')
        original_count = spec.get('original_waypoint_count')
        budget = virtual_blend_time_budget(self.config.limits.motion_timeout, original_count)
        if (spec.get('segment_timeout_seconds') != self.config.limits.motion_timeout
                or spec.get('motion_timeout_seconds') != budget):
            raise ControlError('BLENDED_PLAN_MISMATCH', 'The SDK stream budget differs from the controller')
        plan_meta = spec.get('sdk_command_plan')
        if (not isinstance(plan_meta, dict) or plan_meta.get('scope') != 'sdk_host_stream_unverified'
                or plan_meta.get('command_period_seconds') != SDK_BLENDED_COMMAND_PERIOD_S
                or type(plan_meta.get('command_count')) is not int or plan_meta['command_count'] < 1
                or not isinstance(plan_meta.get('planned_commands_sha256'), str)
                or len(plan_meta['planned_commands_sha256']) != 64
                or type(expected_count) is not int or expected_count < 1
                or not isinstance(expected_sha256, str) or len(expected_sha256) != 64):
            raise ControlError('INVALID_REQUEST', 'The SDK stream plan evidence is invalid')
        speed = number(spec.get('speed_pct'), 'speed_pct', minimum=1, maximum=self.config.limits.max_speed_pct)
        constraints = self._stream_constraints(spec)
        correction_enabled = spec.get('allow_nominal_correction', False)
        if type(correction_enabled) is not bool:
            raise ControlError('INVALID_REQUEST', 'Nominal correction permission must be boolean')
        start = check_joints(spec.get('start_joints_deg'))
        admission = check_joints(spec.get('admission_observed_joints_deg'))
        target = check_joints(spec.get('waypoints_deg')[-1])
        steps, effective = quantize_joints(target)
        if any(abs(a-b) > 1e-10 for a,b in zip(target, effective)):
            raise ControlError('BLENDED_PLAN_MISMATCH', 'The SDK final target is not quantized')
        feedback = self._stream_feedback()
        if feedback.moving is not False:
            raise ControlError('NOT_STATIONARY', 'The SDK must be stopped before streaming')
        if max(abs(a-b) for a,b in zip(feedback.joints_deg, admission)) > self.config.limits.joint_tolerance_deg:
            raise ControlError('STATE_CHANGED', 'The SDK encoder changed after admission')
        if max(abs(a-b) for a,b in zip(feedback.joints_deg, start)) > self.config.limits.joint_tolerance_deg:
            raise ControlError('STATE_CHANGED', 'The SDK start reference changed after admission')
        admitted_collision_counts = tuple(feedback.collision_event_counts)
        with self._lock:
            if self._blended_state is not None and self._blended_state.get('running'):
                raise ControlError('BUSY', 'An SDK blended path is already active')
            cancel = threading.Event()
            self._blended_cancel = cancel
            empty_digest = hashlib.sha256().hexdigest()
            diagnostics = _TrackingDiagnostics()
            self._blended_diagnostics = diagnostics
            state = {'command_id': command_id, 'scope': 'sdk_host_stream_unverified',
                     'planned_sha256': expected_sha256, 'scheduled_count': expected_count,
                     'encoding': 'be-u64-tick-and-5-f64-v1',
                     'sdk_planned_sha256': plan_meta['planned_commands_sha256'],
                     'sdk_planned_count': plan_meta['command_count'],
                     'sdk_command_period_seconds': SDK_BLENDED_COMMAND_PERIOD_S,
                     'sdk_submitted_count': 0, 'sdk_submitted_sha256': empty_digest,
                     'sdk_encoded_count': 0, 'sdk_encoded_sha256': empty_digest,
                     'sdk_submission_first_monotonic': None, 'sdk_submission_last_monotonic': None,
                     'sdk_submission_min_interval_s': None, 'sdk_submission_max_interval_s': None,
                     'sdk_submission_max_lateness_s': None,
                     'sdk_feedback_delay_seconds': 0., 'sdk_feedback_slowed_seconds': 0.,
                     'sdk_envelope_wait_seconds': 0.,
                     # Tracking decisions are disabled. Keep the old trace
                     # keys, but do not report unmeasured values as zero.
                     'sdk_feedback_tracking_enabled': False,
                     'sdk_feedback_max_joint_error_deg': None,
                     'sdk_feedback_max_joint_lag_deg': None,
                     'sdk_feedback_stable_finished_axes': None,
                     'sdk_feedback_stable_residual_axes': None,
                     'sdk_feedback_used_stable_residual_axes': None,
                     'sdk_feedback_dynamic_following_axes': None,
                     'sdk_feedback_used_dynamic_following_axes': None,
                     'sdk_feedback_confirmed_residuals_deg': None,
                     'sdk_feedback_max_additional_lags_deg': None,
                     'sdk_feedback_last_sensor_sequence': None,
                     'sdk_encoding_first_monotonic': None, 'sdk_encoding_last_monotonic': None,
                     'sdk_encoding_min_interval_s': None, 'sdk_encoding_max_interval_s': None,
                     'last_sdk_submitted_target_deg': None, 'last_sdk_encoded_target_deg': None,
                     'last_sdk_submitted_time_s': None, 'last_sdk_encoded_time_s': None,
                     'applied_count': None, 'applied_sha256': None, 'last_applied_joints_deg': None,
                     'completed': False, 'running': True,
                     'plan_verified': False, 'worker_error': None,
                     'stop_diagnostics': None,
                     'intermediate_stop_count': 0, 'stop_requested_during_path': False,
                     'physical_success': None, 'sdk_encoder_is_wire_ack': False,
                     'sensor_is_object_observation': False}
            self._blended_state = state
            self.moves += 1

        def execute():
            try:
                path = prepare_virtual_blend(spec, motion_timeout=budget)
                virtual = planned_sample_digest(path)
                commands = sdk_blended_commands(path)
                if (virtual['sha256'] != expected_sha256 or virtual['sample_count'] != expected_count
                        or len(commands) != plan_meta['command_count']
                        or sdk_blended_command_digest(commands) != plan_meta['planned_commands_sha256']):
                    raise ControlError('BLENDED_PLAN_MISMATCH', 'The SDK worker reconstructed a different path')
                feedback, _ = self._wait_for_stream_feedback(admitted_collision_counts, cancel)
                if feedback is None:
                    return
                if (feedback.moving is not False
                        or max(abs(a-b) for a,b in zip(feedback.joints_deg, admission)) > self.config.limits.joint_tolerance_deg
                        or max(abs(a-b) for a,b in zip(feedback.joints_deg, start)) > self.config.limits.joint_tolerance_deg):
                    raise ControlError('STATE_CHANGED', 'The SDK start changed during path preparation')
                baseline = admitted_collision_counts
                submitted = hashlib.sha256()
                encoded = hashlib.sha256()
                encoded_target = None
                envelope_wait = 0.
                with self._lock:
                    state['plan_verified'] = True
                    state['started_monotonic'] = time.monotonic()
                for index, (when, target_deg) in enumerate(commands, 1):
                    # Submit each planned target once at its scheduled time;
                    # encoder tracking error does not hold or abort the stream.
                    envelope_wait_started = None
                    submitted_at = None
                    target_steps, _ = quantize_joints(target_deg)
                    while not cancel.is_set():
                        observed, waited = self._wait_for_stream_feedback(baseline, cancel)
                        if observed is None:
                            break
                        envelope_wait += waited
                        now = time.monotonic()
                        self._check_stream_observation(observed.joints_deg, constraints)
                        with self._lock:
                            state['sdk_feedback_last_sensor_sequence'] = observed.sensor_sequence
                        due = state['started_monotonic'] + when + envelope_wait
                        if now >= due:
                            checked = observed
                            try:
                                self._check_stream_transition(checked.joints_deg, target_deg, constraints)
                            except ControlError as exc:
                                # If the actual TCP is still inside all original
                                # constraints, a conservative transition box
                                # can become safe as the held target is reached.
                                if exc.code != 'ASYNC_ENVELOPE' or encoded_target is None:
                                    raise
                                if envelope_wait_started is None:
                                    envelope_wait_started = now
                                if now-envelope_wait_started >= SDK_TRACKING_STALL_SECONDS:
                                    raise
                                began_wait = time.monotonic()
                                if cancel.wait(.01):
                                    break
                                envelope_wait += time.monotonic()-began_wait
                                with self._lock:
                                    state['sdk_envelope_wait_seconds'] = envelope_wait
                                continue
                            observed, waited = self._wait_for_stream_feedback(baseline, cancel)
                            if observed is None:
                                break
                            if waited:
                                # Recompute the transition using the new
                                # observation; never send from the old one.
                                envelope_wait += waited
                                continue
                            self._check_stream_observation(observed.joints_deg, constraints)
                            if (observed.sensor_sequence < checked.sensor_sequence
                                    or max(abs(a-b) for a, b in zip(observed.joints_deg, checked.joints_deg))
                                       > self.config.limits.joint_tolerance_deg):
                                raise ControlError('STATE_CHANGED', 'The encoder changed during SDK path checking')
                            adjusted_due = state['started_monotonic'] + when + envelope_wait
                            if time.monotonic() > adjusted_due + SDK_STREAM_MAX_LATENESS_S:
                                raise ControlError('SDK_STREAM_LATE', 'A late SDK target was not replayed')
                            with self._stream_send_lock:
                                if cancel.is_set():
                                    break
                                # Fence the vendor's four-axis write and validate
                                # the newest observation at that send boundary.
                                with self._sdk_runner_lock:
                                    with self.robot._roboid._thread_lock:
                                        with self._lock:
                                            if cancel.is_set():
                                                break
                                            latest = self._snapshot_locked()
                                            try:
                                                final = self._validate_stream_feedback(latest, baseline)
                                            except ControlError as exc:
                                                if (exc.code == 'SENSOR_STALE'
                                                        and self._transient_stream_feedback_gap(latest)):
                                                    continue
                                                raise
                                            self._check_stream_observation(final.joints_deg, constraints)
                                            if (final.sensor_sequence < checked.sensor_sequence
                                                    or max(abs(a-b) for a, b in zip(final.joints_deg, checked.joints_deg))
                                                       > self.config.limits.joint_tolerance_deg):
                                                raise ControlError('STATE_CHANGED', 'The encoder changed before SDK submission')
                                            send_time = time.monotonic()
                                            if send_time > adjusted_due + SDK_STREAM_MAX_LATENESS_S:
                                                raise ControlError('SDK_STREAM_LATE', 'A late SDK target was not replayed')
                                            frame_before = self._motor_frame_sequence
                                            self.robot.unlock()
                                            if cancel.is_set():
                                                break
                                            self.robot.degree_to(*target_deg, speed, wait=False)
                                            submitted_at = time.monotonic()
                                            update_sample_digest(submitted, index, when, target_deg)
                                            previous = state['sdk_submission_last_monotonic']
                                            if previous is None:
                                                state['sdk_submission_first_monotonic'] = submitted_at
                                            else:
                                                interval = submitted_at - previous
                                                minimum = state['sdk_submission_min_interval_s']
                                                maximum = state['sdk_submission_max_interval_s']
                                                state['sdk_submission_min_interval_s'] = interval if minimum is None else min(minimum, interval)
                                                state['sdk_submission_max_interval_s'] = interval if maximum is None else max(maximum, interval)
                                            state['sdk_submission_last_monotonic'] = submitted_at
                                            lateness = max(0., submitted_at - adjusted_due)
                                            state['sdk_submission_max_lateness_s'] = max(state['sdk_submission_max_lateness_s'] or 0., lateness)
                                            state.update(sdk_submitted_count=index, sdk_submitted_sha256=submitted.hexdigest(),
                                                last_sdk_submitted_target_deg=list(target_deg), last_sdk_submitted_time_s=when)
                            if submitted_at is not None:
                                break
                        interval = min(.01, max(.001, due - now)) if now < due else .01
                        if cancel.wait(interval):
                            break
                    if cancel.is_set():
                        break
                    frame_wait_started = time.monotonic()
                    frame = self._wait_for_stream_frame(after_sequence=frame_before, steps=target_steps,
                                                        speed_pct=speed, cancel=cancel)
                    frame_wait = time.monotonic() - frame_wait_started
                    if frame_wait > SDK_STREAM_MAX_LATENESS_S:
                        # A gap while waiting for this target's host frame is
                        # not a reason to replay later targets at once.
                        envelope_wait += frame_wait
                    if frame is None:
                        break
                    encoded_target = target_deg
                    update_sample_digest(encoded, index, when, target_deg)
                    with self._lock:
                        previous = state['sdk_encoding_last_monotonic']
                        if previous is None:
                            state['sdk_encoding_first_monotonic'] = frame['monotonic']
                        else:
                            interval = frame['monotonic'] - previous
                            minimum = state['sdk_encoding_min_interval_s']
                            maximum = state['sdk_encoding_max_interval_s']
                            state['sdk_encoding_min_interval_s'] = interval if minimum is None else min(minimum, interval)
                            state['sdk_encoding_max_interval_s'] = interval if maximum is None else max(maximum, interval)
                        state['sdk_encoding_last_monotonic'] = frame['monotonic']
                        state.update(sdk_encoded_count=index, sdk_encoded_sha256=encoded.hexdigest(),
                                     last_sdk_encoded_target_deg=list(target_deg), last_sdk_encoded_time_s=when,
                                     last_sdk_encoded_monotonic=frame['monotonic'],
                                     last_sdk_encoded_frame=dict(frame))
                with self._lock:
                    if not cancel.is_set() and state['sdk_encoded_count'] == len(commands):
                        state['completed'] = True
            except Exception as exc:
                with self._lock:
                    state['worker_error'] = exc.as_dict() if isinstance(exc, ControlError) else {
                        'code': 'DEVICE_ERROR', 'message': type(exc).__name__, 'details': {}}
                    diagnostics.record_error(state['worker_error'])
                try:
                    self.stop()
                except Exception:
                    pass
            finally:
                with self._lock:
                    state['running'] = False

        thread = threading.Thread(target=execute, name='sdk-blended-stream', daemon=True)
        self._blended_thread = thread
        thread.start()
        return {'command_id': command_id, 'scheduled_count': expected_count, 'planned_sha256': expected_sha256,
                'encoding': 'be-u64-tick-and-5-f64-v1', 'steps': steps, 'effective_target_deg': effective,
                'simulated': False, 'scope': 'sdk_host_stream_unverified',
                'plan_verification': 'pending_worker_reconstruction'}

    def blended_path_trace(self, command_id):
        with self._lock:
            state = self._blended_state
            if state is None or state['command_id'] != command_id:
                return {'command_id': command_id, 'available': False}
            result = {'available': True, **state}
            if (not state['running'] or state['worker_error'] is not None
                    or state['stop_requested_during_path']):
                if self._blended_diagnostics is not None:
                    result['sdk_feedback_diagnostics'] = self._blended_diagnostics.snapshot()
            return result

    def move_joints(self, joints_deg, speed_pct, capture_host_frame=False, position_parent=None):
        # A move already being dispatched when stop begins cannot run after it.
        with self._lock:
            stop_epoch = self._stop_epoch
        joints = check_joints(joints_deg)
        speed = number(speed_pct, "speed_pct", minimum=1, maximum=self.config.limits.max_speed_pct)
        if not self.robot.is_connected():
            raise ControlError("DISCONNECTED", "SDK is not connected")
        steps, effective = quantize_joints(joints)
        if type(capture_host_frame) is not bool:
            raise ControlError('INVALID_REQUEST', 'capture_host_frame must be boolean')
        if position_parent is not None and not capture_host_frame:
            raise ControlError('INVALID_REQUEST', 'A compensated target requires host-frame evidence')
        if capture_host_frame:
            # Keep any old SDK frame from being mistaken for this submission.
            # The request-data flush only commits the SDK's in-memory target;
            # the later encoder output is not a device acknowledgement.
            with self._stream_send_lock, self._sdk_wire_lock, self._sdk_runner_lock:
                with self.robot._roboid._thread_lock:
                    with self._lock:
                        if stop_epoch != self._stop_epoch:
                            raise ControlError('STOP_LATCHED', 'The move was superseded by a stop')
                        if position_parent is not None:
                            basis, expected = position_parent['basis'], position_parent['host_frame']
                            frame = self._latest_motor_frame
                            snapshot = self._validate_stream_feedback(self._snapshot_locked(),
                                position_parent['collision_event_counts'])
                            if (snapshot.moving is not False or snapshot.sensor_sequence < basis['sensor_sequence']
                                    or snapshot.sensor_time < basis['sensor_monotonic']
                                    or not frame or frame.get('mode_hex') != '11'
                                    or frame.get('command_sequence') != expected['command_sequence']
                                    or frame.get('target_segment_start_sequence') != expected['target_segment_start_sequence']
                                    or frame.get('joint_steps') != list(basis['nominal_steps'])
                                    or frame.get('speed_pct') != expected['speed_pct']
                                    or frame.get('frame_sequence', -1) < expected['frame_sequence']
                                    or not 0 <= time.monotonic()-frame['monotonic'] <= self.config.limits.sensor_max_age):
                                raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The parent position target changed before compensation')
                            nominal = check_joints(basis['nominal_target_deg'])
                            geometry = Geometry(tool_forward_mm=self.config.tool_forward_mm,
                                                tool_vertical_mm=self.config.tool_vertical_mm)
                            goal_tcp = geometry.forward(nominal)
                            before_tcp = geometry.forward(basis['joints_deg'])
                            actual_tcp = geometry.forward(snapshot.joints_deg)
                            before_error = [a-b for a,b in zip(nominal, basis['joints_deg'])]
                            current_error = [a-b for a,b in zip(nominal, snapshot.joints_deg)]
                            joint_error = max(map(abs, current_error))
                            tcp_error = distance(actual_tcp, goal_tcp)
                            crossed_joint = any(a*b < 0 and abs(b) > self.config.limits.joint_tolerance_deg
                                                for a,b in zip(before_error, current_error))
                            crossed_tcp = (sum((a-b)*(a-c) for a,b,c in zip(goal_tcp,before_tcp,actual_tcp)) < 0
                                           and tcp_error > position_parent['arrival_tolerance_mm'])
                            if (joint_error > basis['baseline_joint_error_deg']
                                    or tcp_error > basis['baseline_tcp_error_mm'] or crossed_joint or crossed_tcp):
                                raise ControlError('DAMPED_COMPENSATION_DIVERGED', 'Nominal error worsened before compensation')
                            constraints = position_parent['path_constraints']
                            self._check_stream_observation(snapshot.joints_deg, constraints)
                            geometry.check_path(snapshot.joints_deg, effective, self.config.limits)
                            envelope = motion_bounds(geometry, snapshot.joints_deg, effective,
                                                     self.config.limits.joint_tolerance_deg, start_tolerance_deg=0)
                            check_envelope(envelope, self.config.limits,
                                vertical_xy=constraints.get('vertical_xy_mm'), corridor_mm=constraints.get('corridor_mm'),
                                minimum_z_mm=constraints.get('minimum_z_mm'), maximum_z_mm=constraints.get('maximum_z_mm'))
                        frame_before = self._motor_frame_sequence
                        self.robot.unlock()
                        self.robot.degree_to(*joints, speed, wait=False)
                        submitted_at = time.monotonic()
                self.robot._roboid._request_motoring_data()
        else:
            with self._stream_send_lock, self._sdk_runner_lock:
                with self.robot._roboid._thread_lock:
                    with self._lock:
                        if stop_epoch != self._stop_epoch:
                            raise ControlError('STOP_LATCHED', 'The move was superseded by a stop')
                        self.robot.unlock()
                        self.robot.degree_to(*joints, speed, wait=False)
        self.moves += 1
        receipt = {"steps": steps, "effective_target_deg": effective, "speed_pct": speed}
        if capture_host_frame:
            receipt.update(encoding_after_frame_sequence=frame_before, submitted_monotonic=submitted_at)
        return receipt

    def set_gripper(self, state, retain_arm=None):
        if state not in {'open', 'closed'}:
            raise ControlError("INVALID_GRIPPER", "Expected open or closed")
        def command():
            if state == 'open':
                self.robot.open_gripper(wait=False)
            else:
                self.robot.close_gripper(wait=False)
        if retain_arm is None:
            command()
            boundary = {}
        else:
            # Check the newest decoder sample and the last host frame in the
            # same SDK send fence as the SLOT_W2 write. No arm target is sent.
            with self._stream_send_lock, self._sdk_wire_lock, self._sdk_runner_lock:
                with self.robot._roboid._thread_lock:
                    with self._lock:
                        snapshot = self._validate_stream_feedback(self._snapshot_locked(),
                                                                  retain_arm['collision_event_counts'])
                        frame = self._latest_motor_frame
                        if (not frame or frame.get('mode_hex') != '11'
                                or frame.get('command_sequence') != retain_arm['command_sequence']
                                or frame.get('target_segment_start_sequence') != retain_arm['target_segment_start_sequence']
                                or frame.get('joint_steps') != list(retain_arm['target_steps'])
                                or frame.get('speed_pct') != retain_arm['speed_pct']):
                            raise ControlError('POSITION_TARGET_UNCONFIRMED', 'The SDK host position target changed')
                        target = check_joints(retain_arm['target_deg'])
                        nominal = check_joints(retain_arm['nominal_target_deg'])
                        geometry = Geometry(tool_forward_mm=self.config.tool_forward_mm,
                                            tool_vertical_mm=self.config.tool_vertical_mm)
                        observed_tcp = geometry.forward(snapshot.joints_deg)
                        arrival_diagnostics = {
                            'observed_joints_deg': list(snapshot.joints_deg),
                            'joint_error_deg': max(abs(a-b) for a,b in zip(snapshot.joints_deg, target)),
                            'tcp_error_mm': distance(observed_tcp, geometry.forward(target)),
                            'arrival_joint_error_deg': max(abs(a-b) for a,b in zip(snapshot.joints_deg, nominal)),
                            'arrival_tcp_error_mm': distance(observed_tcp, geometry.forward(nominal)),
                            'sensor_sequence': snapshot.sensor_sequence, 'sensor_monotonic': snapshot.sensor_time,
                        }
                        if (snapshot.moving is not False
                                or snapshot.sensor_sequence < retain_arm['sensor_sequence']):
                            raise ControlError('TARGET_NOT_REACHED',
                                'A stopped arm and current encoder sample are required at gripper dispatch',
                                **arrival_diagnostics)
                        self._check_stream_observation(snapshot.joints_deg, retain_arm['path_constraints'])
                        frame_before = self._motor_frame_sequence
                        command()
                        sent_at = time.monotonic()
                        boundary = {'after_frame_sequence': frame_before, 'sent_monotonic': sent_at,
                                    'before_sensor_sequence': snapshot.sensor_sequence,
                                    'arrival_diagnostics': arrival_diagnostics}
                # Commit SLOT_W2 to the SDK packet state before the wire
                # fence opens; this does not write another arm target.
                self.robot._roboid._request_motoring_data()
        self.gripper_commands += 1
        return {"command": state, "object_held_visual": None, **boundary}

    def stop(self):
        # Set cancellation before waiting on the SDK write fence. An in-flight
        # degree_to can finish, but no later target may be written after stop.
        self._blended_cancel.set()
        with self._frame_condition:
            self._stop_epoch += 1
            state = self._blended_state
            active = state is not None and state.get('running')
            if active and not state.get('completed'):
                state['stop_requested_during_path'] = True
                if state.get('stop_diagnostics') is None:
                    # Capture the already available bounded rings immediately
                    # before issuing stop. This is an in-process copy, not a
                    # device RPC and never a substitute for stop feedback.
                    state['stop_diagnostics'] = {
                        'captured_monotonic': time.monotonic(),
                        'recent_motor_frames': [dict(frame) for frame in self.motor_frames],
                        'recent_sensor_packets': [dict(packet) for packet in self.sensor_packets],
                        'sensor_packet_source': 'SDK decoder input; not an independent wire capture',
                    }
            self._frame_condition.notify_all()
        with self._stream_send_lock:
            with self._sdk_wire_lock:
                with self._sdk_runner_lock:
                    with self.robot._roboid._thread_lock:
                        self.robot.stop()
                    # Commit velocity-zero to the SDK's current packet state
                    # before allowing the next _send to start for any MOVE.
                    self.robot._roboid._request_motoring_data()
        self.stops += 1
        return {"requested": True, "connected": bool(self.robot.is_connected())}

    def close(self):
        thread = self._blended_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            self.stop()
            thread.join(self.config.limits.close_timeout)
            if thread.is_alive():
                raise ControlError('SDK_STREAM_STOP_UNCONFIRMED', 'The SDK stream did not stop before close')
        robot,self.robot=self.robot,None
        try:
            if robot is not None:
                try:
                    robot.remove_device_data_changed_listener(self._on_data)
                finally:
                    robot.dispose()
        finally:
            self.lock.close()
