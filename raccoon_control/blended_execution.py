"""Shared virtual blended-path planning and compact application evidence.

This is a virtual worker contract, not a claim about SDK timing or motion.
"""
import math

from .blended_path import plan_blended_path, execution_samples_sha256, update_sample_digest as _update_sample_digest
from .errors import ControlError
from .kinematics import check_joints
from .validation import number

MAX_BLEND_WAYPOINTS = 64
MAX_BLEND_DURATION_S = 60.
MIN_BLEND_PERIOD_S = .002
MAX_BLEND_PERIOD_S = .2
SDK_BLENDED_COMMAND_PERIOD_S = .05


def virtual_blend_time_budget(single_timeout, waypoint_count):
    """Aggregate the existing per-waypoint deadline with a finite cap."""
    timeout = number(single_timeout, 'single_timeout', minimum=.0001, maximum=30.)
    if type(waypoint_count) is not int or not 1 <= waypoint_count <= MAX_BLEND_WAYPOINTS:
        raise ControlError('INVALID_REQUEST', 'A blended path needs 1 to 64 destination waypoints')
    return min(MAX_BLEND_DURATION_S, timeout * waypoint_count)


def prepare_virtual_blend(spec, *, motion_timeout):
    """Reconstruct the same finite, fixed-tick plan in controller and worker."""
    waypoints = spec.get('waypoints_deg')
    durations = spec.get('segment_durations_s')
    if not isinstance(waypoints, (tuple, list)) or not 1 <= len(waypoints) <= MAX_BLEND_WAYPOINTS:
        raise ControlError('INVALID_REQUEST', 'A blended path needs 1 to 64 destination waypoints')
    if not isinstance(durations, (tuple, list)) or len(durations) != len(waypoints):
        raise ControlError('INVALID_REQUEST', 'A blended path needs one duration per waypoint')
    groups = spec.get('segment_groups', [1] * len(waypoints))
    original_count = spec.get('original_waypoint_count', len(waypoints))
    if (not isinstance(groups, (tuple, list)) or type(original_count) is not int
            or len(groups) != original_count or not groups
            or any(type(count) is not int or count < 1 for count in groups)
            or sum(groups) != len(waypoints)):
        raise ControlError('INVALID_REQUEST', 'Invalid original-segment grouping')
    start = check_joints(spec.get('start_joints_deg'))
    checked = [check_joints(point) for point in waypoints]
    period = number(spec.get('sample_period_seconds'), 'sample_period_seconds',
                    minimum=MIN_BLEND_PERIOD_S, maximum=MAX_BLEND_PERIOD_S)
    segment_timeout = number(spec.get('segment_timeout_seconds', motion_timeout),
                             'segment_timeout_seconds', minimum=period, maximum=30.)
    # Check the raw duration before constructing derivatives or expanding ticks.
    durations = [number(x, 'segment_durations_s', minimum=period, maximum=segment_timeout)
                 for x in durations]
    if sum(durations) > motion_timeout:
        raise ControlError('MOTION_TIMEOUT', 'The path exceeds the command time budget')
    try:
        path = plan_blended_path([start, *checked], durations,
            max_velocity_deg_s=spec.get('max_velocity_deg_s'),
            max_acceleration_deg_s2=spec.get('max_acceleration_deg_s2'),
            max_jerk_deg_s3=spec.get('max_jerk_deg_s3'),
            time_quantum_s=period, require_nonstop=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ControlError('INVALID_BLENDED_PATH', str(exc)) from exc
    ticks = sum(path.segment_ticks)
    offset = 0
    for index, count in enumerate(groups):
        group_duration = sum(segment.duration_s for segment in path.segments[offset:offset + count])
        if group_duration > segment_timeout:
            raise ControlError('MOTION_TIMEOUT', 'A blended segment exceeds the original single-command time budget',
                               group_index=index, group_duration_seconds=group_duration,
                               segment_timeout_seconds=segment_timeout)
        offset += count
    # The aggregate bounded motion deadline also bounds the number of ticks.
    sample_budget = math.ceil(motion_timeout / period)
    if path.duration_s > motion_timeout or ticks > sample_budget:
        raise ControlError('MOTION_TIMEOUT', 'The scaled path exceeds the command time or sample budget')
    if not math.isfinite(path.duration_s):
        raise ControlError('INVALID_BLENDED_PATH', 'The trajectory duration must be finite')
    return path


def update_sample_digest(digest, tick, time_s, joints_deg):
    """Hash the virtual target actually applied at a given scheduled tick."""
    _update_sample_digest(digest, tick, time_s, check_joints(joints_deg))


def planned_sample_digest(path):
    digest = execution_samples_sha256((when, check_joints(joints))
                                      for when, joints in path.execution_samples())
    return {'sha256': digest, 'sample_count': sum(path.segment_ticks),
            'encoding': 'be-u64-tick-and-5-f64-v1'}


def sdk_blended_commands(path):
    """Return scheduled quantized host targets, not physical observations.

    RoboID's 20 ms Runner and receive-driven packet loop do not guarantee that
    a particular target is transmitted; the SDK adapter checks encoding later.
    """
    from .kinematics import quantize_joints
    count = math.ceil(path.duration_s / SDK_BLENDED_COMMAND_PERIOD_S)
    _, previous = quantize_joints(path.waypoints_deg[0])
    commands = []
    for index in range(1, count + 1):
        when = min(index * SDK_BLENDED_COMMAND_PERIOD_S, path.duration_s)
        _, target = quantize_joints(path.sample(when).position_deg)
        if target != previous:
            commands.append((when, target))
            previous = target
    return tuple(commands)


def sdk_blended_command_digest(commands):
    """Digest SDK host targets using the same unambiguous tick encoding."""
    if not commands:
        raise ControlError('INVALID_BLENDED_PATH', 'The SDK path has no distinct command target')
    return execution_samples_sha256(commands)
