"""Pure, bounded, C2 joint-space paths through explicitly supplied waypoints.

The trajectory is not a collision check or a hardware controller. Callers must
validate every pair of waypoint endpoints with their existing path checks and
must supply velocity, acceleration and jerk limits appropriate to the actual
execution backend. Bezier position control points stay inside each endpoint
box, so that a valid independent-joint envelope remains applicable.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import struct
from typing import Iterator


JOINT_COUNT = 4
JointVector = tuple[float, float, float, float]


def _real(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite number') from exc
    if not math.isfinite(result) or (positive and result <= 0.):
        raise ValueError(f'{label} must be finite' + (' and positive' if positive else ''))
    return result


def _vector(value, label, *, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != JOINT_COUNT:
        raise ValueError(f'{label} must contain exactly four values')
    return tuple(_real(item, label, positive=positive) for item in value)


def _limits(value, label):
    if isinstance(value, Real) and not isinstance(value, bool):
        bound = _real(value, label, positive=True)
        return (bound,) * JOINT_COUNT
    return _vector(value, label, positive=True)


def update_sample_digest(digest, tick, time_s, joints_deg):
    """Append one ``be-u64-tick-and-5-f64-v1`` command to a digest.

    This verifies representation, not configured robot joint limits. Callers
    retain responsibility for validating the commanded joint angles.
    """
    if type(tick) is not int or not 1 <= tick < (1 << 64):
        raise ValueError('execution tick must be a positive unsigned 64-bit integer')
    time = _real(time_s, 'sample time', positive=True)
    joints = _vector(joints_deg, 'sample target')
    digest.update(struct.pack('>Q5d', tick, time, *joints))


def execution_samples_sha256(samples):
    """Hash ordered future ``(relative_seconds, joints_deg)`` commands.

    The 0-second observation is not an execution command and is rejected.
    The ``be-u64-tick-and-5-f64-v1`` format concatenates, for each command,
    ``struct.pack('>Q5d', tick, time, q1, q2, q3, q4)`` with ticks starting at
    1. There is no header. Numeric values are converted to finite floats. No
    rounding, sorting, or reordering is performed.
    """
    digest = hashlib.sha256()
    previous = 0.
    count = 0
    for item in samples:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('each execution sample must be a time and four-joint target')
        time = _real(item[0], 'sample time', positive=True)
        if time <= previous:
            raise ValueError('execution sample times must be strictly increasing')
        if count >= (1 << 64) - 1:
            raise ValueError('too many execution samples')
        update_sample_digest(digest, count + 1, time, item[1])
        previous = time
        count += 1
    if count == 0:
        raise ValueError('at least one future execution sample is required')
    return digest.hexdigest()


def _derivative(controls, duration_s):
    degree = len(controls) - 1
    rows = []
    for left, right in zip(controls[:-1], controls[1:], strict=True):
        row = tuple(((b - a) / duration_s) * degree for a, b in zip(left, right, strict=True))
        if not all(math.isfinite(x) for x in row):
            raise ValueError('trajectory derivative cannot be represented finitely')
        rows.append(row)
    return tuple(rows)


def _evaluate(controls, progress):
    # de Casteljau interpolation retains the convex-hull interpretation.
    if progress == 0.:
        return controls[0]
    if progress == 1.:
        return controls[-1]
    rows = controls
    while len(rows) > 1:
        rows = tuple(tuple(a + (b - a) * progress for a, b in zip(left, right, strict=True))
                     for left, right in zip(rows[:-1], rows[1:], strict=True))
    return rows[0]


@dataclass(frozen=True)
class BlendedSample:
    time_s: float
    position_deg: JointVector
    velocity_deg_s: JointVector
    acceleration_deg_s2: JointVector
    jerk_deg_s3: JointVector


@dataclass(frozen=True)
class BezierSegment:
    start_s: float
    end_s: float
    duration_s: float
    control_points_deg: tuple[JointVector, ...]
    velocity_control_points_deg_s: tuple[JointVector, ...]
    acceleration_control_points_deg_s2: tuple[JointVector, ...]
    jerk_control_points_deg_s3: tuple[JointVector, ...]

    def sample(self, local_time_s):
        """Evaluate this segment at a local time in its closed interval."""
        local = _real(local_time_s, 'local_time_s')
        if not 0. <= local <= self.duration_s:
            raise ValueError('local time must be inside the segment')
        progress = local / self.duration_s
        return BlendedSample(self.start_s + local,
            _evaluate(self.control_points_deg, progress),
            _evaluate(self.velocity_control_points_deg_s, progress),
            _evaluate(self.acceleration_control_points_deg_s2, progress),
            _evaluate(self.jerk_control_points_deg_s3, progress))


@dataclass(frozen=True)
class BlendedPath:
    waypoints_deg: tuple[JointVector, ...]
    requested_segment_durations_s: tuple[float, ...]
    velocity_limits_deg_s: JointVector
    acceleration_limits_deg_s2: JointVector
    jerk_limits_deg_s3: JointVector
    require_nonstop: bool
    segments: tuple[BezierSegment, ...]
    boundary_times_s: tuple[float, ...]
    node_velocities_deg_s: tuple[JointVector, ...]
    stopped_waypoint_indices: tuple[int, ...]
    duration_s: float
    scale: float
    time_quantum_s: float | None
    segment_ticks: tuple[int, ...] | None
    max_abs_velocity_deg_s: JointVector
    max_abs_acceleration_deg_s2: JointVector
    max_abs_jerk_deg_s3: JointVector

    def as_dict(self):
        """Return a JSON-compatible canonical description of inputs and path."""
        def vectors(rows):
            return [list(row) for row in rows]

        return {
            'schema': 'raccoon-blended-joint-path.v1',
            'waypoints_deg': vectors(self.waypoints_deg),
            'requested_segment_durations_s': list(self.requested_segment_durations_s),
            'max_velocity_deg_s': list(self.velocity_limits_deg_s),
            'max_acceleration_deg_s2': list(self.acceleration_limits_deg_s2),
            'max_jerk_deg_s3': list(self.jerk_limits_deg_s3),
            'time_quantum_s': self.time_quantum_s,
            'require_nonstop': self.require_nonstop,
            'scale': self.scale,
            'duration_s': self.duration_s,
            'boundary_times_s': list(self.boundary_times_s),
            'node_velocities_deg_s': vectors(self.node_velocities_deg_s),
            'stopped_waypoint_indices': list(self.stopped_waypoint_indices),
            'segment_ticks': None if self.segment_ticks is None else list(self.segment_ticks),
            'max_abs_velocity_deg_s': list(self.max_abs_velocity_deg_s),
            'max_abs_acceleration_deg_s2': list(self.max_abs_acceleration_deg_s2),
            'max_abs_jerk_deg_s3': list(self.max_abs_jerk_deg_s3),
            'segments': [{
                'start_s': segment.start_s, 'end_s': segment.end_s, 'duration_s': segment.duration_s,
                'control_points_deg': vectors(segment.control_points_deg),
                'velocity_control_points_deg_s': vectors(segment.velocity_control_points_deg_s),
                'acceleration_control_points_deg_s2': vectors(segment.acceleration_control_points_deg_s2),
                'jerk_control_points_deg_s3': vectors(segment.jerk_control_points_deg_s3),
            } for segment in self.segments],
        }

    @property
    def canonical_spec(self):
        return self.as_dict()

    @property
    def sha256(self):
        raw = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def _segment_and_local_time(self, time):
        """Choose the same exact boundary representation for all sampling."""
        index = min(bisect_right(self.boundary_times_s, time) - 1, len(self.segments) - 1)
        segment = self.segments[index]
        # Use exact local endpoints at a known boundary, avoiding subtraction
        # drift that could otherwise omit or slightly overshoot a waypoint.
        if time == segment.start_s:
            local = 0.
        elif time == segment.end_s:
            local = segment.duration_s
        else:
            local = time - segment.start_s
        return segment, local

    def sample(self, time_s):
        """Evaluate position and three derivatives at a path-relative time."""
        time = _real(time_s, 'time_s')
        if not 0. <= time <= self.duration_s:
            raise ValueError('time must be inside the path')
        segment, local = self._segment_and_local_time(time)
        evaluated = segment.sample(local)
        return BlendedSample(time, evaluated.position_deg, evaluated.velocity_deg_s,
                             evaluated.acceleration_deg_s2, evaluated.jerk_deg_s3)

    def sample_ticks(self) -> Iterator[BlendedSample]:
        """Yield every fixed-period sample, including all waypoint boundaries.

        This requires planning with ``time_quantum_s``. The first sample is at
        zero; command consumers may explicitly skip it if their API expects
        only future points.
        """
        if self.time_quantum_s is None or self.segment_ticks is None:
            raise ValueError('sample_ticks requires a time quantum during planning')
        for tick in range(sum(self.segment_ticks) + 1):
            yield self.sample(tick * self.time_quantum_s)

    def execution_samples(self) -> Iterator[tuple[float, JointVector]]:
        """Yield fixed-period future targets without evaluating derivatives.

        Boundary selection and the position evaluator are shared with
        ``sample_ticks`` so command times and position bits are unchanged.
        The initial observation is not an execution command.
        """
        if self.time_quantum_s is None or self.segment_ticks is None:
            raise ValueError('sample_ticks requires a time quantum during planning')
        for tick in range(1, sum(self.segment_ticks) + 1):
            time = tick * self.time_quantum_s
            segment, local = self._segment_and_local_time(time)
            if not 0. <= local <= segment.duration_s:
                raise ValueError('local time must be inside the segment')
            yield time, _evaluate(segment.control_points_deg, local / segment.duration_s)

    @property
    def execution_sha256(self):
        return execution_samples_sha256(self.execution_samples())


def _node_velocities(waypoints, durations):
    slopes = []
    for left, right, duration in zip(waypoints[:-1], waypoints[1:], durations, strict=True):
        row = tuple((b - a) / duration for a, b in zip(left, right, strict=True))
        if not all(math.isfinite(value) for value in row):
            raise ValueError('waypoint speed cannot be represented finitely')
        slopes.append(row)
    zero = (0.,) * JOINT_COUNT
    velocities = [zero]
    for left, right in zip(slopes[:-1], slopes[1:], strict=True):
        velocities.append(tuple(min(a, b) if a > 0. and b > 0. else
                                -min(-a, -b) if a < 0. and b < 0. else 0.
                                for a, b in zip(left, right, strict=True)))
    return tuple([*velocities, zero])


def _control_points(start, end, start_velocity, end_velocity, duration):
    # Endpoint acceleration zero implies P2 = 2 P1 - P0 and P3 = 2 P4 - P5.
    p1 = tuple(a + v * (duration / 5.) for a, v in zip(start, start_velocity, strict=True))
    p2 = tuple(a + 2. * (b - a) for a, b in zip(start, p1, strict=True))
    p4 = tuple(b - v * (duration / 5.) for b, v in zip(end, end_velocity, strict=True))
    p3 = tuple(b + 2. * (a - b) for a, b in zip(p4, end, strict=True))
    controls = (start, p1, p2, p3, p4, end)
    for axis in range(JOINT_COUNT):
        values = [point[axis] for point in controls]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('trajectory controls cannot be represented finitely')
        low, high = sorted((start[axis], end[axis]))
        ordered = values if start[axis] <= end[axis] else values[::-1]
        if any(value < low or value > high for value in values) or any(a > b for a, b in zip(ordered, ordered[1:])):
            raise ValueError('trajectory control points are not monotone within their endpoint box')
    return controls


def _durations(base_durations, scale, quantum):
    durations, ticks = [], []
    for base in base_durations:
        target = base * scale
        if not math.isfinite(target) or target <= 0.:
            raise ValueError('scaled duration cannot be represented finitely')
        if quantum is None:
            durations.append(target)
            continue
        ratio = target / quantum
        if not math.isfinite(ratio):
            raise ValueError('time quantum requires an unrepresentable number of ticks')
        count = max(1, math.ceil(ratio))
        try:
            duration = count * quantum
        except OverflowError as exc:
            raise ValueError('quantized duration cannot be represented finitely') from exc
        if duration < target:
            count += 1
            try:
                duration = count * quantum
            except OverflowError as exc:
                raise ValueError('quantized duration cannot be represented finitely') from exc
        if not math.isfinite(duration):
            raise ValueError('quantized duration cannot be represented finitely')
        durations.append(duration)
        ticks.append(count)
    return tuple(durations), tuple(ticks) if quantum is not None else None


def _build(waypoints, durations, ticks, quantum):
    velocities = _node_velocities(waypoints, durations)
    segments, boundaries = [], [0.]
    elapsed_ticks = 0
    for index, duration in enumerate(durations):
        start_s = boundaries[-1]
        if ticks is None:
            end_s = start_s + duration
        else:
            elapsed_ticks += ticks[index]
            end_s = elapsed_ticks * quantum
        if not math.isfinite(end_s) or end_s <= start_s:
            raise ValueError('path boundary times cannot be represented distinctly')
        controls = _control_points(waypoints[index], waypoints[index + 1],
                                   velocities[index], velocities[index + 1], duration)
        velocity = _derivative(controls, duration)
        acceleration = _derivative(velocity, duration)
        jerk = _derivative(acceleration, duration)
        segments.append(BezierSegment(start_s, end_s, duration, controls, velocity, acceleration, jerk))
        boundaries.append(end_s)
    maxima = tuple(tuple(max(abs(row[axis]) for segment in segments for row in getattr(segment, field))
                         for axis in range(JOINT_COUNT)) for field in
                   ('velocity_control_points_deg_s', 'acceleration_control_points_deg_s2', 'jerk_control_points_deg_s3'))
    # Finite floating-point control points can lose a very small tangential
    # velocity at a large offset. Reject rather than claim a nonzero C1 node
    # that the actual polynomial no longer represents.
    for index, nominal in enumerate(velocities[1:-1], 1):
        left = segments[index - 1].velocity_control_points_deg_s[-1]
        right = segments[index].velocity_control_points_deg_s[0]
        for intended, before, after in zip(nominal, left, right, strict=True):
            if not (math.isclose(before, intended, rel_tol=1e-8, abs_tol=0.) and
                    math.isclose(after, intended, rel_tol=1e-8, abs_tol=0.)):
                raise ValueError('continuous waypoint velocity cannot be represented accurately')
    for segment in segments:
        for acceleration in (segment.acceleration_control_points_deg_s2[0],
                             segment.acceleration_control_points_deg_s2[-1]):
            for axis, value in enumerate(acceleration):
                if abs(value) > max(1e-12, 1e-10 * maxima[1][axis]):
                    raise ValueError('zero waypoint acceleration cannot be represented accurately')
    return tuple(segments), tuple(boundaries), velocities, maxima


def plan_blended_path(waypoints_deg, segment_durations_s, *, max_velocity_deg_s,
                      max_acceleration_deg_s2, max_jerk_deg_s3, time_quantum_s=None,
                      require_nonstop=True):
    """Plan a monotone, C2 path and conservatively certify derivative limits.

    ``waypoints_deg`` contains four-axis points including the start and final
    target. ``segment_durations_s`` supplies one positive initial duration per
    pair. Limits may be positive scalars or four-axis sequences. A single
    common time scale is increased as needed; optional quantization rounds each
    segment up to full fixed-period ticks and rechecks every derivative bound.

    A waypoint where all four prescribed derivatives vanish is rejected by
    default: it cannot be crossed without a complete stop under this monotone
    construction. ``require_nonstop=False`` exposes such points explicitly in
    ``stopped_waypoint_indices`` for a caller deciding how to handle them.
    """
    if not isinstance(waypoints_deg, (tuple, list)) or len(waypoints_deg) < 2:
        raise ValueError('at least two four-joint waypoints are required')
    waypoints = tuple(_vector(point, 'waypoint') for point in waypoints_deg)
    if not isinstance(segment_durations_s, (tuple, list)) or len(segment_durations_s) != len(waypoints) - 1:
        raise ValueError('one duration per waypoint pair is required')
    base_durations = tuple(_real(value, 'segment duration', positive=True) for value in segment_durations_s)
    limits = (_limits(max_velocity_deg_s, 'velocity limit'),
              _limits(max_acceleration_deg_s2, 'acceleration limit'),
              _limits(max_jerk_deg_s3, 'jerk limit'))
    quantum = None if time_quantum_s is None else _real(time_quantum_s, 'time quantum', positive=True)
    if not isinstance(require_nonstop, bool):
        raise ValueError('require_nonstop must be boolean')
    scale = 1.
    for _ in range(16):
        durations, ticks = _durations(base_durations, scale, quantum)
        segments, boundaries, velocities, maxima = _build(waypoints, durations, ticks, quantum)
        stopped = tuple(index for index, velocity in enumerate(velocities[1:-1], 1)
                        if all(value == 0. for value in velocity))
        if require_nonstop and stopped:
            raise ValueError(f'internal waypoints require a complete stop: {stopped}')
        ratios = []
        for actual, bound, power in zip(maxima, limits, (1., .5, 1. / 3.), strict=True):
            for value, limit in zip(actual, bound, strict=True):
                ratio = value / limit
                if not math.isfinite(ratio):
                    raise ValueError('required time scale cannot be represented finitely')
                ratios.append(ratio ** power)
        factor = max(1., *ratios)
        if all(value <= limit for actual, bound in zip(maxima, limits, strict=True)
               for value, limit in zip(actual, bound, strict=True)):
            return BlendedPath(waypoints, base_durations, *limits, require_nonstop, segments, boundaries,
                               velocities, stopped, boundaries[-1], scale, quantum, ticks, *maxima)
        scaled = scale * factor
        if not math.isfinite(scaled):
            raise ValueError('required time scale cannot be represented finitely')
        next_scale = math.nextafter(max(scale, scaled), math.inf)
        if quantum is not None:
            _, next_ticks = _durations(base_durations, next_scale, quantum)
            if next_ticks == ticks:
                # A small limit violation can repeatedly scale the requested
                # durations without changing any rounded segment duration.
                # Advance past the first representable tick boundary and
                # rebuild all derivatives before deciding whether it is safe.
                boundary = min(math.nextafter(duration, math.inf) / base
                               for base, duration in zip(base_durations, durations, strict=True))
                next_scale = math.nextafter(max(next_scale, boundary), math.inf)
                if not math.isfinite(next_scale):
                    raise ValueError('next quantized time boundary cannot be represented finitely')
                _, next_ticks = _durations(base_durations, next_scale, quantum)
                if next_ticks == ticks:
                    raise ValueError('next quantized time boundary cannot be represented accurately')
        scale = next_scale
    raise ValueError('could not certify derivative limits after time scaling')
