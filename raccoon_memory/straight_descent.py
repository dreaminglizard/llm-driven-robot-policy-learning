"""Checked nominal Cartesian descent for the selected virtual student.

The only inputs are registered slot coordinates, expected support, and the
previous and guarded joint targets. This is not a physical robot controller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, JOINT_LIMITS, check_joints

GEOMETRY = Geometry()
TICKS = 1000
TIMESTEP = .002
FOOTPRINT_MM = 10.


@dataclass(frozen=True)
class CheckedDescent:
    start: tuple
    target: tuple
    # Immutable radian targets, one per existing 2 ms MuJoCo integration tick.
    targets_rad: tuple


def eligible(start, target, destination, has_support):
    p0, p1 = GEOMETRY.forward(start), GEOMETRY.forward(target)
    return (has_support and p1[2] < p0[2]
            and max(math.dist(p[:2], destination) for p in (p0, p1)) <= FOOTPRINT_MM)


def prepare(start, target, destination, has_support, cancel=None):
    """Preflight exactly the 1,000 cubic targets or retain the old joint path."""
    start, target = tuple(check_joints(start)), tuple(check_joints(target))
    info = {'eligible': bool(eligible(start, target, destination, has_support)),
            'changed': False, 'fallback': False}
    if not info['eligible']:
        return None, info

    def interrupted():
        if cancel is not None and cancel.is_set():
            raise InterruptedError('virtual descent cancelled during path preparation')

    interrupted()
    p0, p1 = GEOMETRY.forward(start), GEOMETRY.forward(target)
    pitch0, pitch1 = (90. + sum(q[1:]) for q in (start, target))
    try:
        if max(abs(a - b) for a, b in zip(GEOMETRY.inverse(p0, pitch0), start, strict=True)) > 1e-6:
            info['fallback'] = True
            return None, info
        rows = []
        for tick in range(1, TICKS + 1):
            interrupted()
            t = tick / TICKS
            u = t*t*(3.-2.*t)
            p = [a+u*(b-a) for a, b in zip(p0, p1, strict=True)]
            pitch = pitch0 + u*(pitch1-pitch0)
            row = GEOMETRY.inverse(p, pitch)
            if abs(90.+sum(row[1:])-pitch) > 1e-7:
                raise ValueError('nominal pitch inverse differs')
            rows.append(row)
        if max(abs(a-b) for a, b in zip(rows[-1], target, strict=True)) > 1e-6:
            info['fallback'] = True
            return None, info
        rows[-1] = target
    except ControlError:
        info['fallback'] = True
        return None, info
    interrupted()
    degree_rows = (start, *rows)
    margin = min(min(value-low, high-value) for row in degree_rows
                 for value, (low, high) in zip(row, JOINT_LIMITS, strict=True))
    peak_speed = max(abs(b-a)/TIMESTEP for left, right in zip(degree_rows, degree_rows[1:])
                     for a, b in zip(left, right, strict=True))
    peak_acceleration = max(abs(c-2*b+a)/TIMESTEP**2
                            for left, middle, right in zip(degree_rows, degree_rows[1:], degree_rows[2:])
                            for a, b, c in zip(left, middle, right, strict=True))
    info.update(changed=True, ticks=TICKS, smallest_target_margin_deg=margin,
                new_peak_target_deg_s=peak_speed, new_peak_target_deg_s2=peak_acceleration)
    return CheckedDescent(start, target, tuple(tuple(math.radians(v) for v in row) for row in rows)), info
