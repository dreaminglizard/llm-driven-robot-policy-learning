"""Offline teacher experiment: one common, nominal radial placement bias.

The 4 mm inward target was chosen explicitly from prior *offline* evidence of
outward placement error. It is a hypothesis, not a measured object estimate.
This module supplies a waypoint function to the existing observable teacher;
the emitted low-level decisions are therefore program-generated labels, not
individual actions written by the model. No object coordinates are accessed.
"""
from __future__ import annotations

import math

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from raccoon_memory import scene_nominal_kinematic_teacher as nominal


def biased_waypoints(row, bias_mm=4.0):
    if type(bias_mm) not in (int, float) or not math.isfinite(bias_mm) or not 0.0 <= bias_mm <= 8.0:
        raise ValueError("a finite 0..8 mm inward bias is required")
    source, path, _ = nominal._waypoints(row)
    destination = nominal._one_hot(row, 18, 4)
    x = float(row[74 + 2 * destination]) * 300.0
    y = float(row[75 + 2 * destination]) * 300.0
    radius = math.hypot(x, y)
    if radius <= bias_mm:
        raise ValueError("destination radius cannot accommodate bias")
    level = nominal._one_hot(row, 24, 2)
    geo = Geometry()
    heights = (max(60.0, 35.0 + 20.0 * level), 15.0 + 20.0 * level, 100.0)

    def coordinates(amount):
        scale = (radius - amount) / radius
        return [list(geo.inverse((x * scale, y * scale, z))) for z in heights]

    def safe(amount):
        try:
            points = coordinates(amount)
        except ControlError:
            return False
        return all(lo + 1.0 <= joint <= hi - 1.0
                   for point in points for joint, (lo, hi) in zip(point, JOINT_LIMITS))

    if not safe(0.0):
        raise ValueError("nominal destination has insufficient joint-limit margin")
    low, high = 0.0, float(bias_mm)
    if safe(high):
        low = high
    else:
        for _ in range(24):
            mid = (low + high) / 2.0
            if safe(mid):
                low = mid
            else:
                high = mid
    approach, placement, retreat = coordinates(low)
    path[-2:] = [approach, placement]
    return source, path, retreat
