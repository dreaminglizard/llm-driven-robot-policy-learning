"""A deterministic teacher whose inputs are exactly the student inputs.

Only the 8x82 causal window, its mask, and the last applied joint target
are consulted. Waypoint progress is reconstructed from those values; there
is no private stage counter, object pose, contact, or physical-success input.
The geometry is nominal and this module makes no claim of physical success.
"""
from __future__ import annotations

import math

from raccoon_control.kinematics import Geometry, check_joints
from raccoon_memory import motion_policy, scene_features


TEACHER = "observable_nominal_kinematic_teacher_v2_reached"
# These common nominal tolerances apply to every slot, level and waypoint.
# The measured joint tolerance permits normal simulated servo compliance;
# it is not an observation of an object's actual position or contact.
PRIOR_JOINT_MAX_ERROR_DEG = 1.5
PRIOR_XY_ERROR_MM = 3.0
PRIOR_Z_ERROR_MM = 3.0
OBSERVED_TO_PRIOR_MAX_ERROR_DEG = 2.0
OBSERVED_XY_ERROR_MM = 3.0
OBSERVED_Z_ERROR_MM = 6.0
OBSERVED_PITCH_ERROR_DEG = 3.0


def _one_hot(row, start, count):
    values = list(row[start:start + count])
    if len(values) != count or any(v not in (0, 1) for v in values) or sum(values) != 1:
        raise ValueError("invalid categorical feature")
    return values.index(1)


def _distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _reached(row, prior, goal):
    """Test nominal arrival using current sensed joints and applied target.

    This is a training-teacher decision only. ``goal`` is derived from the
    student's registered nominal scene. No object pose/contact is used.
    """
    observed = [float(value) * motion_policy.JOINT_SCALE_DEG for value in row[:4]]
    if (len(observed) != 4 or any(not math.isfinite(x) for x in observed) or
            len(prior) != 4 or len(goal) != 4):
        raise ValueError("four finite observed and applied joints are required")
    if (max(abs(a-b) for a,b in zip(prior, goal)) > PRIOR_JOINT_MAX_ERROR_DEG or
            max(abs(a-b) for a,b in zip(observed, prior)) > OBSERVED_TO_PRIOR_MAX_ERROR_DEG):
        return False
    geometry = Geometry()
    target_xyz, commanded_xyz, observed_xyz = (geometry.forward(q) for q in (goal, prior, observed))
    def xy_error(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    return (xy_error(commanded_xyz, target_xyz) <= PRIOR_XY_ERROR_MM and
            abs(commanded_xyz[2]-target_xyz[2]) <= PRIOR_Z_ERROR_MM and
            xy_error(observed_xyz, target_xyz) <= OBSERVED_XY_ERROR_MM and
            abs(observed_xyz[2]-target_xyz[2]) <= OBSERVED_Z_ERROR_MM and
            abs(sum(observed[1:])-sum(goal[1:])) <= OBSERVED_PITCH_ERROR_DEG)


def _waypoints(row):
    src, dst = _one_hot(row, 14, 4), _one_hot(row, 18, 4)
    sl, dl = _one_hot(row, 22, 2), _one_hot(row, 24, 2)
    if src == dst:
        raise ValueError("source and destination must differ")
    xy = [(float(row[74 + 2*i]) * 300, float(row[75 + 2*i]) * 300) for i in range(4)]
    geo = Geometry()
    def point(slot, height):
        return list(geo.inverse((*xy[slot], height)))
    source = [point(src, 100), point(src, 40 + 20*sl), point(src, 15 + 20*sl)]
    lift, clear = point(src, 60 + 20*sl), point(src, 130)
    above = point(dst, 130)
    # Small, fixed nominal segments reduce inertial slip; neither the choice
    # nor the segment count depends on object/contact observations.
    n = max(1, math.ceil(abs(above[0] - clear[0]) / 10.0))
    rotation = [[clear[0] + (above[0] - clear[0])*i/n, *clear[1:]] for i in range(1, n+1)]
    # At equal source/destination radii the last rotation already is above.
    route = [*rotation]
    source_radius = math.hypot(*xy[src])
    destination_radius = math.hypot(*xy[dst])
    radial_segments = max(1, math.ceil(abs(source_radius - destination_radius) / 15.0))
    for i in range(1, radial_segments + 1):
        radius = source_radius + (destination_radius - source_radius) * i / radial_segments
        xyz = (xy[dst][0] * radius / destination_radius, xy[dst][1] * radius / destination_radius, 130.)
        target = list(geo.inverse(xyz))
        if _distance(route[-1], target) > 1e-5:
            route.append(target)
    destination = [point(dst, max(60, 35 + 20*dl)), point(dst, 15 + 20*dl)]
    retreat = point(dst, 100)
    return source, [lift, clear, *route, *destination], retreat


def decide(window, mask, previous_target_degrees):
    """Return (action, joint target) using only a causal student input.

    The first open observation approaches the source. Subsequent steps choose
    the nearest previous nominal command within the current gripper phase and
    require both the applied target and observed-joint nominal FK to reach it.
    A recent closed observation distinguishes release from the initial open
    state. Returned joint targets are absolute degrees; for non-MOVE actions
    they remain the supplied applied target. This is a stateless function.
    """
    if (not isinstance(window, (list, tuple)) or len(window) != 8 or
            not isinstance(mask, (list, tuple)) or len(mask) != 8 or
            any(type(m) is not bool for m in mask) or not mask[-1] or
            any(a and not b for a, b in zip(mask, mask[1:]))):
        raise ValueError("a left-padded eight-observation window is required")
    rows = [list(row) for row, valid in zip(window, mask) if valid]
    if any(len(row) != 82 or any(not math.isfinite(float(v)) for v in row) for row in rows):
        raise ValueError("invalid observation features")
    prior = list(check_joints(previous_target_degrees))
    row = rows[-1]
    released = any(_one_hot(old, 6, 3) == 2 for old in rows[:-1])
    return _select_action(row, prior, len(rows), released, _waypoints(row))


def _select_action(row, prior, history_length, prior_closed, waypoints):
    """Shared pure decision rule; waypoints must derive from this row."""
    if row[4] != 1:
        raise ValueError("valid joint observation is required")
    command = _one_hot(row, 6, 3)
    if command not in (1, 2):
        raise ValueError("known open or closed gripper command required")
    source, closed_path, retreat = waypoints
    if command == 1:
        if prior_closed:
            if _reached(row, prior, retreat):
                return motion_policy.DONE, prior
            return motion_policy.MOVE, retreat
        if history_length == 1:
            return motion_policy.MOVE, source[0]
        index = min(range(len(source)), key=lambda i: _distance(prior, source[i]))
        if not _reached(row, prior, source[index]):
            return motion_policy.MOVE, source[index]
        if index == len(source) - 1:
            return motion_policy.CLOSE, prior
        return motion_policy.MOVE, source[index + 1]
    # On first closed observation the previous target is the grasp waypoint.
    path = [source[-1], *closed_path]
    index = min(range(len(path)), key=lambda i: _distance(prior, path[i]))
    if not _reached(row, prior, path[index]):
        return motion_policy.MOVE, path[index]
    if index == len(path) - 1:
        return motion_policy.OPEN, prior
    return motion_policy.MOVE, path[index + 1]
