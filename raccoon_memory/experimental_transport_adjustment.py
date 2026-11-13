"""Device-free candidates for a measured, bounded manipulation experiment.

These helpers do not dispatch, inspect object poses, or certify contact safety.
The caller must pass only freshly observed joints and confirmed applied targets,
preflight every resulting segment, and record the controller's actual target.
This module is not enabled in the product robot session.
"""
from __future__ import annotations

import math

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import check_joints, quantize_joints
from raccoon_control.validation import number, vector


_REJECTED_SHORTCUT = {'ASYNC_ENVELOPE', 'VERTICAL_CORRIDOR', 'PATH_OUTSIDE_WORKSPACE', 'TRANSFER_CLEARANCE'}


def lowered_grasp_target(geometry, observed_joints, source_xy, corridor_mm, lower_mm):
    """Return an untransmitted nominal IK target under the observed TCP.

    The caller must preflight/subdivide this target before sending any motion.
    This checks a nominal TCP corridor only, not fingers, support or table.
    """
    joints = check_joints(observed_joints)
    xy = vector(source_xy, 2, 'source_xy_mm')
    corridor = number(corridor_mm, 'corridor_mm', minimum=0)
    lowering = number(lower_mm, 'lower_mm', minimum=0, maximum=6)
    xyz = geometry.forward(joints)
    if math.dist(xyz[:2], xy) > corridor:
        raise ValueError('preclose_not_at_registered_source')
    desired = (xyz[0], xyz[1], xyz[2] - lowering)
    return tuple(geometry.inverse(desired)), desired


def validated_bias_target(goal, applied, observed, validate):
    """Bound feedback bias to +/-0.5° and validate the replacement segment.

    On an expected corridor/envelope rejection, revalidate the original goal;
    all other errors propagate. A return value is a proposal, not a confirmed
    servo target. Arrival must still be evaluated against the nominal goal.
    """
    nominal = check_joints(goal)
    last_applied, measured = check_joints(applied), check_joints(observed)
    proposal = [g + max(-.5, min(.5, a - m)) for g, a, m in zip(nominal, last_applied, measured)]
    _, candidate = quantize_joints(proposal)
    if max(abs(a - b) for a, b in zip(candidate, nominal)) > .55:
        raise ValueError('predictive_trim_limit')
    try:
        validate(measured, candidate)
    except ControlError as exc:
        if exc.code not in _REJECTED_SHORTCUT:
            raise
        validate(measured, nominal)
        return tuple(nominal), exc.code
    return tuple(candidate), None


def source_floor_target(geometry, requested, source_xy, corridor_mm, nominal_height_mm):
    """Clamp a source-adjacent proposal above the registered nominal height.

    This is a geometric proposal only and never proves the object is held or
    clear of its support. The caller must preflight/subdivide the new path.
    """
    target = check_joints(requested)
    xy = vector(source_xy, 2, 'source_xy_mm')
    corridor = number(corridor_mm, 'corridor_mm', minimum=0)
    floor = number(nominal_height_mm, 'nominal_height_mm')
    x, y, z = geometry.forward(target)
    if math.dist((x, y), xy) <= corridor and z < floor:
        return tuple(geometry.inverse((x, y, floor))), {'from': z, 'to': floor}
    return tuple(target), None
