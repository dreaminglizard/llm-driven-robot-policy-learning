"""Nominal position IK with hard joint limits and nearest feasible tool pitch.

The requested XYZ is fixed. Feasible pitch intervals are delimited by analytic
joint-boundary circle intersections, rather than by clipping joint targets.
Only nominal geometry and the proposed pose are used; no scene state is read.
"""
from __future__ import annotations

import cmath
import math

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from .scene_causal_frame_policy import inverse_or_reject


def _circle_pitch_boundaries(point, rotating, length, lower, upper):
    # |point - rotating*exp(i*pitch)| == length.
    product = point * rotating.conjugate()
    amplitude = abs(product)
    rhs = (abs(point)**2 + abs(rotating)**2 - length**2) / 2
    if amplitude <= 1e-12:
        return []
    ratio = rhs / amplitude
    if not -1-1e-12 <= ratio <= 1+1e-12:
        return []
    center = cmath.phase(product)
    offset = math.acos(max(-1., min(1., ratio)))
    roots = []
    for root in (center-offset, center+offset):
        first = math.ceil((lower-root)/(2*math.pi)-1e-12)
        last = math.floor((upper-root)/(2*math.pi)+1e-12)
        roots.extend(root+turn*2*math.pi for turn in range(first, last+1))
    return roots


def _geometry_inverse(target, geometry):
    x, y, z, degrees = target
    pitch, radius = math.radians(degrees), math.hypot(x, y)
    arm_r = radius-geometry.tool_forward_mm*math.cos(pitch)+geometry.tool_vertical_mm*math.sin(pitch)
    arm_z = z-geometry.base_mm-geometry.tool_forward_mm*math.sin(pitch)-geometry.tool_vertical_mm*math.cos(pitch)
    cosine = (arm_r**2+arm_z**2-geometry.upper_mm**2-geometry.forearm_mm**2)/(2*geometry.upper_mm*geometry.forearm_mm)
    if not math.isfinite(cosine) or not -1-1e-10 <= cosine <= 1+1e-10:
        return [999.]*4, False
    elbow = -math.acos(max(-1., min(1., cosine)))
    shoulder = math.atan2(-arm_r, arm_z)-math.atan2(geometry.forearm_mm*math.sin(elbow),
        geometry.upper_mm+geometry.forearm_mm*math.cos(elbow))
    yaw = math.atan2(-x, y)
    wrist = pitch-math.pi/2-shoulder-elbow
    return [math.degrees(v) for v in (yaw, shoulder, elbow, wrist)], True


def inverse_nearest_feasible_pitch(target, *, geometry=None):
    """Keep XYZ on the policy's negative-elbow, positive-radius IK branch.

    Minimize unwrapped pitch change, or reject without a command.
    """
    if len(target) != 4 or not all(math.isfinite(v) for v in target):
        raise ValueError("one finite nominal XYZ and tool pitch required")
    x, y, z, requested_degrees = map(float, target)
    radius = math.hypot(x, y)
    if radius < 1e-6:
        raise ValueError("nominal target has no defined base yaw")
    yaw = math.degrees(math.atan2(-x, y))
    if not JOINT_LIMITS[0][0] <= yaw <= JOINT_LIMITS[0][1]:
        raise ValueError("nominal target yaw is outside joint limits")
    nominal_geometry = geometry is None
    geometry = Geometry() if nominal_geometry else geometry
    if not isinstance(geometry, Geometry) or any(not math.isfinite(v) for v in
            (geometry.base_mm, geometry.upper_mm, geometry.forearm_mm,
             geometry.tool_forward_mm, geometry.tool_vertical_mm)) or min(geometry.upper_mm, geometry.forearm_mm) <= 0:
        raise ValueError("finite nominal geometry with positive arm lengths required")
    requested = math.radians(requested_degrees)
    lower = math.radians(90+sum(bounds[0] for bounds in JOINT_LIMITS[1:]))
    upper = math.radians(90+sum(bounds[1] for bounds in JOINT_LIMITS[1:]))
    point = complex(radius, z-geometry.base_mm)
    tool = complex(geometry.tool_forward_mm, geometry.tool_vertical_mm)
    cuts = [lower, upper]
    for shoulder in JOINT_LIMITS[1]:
        fixed_upper = geometry.upper_mm * cmath.exp(1j*math.radians(90+shoulder))
        cuts.extend(_circle_pitch_boundaries(point-fixed_upper, tool, geometry.forearm_mm, lower, upper))
    for elbow in (*JOINT_LIMITS[2], -180.):
        reach = abs(geometry.upper_mm+geometry.forearm_mm*cmath.exp(1j*math.radians(elbow)))
        cuts.extend(_circle_pitch_boundaries(point, tool, reach, lower, upper))
    for wrist in JOINT_LIMITS[3]:
        rotating = tool+geometry.forearm_mm*cmath.exp(-1j*math.radians(wrist))
        cuts.extend(_circle_pitch_boundaries(point, rotating, geometry.upper_mm, lower, upper))
    cuts = sorted(set(max(lower, min(upper, value)) for value in cuts))

    def solution(pitch):
        pose = [x, y, z, math.degrees(pitch)]
        joints, reachable = (inverse_or_reject(pose) if nominal_geometry else _geometry_inverse(pose, geometry))
        if not reachable:
            return None
        # The planar atan2 representation can differ by a full shoulder turn.
        joints[1] = (joints[1]+180.) % 360.-180.
        joints[3] = math.degrees(pitch)-90.-joints[1]-joints[2]
        if any(q < lo-1e-7 or q > hi+1e-7 for q, (lo, hi) in zip(joints, JOINT_LIMITS)):
            return None
        joints = [max(lo, min(hi, q)) for q, (lo, hi) in zip(joints, JOINT_LIMITS)]
        if math.dist(geometry.forward(joints), [x, y, z]) > 1e-5:
            return None
        return joints

    candidates = list(cuts)
    candidates.append(max(lower, min(upper, requested)))
    for start, end in zip(cuts, cuts[1:]):
        if solution((start+end)/2) is not None:
            candidates.append(max(start, min(end, requested)))
    feasible = [(abs(p-requested), p, q) for p in candidates if (q := solution(p)) is not None]
    if not feasible:
        raise ValueError("no nominal position IK within joint limits")
    _, selected, joints = min(feasible, key=lambda item: (item[0], item[1]))
    return joints, {"requested_pitch_deg": requested_degrees,
                    "feasible_pitch_deg": math.degrees(selected),
                    "pitch_change_deg": math.degrees(selected-requested),
                    "nominal_position_error_mm": math.dist(geometry.forward(joints), [x, y, z])}
