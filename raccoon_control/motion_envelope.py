"""Conservative nominal TCP bounds when joints progress independently.

Interval arithmetic covers every joint combination between endpoints, including
the supplied encoder tolerance. It does not model physical tool calibration or
whole-link collisions.
"""
import math

from .errors import ControlError
from .kinematics import JOINT_LIMITS, check_joints


def add(*values):
    return sum(v[0] for v in values), sum(v[1] for v in values)


def scale(value, factor):
    ends = (value[0] * factor, value[1] * factor)
    return min(ends), max(ends)


def multiply(a, b):
    products = [x*y for x in a for y in b]
    return min(products), max(products)


def sine(value):
    lo, hi = value
    points = [math.sin(lo), math.sin(hi)]
    for k in range(math.ceil((lo-math.pi/2)/math.pi), math.floor((hi-math.pi/2)/math.pi)+1):
        points.append(math.sin(math.pi/2+k*math.pi))
    return min(points), max(points)


def cosine(value):
    return sine(add(value, (math.pi/2, math.pi/2)))


def arm_bounds(geometry, ranges, depth=6):
    """Split joint intervals to retain the shared q2/q3/pitch correlations."""
    widths=[hi-lo for lo,hi in ranges]
    widest=max(range(3),key=widths.__getitem__)
    if depth and widths[widest]>math.radians(.5):
        lo,hi=ranges[widest]
        middle=(lo+hi)/2
        left,right=list(ranges),list(ranges)
        left[widest],right[widest]=(lo,middle),(middle,hi)
        lr,lz=arm_bounds(geometry,left,depth-1)
        rr,rz=arm_bounds(geometry,right,depth-1)
        return (min(lr[0],rr[0]),max(lr[1],rr[1])),(min(lz[0],rz[0]),max(lz[1],rz[1]))
    q2,q3,q4=ranges
    q23=add(q2,q3)
    pitch=add(q23,q4,(math.pi/2,math.pi/2))
    radius=add(scale(sine(q2),-geometry.upper_mm),scale(sine(q23),-geometry.forearm_mm),
               scale(cosine(pitch),geometry.tool_forward_mm),scale(sine(pitch),-geometry.tool_vertical_mm))
    z=add((geometry.base_mm,geometry.base_mm),scale(cosine(q2),geometry.upper_mm),
          scale(cosine(q23),geometry.forearm_mm),scale(sine(pitch),geometry.tool_forward_mm),
          scale(cosine(pitch),geometry.tool_vertical_mm))
    return radius,z


def bounds(geometry, start, target, tolerance_deg=0, *, start_tolerance_deg=None):
    start, target = check_joints(start), check_joints(target)
    if start_tolerance_deg is None:
        start_tolerance_deg = tolerance_deg
    ranges = [tuple(math.radians(v) for v in (max(limit[0], min(a-start_tolerance_deg,b-tolerance_deg)),
                                             min(limit[1], max(a+start_tolerance_deg,b+tolerance_deg))))
              for a,b,limit in zip(start,target,JOINT_LIMITS)]
    q1,q2,q3,q4 = ranges
    radius,z=arm_bounds(geometry,(q2,q3,q4))
    x = scale(multiply(radius,sine(q1)),-1)
    y = multiply(radius,cosine(q1))
    return {"minimum_mm":(x[0],y[0],z[0]), "maximum_mm":(x[1],y[1],z[1]),
            "radius_mm":radius,"base_yaw_rad":q1,
            "independent_joint_progress":True, "encoder_tolerance_deg":tolerance_deg,
            "start_tolerance_deg":start_tolerance_deg,
            "geometry_source":"nominal", "full_collision_check":False}


def check(envelope, limits, *, vertical_xy=None, corridor_mm=None, minimum_z_mm=None,maximum_z_mm=None):
    low, high = envelope["minimum_mm"], envelope["maximum_mm"]
    reason = None
    if limits.workspace_configured and any(a<lo or b>hi for a,b,lo,hi in zip(low,high,limits.workspace_min_mm,limits.workspace_max_mm)):
        reason = "workspace"
    if minimum_z_mm is not None and low[2]<minimum_z_mm:
        reason = "transport_clearance"
    if maximum_z_mm is not None and high[2]>maximum_z_mm:
        reason = "path_ceiling"
    deviation = None
    if vertical_xy is not None:
        if "radius_mm" in envelope and "base_yaw_rad" in envelope:
            reference_radius=math.hypot(*vertical_xy)
            reference_yaw=math.atan2(-vertical_xy[0],vertical_xy[1])
            angular_cosine=cosine(add(envelope["base_yaw_rad"],(-reference_yaw,-reference_yaw)))
            # A rotating radius cannot occupy every corner of its enclosing XY
            # rectangle. Preserve that correlation for the circular corridor.
            squared=max(r*r+reference_radius**2-2*r*reference_radius*c
                        for r in envelope["radius_mm"] for c in angular_cosine)
            deviation=math.sqrt(max(0,squared))
        else:
            deviation = math.hypot(*(max(abs(lo-c),abs(hi-c)) for lo,hi,c in zip(low[:2],high[:2],vertical_xy)))
        if deviation>corridor_mm:
            reason = "vertical_corridor"
    if reason:
        raise ControlError("ASYNC_ENVELOPE", "Independent joint progress needs smaller Cartesian segments",
                           reason=reason, envelope=envelope, maximum_lateral_deviation_mm=deviation)
    return {**envelope, "maximum_lateral_deviation_mm":deviation}
