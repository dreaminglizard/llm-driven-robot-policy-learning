"""Pure checked path subdivision; no SDK or controller access."""
import math

from .errors import ControlError
from .kinematics import check_joints, quantize_joints
from .validation import vector


def compact_segments(start_joints,segments,validate_segment):
    """Remove waypoints only after validating each replacement segment.

    Original endpoints, final target and validation rules remain unchanged.
    A failed shortcut falls back to a shorter checked segment. This is bounded
    by the existing path's waypoint count and does not dispatch any motion.
    """
    previous=check_joints(start_joints)
    result=[]
    index=0
    while index<len(segments):
        for candidate in range(len(segments)-1,index-1,-1):
            segment=segments[candidate]
            try:
                evidence=validate_segment(previous,segment["effective"])
            except ControlError as exc:
                if candidate==index or exc.code not in {
                        "VERTICAL_CORRIDOR","ASYNC_ENVELOPE",
                        "PATH_OUTSIDE_WORKSPACE","TRANSFER_CLEARANCE"}:
                    raise
                continue
            result.append({**segment,"path":evidence})
            previous=segment["effective"]
            index=candidate+1
            break
    return result


def vertical_segments(geometry,start_joints,target_xyz,validate_segment,*,allow_async_subdivision=False,
                      compact=False):
    """Subdivide into fully checked Cartesian waypoints before dispatch.

    By default only a vertical-corridor failure permits subdivision. The explicit
    asynchronous option also subdivides an independent-joint envelope failure.
    Joint limits and nominal-path workspace failures propagate unchanged. Every
    segment passes full validation before ANY segment is returned.
    """
    start=check_joints(start_joints)
    target=vector(target_xyz,3,"target_mm")
    origin=geometry.forward(start)
    last_error=None
    for count in (1,2,4,8,16,32,64):
        previous=start
        result=[]
        try:
            for index in range(1,count+1):
                fraction=index/count
                xyz=tuple(a+(b-a)*fraction for a,b in zip(origin,target))
                requested=geometry.inverse(xyz)
                _,effective=quantize_joints(requested)
                evidence=validate_segment(previous,effective)
                result.append({"requested":requested,"effective":effective,"path":evidence})
                previous=effective
            return compact_segments(start,result,validate_segment) if compact else result
        except ControlError as exc:
            if exc.code!="VERTICAL_CORRIDOR" and not (allow_async_subdivision and exc.code=="ASYNC_ENVELOPE"):
                raise
            last_error=exc
    raise ControlError(last_error.code,"No checked Cartesian path fits within 64 segments",last_error=last_error.as_dict())


def radial_yaw_segments(geometry,start_joints,target_xyz,validate_segment,*,compact=False):
    """Separate radial travel from base rotation at the transport height.

    Try the larger-radius arc first, then the other ordering if it leaves the
    configured workspace. Neither candidate reaches the controller until every
    segment passes validation. Base turns are split at 20 degrees so a long
    turn does not consume the entire per-command time budget at trial speed.
    """
    start=check_joints(start_joints)
    target=vector(target_xyz,3,"target_mm")
    desired=geometry.inverse(target)
    origin=geometry.forward(start)
    start_radius,target_radius=math.hypot(*origin[:2]),math.hypot(*target[:2])
    orders=(("radial","yaw"),("yaw","radial"))
    if target_radius<start_radius:
        orders=orders[::-1]
    failures=[]
    for order in orders:
        previous=start
        result=[]
        try:
            for phase in order:
                if phase=="radial":
                    angle=math.radians(previous[0])
                    waypoint=(-target_radius*math.sin(angle),target_radius*math.cos(angle),target[2])
                    segments=vertical_segments(geometry,previous,waypoint,validate_segment,
                                               allow_async_subdivision=True,compact=compact)
                    for segment in segments:
                        result.append({**segment,"transport_phase":"radial"})
                    previous=segments[-1]["effective"]
                else:
                    count=max(1,math.ceil(abs(desired[0]-previous[0])/20))
                    yaw_start=previous[0]
                    arm=previous[1:]
                    for index in range(1,count+1):
                        requested=(yaw_start+(desired[0]-yaw_start)*index/count,*arm)
                        _,effective=quantize_joints(requested)
                        evidence=validate_segment(previous,effective)
                        result.append({"requested":requested,"effective":effective,"path":evidence,
                                       "transport_phase":"yaw"})
                        previous=effective
            return result
        except ControlError as exc:
            if exc.code not in {"ASYNC_ENVELOPE","PATH_OUTSIDE_WORKSPACE","TRANSFER_CLEARANCE"}:
                raise
            failures.append(exc.as_dict())
    raise ControlError("TRANSPORT_PATH_UNAVAILABLE","Neither checked radial/yaw ordering fits the workspace",failures=failures)
