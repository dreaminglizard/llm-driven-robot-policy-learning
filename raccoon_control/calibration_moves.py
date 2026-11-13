"""Small commissioning moves before a task workspace has been measured.

These step sizes are commissioning settings, not the robot's physical reach.
No device access, workspace approval or calibration-success inference occurs here.
"""
import math

from .errors import ControlError
from .kinematics import check_joints, distance, quantize_joints
from .validation import number, vector

MAX_JOINT_STEP_DEG = 2.0
MAX_TCP_STEP_MM = 5.0
MAX_SPEED_PCT = 15.0


def prepare_step(geometry, start, action, limits):
    start = check_joints(start)
    speed = number(action.get("speed_pct", 12), "speed_pct", minimum=1,
                   maximum=min(MAX_SPEED_PCT, limits.max_speed_pct))
    if action["kind"] == "jog_joint":
        axis = action.get("joint")
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in range(1, 5):
            raise ControlError("INVALID_JOINT", "Choose joint 1, 2, 3 or 4")
        delta = number(action.get("delta_deg"), "delta_deg",
                       minimum=-MAX_JOINT_STEP_DEG, maximum=MAX_JOINT_STEP_DEG)
        target = list(start)
        target[axis - 1] += delta
        excursion_limit = 12.0
    elif action["kind"] == "jog_tcp":
        delta = vector(action.get("delta_mm"), 3, "delta_mm")
        if distance(delta, (0, 0, 0)) > MAX_TCP_STEP_MM:
            raise ControlError("CALIBRATION_STEP_LIMIT", "One Cartesian calibration step is at most 5 mm")
        origin = geometry.forward(start)
        # Preserve the observed pitch; jogging must not silently level the tool.
        target = geometry.inverse(tuple(a + b for a, b in zip(origin, delta)),
                                  pitch_deg=sum(start[1:]) + 90)
        excursion_limit = 6.0  # Includes the encoder quantization residual.
    else:
        raise ControlError("UNSUPPORTED_ACTION", "Expected a calibration jog")
    steps, effective = quantize_joints(target)
    check_joints(effective)
    # A tiny Cartesian step near a singularity must not turn into a large joint move.
    if max(abs(a - b) for a, b in zip(start, effective)) > 5.0:
        raise ControlError("CALIBRATION_JOINT_CHANGE", "This Cartesian step needs a smaller increment near this pose")
    count = max(1, math.ceil(max(abs(a - b) for a, b in zip(start, effective)) / limits.path_step_deg))
    origin = geometry.forward(start)
    minimum_z, maximum_excursion = math.inf, 0.0
    for i in range(count + 1):
        point = geometry.forward(tuple(a + (b - a) * i / count for a, b in zip(start, effective)))
        minimum_z = min(minimum_z, point[2])
        maximum_excursion = max(maximum_excursion, distance(point, origin))
    if minimum_z < 0:
        raise ControlError("CALIBRATION_BELOW_BASE", "The nominal tool path crosses below the robot's base plane")
    if maximum_excursion > excursion_limit:
        raise ControlError("CALIBRATION_STEP_LIMIT", "The sampled tool path exceeds the calibration step size")
    # Honor an explicitly configured box, but do not require a fabricated one
    # merely to begin collecting calibration measurements.
    if limits.workspace_configured:
        geometry.check_path(start, effective, limits)
    return {
        "kind": action["kind"], "calibration_step": True,
        "requested_joints_deg": target, "effective_target_deg": effective,
        "steps": steps, "target_tcp_mm": geometry.forward(effective),
        "speed_pct": speed, "allow_visual_settle": False,
        "path": {"samples": count + 1, "minimum_tcp_z_mm": minimum_z,
                 "maximum_tcp_excursion_mm": maximum_excursion,
                 "workspace_bounds_checked": limits.workspace_configured,
                 "geometry_source": "nominal", "full_collision_check": False},
    }
