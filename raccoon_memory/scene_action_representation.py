"""Separate joint-target and nominal TCP-plus-pitch output contracts.

The student predicts exactly four continuous values. Task-space inference
passes the predicted XYZ and pitch to the existing single-branch IK once.
There is no teacher target, default pitch, correction, or clipping fallback.
"""
from __future__ import annotations

import math

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, JOINT_LIMITS, check_joints


JOINT = "joint_absolute_v1"
TOOLPOSE = "tcp_xyz_pitch_v1"
REPRESENTATIONS = (JOINT, TOOLPOSE)
JOINT_SCALE = 180.0
XYZ_SCALE = 300.0
PITCH_SCALE = 180.0
PARAMETERIZATION = {JOINT: "absolute", TOOLPOSE: "tcp_xyz_pitch_v1"}


def pose_from_joints(joints):
    q = [float(value) for value in joints]
    if len(q) != 4 or not all(math.isfinite(value) for value in q):
        raise ValueError("four finite joint angles are required")
    xyz = list(Geometry().forward(q))
    pitch = q[1] + q[2] + q[3] + 90.0
    return [*xyz, pitch]


def label_from_joints(joints, representation):
    if representation == JOINT:
        q = [float(value) for value in joints]
        if len(q) != 4 or not all(math.isfinite(value) for value in q):
            raise ValueError("four finite joint angles are required")
        return [value / JOINT_SCALE for value in q]
    if representation == TOOLPOSE:
        x, y, z, pitch = pose_from_joints(joints)
        return [x / XYZ_SCALE, y / XYZ_SCALE, z / XYZ_SCALE, pitch / PITCH_SCALE]
    raise ValueError("unknown output representation")


def tool_axis(pose):
    x, y, _, pitch_deg = pose
    if not all(math.isfinite(float(value)) for value in pose) or math.hypot(x, y) < 1e-6:
        return None
    yaw = math.atan2(-x, y)
    pitch = math.radians(pitch_deg)
    return [-math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)]


def axis_error_deg(predicted_pose, target_pose):
    a, b = tool_axis(predicted_pose), tool_axis(target_pose)
    if a is None or b is None:
        return None
    # Round-off at identical axes may put the dot product a few ulps outside [-1,1].
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(cosine))


def decode_target(values, representation):
    """Return raw pose plus joint proposal or a named invalid proposal."""
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown output representation")
    values = [float(value) for value in values]
    if len(values) != 4:
        raise ValueError("model target must contain four outputs")
    if not all(math.isfinite(value) for value in values):
        return {"valid": False, "reason": "nonfinite_output", "raw_pose": None, "joints_deg": None,
                "minimum_joint_limit_margin_deg": None}
    if representation == JOINT:
        joints = [value * JOINT_SCALE for value in values]
        pose = pose_from_joints(joints)
        try:
            check_joints(joints)
        except ControlError as exc:
            if exc.code != "JOINT_LIMIT":
                raise
            return {"valid": False, "reason": exc.code, "raw_pose": pose, "joints_deg": None,
                    "minimum_joint_limit_margin_deg": None}
    else:
        pose = [values[0] * XYZ_SCALE, values[1] * XYZ_SCALE, values[2] * XYZ_SCALE,
                values[3] * PITCH_SCALE]
        try:
            joints = list(Geometry().inverse(pose[:3], pitch_deg=pose[3]))
        except ControlError as exc:
            if exc.code not in ("IK_SINGULAR", "IK_UNREACHABLE", "IK_MISMATCH", "JOINT_LIMIT"):
                raise
            return {"valid": False, "reason": exc.code, "raw_pose": pose, "joints_deg": None,
                    "minimum_joint_limit_margin_deg": None}
    margin = min(min(value - low, high - value) for value, (low, high) in zip(joints, JOINT_LIMITS))
    return {"valid": True, "reason": None, "raw_pose": pose, "joints_deg": joints,
            "minimum_joint_limit_margin_deg": margin}


def verify_contract(config, representation):
    if representation not in REPRESENTATIONS or not isinstance(config, dict):
        raise ValueError("missing representation contract")
    if (config.get("output_representation") != representation
            or config.get("output_parameterization") != PARAMETERIZATION[representation]
            or config.get("joint_scale_degrees") != JOINT_SCALE
            or (representation == TOOLPOSE and (config.get("xyz_scale_mm") != XYZ_SCALE
                                                  or config.get("pitch_scale_degrees") != PITCH_SCALE))):
        raise ValueError("checkpoint output representation contract differs")
