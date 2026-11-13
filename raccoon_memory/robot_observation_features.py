"""Pure-Python robot observation features shared by collection and learning."""
from __future__ import annotations

import math


FEATURE_NAMES = (
    "joint_1_deg/180", "joint_2_deg/180", "joint_3_deg/180", "joint_4_deg/180",
    "sensor_valid", "moving",
    "gripper_command_none", "gripper_command_open", "gripper_command_closed",
    "gripper_reported_none", "gripper_reported_open", "gripper_reported_closed",
    "gripper_reported_valid", "gripper_feedback_fresh",
)


def _required(mapping, key):
    if key not in mapping:
        raise ValueError(f"missing observation.{key}")
    return mapping[key]


def _flag(mapping, key):
    value = _required(mapping, key)
    if type(value) is not bool:
        raise ValueError(f"observation.{key} must be boolean")
    return value


def encode(observation):
    """Validate and encode the same 14 fields used by motion_policy."""
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    if _required(observation, "schema_version") != "PolicyObservation.phase1":
        raise ValueError("unsupported observation schema")
    joints = _required(observation, "joints_deg")
    if not isinstance(joints, (list, tuple)) or len(joints) != 4:
        raise ValueError("observation.joints_deg must have four joint angles")
    converted = []
    for joint in joints:
        if type(joint) not in (int, float):
            raise ValueError("observation.joints_deg must be a finite number")
        try:
            joint = float(joint)
        except (OverflowError, ValueError):
            raise ValueError("observation.joints_deg must be a finite number") from None
        if not math.isfinite(joint):
            raise ValueError("observation.joints_deg must be a finite number")
        converted.append(joint)
    valid, moving = _flag(observation, "sensor_valid"), _flag(observation, "moving")
    report_valid = _flag(observation, "gripper_reported_valid")
    report_fresh = _flag(observation, "gripper_feedback_fresh")
    if not valid:
        raise ValueError("joint sensor is invalid")
    command = _required(observation, "gripper_command")
    report = _required(observation, "gripper_reported_state")
    if command not in (None, "open", "closed") or report not in (None, "open", "closed"):
        raise ValueError("unsupported gripper state")
    if (report_valid and (not report_fresh or report is None)) or (not report_valid and report is not None):
        raise ValueError("gripper report and validity disagree")
    result = [joint / 180. for joint in converted]
    result.extend((float(valid), float(moving)))
    result.extend(float(command == value) for value in (None, "open", "closed"))
    result.extend(float(report == value) for value in (None, "open", "closed"))
    result.extend((float(report_valid), float(report_fresh)))
    if len(result) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in result):
        raise ValueError("non-finite encoded feature")
    return result
