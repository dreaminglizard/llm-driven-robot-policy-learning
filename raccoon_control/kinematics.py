"""Raccoon geometry in millimetres; no SDK imports and no device I/O.

This checks sampled TCP paths, NOT complete link/self collision geometry.
"""
from dataclasses import dataclass
import math

from .errors import ControlError
from .validation import number, vector

JOINT_LIMITS = ((-120.0, 120.0), (-90.0, 30.0), (-150.0, 0.0), (-105.0, 105.0))
JOINT_COUNT = len(JOINT_LIMITS)
STEPS_PER_DEGREE = 4096.0 / 360.0
ENCODER_STEP_DEG = 1.0 / STEPS_PER_DEGREE


def check_joints(values):
    joints = vector(values, JOINT_COUNT, "joints_deg")
    for i, (value, (low, high)) in enumerate(zip(joints, JOINT_LIMITS)):
        if not low <= value <= high:
            raise ControlError("JOINT_LIMIT", f"Joint {i + 1} is outside its range")
    return joints


def quantize_joints(values):
    requested = check_joints(values)
    # Match the pinned SDK encoder, including its use of Python round().
    steps = tuple(round(x * STEPS_PER_DEGREE) for x in requested)
    return steps, tuple(step / STEPS_PER_DEGREE for step in steps)


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class Geometry:
    base_mm: float = 82.5
    upper_mm: float = 100.0
    forearm_mm: float = 100.0
    tool_forward_mm: float = 80.0
    tool_vertical_mm: float = -6.0

    def forward(self, joints):
        q1, q2, q3, q4 = (math.radians(x) for x in vector(joints, JOINT_COUNT, "joints_deg"))
        pitch = q2 + q3 + q4 + math.pi / 2
        radius = (-self.upper_mm * math.sin(q2) - self.forearm_mm * math.sin(q2 + q3)
                  + self.tool_forward_mm * math.cos(pitch) - self.tool_vertical_mm * math.sin(pitch))
        z = (self.base_mm + self.upper_mm * math.cos(q2) + self.forearm_mm * math.cos(q2 + q3)
             + self.tool_forward_mm * math.sin(pitch) + self.tool_vertical_mm * math.cos(pitch))
        return (-radius * math.sin(q1), radius * math.cos(q1), z)

    def inverse(self, xyz, pitch_deg=0.0):
        x, y, z = vector(xyz, 3, "target_mm")
        pitch = math.radians(number(pitch_deg, "pitch_deg"))
        radius = math.hypot(x, y)
        if radius < 1e-6:
            raise ControlError("IK_SINGULAR", "The base-axis target has no unique yaw")
        arm_r = radius - self.tool_forward_mm * math.cos(pitch) + self.tool_vertical_mm * math.sin(pitch)
        arm_z = z - self.base_mm - self.tool_forward_mm * math.sin(pitch) - self.tool_vertical_mm * math.cos(pitch)
        cosine = (arm_r ** 2 + arm_z ** 2 - self.upper_mm ** 2 - self.forearm_mm ** 2) / (2 * self.upper_mm * self.forearm_mm)
        if not -1.0 - 1e-10 <= cosine <= 1.0 + 1e-10:
            raise ControlError("IK_UNREACHABLE", "Target is beyond the arm's geometric reach")
        q3 = -math.acos(max(-1.0, min(1.0, cosine)))
        q2 = math.atan2(-arm_r, arm_z) - math.atan2(self.forearm_mm * math.sin(q3), self.upper_mm + self.forearm_mm * math.cos(q3))
        q1 = math.atan2(-x, y)
        q4 = pitch - math.pi / 2 - q2 - q3
        joints = check_joints(tuple(math.degrees(v) for v in (q1, q2, q3, q4)))
        if distance(self.forward(joints), (x, y, z)) > 1e-5:
            raise ControlError("IK_MISMATCH", "Forward verification of the IK target failed")
        return joints

    def check_path(self, start, target, limits):
        if limits.workspace_min_mm is None or limits.workspace_max_mm is None:
            raise ControlError("WORKSPACE_UNSPECIFIED", "No workspace bounds have been configured; mock bounds are not a physical workspace")
        start = check_joints(start)
        target = check_joints(target)
        count = max(1, math.ceil(max(abs(a - b) for a, b in zip(start, target)) / limits.path_step_deg))
        minimum_z = math.inf
        for i in range(count + 1):
            q = tuple(a + (b - a) * i / count for a, b in zip(start, target))
            point = self.forward(q)
            minimum_z = min(minimum_z, point[2])
            if any(p < low or p > high for p, low, high in zip(point, limits.workspace_min_mm, limits.workspace_max_mm)):
                raise ControlError("PATH_OUTSIDE_WORKSPACE", "The sampled TCP path leaves the approved workspace", sample=i)
        return {"samples": count + 1, "minimum_tcp_z_mm": minimum_z, "full_collision_check": False}
