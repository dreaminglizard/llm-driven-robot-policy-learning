"""Conditional object-position estimates from registered scene and robot FK.

This is neither object sensing nor a controller. A closed gripper is assumed
to hold the requested object rigidly without slip. Estimates remain unconfirmed
even after a normal command return. Only explicit registration resets them.
"""
from __future__ import annotations

import math

from raccoon_control.kinematics import Geometry
from . import task_catalog


def tool_frame(joints):
    if (not isinstance(joints, (tuple, list)) or len(joints) != 4
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in joints)):
        raise ValueError("four finite observed joint angles required")
    point = Geometry().forward(joints)
    yaw = math.radians(joints[0])
    pitch = math.radians(sum(joints[1:])+90.0)
    radial = (-math.sin(yaw), math.cos(yaw), 0.0)
    tangent = (-math.cos(yaw), -math.sin(yaw), 0.0)
    forward = tuple(x*math.cos(pitch) for x in radial[:2])+(math.sin(pitch),)
    normal = tuple(-x*math.sin(pitch) for x in radial[:2])+(math.cos(pitch),)
    return point, (forward, tangent, normal)


def to_tool(point, frame):
    origin, axes = frame
    delta = [x-y for x, y in zip(point, origin)]
    return tuple(sum(a*b for a, b in zip(delta, axis)) for axis in axes)


def from_tool(offset, frame):
    origin, axes = frame
    return tuple(origin[i]+sum(offset[j]*axes[j][i] for j in range(3)) for i in range(3))


class NominalObjectBelief:
    def __init__(self, *, object_height_mm=20.0, center_lateral_on_close=False):
        if not math.isfinite(object_height_mm) or object_height_mm <= 0:
            raise ValueError("positive nominal object height required")
        self.object_height_mm = float(object_height_mm)
        if type(center_lateral_on_close) is not bool:
            raise ValueError("explicit lateral-centering mode required")
        self.center_lateral_on_close = center_lateral_on_close
        self.expected_scene = None
        self.active = None
        self.held = None
        self.previous_command = None
        self.offset = None
        self.points = {}
        self.states = {}

    def register(self, scene, slots):
        scene = task_catalog.scene(scene)
        if (not isinstance(slots, (tuple, list)) or len(slots) != 4
                or any(not isinstance(xy, (tuple, list)) or len(xy) != 2
                       or any(type(x) not in (float, int) or not math.isfinite(x) for x in xy) for xy in slots)):
            raise ValueError("four finite nominal XY calibration pairs required")
        self.points = {obj: (float(slots[slot][0]), float(slots[slot][1]), (level+.5)*self.object_height_mm)
                       for slot, stack in enumerate(scene) for level, obj in enumerate(stack)}
        self.states = {obj: "registered_nominal" for obj in self.points}
        self.expected_scene, self.active, self.held = scene, None, None
        self.previous_command, self.offset = None, None

    def begin_transfer(self, before, source, destination):
        before = task_catalog.scene(before)
        if self.active is not None or before != self.expected_scene or self.held is not None:
            raise ValueError("expected scene or transfer lifecycle differs")
        after = task_catalog.transfer(before, source, destination)
        self.active = {"object": before[source-1][-1], "after": after,
                       "destination_level": len(after[destination-1])}

    def observe(self, joints, gripper_command):
        if self.active is None or gripper_command not in ("open", "closed"):
            raise ValueError("registered active transfer and observed binary gripper command required")
        frame = tool_frame(joints)
        if gripper_command == "closed":
            if self.held is None:
                if self.previous_command != "open":
                    raise ValueError("a grasp estimate needs a previously observed open gripper")
                self.held = self.active["object"]
                self.offset = to_tool(self.points[self.held], frame)
                if self.center_lateral_on_close:
                    self.offset = (self.offset[0], 0.0, self.offset[2])
                self.states[self.held] = ("conditional_centered_grasp" if self.center_lateral_on_close
                                          else "conditional_rigid_grasp")
            self.points[self.held] = from_tool(self.offset, frame)
        elif self.held is not None:
            # Use the last closed observation: motion or compliance after opening
            # is not evidence that the released object followed the hand.
            obj = self.held
            point = self.points[obj]
            self.points[obj] = (point[0], point[1], (self.active["destination_level"]-.5)*self.object_height_mm)
            self.states[obj] = "released_expected_support"
            self.held, self.offset = None, None
        self.previous_command = gripper_command

    def estimate(self, object_id):
        if object_id not in self.points:
            raise ValueError("object is outside the registered scene")
        return {"position_mm": list(self.points[object_id]), "state": self.states[object_id],
                "confirmed": False, "assumption": ("symmetric_lateral_centering_then_rigid_no_slip"
                    if self.center_lateral_on_close else "requested_object_grasped_without_slip")}

    def finish_transfer(self):
        if self.active is None or self.held is not None or self.previous_command != "open":
            raise ValueError("normal released transfer completion required")
        self.expected_scene, self.active = self.active["after"], None
