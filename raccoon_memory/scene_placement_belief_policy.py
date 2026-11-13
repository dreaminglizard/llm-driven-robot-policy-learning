"""One student with a continuous nominal object-position correction on MOVE.

The field vanishes at the source and at the transfer's observed/predicted peak
height, and approaches the registered object goal during descent. Actions are
selected by the original student; no OPEN gate, physical feedback or new labels
are introduced. Object position remains a conditional joint-based estimate.
"""
from __future__ import annotations

import math

import torch

from raccoon_control.kinematics import Geometry
from . import motion_policy, scene_features, scene_unified_kinematic_policy as unified
from .nominal_bounded_ik import inverse_nearest_feasible_pitch
from .scene_belief_canonical_policy import BeliefCanonicalPolicy

FORMAT = "raccoon-placement-belief-canonical-unified-policy.v1"
GRASP_FORMAT = "raccoon-grasp-placement-belief-canonical-unified-policy.v1"


def attracted_position(proposed, source_xy, goal, peak_height):
    delta = [goal[i]-source_xy[i] for i in range(2)]
    length_squared = sum(v*v for v in delta)
    if length_squared < 1. or peak_height <= goal[2]+1e-9:
        return list(proposed), 0.
    progress = max(0., min(1., sum((proposed[i]-source_xy[i])*delta[i] for i in range(2))/length_squared))
    height = max(0., min(1., (proposed[2]-goal[2])/(peak_height-goal[2])))
    weight = progress*(1.-height)
    return [p+weight*(g-p) for p, g in zip(proposed, goal)], weight


def object_geometry(offset):
    if len(offset) != 3 or not all(math.isfinite(v) for v in offset) or abs(offset[1]) > 1e-8:
        raise ValueError("a finite symmetrically centered conditional grasp is required")
    nominal = Geometry()
    return Geometry(base_mm=nominal.base_mm, upper_mm=nominal.upper_mm, forearm_mm=nominal.forearm_mm,
                    tool_forward_mm=nominal.tool_forward_mm+offset[0],
                    tool_vertical_mm=nominal.tool_vertical_mm+offset[2])


def approach_position(proposed, source_goal, transfer_length, peak_height):
    if transfer_length < 1. or peak_height <= source_goal[2]+1e-9:
        return list(proposed), 0.
    proximity = max(0., min(1., 1.-math.dist(proposed[:2], source_goal[:2])/transfer_length))
    height = max(0., min(1., (proposed[2]-source_goal[2])/(peak_height-source_goal[2])))
    weight = proximity*(1.-height)
    return [p+weight*(g-p) for p, g in zip(proposed, source_goal)], weight


class PlacementBeliefCanonicalPolicy(BeliefCanonicalPolicy):
    def __init__(self, reference):
        super().__init__(reference, center_lateral_on_close=True, joint_limited_pitch=True)
        self._closed_peak_mm = None

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        super().on_scene_registered(scene, nominal_slot_xy_mm)
        self._closed_peak_mm = None

    def on_expected_transfer_committed(self, before, task, after):
        super().on_expected_transfer_committed(before, task, after)
        self._closed_peak_mm = None

    def forward_with_control_state(self, windows, masks, previous):
        logits, angles = super().forward_with_control_state(windows, masks, previous)
        if self.belief.held is None:
            return logits, angles
        position = self.belief.estimate(self.belief.held)["position_mm"]
        self._closed_peak_mm = max(position[2], self._closed_peak_mm if self._closed_peak_mm is not None else position[2])
        if int(logits.argmax()) != motion_policy.MOVE:
            return logits, angles
        joints = (angles[0].double()*180.).tolist()
        if any(abs(v) > 360 for v in joints):
            return logits, angles  # Preserve an upstream rejected proposal.
        geometry = object_geometry(self.belief.offset)
        proposed = geometry.forward(joints)
        self._closed_peak_mm = max(self._closed_peak_mm, proposed[2])
        goal = [*self._nominal_slots[self._task["destination_slot"]-1],
                (self._task["destination_level"]-.5)*self.belief.object_height_mm]
        desired, weight = attracted_position(proposed, self._source_anchor, goal, self._closed_peak_mm)
        diagnostic = {"weight": weight, "proposed_conditional_object_mm": list(proposed),
                      "desired_conditional_object_mm": desired, "registered_goal_mm": goal,
                      "closed_peak_mm": self._closed_peak_mm, "pose_confirmed": False}
        self.last_diagnostic["placement_attraction"] = diagnostic
        if weight <= 1e-12:
            return logits, angles
        try:
            feasible, report = inverse_nearest_feasible_pitch([*desired, sum(joints[1:])+90.], geometry=geometry)
            diagnostic.update(report)
        except ValueError as exc:
            diagnostic["rejected"] = str(exc)
            feasible = [999.]*4
        return logits, angles.new_tensor([feasible])/180.


class GraspPlacementBeliefCanonicalPolicy(PlacementBeliefCanonicalPolicy):
    def __init__(self, reference, *, grasp_tcp_above_center_mm):
        super().__init__(reference)
        if type(grasp_tcp_above_center_mm) not in (float, int) or not math.isfinite(grasp_tcp_above_center_mm) or grasp_tcp_above_center_mm < 0:
            raise ValueError("finite nominal gripper height above the object center required")
        self.grasp_tcp_above_center_mm = float(grasp_tcp_above_center_mm)
        self._approach_peak_mm, self._grasp_started = None, False

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        super().on_scene_registered(scene, nominal_slot_xy_mm)
        self._approach_peak_mm, self._grasp_started = None, False

    def on_expected_transfer_committed(self, before, task, after):
        super().on_expected_transfer_committed(before, task, after)
        self._approach_peak_mm, self._grasp_started = None, False

    def forward_with_control_state(self, windows, masks, previous):
        logits, angles = super().forward_with_control_state(windows, masks, previous)
        if self.belief.held is not None:
            self._grasp_started = True
        if self._grasp_started or int(logits.argmax()) != motion_policy.MOVE:
            return logits, angles
        joints = (angles[0].double()*180.).tolist()
        if any(abs(v) > 360 for v in joints):
            return logits, angles
        geometry = object_geometry((0., 0., -self.grasp_tcp_above_center_mm))
        proposed = geometry.forward(joints)
        self._approach_peak_mm = max(proposed[2], self._approach_peak_mm if self._approach_peak_mm is not None else proposed[2])
        goal = [*self._source_anchor,
                (self._task["source_level"]-.5)*self.belief.object_height_mm]
        distance = math.dist(self._source_anchor, self._nominal_slots[self._task["destination_slot"]-1])
        desired, weight = approach_position(proposed, goal, distance, self._approach_peak_mm)
        diagnostic = {"weight": weight, "proposed_grasp_reference_mm": list(proposed),
            "desired_grasp_reference_mm": desired, "registered_source_goal_mm": goal,
            "nominal_gripper_height_above_center_mm": self.grasp_tcp_above_center_mm,
            "proposed_approach_peak_mm": self._approach_peak_mm, "pose_confirmed": False}
        self.last_diagnostic["grasp_attraction"] = diagnostic
        if weight <= 1e-12:
            return logits, angles
        try:
            feasible, report = inverse_nearest_feasible_pitch([*desired, sum(joints[1:])+90.], geometry=geometry)
            diagnostic.update(report)
        except ValueError as exc:
            diagnostic["rejected"] = str(exc)
            feasible = [999.]*4
        return logits, angles.new_tensor([feasible])/180.


def load_checkpoint(payload, device="cpu"):
    if (payload.get("format") not in (FORMAT, GRASP_FORMAT) or str(device) != "cpu" or
            payload.get("reference_slots_mm") != [list(xy) for xy in scene_features.NOMINAL_SLOT_XY_MM]):
        raise ValueError("unsupported placement-belief checkpoint")
    if payload["format"] == GRASP_FORMAT:
        return GraspPlacementBeliefCanonicalPolicy(unified.load_checkpoint(payload["reference"], device=device),
            grasp_tcp_above_center_mm=payload["configuration"]["grasp_tcp_above_center_mm"]).eval()
    return PlacementBeliefCanonicalPolicy(unified.load_checkpoint(payload["reference"], device=device)).eval()
