"""One reused learned policy expressed in a registered nominal task frame.

The similarity transform uses registered slots and nominal robot FK/IK only.
It supplies neither object poses nor a next waypoint. The learned policy still
chooses every action and target. This experiment adds no new learned weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_features, scene_unified_kinematic_policy as unified
from .scene_causal_frame_policy import inverse_or_reject, pose
from .nominal_bounded_ik import inverse_nearest_feasible_pitch

FORMAT = "raccoon-canonical-task-unified-policy.v1"


def basis(source, destination):
    delta = [b-a for a, b in zip(source, destination)]
    length = math.hypot(*delta)
    if not math.isfinite(length) or length < 1:
        raise ValueError("distinct finite registered source and destination required")
    axis = [d/length for d in delta]
    return list(source), length, axis, [-axis[1], axis[0]]


def map_xy(point, original, following):
    origin, length, axis, normal = original
    target, distance, direction, side = following
    delta = [x-y for x, y in zip(point, origin)]
    along = sum(x*y for x, y in zip(delta, axis))/length
    across = sum(x*y for x, y in zip(delta, normal))/length
    return [target[i]+distance*(along*direction[i]+across*side[i]) for i in range(2)]


def map_joints(joints, original, following, *, input_control=False, joint_limited_pitch=False, diagnostics=None):
    value = pose(torch.tensor([joints], dtype=torch.float64))[0].tolist()
    value[:2] = map_xy(value[:2], original, following)
    mapped, reachable = inverse_or_reject(value)
    if joint_limited_pitch and (not reachable or any(q < lo or q > hi for q, (lo, hi) in zip(mapped, JOINT_LIMITS))):
        mapped, diagnostic = inverse_nearest_feasible_pitch(value)
        if diagnostics is not None:
            diagnostics.append(dict(diagnostic, virtual_input=bool(input_control)))
        reachable = True
    if not reachable or not all(math.isfinite(x) for x in mapped):
        raise ValueError("registered-frame nominal IK is unreachable")
    if input_control:
        if any(q < lo-1e-7 or q > hi+1e-7 for q, (lo, hi) in zip(mapped, JOINT_LIMITS)):
            raise ValueError("virtual input lies outside reference joint limits")
        mapped = [min(hi, max(lo, q)) for q, (lo, hi) in zip(mapped, JOINT_LIMITS)]
    return mapped


class CanonicalTaskPolicy(nn.Module):
    history_length, feature_count = 8, 82

    def __init__(self, reference, *, joint_limited_pitch=False):
        super().__init__()
        if type(joint_limited_pitch) is not bool:
            raise ValueError("joint-limited pitch configuration must be boolean")
        self.joint_limited_pitch = joint_limited_pitch
        self.reference = reference
        self.register_buffer("reference_slots", torch.tensor(scene_features.NOMINAL_SLOT_XY_MM, dtype=torch.float64))
        self.last_diagnostic = None

    def forward_with_control_state(self, windows, masks, previous):
        return self._forward_in_frame(windows, masks, previous)

    def _forward_in_frame(self, windows, masks, previous, *, source_anchor=None):
        if (windows.shape != (1, 8, 82) or masks.shape != (1, 8) or masks.dtype != torch.bool
                or previous.shape != (1, 4) or windows.device.type != "cpu"
                or not bool(torch.isfinite(windows).all() and torch.isfinite(previous).all())):
            raise ValueError("one finite causal CPU observation and exact applied target required")
        n = int(masks.sum())
        if n < 1 or masks[0].tolist() != [False]*(8-n)+[True]*n:
            raise ValueError("invalid causal mask")
        last = windows[0, -1]
        self.last_diagnostic = {"identity_registration": False, "output_rejection": None}
        adjustments = [] if self.joint_limited_pitch else None
        if adjustments is not None:
            self.last_diagnostic["joint_limit_pose_adjustments"] = adjustments
        expected = (self.reference_slots/300.0).flatten().to(last)
        identity = torch.equal(last[74:82], expected)
        if identity and source_anchor is None:
            self.last_diagnostic["identity_registration"] = True
            return self.reference.forward_with_control_state(windows, masks, previous)
        source, destination = int(last[14:18].argmax()), int(last[18:22].argmax())
        if source == destination:
            raise ValueError("source and destination must differ")
        slots = (last[74:82].double().reshape(4, 2)*300.0).tolist()
        if source_anchor is not None:
            if len(source_anchor) != 2 or not all(math.isfinite(v) for v in source_anchor):
                raise ValueError("finite conditional source anchor required")
            if identity and all(abs(a-b) <= 1e-7 for a, b in zip(source_anchor, slots[source])):
                self.last_diagnostic["identity_registration"] = True
                return self.reference.forward_with_control_state(windows, masks, previous)
            # This is the reference policy's expected object anchor, not a
            # mutation of the physical simulator's registered calibration.
            slots[source] = list(source_anchor)
        canonical_slots = self.reference_slots.tolist()
        actual = basis(slots[source], slots[destination])
        canonical = basis(canonical_slots[source], canonical_slots[destination])
        mapped_slots = [v/300.0 for point in slots for v in map_xy(point, actual, canonical)]
        virtual = windows.clone()
        for time in masks[0].nonzero().flatten().tolist():
            row = windows[0, time]
            if not torch.equal(row[14:26], last[14:26]) or not torch.equal(row[74:82], last[74:82]):
                raise ValueError("task or registration changed inside the causal history")
            q = map_joints((row[:4].double()*180.0).tolist(), actual, canonical, input_control=True,
                           joint_limited_pitch=self.joint_limited_pitch, diagnostics=adjustments)
            virtual[0, time, :4] = row.new_tensor(q)/180.0
            virtual[0, time, 74:82] = row.new_tensor(mapped_slots)
        prior = map_joints(previous[0].double().tolist(), actual, canonical, input_control=True,
                           joint_limited_pitch=self.joint_limited_pitch, diagnostics=adjustments)
        logits, target = self.reference.forward_with_control_state(virtual, masks, previous.new_tensor([prior]))
        if int(logits.argmax()) != motion_policy.MOVE:
            return logits, previous/180.0
        try:
            q = map_joints((target[0].double()*180.0).tolist(), canonical, actual,
                           joint_limited_pitch=self.joint_limited_pitch, diagnostics=adjustments)
        except ValueError as exc:
            self.last_diagnostic["output_rejection"] = str(exc)
            q = [999.0]*4
        # The existing outer guard decides whether an actual joint target is legal.
        return logits, target.new_tensor([q])/180.0

    def forward(self, windows, masks):
        raise ValueError("exact previous applied control required")


def load_checkpoint(payload, device="cpu"):
    if (payload.get("format") != FORMAT or str(device) != "cpu"
            or payload.get("reference_slots_mm") != [list(xy) for xy in scene_features.NOMINAL_SLOT_XY_MM]):
        raise ValueError("unsupported registered-frame checkpoint or device")
    return CanonicalTaskPolicy(unified.load_checkpoint(payload["reference"], device=device)).eval()
