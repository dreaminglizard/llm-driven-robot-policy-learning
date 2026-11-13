"""One learned student with conditional proprioceptive source-position tracking.

The pose state is initialized by explicit nominal registration and propagated
from observed joints and binary gripper commands. It never confirms grasp or
placement and never reads simulated object/contact state. Only the source
anchor for the next transfer changes; learned action selection is untouched.
"""
from __future__ import annotations

import torch

from . import scene_features, task_catalog, scene_unified_kinematic_policy as unified
from .nominal_object_belief import NominalObjectBelief
from .scene_canonical_task_policy import CanonicalTaskPolicy

FORMAT = "raccoon-belief-canonical-unified-policy.v1"


def registered_scene(row):
    identifiers = row[30:70].reshape(4, 2, 5).argmax(-1).tolist()
    return task_catalog.scene([[int(obj) for obj in stack if obj] for stack in identifiers])


class BeliefCanonicalPolicy(CanonicalTaskPolicy):
    def __init__(self, reference, *, center_lateral_on_close, joint_limited_pitch=False):
        super().__init__(reference, joint_limited_pitch=joint_limited_pitch)
        self.belief = NominalObjectBelief(center_lateral_on_close=center_lateral_on_close)
        self._registered = False
        self._task = None
        self._source_anchor = None
        self._tracking_error = None
        self._nominal_slots = None
        self._last_action = None

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        self.belief.register(scene, nominal_slot_xy_mm)
        self._registered = True
        self._task, self._source_anchor, self._tracking_error = None, None, None
        self._nominal_slots = [list(xy) for xy in nominal_slot_xy_mm]
        self._last_action = None

    def on_expected_transfer_committed(self, before, task, after):
        try:
            if (self._last_action != 3 or self._task != task or self.belief.expected_scene != task_catalog.scene(before)
                    or task_catalog.transfer(before, task["source_slot"], task["destination_slot"]) != task_catalog.scene(after)):
                raise ValueError("pose-tracking expected commit differs")
            self.belief.finish_transfer()
        except ValueError as exc:
            # Bookkeeping completion is not evidence of a successful grasp.
            # Invalidate the estimate without changing the session's outcome.
            self._tracking_error = str(exc)
        self._task, self._source_anchor = None, None

    def conditional_pose_state(self):
        if not self._registered:
            return {"available": False, "confirmed": False, "reason": "not_registered"}
        return {"available": self._tracking_error is None, "confirmed": False,
                "reason": self._tracking_error,
                "objects": {str(obj): self.belief.estimate(obj) for obj in sorted(self.belief.points)}}

    def forward_with_control_state(self, windows, masks, previous):
        if not self._registered or self._tracking_error is not None:
            raise ValueError("conditional pose tracking requires explicit registration")
        if (windows.shape != (1, 8, 82) or masks.shape != (1, 8) or masks.dtype != torch.bool
                or previous.shape != (1, 4) or windows.device.type != "cpu"
                or not bool(torch.isfinite(windows).all() and torch.isfinite(previous).all())):
            raise ValueError("finite causal scene features required")
        n = int(masks.sum())
        if n < 1 or masks[0].tolist() != [False]*(8-n)+[True]*n:
            raise ValueError("invalid causal history mask")
        row = windows[0, -1]
        expected_slots = row.new_tensor([v/300.0 for xy in self._nominal_slots for v in xy])
        if not torch.equal(row[74:82], expected_slots):
            raise ValueError("calibration changed without explicit registration")
        before = registered_scene(row)
        before, task = scene_features.transfer_task(before, {
            "source_slot": int(row[14:18].argmax())+1,
            "destination_slot": int(row[18:22].argmax())+1,
            "source_level": int(row[22:24].argmax())+1,
            "destination_level": int(row[24:26].argmax())+1,
            "object_id": int(row[26:30].argmax())+1})
        if self._task is None:
            if int(masks.sum()) != 1:
                raise ValueError("a new tracked transfer must start at its first observation")
            self.belief.begin_transfer(before, task["source_slot"], task["destination_slot"])
            self._task = task
            self._source_anchor = self.belief.estimate(task["object_id"])["position_mm"][:2]
        elif self._task != task or before != self.belief.expected_scene:
            raise ValueError("tracked task changed before expected commit")
        grip = ("none", "open", "closed")[int(row[6:9].argmax())]
        self.belief.observe((row[:4].double()*180.0).tolist(), grip)
        output = self._forward_in_frame(windows, masks, previous, source_anchor=self._source_anchor)
        self._last_action = int(output[0].argmax())
        self.last_diagnostic.update(conditional_source_anchor_mm=list(self._source_anchor),
                                    source_pose_confirmed=False)
        return output


def load_checkpoint(payload, device="cpu"):
    if (payload.get("format") != FORMAT or str(device) != "cpu"
            or payload.get("reference_slots_mm") != [list(xy) for xy in scene_features.NOMINAL_SLOT_XY_MM]):
        raise ValueError("unsupported conditional-pose checkpoint")
    return BeliefCanonicalPolicy(unified.load_checkpoint(payload["reference"], device=device),
        center_lateral_on_close=payload["configuration"]["center_lateral_on_close"],
        joint_limited_pitch=payload["configuration"].get("joint_limited_pitch", False)).eval()
