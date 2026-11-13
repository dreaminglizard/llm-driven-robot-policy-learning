"""Execute the causal applied-action student through the shared policy API.

The executor supplies registration, transfer boundaries and confirmed command
returns. A prediction never becomes previous applied control by itself. This
adapter retains the learned recurrent state across transfers and applies no
kinematic attraction, inverse frame transform, or action-routing rule.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_features, task_catalog
from .scene_applied_action_student import AppliedActionStudent, PREVIOUS_APPLIED_ACTIONS, FORMAT, FEATURE_COUNT, HIDDEN_SIZE
from .scene_custom_slots_sim import registered_slot_xy_mm


S0_CHECKPOINT_FORMAT = "raccoon-applied-action-s0-checkpoint.v1"
PAIRED_CHECKPOINT_FORMAT = "raccoon-applied-action-direct-pair-checkpoint.v1"
KINEMATIC_CHECKPOINT_FORMAT = "raccoon-kinematic-applied-action-checkpoint.v1"
TASK_FRAME_CHECKPOINT_FORMAT = "raccoon-task-frame-student-checkpoint.v1"
CHECKPOINT_FORMATS = (S0_CHECKPOINT_FORMAT, PAIRED_CHECKPOINT_FORMAT, KINEMATIC_CHECKPOINT_FORMAT,
                      TASK_FRAME_CHECKPOINT_FORMAT)


class AppliedActionPolicy(nn.Module):
    # Keep the shared evaluator's padded window contract. Only its current row
    # is consumed: earlier rows have already advanced the persistent GRU.
    history_length = scene_features.HISTORY_LENGTH
    feature_count = len(scene_features.FEATURE_NAMES)

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, AppliedActionStudent):
            raise TypeError("an AppliedActionStudent is required")
        self.model = model
        self._scene = None
        self._slots = None
        self._reset_runtime()
        self.eval()

    def _reset_runtime(self):
        self._state = None
        self._task = None
        self._previous_action = None
        self._previous_target = None
        self._opening = None
        self._first_registration_row = True
        self._first_transfer_row = True
        self._pending_action = None
        self._done = False

    @staticmethod
    def _joints(values):
        if not isinstance(values, (tuple, list)) or len(values) != 4:
            raise ValueError("confirmed control requires four joint angles")
        if any(type(q) not in (int, float) or not math.isfinite(q) or not lo <= q <= hi
               for q, (lo, hi) in zip(values, JOINT_LIMITS)):
            raise ValueError("confirmed control must be finite and within joint limits")
        return [float(q) for q in values]

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        before = task_catalog.scene(scene)
        slots = registered_slot_xy_mm(nominal_slot_xy_mm)
        self._reset_runtime()
        self._scene, self._slots = before, slots

    def on_transfer_started(self, before, task, applied_targets):
        """Receive already validated public control, without resetting the GRU."""
        if self._scene is None or self._task is not None or self._pending_action is not None or self._done:
            raise ValueError("a registered, committed transfer boundary is required")
        before, task = scene_features.transfer_task(before, task)
        if before != self._scene or not isinstance(applied_targets, dict):
            raise ValueError("transfer differs from the expected registered scene")
        target = self._joints(applied_targets.get("joints_deg"))
        opening = applied_targets.get("gripper_opening_m")
        if type(opening) not in (int, float) or not math.isclose(opening, .04, abs_tol=1e-9, rel_tol=0):
            raise ValueError("a transfer must start with confirmed open control")
        if self._previous_target is not None and (target != self._previous_target or opening != self._opening):
            raise ValueError("control changed across transfers without a confirmed return")
        self._previous_target, self._opening = target, float(opening)
        self._task = task
        self._first_transfer_row = True

    def _row(self, windows, masks, previous):
        parameter = self.model.action_head.weight
        if (not isinstance(windows, torch.Tensor) or windows.shape != (1, self.history_length, self.feature_count)
                or windows.dtype != parameter.dtype or windows.device != parameter.device
                or not bool(torch.isfinite(windows).all()) or not isinstance(masks, torch.Tensor)
                or masks.shape != (1, self.history_length) or masks.dtype != torch.bool or masks.device != windows.device
                or not bool(masks[0, -1])):
            raise ValueError("one finite current scene feature row is required")
        if (not isinstance(previous, torch.Tensor) or previous.shape != (1, 4)
                or previous.dtype != parameter.dtype or previous.device != parameter.device
                or not torch.equal(previous[0], previous.new_tensor(self._previous_target))):
            raise ValueError("previous target differs from confirmed transmitted control")
        # Only compare registered symbolic inputs. The first fourteen values
        # contain current public robot feedback and remain unchanged.
        dummy = {"schema_version": "PolicyObservation.phase1", "joints_deg": [0, -10, -110, 30],
                 "sensor_valid": True, "moving": False, "gripper_command": "open",
                 "gripper_reported_state": "open", "gripper_reported_valid": True,
                 "gripper_feedback_fresh": True}
        registered = scene_features.encode_scene_features(dummy, self._scene, self._task, self._slots)[14:]
        if not torch.equal(windows[0, -1, 14:], windows.new_tensor(registered)):
            raise ValueError("student inputs differ from the active registered task")
        # Match the loader and encode_step: divide the validated transmitted
        # double first, then convert to float32. Normalizing the rounded tensor
        # here would introduce a second, different rounding path.
        extra = [float(self._previous_action == action) for action in PREVIOUS_APPLIED_ACTIONS]
        extra += [q / motion_policy.JOINT_SCALE_DEG for q in self._previous_target]
        extra += [float(self._first_transfer_row), float(self._first_registration_row)]
        return torch.cat((windows[0, -1], windows.new_tensor(extra))).reshape(1, 1, -1)

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self._task is None or self._previous_target is None:
            raise ValueError("confirmed transfer-start control is required")
        if self._pending_action is not None or self._done:
            raise ValueError("the previous decision still requires command return or expected commit")
        row = self._row(windows, masks, previous_target_degrees)
        output = self.model(row, masks[:, -1:], self._state)
        logits, joints = output.action_logits[:, 0], output.move_joints_deg[:, 0]
        if not bool(torch.isfinite(logits).all() and torch.isfinite(joints).all()):
            raise ValueError("applied-action student returned nonfinite output")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        self._state = output.state.detach()
        self._first_registration_row = self._first_transfer_row = False
        if action == "done":
            self._done = True
        else:
            self._pending_action = action
        return logits, joints / motion_policy.JOINT_SCALE_DEG

    def forward(self, windows, masks):
        raise ValueError("applied-action execution requires confirmed previous control")

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        """Accept only the executor's already checked normal command return."""
        if self._pending_action is None or action != self._pending_action:
            raise ValueError("command return does not match the pending model action")
        sent, actual = self._joints(transmitted_target_deg), self._joints(applied_target_deg)
        if any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0) for a, b in zip(sent, actual)):
            raise ValueError("applied control differs from the transmitted command")
        expected_opening = .04 if action == "open" else 0. if action == "close" else self._opening
        if (type(gripper_opening_m) not in (int, float)
                or not math.isclose(gripper_opening_m, expected_opening, abs_tol=1e-9, rel_tol=0)
                or (action != "move" and sent != self._previous_target)):
            raise ValueError("command return differs from held joints or expected gripper control")
        self._previous_action, self._previous_target = action, sent
        self._opening, self._pending_action = float(gripper_opening_m), None

    def on_expected_transfer_committed(self, before, task, after):
        before, task = scene_features.transfer_task(before, task)
        after = task_catalog.scene(after)
        if (not self._done or self._pending_action is not None or before != self._scene or task != self._task
                or after != task_catalog.transfer(before, task["source_slot"], task["destination_slot"])):
            raise ValueError("expected commit does not match the completed student transfer")
        self._scene, self._task, self._done = after, None, False
        self._first_transfer_row = True


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _state_digest(model):
    """Match the published S0 trainer's named tensor fingerprint."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        descriptor = _encoded({"name": name, "shape": list(value.shape), "dtype": str(value.dtype)})
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(payload):
    """Load a raw, direct-pair or kinematic student for the shared evaluation API.

    File fingerprints and data provenance remain the evaluator's responsibility.
    This validates the architecture, bounded output buffers and saved weights.
    """
    if isinstance(payload, dict) and payload.get("format") == TASK_FRAME_CHECKPOINT_FORMAT:
        from . import scene_task_frame_policy
        return scene_task_frame_policy.load_checkpoint(payload)
    if isinstance(payload, dict) and payload.get("format") == KINEMATIC_CHECKPOINT_FORMAT:
        from . import scene_kinematic_applied_student
        return AppliedActionPolicy(scene_kinematic_applied_student.load_checkpoint(payload))
    contract = {"model_format": FORMAT, "constructor": "AppliedActionStudent()",
                "feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE,
                "actions": list(motion_policy.ACTIONS), "joint_output": "absolute_robot_frame_degrees"}
    if (not isinstance(payload, dict) or payload.get("format") not in CHECKPOINT_FORMATS
            or payload.get("model") != contract or not isinstance(payload.get("config"), dict)
            or payload.get("config_sha256") != hashlib.sha256(_encoded(payload["config"])).hexdigest()
            or not isinstance(payload.get("provenance"), dict) or not isinstance(payload.get("training"), dict)
            or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("unsupported applied-action checkpoint contract")
    state_key = "final_state_sha256"
    if payload["format"] == PAIRED_CHECKPOINT_FORMAT:
        branch = payload.get("branch")
        if branch not in ("a", "b"):
            raise ValueError("unsupported direct-pair checkpoint branch")
        state_key = branch + "_final_state_sha256"
    with torch.random.fork_rng(devices=[]):
        model = AppliedActionStudent().cpu()
    expected, state = model.state_dict(), payload["state_dict"]
    if (set(expected) != set(state) or any(not isinstance(value, torch.Tensor)
            or value.device.type != "cpu" or value.dtype != expected[name].dtype or value.shape != expected[name].shape
            or not bool(torch.isfinite(value).all()) for name, value in state.items())
            or any(not torch.equal(state[name], value) for name, value in model.named_buffers())):
        raise ValueError("applied-action tensors or joint bounds differ")
    model.load_state_dict(state, strict=True)
    if _state_digest(model) != payload["training"].get(state_key):
        raise ValueError("applied-action weights fingerprint differs")
    return AppliedActionPolicy(model.eval())
