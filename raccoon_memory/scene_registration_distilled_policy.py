"""Public-history GRU student and the callback adapter for one registration.

The 51 original polar features retain their original units and normalization.
The appended fields are public past events.  A control event is installed only
after a confirmed control callback; a return-time joint sample is optional and
is never fabricated from the next decision's sample.  No object pose, teacher
state, current action label, or future commit is an input.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy
from .scene_causal_frame_policy import CONTEXT_COUNT, context
from .scene_chunked_recurrent_policy import ChunkedRecurrentPolicy

FORMAT = "raccoon-public-registration-distilled-gru.v1"
START = 4
MODE_COUNT = 5  # MOVE, OPEN, CLOSE, END, START
FEATURE_COUNT = CONTEXT_COUNT + MODE_COUNT + 1 + 4 + 3 + 4 + 4 + 4 + 1


def _vector(values, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if (not isinstance(values, (tuple, list)) or len(values) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f"{name} must contain four finite angles")
    return tuple(float(v) for v in values)


def _target(values):
    values = _vector(values, "applied target")
    if any(not lo - 1e-5 <= v <= hi + 1e-5 for v, (lo, hi) in zip(values, JOINT_LIMITS)):
        raise ValueError("applied target is outside the joint limits")
    return values


def _opening(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .05:
        raise ValueError("applied gripper opening must be a finite value within range")
    return float(value)


class ControlEvent(NamedTuple):
    mode: int
    object_id: tuple[float, ...]
    before_deg: tuple[float, ...]
    after_deg: tuple[float, ...] | None


def encode_public_row(row, previous, *, previous_mode, new_instruction, control_event=None):
    """One decision input, with events from confirmed *earlier* controls only."""
    row = torch.as_tensor(row, dtype=torch.float32, device="cpu").reshape(-1)
    previous = torch.as_tensor(_target(previous), dtype=torch.float32).reshape(1, 4)
    if (row.shape != (82,) or not bool(torch.isfinite(row).all())
            or type(previous_mode) is not int or previous_mode not in range(MODE_COUNT)
            or type(new_instruction) is not bool):
        raise ValueError("invalid public decision input")
    object_id = row[26:30]
    if bool(((object_id != 0) & (object_id != 1)).any()) or float(object_id.sum()) != 1.0:
        raise ValueError("a public task object identity is required")
    if previous_mode == START and not new_instruction:
        raise ValueError("START can occur only on the first instruction")
    geometric = context(row[None], previous, "polar")[0].float()
    mode = torch.zeros(MODE_COUNT)
    mode[previous_mode] = 1
    event_kind = torch.zeros(3)
    event_object = torch.zeros(4)
    event_before = torch.zeros(4)
    event_after = torch.zeros(4)
    after_valid = 0.0
    if control_event is not None:
        if not isinstance(control_event, ControlEvent) or control_event.mode not in (motion_policy.MOVE, motion_policy.OPEN, motion_policy.CLOSE):
            raise ValueError("a confirmed physical control event is required")
        event_kind[control_event.mode] = 1
        event_object = torch.tensor(control_event.object_id, dtype=torch.float32)
        if (event_object.shape != (4,) or bool(((event_object != 0) & (event_object != 1)).any())
                or float(event_object.sum()) != 1):
            raise ValueError("invalid event object identity")
        event_before = torch.tensor(_vector(control_event.before_deg, "pre-event joints")) / 180.0
        if control_event.after_deg is not None:
            event_after = torch.tensor(_vector(control_event.after_deg, "return-time joints")) / 180.0
            after_valid = 1.0
    result = torch.cat((geometric, mode, torch.tensor([float(new_instruction)]), object_id,
                        event_kind, event_object, event_before, event_after, torch.tensor([after_valid])))
    if result.shape != (FEATURE_COUNT,) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid encoded public history")
    return result


class PolicyOutputs(NamedTuple):
    logits: torch.Tensor
    angles: torch.Tensor  # normalized by 180, as in the original student
    hidden: torch.Tensor


class RegistrationDistilledPolicy(ChunkedRecurrentPolicy):
    """Original horizon-one architecture with registration-wide memory."""
    def __init__(self, *, center, scale):
        super().__init__(horizon=1, center=center, scale=scale, output_space="joint")
        self.input = nn.Sequential(nn.Linear(FEATURE_COUNT, 64), nn.Tanh())

    def _project(self, features):
        raw = torch.cat(((features[..., :CONTEXT_COUNT] - self.center) / self.scale,
                         features[..., CONTEXT_COUNT:]), dim=-1)
        return self.input(raw)

    def forward_padded(self, features, previous, valid):
        """Train through every real context row, with no transfer-boundary detach.

        Each batch item is one complete registration; right padding never
        advances its hidden state and never affects its outputs or gradients.
        """
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[-1] != FEATURE_COUNT or features.dtype != torch.float32
                or not isinstance(previous, torch.Tensor) or previous.shape != (*features.shape[:2], 4)
                or previous.dtype != features.dtype or previous.device != features.device
                or not isinstance(valid, torch.Tensor) or valid.shape != features.shape[:2]
                or valid.dtype != torch.bool or valid.device != features.device
                or not bool(valid[:, 0].all()) or bool(((~valid[:, :-1]) & valid[:, 1:]).any())
                or not bool(torch.isfinite(features[valid]).all() and torch.isfinite(previous[valid]).all())):
            raise ValueError("finite nonempty right-padded registration histories are required")
        safe = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        safe_previous = torch.where(valid.unsqueeze(-1), previous, torch.zeros_like(previous))
        lengths = valid.sum(1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(self._project(safe), lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.encoder(packed)
        encoded, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=features.shape[1])
        batch, steps = features.shape[:2]
        logits, angles = self.decode(encoded.reshape(-1, 96), safe_previous.reshape(-1, 4), steps=1)
        logits, angles = logits.reshape(batch, steps, 4), angles.reshape(batch, steps, 4)
        return PolicyOutputs(torch.where(valid[..., None], logits, torch.zeros_like(logits)),
                             torch.where(valid[..., None], angles, torch.zeros_like(angles)), hidden)

    def infer_row(self, features, previous, hidden):
        if features.shape != (1, FEATURE_COUNT) or previous.shape != (1, 4):
            raise ValueError("one encoded public observation is required")
        encoded, hidden = self.encoder(self._project(features).unsqueeze(1), hidden)
        logits, angles = self.decode(encoded[:, 0], previous, steps=1)
        return logits[:, 0], angles[:, 0], hidden


def load_checkpoint(payload, device="cpu"):
    if (str(device) != "cpu" or not isinstance(payload, dict) or payload.get("format") != FORMAT
            or payload.get("configuration") != {"horizon": 1, "output_space": "joint", "feature_count": FEATURE_COUNT}):
        raise ValueError("unsupported registration student checkpoint")
    state = payload["state_dict"]
    model = RegistrationDistilledPolicy(center=state["center"], scale=state["scale"])
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    return model.eval()


class RegistrationAdapter(nn.Module):
    """Wire the student's memory to real registration/control/commit callbacks."""
    history_length, feature_count = 8, 82

    def __init__(self, student, *, return_joint_observer=None):
        super().__init__()
        if not isinstance(student, RegistrationDistilledPolicy):
            raise ValueError("the registration student is required")
        self.student = student.eval()
        self.return_joint_observer = return_joint_observer
        self.start_case()

    def start_case(self):
        self.registrations, self.calls = 0, []
        self._active = False
        self._hidden = None
        self._pending = None
        self._control_event = None
        self._mode = START
        self._new_instruction = False
        self._expected_target = None
        self._expected_opening = None

    def on_scene_registered(self, _scene, _slots):
        self.start_case()
        self.registrations = 1
        self._registered = True

    def on_transfer_started(self, _before, _task, applied_targets):
        if self.registrations != 1 or self._active or self._pending is not None:
            raise ValueError("registration and preceding expected commit are required")
        target = _target(applied_targets["joints_deg"])
        opening = _opening(applied_targets["gripper_opening_m"])
        if self._expected_target is not None and any(abs(a-b) > 3e-5 for a, b in zip(target, self._expected_target)):
            raise ValueError("transfer starts with an unexpected applied target")
        if self._expected_opening is not None and not math.isclose(opening, self._expected_opening, abs_tol=1e-9, rel_tol=0):
            raise ValueError("transfer starts with an unexpected applied gripper opening")
        self._expected_target = target
        self._expected_opening = opening
        self._new_instruction = True
        self._active = True

    def forward_with_control_state(self, windows, masks, previous):
        if (not self._active or self._pending is not None or windows.shape != (1, 8, 82)
                or masks.shape != (1, 8) or masks.dtype != torch.bool or previous.shape != (1, 4)
                or windows.device.type != "cpu" or previous.device.type != "cpu"):
            raise ValueError("one active transfer and confirmed prior control are required")
        count = int(masks.sum())
        if count < 1 or masks[0].tolist() != [False] * (8-count) + [True] * count:
            raise ValueError("invalid public history mask")
        applied = _target(previous[0])
        if any(abs(a-b) > 3e-5 for a, b in zip(applied, self._expected_target)):
            raise ValueError("input target differs from the confirmed applied target")
        row = windows[0, -1]
        features = encode_public_row(row, applied, previous_mode=self._mode,
                                     new_instruction=self._new_instruction, control_event=self._control_event)
        prior = self._hidden
        with torch.no_grad():
            logits, angles, hidden = self.student.infer_row(features[None], previous, prior)
        self._hidden = hidden.detach()
        action = int(logits.argmax(-1))
        self._pending = {"action": action, "pre_joints": tuple((row[:4] * 180).tolist()),
                         "object_id": tuple(row[26:30].tolist())}
        self.calls.append({"mode": self._mode, "new_instruction": self._new_instruction,
                           "prior_hidden": prior is not None, "action": action,
                           "event_after_valid": bool(self._control_event and self._control_event.after_deg is not None),
                           "previous_applied_target_deg": list(applied), "encoded": features.tolist()})
        self._new_instruction = False
        if action != motion_policy.MOVE:
            angles = previous / 180.0
        return logits, angles

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        if self._pending is None or self._pending["action"] == motion_policy.DONE:
            raise ValueError("no physical command is awaiting confirmation")
        expected_action = motion_policy.ACTIONS[self._pending["action"]]
        sent, actual = _target(transmitted_target_deg), _target(applied_target_deg)
        opening = _opening(gripper_opening_m)
        expected_opening = .04 if action == "open" else 0. if action == "close" else self._expected_opening
        if (action != expected_action or any(abs(a-b) > 3e-5 for a, b in zip(sent, actual))
                or (action != "move" and any(abs(a-b) > 3e-5 for a, b in zip(actual, self._expected_target)))
                or expected_opening is None or not math.isclose(opening, expected_opening, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("confirmed control does not match the pending command")
        after = None if self.return_joint_observer is None else _vector(self.return_joint_observer(), "return-time joints")
        self._control_event = ControlEvent(self._pending["action"], self._pending["object_id"],
                                           self._pending["pre_joints"], after)
        self._mode = self._pending["action"]
        self._expected_target = actual
        self._expected_opening = opening
        self._pending = None

    def on_expected_transfer_committed(self, _before, _task, _after):
        if not self._active or self._pending is None or self._pending["action"] != motion_policy.DONE:
            raise ValueError("END and a normal expected-scene commit are required")
        self._pending = None
        self._mode = motion_policy.DONE
        self._active = False
