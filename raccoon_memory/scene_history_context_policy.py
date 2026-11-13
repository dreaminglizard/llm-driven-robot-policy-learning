"""Single manipulation policy with observable prior-transfer context.

The session records only robot observations and normally returned commands.
The previous physical outcome and simulator object states are never used as
policy inputs or to decide whether the next transfer should run. This is a
simulation development implementation, not a hardware controller.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from raccoon_control.kinematics import Geometry
from . import motion_policy, scene_features, scene_unified_kinematic_policy as unified
from .scene_oracle_free_session import OracleFreeSceneModelSession

HISTORY_CONTEXT_SIZE = 29  # valid flag plus close/open measured and applied FK/joints
FORMAT = "raccoon-unified-observable-transfer-history.v1"


def _one_hot(row, start, size):
    values = row[start:start + size]
    if len(values) != size or any(value not in (0.0, 1.0) for value in values) or sum(values) != 1:
        raise ValueError("invalid task encoding in returned observation")
    return values.index(1) + 1


def _xy_relative(xy, nominal):
    radius = math.hypot(*nominal)
    if radius <= 1e-9:
        raise ValueError("nominal slot has no radial direction")
    dx, dy = xy[0] - nominal[0], xy[1] - nominal[1]
    return [(dx * nominal[0] + dy * nominal[1]) / radius / 300,
            (-dx * nominal[1] + dy * nominal[0]) / radius / 300]


def _event_features(event, slot, level):
    row = [float(value) for value in event["policy_features"]]
    prior = [float(value) for value in event["previous_applied_target_deg"]]
    if len(row) != 82 or len(prior) != 4 or not all(math.isfinite(v) for v in row + prior):
        raise ValueError("prior event lacks finite observable joints and returned target")
    nominal = [row[74 + 2 * (slot - 1)] * 300, row[75 + 2 * (slot - 1)] * 300]
    measured_joints = [value * 180 for value in row[:4]]
    height = 15 + 20 * (level - 1)
    geometry = Geometry()
    def relative(joints):
        xyz = geometry.forward(joints)
        return [*_xy_relative(xyz[:2], nominal), (xyz[2] - height) / 300]
    return [*relative(measured_joints), *relative(prior), *row[:4], *(value / 180 for value in prior)]


class TransferHistory:
    """Expected history per object; neither success nor object pose is stored."""

    def __init__(self):
        self._items = {}

    def clear(self):
        self._items.clear()

    def invalidate(self, object_id):
        self._items.pop(object_id, None)

    def record(self, task, decisions):
        """Record only gripper commands whose executor marked them applied."""
        self.invalidate(task["object_id"])
        chosen = {}
        for action in ("close", "open"):
            matches = [row for row in decisions if row.get("action") == action and row.get("result") == "command_applied"]
            if len(matches) != 1:
                return False
            event = matches[0]
            row, prior, command = event.get("policy_features"), event.get("previous_applied_target_deg"), event.get("command")
            if not isinstance(row, list) or not isinstance(prior, list) or not isinstance(command, dict):
                return False
            applied = command.get("joints_deg")
            if (not isinstance(applied, list) or len(applied) != 4 or len(prior) != 4 or
                    any(not math.isclose(float(a), float(b), abs_tol=1e-5, rel_tol=0) for a, b in zip(applied, prior))):
                return False
            if (_one_hot(row, 14, 4) != task["source_slot"] or _one_hot(row, 18, 4) != task["destination_slot"] or
                    _one_hot(row, 22, 2) != task["source_level"] or _one_hot(row, 24, 2) != task["destination_level"] or
                    _one_hot(row, 26, 4) != task["object_id"]):
                raise ValueError("recorded robot input does not match the transfer")
            chosen[action] = {"policy_features": list(row), "previous_applied_target_deg": list(prior)}
        self._items[task["object_id"]] = {"task": copy.deepcopy(task), **chosen}
        return True

    def context_for(self, task):
        item = self._items.get(task["object_id"])
        if (item is None or item["task"]["destination_slot"] != task["source_slot"] or
                item["task"]["destination_level"] != task["source_level"]):
            return [0.0] * HISTORY_CONTEXT_SIZE
        old = item["task"]
        result = [1.0, *_event_features(item["close"], old["source_slot"], old["source_level"]),
                  *_event_features(item["open"], old["destination_slot"], old["destination_level"])]
        if len(result) != HISTORY_CONTEXT_SIZE or not all(math.isfinite(value) for value in result):
            raise ValueError("history context is malformed")
        return result


class HistoryContextPolicy(unified.UnifiedKinematicPolicy):
    """Add a zero-start history projection to the one unified policy."""

    def __init__(self):
        super().__init__()
        self.history_projection = nn.Linear(HISTORY_CONTEXT_SIZE, self.hidden_size, bias=False)
        nn.init.zeros_(self.history_projection.weight)
        self._live_history = None  # not a parameter or checkpointed physical state

    def set_live_history(self, history):
        if history is None:
            self._live_history = None
            return
        values = tuple(float(x) for x in history)
        if len(values) != HISTORY_CONTEXT_SIZE or not all(math.isfinite(x) for x in values):
            raise ValueError("history must be finite and have the fixed size")
        self._live_history = values

    def training_outputs(self, windows, masks, previous_target_degrees, history=None):
        relative, prior = self._context(windows, masks, previous_target_degrees)
        lengths = self.base._validate(windows, masks)
        if history is None:
            if len(windows) == 1 and self._live_history is not None:
                history = windows.new_tensor([self._live_history])
            else:
                history = windows.new_zeros((len(windows), HISTORY_CONTEXT_SIZE))
        if (not isinstance(history, torch.Tensor) or history.shape != (len(windows), HISTORY_CONTEXT_SIZE) or
                history.dtype != windows.dtype or history.device != windows.device or not bool(torch.isfinite(history).all())):
            raise ValueError("history context must match policy batch, dtype and device")
        original = self.base._project_validated(windows, masks)
        first = (self.base.encoder[0](original) + self.context_input(torch.cat((relative, prior), dim=1))
                 + self.history_projection(history))
        encoded = self.base.encoder[3](self.base.encoder[2](self.base.encoder[1](first)))
        logits, base_joints = self.base.action_head(encoded), self.base.joint_head(encoded)
        current = windows[:, -1, :].index_select(1, self._current_columns)
        early = self.early_head(torch.cat((encoded, current), dim=1))
        late = self.late_head(current)
        ungated = base_joints + torch.where((lengths < unified.baseline.MIN_LATE_HISTORY).unsqueeze(1), early, late)
        joints = torch.where((logits.argmax(dim=1) == motion_policy.MOVE).unsqueeze(1), ungated, base_joints)
        return unified.UnifiedOutputs(logits, joints, ungated, lengths, relative, prior)


def from_unified(source):
    if type(source) is not unified.UnifiedKinematicPolicy:
        raise ValueError("a unified single-policy source is required")
    result = HistoryContextPolicy().to(dtype=next(source.parameters()).dtype, device=next(source.parameters()).device)
    missing, extra = result.load_state_dict(source.state_dict(), strict=False)
    if missing != ["history_projection.weight"] or extra:
        raise ValueError("unified source does not match the history policy")
    result.requires_grad_(True)
    result.train(source.training)
    return result


def payload(model):
    if type(model) is not HistoryContextPolicy:
        raise ValueError("history policy required")
    return {"contract": FORMAT, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def load(payload_value):
    if not isinstance(payload_value, dict) or payload_value.get("contract") != FORMAT:
        raise ValueError("history checkpoint contract differs")
    model = HistoryContextPolicy()
    state = payload_value.get("state_dict")
    if not isinstance(state, dict) or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("history checkpoint tensors are invalid")
    model.load_state_dict(state, strict=True)
    return model.eval()


class HistoryContextSession(OracleFreeSceneModelSession):
    """Attach prior returned robot commands to the next single-policy call."""

    def __init__(self, *args, use_history=True, **kwargs):
        super().__init__(*args, **kwargs)
        actors = [item for item in self.student.modules() if isinstance(item, HistoryContextPolicy)]
        if len(actors) != 1:
            raise ValueError("exactly one history-aware policy is required")
        self._history_actor = actors[0]
        self._history = TransferHistory()
        self._use_history = bool(use_history)

    def _register(self, value):
        super()._register(value)
        self._history.clear()
        self._history_actor.set_live_history(None)

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        self._ready()
        _, task = scene_features.transfer_task(self.current, action)
        history = self._history.context_for(task) if self._use_history else [0.0] * HISTORY_CONTEXT_SIZE
        self._history_actor.set_live_history(history)
        event_index = len(self.events)
        try:
            super()._move(action, commit_lock=commit_lock, is_current=is_current,
                          on_transfer_commit=on_transfer_commit)
        except BaseException:
            self._history.invalidate(task["object_id"])
            raise
        else:
            events = [event for event in self.events[event_index:] if event.get("event") == "transfer"]
            if len(events) != 1 or events[0].get("committed") is not True:
                raise RuntimeError("expected transfer event was not committed")
            events[0]["prior_robot_history_available"] = bool(history[0])
            events[0]["prior_robot_history_context"] = list(history)
            events[0]["prior_transfer_physical_success"] = None
            self._history.record(task, events[0]["result"]["decisions"])
        finally:
            self._history_actor.set_live_history(None)
