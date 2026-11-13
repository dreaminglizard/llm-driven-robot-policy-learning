"""Experimental fixed-period, directly commanded trajectory student.

At each tick the model consumes only current and preceding registered-scene
features. Its position, signed velocity, gripper and termination outputs all
come from learned heads. There is no waypoint interpolator in this module.
This is an offline research policy, not a hardware controller or safety guard.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import scene_features


FORMAT = "raccoon-direct-trajectory-student.v1"
JOINT_SCALE_DEG = 180.0
VELOCITY_SCALE_DEG_S = 180.0
SIMULATED_VELOCITY_LIMIT_DEG_S = 180.0
NUMERICAL_POSITION_MARGIN_DEG = 0.001
GRIPPER_STATES = ("open", "closed")
TERMINAL_SEMANTICS = ("done_no_command", "apply_then_finish")
ACTUATION_SEMANTICS = ("position_target_with_velocity_force", "integrated_velocity_with_position_target")
MAX_PARAMETERS = 250_000


def _period(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("step_seconds must be finite and positive")
    return float(value)


def _history_offsets(offsets, history_length):
    if offsets is None:
        return tuple(range(history_length - 1, -1, -1))
    if (not isinstance(offsets, (tuple, list)) or len(offsets) != history_length
            or any(type(value) is not int or value < 0 for value in offsets)
            or offsets[-1] != 0
            or any(left <= right for left, right in zip(offsets, offsets[1:]))):
        raise ValueError("history_offsets must be strictly decreasing nonnegative integers ending at zero")
    return tuple(offsets)


def history_window_offsets(history, offsets):
    """Select real previous observations; missing offsets are left padded."""
    if not isinstance(history, (tuple, list)) or not history:
        raise ValueError("history must contain at least the current observation")
    offsets = _history_offsets(offsets, len(offsets))
    rows, masks = [], []
    for offset in offsets:
        index = len(history)-1-offset
        if index < 0:
            rows.append([0.0] * len(scene_features.FEATURE_NAMES))
            masks.append(False)
        else:
            row = history[index]
            if (not isinstance(row, (list, tuple)) or len(row) != len(scene_features.FEATURE_NAMES)
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
                raise ValueError("history contains an invalid feature row")
            rows.append(list(row))
            masks.append(True)
    return rows, masks


class TrajectoryPolicy(nn.Module):
    """A small history-conditioned network emitting one direct command per tick.

    Position and velocity heads respectively emit degrees / 180 and signed
    degrees per second / 180. Gripper and terminal heads emit logits. Inputs
    are the existing 82-feature observation and a left-padded history mask.
    Neither the next observation nor simulator qvel is a model input.
    """

    def __init__(self, *, step_seconds: float, hidden_size: int = 64,
                 history_length: int = scene_features.HISTORY_LENGTH, position_residual: bool = False,
                 bounded_outputs: bool = False, history_offsets: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        self.step_seconds = _period(step_seconds)
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(history_length) is not int or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        if type(position_residual) is not bool:
            raise ValueError("position_residual must be boolean")
        if type(bounded_outputs) is not bool:
            raise ValueError("bounded_outputs must be boolean")
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.history_offsets = _history_offsets(history_offsets, history_length)
        self.position_residual = position_residual
        self.bounded_outputs = bounded_outputs
        self.terminal_semantics = "done_no_command"
        self.actuation_semantics = "position_target_with_velocity_force"
        self.direct_position_label = None
        self.feature_count = len(scene_features.FEATURE_NAMES)
        limits = torch.tensor(JOINT_LIMITS, dtype=torch.float32)
        self.register_buffer("joint_center_deg", limits.mean(1), persistent=False)
        self.register_buffer("joint_half_range_deg", (limits[:, 1] - limits[:, 0]) / 2
                             - NUMERICAL_POSITION_MARGIN_DEG, persistent=False)
        self.encoder = nn.Sequential(
            nn.Linear(history_length * (self.feature_count + 1), hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.joint_head = nn.Linear(hidden_size, 4)
        self.velocity_head = nn.Linear(hidden_size, 4)
        self.gripper_head = nn.Linear(hidden_size, len(GRIPPER_STATES))
        self.done_head = nn.Linear(hidden_size, 1)
        if sum(p.numel() for p in self.parameters()) > MAX_PARAMETERS:
            raise ValueError(f"model must have at most {MAX_PARAMETERS} parameters")

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        if (not isinstance(windows, torch.Tensor) or not windows.is_floating_point()
                or windows.ndim != 3 or windows.shape[1:] != (self.history_length, self.feature_count)
                or windows.shape[0] == 0 or not bool(torch.isfinite(windows).all())):
            raise ValueError("windows must be a finite batch of fixed-length feature histories")
        if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                or masks.shape != windows.shape[:2] or masks.device != windows.device):
            raise ValueError("masks must be boolean and match the batch and history")
        if not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any()):
            raise ValueError("masks must be left-padded and include the current observation")
        masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
        encoded = self.encoder(torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1))
        position = self.joint_head(encoded)
        velocity = self.velocity_head(encoded)
        if self.bounded_outputs and self.position_residual:
            current_deg = windows[:, -1, :4] * JOINT_SCALE_DEG
            current_in_range = ((current_deg - self.joint_center_deg) / self.joint_half_range_deg).clamp(
                min=-1 + 1e-6, max=1 - 1e-6)
            position = (self.joint_center_deg + self.joint_half_range_deg *
                        torch.tanh(torch.atanh(current_in_range) + position)) / JOINT_SCALE_DEG
        elif self.bounded_outputs:
            position = (self.joint_center_deg + self.joint_half_range_deg * torch.tanh(position)) / JOINT_SCALE_DEG
        elif self.position_residual:
            # Current observed joints are already normalized by 180. This skip
            # connection is inside the network and produces the direct target.
            position = position + windows[:, -1, :4]
        if self.bounded_outputs:
            velocity = torch.tanh(velocity) * (SIMULATED_VELOCITY_LIMIT_DEG_S / VELOCITY_SCALE_DEG_S)
        return (position, velocity,
                self.gripper_head(encoded), self.done_head(encoded).squeeze(-1))


def checkpoint_payload(model: TrajectoryPolicy, *, provenance: dict) -> dict:
    if not isinstance(model, TrajectoryPolicy) or not isinstance(provenance, dict):
        raise ValueError("trajectory model and provenance object are required")
    return {
        "format": FORMAT,
        "configuration": {"step_seconds": model.step_seconds, "hidden_size": model.hidden_size,
                          "history_length": model.history_length, "position_residual": model.position_residual,
                          "bounded_outputs": model.bounded_outputs, "history_offsets": list(model.history_offsets)},
        "output_bounds": ({"joint_limits_deg": [list(pair) for pair in JOINT_LIMITS],
                           "numerical_position_margin_deg": NUMERICAL_POSITION_MARGIN_DEG,
                           "simulation_velocity_abs_deg_s": SIMULATED_VELOCITY_LIMIT_DEG_S,
                           "scope": "simulation research limits, not verified physical limits"}
                          if model.bounded_outputs else None),
        "feature_names": list(scene_features.FEATURE_NAMES),
        "joint_scale_deg": JOINT_SCALE_DEG,
        "velocity_scale_deg_s": VELOCITY_SCALE_DEG_S,
        "gripper_states": list(GRIPPER_STATES),
        "terminal_semantics": model.terminal_semantics,
        "actuation_semantics": model.actuation_semantics,
        "direct_position_label": model.direct_position_label,
        "provenance": copy.deepcopy(provenance),
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }


def load_checkpoint(checkpoint: dict | str | Path) -> TrajectoryPolicy:
    """Load a locally generated research checkpoint; reject contract changes."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT
            or checkpoint.get("feature_names") != list(scene_features.FEATURE_NAMES)
            or checkpoint.get("joint_scale_deg") != JOINT_SCALE_DEG
            or checkpoint.get("velocity_scale_deg_s") != VELOCITY_SCALE_DEG_S
            or checkpoint.get("gripper_states") != list(GRIPPER_STATES)
            or not isinstance(checkpoint.get("configuration"), dict)
            or not isinstance(checkpoint.get("provenance"), dict)
            or not isinstance(checkpoint.get("state_dict"), dict)):
        raise ValueError("unsupported trajectory checkpoint contract")
    try:
        model = TrajectoryPolicy(**checkpoint["configuration"])
        semantics = checkpoint.get("terminal_semantics", "done_no_command")
        if semantics not in TERMINAL_SEMANTICS:
            raise ValueError("unsupported terminal semantics")
        model.terminal_semantics = semantics
        actuation = checkpoint.get("actuation_semantics", "position_target_with_velocity_force")
        if actuation not in ACTUATION_SEMANTICS:
            raise ValueError("unsupported actuation semantics")
        position_label = checkpoint.get("direct_position_label")
        if (actuation == "integrated_velocity_with_position_target"
                and position_label != "current_applied_position_servo_target"):
            raise ValueError("integrated actuation requires the start-of-interval position label")
        model.actuation_semantics = actuation
        model.direct_position_label = position_label
        if model.bounded_outputs and checkpoint.get("output_bounds") != {
                "joint_limits_deg": [list(pair) for pair in JOINT_LIMITS],
                "numerical_position_margin_deg": NUMERICAL_POSITION_MARGIN_DEG,
                "simulation_velocity_abs_deg_s": SIMULATED_VELOCITY_LIMIT_DEG_S,
                "scope": "simulation research limits, not verified physical limits"}:
            raise ValueError("bounded checkpoint limits differ from the current research limits")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (TypeError, KeyError, RuntimeError) as exc:
        raise ValueError("invalid trajectory checkpoint parameters") from exc
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("nonfinite trajectory checkpoint parameters")
    return model.eval()


class TrajectoryPolicyRuntime:
    """One transfer's independent causal history, with no hardware connection.

    The caller must invoke step at the checkpoint's fixed step_seconds and apply
    or reject the returned position and signed velocity as a pair. Values are
    raw network outputs, not validated device commands. Scene is the registered
    or expected layout, never measured hidden simulator object positions.
    """

    def __init__(self, checkpoint: TrajectoryPolicy | dict | str | Path, *, registered_scene,
                 task, nominal_slot_xy_mm=scene_features.NOMINAL_SLOT_XY_MM):
        self.model = (checkpoint if isinstance(checkpoint, TrajectoryPolicy)
                      else load_checkpoint(checkpoint))
        if any(value.device.type != "cpu" for value in self.model.parameters()):
            raise ValueError("trajectory runtime requires a CPU model")
        self.model.eval()
        before, canonical = scene_features.transfer_task(registered_scene, task)
        self.registered_scene = copy.deepcopy(before)
        self.task = canonical
        self.nominal_slot_xy_mm = copy.deepcopy(nominal_slot_xy_mm)
        self.step_seconds = self.model.step_seconds
        self.terminal_semantics = self.model.terminal_semantics
        self.actuation_semantics = self.model.actuation_semantics
        self.direct_position_label = self.model.direct_position_label
        self._history = []

    def reset(self):
        self._history.clear()

    def step(self, observation: dict) -> dict:
        features = scene_features.encode_scene_features(
            observation, self.registered_scene, self.task, self.nominal_slot_xy_mm)
        history = (self._history + [features])[-(self.model.history_offsets[0] + 1):]
        window, mask = history_window_offsets(history, self.model.history_offsets)
        with torch.inference_mode():
            positions, velocities, gripper_logits, done_logits = self.model(
                torch.tensor([window], dtype=torch.float32), torch.tensor([mask], dtype=torch.bool))
        if not all(bool(torch.isfinite(value).all())
                   for value in (positions, velocities, gripper_logits, done_logits)):
            raise ValueError("trajectory model returned nonfinite outputs")
        result = {
            "joint_target_deg": (positions[0] * JOINT_SCALE_DEG).tolist(),
            "joint_velocity_deg_s": (velocities[0] * VELOCITY_SCALE_DEG_S).tolist(),
            "gripper": GRIPPER_STATES[int(gripper_logits[0].argmax())],
            "done": bool(done_logits[0] >= 0),
            "step_seconds": self.step_seconds,
            "terminal_semantics": self.terminal_semantics,
            "actuation_semantics": self.actuation_semantics,
            "direct_position_label": self.direct_position_label,
        }
        self._history = history
        return result
