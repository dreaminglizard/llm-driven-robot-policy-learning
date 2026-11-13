"""An isolated 88-feature policy using only nominal relative geometry.

The external observation remains the existing causal 82-feature window. For
each valid row, six values are appended: registered source and destination
object-center XYZ minus the current nominal FK TCP, divided by 300 mm. The
registered centers assume table Z=0 and identical 20 mm cylinders; they are
not sensed object positions. This module performs no device or simulator I/O.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_control.kinematics import Geometry
from raccoon_memory import motion_policy, scene_features, scene_fk_loss, scene_policy


HISTORY_LENGTH = 8
SOURCE_FEATURE_COUNT = 82
RELATIVE_FEATURE_COUNT = 6
FEATURE_COUNT = SOURCE_FEATURE_COUNT + RELATIVE_FEATURE_COUNT
HIDDEN_SIZE = 64
POSITION_SCALE_MM = 300.0
TABLE_Z_MM = 0.0
CYLINDER_HEIGHT_MM = 20.0
CHECKPOINT_FORMAT = "raccoon-scene-relative-xyz88-exclusive-policy.v1"
FEATURE_NAMES = tuple(scene_features.FEATURE_NAMES) + tuple(
    f"{role}_nominal_center_minus_tcp_{axis}_mm/300"
    for role in ("source", "destination") for axis in ("x", "y", "z")
)
_SOURCE_NAMES = (
    *motion_policy.FEATURE_NAMES[:14],
    *(f"source_slot_{i}" for i in range(1, 5)),
    *(f"destination_slot_{i}" for i in range(1, 5)),
    *(f"source_level_{i}" for i in range(1, 3)),
    *(f"destination_level_{i}" for i in range(1, 3)),
    *(f"moved_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_level_{level}_id_{obj}"
      for slot in range(1, 5) for level in range(1, 3) for obj in range(5)),
    *(f"active_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_{axis}_mm/300" for slot in range(1, 5) for axis in ("x", "y")),
)
_DYNAMIC_NAMES = (
    "joint_1_deg/180", "joint_2_deg/180", "joint_3_deg/180", "joint_4_deg/180",
    "sensor_valid", "moving", "gripper_command_none", "gripper_command_open",
    "gripper_command_closed", "gripper_reported_none", "gripper_reported_open",
    "gripper_reported_closed", "gripper_reported_valid", "gripper_feedback_fresh",
)


def _check_contract() -> Geometry:
    geometry = Geometry()
    if (scene_features.SCHEMA != "raccoon-scene-policy-features.v1"
            or tuple(scene_features.FEATURE_NAMES) != _SOURCE_NAMES
            or tuple(motion_policy.FEATURE_NAMES[:14]) != _DYNAMIC_NAMES
            or scene_features.HISTORY_LENGTH != HISTORY_LENGTH
            or scene_features.SLOT_SCALE_MM != POSITION_SCALE_MM
            or motion_policy.JOINT_SCALE_DEG != 180.0
            or tuple(motion_policy.ACTIONS) != ("move", "open", "close", "done")
            or geometry != Geometry(base_mm=82.5, upper_mm=100.0, forearm_mm=100.0,
                                    tool_forward_mm=80.0, tool_vertical_mm=-6.0)):
        raise ValueError("existing feature, action, scale or nominal geometry contract differs")
    return geometry


def _one_hot_index(values: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(((values == 0) | (values == 1)).all()) or not bool((values.sum(dim=1) == 1).all()):
        raise ValueError(f"{name} must be one-hot on each valid observation")
    return values.argmax(dim=1)


def extend_scene_features(windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Append six nominal relative values to each valid 82-feature row.

    Inputs are finite floating [batch, 8, 82] windows and left-padded boolean
    [batch, 8] masks. Existing values are copied unchanged, including padding;
    the six added values are zero on padding. Each row uses its own current
    joints and registered task. The caller owns the causal history boundary.
    """
    geometry = _check_contract()
    if (not isinstance(windows, torch.Tensor) or not windows.is_floating_point()
            or windows.ndim != 3 or windows.shape[1:] != (HISTORY_LENGTH, SOURCE_FEATURE_COUNT)
            or not windows.shape[0] or not bool(torch.isfinite(windows).all())):
        raise ValueError("windows must be nonempty finite floating tensors of shape (batch, 8, 82)")
    if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
            or masks.shape != windows.shape[:2] or masks.device != windows.device
            or not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any())):
        raise ValueError("masks must be left-padded and include the current observation")

    rows = windows[masks]
    if not bool((rows[:, 4] == 1).all()):
        raise ValueError("joint sensor must be valid for every observation")
    source = _one_hot_index(rows[:, 14:18], "source slot")
    destination = _one_hot_index(rows[:, 18:22], "destination slot")
    source_level = _one_hot_index(rows[:, 22:24], "source level")
    destination_level = _one_hot_index(rows[:, 24:26], "destination level")
    if bool((source == destination).any()):
        raise ValueError("source and destination slots must differ")
    joint_degrees = rows[:, :4] * motion_policy.JOINT_SCALE_DEG
    if not bool(torch.isfinite(joint_degrees).all()):
        raise ValueError("joint angles must remain finite after conversion to degrees")
    tcp_mm, _ = scene_fk_loss.nominal_forward(joint_degrees, geometry)
    if not bool(torch.isfinite(tcp_mm).all()):
        raise ValueError("nominal TCP must be finite")

    registered_xy = rows[:, 74:82].reshape(-1, 4, 2)  # already divided by 300 mm
    row_index = torch.arange(rows.shape[0], device=rows.device)
    tcp_scaled = tcp_mm / POSITION_SCALE_MM

    def relative(slot: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        # Encoded level index is zero-based: level 1 center=10 mm; level 2=30 mm.
        center_z = (TABLE_Z_MM + CYLINDER_HEIGHT_MM * (level.to(rows.dtype) + 0.5)) / POSITION_SCALE_MM
        return torch.cat((registered_xy[row_index, slot] - tcp_scaled[:, :2],
                          (center_z - tcp_scaled[:, 2]).unsqueeze(1)), dim=1)

    additions = torch.cat((relative(source, source_level), relative(destination, destination_level)), dim=1)
    if not bool(torch.isfinite(additions).all()):
        raise ValueError("relative nominal features must be finite")
    padded_additions = windows.new_zeros((*windows.shape[:2], RELATIVE_FEATURE_COUNT))
    padded_additions[masks] = additions
    return torch.cat((windows, padded_additions), dim=-1)


class SceneRelativePolicy(nn.Module):
    """Exclusive MLP64: 82 original plus 6 nominal features at each of 8 times."""

    def __init__(self):
        super().__init__()
        _check_contract()
        self.architecture = "relative_xyz_mlp"
        self.history_length, self.source_feature_count = HISTORY_LENGTH, SOURCE_FEATURE_COUNT
        self.feature_count, self.hidden_size = FEATURE_COUNT, HIDDEN_SIZE
        self.encoder = nn.Sequential(nn.Linear(HISTORY_LENGTH * (FEATURE_COUNT + 1), HIDDEN_SIZE),
                                     nn.ReLU(), nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU())
        self.action_head = nn.Linear(HIDDEN_SIZE, len(motion_policy.ACTIONS))
        self.joint_head = nn.Linear(HIDDEN_SIZE, 4)
        if sum(parameter.numel() for parameter in self.parameters()) > scene_policy.MAX_PARAMETERS:
            raise ValueError("parameter limit exceeded")

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        extended = extend_scene_features(windows, masks)
        masked = extended.masked_fill(~masks.unsqueeze(-1), 0.0)
        values = torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1)
        encoded = self.encoder(values)
        return self.action_head(encoded), self.joint_head(encoded)


def initialize_from_scene_policy(source: scene_policy.ScenePolicy) -> SceneRelativePolicy:
    """Copy the existing MLP exactly, zeroing only the six new columns per time.

    No optimizer is created or copied. The caller chooses its checkpoint and
    optimizer; neither source tensors nor the caller's CPU RNG state is changed.
    """
    _check_contract()
    if (type(source) is not scene_policy.ScenePolicy
            or (source.architecture, source.hidden_size, source.history_length, source.feature_count)
            != ("mlp", HIDDEN_SIZE, HISTORY_LENGTH, SOURCE_FEATURE_COUNT)):
        raise ValueError("source must be the existing 82-feature history-8 MLP64 policy")
    state = source.state_dict()
    if any(not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("source parameters must be finite floating tensors")
    devices = {value.device for value in state.values()}
    dtypes = {value.dtype for value in state.values()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("source parameters must share a device and floating dtype")
    with torch.random.fork_rng(devices=[]):
        target = SceneRelativePolicy().to(device=next(iter(devices)), dtype=next(iter(dtypes)))
    target_state = target.state_dict()
    if set(state) != set(target_state) or tuple(state["encoder.0.weight"].shape) != (64, 8 * 83):
        raise ValueError("source parameter names or first-layer shape differ")
    with torch.no_grad():
        for name, value in target_state.items():
            if name == "encoder.0.weight":
                continue
            if value.shape != state[name].shape:
                raise ValueError(f"source parameter shape differs: {name}")
            value.copy_(state[name])
        old = state["encoder.0.weight"].reshape(HIDDEN_SIZE, HISTORY_LENGTH, SOURCE_FEATURE_COUNT + 1)
        new = target_state["encoder.0.weight"].reshape(HIDDEN_SIZE, HISTORY_LENGTH, FEATURE_COUNT + 1)
        new.zero_()
        new[:, :, :SOURCE_FEATURE_COUNT].copy_(old[:, :, :SOURCE_FEATURE_COUNT])
        new[:, :, FEATURE_COUNT].copy_(old[:, :, SOURCE_FEATURE_COUNT])
    target.load_state_dict(target_state, strict=True)
    target.train(source.training)
    return target


def checkpoint_contract() -> dict:
    """Return an explicit portable format for this distinct input contract."""
    _check_contract()
    return {
        "format": CHECKPOINT_FORMAT,
        "source_schema": scene_features.SCHEMA,
        "source_feature_names": list(scene_features.FEATURE_NAMES),
        "extended_feature_names": list(FEATURE_NAMES),
        "history_length": HISTORY_LENGTH, "hidden_size": HIDDEN_SIZE,
        "external_window_features": SOURCE_FEATURE_COUNT, "extended_window_features": FEATURE_COUNT,
        "position_scale_mm": POSITION_SCALE_MM, "table_z_mm": TABLE_Z_MM,
        "cylinder_height_mm": CYLINDER_HEIGHT_MM,
        "nominal_geometry_mm": {"base": 82.5, "upper": 100.0, "forearm": 100.0,
                                "tool_forward": 80.0, "tool_vertical": -6.0},
        "action_names": list(motion_policy.ACTIONS),
        "joint_parameterization": "absolute_degrees_div_180",
    }


def checkpoint_payload(model: SceneRelativePolicy) -> dict:
    if type(model) is not SceneRelativePolicy:
        raise ValueError("relative policy required")
    payload = checkpoint_contract()
    payload["state_dict"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict) -> SceneRelativePolicy:
    """Load only finite float32 CPU tensors with the exact relative contract."""
    contract = checkpoint_contract()
    if not isinstance(payload, dict) or any(payload.get(k) != value for k, value in contract.items()):
        raise ValueError("relative checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("relative checkpoint requires a state dictionary")
    with torch.random.fork_rng(devices=[]):
        model = SceneRelativePolicy()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("relative checkpoint parameter names differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError("relative checkpoint parameter shape, dtype or value differs: " + name)
    model.load_state_dict(state, strict=True)
    return model
