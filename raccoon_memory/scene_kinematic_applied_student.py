"""Add public nominal kinematics to the same causal applied-action student.

One learned projection combines nominal TCP/task features with the existing
92 public inputs before the same GRU. Its zero initialization preserves the
reference student's initial computation. The action and absolute joint heads
remain learned; no action router, simulator state or geometric target corrector
is used. Nominal geometry is a calibrated assumption, not object observation.
Inputs that overflow nominal FK are rejected; warm-start equivalence concerns
the physical public-input domain, not arbitrary unbounded floating values.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_control.kinematics import Geometry
from . import motion_policy, scene_features
from .scene_applied_action_student import AppliedActionStudent, FEATURE_COUNT, HIDDEN_SIZE
from .scene_fk_loss import nominal_forward


FORMAT = "raccoon-scene-kinematic-applied-action-student.v1"
CHECKPOINT_FORMAT = "raccoon-kinematic-applied-action-checkpoint.v1"
DERIVED_FEATURE_NAMES = tuple(
    f"{name}_{axis}" for name in (
        "current_tcp_mm/300", "previous_target_tcp_mm/300",
        "source_grasp_tcp_mm/300", "destination_grasp_tcp_mm/300",
        "source_minus_current_tcp_mm/300", "destination_minus_current_tcp_mm/300",
        "source_minus_previous_target_tcp_mm/300", "destination_minus_previous_target_tcp_mm/300",
        "current_tool_axis", "previous_target_tool_axis") for axis in ("x", "y", "z"))
DERIVED_FEATURE_COUNT = len(DERIVED_FEATURE_NAMES)
_GEOMETRY = Geometry()
NOMINAL_CONSTANTS = (_GEOMETRY.base_mm, _GEOMETRY.upper_mm, _GEOMETRY.forearm_mm,
                     _GEOMETRY.tool_forward_mm, _GEOMETRY.tool_vertical_mm,
                     20.0, 15.0, scene_features.SLOT_SCALE_MM)


def nominal_features(inputs, constants):
    """Compute only from public joints, applied target and registered task XY."""
    if (not isinstance(inputs, torch.Tensor) or inputs.shape[-1] != FEATURE_COUNT
            or not inputs.is_floating_point() or not isinstance(constants, torch.Tensor)
            or constants.shape != (8,) or constants.device != inputs.device or constants.dtype != inputs.dtype):
        raise ValueError("public 92-feature inputs and matching nominal constants are required")
    geometry = Geometry(*constants[:5].detach().cpu().tolist())
    joints = torch.stack((inputs[..., :4], inputs[..., 86:90]), dim=-2) * motion_policy.JOINT_SCALE_DEG
    positions, axes = nominal_forward(joints, geometry)
    current, previous = positions.unbind(dim=-2)
    current, previous = current / constants[7], previous / constants[7]
    current_axis, previous_axis = axes.unbind(dim=-2)
    slots = inputs[..., 74:82].reshape(*inputs.shape[:-1], 4, 2)
    source_xy = (slots * inputs[..., 14:18, None]).sum(dim=-2)
    destination_xy = (slots * inputs[..., 18:22, None]).sum(dim=-2)
    # Registered layers are one or two. The nominal grasp TCP is 15mm above
    # the table for layer one and 35mm for layer two, using the known 20mm part.
    source_z = (inputs[..., 23:24] * constants[5] + constants[6]) / constants[7]
    destination_z = (inputs[..., 25:26] * constants[5] + constants[6]) / constants[7]
    source = torch.cat((source_xy, source_z), dim=-1)
    destination = torch.cat((destination_xy, destination_z), dim=-1)
    return torch.cat((current, previous, source, destination, source - current, destination - current,
                      source - previous, destination - previous, current_axis, previous_axis), dim=-1)


class NominalFeatureGRU(nn.GRU):
    """A GRU with a learned zero-initialized nominal-feature input projection."""

    def __init__(self):
        super().__init__(FEATURE_COUNT, HIDDEN_SIZE, batch_first=True)
        self.feature_projection = nn.Linear(DERIVED_FEATURE_COUNT, FEATURE_COUNT, bias=False)
        nn.init.zeros_(self.feature_projection.weight)
        self.register_buffer("nominal_constants", torch.tensor(NOMINAL_CONSTANTS, dtype=torch.float32))

    def forward(self, inputs, hx=None):
        extra = nominal_features(inputs, self.nominal_constants)
        if not bool(torch.isfinite(extra).all()):
            raise ValueError("public inputs overflow nominal kinematic features")
        return super().forward(inputs + self.feature_projection(extra), hx)


class KinematicAppliedActionStudent(AppliedActionStudent):
    def __init__(self):
        super().__init__()
        self.gru = NominalFeatureGRU()


def warm_start(reference):
    """Copy the complete raw student and leave only the new projection at zero."""
    if type(reference) is not AppliedActionStudent:
        raise TypeError("a raw AppliedActionStudent checkpoint is required")
    with torch.random.fork_rng(devices=[]):
        model = KinematicAppliedActionStudent().cpu()
    original, state = reference.state_dict(), model.state_dict()
    added = {"gru.feature_projection.weight", "gru.nominal_constants"}
    if set(state) - set(original) != added or set(original) - set(state):
        raise ValueError("raw student tensor names differ from the warm-start contract")
    for name, value in original.items():
        if (value.device.type != "cpu" or value.dtype != state[name].dtype or value.shape != state[name].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError("raw student tensors differ from the warm-start contract")
        if name in ("joint_center_deg", "joint_half_range_deg") and not torch.equal(value, state[name]):
            raise ValueError("raw student joint limits differ")
        state[name] = value.detach().clone()
    model.load_state_dict(state, strict=True)
    return model.eval()


def model_contract():
    return {"model_format": FORMAT, "constructor": "KinematicAppliedActionStudent()",
            "feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE,
            "actions": list(motion_policy.ACTIONS), "joint_output": "absolute_robot_frame_degrees",
            "derived_feature_names": list(DERIVED_FEATURE_NAMES),
            "nominal_constants": list(NOMINAL_CONSTANTS),
            "input_projection": "raw_plus_learned_nominal_features_zero_initialized"}


def load_checkpoint(payload):
    """Read this structure's checkpoint; training provenance is checked separately."""
    from .scene_applied_action_policy import _encoded, _state_digest
    import hashlib
    if (not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("model") != model_contract() or not isinstance(payload.get("config"), dict)
            or payload.get("config_sha256") != hashlib.sha256(_encoded(payload["config"])).hexdigest()
            or not isinstance(payload.get("provenance"), dict) or not isinstance(payload.get("training"), dict)
            or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("unsupported kinematic applied-action checkpoint contract")
    with torch.random.fork_rng(devices=[]):
        model = KinematicAppliedActionStudent().cpu()
    expected, state = model.state_dict(), payload["state_dict"]
    if (set(state) != set(expected) or any(not isinstance(value, torch.Tensor)
            or value.device.type != "cpu" or value.dtype != expected[name].dtype or value.shape != expected[name].shape
            or not bool(torch.isfinite(value).all()) for name, value in state.items())
            or any(not torch.equal(state[name], value) for name, value in model.named_buffers())):
        raise ValueError("kinematic student tensors or nominal geometry differ")
    model.load_state_dict(state, strict=True)
    if _state_digest(model) != payload["training"].get("final_state_sha256"):
        raise ValueError("kinematic student weights fingerprint differs")
    return model.eval()
