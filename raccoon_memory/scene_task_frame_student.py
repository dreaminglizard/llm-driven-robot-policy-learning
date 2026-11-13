"""Fresh shared-slot/LSTM policy in registered destination-relative coordinates.

The raw input is the same causal 92-value public observation used by the applied
action dataset. Only registered nominal geometry and current/previous applied
joint angles enter the feature transform. The destination grasp point is a
fixed reference for one transfer, not a measurement of an object's position.
All parameters are freshly initialized; no prior student weights are required.

MOVE output is (relative XYZ / 300mm, unwrapped tool pitch / 180 degrees).
The other actions ignore this output. Static exact-pose IK belongs to the
execution adapter; it neither chooses actions nor provides a teaching target.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import torch
from torch import nn

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import motion_policy
from .scene_applied_action_student import (
    FEATURE_COUNT, PREVIOUS_ACTION_SLICE, PREVIOUS_TARGET_SLICE,
    TRANSFER_START_INDEX, REGISTRATION_START_INDEX,
)
from .scene_fk_loss import nominal_forward


FORMAT = "raccoon-scene-task-frame-student.v1"
CHECKPOINT_FORMAT = "raccoon-task-frame-student-checkpoint.v1"
HIDDEN_SIZE = 96
SLOT_FEATURE_COUNT = 16
SLOT_EMBEDDING_SIZE = 32
GLOBAL_FEATURE_COUNT = 62
ENCODED_FEATURE_COUNT = SLOT_EMBEDDING_SIZE * 3 + GLOBAL_FEATURE_COUNT
_GEOMETRY = Geometry()
NOMINAL_CONSTANTS = (
    _GEOMETRY.base_mm, _GEOMETRY.upper_mm, _GEOMETRY.forearm_mm,
    _GEOMETRY.tool_forward_mm, _GEOMETRY.tool_vertical_mm,
    20.0, 15.0, 300.0,
)


class TaskFrameOutputs(NamedTuple):
    action_logits: torch.Tensor   # [batch,time,4] in motion_policy.ACTIONS order
    move_pose: torch.Tensor       # [batch,time,4], relative XYZ/300, pitch/180
    state: tuple[torch.Tensor, torch.Tensor]  # hidden and cell, each [1,batch,96]


def _inputs(features):
    if (not isinstance(features, torch.Tensor) or features.ndim < 1
            or features.shape[-1] != FEATURE_COUNT or not features.is_floating_point()):
        raise ValueError("floating public inputs with last dimension 92 are required")


def _constants(features, constants):
    if constants is None:
        return features.new_tensor(NOMINAL_CONSTANTS)
    if (not isinstance(constants, torch.Tensor) or constants.shape != (8,)
            or constants.device != features.device or constants.dtype != features.dtype
            or not bool(torch.isfinite(constants).all()) or bool((constants[[1, 2, 5, 7]] <= 0).any())):
        raise ValueError("matching finite nominal geometry constants are required")
    return constants


def frame_origin(features, constants=None):
    """Return the registered destination grasp point in nominal world mm.

    Axes are parallel to the robot's world frame. A source-to-destination
    direction is never normalized, so there is no directional-frame singularity.
    For layer one z=15mm and layer two z=35mm with the declared constants.
    """
    _inputs(features)
    constants = _constants(features, constants)
    destination = features[..., 18:22]
    level = features[..., 24:26]
    if (not bool(torch.isfinite(features[..., 74:82]).all())
            or bool(((destination != 0) & (destination != 1)).any())
            or bool((destination.sum(-1) != 1).any())
            or bool(((level != 0) & (level != 1)).any()) or bool((level.sum(-1) != 1).any())):
        raise ValueError("a registered destination and layer are required")
    slots = features[..., 74:82].reshape(*features.shape[:-1], 4, 2)
    xy = (slots * destination[..., None]).sum(-2) * constants[7]
    z = features[..., 25:26] * constants[5] + constants[6]
    return torch.cat((xy, z), dim=-1)


def applied_targets_to_pose(features, joints_deg, constants=None):
    """Encode confirmed MOVE labels without reading any actual object state.

    Call on valid, confirmed MOVE rows only. Context/non-MOVE/NaN targets are
    not inferred or replaced. Dataset provenance remains the caller's concern.
    """
    _inputs(features)
    constants = _constants(features, constants)
    if (not isinstance(joints_deg, torch.Tensor) or joints_deg.shape != (*features.shape[:-1], 4)
            or joints_deg.dtype != features.dtype or joints_deg.device != features.device
            or not bool(torch.isfinite(joints_deg).all())):
        raise ValueError("matching finite confirmed MOVE targets in degrees are required")
    limits = joints_deg.new_tensor(JOINT_LIMITS)
    if bool(((joints_deg < limits[:, 0] - 1e-4) | (joints_deg > limits[:, 1] + 1e-4)).any()):
        raise ValueError("confirmed MOVE target is outside joint limits")
    geometry = Geometry(*constants[:5].detach().cpu().tolist())
    xyz, _ = nominal_forward(joints_deg, geometry)
    pitch = joints_deg[..., 1:].sum(-1, keepdim=True) + 90.0
    return torch.cat(((xyz - frame_origin(features, constants)) / constants[7], pitch / 180.0), dim=-1)


def pose_to_world(features, move_pose, constants=None):
    """Decode a proposed pose to world XYZ mm and unwrapped pitch degrees.

    This is a coordinate transform, not IK, reachability or collision checking.
    """
    _inputs(features)
    constants = _constants(features, constants)
    if (not isinstance(move_pose, torch.Tensor) or move_pose.shape != (*features.shape[:-1], 4)
            or move_pose.device != features.device or move_pose.dtype != features.dtype
            or not bool(torch.isfinite(move_pose).all())):
        raise ValueError("matching finite destination-relative poses are required")
    world = torch.cat((frame_origin(features, constants) + move_pose[..., :3] * constants[7],
                       move_pose[..., 3:4] * 180.0), dim=-1)
    if not bool(torch.isfinite(world).all()):
        raise ValueError("task-frame pose overflows nominal world coordinates")
    return world


def public_features(features, *, include_object_ids=True, constants=None):
    """Return four shared slot tokens and causal robot/task features.

    Both ID conditions have exactly the same widths. Removing ID leaves each
    layer's occupancy, task layers, source/destination slots and all geometry.
    The memory/planning system outside this model still retains object IDs.
    """
    _inputs(features)
    constants = _constants(features, constants)
    if type(include_object_ids) is not bool:
        raise ValueError("include_object_ids must be boolean")
    slots = features[..., 74:82].reshape(*features.shape[:-1], 4, 2)
    nodes = features[..., 30:70].reshape(*features.shape[:-1], 4, 2, 5)
    origin = frame_origin(features, constants)
    # The explicit empty channel survives the ID-masked data view. Do not
    # derive occupancy from channels that the ablation intentionally removes.
    occupied = 1.0 - nodes[..., 0]
    ids = nodes[..., 1:].reshape(*features.shape[:-1], 4, 8)
    if not include_object_ids:
        ids = torch.zeros_like(ids)
    slot_tokens = torch.cat((slots, slots - origin[..., None, :2] / constants[7],
        features[..., 14:18, None], features[..., 18:22, None], occupied, ids), dim=-1)

    geometry = Geometry(*constants[:5].detach().cpu().tolist())
    q = torch.stack((features[..., :4], features[..., PREVIOUS_TARGET_SLICE]), dim=-2) * 180.0
    xyz, axes = nominal_forward(q, geometry)
    current, previous = xyz.unbind(-2)
    current_axis, previous_axis = axes.unbind(-2)
    source_xy = (slots * features[..., 14:18, None]).sum(-2) * constants[7]
    source_z = features[..., 23:24] * constants[5] + constants[6]
    source = torch.cat((source_xy, source_z), dim=-1)
    moved, active = features[..., 26:30], features[..., 70:74]
    if not include_object_ids:
        moved, active = torch.zeros_like(moved), torch.zeros_like(active)
    public = torch.cat((features[..., :14], features[..., 82:92], features[..., 22:26], moved, active,
        current / constants[7], previous / constants[7], (current - origin) / constants[7],
        (previous - origin) / constants[7], (source - origin) / constants[7], origin / constants[7],
        current_axis, previous_axis, (q[..., 0, 1:].sum(-1, keepdim=True) + 90.0) / 180.0,
        (q[..., 1, 1:].sum(-1, keepdim=True) + 90.0) / 180.0), dim=-1)
    if (slot_tokens.shape[-1] != SLOT_FEATURE_COUNT or public.shape[-1] != GLOBAL_FEATURE_COUNT
            or not bool(torch.isfinite(slot_tokens).all() and torch.isfinite(public).all())):
        raise ValueError("public inputs overflow task-frame features")
    return slot_tokens, public


def _validate_rows(present, *, include_object_ids=True):
    if not bool(torch.isfinite(present).all()):
        raise ValueError("valid public inputs must be finite")
    for section in (slice(14, 18), slice(18, 22), slice(22, 24), slice(24, 26), PREVIOUS_ACTION_SLICE):
        group = present[:, section]
        if bool(((group != 0) & (group != 1)).any()) or bool((group.sum(-1) != 1).any()):
            raise ValueError("categorical public inputs must be one-hot")
    if bool((present[:, 14:18] == present[:, 18:22]).all(-1).any()):
        raise ValueError("source and destination slots must differ")
    moved = present[:, 26:30]
    nodes = present[:, 30:70].reshape(-1, 4, 2, 5)
    if bool(((nodes != 0) & (nodes != 1)).any()) or bool(((moved != 0) & (moved != 1)).any()):
        raise ValueError("registered object channels must be binary")
    if include_object_ids:
        if bool((nodes.sum(-1) != 1).any()) or bool((moved.sum(-1) != 1).any()):
            raise ValueError("registered scene layers and moved ID must be one-hot")
    elif (bool((nodes.sum(-1) > 1).any()) or bool((moved.sum(-1) > 1).any())):
        raise ValueError("ID-masked scene channels must be empty or one-hot")
    flags = present[:, [*range(70, 74), TRANSFER_START_INDEX, REGISTRATION_START_INDEX]]
    if bool(((flags != 0) & (flags != 1)).any()):
        raise ValueError("public active-object and boundary flags must be binary")
    targets = present[:, PREVIOUS_TARGET_SLICE] * 180.0
    limits = targets.new_tensor(JOINT_LIMITS)
    if bool(((targets < limits[:, 0] - 1e-4) | (targets > limits[:, 1] + 1e-4)).any()):
        raise ValueError("previous applied target is outside joint limits")


class TaskFrameStudent(nn.Module):
    """Shared slot encoder, fresh causal LSTM96, action and Cartesian heads."""

    def __init__(self, include_object_ids=True):
        super().__init__()
        if type(include_object_ids) is not bool:
            raise ValueError("include_object_ids must be boolean")
        self.include_object_ids = include_object_ids
        self.slot_encoder = nn.Sequential(nn.Linear(SLOT_FEATURE_COUNT, SLOT_EMBEDDING_SIZE), nn.SiLU(),
                                          nn.Linear(SLOT_EMBEDDING_SIZE, SLOT_EMBEDDING_SIZE), nn.SiLU())
        self.encoder = nn.Sequential(nn.Linear(ENCODED_FEATURE_COUNT, HIDDEN_SIZE), nn.SiLU())
        self.lstm = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.action_head = nn.Linear(HIDDEN_SIZE, len(motion_policy.ACTIONS))
        self.pose_head = nn.Linear(HIDDEN_SIZE, 4)
        self.register_buffer("nominal_constants", torch.tensor(NOMINAL_CONSTANTS, dtype=torch.float32))

    def forward(self, features, valid_mask, state=None):
        parameter = self.action_head.weight
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or min(features.shape[:2]) < 1 or features.shape[-1] != FEATURE_COUNT
                or features.dtype != parameter.dtype or features.device != parameter.device):
            raise ValueError("features must be [batch,time,92] on the model dtype and device")
        batch, steps = features.shape[:2]
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (batch, steps)
                or valid_mask.dtype != torch.bool or valid_mask.device != features.device):
            raise ValueError("validity mask must be boolean [batch,time] on the feature device")
        _validate_rows(features[valid_mask], include_object_ids=self.include_object_ids)
        if state is None:
            first = valid_mask.to(torch.int64).argmax(-1)
            boundary = features[torch.arange(batch, device=features.device), first, REGISTRATION_START_INDEX]
            if bool((valid_mask.any(-1) & (boundary != 1)).any()):
                raise ValueError("a sequence without state must begin at registration")
            hidden = cell = features.new_zeros((1, batch, HIDDEN_SIZE))
        else:
            if (not isinstance(state, tuple) or len(state) != 2 or any(
                    not isinstance(value, torch.Tensor) or value.shape != (1, batch, HIDDEN_SIZE)
                    or value.dtype != features.dtype or value.device != features.device
                    or not bool(torch.isfinite(value).all()) for value in state)):
                raise ValueError("state must be finite hidden/cell [1,batch,96] tensors")
            hidden, cell = state

        # Use a real current row to sanitize padding before geometric validation.
        # A completely padded batch contributes no output, state update or loss.
        clean = features.new_zeros(features.shape)
        if bool(valid_mask.any()):
            clean[:] = features[valid_mask][0]
        else:
            clean[..., 14] = clean[..., 19] = clean[..., 22] = clean[..., 24] = 1.0
            clean[..., 26] = clean[..., 70] = clean[..., 82] = 1.0
            clean[..., 30:70:5] = 1.0
            clean[..., 86:90] = clean.new_tensor([0., -10., -110., 30.]) / 180.0
        clean = torch.where(valid_mask[..., None], features, clean)
        tokens, public = public_features(clean, include_object_ids=self.include_object_ids,
                                         constants=self.nominal_constants)
        slots = self.slot_encoder(tokens)
        source = (slots * clean[..., 14:18, None]).sum(-2)
        destination = (slots * clean[..., 18:22, None]).sum(-2)
        encoded = self.encoder(torch.cat((slots.mean(-2), source, destination, public), dim=-1))
        logits, poses = [], []
        for step in range(steps):
            valid = valid_mask[:, step]
            reset = valid & (clean[:, step, REGISTRATION_START_INDEX] == 1)
            before_h = torch.where(reset[None, :, None], torch.zeros_like(hidden), hidden)
            before_c = torch.where(reset[None, :, None], torch.zeros_like(cell), cell)
            _, (next_h, next_c) = self.lstm(encoded[:, step:step + 1], (before_h, before_c))
            hidden = torch.where(valid[None, :, None], next_h, hidden)
            cell = torch.where(valid[None, :, None], next_c, cell)
            action, pose = self.action_head(hidden[0]), self.pose_head(hidden[0])
            logits.append(torch.where(valid[:, None], action, torch.zeros_like(action)))
            poses.append(torch.where(valid[:, None], pose, torch.zeros_like(pose)))
        return TaskFrameOutputs(torch.stack(logits, 1), torch.stack(poses, 1), (hidden, cell))


def model_contract(include_object_ids=True):
    if type(include_object_ids) is not bool:
        raise ValueError("include_object_ids must be boolean")
    return {"model_format": FORMAT, "constructor": "TaskFrameStudent(include_object_ids=...)",
        "feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE, "cell": "LSTM",
        "slot_feature_count": SLOT_FEATURE_COUNT, "slot_embedding_size": SLOT_EMBEDDING_SIZE,
        "global_feature_count": GLOBAL_FEATURE_COUNT, "include_object_ids": include_object_ids,
        "actions": list(motion_policy.ACTIONS),
        "move_output": "destination_nominal_grasp_relative_world_xyz_mm/300_and_unwrapped_pitch_deg/180",
        "nominal_constants": list(NOMINAL_CONSTANTS),
        "inverse": "analytic_exact_xyz_unwrapped_pitch_both_signed_radius_branches",
        "inverse_branch_selection": "nearest_confirmed_applied_joints_else_positive_radius_first",
        "inverse_zero_radius": "reject_below_1e-6_mm"}


def load_checkpoint(payload):
    """Validate model/config/weight binding; source provenance is checked outside."""
    from .scene_applied_action_policy import _encoded, _state_digest
    contract = payload.get("model") if isinstance(payload, dict) else None
    flag = contract.get("include_object_ids") if isinstance(contract, dict) else None
    if (not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or type(flag) is not bool
            or contract != model_contract(flag) or not isinstance(payload.get("config"), dict)
            or payload.get("config_sha256") != hashlib.sha256(_encoded(payload["config"])).hexdigest()
            or not isinstance(payload.get("provenance"), dict) or not isinstance(payload.get("training"), dict)
            or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("unsupported task-frame student checkpoint contract")
    with torch.random.fork_rng(devices=[]):
        model = TaskFrameStudent(flag).cpu()
    expected, state = model.state_dict(), payload["state_dict"]
    if (set(expected) != set(state) or any(not isinstance(value, torch.Tensor)
            or value.device.type != "cpu" or value.dtype != expected[name].dtype or value.shape != expected[name].shape
            or not bool(torch.isfinite(value).all()) for name, value in state.items())
            or any(not torch.equal(state[name], value) for name, value in model.named_buffers())):
        raise ValueError("task-frame student tensors or nominal geometry differ")
    model.load_state_dict(state, strict=True)
    if _state_digest(model) != payload["training"].get("final_state_sha256"):
        raise ValueError("task-frame student weights fingerprint differs")
    return model.eval()
