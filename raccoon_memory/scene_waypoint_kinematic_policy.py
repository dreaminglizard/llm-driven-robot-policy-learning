"""One learned action policy with a nominal-geometry waypoint output basis.

The policy chooses its own action and waypoint.  The basis uses only the
registered slot coordinates and nominal arm geometry already in the actor
input.  It does not invoke a teacher, inspect physical object state, or switch
between policies.  Its output is still subject to the normal control guard.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import NamedTuple

import torch
from torch import nn

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from raccoon_memory import motion_policy, scene_unified_kinematic_policy as unified

MAX_WAYPOINTS = 64
MAX_CORRECTION_DEG = 1.25
CONTRACT = {"format": "raccoon-nominal-waypoint-single-policy.v1",
            "window": [8, 82], "previous_target": "confirmed_applied_joint_degrees",
            "geometry": "registered_nominal_slots_and_nominal_arm_only",
            "actions": list(motion_policy.ACTIONS), "maximum_waypoints": MAX_WAYPOINTS,
            "maximum_correction_degrees_l2": MAX_CORRECTION_DEG,
            "runtime_teacher": False, "runtime_policy_switch": False}


def _one_hot(values):
    if any(v not in (0., 1.) for v in values) or sum(values) != 1.:
        raise ValueError("nominal task features must be one-hot")
    return values.index(1.)


@lru_cache(maxsize=2048)
def _nominal_basis(key):
    if len(key) != 20 or any(not math.isfinite(v) for v in key):
        raise ValueError("nominal geometry must be finite")
    src, dst = _one_hot(key[:4]), _one_hot(key[4:8])
    sl, dl = _one_hot(key[8:10]), _one_hot(key[10:12])
    if src == dst:
        raise ValueError("source and destination must differ")
    xy = [(key[12 + 2*i] * 300., key[13 + 2*i] * 300.) for i in range(4)]
    geometry = Geometry()

    def point(slot, height):
        return list(geometry.inverse((*xy[slot], height)))

    source = [point(src, 100.), point(src, 40. + 20.*sl), point(src, 15. + 20.*sl)]
    lift, clear, above = point(src, 60. + 20.*sl), point(src, 130.), point(dst, 130.)
    count = max(1, math.ceil(abs(above[0] - clear[0]) / 10.))
    route = [[clear[0] + (above[0] - clear[0])*i/count, *clear[1:]] for i in range(1, count+1)]
    source_radius, destination_radius = math.hypot(*xy[src]), math.hypot(*xy[dst])
    if destination_radius <= 0:
        raise ValueError("destination radius must be positive")
    count = max(1, math.ceil(abs(source_radius - destination_radius) / 15.))
    for i in range(1, count + 1):
        radius = source_radius + (destination_radius - source_radius) * i / count
        target = list(geometry.inverse((xy[dst][0]*radius/destination_radius,
                                       xy[dst][1]*radius/destination_radius, 130.)))
        if math.dist(route[-1], target) > 1e-5:
            route.append(target)
    destination = [point(dst, max(60., 35. + 20.*dl)), point(dst, 15. + 20.*dl)]
    basis = source + [lift, clear, *route, *destination, point(dst, 100.)]
    if not 1 <= len(basis) <= MAX_WAYPOINTS:
        raise ValueError("nominal geometry has too many waypoint proposals")
    return tuple(tuple(q / motion_policy.JOINT_SCALE_DEG for q in point) for point in basis)


def nominal_basis(windows):
    """Padded, normalized candidate joints and a mask, from actor inputs."""
    if (not isinstance(windows, torch.Tensor) or windows.ndim != 3 or windows.shape[1:] != (8, 82)
            or windows.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(windows).all())):
        raise ValueError("finite floating [batch,8,82] windows required")
    rows = windows[:, -1, :].detach().cpu()
    keys = torch.cat((rows[:, 14:26], rows[:, 74:82]), dim=1).tolist()
    unique = {tuple(key): _nominal_basis(tuple(key)) for key in keys}
    candidates = windows.new_zeros((len(keys), MAX_WAYPOINTS, 4))
    valid = torch.zeros((len(keys), MAX_WAYPOINTS), dtype=torch.bool, device=windows.device)
    for index, key in enumerate(keys):
        points = unique[tuple(key)]
        candidates[index, :len(points)] = windows.new_tensor(points)
        valid[index, :len(points)] = True
    return candidates, valid


class WaypointOutputs(NamedTuple):
    action_logits: torch.Tensor
    absolute_joints: torch.Tensor
    waypoint_logits: torch.Tensor
    correction_degrees: torch.Tensor
    selected_waypoint: torch.Tensor


class NominalWaypointPolicy(unified.UnifiedKinematicPolicy):
    """All actions and waypoint choices come from a single learned model."""

    def __init__(self):
        super().__init__()
        width = unified.HIDDEN_SIZE + 22
        self.waypoint_head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, MAX_WAYPOINTS))
        self.residual_head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.residual_head[2].weight)
        nn.init.zeros_(self.residual_head[2].bias)

    def waypoint_outputs(self, windows, masks, previous_target_degrees, basis=None):
        base = super().training_outputs(windows, masks, previous_target_degrees)
        relative, prior = self._context(windows, masks, previous_target_degrees)
        projected = self.base._project_validated(windows, masks)
        first = self.base.encoder[0](projected) + self.context_input(torch.cat((relative, prior), dim=1))
        encoded = self.base.encoder[3](self.base.encoder[2](self.base.encoder[1](first)))
        row = windows[:, -1, :]
        features = torch.cat((encoded, relative, prior, row[:, :4], row[:, 6:14]), dim=1)
        candidates, valid = nominal_basis(windows) if basis is None else basis
        if (candidates.shape != (len(windows), MAX_WAYPOINTS, 4) or
                valid.shape != (len(windows), MAX_WAYPOINTS) or valid.dtype != torch.bool or
                candidates.dtype != windows.dtype or candidates.device != windows.device or
                valid.device != windows.device or not bool(valid.any(1).all()) or
                not bool(torch.isfinite(candidates).all())):
            raise ValueError("nominal waypoint basis differs from the model input")
        logits = self.waypoint_head(features).masked_fill(~valid, -1e9)
        correction = torch.tanh(self.residual_head(features)) * MAX_CORRECTION_DEG
        norm = correction.norm(dim=1, keepdim=True).clamp(min=1e-12)
        correction = correction * (MAX_CORRECTION_DEG / norm).clamp(max=1.)
        selected = logits.argmax(dim=1)
        rows = torch.arange(len(windows), device=windows.device)
        proposal = candidates[rows, selected] + correction / motion_policy.JOINT_SCALE_DEG
        limits = windows.new_tensor(JOINT_LIMITS) / motion_policy.JOINT_SCALE_DEG
        proposal = torch.maximum(limits[:, 0], torch.minimum(limits[:, 1], proposal))
        target = torch.where((base.action_logits.argmax(dim=1) == motion_policy.MOVE).unsqueeze(1),
                             proposal, base.absolute_joints)
        return WaypointOutputs(base.action_logits, target, logits, correction, selected)

    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        out = self.waypoint_outputs(windows, masks, previous_target_degrees)
        return out.action_logits, out.absolute_joints


def initialize_from_unified(source):
    if type(source) is not unified.UnifiedKinematicPolicy or source.joint_branch is not None:
        raise ValueError("a branch-free unified policy is required")
    with torch.random.fork_rng(devices=[]):
        result = NominalWaypointPolicy().to(dtype=next(source.parameters()).dtype,
                                            device=next(source.parameters()).device)
    missing, unexpected = result.load_state_dict(source.state_dict(), strict=False)
    if unexpected or set(missing) != {f"{head}.{layer}.{field}" for head in ("waypoint_head", "residual_head")
                                      for layer in (0, 2) for field in ("weight", "bias")}:
        raise ValueError("source tensors differ")
    return result


def checkpoint_payload(model):
    if type(model) is not NominalWaypointPolicy or any(not p.requires_grad for p in model.parameters()):
        raise ValueError("a fully trainable nominal-waypoint policy is required")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("nonfinite model parameters")
    return {"contract": CONTRACT, "state_dict": state}


def load_checkpoint(payload, *, device="cpu"):
    if not isinstance(payload, dict) or payload.get("contract") != CONTRACT:
        raise ValueError("nominal-waypoint checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "base.encoder.0.weight" not in state:
        raise ValueError("model state is required")
    weight = state["base.encoder.0.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu" or
            weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("model state must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]):
        model = NominalWaypointPolicy().to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected) or any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != weight.dtype or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())
            for name, value in state.items()):
        raise ValueError("model tensor names, shapes, dtype or values differ")
    model.load_state_dict(state, strict=True)
    return model.to(device=device).eval()
