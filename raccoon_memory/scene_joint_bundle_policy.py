"""MLP64 with fixed joint-target centers, a selector, and per-center residuals.

The external input is the same causal 82-feature, eight-observation window as
ScenePolicy. The policy has no teacher inputs: forward selects the first
maximum-scoring center and returns its center plus its residual, in units of
degrees / 180. It never clamps that prediction or substitutes another center.

The separate training loss assigns each supervised move target to its nearest
fixed center using squared Euclidean distance (first index on ties). Non-move
joint labels are not read. Centers must be prepared and audited from training
targets by the caller; this module does not fit centers or load a dataset.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, scene_policy


HISTORY_LENGTH = 8
FEATURE_COUNT = 82
HIDDEN_SIZE = 64
BUNDLE_COUNT = 4
JOINT_COUNT = 4
CHECKPOINT_FORMAT = "raccoon-scene-joint-bundle-residual.v1"


def _centers_tensor(values, *, dtype=None, device=None) -> torch.Tensor:
    """Copy four finite centers, already normalized by 180 degrees."""
    try:
        tensor = torch.as_tensor(values, dtype=dtype, device=device).detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("centers must be a finite floating 4x4 array") from exc
    if (tensor.shape != (BUNDLE_COUNT, JOINT_COUNT) or not tensor.is_floating_point()
            or tensor.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(tensor).all())):
        raise ValueError("centers must be a finite float32/float64 4x4 array")
    low = tensor.new_tensor([bounds[0] / 180.0 for bounds in JOINT_LIMITS])
    high = tensor.new_tensor([bounds[1] / 180.0 for bounds in JOINT_LIMITS])
    if bool(((tensor < low) | (tensor > high)).any()):
        raise ValueError("normalized center is outside the joint limits")
    return tensor


class BundleLoss(NamedTuple):
    total: torch.Tensor
    action_ce: torch.Tensor
    bundle_ce: torch.Tensor
    residual_mse: torch.Tensor
    move_count: int


class SceneJointBundlePolicy(nn.Module):
    """Predict action, four bundle scores, and four four-joint residuals."""

    def __init__(self, centers):
        super().__init__()
        self.architecture = "joint_bundle_mlp"
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.register_buffer("centers", _centers_tensor(centers))
        self.encoder = nn.Sequential(nn.Linear(HISTORY_LENGTH * (FEATURE_COUNT + 1), HIDDEN_SIZE), nn.ReLU(),
                                     nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU())
        self.action_head = nn.Linear(HIDDEN_SIZE, len(motion_policy.ACTIONS))
        self.bundle_head = nn.Linear(HIDDEN_SIZE, BUNDLE_COUNT)
        self.residual_head = nn.Linear(HIDDEN_SIZE, BUNDLE_COUNT * JOINT_COUNT)
        if sum(parameter.numel() for parameter in self.parameters()) > scene_policy.MAX_PARAMETERS:
            raise ValueError("parameter limit exceeded")
        self.to(device=self.centers.device, dtype=self.centers.dtype)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor):
        """Return (action logits [N,4], bundle logits [N,4], residuals [N,4,4])."""
        # Reuse the exact shape, finite-value, and left-padding validation.
        scene_policy.ScenePolicy._validate(self, windows, masks)
        masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
        values = torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1)
        encoded = self.encoder(values)
        action_logits = self.action_head(encoded)
        bundle_logits = self.bundle_head(encoded)
        residuals = self.residual_head(encoded).reshape(-1, BUNDLE_COUNT, JOINT_COUNT)
        if not all(bool(torch.isfinite(x).all()) for x in (action_logits, bundle_logits, residuals)):
            raise ValueError("policy produced non-finite output")
        return action_logits, bundle_logits, residuals

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        """Return existing evaluator outputs: action logits and absolute joint targets."""
        action_logits, bundle_logits, residuals = self.training_outputs(windows, masks)
        chosen = bundle_logits.argmax(dim=1)  # PyTorch chooses the first index for ties.
        indices = torch.arange(chosen.shape[0], device=chosen.device)
        return action_logits, self.centers[chosen] + residuals[indices, chosen]

    def supervised_loss(self, outputs, actions: torch.Tensor, joint_targets: torch.Tensor) -> BundleLoss:
        """CE(action) + CE(nearest move center) + 100*MSE(move residual).

        Only the move rows of ``joint_targets`` are accessed; entries for
        open, close, or done can be missing/NaN placeholders. The centers and
        target assignments are fixed and do not receive gradients.
        """
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("outputs must be the tuple from training_outputs")
        action_logits, bundle_logits, residuals = outputs
        if (not isinstance(action_logits, torch.Tensor) or action_logits.ndim != 2
                or action_logits.shape[1] != 4 or action_logits.shape[0] < 1):
            raise ValueError("action logits must have shape (nonempty batch, 4)")
        count = action_logits.shape[0]
        if (not isinstance(bundle_logits, torch.Tensor) or bundle_logits.shape != (count, 4)
                or not isinstance(residuals, torch.Tensor) or residuals.shape != (count, 4, 4)
                or any(not x.is_floating_point() or x.device != self.centers.device
                       or x.dtype != self.centers.dtype or not bool(torch.isfinite(x).all())
                       for x in (action_logits, bundle_logits, residuals))):
            raise ValueError("training outputs have incompatible shape, dtype, device or finite values")
        if (not isinstance(actions, torch.Tensor) or actions.dtype != torch.long
                or actions.shape != (count,) or actions.device != self.centers.device
                or bool(((actions < 0) | (actions >= len(motion_policy.ACTIONS))).any())):
            raise ValueError("actions must be valid integer action indices")
        if (not isinstance(joint_targets, torch.Tensor) or joint_targets.shape != (count, 4)
                or not joint_targets.is_floating_point() or joint_targets.dtype != self.centers.dtype
                or joint_targets.device != self.centers.device):
            raise ValueError("joint targets must have shape (batch, 4) and match policy dtype/device")
        if self.centers.requires_grad:
            raise ValueError("centers must remain fixed")
        action_ce = F.cross_entropy(action_logits, actions)
        move = actions == motion_policy.MOVE
        move_count = int(move.sum().item())
        if move_count:
            targets = joint_targets[move].detach()
            if not bool(torch.isfinite(targets).all()):
                raise ValueError("move joint targets must be finite")
            low = targets.new_tensor([bounds[0] / 180.0 for bounds in JOINT_LIMITS])
            high = targets.new_tensor([bounds[1] / 180.0 for bounds in JOINT_LIMITS])
            if bool(((targets < low) | (targets > high)).any()):
                raise ValueError("normalized move target is outside the joint limits")
            nearest = ((targets[:, None, :] - self.centers[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
            target_residual = targets - self.centers[nearest]
            move_residuals = residuals[move]
            rows = torch.arange(move_count, device=nearest.device)
            bundle_ce = F.cross_entropy(bundle_logits[move], nearest)
            residual_mse = F.mse_loss(move_residuals[rows, nearest], target_residual)
        else:
            # Keep the outputs in the graph with zero gradient, without reading
            # non-move target placeholders (which need not even be finite).
            bundle_ce = bundle_logits.sum() * 0.0
            residual_mse = residuals.sum() * 0.0
        return BundleLoss(action_ce + bundle_ce + 100.0 * residual_mse,
                          action_ce, bundle_ce, residual_mse, move_count)


def initialize_from_scene_policy(source: scene_policy.ScenePolicy, centers) -> SceneJointBundlePolicy:
    """Independently copy an MLP64; initialize each residual as old output-center.

    Bundle scores start at zero. No optimizer is copied, and source tensors,
    supplied centers, and the caller's global Torch random stream are unchanged.
    """
    if (type(source) is not scene_policy.ScenePolicy
            or (source.architecture, source.hidden_size, source.history_length, source.feature_count)
            != ("mlp", HIDDEN_SIZE, HISTORY_LENGTH, FEATURE_COUNT)):
        raise ValueError("source must be the existing 82-feature history-8 MLP64 ScenePolicy")
    state = source.state_dict()
    shapes = {"encoder.0.weight": (64, 664), "encoder.0.bias": (64,),
              "encoder.2.weight": (64, 64), "encoder.2.bias": (64,),
              "action_head.weight": (4, 64), "action_head.bias": (4,),
              "joint_head.weight": (4, 64), "joint_head.bias": (4,)}
    if set(state) != set(shapes):
        raise ValueError("source parameter names differ from MLP64")
    for name, shape in shapes.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype not in (torch.float32, torch.float64) or value.device.type == "meta"
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"source parameter is invalid: {name}")
    devices, dtypes = {v.device for v in state.values()}, {v.dtype for v in state.values()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("source parameters must share device and dtype")
    device, dtype = next(iter(devices)), next(iter(dtypes))
    center_copy = _centers_tensor(centers, dtype=dtype, device=device)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        target = SceneJointBundlePolicy(center_copy.cpu())
    target = target.to(device=device, dtype=dtype)
    with torch.no_grad():
        target.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
        target.action_head.load_state_dict(source.action_head.state_dict(), strict=True)
        target.bundle_head.weight.zero_()
        target.bundle_head.bias.zero_()
        target.residual_head.weight.reshape(4, 4, 64).copy_(state["joint_head.weight"].unsqueeze(0).expand(4, -1, -1))
        target.residual_head.bias.reshape(4, 4).copy_(state["joint_head.bias"].unsqueeze(0) - target.centers)
    target.train(source.training)
    return target


def checkpoint_contract() -> dict:
    """Return the distinct policy contract; caller records data and model provenance."""
    return {"format": CHECKPOINT_FORMAT, "architecture": "joint_bundle_mlp",
            "feature_schema": scene_features.SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
            "actions": list(motion_policy.ACTIONS), "history_length": 8, "hidden_size": 64,
            "bundle_count": 4, "joint_count": 4, "joint_scale_degrees": 180.0,
            "center_assignment": "nearest_squared_distance_first_index",
            "inference_selection": "argmax_first_index_no_clamp_no_fallback"}


def checkpoint_payload(model: SceneJointBundlePolicy) -> dict:
    """Copy model tensors to CPU; caller adds provenance and optimizer separately."""
    if type(model) is not SceneJointBundlePolicy:
        raise ValueError("joint bundle policy required")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    load_checkpoint(payload)  # Validate before returning a publishable model payload.
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneJointBundlePolicy:
    """Restore this format only, verifying center and parameter shape/dtype/value."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("joint bundle checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "centers" not in state:
        raise ValueError("joint bundle checkpoint requires centers and state dictionary")
    center = state["centers"]
    if not isinstance(center, torch.Tensor) or center.device.type != "cpu":
        raise ValueError("checkpoint centers must be a CPU tensor")
    _centers_tensor(center)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneJointBundlePolicy(center).to(dtype=center.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != center.dtype or value.shape != expected[name].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.eval()
    return model
