"""Frozen scene MLP with a trainable correction for short history MOVE outputs.

Inference accepts only the existing observation window and mask. Its gate uses
the frozen policy's predicted action; teacher actions appear only in the
separate supervised-loss method. Joint values use the existing degrees / 180
units, and neither that method nor inference clips or replaces a prediction.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_memory import motion_policy, scene_features, scene_policy


HISTORY_LENGTH = 8
FEATURE_COUNT = 82
HIDDEN_SIZE = 64
JOINT_COUNT = 4
MAX_CORRECTED_HISTORY = 3
CHECKPOINT_FORMAT = "raccoon-scene-early-history-correction.v1"


class CorrectionTrainingOutputs(NamedTuple):
    action_logits: torch.Tensor
    base_joints: torch.Tensor
    correction: torch.Tensor
    history_lengths: torch.Tensor


class CorrectionLoss(NamedTuple):
    total: torch.Tensor
    joint_mse: torch.Tensor
    supervised_count: int


class SceneEarlyHistoryCorrectionPolicy(nn.Module):
    """Keep an independent frozen MLP64 and train only a 64-to-4 linear head."""

    def __init__(self):
        super().__init__()
        self.architecture = "early_history_correction_mlp"
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.base = scene_policy.ScenePolicy(architecture="mlp", hidden_size=HIDDEN_SIZE,
                                             history_length=HISTORY_LENGTH)
        self.base.requires_grad_(False)
        self.correction_head = nn.Linear(HIDDEN_SIZE, JOINT_COUNT)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> CorrectionTrainingOutputs:
        """Return ungated correction for teacher-move supervision, without teacher inputs."""
        with torch.no_grad():
            lengths = self.base._validate(windows, masks)
            masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
            values = torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1)
            encoded = self.base.encoder(values)
            logits = self.base.action_head(encoded)
            joints = self.base.joint_head(encoded)
        return CorrectionTrainingOutputs(logits, joints, self.correction_head(encoded), lengths)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Correct only H<=3 with predicted MOVE; return ordinary evaluator outputs.

        Nonfinite model outputs are returned to the existing evaluator so they
        are classified as model failures. A nonfinite unused correction does
        not alter the frozen output for a non-MOVE or long-history row.
        """
        out = self.training_outputs(windows, masks)
        applies = (out.history_lengths <= MAX_CORRECTED_HISTORY) & (out.action_logits.argmax(dim=1) == motion_policy.MOVE)
        joints = torch.where(applies.unsqueeze(1), out.base_joints + out.correction, out.base_joints)
        return out.action_logits, joints

    def supervised_loss(self, outputs: CorrectionTrainingOutputs, actions: torch.Tensor,
                        joint_targets: torch.Tensor) -> CorrectionLoss:
        """100*MSE on teacher MOVE rows with H<=3, regardless of predicted action.

        Targets are normalized by 180 exactly once before this call. Non-MOVE
        and H>=4 target placeholders are never read. If there are no eligible
        rows, the returned count and loss are zero; the caller must skip the
        optimizer step for that batch to avoid updating Adam's momentum.
        """
        if not isinstance(outputs, CorrectionTrainingOutputs):
            raise ValueError("outputs must come from training_outputs")
        logits, base, correction, lengths = outputs
        weight = self.correction_head.weight
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 4
                or logits.shape[0] == 0):
            raise ValueError("action logits must have shape (nonempty batch, 4)")
        n = logits.shape[0]
        if (not isinstance(base, torch.Tensor) or not isinstance(correction, torch.Tensor)
                or base.shape != (n, 4) or correction.shape != (n, 4)
                or any(not x.is_floating_point() or x.dtype != weight.dtype or x.device != weight.device
                       for x in (logits, base, correction))):
            raise ValueError("outputs must match the policy's shape, dtype, and device")
        if (not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long
                or lengths.shape != (n,) or lengths.device != weight.device
                or bool(((lengths < 1) | (lengths > HISTORY_LENGTH)).any())):
            raise ValueError("history lengths must be integers from 1 to 8")
        if (not isinstance(actions, torch.Tensor) or actions.dtype != torch.long
                or actions.shape != (n,) or actions.device != weight.device
                or bool(((actions < 0) | (actions >= len(motion_policy.ACTIONS))).any())):
            raise ValueError("actions must contain valid teacher action indices")
        if (not isinstance(joint_targets, torch.Tensor) or joint_targets.shape != (n, 4)
                or joint_targets.dtype != weight.dtype or joint_targets.device != weight.device):
            raise ValueError("joint targets must match the policy dtype and have shape (batch, 4)")
        if any(p.requires_grad for p in self.base.parameters()):
            raise ValueError("base policy must remain frozen")
        selected = (lengths <= MAX_CORRECTED_HISTORY) & (actions == motion_policy.MOVE)
        count = int(selected.sum().item())
        if not count:
            zero = self.correction_head.weight.sum() * 0.0 + self.correction_head.bias.sum() * 0.0
            return CorrectionLoss(zero, zero, 0)
        prediction = base[selected] + correction[selected]
        target = joint_targets[selected].detach()
        if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
            raise ValueError("supervised joint targets and predictions must be finite")
        mse = F.mse_loss(prediction, target)
        return CorrectionLoss(100.0 * mse, mse, count)


def _validate_source(source: scene_policy.ScenePolicy):
    if (type(source) is not scene_policy.ScenePolicy
            or (source.architecture, source.hidden_size, source.history_length, source.feature_count)
            != ("mlp", HIDDEN_SIZE, HISTORY_LENGTH, FEATURE_COUNT)):
        raise ValueError("source must be the existing 82-feature history-8 MLP64 ScenePolicy")
    shapes = {"encoder.0.weight": (64, 664), "encoder.0.bias": (64,),
              "encoder.2.weight": (64, 64), "encoder.2.bias": (64,),
              "action_head.weight": (4, 64), "action_head.bias": (4,),
              "joint_head.weight": (4, 64), "joint_head.bias": (4,)}
    state = source.state_dict()
    if set(state) != set(shapes):
        raise ValueError("source parameter names differ from MLP64")
    for name, shape in shapes.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype not in (torch.float32, torch.float64) or tensor.device.type == "meta"
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"source parameter is invalid: {name}")
    devices = {t.device for t in state.values()}
    dtypes = {t.dtype for t in state.values()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("source parameters must share device and dtype")
    return next(iter(devices)), next(iter(dtypes))


def initialize_from_scene_policy(source: scene_policy.ScenePolicy) -> SceneEarlyHistoryCorrectionPolicy:
    """Copy the baseline independently, freeze it, and zero the 260 new parameters."""
    device, dtype = _validate_source(source)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        result = SceneEarlyHistoryCorrectionPolicy()
    result = result.to(device=device, dtype=dtype)
    result.base.load_state_dict(source.state_dict(), strict=True)
    result.base.requires_grad_(False)
    result.train(source.training)
    return result


def checkpoint_contract() -> dict:
    """Distinct format: callers bind baseline, training data and provenance separately."""
    return {"format": CHECKPOINT_FORMAT, "architecture": "early_history_correction_mlp",
            "feature_schema": scene_features.SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
            "actions": list(motion_policy.ACTIONS), "history_length": HISTORY_LENGTH,
            "feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE, "joint_count": JOINT_COUNT,
            "joint_scale_degrees": 180.0, "max_corrected_history": MAX_CORRECTED_HISTORY,
            "inference_gate": "history_1_to_3_and_predicted_move",
            "supervision_gate": "history_1_to_3_and_teacher_move",
            "output": "absolute_base_plus_correction_no_clipping"}


def checkpoint_payload(model: SceneEarlyHistoryCorrectionPolicy) -> dict:
    """Make independent CPU tensors; caller appends training provenance and optimizer."""
    if type(model) is not SceneEarlyHistoryCorrectionPolicy:
        raise ValueError("early history correction policy required")
    if any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("base policy must remain frozen")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneEarlyHistoryCorrectionPolicy:
    """Restore this distinct checkpoint contract and reestablish the frozen base."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("early history correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "correction_head.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["correction_head.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneEarlyHistoryCorrectionPolicy().to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != weight.dtype or value.shape != expected[name].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.base.requires_grad_(False)
    model.eval()
    return model
