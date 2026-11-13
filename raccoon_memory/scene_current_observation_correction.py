"""Early-history correction using frozen encoding plus current observation.

The ordinary 82-feature window and mask remain the only inference inputs. A
fully validated left-padded window always has a valid last row. Its moving
column is omitted, and the other 81 values follow the frozen 64-dimensional
no-moving representation in the correction head. No teacher action, phase,
privileged simulator state, or future observation is accepted by inference.

The source 64-to-4 correction and its frozen core are independently copied;
the new 81 columns start at zero. The inherited inference and supervision
gates keep the original early-history semantics, and no target is clipped.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_memory import motion_policy, scene_early_history_correction as early
from raccoon_memory import scene_features, scene_no_moving_policy as no_moving


HISTORY_LENGTH = 8
FEATURE_COUNT = 82
HIDDEN_SIZE = 64
MOVING_INDEX = 5
CURRENT_COLUMNS = tuple(index for index in range(FEATURE_COUNT) if index != MOVING_INDEX)
CURRENT_FEATURE_COUNT = len(CURRENT_COLUMNS)
HEAD_FEATURE_COUNT = HIDDEN_SIZE + CURRENT_FEATURE_COUNT
CHECKPOINT_FORMAT = "raccoon-scene-current-observation-early-correction.v1"


def _check_contract() -> None:
    no_moving._check_contract()
    if (scene_features.HISTORY_LENGTH != HISTORY_LENGTH
            or len(scene_features.FEATURE_NAMES) != FEATURE_COUNT
            or scene_features.FEATURE_NAMES[MOVING_INDEX] != "moving"
            or len(CURRENT_COLUMNS) != 81 or HEAD_FEATURE_COUNT != 145
            or tuple(motion_policy.ACTIONS) != ("move", "open", "close", "done")):
        raise ValueError("scene feature or action contract differs")


class SceneCurrentObservationCorrectionPolicy(early.SceneEarlyHistoryCorrectionPolicy):
    """A 145-to-4 head on an independent frozen no-moving policy."""

    def __init__(self):
        # Avoid constructing the ordinary source class's 664-input encoder.
        nn.Module.__init__(self)
        _check_contract()
        self.architecture = "current_observation_early_correction_mlp"
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.base = no_moving.SceneNoMovingPolicy()
        self.base.requires_grad_(False)
        self.register_buffer("_current_columns", torch.tensor(CURRENT_COLUMNS, dtype=torch.long), persistent=False)
        self.correction_head = nn.Linear(HEAD_FEATURE_COUNT, 4)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def _frozen_outputs(self, windows: torch.Tensor, masks: torch.Tensor):
        with torch.no_grad():
            # The source validator checks shape, dtype, all finite values,
            # left padding, and a valid final observation before we index it.
            encoded, lengths = self.base._encode_with_lengths(windows, masks)
            current = windows[:, -1, :].index_select(1, self._current_columns)
            features = torch.cat((encoded, current), dim=1)
            logits = self.base.action_head(encoded)
            joints = self.base.joint_head(encoded)
        return features, logits, joints, lengths

    def head_features(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return [batch,145]: frozen 64-vector followed by current 81 features."""
        features, _, _, _ = self._frozen_outputs(windows, masks)
        return features

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> early.CorrectionTrainingOutputs:
        """Return base outputs and an ungated correction for teacher-MOVE fitting."""
        features, logits, joints, lengths = self._frozen_outputs(windows, masks)
        return early.CorrectionTrainingOutputs(logits, joints, self.correction_head(features), lengths)


def initialize_from_correction(source: no_moving.SceneNoMovingCorrectionPolicy) -> SceneCurrentObservationCorrectionPolicy:
    """Copy the source core and old head; initialize the 81 new columns to zero.

    No source tensor is modified or shared, no optimizer is copied, and model
    construction preserves the caller's CPU random state.
    """
    _check_contract()
    if type(source) is not no_moving.SceneNoMovingCorrectionPolicy:
        raise ValueError("source must be the existing no-moving early-history correction")
    payload = no_moving.checkpoint_payload(source)
    state = payload["state_dict"]
    param = source.correction_head.weight
    if tuple(param.shape) != (4, HIDDEN_SIZE):
        raise ValueError("source correction head must have 64 inputs")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneCurrentObservationCorrectionPolicy()
    model = model.to(device=param.device, dtype=param.dtype)
    base_state = {name.removeprefix("base."): value for name, value in state.items() if name.startswith("base.")}
    model.base.load_state_dict(base_state, strict=True)
    model.base.requires_grad_(False)
    with torch.no_grad():
        model.correction_head.weight[:, :HIDDEN_SIZE].copy_(state["correction_head.weight"])
        model.correction_head.weight[:, HIDDEN_SIZE:].zero_()
        model.correction_head.bias.copy_(state["correction_head.bias"])
    model.train(source.training)
    return model


def checkpoint_contract() -> dict:
    """Explicit format distinct from the 64-input correction checkpoint."""
    _check_contract()
    return {"format": CHECKPOINT_FORMAT, "architecture": "current_observation_early_correction_mlp",
            "base_contract": no_moving.checkpoint_contract(),
            "feature_schema": scene_features.SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
            "actions": list(motion_policy.ACTIONS), "history_length": HISTORY_LENGTH,
            "external_feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE,
            "current_feature_columns": list(CURRENT_COLUMNS), "current_feature_count": CURRENT_FEATURE_COUNT,
            "head_feature_count": HEAD_FEATURE_COUNT, "joint_scale_degrees": 180.0,
            "input": "frozen_encoding_64_then_last_valid_observation_without_moving_81",
            "max_corrected_history": early.MAX_CORRECTED_HISTORY,
            "inference_gate": "history_1_to_3_and_predicted_move",
            "supervision_gate": "history_1_to_3_and_teacher_move",
            "output": "absolute_base_plus_correction_no_clipping"}


def checkpoint_payload(model: SceneCurrentObservationCorrectionPolicy) -> dict:
    """Copy tensors to CPU; the caller records provenance and optimizer separately."""
    if type(model) is not SceneCurrentObservationCorrectionPolicy:
        raise ValueError("current-observation correction policy required")
    if any(p.requires_grad for p in model.base.parameters()):
        raise ValueError("correction base must remain frozen")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneCurrentObservationCorrectionPolicy:
    """Restore the explicit expanded-head format and reestablish its frozen core."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("current-observation correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "correction_head.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["correction_head.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneCurrentObservationCorrectionPolicy().to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != weight.dtype or tensor.shape != expected[name].shape
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.base.requires_grad_(False)
    model.eval()
    return model
