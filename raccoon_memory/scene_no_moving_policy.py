"""Scene MLP64 that omits ``moving`` inside the model, not from observations.

Inputs stay the original 82-feature observation windows and validity masks.
They are validated in full, including moving, before the fixed projection
removes that one feature from every masked frame. Nothing edits the input,
sensor logs, the upstream encoder, or the execution controller.

The independent early-history wrapper reuses the existing correction loss and
inference gate but obtains representations from this projected encoder. Its
head starts at zero and must be fitted for its own frozen base; a head fitted
to a different baseline is not implicitly reused.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_memory import motion_policy, scene_early_history_correction as early
from raccoon_memory import scene_features, scene_policy


HISTORY_LENGTH = 8
FEATURE_COUNT = 82
HIDDEN_SIZE = 64
MOVING_INDEX = 5
FRAME_WIDTH = FEATURE_COUNT + 1
FLAT_INPUT_COUNT = HISTORY_LENGTH * FRAME_WIDTH
OMITTED_COLUMNS = tuple(frame * FRAME_WIDTH + MOVING_INDEX for frame in range(HISTORY_LENGTH))
KEPT_COLUMNS = tuple(column for column in range(FLAT_INPUT_COUNT) if column not in OMITTED_COLUMNS)
PROJECTED_INPUT_COUNT = len(KEPT_COLUMNS)
BASE_FORMAT = "raccoon-scene-no-moving-mlp.v1"
CORRECTION_FORMAT = "raccoon-scene-no-moving-early-history-correction.v1"


def _check_contract() -> None:
    if (scene_features.SCHEMA != "raccoon-scene-policy-features.v1"
            or scene_features.HISTORY_LENGTH != HISTORY_LENGTH
            or len(scene_features.FEATURE_NAMES) != FEATURE_COUNT
            or scene_features.FEATURE_NAMES[MOVING_INDEX] != "moving"
            or tuple(motion_policy.ACTIONS) != ("move", "open", "close", "done")
            or PROJECTED_INPUT_COUNT != 656):
        raise ValueError("scene feature or action contract differs")


class SceneNoMovingPolicy(nn.Module):
    """MLP64 with 656 inputs after omitting the eight flattened moving columns."""

    def __init__(self):
        super().__init__()
        _check_contract()
        self.architecture = "no_moving_mlp"
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.register_buffer("_kept_columns", torch.tensor(KEPT_COLUMNS, dtype=torch.long), persistent=False)
        self.encoder = nn.Sequential(nn.Linear(PROJECTED_INPUT_COUNT, HIDDEN_SIZE), nn.ReLU(),
                                     nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU())
        self.action_head = nn.Linear(HIDDEN_SIZE, len(motion_policy.ACTIONS))
        self.joint_head = nn.Linear(HIDDEN_SIZE, 4)

    def _validate(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        # This is the unmodified source policy's validation, before projection.
        return scene_policy.ScenePolicy._validate(self, windows, masks)

    def project_inputs(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return [batch, 656] with unchanged non-moving and mask columns.

        The supplied windows and masks are never modified. Invalid observations
        remain invalid even if the offending value is in the omitted column.
        """
        self._validate(windows, masks)
        return self._project_validated(windows, masks)

    def _project_validated(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
        flat = torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1)
        return flat.index_select(1, self._kept_columns)

    def _encode_with_lengths(self, windows: torch.Tensor, masks: torch.Tensor):
        lengths = self._validate(windows, masks)
        return self.encoder(self._project_validated(windows, masks)), lengths

    def encode(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return the 64-dimensional encoder representation for fitting."""
        encoded, _ = self._encode_with_lengths(windows, masks)
        return encoded

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        encoded = self.encode(windows, masks)
        return self.action_head(encoded), self.joint_head(encoded)


class SceneNoMovingCorrectionPolicy(early.SceneEarlyHistoryCorrectionPolicy):
    """Existing early-history correction semantics on an independent projected base."""

    def __init__(self):
        # The parent constructor creates the ordinary 664-input base.
        nn.Module.__init__(self)
        self.architecture = "no_moving_early_history_correction_mlp"
        self.history_length, self.feature_count, self.hidden_size = HISTORY_LENGTH, FEATURE_COUNT, HIDDEN_SIZE
        self.base = SceneNoMovingPolicy()
        self.base.requires_grad_(False)
        self.correction_head = nn.Linear(HIDDEN_SIZE, 4)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> early.CorrectionTrainingOutputs:
        """Return frozen base outputs and an ungated head for teacher-MOVE fitting."""
        with torch.no_grad():
            encoded, lengths = self.base._encode_with_lengths(windows, masks)
            logits = self.base.action_head(encoded)
            joints = self.base.joint_head(encoded)
        return early.CorrectionTrainingOutputs(logits, joints, self.correction_head(encoded), lengths)


def initialize_from_scene_policy(source: scene_policy.ScenePolicy) -> SceneNoMovingPolicy:
    """Copy an ordinary MLP64, deleting only its moving columns.

    The copied model is independent of the source. No optimizer is copied and
    construction does not advance the caller's CPU random stream.
    """
    _check_contract()
    device, dtype = early._validate_source(source)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneNoMovingPolicy()
    model = model.to(device=device, dtype=dtype)
    source_state = source.state_dict()
    with torch.no_grad():
        model.encoder[0].weight.copy_(source_state["encoder.0.weight"][:, list(KEPT_COLUMNS)])
        model.encoder[0].bias.copy_(source_state["encoder.0.bias"])
        model.encoder[2].load_state_dict(source.encoder[2].state_dict(), strict=True)
        model.action_head.load_state_dict(source.action_head.state_dict(), strict=True)
        model.joint_head.load_state_dict(source.joint_head.state_dict(), strict=True)
    model.train(source.training)
    return model


def initialize_correction(source: SceneNoMovingPolicy) -> SceneNoMovingCorrectionPolicy:
    """Freeze an independent copy of this base and create a zero correction head."""
    if type(source) is not SceneNoMovingPolicy:
        raise ValueError("source must be the projected no-moving policy")
    # Validate the complete source before binding it to a correction head.
    payload = checkpoint_payload(source)
    state = payload["state_dict"]
    param = source.encoder[0].weight
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = SceneNoMovingCorrectionPolicy()
    model = model.to(device=param.device, dtype=param.dtype)
    model.base.load_state_dict(state, strict=True)
    model.base.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract(*, correction: bool = False) -> dict:
    _check_contract()
    if type(correction) is not bool:
        raise ValueError("correction must be a boolean")
    result = {"format": CORRECTION_FORMAT if correction else BASE_FORMAT,
              "architecture": "no_moving_early_history_correction_mlp" if correction else "no_moving_mlp",
              "feature_schema": scene_features.SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
              "actions": list(motion_policy.ACTIONS), "history_length": HISTORY_LENGTH,
              "external_feature_count": FEATURE_COUNT, "hidden_size": HIDDEN_SIZE,
              "flattened_input_count": FLAT_INPUT_COUNT, "projected_input_count": PROJECTED_INPUT_COUNT,
              "omitted_columns": list(OMITTED_COLUMNS), "joint_scale_degrees": 180.0,
              "input_projection": "mask_each_frame_then_append_mask_then_remove_moving_columns"}
    if correction:
        result.update({"max_corrected_history": early.MAX_CORRECTED_HISTORY,
                       "inference_gate": "history_1_to_3_and_predicted_move",
                       "supervision_gate": "history_1_to_3_and_teacher_move",
                       "output": "absolute_base_plus_correction_no_clipping"})
    return result


def checkpoint_payload(model: nn.Module) -> dict:
    """Make an independent CPU state with an explicit base or correction contract."""
    if type(model) is SceneNoMovingPolicy:
        correction = False
    elif type(model) is SceneNoMovingCorrectionPolicy:
        correction = True
        if any(p.requires_grad for p in model.base.parameters()):
            raise ValueError("correction base must remain frozen")
    else:
        raise ValueError("no-moving policy or correction policy required")
    payload = {"contract": checkpoint_contract(correction=correction),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> nn.Module:
    """Restore only one of the two explicit formats, verifying every tensor."""
    if not isinstance(payload, dict):
        raise ValueError("no-moving checkpoint must be an object")
    contract = payload.get("contract")
    if contract == checkpoint_contract():
        cls = SceneNoMovingPolicy
    elif contract == checkpoint_contract(correction=True):
        cls = SceneNoMovingCorrectionPolicy
    else:
        raise ValueError("no-moving checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint state dictionary is required")
    first = next(iter(state.values()))
    if (not isinstance(first, torch.Tensor) or first.device.type != "cpu"
            or first.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = cls().to(dtype=first.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != first.dtype or tensor.shape != expected[name].shape
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    if type(model) is SceneNoMovingCorrectionPolicy:
        model.base.requires_grad_(False)
    model.eval()
    return model
