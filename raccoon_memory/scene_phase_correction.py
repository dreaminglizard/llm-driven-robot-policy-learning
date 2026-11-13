"""Frozen current-observation policy with a selected late-history correction.

The only inference inputs are the existing 82-feature history and mask.  A
145-to-7 selector and seven 145-to-4 residuals use the original frozen
64-vector plus the current 81 non-moving features. No phase label, waypoint,
object truth, future observation, clipping or sequencing rule is accepted.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from raccoon_memory import motion_policy
from raccoon_memory import scene_current_observation_correction as current
from raccoon_memory import scene_early_history_correction as early


STAGES = ("grasp_lift", "clearance_lift", "transit_turn", "destination_above",
          "destination_preposition", "placement", "retreat")
HEAD_FEATURE_COUNT = current.HEAD_FEATURE_COUNT
MIN_LATE_HISTORY = early.MAX_CORRECTED_HISTORY + 1
CHECKPOINT_FORMAT = "raccoon-scene-phase-selected-correction.v1"


class PhaseOutputs(NamedTuple):
    features: torch.Tensor
    action_logits: torch.Tensor
    base_joints: torch.Tensor
    early_correction: torch.Tensor
    phase_logits: torch.Tensor
    phase_index: torch.Tensor
    phase_correction: torch.Tensor
    history_lengths: torch.Tensor


class PhaseDecision(NamedTuple):
    action_logits: torch.Tensor
    joints: torch.Tensor
    phase_logits: torch.Tensor
    phase_index: torch.Tensor
    late_correction_gate: torch.Tensor
    selector_valid: torch.Tensor
    history_lengths: torch.Tensor


class ScenePhaseCorrectionPolicy(nn.Module):
    """Keep the full prior policy frozen and select one fixed late residual."""

    def __init__(self):
        super().__init__()
        current._check_contract()
        self.architecture = "phase_selected_current_observation_correction_mlp"
        self.history_length = current.HISTORY_LENGTH
        self.feature_count = current.FEATURE_COUNT
        self.hidden_size = current.HIDDEN_SIZE
        self.frozen_policy = current.SceneCurrentObservationCorrectionPolicy()
        self.frozen_policy.requires_grad_(False)
        self.frozen_policy.eval()
        self.phase_selector = nn.Linear(HEAD_FEATURE_COUNT, len(STAGES))
        self.phase_heads = nn.ModuleList(nn.Linear(HEAD_FEATURE_COUNT, 4) for _ in STAGES)
        nn.init.zeros_(self.phase_selector.weight)
        nn.init.zeros_(self.phase_selector.bias)
        for head in self.phase_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.phase_selector.requires_grad_(False)
        self.phase_heads.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_policy.eval()
        return self

    def head_features(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.frozen_policy.head_features(windows, masks)

    def phase_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> PhaseOutputs:
        """Expose selector decisions for recorded-input verification."""
        with torch.no_grad():
            features, logits, joints, lengths = self.frozen_policy._frozen_outputs(windows, masks)
            early_correction = self.frozen_policy.correction_head(features)
            phase_logits = self.phase_selector(features)
            phase_index = phase_logits.argmax(dim=1)
            candidates = torch.stack([head(features) for head in self.phase_heads], dim=1)
            selected = candidates[torch.arange(len(features), device=features.device), phase_index]
        return PhaseOutputs(features, logits, joints, early_correction, phase_logits, phase_index, selected, lengths)

    def forward_with_diagnostics(self, windows: torch.Tensor, masks: torch.Tensor) -> PhaseDecision:
        """Return outputs and same-computation phase choice/gate for logging.

        The gate is a model decision, not proof a command was applied.  When
        selector_valid is false, phase_index is not a meaningful selection.
        """
        out = self.phase_outputs(windows, masks)
        is_move = out.action_logits.argmax(dim=1) == motion_policy.MOVE
        early_gate = (out.history_lengths < MIN_LATE_HISTORY) & is_move
        late_gate = (out.history_lengths >= MIN_LATE_HISTORY) & is_move
        original = torch.where(early_gate.unsqueeze(1), out.base_joints + out.early_correction, out.base_joints)
        candidate = out.base_joints + out.phase_correction
        # A nonfinite selector cannot make a meaningful argmax. Propagate a
        # nonfinite proposal for an applied row so the existing evaluator can
        # reject it; unused rows retain the original outputs exactly.
        selector_valid = torch.isfinite(out.phase_logits).all(dim=1)
        candidate = torch.where(selector_valid.unsqueeze(1), candidate,
                                torch.full_like(candidate, float("nan")))
        joints = torch.where(late_gate.unsqueeze(1), candidate, original)
        return PhaseDecision(out.action_logits, joints, out.phase_logits, out.phase_index, late_gate,
                             selector_valid, out.history_lengths)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the existing evaluator's action-logits and joint contract."""
        result = self.forward_with_diagnostics(windows, masks)
        return result.action_logits, result.joints


def initialize_from_current(source: current.SceneCurrentObservationCorrectionPolicy) -> ScenePhaseCorrectionPolicy:
    """Copy the full current policy without sharing tensors; zero new heads."""
    if type(source) is not current.SceneCurrentObservationCorrectionPolicy:
        raise ValueError("source must be the existing 145-input current-observation correction policy")
    payload = current.checkpoint_payload(source)
    parameter = source.correction_head.weight
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = ScenePhaseCorrectionPolicy()
    model = model.to(device=parameter.device, dtype=parameter.dtype)
    model.frozen_policy.load_state_dict(payload["state_dict"], strict=True)
    model.frozen_policy.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract() -> dict:
    return {"format": CHECKPOINT_FORMAT, "architecture": "phase_selected_current_observation_correction_mlp",
            "frozen_policy_contract": current.checkpoint_contract(), "history_length": current.HISTORY_LENGTH,
            "external_feature_count": current.FEATURE_COUNT, "hidden_size": current.HIDDEN_SIZE,
            "head_feature_count": HEAD_FEATURE_COUNT, "stage_names": list(STAGES), "joint_count": 4,
            "joint_scale_degrees": 180.0,
            "head_input": "frozen_encoding_64_then_current_observation_without_moving_81",
            "early_gate": "history_1_to_3_and_predicted_move_existing_head",
            "late_gate": "history_4_to_8_and_predicted_move_selector_argmax",
            "selector": "linear_145_to_7", "residuals": "seven_linear_145_to_4",
            "output": "absolute_base_plus_applicable_correction_no_clipping"}


def checkpoint_payload(model: ScenePhaseCorrectionPolicy) -> dict:
    if type(model) is not ScenePhaseCorrectionPolicy:
        raise ValueError("phase correction policy required")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("existing policy and phase parameters must remain frozen")
    payload = {"contract": checkpoint_contract(),
               "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> ScenePhaseCorrectionPolicy:
    """Restore only the new format and validate every tensor before loading."""
    if not isinstance(payload, dict) or payload.get("contract") != checkpoint_contract():
        raise ValueError("phase correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "phase_selector.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["phase_selector.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = ScenePhaseCorrectionPolicy().to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != weight.dtype
                or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.requires_grad_(False)
    model.eval()
    return model
