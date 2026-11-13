"""Current-observation nonlinear late correction on a frozen scene policy.

Inference still receives the original 82-feature history and mask.  Only the
additional late correction is independent of history: it receives the last
valid observation without the moving feature.  The frozen policy continues
to use history for its action and base joint predictions.
"""
from __future__ import annotations

import re
from typing import NamedTuple

import torch
from torch import nn

from raccoon_memory import motion_policy
from raccoon_memory import scene_current_observation_correction as current
from raccoon_memory import scene_early_history_correction as early


INITIALIZATION_SEED = 20261030
CURRENT_FEATURE_COUNT = current.CURRENT_FEATURE_COUNT
LATE_HIDDEN_SIZE = 64
MIN_LATE_HISTORY = early.MAX_CORRECTED_HISTORY + 1
CHECKPOINT_FORMAT = "raccoon-scene-current-late-nonlinear-correction.v1"


def _checked_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("late-head initialization seed must be an integer in [0, 2**32)")
    return seed


class CurrentLateOutputs(NamedTuple):
    """One calculation for fitting and for inspecting the actual decision.

    The correction tensors are ungated, so teacher-MOVE supervision may use
    ``late_correction`` independently of the student's inference decision.
    The boolean gates describe which correction contributes to ``joints``.
    """

    features: torch.Tensor
    action_logits: torch.Tensor
    base_joints: torch.Tensor
    early_correction: torch.Tensor
    late_correction: torch.Tensor
    history_lengths: torch.Tensor
    early_correction_gate: torch.Tensor
    late_correction_gate: torch.Tensor
    joints: torch.Tensor


class SceneCurrentLateNonlinearPolicy(nn.Module):
    """Freeze the entire prior policy and train only the 81-to-64-to-4 head."""

    def __init__(self, *, initialization_seed=INITIALIZATION_SEED):
        super().__init__()
        current._check_contract()
        self.initialization_seed = _checked_seed(initialization_seed)
        self.architecture = "current_observation_late_nonlinear_correction_mlp"
        self.history_length = current.HISTORY_LENGTH
        self.feature_count = current.FEATURE_COUNT
        self.hidden_size = current.HIDDEN_SIZE

        # Construct on CPU without consuming or changing the caller's RNG.
        # Seeding immediately before the first Linear makes its parameters
        # exactly PyTorch's default initialization for the recorded seed.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            self.frozen_policy = current.SceneCurrentObservationCorrectionPolicy()
            torch.random.default_generator.manual_seed(self.initialization_seed)
            self.late_head = nn.Sequential(
                nn.Linear(CURRENT_FEATURE_COUNT, LATE_HIDDEN_SIZE, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(LATE_HIDDEN_SIZE, 4, dtype=torch.float32),
            )
            nn.init.zeros_(self.late_head[2].weight)
            nn.init.zeros_(self.late_head[2].bias)
        self.frozen_policy.to(dtype=torch.float32)
        self.frozen_policy.requires_grad_(False)
        self.frozen_policy.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_policy.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.frozen_policy.requires_grad_(False)
        return self

    def head_features(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return the validated last observation's 81 non-moving values."""
        features = self.frozen_policy.head_features(windows, masks)
        return features[:, current.HIDDEN_SIZE:]

    def diagnostics(self, windows: torch.Tensor, masks: torch.Tensor) -> CurrentLateOutputs:
        """Return the exact components, gates and proposal used by forward."""
        with torch.no_grad():
            features145, logits, base_joints, lengths = self.frozen_policy._frozen_outputs(windows, masks)
            early_correction = self.frozen_policy.correction_head(features145)
            features = features145[:, current.HIDDEN_SIZE:]
        late_correction = self.late_head(features)
        is_move = logits.argmax(dim=1) == motion_policy.MOVE
        early_gate = (lengths < MIN_LATE_HISTORY) & is_move
        late_gate = (lengths >= MIN_LATE_HISTORY) & is_move
        existing = torch.where(early_gate.unsqueeze(1), base_joints + early_correction, base_joints)
        joints = torch.where(late_gate.unsqueeze(1), base_joints + late_correction, existing)
        return CurrentLateOutputs(features, logits, base_joints, early_correction, late_correction,
                                  lengths, early_gate, late_gate, joints)

    def training_outputs(self, windows: torch.Tensor, masks: torch.Tensor) -> CurrentLateOutputs:
        """Expose the ungated late head along with the actual inference result."""
        return self.diagnostics(windows, masks)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.diagnostics(windows, masks)
        return out.action_logits, out.joints


def initialize_from_current(source: current.SceneCurrentObservationCorrectionPolicy) -> SceneCurrentLateNonlinearPolicy:
    """Independently copy the complete existing policy and initialize the new head."""
    if type(source) is not current.SceneCurrentObservationCorrectionPolicy:
        raise ValueError("source must be the existing current-observation correction policy")
    payload = current.checkpoint_payload(source)
    parameter = source.correction_head.weight
    model = SceneCurrentLateNonlinearPolicy().to(device=parameter.device, dtype=parameter.dtype)
    model.frozen_policy.load_state_dict(payload["state_dict"], strict=True)
    model.frozen_policy.requires_grad_(False)
    model.train(source.training)
    return model


def checkpoint_contract(*, initialization_seed=INITIALIZATION_SEED) -> dict:
    """Contract distinct from both the existing and the linear late policies."""
    current._check_contract()
    return {
        "format": CHECKPOINT_FORMAT,
        "architecture": "current_observation_late_nonlinear_correction_mlp",
        "frozen_policy_contract": current.checkpoint_contract(),
        "history_length": current.HISTORY_LENGTH,
        "external_feature_count": current.FEATURE_COUNT,
        "current_feature_columns": list(current.CURRENT_COLUMNS),
        "current_feature_count": CURRENT_FEATURE_COUNT,
        "late_hidden_size": LATE_HIDDEN_SIZE,
        "joint_count": 4,
        "joint_scale_degrees": 180.0,
        "head_input": "last_valid_observation_without_moving_81",
        "head_architecture": "linear_81_64_relu_linear_64_4",
        "initialization": f"pytorch_default_first_linear_seed_{_checked_seed(initialization_seed)}_last_linear_zero",
        "early_gate": "history_1_to_3_and_predicted_move_existing_head",
        "late_gate": "history_4_to_8_and_predicted_move_new_head",
        "output": "absolute_base_plus_applicable_correction_no_clipping",
    }


def checkpoint_payload(model: SceneCurrentLateNonlinearPolicy) -> dict:
    """Return independent CPU tensors after checking the frozen policy."""
    if type(model) is not SceneCurrentLateNonlinearPolicy:
        raise ValueError("current-observation nonlinear late correction policy required")
    if any(parameter.requires_grad for parameter in model.frozen_policy.parameters()):
        raise ValueError("entire existing policy must remain frozen")
    if any(module.training for module in model.frozen_policy.modules()):
        raise ValueError("entire existing policy must remain in evaluation mode")
    payload = {
        "contract": checkpoint_contract(initialization_seed=model.initialization_seed),
        "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
    }
    load_checkpoint(payload)
    return payload


def load_checkpoint(payload: dict, *, device="cpu") -> SceneCurrentLateNonlinearPolicy:
    """Strictly restore only this format and freeze the complete prior policy."""
    contract = payload.get("contract") if isinstance(payload, dict) else None
    initialization = contract.get("initialization") if isinstance(contract, dict) else None
    # The existing v1 contract stores the seed in this descriptor.
    match = (re.fullmatch(r"pytorch_default_first_linear_seed_(0|[1-9][0-9]{0,9})_last_linear_zero", initialization)
             if isinstance(initialization, str) else None)
    seed = int(match[1]) if match else None
    if seed is None or seed >= 2**32 or contract != checkpoint_contract(initialization_seed=seed):
        raise ValueError("current-observation nonlinear late correction checkpoint contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "late_head.0.weight" not in state:
        raise ValueError("checkpoint state dictionary is required")
    weight = state["late_head.0.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
            or weight.dtype not in (torch.float32, torch.float64)):
        raise ValueError("checkpoint tensors must be CPU float32 or float64")
    model = SceneCurrentLateNonlinearPolicy(initialization_seed=seed).to(dtype=weight.dtype)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names differ")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != weight.dtype
                or tensor.shape != expected[name].shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"checkpoint parameter shape, dtype, device, or value differs: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.frozen_policy.requires_grad_(False)
    model.eval()
    return model
