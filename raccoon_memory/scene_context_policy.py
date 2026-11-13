"""Isolated 188-input MLP candidates; no training or device control.

Both output contracts receive the existing causal observation window. The
encoder validates and separates dynamic history and registered context.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_memory import motion_policy, scene_context_encoder, scene_features

EXCLUSIVE = "exclusive"
SIMULTANEOUS = "simultaneous"
OUTPUT_MODES = (EXCLUSIVE, SIMULTANEOUS)
EXCLUSIVE_ACTION_NAMES = ("move", "open", "close", "done")
GRIPPER_NAMES = ("open", "closed")
STOP_NAMES = ("continue", "done")
CHECKPOINT_FORMATS = {
    EXCLUSIVE: "raccoon-scene-context188-exclusive-policy.v1",
    SIMULTANEOUS: "raccoon-scene-context188-simultaneous-policy.v1",
}
PARAMETER_COUNT = 16_776


def _check_mode(output_mode: str) -> None:
    if not isinstance(output_mode, str) or output_mode not in OUTPUT_MODES:
        raise ValueError("output mode must be exclusive or simultaneous")


def _check_source_contract() -> None:
    if (scene_features.SCHEMA != "raccoon-scene-policy-features.v1"
            or tuple(scene_features.FEATURE_NAMES) != scene_context_encoder._EXPECTED_FEATURE_NAMES
            or scene_features.HISTORY_LENGTH != 8
            or tuple(motion_policy.ACTIONS) != EXCLUSIVE_ACTION_NAMES
            or motion_policy.JOINT_SCALE_DEG != 180.0):
        raise ValueError("source feature, action or joint-scale contract differs")


class SceneContextPolicy(nn.Module):
    """MLP64 with either the exclusive or simultaneous action heads."""

    def __init__(self, output_mode: str):
        super().__init__()
        _check_mode(output_mode)
        _check_source_contract()
        self.output_mode = output_mode
        self.history_length, self.feature_count, self.hidden_size = 8, 82, 64
        self.encoder = nn.Sequential(
            nn.Linear(scene_context_encoder.OUTPUT_COUNT, 64, device="cpu", dtype=torch.float32), nn.ReLU(),
            nn.Linear(64, 64, device="cpu", dtype=torch.float32), nn.ReLU(),
        )
        self.joint_head = nn.Linear(64, 4, device="cpu", dtype=torch.float32)
        if output_mode == EXCLUSIVE:
            self.action_head = nn.Linear(64, 4, device="cpu", dtype=torch.float32)
        else:
            self.gripper_head = nn.Linear(64, 2, device="cpu", dtype=torch.float32)
            self.stop_head = nn.Linear(64, 2, device="cpu", dtype=torch.float32)
        if sum(parameter.numel() for parameter in self.parameters()) != PARAMETER_COUNT:
            raise ValueError("context policy parameter count differs")

    def forward(self, windows: torch.Tensor, masks: torch.Tensor):
        values = scene_context_encoder.encode_history_context(windows, masks)
        encoded = self.encoder(values)
        joints = self.joint_head(encoded)
        if self.output_mode == EXCLUSIVE:
            return self.action_head(encoded), joints
        return joints, self.gripper_head(encoded), self.stop_head(encoded)


def paired_random_models(seed: int) -> tuple[SceneContextPolicy, SceneContextPolicy]:
    """Return fresh numerically paired CPU models without changing caller RNG.

    Random action-head rows 0:2 are copied to the gripper head and rows 2:4
    to the stop head. These numerical correspondences do not imply that the
    different classes have identical semantics or produce the same action.
    """
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be an integer in the unsigned 64-bit range")
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        exclusive = SceneContextPolicy(EXCLUSIVE)
        simultaneous = SceneContextPolicy(SIMULTANEOUS)
    simultaneous.encoder.load_state_dict(exclusive.encoder.state_dict(), strict=True)
    simultaneous.joint_head.load_state_dict(exclusive.joint_head.state_dict(), strict=True)
    with torch.no_grad():
        simultaneous.gripper_head.weight.copy_(exclusive.action_head.weight[:2])
        simultaneous.gripper_head.bias.copy_(exclusive.action_head.bias[:2])
        simultaneous.stop_head.weight.copy_(exclusive.action_head.weight[2:])
        simultaneous.stop_head.bias.copy_(exclusive.action_head.bias[2:])
    return exclusive, simultaneous


def checkpoint_contract(output_mode: str) -> dict:
    """Describe an input/output contract; this does not assert training."""
    _check_mode(output_mode)
    _check_source_contract()
    heads = ({"return_order": ["action_logits", "joint_output"],
              "action_names": list(EXCLUSIVE_ACTION_NAMES)} if output_mode == EXCLUSIVE else
             {"return_order": ["joint_output", "gripper_logits", "stop_logits"],
              "gripper_names": list(GRIPPER_NAMES), "stop_names": list(STOP_NAMES)})
    return {
        "format": CHECKPOINT_FORMATS[output_mode],
        "model_config": {"architecture": "context_mlp", "hidden_size": 64, "history_length": 8,
                         "source_features": 82, "encoded_features": 188,
                         "joint_parameterization": "absolute_degrees_div_180", "output_mode": output_mode,
                         "parameter_dtype": "float32"},
        "source_feature_schema": scene_features.SCHEMA,
        "source_feature_names": list(scene_features.FEATURE_NAMES),
        "input_schema": scene_context_encoder.SCHEMA,
        "input_layout": {"dynamic": 112, "mask": 8, "current_context": 68,
                         "order": "dynamic_history_then_mask_then_current_context"},
        "head_contract": {"joint_outputs": 4, **heads},
    }


def checkpoint_payload(model: SceneContextPolicy) -> dict:
    """Create an in-memory CPU state package without assigning training status."""
    if not isinstance(model, SceneContextPolicy):
        raise ValueError("a context policy is required")
    payload = checkpoint_contract(model.output_mode)
    payload["state_dict"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    # A separately modified or non-FP32 model must not be serialized under this contract.
    load_checkpoint(payload, model.output_mode)
    return payload


def load_checkpoint(payload: dict, expected_output_mode: str) -> SceneContextPolicy:
    """Validate and load this 188-input contract; reject old or crossed formats."""
    contract = checkpoint_contract(expected_output_mode)
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in contract.items()):
        raise ValueError("context checkpoint input or output contract differs")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("context checkpoint requires a state dictionary")
    with torch.random.fork_rng(devices=[]):
        model = SceneContextPolicy(expected_output_mode)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("context checkpoint parameter names differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError("context checkpoint parameter shape, dtype or value differs: " + name)
    model.load_state_dict(state, strict=True)
    return model
