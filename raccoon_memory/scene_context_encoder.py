"""Contract-preserving input preparation for a future 188-input policy.

The caller supplies only its existing eight-row past/current window and mask.
This module has no learned parameters, output head, simulator or device access.
"""
from __future__ import annotations

import torch

from raccoon_memory import scene_features

HISTORY_LENGTH = 8
DYNAMIC_COUNT = 14
CONTEXT_COUNT = 68
FEATURE_COUNT = DYNAMIC_COUNT + CONTEXT_COUNT
OUTPUT_COUNT = HISTORY_LENGTH * DYNAMIC_COUNT + HISTORY_LENGTH + CONTEXT_COUNT
SCHEMA = "raccoon-scene-history-context-input.v1"

_DYNAMIC_NAMES = (
    "joint_1_deg/180", "joint_2_deg/180", "joint_3_deg/180", "joint_4_deg/180",
    "sensor_valid", "moving",
    "gripper_command_none", "gripper_command_open", "gripper_command_closed",
    "gripper_reported_none", "gripper_reported_open", "gripper_reported_closed",
    "gripper_reported_valid", "gripper_feedback_fresh",
)
_CONTEXT_NAMES = (
    *(f"source_slot_{i}" for i in range(1, 5)),
    *(f"destination_slot_{i}" for i in range(1, 5)),
    *(f"source_level_{i}" for i in range(1, 3)),
    *(f"destination_level_{i}" for i in range(1, 3)),
    *(f"moved_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_level_{level}_id_{obj}"
      for slot in range(1, 5) for level in range(1, 3) for obj in range(5)),
    *(f"active_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_{axis}_mm/300" for slot in range(1, 5) for axis in ("x", "y")),
)
_EXPECTED_FEATURE_NAMES = _DYNAMIC_NAMES + _CONTEXT_NAMES


def encode_history_context(windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return dynamic history, masks, and the current registered context.

    `windows` is (batch, 8, 82); `masks` is a boolean, left-padded
    (batch, 8) mask whose last value is true. On valid rows the 68 context
    features must be exactly equal. Padded rows are ignored and zeroed in the
    resulting dynamic block. The output layout is 112 dynamic values in time
    order, 8 mask values, then the current row's 68 context values.

    The function does not establish whether the caller built its window from
    past/current observations; the caller must maintain that causal boundary.
    """
    if (scene_features.SCHEMA != "raccoon-scene-policy-features.v1"
            or scene_features.HISTORY_LENGTH != HISTORY_LENGTH
            or tuple(scene_features.FEATURE_NAMES) != _EXPECTED_FEATURE_NAMES):
        raise ValueError("scene feature schema or feature ordering differs")
    if (not isinstance(windows, torch.Tensor) or not windows.is_floating_point()
            or windows.ndim != 3 or windows.shape[1:] != (HISTORY_LENGTH, FEATURE_COUNT)
            or not windows.shape[0] or not bool(torch.isfinite(windows).all())):
        raise ValueError("windows must be nonempty finite floating tensors of shape (batch, 8, 82)")
    if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
            or masks.shape != windows.shape[:2] or masks.device != windows.device):
        raise ValueError("masks must be boolean and match the batch, history and device")
    if not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any()):
        raise ValueError("each mask must be left-padded and include the current observation")

    context = windows[:, -1, DYNAMIC_COUNT:]
    contexts_equal = (windows[:, :, DYNAMIC_COUNT:] == context.unsqueeze(1)).all(dim=-1)
    if not bool((contexts_equal | ~masks).all()):
        raise ValueError("registered context differs between valid history rows")
    dynamic = windows[:, :, :DYNAMIC_COUNT].masked_fill(~masks.unsqueeze(-1), 0.0)
    return torch.cat((dynamic.reshape(windows.shape[0], -1), masks.to(windows.dtype), context), dim=1)
