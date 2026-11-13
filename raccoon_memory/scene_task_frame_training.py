"""Packed computation for a complete registered TaskFrame training history.

This is an optional training computation, not a policy or execution adapter.
It requires one registration at the first row, nonempty histories and right
padding. The model's ordinary forward handles streaming inference and more
general boundaries. Floating point results need not be bitwise identical to
the stepwise training implementation; a run must record which it uses.
"""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import scene_task_frame_student as student


FORMAT = "raccoon-task-frame-packed-registration-training.v1"


def forward_registration(model, features, valid_mask):
    """Return the same output fields from one causal LSTM call per batch.

    No padding advances the recurrent state. Unlabeled context is retained;
    the caller's original label and confirmed-MOVE masks still select losses.
    This function does not accept prior state or a partial registration.
    """
    if type(model) is not student.TaskFrameStudent or not model.training:
        raise ValueError("a raw TaskFrameStudent in training mode is required")
    parameter = model.action_head.weight
    if (not isinstance(features, torch.Tensor) or features.ndim != 3 or min(features.shape[:2]) < 1
            or features.shape[-1] != student.FEATURE_COUNT
            or features.dtype != parameter.dtype or features.device != parameter.device):
        raise ValueError("features must be [batch,time,92] on the model dtype and device")
    batch, steps = features.shape[:2]
    if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != (batch, steps)
            or valid_mask.dtype != torch.bool or valid_mask.device != features.device):
        raise ValueError("validity mask must be boolean [batch,time] on the feature device")
    lengths = valid_mask.sum(-1)
    right_padding = torch.arange(steps, device=features.device)[None, :] < lengths[:, None]
    if bool((lengths == 0).any()) or not torch.equal(valid_mask, right_padding):
        raise ValueError("packed training needs nonempty histories and right padding")
    student._validate_rows(features[valid_mask], include_object_ids=model.include_object_ids)
    if (bool((features[:, 0, student.REGISTRATION_START_INDEX] != 1).any()) or
            bool((features[:, 1:, student.REGISTRATION_START_INDEX][valid_mask[:, 1:]] != 0).any())):
        raise ValueError("packed training needs one complete registration at the first row")
    clean = torch.where(valid_mask[..., None], features, features[valid_mask][0])
    tokens, public = student.public_features(clean, include_object_ids=model.include_object_ids,
                                            constants=model.nominal_constants)
    slots = model.slot_encoder(tokens)
    source = (slots * clean[..., 14:18, None]).sum(-2)
    destination = (slots * clean[..., 18:22, None]).sum(-2)
    encoded = model.encoder(torch.cat((slots.mean(-2), source, destination, public), dim=-1))
    packed = pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed, state = model.lstm(packed)
    decoded, _ = pad_packed_sequence(packed, batch_first=True, total_length=steps)
    action, pose = model.action_head(decoded), model.pose_head(decoded)
    return student.TaskFrameOutputs(
        torch.where(valid_mask[..., None], action, torch.zeros_like(action)),
        torch.where(valid_mask[..., None], pose, torch.zeros_like(pose)), state)
