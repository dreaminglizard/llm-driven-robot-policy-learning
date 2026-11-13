"""The same joint-output GRU with one explicit registration-boundary choice.

The 51 inputs, decoder, and 89,352 learned parameters are inherited from the
existing direct student.  A caller starts each *registration* once.  Within a
registration, the existing history mask marks the first observation of each
transfer; it is bookkeeping for resetting state, not a policy feature.
"""
from __future__ import annotations

import torch

from . import motion_policy
from . import scene_chunked_recurrent_policy as original
from .scene_causal_frame_policy import CONTEXT_COUNT, context

FORMAT = "raccoon-direct-boundary-recurrent-pair.v1"
MODES = ("transfer_reset", "registration_reset")


class BoundaryRecurrentPolicy(original.ChunkedRecurrentPolicy):
    """A single model, never a router between models or action rules."""

    def __init__(self, *, center, scale, mode):
        if mode not in MODES:
            raise ValueError("unknown recurrent boundary mode")
        super().__init__(horizon=1, center=center, scale=scale, output_space="joint")
        self.mode = mode
        self._registration_started = False

    def begin_registration(self):
        """Call exactly when a new real registration begins, before its first row."""
        self._hidden = None
        self._registration_started = True

    def forward_with_control_state(self, windows, masks, previous):
        if (windows.shape != (1, 8, 82) or masks.shape != (1, 8)
                or masks.dtype != torch.bool or previous.shape != (1, 4)
                or windows.device.type != "cpu" or previous.device.type != "cpu"):
            raise ValueError("one CPU observation and its applied control are required")
        if not self._registration_started:
            raise ValueError("begin_registration must be called before the first observation")
        count = int(masks.sum().item())
        if count < 1 or masks[0].tolist() != [False] * (8-count) + [True] * count:
            raise ValueError("invalid transfer history mask")
        if self._hidden is None and count != 1:
            raise ValueError("a registration must start at a transfer's first observation")
        if self.mode == "transfer_reset" and count == 1:
            self._hidden = None
        raw = context(windows[:, -1], previous, "polar").to(torch.float32)
        encoded, hidden = self.encoder(self.input((raw-self.center)/self.scale).unsqueeze(1), self._hidden)
        self._hidden = hidden.detach()
        logits, angles = self.decode(encoded[:, 0], previous, steps=1)
        target = angles[:, 0]
        if int(logits[:, 0].argmax()) != motion_policy.MOVE:
            target = previous/180.0
        self.last_diagnostic = {"output_space": "joint", "nominal_target_pose_mm_deg": None,
                                "ik_geometrically_reachable": None, "pose_rejection": None}
        return logits[:, 0], target

    def forward_padded(self, features, previous, valid, transfer_start):
        """Train on the same right-padded registration batches in both modes.

        Each real observation, including a transfer's DONE row, occurs once.
        Segments are the same for both models; only the reset mask differs.
        Padded timesteps cannot influence a later valid observation.
        """
        if (features.ndim != 3 or features.shape[-1] != CONTEXT_COUNT
                or previous.shape != (*features.shape[:2], 4)
                or valid.shape != features.shape[:2] or transfer_start.shape != valid.shape
                or valid.dtype != torch.bool or transfer_start.dtype != torch.bool
                or not bool(torch.isfinite(features).all() and torch.isfinite(previous).all())):
            raise ValueError("invalid batched public observations")
        if (not bool(valid[:, 0].all()) or not bool(transfer_start[:, 0].all())
                or bool((transfer_start & ~valid).any())
                or bool(((~valid[:, :-1]) & valid[:, 1:]).any())):
            raise ValueError("registrations must be nonempty, right padded, and start a transfer")
        boundaries = [0] + (torch.nonzero(transfer_start[:, 1:].any(0), as_tuple=True)[0]+1).tolist()
        boundaries.append(features.shape[1])
        projected = self.input((features-self.center)/self.scale)
        hidden = None
        parts = []
        for start, end in zip(boundaries, boundaries[1:]):
            if hidden is not None and self.mode == "transfer_reset":
                hidden = hidden * (~transfer_start[:, start])[None, :, None]
            encoded, hidden = self.encoder(projected[:, start:end], hidden)
            parts.append(encoded)
        encoded = torch.cat(parts, dim=1)
        batch, length = features.shape[:2]
        logits, angles = self.decode(encoded.reshape(-1, 96), previous.reshape(-1, 4), steps=1)
        return logits.reshape(batch, length, 4), angles.reshape(batch, length, 4)


def from_initial_checkpoint(payload, *, mode):
    """Copy an existing joint/horizon-one student without changing its state."""
    if payload.get("format") != original.FORMAT or payload.get("configuration") != {
            "horizon": 1, "output_space": "joint"}:
        raise ValueError("the existing joint-output horizon-one checkpoint is required")
    state = payload["state_dict"]
    model = BoundaryRecurrentPolicy(center=state["center"], scale=state["scale"], mode=mode)
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    return model


def load_checkpoint(payload, device="cpu"):
    if payload.get("format") != FORMAT or str(device) != "cpu":
        raise ValueError("unsupported paired recurrent checkpoint or device")
    config, state = payload["configuration"], payload["state_dict"]
    if set(config) != {"mode", "horizon", "output_space"} or config["horizon"] != 1 or config["output_space"] != "joint":
        raise ValueError("unsupported paired recurrent checkpoint configuration")
    model = BoundaryRecurrentPolicy(center=state["center"], scale=state["scale"], mode=config["mode"])
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint")
    return model.eval()
