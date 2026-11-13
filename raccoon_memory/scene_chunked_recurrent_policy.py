"""One causal recurrent student with a short autoregressive command decoder.

The encoder carries all observations within a transfer. Only the first decoded
command is executed; future command heads are auxiliary training predictions.
The joint-output variant learns targets inside the existing joint limits. The
polar-output variant learns nominal source-frame poses and rejects invalid IK
results through the existing guard. Task output instead expresses the pose
along/across the registered source-to-destination axis. No output variant uses
a teacher, stage router, or
object-state input at runtime.
"""
from __future__ import annotations

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy
from .scene_causal_frame_policy import CONTEXT_COUNT, context, decode_pose, inverse_or_reject

FORMAT = "raccoon-chunked-recurrent-direct-policy.v1"


class ChunkedRecurrentPolicy(nn.Module):
    history_length, feature_count = 8, 82

    def __init__(self, *, horizon, center, scale, output_space="joint"):
        super().__init__()
        if horizon not in (1, 3):
            raise ValueError("this controlled comparison uses horizon one or three")
        self.horizon = horizon
        if output_space not in ("joint", "polar", "task"):
            raise ValueError("unknown learned target representation")
        self.output_space = output_space
        self.register_buffer("center", center.detach().clone().to(torch.float32))
        self.register_buffer("scale", scale.detach().clone().to(torch.float32))
        if (self.center.shape != (CONTEXT_COUNT,) or self.scale.shape != self.center.shape
                or not bool(torch.isfinite(self.center).all() and torch.isfinite(self.scale).all())
                or bool((self.scale <= 0).any())):
            raise ValueError("finite geometric feature normalization is required")
        limits = torch.tensor(JOINT_LIMITS, dtype=torch.float32)
        self.register_buffer("joint_center", limits.mean(1))
        self.register_buffer("joint_half_range", (limits[:, 1]-limits[:, 0])/2-1e-4)
        self.input = nn.Sequential(nn.Linear(CONTEXT_COUNT, 64), nn.Tanh())
        self.encoder = nn.GRU(64, 96, batch_first=True)
        self.decoder_initial = nn.Sequential(nn.Linear(96, 64), nn.Tanh())
        self.decoder = nn.GRUCell(96+8, 64)
        self.output = nn.Linear(64, 8)
        self._hidden = None
        self.last_diagnostic = None

    def decode(self, encoded, previous, *, steps):
        if steps not in (1, self.horizon):
            raise ValueError("invalid decoder horizon")
        hidden = self.decoder_initial(encoded)
        initial_command = previous/180.0 if self.output_space == "joint" else previous
        command = torch.cat((encoded.new_zeros((len(encoded), 4)), initial_command), dim=1)
        logits, angles = [], []
        for _ in range(steps):
            hidden = self.decoder(torch.cat((encoded, command), dim=1), hidden)
            output = self.output(hidden)
            action = output[:, :4]
            target = ((self.joint_center+self.joint_half_range*torch.tanh(output[:, 4:]))/180.0
                      if self.output_space == "joint" else output[:, 4:])
            logits.append(action)
            angles.append(target)
            probability = action.softmax(-1)
            # This is an internal decoder input, never a blended physical goal.
            following = probability[:, :1]*target+(1-probability[:, :1])*command[:, 4:]
            command = torch.cat((probability, following), dim=1)
        return torch.stack(logits, dim=1), torch.stack(angles, dim=1)

    def forward_sequence(self, features, previous):
        if features.ndim != 3 or features.shape[-1] != CONTEXT_COUNT or previous.shape != (*features.shape[:2], 4):
            raise ValueError("full causal sequences and applied targets are required")
        encoded, _ = self.encoder(self.input((features-self.center)/self.scale))
        batch, length = encoded.shape[:2]
        # Pose heads use the observed previous polar pose for initial decoder
        # conditioning. Their target representation can differ from this input.
        decoder_previous = previous if self.output_space == "joint" else features[:, :, 4:8]
        logits, angles = self.decode(encoded.reshape(-1, 96), decoder_previous.reshape(-1, 4), steps=self.horizon)
        return logits.reshape(batch, length, self.horizon, 4), angles.reshape(batch, length, self.horizon, 4)

    def forward_with_control_state(self, windows, masks, previous):
        if (windows.shape != (1, 8, 82) or masks.shape != (1, 8) or masks.dtype != torch.bool
                or previous.shape != (1, 4) or windows.device.type != "cpu"):
            raise ValueError("one CPU transfer observation and exact applied target are required")
        valid = int(masks.sum().item())
        if valid < 1 or masks[0].tolist() != [False]*(8-valid)+[True]*valid:
            raise ValueError("invalid causal history mask")
        if valid == 1:
            self._hidden = None
        elif self._hidden is None:
            raise ValueError("the recurrent state must start at the transfer's first observation")
        raw = context(windows[:, -1], previous, "polar").to(torch.float32)
        encoded, hidden = self.encoder(self.input((raw-self.center)/self.scale).unsqueeze(1), self._hidden)
        self._hidden = hidden.detach()
        decoder_previous = previous if self.output_space == "joint" else raw[:, 4:8]
        logits, angles = self.decode(encoded[:, 0], decoder_previous, steps=1)
        target = angles[:, 0]
        self.last_diagnostic = {"output_space": self.output_space, "nominal_target_pose_mm_deg": None,
                                "ik_geometrically_reachable": None, "pose_rejection": None}
        if self.output_space != "joint" and int(logits[:, 0].argmax()) == motion_policy.MOVE:
            try:
                value = decode_pose(target[0].detach().numpy(), windows[0, -1].double(), self.output_space)
                joints, reachable = inverse_or_reject(value)
                self.last_diagnostic.update(nominal_target_pose_mm_deg=value.tolist(), ik_geometrically_reachable=reachable)
            except ValueError as exc:
                joints = [999.0]*4
                self.last_diagnostic.update(ik_geometrically_reachable=False, pose_rejection=str(exc))
            target = torch.tensor([joints], dtype=windows.dtype)/180.0
        if int(logits[:, 0].argmax()) != motion_policy.MOVE:
            target = previous/180.0
        return logits[:, 0], target

    def forward(self, windows, masks):
        raise ValueError("the exact previous applied target is required")


def load_checkpoint(payload, device="cpu"):
    if payload.get("format") != FORMAT or str(device) != "cpu":
        raise ValueError("unsupported recurrent checkpoint or device")
    configuration = payload["configuration"]
    model = ChunkedRecurrentPolicy(horizon=configuration["horizon"],
        center=payload["state_dict"]["center"], scale=payload["state_dict"]["scale"],
        output_space=configuration.get("output_space", "joint"))
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("nonfinite recurrent checkpoint")
    return model.eval()
