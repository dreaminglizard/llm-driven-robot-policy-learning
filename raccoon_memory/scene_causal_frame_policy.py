"""One causal Markov student with separate action and MOVE regression heads.

This is an experimental HMM/regression policy, not a TP-HSMM reproduction.
Its latent state is updated from robot observations and past applied targets.
No object coordinates, demonstration index, future target, or success is read.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from raccoon_control.kinematics import Geometry
from . import motion_policy, scene_fk_loss


FORMAT = "raccoon-causal-frame-markov-policy.v1"
FRAMES = ("world", "task", "polar")
CONTEXT_COUNT = 51


def geometry_rows(rows):
    xy = rows[:, 74:82].reshape(-1, 4, 2) * 300.0
    batch = torch.arange(len(rows), device=rows.device)
    source = xy[batch, rows[:, 14:18].argmax(1)]
    destination = xy[batch, rows[:, 18:22].argmax(1)]
    difference = destination - source
    length = torch.linalg.vector_norm(difference, dim=1, keepdim=True)
    if bool((length < 1.0).any()):
        raise ValueError("registered source and destination must be distinct")
    axis = difference / length
    normal = torch.stack((-axis[:, 1], axis[:, 0]), dim=1)
    return source, destination, length, axis, normal, xy


def pose(joints):
    xyz, _ = scene_fk_loss.nominal_forward(joints)
    return torch.cat((xyz, (joints[:, 1:].sum(1) + 90.0).unsqueeze(1)), dim=1)


def encode_pose(values, rows, frame):
    if frame == "world":
        return values / values.new_tensor((300.0, 300.0, 100.0, 180.0))
    if frame == "polar":
        source, _, _, _, _, _ = geometry_rows(rows)
        source_yaw = torch.atan2(-source[:, 0], source[:, 1])
        yaw = torch.atan2(-values[:, 0], values[:, 1])
        radius = torch.linalg.vector_norm(values[:, :2], dim=1)
        source_radius = torch.linalg.vector_norm(source, dim=1)
        return torch.stack(((yaw-source_yaw)/math.pi, (radius-source_radius)/100.0,
                            values[:, 2]/100.0, values[:, 3]/180.0), dim=1)
    if frame != "task":
        raise ValueError("unknown coordinate frame")
    source, _, length, axis, normal, _ = geometry_rows(rows)
    difference = values[:, :2] - source
    return torch.stack(((difference * axis).sum(1) / length[:, 0],
                        (difference * normal).sum(1) / length[:, 0],
                        values[:, 2] / 100.0, values[:, 3] / 180.0), dim=1)


def decode_pose(values, row, frame):
    if frame == "world":
        return np.asarray(values, dtype=np.float64) * (300.0, 300.0, 100.0, 180.0)
    if frame == "polar":
        source, _, _, _, _, _ = geometry_rows(row.unsqueeze(0))
        source = source[0].numpy()
        yaw = math.atan2(-source[0], source[1]) + float(values[0])*math.pi
        radius = float(np.linalg.norm(source)) + float(values[1])*100.0
        if radius <= 1e-6 or abs(yaw) > math.pi:
            raise ValueError("polar regression predicts an invalid radius or yaw branch")
        return np.asarray([-radius*math.sin(yaw), radius*math.cos(yaw), values[2]*100.0, values[3]*180.0])
    source, _, length, axis, normal, _ = geometry_rows(row.unsqueeze(0))
    xy = source[0].numpy() + float(length[0, 0]) * (values[0] * axis[0].numpy() + values[1] * normal[0].numpy())
    return np.asarray([*xy, values[2] * 100.0, values[3] * 180.0])


def context(rows, previous, frame):
    if (rows.ndim != 2 or rows.shape[1] != 82 or previous.shape != (len(rows), 4)
            or not bool(torch.isfinite(rows).all() and torch.isfinite(previous).all())):
        raise ValueError("finite current scene features and applied targets are required")
    rows, previous = rows.to(torch.float64), previous.to(torch.float64)
    source, destination, length, axis, normal, xy = geometry_rows(rows)
    if frame == "world":
        slots = xy.reshape(-1, 8) / 300.0
        goals = torch.cat((source, destination), dim=1) / 300.0
        base = rows.new_zeros((len(rows), 2))
    elif frame == "task":
        differences = xy - source.unsqueeze(1)
        slots = torch.stack(((differences * axis.unsqueeze(1)).sum(2) / length,
                             (differences * normal.unsqueeze(1)).sum(2) / length), dim=2).reshape(-1, 8)
        goals = rows.new_tensor((0.0, 0.0, 1.0, 0.0)).expand(len(rows), -1)
        base = torch.stack(((-source * axis).sum(1), (-source * normal).sum(1)), dim=1) / length
    elif frame == "polar":
        yaw = torch.atan2(-xy[:, :, 0], xy[:, :, 1])
        radius = torch.linalg.vector_norm(xy, dim=2)
        source_yaw = torch.atan2(-source[:, 0], source[:, 1])
        source_radius = torch.linalg.vector_norm(source, dim=1)
        slots = torch.stack(((yaw-source_yaw[:, None])/math.pi,
                              (radius-source_radius[:, None])/100.0), dim=2).reshape(-1, 8)
        destination_yaw = torch.atan2(-destination[:, 0], destination[:, 1])
        destination_radius = torch.linalg.vector_norm(destination, dim=1)
        goals = torch.stack((torch.zeros_like(source_yaw), torch.zeros_like(source_yaw),
                             (destination_yaw-source_yaw)/math.pi, (destination_radius-source_radius)/100.0), dim=1)
        base = torch.stack((source_yaw/math.pi, source_radius/300.0), dim=1)
    else:
        raise ValueError("unknown coordinate frame")
    occupancy = (1.0 - rows[:, 30:70].reshape(-1, 4, 2, 5)[:, :, :, 0]).sum(2) / 2.0
    levels = torch.stack((rows[:, 22:24].argmax(1) + 1, rows[:, 24:26].argmax(1) + 1), dim=1) / 2.0
    current_pose = pose(rows[:, :4] * 180.0)
    heights = (levels * 2.0 - 0.5) * 20.0
    source_xyz = torch.cat((source, heights[:, 0:1]), dim=1)
    destination_xyz = torch.cat((destination, heights[:, 1:2]), dim=1)
    offsets = torch.stack((current_pose[:, :3] - source_xyz, current_pose[:, :3] - destination_xyz), dim=1)
    if frame in ("task", "polar"):
        offsets = torch.stack(((offsets[:, :, :2] * axis.unsqueeze(1)).sum(2),
                               (offsets[:, :, :2] * normal.unsqueeze(1)).sum(2), offsets[:, :, 2]), dim=2)
    result = torch.cat((encode_pose(pose(rows[:, :4] * 180.0), rows, frame),
                        encode_pose(pose(previous), rows, frame), rows[:, :4], previous / 180.0,
                        rows[:, 6:14], levels, slots, occupancy, base, length / 300.0, goals,
                        offsets.reshape(-1, 6) / 100.0), dim=1)
    if result.shape != (len(rows), CONTEXT_COUNT) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid causal geometric context")
    return result


def move_labels(data, frame):
    """No non-MOVE target is evaluated, transformed, or fitted."""
    rows = data["windows"][:, -1].to(torch.float64)
    selected = data["actions"] == motion_policy.MOVE
    result = torch.full((len(rows), 4), float("nan"), dtype=torch.float64)
    result[selected] = encode_pose(pose(data["targets"][selected].to(torch.float64) * 180.0), rows[selected], frame)
    return result


def logsumexp(values, axis=None):
    largest = np.max(values, axis=axis, keepdims=True)
    result = largest + np.log(np.exp(values - largest).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def filter_step(log_previous, log_initial, log_transition, log_observation):
    prior = log_initial if log_previous is None else logsumexp(log_previous[:, None] + log_transition, axis=0)
    posterior = prior + log_observation
    return posterior - logsumexp(posterior)


def forward_backward(emission, log_initial, log_transition):
    """Training-only smoother; runtime uses filter_step without labels/future."""
    n, k = emission.shape
    alpha, beta = np.empty((n, k)), np.zeros((n, k))
    alpha[0] = log_initial + emission[0]
    for i in range(1, n):
        alpha[i] = emission[i] + logsumexp(alpha[i-1, :, None] + log_transition, axis=0)
    likelihood = float(logsumexp(alpha[-1]))
    for i in range(n-2, -1, -1):
        beta[i] = logsumexp(log_transition + emission[i+1][None, :] + beta[i+1][None, :], axis=1)
    gamma = np.exp(alpha + beta - likelihood)
    gamma /= gamma.sum(axis=1, keepdims=True)
    transitions = np.zeros((k, k))
    for i in range(n-1):
        xi = np.exp(alpha[i, :, None] + log_transition + emission[i+1][None, :] + beta[i+1][None, :] - likelihood)
        transitions += xi / xi.sum()
    return gamma, transitions, likelihood


def gaussian_log_density(values, mean, variance):
    return -0.5 * (np.log(2.0 * math.pi * variance) + (values - mean) ** 2 / variance).sum(axis=-1)


def inverse_or_reject(target):
    """Return nominal IK or a finite out-of-range proposal; no forced projection."""
    x, y, z, pitch_deg = (float(value) for value in target)
    geometry = Geometry()
    pitch, radius = math.radians(pitch_deg), math.hypot(x, y)
    arm_r = radius - geometry.tool_forward_mm * math.cos(pitch) + geometry.tool_vertical_mm * math.sin(pitch)
    arm_z = z - geometry.base_mm - geometry.tool_forward_mm * math.sin(pitch) - geometry.tool_vertical_mm * math.cos(pitch)
    cosine = (arm_r * arm_r + arm_z * arm_z - geometry.upper_mm ** 2 - geometry.forearm_mm ** 2) / (2 * geometry.upper_mm * geometry.forearm_mm)
    if radius < 1e-6 or not math.isfinite(cosine) or not -1.0-1e-10 <= cosine <= 1.0+1e-10:
        return [999.0] * 4, False
    q3 = -math.acos(max(-1.0, min(1.0, cosine)))
    q2 = math.atan2(-arm_r, arm_z) - math.atan2(geometry.forearm_mm * math.sin(q3), geometry.upper_mm + geometry.forearm_mm * math.cos(q3))
    q1 = math.atan2(-x, y)
    q4 = pitch - math.pi/2 - q2 - q3
    result = [math.degrees(value) for value in (q1, q2, q3, q4)]
    return result, True


class CausalFramePolicy(nn.Module):
    history_length, feature_count = 8, 82

    def __init__(self, payload):
        super().__init__()
        if payload.get("format") != FORMAT or payload.get("frame") not in FRAMES:
            raise ValueError("unsupported causal frame checkpoint")
        self.frame = payload["frame"]
        values = payload["parameters"]
        for key in ("center", "scale", "initial", "transition", "mean", "variance", "action", "regression", "move_support"):
            tensor = values[key].detach().clone().to(torch.float64)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("nonfinite learned parameters")
            self.register_buffer(key, tensor)
        k = len(self.initial)
        if (self.center.shape != (CONTEXT_COUNT,) or self.scale.shape != (CONTEXT_COUNT,)
                or self.mean.shape != (k, CONTEXT_COUNT) or self.variance.shape != self.mean.shape
                or self.transition.shape != (k, k) or self.action.shape != (k, 4)
                or self.regression.shape != (k, CONTEXT_COUNT+1, 4) or self.move_support.shape != (k,)
                or bool((self.scale <= 0).any() or (self.variance <= 0).any())
                or bool((self.initial <= 0).any() or (self.transition <= 0).any() or (self.action <= 0).any())
                or not torch.allclose(self.initial.sum(), self.initial.new_tensor(1.0), atol=1e-8, rtol=0)
                or not torch.allclose(self.transition.sum(1), self.initial.new_ones(k), atol=1e-8, rtol=0)
                or not torch.allclose(self.action.sum(1), self.initial.new_ones(k), atol=1e-8, rtol=0)):
            raise ValueError("invalid causal parameter shapes or normalized probabilities")
        self._log_alpha = None
        self.last_diagnostic = None

    def forward_with_control_state(self, windows, masks, previous):
        if (windows.shape != (1, 8, 82) or masks.shape != (1, 8) or masks.dtype != torch.bool
                or not bool(masks[0, -1]) or previous.shape != (1, 4) or windows.device.type != "cpu"):
            raise ValueError("causal policy expects one CPU episode and the existing causal history contract")
        valid = int(masks.sum().item())
        if masks[0].tolist() != [False] * (8-valid) + [True] * valid:
            raise ValueError("invalid history validity mask")
        if valid == 1:
            self._log_alpha = None
        elif self._log_alpha is None:
            raise ValueError("the causal state must start at the transfer's first observation")
        row = windows[:, -1].to(torch.float64)
        raw = context(row, previous.to(torch.float64), self.frame).numpy()[0]
        x = (raw - self.center.numpy()) / self.scale.numpy()
        likelihood = gaussian_log_density(x[None, :], self.mean.numpy(), self.variance.numpy())
        self._log_alpha = filter_step(self._log_alpha, np.log(self.initial.numpy()),
                                      np.log(self.transition.numpy()), likelihood)
        posterior = np.exp(self._log_alpha)
        action_probability = posterior @ self.action.numpy()
        action = int(action_probability.argmax())
        state = int((posterior * self.action.numpy()[:, motion_policy.MOVE]).argmax())
        angles = previous[0].tolist()
        reachable, target = None, None
        rejection = None
        if action == motion_policy.MOVE:
            if self.move_support[state] <= 1e-3:
                angles, reachable = [999.0]*4, False
            else:
                encoded = np.r_[1.0, x] @ self.regression[state].numpy()
                try:
                    target = decode_pose(encoded, row[0], self.frame)
                    angles, reachable = inverse_or_reject(target)
                except ValueError as exc:
                    angles, reachable, rejection = [999.0]*4, False, str(exc)
        self.last_diagnostic = {"posterior": posterior.tolist(), "action_probability": action_probability.tolist(),
                                "move_component": state, "nominal_target_pose_mm_deg": target.tolist() if target is not None else None,
                                "ik_geometrically_reachable": reachable, "pose_rejection": rejection}
        return (torch.tensor(np.log(np.maximum(action_probability, 1e-30))[None], dtype=windows.dtype),
                torch.tensor([angles], dtype=windows.dtype) / 180.0)

    def forward(self, windows, masks):
        raise ValueError("the exact previous applied target is required")


def load_checkpoint(payload, device="cpu"):
    if str(device) != "cpu":
        raise ValueError("this causal prototype is CPU-only")
    return CausalFramePolicy(payload).eval()
