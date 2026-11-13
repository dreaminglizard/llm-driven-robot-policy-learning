"""Function-preserving widening of the existing scene MLP, with trainable new paths.

This only constructs a policy; it neither loads data nor runs a simulator. The
original 64 units occupy the first half of each 128-unit layer. Initially,
connections from new units into old units and the output heads are zero, so
the extra paths do not affect the original function. New units themselves
have distinct, seeded random weights. The zero connections can acquire a
gradient immediately; gradients then also reach the new units' incoming
weights in later updates. No optimizer or optimizer state is copied.

The two functions are mathematically equal at initialization. Floating-point
matrix multiplication can differ slightly with the larger matrix shape, so
callers must check numerical output, action, and limit decisions on their
actual inputs, particularly near ties or joint limits.
"""
from __future__ import annotations

import math

import torch

from raccoon_memory import scene_policy


SOURCE_WIDTH = 64
TARGET_WIDTH = 128
HISTORY_LENGTH = 8
FEATURE_COUNT = 82


def widen_scene_policy(source: scene_policy.ScenePolicy, *, seed: int) -> scene_policy.ScenePolicy:
    """Return an independently allocated MLP128 initialized from an MLP64.

    ``seed`` controls only the new units using a private CPU generator; the
    caller's global Torch random stream is unchanged. Source device, dtype,
    and train/eval mode are retained. All returned parameters are trainable.
    The caller is responsible for checkpoint provenance and a fresh optimizer.
    """
    if (type(source) is not scene_policy.ScenePolicy
            or (source.architecture, source.hidden_size, source.history_length, source.feature_count)
            != ("mlp", SOURCE_WIDTH, HISTORY_LENGTH, FEATURE_COUNT)):
        raise ValueError("source must be the existing 82-feature, history-8 MLP64 ScenePolicy")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 to 2**63 - 1")

    old = source.state_dict()
    expected_shapes = {
        "encoder.0.weight": (64, 8 * 83), "encoder.0.bias": (64,),
        "encoder.2.weight": (64, 64), "encoder.2.bias": (64,),
        "action_head.weight": (4, 64), "action_head.bias": (4,),
        "joint_head.weight": (4, 64), "joint_head.bias": (4,),
    }
    if set(old) != set(expected_shapes):
        raise ValueError("source parameter names differ from the MLP64 contract")
    for name, shape in expected_shapes.items():
        value = old[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or value.device.type == "meta"
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"source parameter shape or finite floating value differs: {name}")
    devices = {value.device for value in old.values()}
    dtypes = {value.dtype for value in old.values()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("source parameters must share a device and dtype")

    # Construction initializes on the CPU, so isolate its draws as well as
    # using a local generator for the actual new-path initialization.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        target = scene_policy.ScenePolicy("mlp", TARGET_WIDTH, HISTORY_LENGTH)
    target = target.to(device=next(iter(devices)), dtype=next(iter(dtypes)))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def random_uniform(shape: tuple[int, ...], fan_in: int) -> torch.Tensor:
        bound = 1.0 / math.sqrt(fan_in)
        return torch.empty(shape, dtype=torch.float32, device="cpu").uniform_(
            -bound, bound, generator=generator)

    with torch.no_grad():
        first, second = target.encoder[0], target.encoder[2]
        first.weight[:64].copy_(old["encoder.0.weight"])
        first.bias[:64].copy_(old["encoder.0.bias"])
        first.weight[64:].copy_(random_uniform((64, 8 * 83), 8 * 83))
        first.bias[64:].copy_(random_uniform((64,), 8 * 83))

        second.weight[:64, :64].copy_(old["encoder.2.weight"])
        second.weight[:64, 64:].zero_()
        second.bias[:64].copy_(old["encoder.2.bias"])
        second.weight[64:, :].copy_(random_uniform((64, 128), 128))
        second.bias[64:].copy_(random_uniform((64,), 128))

        for name, head in (("action_head", target.action_head), ("joint_head", target.joint_head)):
            head.weight[:, :64].copy_(old[f"{name}.weight"])
            head.weight[:, 64:].zero_()
            head.bias.copy_(old[f"{name}.bias"])
    target.train(source.training)
    return target
