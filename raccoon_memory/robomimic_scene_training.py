"""Training-batch and recoverable-state helpers for the fixed scene BC run."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid

import numpy as np
import torch


RESUME_SCHEMA = "raccoon-robomimic-scene-resume.v1"


def state_dict_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            state_dict_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(
            state_dict_equal(a, b) for a, b in zip(first, second))
    return first == second


def batch_from_rows(index: dict, rows: np.ndarray) -> dict:
    """Read only the already selected episode-aligned chunks from train index."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0 or np.any(rows < 0) or np.any(rows >= len(index["actions"])):
        raise ValueError("invalid selected chunk rows")
    def tensor(name):
        # A writable copy avoids exposing a read-only mmap to torch.
        return torch.from_numpy(np.array(index[name][rows], copy=True))
    return {"obs": {"scene": tensor("observations")}, "actions": tensor("actions"),
            "action_mask": tensor("action_mask"), "observation_mask": tensor("observation_mask")}


def install_finite_gradient_guard(model):
    """Reject nonfinite gradients during the original backward, before Adam."""
    def check(gradient):
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite policy gradient; optimizer step not executed")
        return gradient
    return [parameter.register_hook(check) for parameter in model.nets.parameters() if parameter.requires_grad]


def model_weight_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.nets.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_resume_checkpoint(path: Path | str, *, model, config, sampler_state: dict,
                           identity: dict, updates_completed: int, cumulative: dict, block_metrics: list):
    """Atomically publish a new checkpoint; interrupted partials are retained."""
    path = Path(path)
    if type(updates_completed) is not int or updates_completed < 0:
        raise ValueError("invalid progress")
    steps = {int(value["step"].item()) for value in model.optimizers["policy"].state.values() if "step" in value}
    if steps != (set() if updates_completed == 0 else {updates_completed}):
        raise ValueError("optimizer state and progress disagree")
    payload = {"schema": RESUME_SCHEMA, "identity": identity, "updates_completed": updates_completed,
               "model": model.serialize(), "config": config.dump(), "torch_rng_state": torch.get_rng_state(),
               "sampler_rng_state": sampler_state, "cumulative": cumulative, "block_metrics": block_metrics,
               "model_weight_sha256": model_weight_sha256(model)}
    if path.exists():
        raise FileExistsError("checkpoint already exists")
    partial = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # A hard link publishes atomically without replacing a concurrent file.
    os.link(partial, path)
    partial.unlink()


def load_resume_checkpoint(path: Path | str, *, model, config, expected_identity: dict, sampler):
    """Restore official model, optimizer, model RNG and sampler RNG exactly."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("schema") != RESUME_SCHEMA or payload.get("identity") != expected_identity
            or payload.get("config") != config.dump()):
        raise ValueError("checkpoint run identity or model config differs")
    model.deserialize(payload["model"], load_optimizers=True)
    if model_weight_sha256(model) != payload["model_weight_sha256"]:
        raise ValueError("restored model fingerprint differs")
    steps = {int(value["step"].item()) for value in model.optimizers["policy"].state.values() if "step" in value}
    expected_steps = set() if payload["updates_completed"] == 0 else {payload["updates_completed"]}
    if steps != expected_steps:
        raise ValueError("restored optimizer progress differs")
    sampler.load_state_dict(payload["sampler_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload
