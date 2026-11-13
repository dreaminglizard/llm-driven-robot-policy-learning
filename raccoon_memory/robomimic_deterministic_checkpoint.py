"""Separate initialization, recovery and final envelopes for deterministic BC-RNN.

No GMM checkpoint format or official source is modified. Recovery records Adam,
the original sampler and CPU RNG; a final envelope is not a recovery point.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import uuid

import torch

from raccoon_memory.robomimic_deterministic_core import _require_deterministic_core, make_official_deterministic_model
from raccoon_memory.robomimic_scene_sequences import sha256_file
from raccoon_memory.robomimic_scene_training import model_weight_sha256

INITIAL_SCHEMA = "raccoon-robomimic-deterministic-initial.v1"
RESUME_SCHEMA = "raccoon-robomimic-deterministic-resume.v1"
FINAL_SCHEMA = "raccoon-robomimic-deterministic-core.v1"
OFFICIAL_COMMIT = "d309eaecc18acf4152a830a895a6984b8ac71b05"


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("a SHA-256 fingerprint is required")


def optimizer_steps(model):
    if not isinstance(model.optimizers.get("policy"), torch.optim.Adam):
        raise ValueError("the official policy Adam is required")
    result = set()
    for state in model.optimizers["policy"].state.values():
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all():
            raise ValueError("invalid Adam step")
        value = float(step.item())
        if value != int(value):
            raise ValueError("Adam step is not integral")
        result.add(int(value))
    return result


def _check_progress(model, completed):
    _require_deterministic_core(model)
    if type(completed) is not int or completed < 0:
        raise ValueError("invalid completed update count")
    if optimizer_steps(model) != (set() if completed == 0 else {completed}):
        raise ValueError("Adam state does not match completed updates")
    optimizer = model.optimizers["policy"]
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("Adam repeats a policy parameter")
    if completed and set(optimizer.state) != set(parameters):
        raise ValueError("Adam state is missing a policy parameter")
    for parameter, state in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError("Adam moments differ from the policy parameters")


def _check_config(model, config):
    if config.dump() != model.global_config.dump():
        raise ValueError("deterministic model and supplied config differ")


def _atomic_torch_save(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite an existing checkpoint")
    partial = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    try:
        partial.unlink()
    except OSError:
        pass


def save_initial(path, *, model, config, seed, base_config_sha256, dataset_sha256, evidence):
    _check_progress(model, 0)
    _check_config(model, config)
    _sha(base_config_sha256)
    _sha(dataset_sha256)
    if (type(seed) is not int or evidence.get("seed") != seed or evidence.get("actual_updates") != 0
            or evidence.get("optimizer_state_empty") is not True
            or evidence.get("deterministic_after_sha256") != model_weight_sha256(model)
            or config.dump() != model.global_config.dump()):
        raise ValueError("deterministic initialization evidence differs")
    payload = {"schema": INITIAL_SCHEMA, "seed": seed, "base_config_sha256": base_config_sha256,
               "dataset_sha256": dataset_sha256, "official_commit": OFFICIAL_COMMIT,
               "config": config.dump(), "model": model.serialize(), "evidence": copy.deepcopy(evidence),
               "model_weight_sha256": model_weight_sha256(model), "torch_rng_state": torch.get_rng_state()}
    _atomic_torch_save(path, payload)


def load_initial(path, *, model, config, seed, base_config_sha256, dataset_sha256, expected_sha256):
    _check_config(model, config)
    if sha256_file(path) != expected_sha256:
        raise ValueError("prepared deterministic initialization file differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != INITIAL_SCHEMA or payload.get("seed") != seed
            or payload.get("official_commit") != OFFICIAL_COMMIT or payload.get("config") != config.dump()
            or payload.get("base_config_sha256") != base_config_sha256 or payload.get("dataset_sha256") != dataset_sha256):
        raise ValueError("prepared deterministic initialization identity differs")
    model.deserialize(payload["model"], load_optimizers=True)
    _check_progress(model, 0)
    if (model_weight_sha256(model) != payload["model_weight_sha256"]
            or payload["evidence"]["deterministic_after_sha256"] != payload["model_weight_sha256"]
            or sha256_file(path) != expected_sha256):
        raise ValueError("prepared deterministic weights or file changed")
    torch.set_rng_state(payload["torch_rng_state"])
    return payload["evidence"]


def save_recovery(path, *, model, config, sampler, identity, completed, cumulative, metrics):
    _check_progress(model, completed)
    _check_config(model, config)
    payload = {"schema": RESUME_SCHEMA, "identity": copy.deepcopy(identity), "updates_completed": completed,
               "config": config.dump(), "model": model.serialize(), "model_weight_sha256": model_weight_sha256(model),
               "torch_rng_state": torch.get_rng_state(), "sampler_rng_state": sampler.state_dict(),
               "cumulative": copy.deepcopy(cumulative), "block_metrics": copy.deepcopy(metrics)}
    _atomic_torch_save(path, payload)


def load_recovery(path, *, model, config, sampler, identity):
    _check_config(model, config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != RESUME_SCHEMA or payload.get("identity") != identity or payload.get("config") != config.dump():
        raise ValueError("deterministic recovery identity differs")
    model.deserialize(payload["model"], load_optimizers=True)
    _check_progress(model, payload["updates_completed"])
    if model_weight_sha256(model) != payload["model_weight_sha256"]:
        raise ValueError("deterministic recovery weights differ")
    sampler.load_state_dict(payload["sampler_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload


def save_final(path, *, model, config, training_spec_sha256, dataset_sha256, seed, updates):
    _check_progress(model, updates)
    _check_config(model, config)
    _sha(training_spec_sha256)
    _sha(dataset_sha256)
    if type(seed) is not int or updates <= 0 or config.dump() != model.global_config.dump():
        raise ValueError("invalid deterministic final provenance")
    payload = {"schema": FINAL_SCHEMA, "official_commit": OFFICIAL_COMMIT,
               "training_spec_sha256": training_spec_sha256, "dataset_sha256": dataset_sha256,
               "training_seed": seed, "optimizer_updates": updates,
               "observation_key": "scene", "observation_dim": 82, "action_dim": 5,
               "config": config.dump(), "model": model.serialize(), "model_weight_sha256": model_weight_sha256(model)}
    _atomic_torch_save(path, payload)


def load_final(path, *, model, config, training_spec_sha256, dataset_sha256, seed, updates):
    _check_config(model, config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != FINAL_SCHEMA or payload.get("official_commit") != OFFICIAL_COMMIT
            or payload.get("training_spec_sha256") != training_spec_sha256 or payload.get("dataset_sha256") != dataset_sha256
            or payload.get("training_seed") != seed or payload.get("optimizer_updates") != updates
            or payload.get("observation_key") != "scene" or payload.get("observation_dim") != 82
            or payload.get("action_dim") != 5 or payload.get("config") != config.dump()):
        raise ValueError("deterministic final checkpoint identity differs")
    model.deserialize(payload["model"], load_optimizers=True)
    _check_progress(model, updates)
    if model_weight_sha256(model) != payload["model_weight_sha256"]:
        raise ValueError("deterministic final model fingerprint differs")
    return payload


def restore_final(path, *, training_spec_sha256, dataset_sha256, seed, updates, device="cpu"):
    if torch.device(device).type != "cpu":
        raise ValueError("the deterministic evaluation contract is CPU-only")
    digest = sha256_file(path)
    model, config = make_official_deterministic_model(device)
    payload = load_final(path, model=model, config=config, training_spec_sha256=training_spec_sha256,
                         dataset_sha256=dataset_sha256, seed=seed, updates=updates)
    if sha256_file(path) != digest:
        raise ValueError("deterministic final checkpoint changed while loading")
    return model, config, {"schema": payload["schema"], "seed": seed, "optimizer_updates": updates,
                           "checkpoint_sha256": digest, "model_weight_sha256": payload["model_weight_sha256"],
                           "training_spec_sha256": training_spec_sha256, "dataset_sha256": dataset_sha256}
