"""Copy only the verified original zero-update LSTM into deterministic BC-RNN.

This is initialization preparation, not a checkpoint or a training runner.
The original files are read only. The deterministic action head keeps the
official initialization for the requested seed and Adam remains untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from raccoon_memory.robomimic_deterministic_core import (
    _require_deterministic_core,
    make_official_deterministic_model,
)
from raccoon_memory.robomimic_scene_sequences import sha256_file
from raccoon_memory.robomimic_scene_training import RESUME_SCHEMA, model_weight_sha256, state_dict_equal


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "artifacts/architecture_research/20261031_bcrnn_sliding_window_prepared_v1/initial_checkpoint_binding.json"
BINDING_SHA256 = "77dd9fa800c3879445222a516c0f31df8947269289f62936c23a17aef1637d55"
BASE_CONFIG_SHA256 = "736c977497bc4289135fcae0cc53cfbb3c30f8c85a2d9ea06de38d4381b08c55"
DATA_SHA256 = "bbde3f4209b59c851b53cda9e92737d1ba9fbdd8ab1617d3b2585299838b7e27"
INDEX_SHA256 = "bff5a1d933a45922236369e0210ca0bfa45b3fe540f26fcf323d86d4233ea569"
SEEDS = (20261030, 20261031, 20261101)
LSTM_PREFIX = "policy.nets.rnn.nets."
LSTM_NAMES = tuple(LSTM_PREFIX + f"{part}_l{layer}" for layer in (0, 1)
                   for part in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
HEAD_NAMES = ("policy.nets.decoder.nets.action.weight", "policy.nets.decoder.nets.action.bias")


def _tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _verified_original_initial(seed):
    """Read the previously frozen original 0-step checkpoint for this seed."""
    from robomimic.config import config_factory
    from raccoon_memory.robomimic_scene_checkpoint import _check_config

    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the three fixed original seeds")
    if sha256_file(BINDING) != BINDING_SHA256:
        raise ValueError("original initialization binding differs")
    binding = json.loads(BINDING.read_text(encoding="utf-8"))[str(seed)]
    relative = f"artifacts/architecture_research/20261031_robomimic_scene_first_training_prepared_v1/training/seed_{seed}/checkpoints/step_000000.pt"
    if (binding.get("path") != relative or binding.get("completed_updates") != 0
            or binding.get("optimizer_state_empty") is not True):
        raise ValueError("original initialization binding is not a zero-step checkpoint")
    path = ROOT / relative
    digest = sha256_file(path)
    if digest != binding["checkpoint_sha256"]:
        raise ValueError("original zero-step checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise ValueError("original zero-step checkpoint changed while reading")
    identity = payload.get("identity", {})
    if (payload.get("schema") != RESUME_SCHEMA or payload.get("updates_completed") != 0
            or identity.get("seed") != seed or identity.get("training_config_sha256") != BASE_CONFIG_SHA256
            or identity.get("source_data_sha256") != DATA_SHA256 or identity.get("index_manifest_sha256") != INDEX_SHA256
            or payload.get("cumulative", {}).get("updates") != 0
            or payload.get("cumulative", {}).get("sampled_chunks") != 0
            or payload.get("cumulative", {}).get("valid_actions") != 0):
        raise ValueError("original zero-step checkpoint provenance or counts differ")
    _check_config(config_factory("bc", dic=json.loads(payload["config"])))
    model_state = payload["model"]
    if (model_state["optimizers"]["policy"]["state"]
            or payload["model_weight_sha256"] != binding["model_weight_sha256"]
            or _tensor_fingerprint(model_state["nets"]) != binding["model_weight_sha256"]
            or hashlib.sha256(payload["torch_rng_state"].numpy().tobytes()).hexdigest()
            != binding["torch_rng_state_sha256"]):
        raise ValueError("original model, RNG or empty Adam does not match its binding")
    return model_state["nets"], {"seed": seed, "original_checkpoint": relative,
                                  "original_checkpoint_sha256": digest,
                                  "original_model_sha256": binding["model_weight_sha256"],
                                  "binding_sha256": BINDING_SHA256}


def initialize_shared_lstm(model, *, seed):
    """Initialize one fresh, same-seed official deterministic CPU model in place.

    Source and all target compatibility checks precede the first tensor copy.
    The function does not create an optimizer or modify global Torch RNG state.
    It returns evidence that callers can bind to their own experiment manifest.
    """
    _require_deterministic_core(model)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the three fixed original seeds")
    if torch.device(model.device).type != "cpu":
        raise ValueError("the fixed initialization preparation uses the CPU model")
    parameters = dict(model.nets.named_parameters())
    if set(parameters) != set(LSTM_NAMES + HEAD_NAMES):
        raise ValueError("deterministic parameter names differ from the official low-dimensional core")
    optimizer = model.optimizers.get("policy")
    if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
        raise ValueError("a fresh empty official Adam is required")
    group_parameters = [item for group in optimizer.param_groups for item in group["params"]]
    if (len(group_parameters) != len(parameters) or len({id(item) for item in group_parameters}) != len(parameters)
            or {id(item) for item in group_parameters} != {id(item) for item in parameters.values()}
            or any(item.grad is not None or not item.requires_grad for item in parameters.values())):
        raise ValueError("Adam must refer exactly to all fresh trainable parameters")

    # The output head must be the official initial output for this same seed,
    # not just an arbitrary value left untouched by the copy.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        official_initial, _ = make_official_deterministic_model("cpu")
    before = {name: value.detach().clone() for name, value in model.nets.state_dict().items()}
    expected = official_initial.nets.state_dict()
    if before.keys() != expected.keys() or any(not torch.equal(value, expected[name]) for name, value in before.items()):
        raise ValueError("deterministic model is not the same-seed official initial model")
    official_optimizer = official_initial.optimizers["policy"]
    target_name = {id(value): name for name, value in parameters.items()}
    official_name = {id(value): name for name, value in official_initial.nets.named_parameters()}
    target_groups = [[target_name[id(value)] for value in group["params"]] for group in optimizer.param_groups]
    official_groups = [[official_name[id(value)] for value in group["params"]]
                       for group in official_optimizer.param_groups]
    if (official_optimizer.state or not state_dict_equal(optimizer.state_dict(), official_optimizer.state_dict())
            or target_groups != official_groups):
        raise ValueError("Adam settings or parameter order differ from the official initialization")

    source, evidence = _verified_original_initial(seed)
    source_lstm = {name: value for name, value in source.items() if name.startswith(LSTM_PREFIX)}
    target_lstm = {name: value for name, value in before.items() if name.startswith(LSTM_PREFIX)}
    if set(source_lstm) != set(LSTM_NAMES) or set(target_lstm) != set(LSTM_NAMES):
        raise ValueError("original and deterministic LSTM names differ")
    for name in LSTM_NAMES:
        src, dst = source_lstm[name], parameters[name]
        if (not isinstance(src, torch.Tensor) or src.shape != dst.shape or src.dtype != dst.dtype
                or not torch.isfinite(src).all()):
            raise ValueError(f"original and deterministic LSTM tensor differs: {name}")

    parameter_ids = {name: id(value) for name, value in parameters.items()}
    group_ids = [[id(value) for value in group["params"]] for group in optimizer.param_groups]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    before_fingerprint = model_weight_sha256(model)
    head_fingerprint = _tensor_fingerprint({name: before[name] for name in HEAD_NAMES})
    with torch.no_grad():
        for name in LSTM_NAMES:
            parameters[name].copy_(source_lstm[name])
    after = model.nets.state_dict()
    if (any(not torch.equal(after[name], source_lstm[name]) for name in LSTM_NAMES)
            or any(not torch.equal(value, before[name]) for name, value in after.items() if name not in LSTM_NAMES)
            or {name: id(value) for name, value in model.nets.named_parameters()} != parameter_ids
            or [[id(value) for value in group["params"]] for group in optimizer.param_groups] != group_ids
            or not state_dict_equal(optimizer.state_dict(), optimizer_before) or optimizer.state):
        raise RuntimeError("in-place LSTM initialization invariants failed")
    return {**evidence, "lstm_names": list(LSTM_NAMES),
            "lstm_sha256": _tensor_fingerprint(source_lstm),
            "deterministic_before_sha256": before_fingerprint,
            "deterministic_after_sha256": model_weight_sha256(model),
            "official_head_sha256": head_fingerprint,
            "parameter_objects_preserved": True, "optimizer_references_preserved": True,
            "optimizer_state_empty": True, "actual_updates": 0}
