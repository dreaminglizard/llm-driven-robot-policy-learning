"""Scene-core checkpoint envelope around official robomimic serialization.

The envelope does not invent official environment metadata or include a finish
model. Optimizer state is stored; exact training resumption also needs the
caller's sampler position and random-state bookkeeping.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

import torch

from raccoon_memory.robomimic_scene_sequences import ACTION_COUNT, EXPECTED_DATA_SHA256, FEATURE_COUNT, HORIZON, OBS_KEY

SCHEMA = "raccoon-robomimic-scene-core.v1"
OFFICIAL_COMMIT = "d309eaecc18acf4152a830a895a6984b8ac71b05"


def _check_official_source():
    import robomimic
    root = Path(robomimic.__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    if git("rev-parse", "HEAD") != OFFICIAL_COMMIT or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("loaded robomimic source is not the fixed unmodified commit")


def _check_config(config):
    if (config.algo_name != "bc" or not config.algo.gmm.enabled or not config.algo.rnn.enabled
            or config.algo.gmm.num_modes != 5 or not config.algo.gmm.low_noise_eval
            or config.algo.gmm.min_std != 0.0001 or config.algo.gmm.std_activation != "softplus"
            or config.algo.rnn.rnn_type != "LSTM" or config.algo.rnn.hidden_dim != 400
            or config.algo.rnn.num_layers != 2 or config.algo.rnn.horizon != HORIZON
            or config.algo.rnn.open_loop or config.algo.rnn.kwargs.bidirectional or config.train.seq_length != HORIZON
            or list(config.algo.actor_layer_dims) != []
            or config.algo.optim_params.policy.optimizer_type != "adam"
            or config.algo.optim_params.policy.learning_rate.initial != 1e-4
            or list(config.observation.modalities.obs.low_dim) != [OBS_KEY]
            or any(config.observation.modalities.obs[key] for key in config.observation.modalities.obs if key != "low_dim")
            or any(config.observation.modalities.goal[key] for key in config.observation.modalities.goal)):
        raise ValueError("checkpoint config differs from the fixed official scene core")


def save_scene_core_checkpoint(path: Path | str, model, config, *, dataset_sha256: str,
                               training_spec_sha256: str, training_seed: int, optimizer_updates: int):
    """Write a new checkpoint, refusing to overwrite an existing file."""
    from robomimic.algo.bc import BC_RNN_GMM

    _check_official_source()
    _check_config(config)
    if not isinstance(model, BC_RNN_GMM) or model.nets["policy"].use_tanh:
        raise ValueError("expected the official BC_RNN_GMM with use_tanh=False")
    if json.loads(config.dump()) != json.loads(model.global_config.dump()):
        raise ValueError("supplied config differs from the model config")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in (dataset_sha256, training_spec_sha256)):
        raise ValueError("fixed dataset and training specification SHA-256 values are required")
    if type(training_seed) is not int or type(optimizer_updates) is not int or optimizer_updates < 0:
        raise ValueError("training seed and nonnegative completed update count are required")
    observed_steps = {int(state["step"].item()) for state in model.optimizers["policy"].state.values() if "step" in state}
    if observed_steps != (set() if optimizer_updates == 0 else {optimizer_updates}):
        raise ValueError("declared updates differ from the official Adam state")
    payload = {"schema": SCHEMA, "official_commit": OFFICIAL_COMMIT,
               "dataset_sha256": dataset_sha256, "training_spec_sha256": training_spec_sha256,
               "training_seed": training_seed, "optimizer_updates": optimizer_updates,
               "algo_name": "bc", "observation_key": OBS_KEY, "observation_dim": FEATURE_COUNT,
               "action_dim": ACTION_COUNT, "config": config.dump(), "model": model.serialize()}
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint already exists")
    partial = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def load_scene_core_checkpoint(path: Path | str, *, device: torch.device | str = "cpu",
                               load_optimizers: bool = True, expected_spec_sha256: str | None = None,
                               expected_dataset_sha256: str = EXPECTED_DATA_SHA256):
    """Return the original official model, its config and the saved provenance."""
    from robomimic.algo import algo_factory
    from robomimic.algo.bc import BC_RNN_GMM
    from robomimic.config import config_factory
    import robomimic.utils.obs_utils as ObsUtils

    _check_official_source()
    payload = torch.load(Path(path), map_location=torch.device(device), weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("official_commit") != OFFICIAL_COMMIT
            or payload.get("dataset_sha256") != expected_dataset_sha256
            or payload.get("algo_name") != "bc" or payload.get("observation_key") != OBS_KEY
            or payload.get("observation_dim") != FEATURE_COUNT or payload.get("action_dim") != ACTION_COUNT):
        raise ValueError("checkpoint provenance or shape contract differs")
    if expected_spec_sha256 is not None and payload.get("training_spec_sha256") != expected_spec_sha256:
        raise ValueError("checkpoint training specification differs")
    config = config_factory("bc", dic=json.loads(payload["config"]))
    _check_config(config)
    ObsUtils.initialize_obs_utils_with_config(config)
    model = algo_factory("bc", config=config, obs_key_shapes=OrderedDict([(OBS_KEY, (FEATURE_COUNT,))]),
                         ac_dim=ACTION_COUNT, device=torch.device(device))
    if not isinstance(model, BC_RNN_GMM) or model.nets["policy"].use_tanh:
        raise ValueError("restored model does not use the fixed official core")
    model.deserialize(payload["model"], load_optimizers=load_optimizers)
    provenance = {key: payload[key] for key in ("schema", "official_commit", "dataset_sha256",
                  "training_spec_sha256", "training_seed", "optimizer_updates")}
    return model, config, provenance
