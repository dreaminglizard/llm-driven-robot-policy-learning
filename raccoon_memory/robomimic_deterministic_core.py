"""Isolated official low-dimensional MG BC-RNN and valid-action L2 adapter.

Import does not create a model, read a dataset, update weights, or run physics.
This candidate is not the existing GMM checkpoint or completion-head format.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_memory.robomimic_completion import check_official_source
from raccoon_memory.robomimic_scene_sequences import ACTION_COUNT, FEATURE_COUNT, HORIZON, OBS_KEY


PAPER_CONFIG_SHA256 = "bb1ccfdc2c13b541533adc54e9bb3579aac351a17a3bd5bf461158f4953c1dea"


def _require_deterministic_core(model):
    from robomimic.algo.bc import BC_RNN

    # BC_RNN_GMM subclasses BC_RNN, so an isinstance check is insufficient.
    if type(model) is not BC_RNN:
        raise TypeError("expected the exact official deterministic BC_RNN")
    if (model._rnn_is_open_loop is not False or model._rnn_horizon != HORIZON
            or model.ac_dim != ACTION_COUNT or set(model.obs_key_shapes) != {OBS_KEY}
            or tuple(model.obs_key_shapes[OBS_KEY]) != (FEATURE_COUNT,)
            or model.algo_config.gmm.enabled or model.algo_config.gaussian.enabled or model.algo_config.vae.enabled
            or model.algo_config.loss.l2_weight != 1.0 or model.algo_config.loss.l1_weight != 0.0
            or model.algo_config.loss.cos_weight != 0.0):
        raise ValueError("deterministic recurrent observation or action contract differs")
    lstms = [layer for layer in model.nets.modules() if isinstance(layer, nn.LSTM)]
    if len(lstms) != 1 or lstms[0].hidden_size != 400 or lstms[0].num_layers != 2 or lstms[0].bidirectional:
        raise ValueError("expected the official two-layer unidirectional 400-unit LSTM")


def make_official_deterministic_model(device: torch.device | str = "cpu"):
    """Instantiate the official paper's low-dimensional machine-generated BC_RNN."""
    from robomimic.algo import algo_factory
    from robomimic.config import config_factory
    from robomimic.scripts import generate_paper_configs
    import robomimic.utils.obs_utils as obs_utils

    check_official_source()
    if hashlib.sha256(Path(generate_paper_configs.__file__).read_bytes()).hexdigest() != PAPER_CONFIG_SHA256:
        raise ValueError("official paper configuration source differs")
    config = config_factory(algo_name="bc")
    # The task name is used by the official generator only to check real/raw data.
    config = generate_paper_configs.modify_bc_rnn_config_for_dataset(
        config, task_name="lift", dataset_type="mg", hdf5_type="low_dim")
    config.train.cuda = torch.device(device).type == "cuda"
    config.observation.modalities.obs.low_dim = [OBS_KEY]
    config.observation.modalities.obs.rgb = []
    config.observation.modalities.obs.depth = []
    config.observation.modalities.obs.scan = []
    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []
    if (config.algo.gmm.enabled or config.algo.gaussian.enabled or config.algo.vae.enabled
            or not config.algo.rnn.enabled or config.train.seq_length != HORIZON
            or config.algo.rnn.horizon != HORIZON or config.algo.rnn.hidden_dim != 400
            or config.algo.rnn.rnn_type != "LSTM" or config.algo.rnn.num_layers != 2
            or config.algo.rnn.open_loop or config.algo.rnn.kwargs.bidirectional
            or list(config.algo.actor_layer_dims) != []
            or config.algo.loss.l2_weight != 1.0 or config.algo.loss.l1_weight != 0.0
            or config.algo.loss.cos_weight != 0.0
            or config.algo.optim_params.policy.learning_rate.initial != 1e-4):
        raise ValueError("official low-dimensional MG BC_RNN configuration differs")
    obs_utils.initialize_obs_utils_with_config(config)
    model = algo_factory("bc", config=config, obs_key_shapes=OrderedDict([(OBS_KEY, (FEATURE_COUNT,))]),
                         ac_dim=ACTION_COUNT, device=torch.device(device))
    _require_deterministic_core(model)
    return model, config


def masked_action_l2(model, batch: dict):
    """Mean squared error over five outputs at action-supervised positions only.

    The recurrent network receives all real context observations. Prefix and
    terminal observations have no action loss; right padding has neither.
    Return None for an empty mask, before any policy forward call.
    """
    _require_deterministic_core(model)
    if not isinstance(batch, dict) or not isinstance(batch.get("obs"), dict) or set(batch["obs"]) != {OBS_KEY}:
        raise ValueError("the only observation key must be scene")
    obs, actions = batch["obs"][OBS_KEY], batch.get("actions")
    mask, obs_mask = batch.get("action_mask"), batch.get("observation_mask")
    if (not isinstance(obs, torch.Tensor) or not obs.is_floating_point() or obs.ndim != 3 or obs.shape[0] == 0
            or tuple(obs.shape[1:]) != (HORIZON, FEATURE_COUNT)
            or not isinstance(actions, torch.Tensor) or not actions.is_floating_point()
            or tuple(actions.shape) != (*obs.shape[:2], ACTION_COUNT)
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(obs.shape[:2])
            or not isinstance(obs_mask, torch.Tensor) or obs_mask.dtype != torch.bool or tuple(obs_mask.shape) != tuple(mask.shape)
            or not (obs.device == actions.device == mask.device == obs_mask.device)):
        raise ValueError("observation, target or supervision mask shape, type or device differs")
    if torch.any(mask & ~obs_mask) or torch.any((~obs_mask[:, :-1]) & obs_mask[:, 1:]):
        raise ValueError("padding cannot be supervised and must be on the right")
    if not torch.isfinite(obs).all() or not torch.isfinite(actions[mask]).all():
        raise ValueError("observations and supervised action targets must be finite")
    count = int(mask.sum().item())
    if count == 0:
        return None, {"valid_action_count": 0, "skipped": True}
    safe_actions = torch.where(mask.unsqueeze(-1), actions, torch.zeros_like(actions))
    processed = model.process_batch_for_training({"obs": batch["obs"], "actions": safe_actions})
    predicted = model.nets["policy"](obs_dict=processed["obs"], goal_dict=processed["goal_obs"])
    if tuple(predicted.shape) != tuple(processed["actions"].shape):
        raise ValueError("official deterministic action shape differs")
    selected_mask = mask.to(device=predicted.device)
    selected = predicted[selected_mask]
    target = processed["actions"][selected_mask]
    if not torch.isfinite(selected).all() or not torch.isfinite(target).all():
        raise ValueError("supervised deterministic predictions or targets are nonfinite")
    loss = F.mse_loss(selected, target, reduction="mean")
    if not torch.isfinite(loss):
        raise ValueError("masked deterministic L2 loss is nonfinite")
    return loss, {"valid_action_count": count, "skipped": False}


@dataclass(frozen=True)
class DeterministicRNNDecision:
    action: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor
    decision_count: int


class DeterministicRNNObserver:
    """Copy the state after exactly one original deterministic get_action call."""

    def __init__(self, model):
        _require_deterministic_core(model)
        self.model = model
        self._needs_reset = True
        self._counter = 0

    def reset(self):
        self._needs_reset = True
        self.model.reset()
        self._counter = 0
        self._needs_reset = False

    def step(self, observation: torch.Tensor) -> DeterministicRNNDecision:
        if self._needs_reset:
            raise RuntimeError("reset the deterministic recurrent core before a new episode or after an error")
        if self.model.nets.training or self.model._rnn_is_open_loop is not False:
            raise RuntimeError("deterministic recurrent core must be eval and closed-loop")
        try:
            if (not isinstance(observation, torch.Tensor) or not observation.is_floating_point()
                    or tuple(observation.shape) != (1, FEATURE_COUNT) or not torch.isfinite(observation).all()):
                raise ValueError("current observation must be a finite [1,82] tensor")
            if self.model._rnn_counter != self._counter:
                raise RuntimeError("deterministic recurrent core was advanced outside this observer")
            current = observation.detach().to(device=self.model.device, dtype=torch.float32)
            with torch.no_grad():
                action = self.model.get_action({OBS_KEY: current})
                state = self.model._rnn_hidden_state
            if (self.model._rnn_counter != self._counter + 1 or not isinstance(action, torch.Tensor)
                    or tuple(action.shape) != (1, ACTION_COUNT) or not torch.isfinite(action).all()
                    or not isinstance(state, tuple) or len(state) != 2
                    or any(not isinstance(x, torch.Tensor) or tuple(x.shape) != (2, 1, 400)
                           or not torch.isfinite(x).all() for x in state)):
                raise RuntimeError("official deterministic action or post-observation state differs")
            self._counter += 1
            return DeterministicRNNDecision(action.detach().clone(), state[0].detach().clone(),
                                            state[1].detach().clone(), self._counter)
        except Exception:
            self._needs_reset = True
            raise
