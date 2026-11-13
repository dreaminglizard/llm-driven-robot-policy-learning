"""Masked action NLL around the original robomimic BC_RNN_GMM implementation.

The original policy, distribution, optimizer, and optimizer-step code are used
unchanged. Only which action log probabilities enter the mean is adapted.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.distributions as D

from raccoon_memory.robomimic_scene_sequences import ACTION_COUNT, FEATURE_COUNT, HORIZON, OBS_KEY


def make_official_model(device: torch.device | str = "cpu"):
    """Instantiate the fixed official robosuite-example core for 82 inputs and 5 actions."""
    from robomimic.algo import algo_factory
    from robomimic.algo.bc import BC_RNN_GMM
    from robomimic.config import config_factory
    import robomimic.utils.obs_utils as ObsUtils

    config = config_factory(algo_name="bc")
    config.train.seq_length = HORIZON
    config.train.cuda = torch.device(device).type == "cuda"
    config.algo.actor_layer_dims = ()
    config.algo.gmm.enabled = True
    config.algo.gmm.num_modes = 5
    config.algo.gmm.min_std = 0.0001
    config.algo.gmm.std_activation = "softplus"
    config.algo.gmm.low_noise_eval = True
    config.algo.rnn.enabled = True
    config.algo.rnn.horizon = HORIZON
    config.algo.rnn.hidden_dim = 400
    config.algo.rnn.rnn_type = "LSTM"
    config.algo.rnn.num_layers = 2
    config.algo.rnn.open_loop = False
    config.algo.rnn.kwargs.bidirectional = False
    config.algo.optim_params.policy.learning_rate.initial = 1e-4
    config.observation.modalities.obs.low_dim = [OBS_KEY]
    config.observation.modalities.obs.rgb = []
    config.observation.modalities.obs.depth = []
    config.observation.modalities.obs.scan = []
    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []
    ObsUtils.initialize_obs_utils_with_config(config)
    model = algo_factory("bc", config=config, obs_key_shapes=OrderedDict([(OBS_KEY, (FEATURE_COUNT,))]),
                         ac_dim=ACTION_COUNT, device=torch.device(device))
    if not isinstance(model, BC_RNN_GMM) or model.nets["policy"].use_tanh:
        raise RuntimeError("official BC_RNN_GMM with default use_tanh=False is required")
    return model, config


def masked_action_nll(model, batch: dict):
    """Return the mean NLL over true action labels, or None when there are none.

    The official policy builds its GMM for the observation sequence. We select
    valid distribution parameters and actual actions before any log_prob call.
    Prefix, terminal and pad observations remain causal recurrent context.
    """
    if set(batch["obs"]) != {OBS_KEY}:
        raise ValueError("the only policy input must be the stored scene feature vector")
    obs, actions, mask = batch["obs"][OBS_KEY], batch["actions"], batch["action_mask"]
    if obs.ndim != 3 or obs.shape[1:] != (HORIZON, FEATURE_COUNT):
        raise ValueError("expected observations [B,10,82]")
    if actions.shape != (*obs.shape[:2], ACTION_COUNT) or mask.shape != obs.shape[:2]:
        raise ValueError("actions or action mask shape is inconsistent")
    if mask.dtype != torch.bool:
        raise ValueError("action mask must be boolean")
    valid_obs = batch.get("observation_mask")
    if valid_obs is not None:
        if valid_obs.shape != mask.shape or valid_obs.dtype != torch.bool:
            raise ValueError("observation mask shape or type is invalid")
        if torch.any(mask & ~valid_obs):
            raise ValueError("padding cannot have an action label")
        if torch.any((~valid_obs[:, :-1]) & valid_obs[:, 1:]):
            raise ValueError("observation padding must be on the right")
    if not torch.isfinite(obs).all() or not torch.isfinite(actions[mask]).all():
        raise ValueError("model observations and supervised actions must be finite")
    count = int(mask.sum().item())
    if count == 0:
        return None, {"valid_action_count": 0, "skipped": True}
    safe_actions = torch.where(mask.unsqueeze(-1), actions, torch.zeros_like(actions))
    processed = model.process_batch_for_training({"obs": batch["obs"], "actions": safe_actions})
    distribution = model.nets["policy"].forward_train(obs_dict=processed["obs"], goal_dict=processed["goal_obs"])
    components = distribution.component_distribution
    if (not isinstance(distribution, D.MixtureSameFamily) or not isinstance(components, D.Independent)
            or not isinstance(components.base_dist, D.Normal) or components.reinterpreted_batch_ndims != 1
            or distribution.batch_shape != obs.shape[:2] or distribution.event_shape != (ACTION_COUNT,)):
        raise ValueError("official GMM distribution shape or type differs")
    mask_on_device = mask.to(components.base_dist.loc.device)
    base = components.base_dist
    selected_components = D.Independent(D.Normal(base.loc[mask_on_device], base.scale[mask_on_device]), 1)
    selected_actions = processed["actions"][mask_on_device]
    # This is PyTorch MixtureSameFamily.log_prob's original operation order,
    # applied after selecting valid batch/time positions. Reconstructing a
    # Categorical here would normalize its logits an extra time.
    log_prob_x = selected_components.log_prob(selected_actions.unsqueeze(-2))
    log_mix_prob = torch.log_softmax(distribution.mixture_distribution.logits[mask_on_device], dim=-1)
    selected = torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)
    if not torch.isfinite(selected).all():
        raise ValueError("valid GMM log probabilities must be finite")
    return -selected.mean(), {"valid_action_count": count, "skipped": False}


def masked_train_on_batch(model, batch: dict, *, validate: bool = False) -> dict:
    """Call the original optimizer step on the masked loss; skip empty batches.

    Like the official epoch loop, the scheduler's on_gradient_step hook runs
    only after an actual optimizer update. A batch with no labels does neither.
    """
    if model.nets.training == validate:
        raise ValueError("set the official model to train or eval mode before calling")
    with torch.no_grad() if validate else nullcontext():
        loss, info = masked_action_nll(model, batch)
        if loss is None:
            return {**info, "loss": None, "optimizer_updated": False}
        result = {**info, "loss": float(loss.detach().cpu()), "optimizer_updated": False}
        if not validate:
            step = model._train_step({"action_loss": loss})
            model.on_gradient_step()
            result.update(optimizer_updated=True, policy_grad_norms=float(step["policy_grad_norms"]))
        return result
