"""Valid-action optimization around the unchanged official deterministic BC-RNN."""
from __future__ import annotations

from contextlib import nullcontext

import torch

from raccoon_memory.robomimic_deterministic_core import masked_action_l2


def masked_train_on_batch(model, batch, *, validate=False):
    """Use official Adam for masked five-axis MSE; an empty mask never updates."""
    if model.nets.training == validate:
        raise ValueError("set deterministic model to the requested train/eval mode")
    with torch.no_grad() if validate else nullcontext():
        loss, info = masked_action_l2(model, batch)
        if loss is None:
            return {**info, "loss": None, "optimizer_updated": False}
        result = {**info, "loss": float(loss.detach().cpu()), "optimizer_updated": False}
        if not validate:
            step = model._train_step({"action_loss": loss})
            model.on_gradient_step()
            result.update(optimizer_updated=True, policy_grad_norms=float(step["policy_grad_norms"]))
        return result
