"""Prepare and train only the project completion head over a frozen core.

This module does not start a run or read a dataset on import. The caller must
explicitly supply episodes and an already-trained official action core.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from raccoon_memory import scene_sequence_data
from raccoon_memory.robomimic_completion import CompletionHead, FrozenOfficialCore, INPUT_SIZE, bind_head


def extract_episode(core: FrozenOfficialCore, episode: dict) -> dict:
    """Extract one post-observation state per source row, resetting at start.

    The sampled official actions are discarded, including for terminal rows.
    Prefix observations advance the core and have no completion loss target.
    """
    if not isinstance(core, FrozenOfficialCore):
        raise TypeError("a frozen official core is required")
    if core.synthetic_test and episode.get("synthetic_test") is not True:
        raise ValueError("an untrained synthetic core may only process synthetic test episodes")
    scene_sequence_data.validate_episode(episode)
    core.assert_frozen()
    core.reset_episode()
    vectors, labels, mask, counters = [], [], [], []
    try:
        for index, record in enumerate(episode["records"]):
            observation = torch.tensor(record["features"], dtype=torch.float32).reshape(1, 82)
            features, _discarded_action, counter = core._observe_once(observation)
            if counter != index + 1:
                raise RuntimeError("official core did not process exactly one observation per record")
            supervised = record["terminal_supervision_valid"]
            vectors.append(features.squeeze(0).detach().cpu())
            mask.append(supervised)
            labels.append(int(record["terminal_label"]) if supervised else -1)
            counters.append(counter)
        core.assert_frozen()
    finally:
        # The next episode must explicitly invoke the official reset again.
        core._needs_reset = True
    return {"episode_id": episode["episode_id"], "split": episode["split"], "features": torch.stack(vectors),
            "terminal_label": torch.tensor(labels, dtype=torch.long),
            "terminal_mask": torch.tensor(mask, dtype=torch.bool),
            "decision_counts": counters, "core_weights_sha256": core.weights_sha256,
            "core_checkpoint_sha256": core.checkpoint_file_sha256,
            "core_synthetic_mode": core.synthetic_test}


def collate_extracted(episodes: list[dict]) -> dict:
    """Concatenate already-reset episodes of one split and one restored core."""
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("at least one extracted episode is required")
    fingerprint = episodes[0].get("core_weights_sha256")
    checkpoint = episodes[0].get("core_checkpoint_sha256")
    synthetic = episodes[0].get("core_synthetic_mode")
    split = episodes[0].get("split")
    if (not isinstance(fingerprint, str) or type(synthetic) is not bool
            or "core_checkpoint_sha256" not in episodes[0] or split not in ("train", "validation")):
        raise ValueError("extracted core identity or scope is missing")
    for item in episodes:
        x, y, mask = item.get("features"), item.get("terminal_label"), item.get("terminal_mask")
        if item.get("split") != split:
            raise ValueError("extracted episodes must have the same split")
        if (item.get("core_weights_sha256") != fingerprint or item.get("core_synthetic_mode") is not synthetic
                or "core_checkpoint_sha256" not in item or item["core_checkpoint_sha256"] != checkpoint
                or not isinstance(item.get("episode_id"), str) or not isinstance(x, torch.Tensor)
                or not x.is_floating_point() or x.ndim != 2 or x.shape[1] != INPUT_SIZE
                or not isinstance(y, torch.Tensor) or y.dtype != torch.long or tuple(y.shape) != (len(x),)
                or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != (len(x),)
                or x.device != y.device or x.device != mask.device):
            raise ValueError("extracted episode shapes, core weights or scope differ")
    return {"episode_ids": [item["episode_id"] for item in episodes], "split": split,
            "features": torch.cat([item["features"] for item in episodes]),
            "terminal_label": torch.cat([item["terminal_label"] for item in episodes]),
            "terminal_mask": torch.cat([item["terminal_mask"] for item in episodes]),
            "core_weights_sha256": fingerprint, "core_checkpoint_sha256": checkpoint,
            "core_synthetic_mode": synthetic}


def completion_loss(head: CompletionHead, features: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    """Mean CE on explicitly supervised rows; prefix/padding never enter CE."""
    if not isinstance(head, CompletionHead):
        raise TypeError("a project completion head is required")
    if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
            or features.ndim != 2 or features.shape[1] != INPUT_SIZE
            or not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or tuple(labels.shape) != (features.shape[0],)
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or tuple(mask.shape) != (features.shape[0],)
            or labels.device != features.device or mask.device != features.device):
        raise ValueError("completion features, labels or supervision mask have invalid shapes or types")
    selected_features, selected_labels = features[mask], labels[mask]
    count = int(mask.sum().item())
    if count == 0:
        return None, 0
    if (not torch.isfinite(selected_features).all()
            or not torch.all((selected_labels == 0) | (selected_labels == 1))):
        raise ValueError("supervised completion features or labels are invalid")
    device = next(head.parameters()).device
    logits = head(selected_features.detach().to(device=device, dtype=torch.float32))
    loss = F.cross_entropy(logits, selected_labels.to(device=device))
    if not torch.isfinite(loss):
        raise ValueError("completion loss is not finite")
    return loss, count


def make_head_optimizer(core: FrozenOfficialCore, head: CompletionHead, *, learning_rate: float):
    if not isinstance(core, FrozenOfficialCore) or not isinstance(head, CompletionHead):
        raise TypeError("a frozen core and project completion head are required")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("a finite positive learning rate must be explicitly supplied")
    bind_head(core, head)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(learning_rate))
    assert_head_only_optimizer(core, head, optimizer)
    return optimizer


def assert_head_only_optimizer(core: FrozenOfficialCore, head: CompletionHead, optimizer) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("a torch optimizer is required")
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    head_params = list(head.parameters())
    core_ids = {id(p) for p in core.algorithm.nets.parameters()}
    core_storages = {(p.device, p.untyped_storage().data_ptr()) for p in
                     list(core.algorithm.nets.parameters()) + list(core.algorithm.nets.buffers())}
    if (len(optimized) != len(head_params) or len({id(p) for p in optimized}) != len(optimized)
            or {id(p) for p in optimized} != {id(p) for p in head_params}
            or any(id(p) in core_ids or (p.device, p.untyped_storage().data_ptr()) in core_storages for p in optimized)
            or not all(p.requires_grad for p in head_params)):
        raise ValueError("completion optimizer must contain every head parameter and no core parameter or storage")


def train_head_step(core: FrozenOfficialCore, head: CompletionHead, optimizer, batch: dict) -> dict:
    """One explicit update, with an unchanged frozen-core fingerprint check."""
    if not isinstance(batch, dict) or batch.get("split") != "train":
        raise ValueError("completion head updates require the train split")
    core.assert_frozen()
    bind_head(core, head)
    assert_head_only_optimizer(core, head, optimizer)
    if (batch.get("core_weights_sha256") != core.weights_sha256
            or "core_checkpoint_sha256" not in batch
            or batch["core_checkpoint_sha256"] != core.checkpoint_file_sha256):
        raise ValueError("extracted completion features belong to a different action core")
    if batch.get("core_synthetic_mode") is not core.synthetic_test:
        raise ValueError("completion batch and core synthetic-test scope differ")
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss, count = completion_loss(head, batch["features"], batch["terminal_label"], batch["terminal_mask"])
    if loss is None:
        core.assert_frozen()
        return {"updated": False, "supervised_rows": 0, "loss": None, "core_weights_sha256": core.weights_sha256}
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("completion head gradient is not finite")
    # A core gradient or unfreeze is an error before any optimizer mutation.
    core.assert_frozen()
    optimizer.step()
    core.assert_frozen()
    return {"updated": True, "supervised_rows": count, "loss": float(loss.detach().cpu()),
            "core_weights_sha256": core.weights_sha256}
