"""Materialize fixed candidate draws without changing source or student inputs."""
from __future__ import annotations

import numpy as np

from raccoon_memory.robomimic_sliding_scene_data import FIELDS, MAX_VARIANTS, SCHEDULE_DTYPE


def materialize_batch(arrays, metadata, draws):
    """Return NumPy rows and masks; only observations contain model input.

    Existing aligned rows supply every observed time step. Future time steps
    remain separate sequence entries, and unsupervised slots stay masked.
    """
    draws = np.asarray(draws)
    if draws.ndim != 1 or draws.size == 0 or draws.dtype != SCHEDULE_DTYPE:
        raise ValueError("expected a nonempty fixed candidate batch")
    unit, variant = draws["unit"], draws["variant"]
    choice, old_row, start = draws["baseline_choice"], draws["baseline_row"], draws["start"]
    if (np.any(unit >= len(metadata["counts"])) or np.any(variant >= MAX_VARIANTS)
            or np.any(choice >= 2) or np.any(metadata["counts"][unit, variant] == 0)
            or np.any(metadata["rows"][unit, variant, choice] != old_row)):
        raise ValueError("draw does not bind an original parent, ID and row")
    choices = metadata["starts"][unit, variant]
    if np.any(~np.any(choices == start[:, None], axis=1)):
        raise ValueError("draw includes an unapproved start")
    direct = metadata["direct_units"][unit]
    if np.any(start[direct] != choice[direct] * 10):
        raise ValueError("direct demonstrations must use their original window")
    positions = start.astype(np.int16)[:, None] + np.arange(10, dtype=np.int16)
    safe_positions = np.minimum(positions, 19)
    row = metadata["rows"][unit[:, None], variant[:, None], safe_positions // 10]
    safe_row = np.maximum(row, 0)
    within = safe_positions % 10
    observed = (positions < 20) & (row >= 0) & arrays["observation_mask"][safe_row, within]
    result = {}
    for name in FIELDS:
        data = np.array(arrays[name][safe_row, within], copy=True)
        data[~observed] = np.nan if name == "source_times" else -1 if name == "terminal_label" else 0
        result[name] = data
    if (not np.array_equal(result["observation_mask"], observed)
            or np.any(result["action_mask"] & ~observed)
            or not np.array_equal(result["action_mask"].sum(axis=1), draws["valid_actions"])):
        raise ValueError("materialized observations or valid-action counts differ")
    return result
