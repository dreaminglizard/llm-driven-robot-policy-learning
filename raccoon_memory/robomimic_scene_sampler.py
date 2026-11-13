"""Uniform unit -> ID variant -> valid chunk sampling for the first BC run."""
from __future__ import annotations

import copy
import numpy as np

SAMPLE_DTYPE = np.dtype([("unit", "<u2"), ("variant", "u1"), ("choice", "u1"), ("row", "<u2")])


class BalancedSceneSampler:
    """Independent draws with replacement and an explicitly serializable RNG."""

    def __init__(self, lookup: np.ndarray, variant_counts: np.ndarray, chunk_counts: np.ndarray, *, seed: int):
        self.lookup = np.asarray(lookup)
        self.variant_counts = np.asarray(variant_counts)
        self.chunk_counts = np.asarray(chunk_counts)
        if (self.lookup.ndim != 3 or self.lookup.shape[0] != len(self.variant_counts)
                or self.lookup.shape[:2] != self.chunk_counts.shape
                or len(self.variant_counts) == 0 or len(self.variant_counts) >= 65536):
            raise ValueError("lookup dimensions are inconsistent")
        for unit, count in enumerate(self.variant_counts):
            if not 1 <= int(count) <= self.lookup.shape[1] or int(count) > 255:
                raise ValueError("unit has invalid variant count")
            for variant in range(int(count)):
                chunks = int(self.chunk_counts[unit, variant])
                if not 1 <= chunks <= self.lookup.shape[2] or chunks > 255:
                    raise ValueError("variant has no valid action chunk")
                rows = self.lookup[unit, variant, :chunks]
                if np.any(rows < 0) or np.any(rows >= 65536):
                    raise ValueError("invalid chunk row")
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def draw(self, batch_size: int) -> np.ndarray:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch size must be positive")
        unit = self.rng.integers(0, len(self.variant_counts), size=batch_size, dtype=np.uint16)
        variant = self.rng.integers(0, self.variant_counts[unit], size=batch_size, dtype=np.uint8)
        choice = self.rng.integers(0, self.chunk_counts[unit, variant], size=batch_size, dtype=np.uint8)
        row = self.lookup[unit, variant, choice]
        out = np.empty(batch_size, dtype=SAMPLE_DTYPE)
        out["unit"], out["variant"], out["choice"], out["row"] = unit, variant, choice, row
        return out

    def state_dict(self) -> dict:
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state: dict) -> None:
        if state.get("bit_generator") != "PCG64":
            raise ValueError("wrong sampler RNG")
        self.rng.bit_generator.state = copy.deepcopy(state)


def validate_draws(draws: np.ndarray, lookup: np.ndarray, variant_counts: np.ndarray, chunk_counts: np.ndarray):
    """Check that every stored planned draw selects the recorded valid row."""
    if draws.dtype != SAMPLE_DTYPE:
        raise ValueError("unexpected binary sample dtype")
    unit, variant, choice, row = (draws[name] for name in SAMPLE_DTYPE.names)
    if (np.any(unit >= len(variant_counts)) or np.any(variant >= variant_counts[unit])
            or np.any(choice >= chunk_counts[unit, variant])
            or np.any(lookup[unit, variant, choice] != row)):
        raise ValueError("planned sample references an invalid unit, variant, or chunk")
