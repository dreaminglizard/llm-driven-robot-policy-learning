"""Read-only overlapping windows over the frozen aligned scene index.

Physical and continuation-teacher episodes may start at any actual record
whose next ten positions include a real supervised command. Direct-teacher
episodes retain exactly their original aligned chunks and student prefix.
No new command, observation, model, or simulated experience is generated.
"""
from __future__ import annotations

import numpy as np

HORIZON = 10
MAX_RECORDS = 20
MAX_VARIANTS = 24
FIELDS = ("observations", "actions", "observation_mask", "action_mask",
          "terminal_mask", "terminal_label", "source_times", "joint_residuals")
SCHEDULE_DTYPE = np.dtype([("unit", "<u2"), ("variant", "u1"), ("start", "u1"),
                           ("baseline_choice", "u1"), ("baseline_row", "<u2"),
                           ("valid_actions", "u1")])


def build_metadata(units, arrays):
    """Create small candidate lookups; each data row still belongs to the old index."""
    if not isinstance(units, list) or not units or len(units) >= 65536:
        raise ValueError("a bounded ordered unit list is required")
    action, observed = np.asarray(arrays["action_mask"]), np.asarray(arrays["observation_mask"])
    if (action.ndim != 2 or action.shape[1] != HORIZON or observed.shape != action.shape
            or action.dtype != np.bool_ or observed.dtype != np.bool_ or np.any(action & ~observed)):
        raise ValueError("original masks are inconsistent")
    n = len(units)
    rows = np.full((n, MAX_VARIANTS, 2), -1, dtype=np.int32)
    counts = np.zeros((n, MAX_VARIANTS), dtype=np.uint8)
    starts = np.full((n, MAX_VARIANTS, MAX_RECORDS), -1, dtype=np.int8)
    labels = np.zeros((n, MAX_VARIANTS, MAX_RECORDS), dtype=np.uint8)
    absolute_action = np.zeros((n, MAX_VARIANTS, MAX_RECORDS), dtype=np.bool_)
    absolute_observed = np.zeros((n, MAX_VARIANTS, MAX_RECORDS), dtype=np.bool_)
    direct = np.zeros(n, dtype=np.bool_)
    for unit_i, unit in enumerate(units):
        kind, variants = unit.get("kind"), unit.get("variants")
        if kind not in ("physical", "teacher", "direct") or not isinstance(variants, list) or not 1 <= len(variants) <= MAX_VARIANTS:
            raise ValueError("invalid unit kind or variants")
        direct[unit_i] = kind == "direct"
        for variant_i, variant in enumerate(variants):
            old_rows, old_starts = variant.get("chunk_rows"), variant.get("start_record_indices")
            if (not isinstance(old_rows, list) or not 1 <= len(old_rows) <= 2
                    or old_starts != [10 * i for i in range(len(old_rows))]
                    or any(type(r) is not int or r < 0 or r >= len(action) for r in old_rows)):
                raise ValueError("original chunk rows or phases differ")
            if kind != "direct" and len(old_rows) != 2:
                raise ValueError("physical and continuation episodes require both original chunks")
            rows[unit_i, variant_i, :len(old_rows)] = old_rows
            for chunk_i, row in enumerate(old_rows):
                offset = chunk_i * HORIZON
                absolute_action[unit_i, variant_i, offset:offset + HORIZON] = action[row]
                absolute_observed[unit_i, variant_i, offset:offset + HORIZON] = observed[row]
            aa = absolute_action[unit_i, variant_i]
            oo = absolute_observed[unit_i, variant_i]
            if np.any((~oo[:-1]) & oo[1:]):
                raise ValueError("original episode has an observation after padding")
            if kind == "direct":
                choices = old_starts
            else:
                choices = [i for i in range(MAX_RECORDS) if oo[i] and np.any(aa[i:i + HORIZON])]
            if not 1 <= len(choices) <= MAX_RECORDS:
                raise ValueError("episode has no bounded supervised window")
            for choice_i, start in enumerate(choices):
                number = int(np.count_nonzero(aa[start:start + HORIZON]))
                if number < 1:
                    raise ValueError("a candidate window has no real action label")
                starts[unit_i, variant_i, choice_i] = start
                labels[unit_i, variant_i, choice_i] = number
            counts[unit_i, variant_i] = len(choices)
    return {"rows": rows, "counts": counts, "starts": starts, "valid_actions": labels,
            "absolute_action_mask": absolute_action, "absolute_observation_mask": absolute_observed,
            "direct_units": direct}


def reconstruct_window(arrays, metadata, *, unit: int, variant: int, start: int,
                       baseline_choice: int, baseline_row: int):
    """Make one masked window using only pre-existing observations and commands.

    Values from the future within this training sequence are later time steps,
    never appended to an earlier observation. Direct-teacher windows are exact
    copies of their original row; their unlabelled prefix is not discarded.
    """
    if (type(unit) is not int or type(variant) is not int or type(start) is not int
            or not 0 <= unit < len(metadata["counts"]) or not 0 <= variant < MAX_VARIANTS):
        raise ValueError("invalid unit, variant, or start")
    count = int(metadata["counts"][unit, variant])
    allowed = metadata["starts"][unit, variant, :count]
    if start not in allowed:
        raise ValueError("window start is not in the fixed candidate index")
    old_rows = metadata["rows"][unit, variant]
    if (type(baseline_choice) is not int or not 0 <= baseline_choice < 2
            or int(old_rows[baseline_choice]) != baseline_row):
        raise ValueError("baseline unit, variant and original row are not bound")
    if metadata["direct_units"][unit]:
        if start != baseline_choice * HORIZON:
            raise ValueError("direct demonstrations must retain the original aligned chunk")
        return {name: np.array(arrays[name][baseline_row], copy=True) for name in FIELDS}
    result = {}
    for name in FIELDS:
        sample = arrays[name]
        fill = np.nan if name == "source_times" else -1 if name == "terminal_label" else 0
        result[name] = np.full((HORIZON, *sample.shape[2:]), fill, dtype=sample.dtype)
    for offset in range(HORIZON):
        pos = start + offset
        if pos >= MAX_RECORDS:
            break
        chunk_i, within = divmod(pos, HORIZON)
        source_row = int(old_rows[chunk_i])
        if source_row < 0 or not bool(arrays["observation_mask"][source_row, within]):
            continue
        for name in FIELDS:
            result[name][offset] = arrays[name][source_row, within]
    if np.any(result["action_mask"] & ~result["observation_mask"]):
        raise ValueError("a padded observation cannot contain a supervised action")
    if not bool(result["action_mask"].any()):
        raise ValueError("candidate window contains no real action")
    return result


def fixed_candidate_draw(original, metadata, rng):
    """Map a fixed original draw to a new start without changing parent/ID order."""
    if original.ndim != 1:
        raise ValueError("one flat batch of original samples is required")
    unit, variant = original["unit"], original["variant"]
    if (np.any(unit >= len(metadata["counts"])) or np.any(variant >= MAX_VARIANTS)
            or np.any(metadata["counts"][unit, variant] == 0)):
        raise ValueError("original parent or ID representation is not in the index")
    choice = original["choice"]
    if (np.any(choice >= 2) or np.any(metadata["rows"][unit, variant, choice] != original["row"])):
        raise ValueError("original sample no longer matches its frozen row")
    high = metadata["counts"][unit, variant]
    candidate_choice = rng.integers(0, high, size=len(original), dtype=np.uint8)
    direct = metadata["direct_units"][unit]
    candidate_choice[direct] = choice[direct]
    start = metadata["starts"][unit, variant, candidate_choice]
    if np.any(start < 0):
        raise ValueError("candidate selection points at a missing window")
    output = np.empty(len(original), dtype=SCHEDULE_DTYPE)
    output["unit"], output["variant"], output["start"] = unit, variant, start
    output["baseline_choice"], output["baseline_row"] = choice, original["row"]
    output["valid_actions"] = metadata["valid_actions"][unit, variant, candidate_choice]
    return output
