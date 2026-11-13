"""Enumerate the requested four-slot, one-to-four-object manipulation scope.

This is a task specification, not a controller or a learned policy. It does not
change the legacy four-object world or claim that listed tasks have been run.
Stacks are bottom-to-top; IDs are registered identities, not sensed colors.
"""
from __future__ import annotations

from itertools import combinations, permutations, product

SLOT_COUNT = 4
OBJECT_IDS = (1, 2, 3, 4)
MAX_HEIGHT = 2
KINDS = {(1, 1): "move_to_empty", (1, 2): "stack",
         (2, 1): "unstack", (2, 2): "transfer_between_stacks"}


def scene(value):
    if not isinstance(value, (tuple, list)) or len(value) != SLOT_COUNT:
        raise ValueError("exactly four stacks are required")
    if any(not isinstance(stack, (tuple, list)) for stack in value):
        raise ValueError("each stack must be a list or tuple")
    result = tuple(tuple(stack) for stack in value)
    flat = [obj for stack in result for obj in stack]
    if (not 1 <= len(flat) <= 4 or len(set(flat)) != len(flat)
            or any(type(obj) is not int or obj not in OBJECT_IDS for obj in flat)
            or any(len(stack) > MAX_HEIGHT for stack in result)):
        raise ValueError("use one to four unique IDs in 1..4, at most two per slot")
    return result


def count_layouts(total=None):
    if total is not None and (type(total) is not int or not 1 <= total <= 4):
        raise ValueError("total must be an integer from one to four")
    return tuple(counts for counts in product(range(MAX_HEIGHT + 1), repeat=SLOT_COUNT)
                 if 1 <= sum(counts) <= 4 and (total is None or sum(counts) == total))


def from_counts(counts, order=None):
    if (not isinstance(counts, (tuple, list)) or len(counts) != SLOT_COUNT
            or any(type(n) is not int or not 0 <= n <= MAX_HEIGHT for n in counts)
            or not 1 <= sum(counts) <= 4):
        raise ValueError("counts must be four integers in 0..2 with total in 1..4")
    order = tuple(range(1, sum(counts) + 1)) if order is None else tuple(order)
    if len(order) != sum(counts):
        raise ValueError("object order does not match counts")
    result, offset = [], 0
    for count in counts:
        result.append(order[offset:offset + count])
        offset += count
    return scene(result)


def identity_scenes(total):
    """Include every active-ID subset and bottom-to-top identity ordering."""
    for counts in count_layouts(total):
        for active in combinations(OBJECT_IDS, total):
            for order in permutations(active):
                yield from_counts(counts, order)


def transfer(value, source_slot, destination_slot):
    before = scene(value)
    if (type(source_slot) is not int or type(destination_slot) is not int
            or not 1 <= source_slot <= 4 or not 1 <= destination_slot <= 4
            or source_slot == destination_slot):
        raise ValueError("source and destination must be different slots in 1..4")
    source, destination = source_slot - 1, destination_slot - 1
    if not before[source] or len(before[destination]) >= MAX_HEIGHT:
        raise ValueError("source is empty or destination is full")
    after = [list(stack) for stack in before]
    moved = after[source].pop()
    after[destination].append(moved)
    return scene(after)


def one_step_cases():
    """Enumerate count-layout transitions; identity permutations are separate."""
    cases = []
    for counts in count_layouts():
        before = from_counts(counts)
        for source in range(1, 5):
            for destination in range(1, 5):
                if source == destination or not counts[source - 1] or counts[destination - 1] >= MAX_HEIGHT:
                    continue
                source_level, destination_level = counts[source - 1], counts[destination - 1] + 1
                cases.append({
                    "case_id": "layout_" + "".join(map(str, counts)) + f"_s{source}_d{destination}",
                    "object_count": sum(counts), "counts_before": counts,
                    "canonical_scene_before": before,
                    "canonical_scene_after": transfer(before, source, destination),
                    "moved_object_id": before[source - 1][-1],
                    "task": {"source_slot": source, "destination_slot": destination,
                             "source_level": source_level, "destination_level": destination_level},
                    "kind": KINDS[(source_level, destination_level)],
                    "execution_status": "not_executed_by_catalogue", "success": None,
                })
    return tuple(cases)
