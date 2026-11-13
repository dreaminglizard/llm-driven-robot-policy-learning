"""Shared student inputs for variable-object, goal-conditioned manipulation.

The scene is registered/expected symbolic state at transfer start. Slot XY is
registered nominal calibration, never the simulator's sampled object position.
Object poses, contacts, stage indices, timestamps and future targets are not read.
"""
from __future__ import annotations
import math

from . import robot_observation_features, task_catalog

NOMINAL_SLOT_XY_MM = ((150.0, 150.0), (150.0, 0.0), (-150.0, 150.0), (-150.0, 0.0))
SLOT_SCALE_MM = 300.0
HISTORY_LENGTH = 8
FEATURE_NAMES = (
    *robot_observation_features.FEATURE_NAMES,
    *(f"source_slot_{i}" for i in range(1, 5)),
    *(f"destination_slot_{i}" for i in range(1, 5)),
    *(f"source_level_{i}" for i in range(1, 3)),
    *(f"destination_level_{i}" for i in range(1, 3)),
    *(f"moved_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_level_{level}_id_{obj}"
      for slot in range(1, 5) for level in range(1, 3) for obj in range(5)),
    *(f"active_object_id_{i}" for i in range(1, 5)),
    *(f"registered_slot_{slot}_{axis}_mm/300" for slot in range(1, 5) for axis in ("x", "y")),
)
SCHEMA = "raccoon-scene-policy-features.v1"


def transfer_task(registered_scene, task):
    before = task_catalog.scene(registered_scene)
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    source, destination = task.get("source_slot"), task.get("destination_slot")
    after = task_catalog.transfer(before, source, destination)
    canonical = {"source_slot": source, "destination_slot": destination,
                 "source_level": len(before[source - 1]),
                 "destination_level": len(after[destination - 1]),
                 "object_id": before[source - 1][-1]}
    for key, expected in canonical.items():
        if key in task and (type(task[key]) is not int or task[key] != expected):
            raise ValueError(f"task.{key} contradicts the registered scene")
    return before, canonical


def encode_scene_features(observation, registered_scene, task, slot_xy_mm=NOMINAL_SLOT_XY_MM):
    before, goal = transfer_task(registered_scene, task)
    # The pure-Python arm-state contract matches motion_policy's first 14
    # features without loading the training framework during collection.
    encoded = robot_observation_features.encode(observation)
    for field, values in (("source_slot", range(1, 5)), ("destination_slot", range(1, 5)),
                          ("source_level", range(1, 3)), ("destination_level", range(1, 3)),
                          ("object_id", range(1, 5))):
        encoded.extend(float(goal[field] == value) for value in values)
    for stack in before:
        for level in range(2):
            actual = stack[level] if level < len(stack) else 0
            encoded.extend(float(actual == value) for value in range(5))
    active = {obj for stack in before for obj in stack}
    encoded.extend(float(obj in active) for obj in range(1, 5))
    if not isinstance(slot_xy_mm, (tuple, list)) or len(slot_xy_mm) != 4:
        raise ValueError("four registered nominal XY coordinates are required")
    for xy in slot_xy_mm:
        if not isinstance(xy, (tuple, list)) or len(xy) != 2:
            raise ValueError("each registered slot needs X and Y")
        for coordinate in xy:
            if type(coordinate) not in (int, float) or not math.isfinite(coordinate):
                raise ValueError("registered XY must be finite numbers")
            encoded.append(coordinate / SLOT_SCALE_MM)
    if len(encoded) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in encoded):
        raise ValueError("invalid encoded scene features")
    return encoded


def history_window(history, length=HISTORY_LENGTH):
    """Left pad past/current encoded observations; return a validity mask."""
    if type(length) is not int or length < 1 or not isinstance(history, (list, tuple)) or not history:
        raise ValueError("a nonempty history and positive integer length are required")
    selected = history[-length:]
    for row in selected:
        if (not isinstance(row, (list, tuple)) or len(row) != len(FEATURE_NAMES)
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
            raise ValueError("history contains an invalid feature row")
    missing = length - len(selected)
    return [[0.0] * len(FEATURE_NAMES) for _ in range(missing)] + [list(row) for row in selected], \
        [False] * missing + [True] * len(selected)
