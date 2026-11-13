"""Offline, episode-ordered intermediate for existing scene-policy records.

This is not an official robotics-policy format.  The only model-input vector is
the existing 82-feature whitelist; clocks and provenance are metadata.  A
teacher command, terminal label, and unlabelled student prefix are distinct.
"""
from __future__ import annotations

import math
from typing import Any


SCHEMA = "raccoon-scene-sequence-intermediate.v1"
FEATURE_COUNT = 82
PERIOD_SECONDS = 1.0
PERIOD_TOLERANCE = 1e-6


def count_episode(counter, split: str, source_kind: str) -> None:
    """Keep the split and source-kind counters distinct even for validation."""
    if split not in ("train", "validation") or source_kind not in ("bundle_train", "validation", "teacher", "direct"):
        raise ValueError("unsupported split or source kind")
    counter["episodes"] += 1
    counter[f"{split}_episodes"] += 1
    counter[f"source_kind_{source_kind}_episodes"] += 1


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def source_time(observation: dict) -> float:
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    return _number(observation.get("observed_monotonic"), "source observation time")


def programmatic_stream(episode: dict):
    """Yield recorded pre-command observations and the last post-observation.

    A post-observation between commands must equal the next pre-observation;
    it is not duplicated or supplied to a prior command as an input.
    """
    transitions = episode.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("episode must have transitions")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict) or transition.get("applied") is not True:
            raise ValueError("teacher transition must be an applied command")
        obs, after = transition.get("observation"), transition.get("next_observation")
        if abs(source_time(after) - source_time(obs) - PERIOD_SECONDS) > PERIOD_TOLERANCE:
            raise ValueError("teacher command duration is not one recorded second")
        if index and transitions[index - 1].get("next_observation") != obs:
            raise ValueError("consecutive recorded observations differ")
        yield "teacher_command", obs, transition
    yield "terminal_observation", transitions[-1]["next_observation"], None


def direct_stream(episode: dict, samples: list[dict]):
    """Keep the old student prefix once and attach teacher labels afterward."""
    initial = episode.get("initial_student_observations")
    if (not isinstance(initial, list) or len(initial) != 2 or not samples
            or initial[1] != samples[0].get("policy_observation")):
        raise ValueError("direct teacher's first observation must equal the second student observation")
    if (len(samples) != episode.get("sample_count")
            or [x.get("decision_index") for x in samples] != list(range(len(samples)))
            or any(x.get("episode_id") != episode.get("episode_id") for x in samples)):
        raise ValueError("direct samples do not preserve episode identity and order")
    yield "student_context", initial[0], None
    for i, sample in enumerate(samples):
        is_last = i == len(samples) - 1
        if (sample.get("action") == 3) != is_last:
            raise ValueError("direct teacher must have exactly one final explicit done")
        yield "terminal_observation" if is_last else "teacher_command", sample["policy_observation"], sample


def make_record(*, kind: str, features: list[float], time_seconds: float,
                cache_ref: dict | None, command: dict | None = None,
                terminal_origin: str | None = None) -> dict:
    if kind not in ("student_context", "teacher_command", "terminal_observation"):
        raise ValueError("unsupported record kind")
    if not isinstance(features, list) or len(features) != FEATURE_COUNT:
        raise ValueError("features must be the 82-value existing whitelist")
    features = [_number(x, "feature") for x in features]
    time_seconds = _number(time_seconds, "source time")
    if kind == "student_context":
        if cache_ref is not None or command is not None or terminal_origin is not None:
            raise ValueError("student prefix must have no teacher or terminal supervision")
        action_mask, terminal_mask, terminal = False, False, None
    elif kind == "teacher_command":
        if not isinstance(cache_ref, dict) or not isinstance(command, dict) or terminal_origin is not None:
            raise ValueError("teacher command and cache lineage are required")
        q, opening = command.get("joint_target_deg"), command.get("gripper_opening_m")
        if not isinstance(q, list) or len(q) != 4:
            raise ValueError("command needs four actual joint targets")
        for x in q:
            _number(x, "joint target")
        opening = _number(opening, "gripper target")
        if (command.get("gripper_target") not in ("open", "closed")
                or opening != (0.04 if command["gripper_target"] == "open" else 0.0)
                or command.get("duration_seconds") != PERIOD_SECONDS
                or command.get("legacy_action") not in ("move", "open", "close")):
            raise ValueError("unsupported original gripper target, duration or action")
        action_mask, terminal_mask, terminal = True, True, False
    else:
        if (not isinstance(cache_ref, dict) or command is not None
                or terminal_origin not in ("generated_last_post_observation", "explicit_teacher_finish")):
            raise ValueError("terminal must contain only a labeled no-command observation")
        action_mask, terminal_mask, terminal = False, True, True
    return {"kind": kind, "features": features, "source_time_seconds": time_seconds,
            "servo_target": command, "servo_supervision_valid": action_mask,
            "terminal_label": terminal, "terminal_supervision_valid": terminal_mask,
            "terminal_origin": terminal_origin, "source_cache_row": cache_ref}


def validate_episode(episode: dict) -> None:
    if (not isinstance(episode, dict) or episode.get("schema") != SCHEMA
            or episode.get("split") not in ("train", "validation")
            or not isinstance(episode.get("episode_id"), str) or not episode["episode_id"]):
        raise ValueError("episode identity, schema or split is invalid")
    records = episode.get("records")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("episode has too few observations")
    prefix = episode.get("context_prefix_observations")
    if type(prefix) is not int or prefix not in (0, 1) or episode.get("first_teacher_record_index") != prefix:
        raise ValueError("reset/prefix position is invalid")
    if episode.get("reset_hidden_before_record_index") != 0:
        raise ValueError("hidden state must reset before the first source observation")
    if [r.get("kind") for r in records[:prefix]] != ["student_context"] * prefix:
        raise ValueError("student prefix position differs")
    if any(r.get("kind") != "teacher_command" for r in records[prefix:-1]):
        raise ValueError("only original teacher commands may precede the terminal")
    if records[-1].get("kind") != "terminal_observation" or len(records[prefix:-1]) < 1:
        raise ValueError("episode must end in one no-command terminal observation")
    times = []
    for r in records:
        # Rebuild the entry to reject invalid masks or fabricated terminal actions.
        checked = make_record(kind=r.get("kind"), features=r.get("features"),
                              time_seconds=r.get("source_time_seconds"), cache_ref=r.get("source_cache_row"),
                              command=r.get("servo_target"), terminal_origin=r.get("terminal_origin"))
        for key in ("servo_supervision_valid", "terminal_supervision_valid", "terminal_label"):
            if r.get(key) is not checked[key]:
                raise ValueError("supervision mask or terminal label differs")
        times.append(checked["source_time_seconds"])
    if any(abs(b - a - PERIOD_SECONDS) > PERIOD_TOLERANCE for a, b in zip(times, times[1:])):
        raise ValueError("source observations are not in one-second order")
