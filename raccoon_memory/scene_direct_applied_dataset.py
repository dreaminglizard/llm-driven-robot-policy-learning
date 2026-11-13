"""Causal 92-feature episodes from the fixed train-only direct corpus.

The pre-existing direct-corpus verifier checks the 113 selected transfer
records, source evidence, split, returns, and post-DONE assessment.  This
adapter additionally reads the pinned original event logs.  It preserves the
45 context-only observations, recovering their applied actions from the
events rather than treating their sanitized -1 values as actions or labels.
The twelve recorded transfers in five sequence IDs are joined into five
registration histories.  An expected-scene update is only an expected commit.

MOVE supervision is the actual returned simulator joint command in degrees;
OPEN/CLOSE/DONE have no joint target.  No inverse frame mapping, hidden object
state, operator proposal, later assessment or source ID enters model features.
An ordinary direct registration's initial target is recorded in the first
observation; a separate initial servo-return receipt is not asserted.  The
direct policy's recorded moving bit likewise is not independently present in
its visible observation event.  Takeover initial controls and observations
have additional evidence checked by the existing takeover verifier.

This is data loading and validation only.  It does not train, step a simulator,
contact a robot, or open validation, test or pilot trajectories.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

import torch

from . import motion_policy, scene_applied_action_student as student
from .scene_applied_action_dataset import Exclusion, TrainingEpisode


ROOT = Path(__file__).resolve().parents[1]
CORPUS_RECORD = ROOT / "artifacts/architecture_research/20261103_chunked_current_takeovers_seed20261101_v1/result.json"
CORPUS_RECORD_SHA256 = "ddd7a50aaf632d8206c7c44c449f6bcbe6c6b841f238d504c3b7caf6e9cee8b8"
SPLIT_SHA256 = "13b568c881b042bdf85c94c952db77cb02f145ba6e89056cc046c65f97a27aa7"
FORMAT = "raccoon-scene-direct-applied-dataset.v1"
IDENTITY_LIMITED_TAKEOVERS = frozenset({
    "n1_s1_d1_other00-s1-d4-p1030",
    "n3_s1_d2_other01-s2-d3-p1029",
    "n2_s2_d1_other00-s4-d1-p1029",
})
_BATCH_FILE = re.compile(r"direct_trajectory_([A-Z])(?:_move([1-9][0-9]*))?\.pt\Z")


@dataclass(frozen=True)
class DirectDatasetResult:
    episodes: tuple[TrainingEpisode, ...]
    excluded: tuple[Exclusion, ...]
    source_record_sha256: str
    split_sha256: str
    counts: dict


class _Reject(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Reject(reason)


def _key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_pinned(path, hashes):
    expected = hashes.get(_key(path))
    _require(isinstance(expected, str), "source_fingerprint_missing")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise _Reject("source_file_missing") from exc
    _require(_digest(raw) == expected, "source_fingerprint_mismatch")
    return raw


def _joints(value, reason):
    try:
        return student._applied_target(value)
    except (TypeError, ValueError, OverflowError):
        raise _Reject(reason) from None


def _near(a, b, tolerance=3e-5):
    return len(a) == len(b) == 4 and all(math.isclose(x, y, abs_tol=tolerance, rel_tol=0)
                                          for x, y in zip(a, b))


def _scene(value):
    _require(isinstance(value, (tuple, list)) and len(value) == 4 and
             all(isinstance(stack, (tuple, list)) for stack in value), "invalid_expected_scene")
    return tuple(tuple(stack) for stack in value)


def _event_files(episode, hashes, cache):
    trajectory = Path(episode["trajectory"])
    directory = trajectory.parent
    if directory not in cache:
        event_file = directory / "interactive_events.jsonl"
        raw = _read_pinned(event_file, hashes)
        try:
            events = [json.loads(line) for line in raw.splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _Reject("invalid_event_log") from exc
        _require(events and all(isinstance(event, dict) for event in events), "invalid_event_log")
        cache[directory] = events
    try:
        payload = torch.load(io.BytesIO(_read_pinned(trajectory, hashes)), map_location="cpu", weights_only=True)
    except _Reject:
        raise
    except Exception as exc:
        raise _Reject("invalid_trajectory") from exc
    _require(isinstance(payload, dict), "invalid_trajectory")
    return payload, cache[directory]


def _direct_records(episode, payload, events):
    """Return per-decision observations/returns and their event positions."""
    name = Path(episode["trajectory"]).name
    match = _BATCH_FILE.fullmatch(name)
    if name == "direct_trajectory.pt":
        _require(not episode.get("sequence_id"), "sequence_identity_mismatch")
        observations = [(i, e) for i, e in enumerate(events) if e.get("event") == "observation"]
        decisions = [(i, e) for i, e in enumerate(events) if e.get("event") == "decision"]
        alias = None
    else:
        _require(match is not None, "unsupported_direct_trajectory_name")
        alias = match.group(1)
        observations, decisions = [], []
        for i, event in enumerate(events):
            if event.get("event") not in ("observations", "decisions"):
                continue
            scenes = event.get("scenes")
            _require(isinstance(scenes, dict), "invalid_event_log")
            if alias in scenes:
                (observations if event["event"] == "observations" else decisions).append((i, scenes[alias]))
    _require(len(observations) == len(decisions) and observations, "missing_observation_or_decision")
    _require(all(isinstance(o, dict) and isinstance(d, dict) and oi < di
                 for (oi, o), (di, d) in zip(observations, decisions)), "event_order_mismatch")
    _require(all(decisions[i][0] < observations[i+1][0] for i in range(len(decisions)-1)),
             "event_order_mismatch")
    if episode.get("sequence_id"):
        _require(match is not None and match.group(2) is not None and
                 int(match.group(2)) == episode.get("sequence_move_index") and
                 episode["sequence_id"].endswith(":" + alias), "sequence_identity_mismatch")
        boundaries = [i for i, (_, d) in enumerate(decisions) if (d.get("submitted") or {}).get("action") == "done"]
        part = episode["sequence_move_index"] - 1
        _require(part >= 0 and part < len(boundaries) and boundaries[-1] == len(decisions)-1,
                 "sequence_boundary_missing")
        first = 0 if part == 0 else boundaries[part-1] + 1
        end = boundaries[part] + 1
        observations, decisions = observations[first:end], decisions[first:end]
    else:
        _require(match is None or match.group(2) is None, "sequence_identity_mismatch")
    _require(len(observations) == len(decisions) == episode["length"], "episode_length_mismatch")
    rows = []
    for (oi, obs), (di, decision) in zip(observations, decisions):
        submitted = decision.get("submitted")
        _require(isinstance(submitted, dict), "missing_action_submission")
        rows.append({"observation": obs, "observation_event": oi, "decision_event": di,
                     "action": submitted.get("action"), "result": decision.get("result"),
                     "actual": decision.get("actual_applied_target_deg")})
    return rows, alias


def _takeover_records(episode, payload, events):
    _require(payload.get("schema") == "terra-student-prefix-direct-takeover.v1" and
             isinstance(payload.get("rows"), list) and len(payload["rows"]) == episode["length"],
             "unsupported_takeover_trajectory")
    prefix = episode.get("student_prefix_length")
    _require(type(prefix) is int and 0 < prefix < episode["length"], "takeover_prefix_mismatch")
    student_obs = [(i, e) for i, e in enumerate(events) if e.get("event") == "student_observation"]
    teacher_obs = [(i, e) for i, e in enumerate(events) if e.get("event") == "teacher_observation"]
    _require(len(student_obs) == prefix and len(teacher_obs) == episode["length"] - prefix,
             "takeover_observation_missing")
    rows = []
    for index, ((event_index, observation), raw) in enumerate(zip(student_obs + teacher_obs, payload["rows"])):
        _require(observation.get("index") == index and isinstance(raw, dict), "takeover_observation_mismatch")
        if index < prefix:
            decision = raw.get("student_decision")
            _require(raw.get("phase") == "student" and isinstance(decision, dict), "takeover_prefix_mismatch")
            returned = decision.get("servo_targets_after_return")
            _require(isinstance(returned, dict), "command_return_unconfirmed")
            action, result, actual = decision.get("action"), decision.get("result"), returned.get("joints_deg")
        else:
            submitted = raw.get("submitted")
            _require(raw.get("phase") == "teacher" and isinstance(submitted, dict), "takeover_phase_mismatch")
            action, result, actual = submitted.get("action"), raw.get("teacher_result"), raw.get("actual_applied_target_deg")
        rows.append({"observation": observation, "observation_event": event_index,
                     "action": action, "result": result, "actual": actual})
    return rows


def _check_expected_transition(left, right, alias, events):
    previous_episode, previous_payload, previous_rows = left
    episode, payload, rows = right
    _require(previous_episode["sequence_id"] == episode["sequence_id"] and
             episode["sequence_move_index"] == previous_episode["sequence_move_index"] + 1,
             "sequence_boundary_missing")
    _require(_scene(previous_payload.get("expected_after")) == _scene(payload.get("expected_before")),
             "expected_scene_discontinuity")
    start, end = previous_rows[-1]["decision_event"], rows[0]["observation_event"]
    updates = [event["scenes"][alias] for event in events[start+1:end]
               if event.get("event") == "expected_scene_updated" and
               isinstance(event.get("scenes"), dict) and alias in event["scenes"]]
    _require(len(updates) == 1 and updates[0].get("physical_confirmation") == "not_inspected" and
             _scene(updates[0].get("expected_scene")) == _scene(payload.get("expected_before")) and
             updates[0].get("task") == rows[0]["observation"].get("task"), "expected_commit_unconfirmed")


def _convert_group(group, data, hashes, cache):
    _require(bool(group), "empty_registration")
    sequence = group[0].get("sequence_id")
    if sequence:
        group = sorted(group, key=lambda e: e.get("sequence_move_index", -1))
        _require([e.get("sequence_move_index") for e in group] == list(range(1, len(group)+1)) and
                 len(group) in (2, 3) and all(e.get("sequence_id") == sequence for e in group) and
                 len({str(Path(e["trajectory"]).parent) for e in group}) == 1,
                 "sequence_boundary_missing")
    else:
        _require(len(group) == 1, "unexpected_registration_group")
    features, actions, targets, label_mask, applied = [], [], [], [], []
    previous_action, last_actual, previous_part, prior_alias, prior_events = None, None, None, None, None
    is_takeover = group[0].get("source_kind") == "student_prefix_direct_takeover"
    for transfer_index, episode in enumerate(group):
        payload, events = _event_files(episode, hashes, cache)
        if is_takeover:
            _require(len(group) == 1, "takeover_sequence_unsupported")
            rows, alias = _takeover_records(episode, payload, events), None
        else:
            rows, alias = _direct_records(episode, payload, events)
            if previous_part is not None:
                _require(alias == prior_alias and events is prior_events, "sequence_identity_mismatch")
                _check_expected_transition(previous_part, (episode, payload, rows), alias, events)
        prefix = episode.get("student_prefix_length", 0)
        excluded = {entry["episode_row_index"] for entry in episode.get("excluded_teacher_decisions", [])}
        for offset, row in enumerate(rows):
            index = episode["start"] + offset
            _require(isinstance(payload.get("windows"), torch.Tensor) and
                     isinstance(payload.get("masks"), torch.Tensor) and
                     torch.equal(data["windows"][index], payload["windows"][offset]) and
                     torch.equal(data["masks"][index], payload["masks"][offset]), "source_window_mismatch")
            observed = data["windows"][index, -1]
            _require(observed.shape == (82,) and bool(torch.isfinite(observed).all()), "invalid_observation")
            observation = row["observation"]
            _require(isinstance(observation, dict), "invalid_observation")
            recorded_previous = _joints(observation.get("previous_applied_target_deg"), "missing_previous_target")
            _require(torch.equal(data["previous"][index], torch.tensor(recorded_previous, dtype=torch.float32)),
                     "previous_input_mismatch")
            if last_actual is not None:
                _require(_near(recorded_previous, last_actual), "previous_applied_target_mismatch")
            valid_label = bool(data["label_mask"][index])
            if is_takeover:
                expected_label = offset >= prefix and offset not in excluded
                _require(valid_label == expected_label, "takeover_label_mask_mismatch")
            else:
                _require(valid_label, "direct_label_mask_mismatch")
            action, result = row["action"], row["result"]
            _require(action in motion_policy.ACTIONS, "unknown_action")
            if valid_label:
                _require(int(data["actions"][index]) == motion_policy.ACTIONS.index(action), "action_label_mismatch")
                _require(torch.equal(data["targets"][index], payload["targets"][offset]),
                         "source_target_mismatch")
            else:
                _require(int(data["actions"][index]) == -1 and bool((data["targets"][index] == 0).all()),
                         "context_label_leak")
            extras = torch.tensor(
                [float(previous_action == a) for a in student.PREVIOUS_APPLIED_ACTIONS] +
                [v / motion_policy.JOINT_SCALE_DEG for v in recorded_previous] +
                [float(offset == 0), float(transfer_index == 0 and offset == 0)], dtype=torch.float32)
            features.append(torch.cat((observed, extras)))
            target, is_move = [float("nan")] * 4, False
            if action == "done":
                _require(result == "done_no_command" and offset == len(rows)-1 and valid_label,
                         "done_return_unconfirmed")
            else:
                _require(result == "command_applied", "command_return_unconfirmed")
                actual = _joints(row["actual"], "command_return_unconfirmed")
                if action != "move":
                    _require(_near(actual, recorded_previous), "gripper_changed_joint_target")
                if valid_label:
                    tensor_target = (data["targets"][index] * motion_policy.JOINT_SCALE_DEG).tolist()
                    _require(_near(actual, tensor_target), "recorded_teacher_target_mismatch")
                    if action == "move":
                        target, is_move = actual, True
                previous_action, last_actual = action, actual
            actions.append(motion_policy.ACTIONS.index(action) if valid_label else -1)
            targets.append(target)
            label_mask.append(valid_label)
            applied.append(is_move)
        _require(rows[-1]["action"] == "done", "missing_done")
        previous_part, prior_alias, prior_events = (episode, payload, rows), alias, events
    n = len(actions)
    case_ids = tuple(e["case_id"] for e in group)
    evidence = {
        "case_id": group[0]["case_id"], "case_ids": case_ids,
        "source_trajectories": tuple(e["trajectory"] for e in group),
        "sequence_id": sequence, "transfer_count": len(group),
        "source_kind": "student_prefix_direct_takeover" if is_takeover else "direct_operator",
        "initial_target_source": ("verified_initial_applied_control" if is_takeover else
                                  "first_observation_previous_target_no_separate_servo_return"),
        "previous_action_at_registration": "unknown",
        "moving_source": ("recorded_observation_event_verified" if is_takeover else
                          "recorded_policy_feature_not_in_visible_observation_event"),
        "identity_limited_takeover": group[0]["case_id"] in IDENTITY_LIMITED_TAKEOVERS and is_takeover,
        "assigned_model": group[0].get("assigned_model"),
        "runtime_model_identity_independently_verified": False,
        "post_transfer_scene": "expected_only" if sequence else None,
        "context_only_rows": n - sum(label_mask),
    }
    return TrainingEpisode(torch.stack(features), torch.tensor(actions, dtype=torch.long),
                           torch.tensor(targets, dtype=torch.float32), torch.ones(n, dtype=torch.bool),
                           torch.tensor(label_mask, dtype=torch.bool), torch.tensor(applied, dtype=torch.bool), evidence)


def _group_records(episodes):
    groups, positions = [], {}
    for episode in episodes:
        sequence = episode.get("sequence_id")
        key = ("sequence", sequence) if sequence else ("trajectory", _key(episode["trajectory"]))
        if key not in positions:
            positions[key] = len(groups)
            groups.append([])
        groups[positions[key]].append(episode)
    return groups


def _build_verified_episodes(data, source, record):
    """Convert already verified records; exposed for focused synthetic tests."""
    expected = record["episodes"]
    _require(len(source["episodes"]) == len(expected) == 113 and
             len(data["actions"]) == len(data["label_mask"]) == 2474, "verified_corpus_shape_mismatch")
    fields = ("case_id", "trajectory", "start", "length", "sequence_id", "sequence_move_index",
              "source_kind", "student_prefix_length", "excluded_teacher_decisions")
    for actual, original in zip(source["episodes"], expected):
        _require(all(actual.get(field) == original.get(field) for field in fields), "verified_corpus_provenance_mismatch")
    hashes = {_key(path): digest for path, digest in record["source_sha256"].items()}
    groups, cache, episodes, excluded = _group_records(source["episodes"]), {}, [], []
    for group in groups:
        try:
            episode = _convert_group(group, data, hashes, cache)
        except _Reject as exc:
            excluded.append(Exclusion(group[0]["case_id"], str(exc), {"transfers": tuple(group)}))
        except (KeyError, TypeError, ValueError, OverflowError, IndexError) as exc:
            excluded.append(Exclusion(group[0]["case_id"], f"invalid_record: {type(exc).__name__}",
                                      {"transfers": tuple(group)}))
        else:
            episodes.append(episode)
    counts = Counter()
    for episode in episodes:
        counts["observations"] += len(episode.actions)
        counts["supervised_actions"] += int(episode.label_mask.sum())
        counts["applied_move_targets"] += int(episode.applied_move_mask.sum())
        counts["context_only"] += int((~episode.label_mask).sum())
        counts["transfers"] += episode.evidence["transfer_count"]
        for action, name in enumerate(motion_policy.ACTIONS):
            counts[f"action_{name}"] += int(((episode.actions == action) & episode.label_mask).sum())
    counts["registration_episodes"] = len(episodes)
    counts["excluded_registration_episodes"] = len(excluded)
    return DirectDatasetResult(tuple(episodes), tuple(excluded), CORPUS_RECORD_SHA256, SPLIT_SHA256, dict(counts))


def load_direct_training_episodes():
    """Verify and load the fixed direct corpus without training or execution.

    A changed corpus record or split is a global failure; a later individual
    registration inconsistency is retained as an exclusion.  The source is
    explicitly restricted by the existing verifiers to the training split.
    Python optimization is unsupported because the source audit uses asserts.
    """
    if sys.flags.optimize:
        raise RuntimeError("direct corpus verification requires Python without optimization (-O or PYTHONOPTIMIZE)")

    from scripts import train_unified_direct_retention as verified

    raw = CORPUS_RECORD.read_bytes()
    if _digest(raw) != CORPUS_RECORD_SHA256:
        raise ValueError("direct corpus record fingerprint mismatch")
    record = json.loads(raw)["teacher_source"]
    if record.get("split_sha256") != SPLIT_SHA256:
        raise ValueError("direct corpus split fingerprint mismatch")
    provenance, _ = verified.capture_split_provenance(include_experience=False)
    if provenance["direct_corpus_record_sha256"] != CORPUS_RECORD_SHA256:
        raise ValueError("direct corpus record changed during verification")
    data, source = verified.load_direct_corpus()
    verified.validate_split_provenance(provenance, direct_source=source)
    verified.verify_current_runtime()
    result = _build_verified_episodes(data, source, record)
    verified.validate_split_provenance(provenance, direct_source=source)
    verified.verify_current_runtime()
    return result
