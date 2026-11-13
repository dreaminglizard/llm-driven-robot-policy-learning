"""Compact student visits, with causal windows reconstructed per episode.

Only the current 82-feature robot/registered-scene row is stored per decision.
The caller must supply independently fixed model fingerprints when loading.
Recorded actions and command results remain separate from later teacher labels.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, scene_policy


SCHEMA = "unified-student-train-visits.v2"
ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "configs/scene_split_plan.v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_INITIAL = "validated_initial_control"
_SOURCE_RETURNED = "normal_simulated_command_return"
_UNRETURNED = {"guard_unreturned_no_command": False, "output_rejected_no_command": True}


def _fingerprints(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("fixed model fingerprints are required")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, dict):
            raise ValueError("invalid model fingerprint entry")
        checkpoint, state = item.get("checkpoint_sha256"), item.get("model_state_sha256")
        if (not isinstance(checkpoint, str) or not _SHA256.fullmatch(checkpoint) or
                not isinstance(state, str) or not _SHA256.fullmatch(state)):
            raise ValueError("model checkpoint and state SHA-256 are required")
        result[key] = (checkpoint, state)
    return result


def _split_plan(expected_sha):
    raw = SPLIT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("student visit split plan differs")
    plan = json.loads(raw)
    train, validation, test = (set(plan[name]) for name in
                               ("train_groups", "validation_groups", "test_groups"))
    if not train or train & (validation | test) or validation & test:
        raise ValueError("training and held-out groups overlap")
    low, high = plan["seed_ranges_inclusive"]["train"]
    if type(low) is not int or type(high) is not int or low > high:
        raise ValueError("invalid training seed range")
    return train, low, high


def _joints(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value) or
            any(not low <= v <= high for v, (low, high) in zip(value, JOINT_LIMITS))):
        raise ValueError("invalid applied joint target")
    return list(value)


def _opening(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .05:
        raise ValueError("invalid applied gripper target")
    return float(value)


def _same_target(stored, actual):
    return torch.equal(stored, torch.tensor(actual, dtype=torch.float32))


def _one_hot(values):
    return bool((((values == 0) | (values == 1)).all() & (values.sum() == 1)).item())


def _validate_feature(row, expected_scene, gripper):
    if not bool(torch.isfinite(row).all()) or not torch.equal(row[14:], expected_scene):
        raise ValueError("observation does not match its registered task")
    if row[4].item() != 1 or row[5].item() not in (0, 1):
        raise ValueError("invalid joint observation")
    command, report = row[6:9], row[9:12]
    if (not _one_hot(command) or not _one_hot(report) or
            command[1 if gripper == "open" else 2].item() != 1):
        raise ValueError("gripper observation disagrees with command history")
    valid, fresh = row[12].item(), row[13].item()
    if (valid not in (0, 1) or fresh not in (0, 1) or
            (valid == 1 and (fresh != 1 or report[0].item() == 1)) or
            (valid == 0 and report[0].item() != 1)):
        raise ValueError("invalid gripper report")


def validate_payload(payload, *, expected_model_fingerprints):
    """Validate a v2 payload and return causal teacher inputs and provenance.

    The expected fingerprints must come from the caller, not from this payload.
    ``student_actions == -1`` means no accepted student action was recorded;
    such rows remain available for teacher relabeling when their input is valid.
    """
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("unsupported compact student visit schema")
    if "windows" in payload or "masks" in payload:
        raise ValueError("compact visits must not store repeated windows")
    expected = _fingerprints(expected_model_fingerprints)
    if _fingerprints(payload.get("model_fingerprints")) != expected:
        raise ValueError("student visit fixed model fingerprints differ")
    groups, seed_low, seed_high = _split_plan(payload.get("split_plan_sha256"))
    settings = payload.get("settings")
    if (not isinstance(settings, dict) or settings.get("training_groups_only") is not True or
            settings.get("physical_success_labels") is not False or
            settings.get("physical_assessment_during_collection") is not False or
            settings.get("teacher_relabeling") is not False or settings.get("actual_hardware") is not False or
            settings.get("model_switching_within_episode") is not False):
        raise ValueError("student visit collection settings differ")
    if not isinstance(payload.get("source_sha256"), dict):
        raise ValueError("student visit source fingerprints are missing")
    features, previous, actions = (payload.get(name) for name in
                                   ("features", "previous_target_degrees", "student_actions"))
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
           for value in (features, previous, actions)):
        raise ValueError("student visit tensors must be on CPU")
    n = features.shape[0] if features.ndim else -1
    if (features.shape != (n, 82) or features.dtype != torch.float32 or
            previous.shape != (n, 4) or previous.dtype != torch.float32 or
            actions.shape != (n,) or actions.dtype != torch.long or
            not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(previous).all()) or
            bool(((actions < -1) | (actions >= len(motion_policy.ACTIONS))).any())):
        raise ValueError("student visit tensor shapes or values differ")
    bounds = previous.new_tensor(JOINT_LIMITS)
    if bool(((previous < bounds[:, 0]) | (previous > bounds[:, 1])).any()):
        raise ValueError("previous target outside joint limits")
    origins, episodes = payload.get("previous_target_source"), payload.get("episodes")
    if (not isinstance(origins, list) or len(origins) != n or
            not isinstance(episodes, list) or not episodes):
        raise ValueError("student visit episode provenance is missing")

    windows = torch.zeros((n, 8, 82), dtype=torch.float32)
    masks = torch.zeros((n, 8), dtype=torch.bool)
    cursor, identifiers, outcomes = 0, set(), Counter()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("invalid student visit episode")
        a, b, eid = episode.get("row_start"), episode.get("row_end"), episode.get("episode_id")
        if (type(a) is not int or type(b) is not int or a != cursor or b < a or b > n or
                not isinstance(eid, str) or not eid or eid in identifiers):
            raise ValueError("invalid or overlapping episode ranges")
        identifiers.add(eid)
        scene, task = episode.get("registered_scene"), episode.get("task")
        group, physics_seed = episode.get("split_group"), episode.get("physics_seed")
        model_key = episode.get("model_key")
        if model_key is None and type(episode.get("model_training_seed")) is int:
            model_key = str(episode["model_training_seed"])
        if (not isinstance(group, str) or group not in groups or type(physics_seed) is not int or
                not seed_low <= physics_seed <= seed_high or not isinstance(model_key, str) or model_key not in expected or
                scene_policy.split_group_for(scene, task) != group or
                episode.get("physical_assessment_performed") is not False or
                episode.get("physical_success") is not None or episode.get("actual_hardware") is not False):
            raise ValueError("episode does not match the training split or model")
        decisions, termination = episode.get("decisions"), episode.get("termination")
        if (not isinstance(decisions, list) or len(decisions) != b - a or
                termination not in ("model_done", "decision_limit", "out_of_range_target", "error", "interrupted")):
            raise ValueError("episode decisions or termination differ")
        outcomes[termination] += 1
        if a == b:
            if termination not in ("error", "interrupted") or not isinstance(episode.get("error"), dict):
                raise ValueError("empty episode must retain its error or interruption")
            cursor = b
            continue
        initial = episode.get("initial_validated_control")
        if not isinstance(initial, dict):
            raise ValueError("validated initial control is missing")
        applied = _joints(initial.get("joints_deg"))
        opening = _opening(initial.get("gripper_opening_m"))
        if opening < .025:
            raise ValueError("student transfer did not begin with an open gripper")
        gripper = "open"
        dummy = {"schema_version": "PolicyObservation.phase1", "joints_deg": [0., -10., -90., 10.],
                 "sensor_valid": True, "moving": False, "gripper_command": "open",
                 "gripper_reported_state": "open", "gripper_reported_valid": True,
                 "gripper_feedback_fresh": True}
        expected_scene = torch.tensor(scene_features.encode_scene_features(dummy, scene, task)[14:], dtype=torch.float32)
        for offset, decision in enumerate(decisions):
            row = a + offset
            if (not isinstance(decision, dict) or decision.get("index") != offset or
                    "policy_features" in decision or "valid_history" in decision):
                raise ValueError("invalid or duplicated decision record")
            _validate_feature(features[row], expected_scene, gripper)
            if (origins[row] != (_SOURCE_INITIAL if offset == 0 else _SOURCE_RETURNED) or
                    not _same_target(previous[row], applied)):
                raise ValueError("previous target does not match the normal command history")
            action = int(actions[row])
            result, guard_returned = decision.get("result"), decision.get("guard_returned")
            final = row == b - 1
            if action == -1:
                if (not final or result not in _UNRETURNED or guard_returned is not _UNRETURNED[result] or
                        decision.get("action") is not None or "command" in decision or
                        "attempted_command" in decision or termination not in ("error", "interrupted")):
                    raise ValueError("missing student action lacks an explicit final unreturned record")
                continue
            if (guard_returned is not True or decision.get("action") != motion_policy.ACTIONS[action] or
                    not _same_target(previous[row], _joints(decision.get("previous_applied_target_deg"))) or
                    result not in ("command_applied", "command_unconfirmed", "no_command", "done_no_command")):
                raise ValueError("student action or command record differs")
            if result != "command_applied":
                if not final or "command" in decision:
                    raise ValueError("only a normal command return may precede another decision")
                if result == "command_unconfirmed":
                    attempted = decision.get("attempted_command")
                    if (action == motion_policy.DONE or not isinstance(attempted, dict) or
                            attempted.get("source") != "shared_student_policy" or attempted.get("seconds") != 1.0):
                        raise ValueError("unconfirmed command attempt is missing")
                    _joints(attempted.get("joints_deg"))
                    _opening(attempted.get("gripper_opening_m"))
                elif "attempted_command" in decision:
                    raise ValueError("a no-command result cannot include a command attempt")
                guard = decision.get("guard")
                if guard is not None and (not isinstance(guard, dict) or guard.get("actual_applied_joints_deg") is not None):
                    raise ValueError("unconfirmed command was marked as applied")
                if result == "done_no_command" and (action != motion_policy.DONE or termination != "model_done"):
                    raise ValueError("DONE record and episode termination differ")
                if result != "done_no_command" and termination == "model_done":
                    raise ValueError("model_done requires a DONE record")
                continue
            command = decision.get("command")
            if (action == motion_policy.DONE or not isinstance(command, dict) or
                    command.get("source") != "shared_student_policy" or command.get("seconds") != 1.0 or
                    "attempted_command" in decision):
                raise ValueError("normal command record is invalid")
            target = _joints(command.get("joints_deg"))
            new_opening = _opening(command.get("gripper_opening_m"))
            guard = decision.get("guard")
            if guard is not None:
                if not isinstance(guard, dict):
                    raise ValueError("invalid guard record")
                actual = _joints(guard.get("actual_applied_joints_deg"))
                if any(not math.isclose(x, y, abs_tol=1e-7, rel_tol=0) for x, y in zip(actual, target)):
                    raise ValueError("returned and applied joint targets differ")
            if action != motion_policy.MOVE and not _same_target(previous[row], target):
                raise ValueError("gripper command changed the applied arm target")
            expected_opening = .04 if action == motion_policy.OPEN else 0. if action == motion_policy.CLOSE else opening
            if not math.isclose(new_opening, expected_opening, abs_tol=1e-9, rel_tol=0):
                raise ValueError("returned gripper command differs from the selected action")
            applied, opening = target, new_opening
            if action == motion_policy.OPEN:
                gripper = "open"
            elif action == motion_policy.CLOSE:
                gripper = "closed"
        if termination == "model_done" and decisions[-1].get("result") != "done_no_command":
            raise ValueError("model_done requires a final DONE record")
        for offset in range(8):
            if b - a > offset:
                windows[a + offset:b, 7 - offset] = features[a:b - offset]
                masks[a + offset:b, 7 - offset] = True
        cursor = b
    if cursor != n:
        raise ValueError("episode ranges do not cover the observation rows")
    pool = {"windows": windows, "masks": masks, "previous_target_degrees": previous,
            "previous_target_source": list(origins), "student_actions": actions,
            "student_action_available": actions >= 0}
    stats = {"episodes": len(episodes), "rows": n, "returned_student_actions": int((actions >= 0).sum()),
             "unavailable_student_actions": int((actions < 0).sum()), "termination": dict(outcomes),
             "physical_assessment_performed": False}
    return pool, stats


def pack_visits(episodes, rows, *, split_plan_sha256, model_fingerprints, source_sha256, settings):
    """Compact captured rows only after exact episode-local reconstruction."""
    if not isinstance(rows, list):
        raise ValueError("captured student rows are required")
    features, previous, origins, actions = [], [], [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid captured student row")
        window, mask, prior = row.get("window"), row.get("mask"), row.get("previous")
        if (not isinstance(window, torch.Tensor) or window.device.type != "cpu" or
                window.shape != (8, 82) or window.dtype != torch.float32 or
                not isinstance(mask, torch.Tensor) or mask.device.type != "cpu" or
                mask.shape != (8,) or mask.dtype != torch.bool or
                not isinstance(prior, torch.Tensor) or prior.device.type != "cpu" or
                prior.shape != (4,) or prior.dtype != torch.float32 or
                type(row.get("student_action")) is not int):
            raise ValueError("captured window, mask or previous target differs")
        features.append(window[-1].clone())
        previous.append(prior.clone())
        origins.append(row.get("previous_source"))
        actions.append(row["student_action"])
    payload = {"schema": SCHEMA, "features": torch.stack(features) if features else torch.empty((0, 82), dtype=torch.float32),
               "previous_target_degrees": torch.stack(previous) if previous else torch.empty((0, 4), dtype=torch.float32),
               "previous_target_source": origins, "student_actions": torch.tensor(actions, dtype=torch.long),
               "episodes": copy.deepcopy(episodes), "split_plan_sha256": split_plan_sha256,
               "model_fingerprints": copy.deepcopy(model_fingerprints),
               "source_sha256": copy.deepcopy(source_sha256), "settings": copy.deepcopy(settings)}
    restored, _ = validate_payload(payload, expected_model_fingerprints=model_fingerprints)
    for index, row in enumerate(rows):
        if (not torch.equal(row["window"], restored["windows"][index]) or
                not torch.equal(row["mask"], restored["masks"][index])):
            raise ValueError("captured window or mask differs from episode-local reconstruction")
    return payload


def load_visits(path, *, expected_model_fingerprints):
    """Load compact visits without changing the independent v1 loader."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return validate_payload(payload, expected_model_fingerprints=expected_model_fingerprints)
