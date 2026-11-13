"""Load verified v2 corrected experiences as causal direct-student episodes.

Only ``train-corrected-student-experience.v2`` is understood.  The caller
supplies independent, externally trusted fingerprints for the run manifest,
summary, and supported collector.  The manifest binds the collection plan,
collector, checkpoint, planned case IDs and split-plan fingerprint; it is not
an authorization to trust a different manifest or summary.  The split plan's
entire train group/seed scope is used, never a hard-coded smoke-case list.

The 82 recorded policy features, prior *confirmed applied* physical action and
target, and registration/transfer flags are the only model inputs.  At the
first registration, the target comes from validated initial control.  DONE
does not replace the last applied action.  Expected-only scene commits update
the symbolic registration for the next transfer; they are not observations or
proof of physical success.  Hidden correction diagnostics, raw proposals,
future, data-source IDs and after-the-fact assessments never enter tensors.

An episode is admitted for main supervision only if every transfer has an
independent completed success assessment.  Failed, unassessed, malformed and
otherwise ineligible cases retain their report and exclusion reason; source
files are never modified.  MOVE targets are verified applied robot joint
commands, never raw proposals.  Non-MOVE target cells are NaN and masked out.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_applied_action_student as student
from . import scene_features, scene_policy, task_catalog


SCHEMA = "train-corrected-student-experience.v2"
SPLIT_SCHEMA = "raccoon-scene-split-plan.v1"
COLLECTOR_KEY = "scripts/collect_train_corrected_experience.py"
SPLIT_KEY = "configs/scene_split_plan.v1.json"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_EQUIVALENCE = frozenset({"registration", "expected_commits", "decision_features",
    "history_masks", "previous_applied_targets", "logits", "guard_outputs",
    "command_returns", "applied_commands", "terminations", "conditional_pose_state"})
_MAX_COMPRESSED = 64 * 1024 * 1024
_MAX_DECOMPRESSED = 256 * 1024 * 1024


@dataclass(frozen=True)
class TrustedFingerprints:
    """SHA-256 values fixed independently of the data being loaded.

    ``collector_sha256`` must identify a collector whose v2 fields have been
    reviewed for this loader, not merely a hash copied from an unknown run.
    A new summary/manifest needs its own independently supplied hashes.
    """

    manifest_sha256: str
    summary_sha256: str
    collector_sha256: str


@dataclass(frozen=True)
class TrainingEpisode:
    features: torch.Tensor                  # [time, 92]
    actions: torch.Tensor                   # [time], MOVE/OPEN/CLOSE/DONE
    applied_move_targets_deg: torch.Tensor  # [time, 4], NaN outside MOVE
    valid_mask: torch.Tensor                # [time]
    label_mask: torch.Tensor                # [time]
    applied_move_mask: torch.Tensor         # [time]
    evidence: dict                          # never passed to the model


@dataclass(frozen=True)
class Exclusion:
    case_id: str
    reason: str
    report: dict  # Preserve status/evidence separately; never a model input.
    state: dict | None = None


@dataclass(frozen=True)
class DatasetResult:
    episodes: tuple[TrainingEpisode, ...]
    excluded: tuple[Exclusion, ...]
    summary_sha256: str
    manifest_sha256: str


class _Reject(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Reject(reason)


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _read_pinned_json(path, expected, label):
    if not _sha(expected):
        raise ValueError(f"{label} requires an independently fixed SHA-256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{label} fingerprint mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _split_scope(path, expected):
    plan = _read_pinned_json(path, expected, "split plan")
    if plan.get("schema") != SPLIT_SCHEMA:
        raise ValueError("unsupported split-plan schema")
    sets = []
    for name in ("train_groups", "validation_groups", "test_groups"):
        values = plan.get(name)
        if (not isinstance(values, list) or not values or
                any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
            raise ValueError("invalid split groups")
        sets.append(set(values))
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("split groups overlap")
    ranges = plan.get("seed_ranges_inclusive")
    if not isinstance(ranges, dict):
        raise ValueError("split seed ranges are missing")
    parsed = []
    for name in ("pilot", "train", "validation", "test"):
        limits = ranges.get(name)
        if (not isinstance(limits, list) or len(limits) != 2 or
                any(type(v) is not int for v in limits) or limits[0] > limits[1]):
            raise ValueError("invalid split seed range")
        parsed.append(tuple(limits))
    for i, (lo, hi) in enumerate(parsed):
        if any(lo <= other_hi and other_lo <= hi for other_lo, other_hi in parsed[i + 1:]):
            raise ValueError("split seed ranges overlap")
    return sets[0], parsed[1]


def _vector(value, reason):
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu" and value.shape == (4,) and value.dtype == torch.float32, reason)
        value = value.tolist()
    _require(isinstance(value, (tuple, list)) and len(value) == 4, reason)
    out = []
    for v, (lo, hi) in zip(value, JOINT_LIMITS):
        _require(type(v) in (int, float), reason)
        try:
            v = float(v)
        except (ValueError, OverflowError):
            raise _Reject(reason) from None
        _require(math.isfinite(v) and lo - 1e-4 <= v <= hi + 1e-4, reason)
        out.append(v)
    return out


def _same(a, b, atol=1e-7):
    return all(abs(x - y) <= atol for x, y in zip(a, b))


def _scene(value, reason):
    try:
        return task_catalog.scene(value)
    except (TypeError, ValueError):
        raise _Reject(reason) from None


def _one_hot(row):
    return bool((((row == 0) | (row == 1)).all() & (row.sum() == 1)).item())


def _observation(row, expected_scene, gripper):
    _require(bool(torch.isfinite(row).all()) and torch.equal(row[14:], expected_scene), "observation_mismatch")
    _require(row[4].item() == 1 and row[5].item() in (0, 1), "invalid_robot_observation")
    command, report = row[6:9], row[9:12]
    _require(_one_hot(command) and _one_hot(report) and command[1 if gripper == "open" else 2] == 1,
             "gripper_history_mismatch")
    valid, fresh = row[12].item(), row[13].item()
    _require(valid in (0, 1) and fresh in (0, 1) and
             ((valid == 1 and fresh == 1 and report[0] == 0) or (valid == 0 and report[0] == 1)),
             "invalid_gripper_observation")


def _assessment(report):
    _require("training_eligible" in report and
             (report["training_eligible"] is True or report["training_eligible"] is None),
             "training_ineligible_or_invalid")
    if "execution_error" in report:
        _require(report["execution_error"] is None, "execution_error")
    for key in ("capture_error", "live_error", "integrity_error", "stop_error", "score_error"):
        _require(key in report and report[key] is None, f"{key}_or_unconfirmed")
    _require("capture_error" in report and report["capture_error"] is None, "capture_error_or_unconfirmed")
    _require("score_error" in report and report["score_error"] is None, "assessment_error_or_unconfirmed")
    _require(report.get("equivalence_identical") is True, "equivalence_unconfirmed")
    completion = report.get("completion")
    _require(isinstance(completion, dict) and completion.get("status") == "evaluation_complete",
             "assessment_unconfirmed")
    names = ("planned_transfer_count", "attempted_transfer_count", "expected_commit_count",
             "assessment_record_count", "assessed_transfer_count")
    counts = [completion.get(name) for name in names]
    _require(all(type(n) is int and n > 0 for n in counts) and len(set(counts)) == 1,
             "assessment_incomplete")
    statuses = completion.get("assessment_status_by_transfer")
    _require(isinstance(statuses, list) and len(statuses) == counts[0] and
             all(s in ("success", "failure", "not_assessed") for s in statuses), "assessment_unconfirmed")
    for key, status in (("success_count", "success"), ("failure_count", "failure"),
                        ("unassessed_count", "not_assessed")):
        _require(type(completion.get(key)) is int and completion[key] == statuses.count(status),
                 "assessment_counts_mismatch")
    _require(completion.get("all_transfers_succeeded") is (all(s == "success" for s in statuses)),
             "assessment_counts_mismatch")
    _require("failure" not in statuses, "physical_failure")
    _require("not_assessed" not in statuses, "assessment_unconfirmed")
    return statuses


def _case_execution_state(report, state):
    _require(isinstance(state, dict) and state.get("case_id") == report["case_id"] and
             state.get("execution_status") == "executed" and "error" in state and state["error"] is None,
             "case_execution_state_unconfirmed")
    completion = report.get("completion")
    _require(isinstance(completion, dict), "assessment_unconfirmed")
    _require(state.get("experience") == report.get("experience") and
             state.get("assessment") == report.get("assessment") and
             "training_eligible" in state and "training_eligible" in report and
             state["training_eligible"] is report["training_eligible"] and
             state.get("assessment_status_by_transfer") == completion.get("assessment_status_by_transfer"),
             "case_execution_state_mismatch")


def _assessment_file(base, report, statuses):
    """Verify the separately persisted assessment without using it as input."""
    name, expected = report.get("assessment"), report.get("assessment_sha256")
    _require(isinstance(name, str) and name.endswith(".assessment.json.gz") and
             "/" not in name and "\\" not in name and Path(name).name == name and _sha(expected),
             "assessment_reference_unconfirmed")
    path = base / name
    try:
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= _MAX_COMPRESSED,
                 "assessment_file_missing_or_oversize")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected, "assessment_hash_mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            decoded = stream.read(_MAX_DECOMPRESSED + 1)
        _require(len(decoded) <= _MAX_DECOMPRESSED, "assessment_oversize")
        value = json.loads(decoded)
    except _Reject:
        raise
    except Exception as exc:
        raise _Reject("assessment_decode_error") from exc
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA, "unsupported_assessment_schema")
    _require("training_eligible" in value and value["training_eligible"] is report["training_eligible"],
             "assessment_evidence_mismatch")
    for key in ("capture_error", "live_error", "integrity_error", "score_error"):
        _require(key in value and value[key] is None, f"{key}_or_unconfirmed")
    _require(value.get("assessment_skipped_due_to_execution_error") is False, "assessment_unconfirmed")
    if "execution_error" in value:
        _require(value["execution_error"] is None, "execution_error")
    _require(value.get("case_id") == report["case_id"] and value.get("completion") == report["completion"] and
             value.get("experience_file") == report.get("experience") and
             value.get("experience_sha256") == report.get("experience_sha256") and
             value.get("rows") == report.get("rows") and value.get("capture_error", False) is None and
             value.get("score_error", False) is None and value.get("experience_stored_before_assessment") is True and
             value.get("physical_success_not_used_for_row_selection") is True and
             value.get("source_kind") == "existing_student_plus_motion_correction_experience" and
             value.get("direct_terra_demonstration") is False, "assessment_evidence_mismatch")
    equivalence = value.get("equivalence")
    _require(isinstance(equivalence, dict) and equivalence.get("identical") is True and
             isinstance(equivalence.get("compared"), list) and
             _REQUIRED_EQUIVALENCE.issubset(equivalence["compared"]) and
             equivalence.get("decision_count") == report.get("rows") and
             equivalence.get("transfer_count") == len(statuses), "equivalence_unconfirmed")
    scores = value.get("offline_scores")
    _require(isinstance(scores, list) and len(scores) == len(statuses), "assessment_incomplete")
    for index, (score, status) in enumerate(zip(scores, statuses), 1):
        _require(isinstance(score, dict) and score.get("label") == f"transfer_{index}" and
                 score.get("status") == status and score.get("physical_hold_matches") is True,
                 "assessment_evidence_mismatch")
        evidence = score.get("assessment")
        _require(isinstance(evidence, dict) and evidence.get("source") == "mujoco_simulation" and
                 evidence.get("simulated") is True and evidence.get("actual_hardware") is False and
                 evidence.get("success") is True and isinstance(evidence.get("checks"), dict) and
                 bool(evidence["checks"]) and all(v is True for v in evidence["checks"].values()) and
                 type(evidence.get("sample_count")) is int and evidence["sample_count"] == 501 and
                 type(evidence.get("stable_interval_seconds")) in (int, float) and
                 math.isfinite(evidence["stable_interval_seconds"]) and
                 math.isclose(evidence["stable_interval_seconds"], 1.0, abs_tol=1e-6, rel_tol=0),
                 "physical_success_unconfirmed")
    return value


def _decode_experience(base, report):
    name, expected = report.get("experience"), report.get("experience_sha256")
    _require(isinstance(name, str) and name.endswith(".experience.pt.gz") and
             "/" not in name and "\\" not in name and Path(name).name == name and _sha(expected),
             "invalid_experience_reference")
    path = base / name
    try:
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= _MAX_COMPRESSED,
                 "experience_file_missing_or_oversize")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected, "experience_hash_mismatch")
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            decoded = stream.read(_MAX_DECOMPRESSED + 1)
        _require(len(decoded) <= _MAX_DECOMPRESSED, "experience_oversize")
        value = torch.load(io.BytesIO(decoded), map_location="cpu", weights_only=True)
    except _Reject:
        raise
    except Exception as exc:
        # weights_only forbids arbitrary object construction. Corrupt or new
        # encodings are excluded, never retried with unrestricted unpickling.
        raise _Reject("experience_decode_error") from exc
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA, "unsupported_experience_schema")
    return value


def _convert(payload, report, assessment, manifest, train_groups, train_seed_range, statuses, collector_sha):
    _require(payload.get("source_kind") == "existing_student_plus_motion_correction_experience" and
             payload.get("direct_terra_demonstration") is False, "unsupported_experience_source")
    _require("capture_error" in payload and payload["capture_error"] is None, "capture_error_or_unconfirmed")
    _require("training_eligible" in payload and payload["training_eligible"] is report["training_eligible"],
             "training_eligibility_mismatch")
    if "execution_error" in payload:
        _require(payload["execution_error"] is None, "execution_error")
    provenance = payload.get("provenance")
    _require(isinstance(provenance, dict), "missing_provenance")
    for name, expected in (("plan_sha256", manifest["plan_sha256"]),
                           ("collector_source_sha256", collector_sha),
                           ("checkpoint_sha256", manifest["checkpoint_sha256"])):
        _require(provenance.get(name) == expected, "provenance_mismatch")
    _require(_sha(provenance.get("model_state_sha256")), "missing_model_state_fingerprint")
    for name, expected in (("only_training_groups_and_physics_seed", True), ("runtime_teacher", False),
                           ("live_object_or_contact_state_input", False),
                           ("physical_assessment_after_experience_persisted", True),
                           ("physical_assessment_used_for_target_selection", False),
                           ("expected_commit_is_physical_confirmation", False), ("actual_hardware", False)):
        _require(provenance.get(name) is expected, "provenance_mismatch")
    servo = provenance.get("servo_profile")
    _require(isinstance(servo, dict) and servo.get("ramp_ms") == 500 and
             servo.get("shape") == "minimum_jerk" and servo.get("threshold_deg") == 0.0,
             "unsupported_servo_profile")
    equivalence = payload.get("equivalence")
    _require(isinstance(equivalence, dict) and equivalence.get("identical") is True and
             isinstance(equivalence.get("compared"), list) and
             _REQUIRED_EQUIVALENCE.issubset(equivalence["compared"]), "equivalence_unconfirmed")
    case, registration = payload.get("case"), payload.get("registration")
    _require(isinstance(case, dict) and isinstance(registration, dict), "missing_registration")
    _require(case.get("case_id") == report["case_id"], "case_id_mismatch")
    for source in (case, provenance, report):
        if "split" in source:
            _require(source["split"] == "train", "non_training_split")
    group, seed = case.get("initial_group"), case.get("physics_seed")
    _require(group in train_groups and type(seed) is int and train_seed_range[0] <= seed <= train_seed_range[1],
             "non_training_group_or_seed")
    _require(assessment.get("initial_group") == group and assessment.get("physics_seed") == seed and
             assessment.get("registration_id") == case.get("registration_id"), "assessment_evidence_mismatch")
    current_scene = _scene(case.get("initial_scene"), "invalid_initial_scene")
    _require(registration.get("event") == "registered" and registration.get("memory_cleared") is True and
             registration.get("scene_source") == "explicit_registration" and
             registration.get("physical_assessment_performed") is False and
             registration.get("control_target_source") == "validated_explicit_registration" and
             _scene(registration.get("scene"), "invalid_registration_scene") == current_scene,
             "invalid_registration")
    slots = registration.get("nominal_slot_xy_mm")
    _require(slots == case.get("registered_slot_xy_mm"), "registered_slots_mismatch")
    initial = payload.get("initial_targets")
    _require(isinstance(initial, dict), "initial_control_unconfirmed")
    previous = _vector(initial.get("joints_deg"), "initial_control_unconfirmed")
    opening = initial.get("gripper_opening_m")
    _require(type(opening) in (int, float) and math.isfinite(opening) and abs(opening - 0.04) < 1e-9,
             "initial_control_unconfirmed")
    gripper, previous_action = "open", None
    rows, stages, moves = payload.get("rows"), payload.get("stages"), case.get("moves")
    _require(isinstance(rows, list) and isinstance(stages, list) and isinstance(moves, list) and
             len(stages) == len(moves) == len(statuses) and len(rows) == report.get("rows") and
             equivalence.get("decision_count") == len(rows) and equivalence.get("transfer_count") == len(stages),
             "episode_shape_mismatch")
    all_features, actions, move_targets, applied_moves = [], [], [], []
    cursor = 0
    for stage_index, (stage, move) in enumerate(zip(stages, moves)):
        _require(isinstance(stage, dict) and isinstance(move, dict) and stage.get("error") is None and
                 type(stage.get("row_start")) is int and stage["row_start"] == cursor and
                 type(stage.get("row_end")) is int and cursor < stage["row_end"] <= len(rows),
                 "stage_capture_invalid")
        end = stage["row_end"]
        event = stage.get("event")
        _require(isinstance(event, dict) and event.get("event") == "transfer" and
                 event.get("committed") is True and event.get("commit_kind") == "expected_only" and
                 event.get("physical_success") is None, "invalid_expected_commit")
        result = event.get("result")
        _require(isinstance(result, dict) and result.get("termination") == "model_done" and
                 result.get("physical_success") is None and result.get("teacher_assistance_count") == 0 and
                 result.get("actual_hardware") is False and isinstance(result.get("decisions"), list) and
                 len(result["decisions"]) == end - cursor, "stage_result_unconfirmed")
        _require(_scene(event.get("before"), "invalid_expected_scene") == current_scene and
                 _scene(move.get("before"), "invalid_expected_scene") == current_scene and
                 move.get("index") == stage_index + 1 and move.get("task") == event.get("task"),
                 "transfer_sequence_mismatch")
        task = event.get("task")
        try:
            before, goal = scene_features.transfer_task(current_scene, task)
            next_scene = task_catalog.transfer(before, goal["source_slot"], goal["destination_slot"])
            computed_group = scene_policy.split_group_for(before, goal)
            dummy = {"schema_version": "PolicyObservation.phase1", "joints_deg": [0, -10, -110, 30],
                     "sensor_valid": True, "moving": False, "gripper_command": "open",
                     "gripper_reported_state": "open", "gripper_reported_valid": True,
                     "gripper_feedback_fresh": True}
            scene_row = torch.tensor(scene_features.encode_scene_features(dummy, before, goal, slots)[14:],
                                     dtype=torch.float32)
        except (TypeError, ValueError, KeyError):
            raise _Reject("invalid_registered_task") from None
        _require(computed_group in train_groups and event.get("split_group") == computed_group and
                 move.get("split_group") == computed_group, "non_training_group_or_seed")
        if stage_index == 0:
            _require(computed_group == group and case.get("source") == goal["source_slot"] and
                     case.get("destination") == goal["destination_slot"], "initial_task_scope_mismatch")
        _require(_scene(event.get("planned_after"), "invalid_expected_scene") == next_scene and
                 _scene(move.get("expected_after"), "invalid_expected_scene") == next_scene,
                 "transfer_sequence_mismatch")
        physical = assessment["offline_scores"][stage_index]["assessment"]
        _require(physical.get("source_slot") == goal["source_slot"] and
                 physical.get("destination_slot") == goal["destination_slot"] and
                 physical.get("moved_object_id") == goal["object_id"] and
                 _scene(physical.get("before_layout"), "assessment_evidence_mismatch") == before and
                 _scene(physical.get("after_layout"), "assessment_evidence_mismatch") == next_scene,
                 "assessment_evidence_mismatch")
        history = []
        for offset, index in enumerate(range(cursor, end)):
            row = rows[index]
            _require(isinstance(row, dict) and row.get("capture_status") == "decision_recorded", "row_capture_invalid")
            decision = row.get("decision")
            _require(isinstance(decision, dict) and decision == result["decisions"][offset] and
                     decision.get("index") == offset, "decision_mismatch")
            window, mask = row.get("policy_window"), row.get("policy_mask")
            _require(isinstance(window, torch.Tensor) and window.device.type == "cpu" and
                     window.dtype == torch.float32 and window.shape == (8, 82) and
                     isinstance(mask, torch.Tensor) and mask.device.type == "cpu" and
                     mask.dtype == torch.bool and mask.shape == (8,), "invalid_policy_window")
            try:
                observed = torch.tensor(decision["policy_features"], dtype=torch.float32)
            except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
                raise _Reject("invalid_policy_observation") from None
            _require(observed.shape == (82,), "invalid_policy_observation")
            _observation(observed, scene_row, gripper)
            history.append(observed)
            n = min(len(history), 8)
            expected_window = torch.zeros((8, 82), dtype=torch.float32)
            expected_window[-n:] = torch.stack(history[-n:])
            expected_mask = torch.tensor([False] * (8 - n) + [True] * n)
            _require(torch.equal(window, expected_window) and torch.equal(mask, expected_mask) and
                     decision.get("valid_history") == expected_mask.tolist(), "noncausal_or_invalid_window")
            stored_previous = _vector(row.get("previous_applied_target_deg"), "invalid_previous_target")
            decision_previous = _vector(decision.get("previous_applied_target_deg"), "invalid_previous_target")
            # The recorded model input is the float32 form of the previous
            # transmitted target.  The independently confirmed servo return
            # can differ in float64 by a few ULPs; rounding both independently
            # to float32 is not a reliable equality test at a midpoint.
            _require(torch.equal(torch.tensor(decision_previous, dtype=torch.float32),
                                 torch.tensor(stored_previous, dtype=torch.float32)) and
                     _same(previous, decision_previous), "previous_target_mismatch")
            prev_one_hot = [float(previous_action == action) for action in student.PREVIOUS_APPLIED_ACTIONS]
            # Match encode_step: normalize the verified transmitted double
            # first and only then convert the feature to float32.
            extras = torch.tensor(prev_one_hot + [v / motion_policy.JOINT_SCALE_DEG for v in decision_previous] +
                                  [float(offset == 0), float(index == 0)], dtype=torch.float32)
            all_features.append(torch.cat((observed, extras)))
            action = decision.get("action")
            _require(action in motion_policy.ACTIONS, "unknown_action")
            guard = decision.get("guard")
            _require(isinstance(guard, dict) and guard.get("model_action") == action, "command_return_unconfirmed")
            target = [float("nan")] * 4
            is_move = False
            if action == "done":
                _require(decision.get("result") == "done_no_command" and "command" not in decision and
                         guard.get("actual_applied_joints_deg") is None and index == end - 1,
                         "command_return_unconfirmed")
            else:
                command = decision.get("command")
                _require(decision.get("result") == "command_applied" and isinstance(command, dict) and
                         command.get("source") == "shared_student_policy" and "attempted_command" not in decision,
                         "command_return_unconfirmed")
                sent = _vector(command.get("joints_deg"), "command_return_unconfirmed")
                actual = _vector(guard.get("actual_applied_joints_deg"), "command_return_unconfirmed")
                _require(_same(sent, actual), "applied_command_mismatch")
                if action == "move":
                    guarded = _vector(guard.get("guard_returned_target_deg"), "guard_return_unconfirmed")
                    _require(_same(guarded, sent), "guard_command_mismatch")
                duration, next_opening = command.get("seconds"), command.get("gripper_opening_m")
                _require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0 and
                         type(next_opening) in (int, float) and math.isfinite(next_opening),
                         "command_return_unconfirmed")
                expected_opening = 0.04 if action == "open" else 0.0 if action == "close" else opening
                _require(abs(next_opening - expected_opening) < 1e-9 and
                         (action == "move" or _same(previous, actual)), "applied_command_mismatch")
                if action == "move":
                    target, is_move = actual, True
                else:
                    gripper = "open" if action == "open" else "closed"
                previous, previous_action, opening = actual, action, next_opening
            actions.append(motion_policy.ACTIONS.index(action))
            move_targets.append(target)
            applied_moves.append(is_move)
        _require(actions[-1] == motion_policy.DONE, "missing_done")
        cursor, current_scene = end, next_scene
    _require(cursor == len(rows), "episode_shape_mismatch")
    length = len(actions)
    return TrainingEpisode(torch.stack(all_features), torch.tensor(actions, dtype=torch.long),
                           torch.tensor(move_targets, dtype=torch.float32), torch.ones(length, dtype=torch.bool),
                           torch.ones(length, dtype=torch.bool), torch.tensor(applied_moves, dtype=torch.bool),
                           {"case_id": case["case_id"], "experience": report["experience"],
                            "experience_sha256": report["experience_sha256"], "group": group, "seed": seed,
                            "assessment": report["assessment"], "assessment_sha256": report["assessment_sha256"],
                            "assessment_status_by_transfer": tuple(statuses),
                            "post_transfer_scene": "expected_only"})


def load_training_episodes(summary_path, *, manifest_path, split_plan_path, fingerprints):
    """Return admitted episodes and retained exclusions from a pinned v2 run.

    The three fingerprints must come from a trusted review/manifest process,
    independently of the input files.  This function checks the supported v2
    schema, manifest-selected cases, collector and plan provenance, all train
    groups and the train seed range.  Unsupported formats are rejected rather
    than guessed.  Source files, including failed cases, remain untouched.
    """
    if not isinstance(fingerprints, TrustedFingerprints) or not _sha(fingerprints.collector_sha256):
        raise ValueError("trusted manifest, summary and supported collector fingerprints are required")
    manifest = _read_pinned_json(manifest_path, fingerprints.manifest_sha256, "manifest")
    summary = _read_pinned_json(summary_path, fingerprints.summary_sha256, "summary")
    if manifest.get("schema") != SCHEMA or summary.get("schema") != SCHEMA:
        raise ValueError("unsupported collection schema")
    source = manifest.get("source_sha256")
    ids = manifest.get("planned_case_ids")
    if (not isinstance(source, dict) or source.get(COLLECTOR_KEY) != fingerprints.collector_sha256 or
            not _sha(source.get(SPLIT_KEY)) or not _sha(manifest.get("plan_sha256")) or
            not _sha(manifest.get("checkpoint_sha256")) or
            manifest.get("selected_before_physical_execution") is not True or
            manifest.get("scoring_used_to_choose_later_cases") is not False or
            manifest.get("training_performed") is not False or
            not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError("unsupported or untrusted collection manifest")
    train_groups, train_seed_range = _split_scope(split_plan_path, source[SPLIT_KEY])
    mode = manifest.get("mode")
    if (not isinstance(mode, str) or not mode or any(word in mode.lower() for word in ("pilot", "validation", "test")) or
            summary.get("mode") != mode or
            ("split" in summary and summary["split"] != "train") or
            ("split" in manifest and manifest["split"] != "train") or
            summary.get("source_kind") != "existing_student_plus_motion_correction_experience" or
            summary.get("direct_terra_demonstration") is not False or
            summary.get("experience_saved_before_scoring") is not True or
            summary.get("scoring_used_to_choose_later_cases") is not False or
            summary.get("training_performed") is not False):
        raise ValueError("unsupported or incomplete collection summary")
    reports = summary.get("reports")
    if (not isinstance(reports, list) or summary.get("planned_case_count") != len(ids) or
            summary.get("returned_case_count") != len(reports) or
            any(not isinstance(r, dict) or not isinstance(r.get("case_id"), str) for r in reports)):
        raise ValueError("invalid collection summary reports")
    report_ids = [r["case_id"] for r in reports]
    if len(set(report_ids)) != len(report_ids) or not set(report_ids).issubset(ids):
        raise ValueError("summary cases differ from the trusted plan")
    states = summary.get("case_states")
    state_map = {}
    if isinstance(states, list) and all(isinstance(s, dict) and isinstance(s.get("case_id"), str) for s in states):
        state_ids = [s["case_id"] for s in states]
        if (len(state_ids) == len(set(state_ids)) and set(state_ids) == set(ids) and
                all(s.get("execution_status") in ("executed", "execution_error", "not_started") for s in states) and
                type(summary.get("not_started_case_count")) is int and
                type(summary.get("execution_error_case_count")) is int and
                summary["not_started_case_count"] == sum(s["execution_status"] == "not_started" for s in states) and
                summary["execution_error_case_count"] == sum(s["execution_status"] == "execution_error" for s in states)):
            state_map = {s["case_id"]: s for s in states}
    episodes, excluded = [], []
    base = Path(summary_path).resolve().parent
    summary_error = (summary.get("status") != "returned" or "error" not in summary or
                     summary["error"] is not None or
                     ("execution_error" in summary and summary["execution_error"] is not None))
    for report in reports:
        state = state_map.get(report["case_id"])
        try:
            _require(not summary_error, "summary_execution_error_or_unconfirmed")
            _case_execution_state(report, state)
            statuses = _assessment(report)
            assessment = _assessment_file(base, report, statuses)
            payload = _decode_experience(base, report)
            episode = _convert(payload, report, assessment, manifest, train_groups, train_seed_range,
                               statuses, fingerprints.collector_sha256)
        except _Reject as exc:
            excluded.append(Exclusion(report["case_id"], str(exc), dict(report), dict(state) if state else None))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            excluded.append(Exclusion(report["case_id"], f"invalid_experience: {type(exc).__name__}", dict(report),
                                      dict(state) if state else None))
        else:
            episodes.append(episode)
    for case_id in ids:
        if case_id not in report_ids:
            excluded.append(Exclusion(case_id, "summary_execution_error_or_unconfirmed" if summary_error
                                      else "case_not_returned", {}, state_map.get(case_id)))
    return DatasetResult(tuple(episodes), tuple(excluded), fingerprints.summary_sha256, fingerprints.manifest_sha256)
