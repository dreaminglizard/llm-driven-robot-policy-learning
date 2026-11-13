"""Gate and convert the fixed public model-plus-program teacher collection."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from . import motion_policy, scene_features, scene_policy, task_catalog
from .scene_registration_distilled_data import build_registration

SCHEMA = "public-corrected-teacher-192.v1"
ROOT = Path(__file__).resolve().parents[1]
TEACHER_CHECKPOINT = ROOT / "artifacts/architecture_research/20261103_grasp_placement_student_1101_v1/candidate.pt"
KINDS = ("move_to_empty", "stack", "unstack", "transfer_between_stacks")
DIRECTIONS = tuple((s, d) for s in range(1, 5) for d in range(1, 5) if s != d)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gz_json(path, expected):
    _need(_sha(path) == expected, "assessment fingerprint differs")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _gz_torch(path, expected):
    _need(_sha(path) == expected, "experience fingerprint differs")
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def _safe_name(value):
    _need(isinstance(value, str) and value == Path(value).name and "/" not in value and "\\" not in value,
          "collection file name is invalid")
    return value


def _same_angles(a, b, tolerance=1e-7):
    # Compare the recorded double values before float32 conversion.  Two nearly
    # identical doubles can straddle a float32 midpoint and differ by one ULP.
    return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
            and len(a) == len(b) == 4
            and all(type(x) in (int, float) and type(y) in (int, float)
                    and math.isfinite(x) and math.isfinite(y) and abs(x-y) <= tolerance
                    for x, y in zip(a, b)))


def verify_gate(directory):
    """Recompute the 192-case and 48-combination success gate from reports.

    Reports select successful registrations only; their physical assessment
    contents are never transformed into a policy observation or action label.
    """
    directory = Path(directory)
    plan_path, summary_path, live_path = (directory / name for name in ("plan.json", "summary.json", "live_complete.json"))
    plan_sha = _sha(plan_path)
    plan, summary, live = (json.loads(path.read_text(encoding="utf-8")) for path in (plan_path, summary_path, live_path))
    _need(all(x.get("schema") == SCHEMA for x in (plan, summary, live))
          and summary.get("plan_sha256") == live.get("plan_sha256") == plan_sha
          and plan.get("source_kind") == summary.get("source_kind") == "model_plus_program"
          and plan.get("selection_before_policy_execution") is True and plan.get("live_scoring") is False
          and plan.get("actual_hardware") is False and plan.get("reserved_data_access") is False
          and summary.get("no_case_replacement") is True and summary.get("actual_hardware") is False
          and summary.get("training_performed") is False
          and live.get("assessment_performed") is False and live.get("returned_count") == 192,
          "collection evidence or completed-live gate differs")
    cases, reports, records = plan["cases"], summary["reports"], live["cases"]
    _need(len(cases) == len(reports) == len(records) == 192 and summary.get("planned_registrations") == 192
          and summary.get("planned_moves") == 288 and sum(len(c["moves"]) for c in cases) == 288
          and len({c["case_id"] for c in cases}) == 192,
          "collection denominator or case identities differ")
    train_groups = set(plan.get("train_groups", []))
    _need(bool(train_groups) and all(type(c.get("physics_seed")) is int and 1000 <= c["physics_seed"] <= 1099
          and all(m.get("split_group") in train_groups for m in c["moves"]) for c in cases),
          "a collection case is outside the declared training scope")
    design = Counter((c["kind"], c["source"], c["destination"], len(c["moves"])) for c in cases)
    _need(all(design[k, s, d, n] == 2 for k in KINDS for s, d in DIRECTIONS for n in (1, 2)),
          "fixed 48-combination collection design differs")
    _need(all(len(c["moves"]) != 2 or
              (c["moves"][1]["task"]["source_slot"], c["moves"][1]["task"]["destination_slot"])
              == (c["destination"], c["source"]) for c in cases),
          "fixed second-move reversal design differs")
    source_files = plan.get("source_sha256")
    _need(isinstance(source_files, dict) and bool(source_files), "collection source manifest is missing")
    pinned_sources = {}
    for relative, expected in source_files.items():
        path = (ROOT / relative).resolve()
        _need(path.is_relative_to(ROOT.resolve()) and path.is_file() and _sha(path) == expected,
              "collection program fingerprint differs")
        pinned_sources[str(path)] = expected
    _need(TEACHER_CHECKPOINT.is_file() and _sha(TEACHER_CHECKPOINT) == plan.get("checkpoint_sha256"),
          "collection teacher checkpoint differs")
    pinned_sources[str(TEACHER_CHECKPOINT.resolve())] = plan["checkpoint_sha256"]
    coverage, counts, chosen = Counter(), Counter(), []
    hashes = {str(p.resolve()): _sha(p) for p in (plan_path, summary_path, live_path)}
    hashes.update(pinned_sources)
    for i, (case, report, record) in enumerate(zip(cases, reports, records)):
        name = case["case_id"]
        _need(name == report.get("case_id") == record.get("case_id")
              and report.get("kind") == case["kind"]
              and report.get("direction") == f"{case['source']}->{case['destination']}"
              and report.get("experience") == record.get("experience")
              and report.get("experience_sha256") == record.get("experience_sha256")
              and record.get("live_error") is None and record.get("capture_error") is None,
              "collection case evidence differs")
        score_path = directory / "scores" / _safe_name(report["assessment"])
        score = _gz_json(score_path, report["assessment_sha256"])
        hashes[str(score_path.resolve())] = report["assessment_sha256"]
        _need(score.get("schema") == SCHEMA and score.get("case_id") == name
              and score.get("plan_sha256") == plan_sha and score.get("source_kind") == "model_plus_program"
              and score.get("experience_sha256") == report["experience_sha256"]
              and score.get("all_live_finished_before_scoring") is True,
              "assessment does not bind the public experience")
        scores = score.get("scores", [])
        valid = (len(scores) == len(case["moves"]) and all(
            x.get("status") == "success" and x.get("physical_hold_matches") is True
            and isinstance(x.get("assessment"), dict) and x["assessment"].get("sample_count") == 501
            and math.isclose(x["assessment"].get("stable_interval_seconds", 0.), 1., abs_tol=1e-6, rel_tol=0)
            for x in scores))
        status = "success" if valid else "failure" if any(x.get("status") == "failure" for x in scores) else "unconfirmed"
        _need(status == score.get("status") == report.get("status"), "assessment status differs")
        counts[status] += 1
        if valid:
            _need(score.get("score_error") is None, "successful registration has a scoring error")
            for number, (move, scored) in enumerate(zip(case["moves"], scores), 1):
                physical = scored["assessment"]
                _need(scored.get("label") == f"transfer_{number}" and physical.get("success") is True
                      and physical.get("source_slot") == move["task"]["source_slot"]
                      and physical.get("destination_slot") == move["task"]["destination_slot"]
                      and physical.get("moved_object_id") == move["task"]["object_id"]
                      and task_catalog.scene(physical.get("before_layout")) == task_catalog.scene(move["before"])
                      and task_catalog.scene(physical.get("after_layout")) == task_catalog.scene(move["expected_after"]),
                      "successful assessment does not match the expected task")
            coverage[case["kind"], case["source"], case["destination"]] += 1
            experience_path = directory / "live" / _safe_name(report["experience"])
            _need(_sha(experience_path) == report["experience_sha256"], "successful experience fingerprint differs")
            hashes[str(experience_path.resolve())] = report["experience_sha256"]
            chosen.append((case, report, experience_path))
    expected_coverage = {f"{k}:{s}->{d}": coverage[k, s, d] for k in KINDS for s, d in DIRECTIONS}
    _need(dict(counts) == summary.get("status_counts")
          and expected_coverage == summary.get("successful_registrations_by_kind_direction")
          and summary.get("successful_experience_files") == [r["experience"] for _, r, _ in chosen],
          "summary differs from independent report counts")
    _need(counts["success"] >= 144 and all(coverage[k, s, d] >= 1 for k in KINDS for s, d in DIRECTIONS)
          and summary.get("collection_success_gate_passed") is True and summary.get("zero_success_combinations") == [],
          "training gate failed: fewer than 144 successes or a missing combination")
    return chosen, {"plan_sha256": plan_sha, "summary_sha256": _sha(summary_path),
                    "status_counts": dict(counts), "successful_by_kind_direction": expected_coverage,
                    "source_files_sha256": hashes, "planned_registrations": 192,
                    "train_groups": sorted(train_groups),
                    "two_move_limitation": "all second moves reverse the first; not all direction pairs are covered"}


def convert_experience(payload, case, expected_plan_sha, expected_experience_sha):
    """Convert one successful, fully verified public trace; ignore private fields."""
    _need(payload.get("schema") == SCHEMA and payload.get("source_kind") == "model_plus_program"
          and payload.get("direct_terra_demonstration") is False and payload.get("plan_sha256") == expected_plan_sha
          and payload.get("case") == case and payload.get("capture_error") is None and payload.get("live_error") is None,
          "public registration provenance differs")
    registration = payload.get("registration", {})
    _need(registration.get("event") == "registered" and registration.get("scene_source") == "explicit_registration"
          and registration.get("physical_assessment_performed") is False
          and registration.get("control_target_source") == "validated_explicit_registration"
          and registration.get("nominal_slot_xy_mm") == case["registered_slot_xy_mm"]
          and task_catalog.scene(registration.get("scene")) == task_catalog.scene(case["initial_scene"]),
          "explicit registration differs")
    stages, raw_rows, samples = payload["stages"], payload["rows"], payload["public_observations"]
    _need(len(stages) == len(case["moves"]) and isinstance(samples, list), "transfer count differs")
    sample_index, consumed_samples = {}, set()
    last_time, last_transfer = -float("inf"), 0
    for seq, sample in enumerate(samples):
        obs = sample.get("observation", {})
        timestamp = obs.get("observed_monotonic")
        transfer = sample.get("transfer_index")
        _need(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= last_time
              and type(transfer) is int and 1 <= transfer <= len(stages) and transfer >= last_transfer,
              "public observation timestamps or transfer order differ")
        last_time, last_transfer = timestamp, transfer
        if sample.get("phase") in ("pre_decision", "post_command"):
            key = (sample.get("transfer_index"), sample.get("decision_index"), sample["phase"])
            _need(key not in sample_index, "duplicated public observation")
            sample_index[key] = (seq, sample)
    rows, previous, actions, targets, starts, applied, applied_targets, committed, returns = ([] for _ in range(9))
    cursor = 0
    scene = task_catalog.scene(case["initial_scene"])
    prior = list(payload["initial_targets"]["joints_deg"])
    opening = payload["initial_targets"]["gripper_opening_m"]
    _need(type(opening) in (int, float) and math.isfinite(opening)
          and math.isclose(opening, .04, abs_tol=1e-9, rel_tol=0), "initial gripper control differs")
    gripper = "open"
    for number, (stage, move) in enumerate(zip(stages, case["moves"]), 1):
        event = stage["event"]
        result = event.get("result") or {}
        _need(stage.get("error") is None and stage.get("row_start") == cursor and stage.get("row_end", cursor) > cursor
              and event.get("committed") is True and event.get("commit_kind") == "expected_only"
              and event.get("physical_success") is None and result.get("termination") == "model_done"
              and result.get("physical_success") is None and result.get("teacher_assistance_count") == 0
              and result.get("actual_hardware") is False and task_catalog.scene(event.get("before")) == scene
              and move.get("index") == number and event.get("task") == move.get("task"),
              "expected transfer and captured stage differ")
        before, task = scene_features.transfer_task(scene, move["task"])
        following = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
        _need(following == task_catalog.scene(move["expected_after"]) == task_catalog.scene(event["planned_after"])
              and scene_policy.split_group_for(before, task) == move["split_group"] == event["split_group"],
              "expected scene or task differs")
        decisions = result.get("decisions", [])
        _need(len(decisions) == stage["row_end"] - cursor, "decision count differs")
        history, previous_sequence = [], -1
        for index, decision in enumerate(decisions):
            record = raw_rows[cursor + index]
            _need(record.get("capture_status") == "decision_recorded" and record.get("decision") == decision
                  and decision.get("index") == index, "decision trace differs")
            key = (number, index, "pre_decision")
            _need(key in sample_index, "public pre-decision sample is missing")
            sequence, sample = sample_index[key]
            consumed_samples.add(key)
            _need(sequence > previous_sequence and sample.get("confirmed_applied") is None, "public observation order differs")
            obs = sample["observation"]
            _need(obs.get("gripper_command") == gripper, "public gripper command differs from the applied history")
            encoded = torch.tensor(scene_features.encode_scene_features(obs, before, task, case["registered_slot_xy_mm"]), dtype=torch.float32)
            window, mask = record["policy_window"], record["policy_mask"]
            history.append(encoded.tolist())
            expected_window, expected_mask = scene_features.history_window(history, 8)
            _need(torch.equal(window, torch.tensor(expected_window, dtype=torch.float32))
                  and torch.equal(mask, torch.tensor(expected_mask, dtype=torch.bool))
                  and decision["valid_history"] == expected_mask
                  and torch.equal(encoded, torch.tensor(decision["policy_features"], dtype=torch.float32))
                  and torch.allclose(torch.as_tensor(prior), record["previous_applied_target_deg"], atol=4e-5, rtol=0),
                  "public observation or prior control differs")
            action = motion_policy.ACTIONS.index(decision["action"])
            guard = decision.get("guard", {})
            _need(isinstance(guard, dict) and guard.get("model_action") == decision["action"],
                  "guard action and confirmed decision differ")
            target, after = [0.0] * 4, None
            if action == motion_policy.DONE:
                _need(index == len(decisions)-1 and decision.get("result") == "done_no_command"
                      and "command" not in decision and (number, index, "post_command") not in sample_index,
                      "END is not a physical command")
                actual = None
            else:
                _need(decision.get("result") == "command_applied" and isinstance(decision.get("command"), dict)
                      and decision["command"].get("source") == "shared_student_policy"
                      and decision["command"].get("seconds") == 1., "command was not confirmed")
                actual = guard.get("actual_applied_joints_deg")
                _need(_same_angles(actual, decision["command"]["joints_deg"]),
                      "guard and actual applied target differ")
                next_opening = decision["command"].get("gripper_opening_m")
                expected_opening = .04 if action == motion_policy.OPEN else 0. if action == motion_policy.CLOSE else opening
                _need(type(next_opening) in (int, float) and math.isfinite(next_opening)
                      and math.isclose(next_opening, expected_opening, abs_tol=1e-9, rel_tol=0),
                      "confirmed gripper opening differs from the action")
                if action == motion_policy.MOVE:
                    target = [v / 180. for v in actual]
                    _need(_same_angles(actual, guard["guard_returned_target_deg"]),
                          "MOVE label is not the protected applied control")
                post_key = (number, index, "post_command")
                _need(post_key in sample_index, "confirmed command has no recorded return-time observation")
                consumed_samples.add(post_key)
                post_seq, post = sample_index[post_key]
                post_obs = post.get("observation", {})
                _need(post_seq > sequence and post.get("confirmed_applied") is True
                      and post_obs.get("sensor_valid") is True
                      and post_obs.get("observed_monotonic", -1) > obs["observed_monotonic"],
                      "invalid return-time observation")
                after = post_obs["joints_deg"]
                previous_sequence = post_seq
                opening = next_opening
                if action != motion_policy.MOVE:
                    gripper = "open" if action == motion_policy.OPEN else "closed"
            rows.append(encoded); previous.append(torch.tensor(prior)); actions.append(action); targets.append(target)
            starts.append(index == 0); applied.append(action); applied_targets.append(actual)
            committed.append(action == motion_policy.DONE); returns.append(after)
            if actual is not None:
                prior = list(actual)
        cursor = stage["row_end"]
        scene = following
    _need(cursor == len(raw_rows) and consumed_samples == set(sample_index),
          "unmatched captured rows or public decision observations")
    return build_registration(rows=torch.stack(rows), previous=torch.stack(previous), actions=actions, targets=targets,
        labels=[True]*len(actions), instruction_start=starts, applied_actions=applied,
        applied_targets_deg=applied_targets, committed=committed, return_joints=returns,
        evidence={"source": "model_plus_program", "case_id": case["case_id"],
                  "kind_direction": (case["kind"], case["source"], case["destination"]),
                  "experience_sha256": expected_experience_sha})


def load_successful(directory):
    chosen, evidence = verify_gate(directory)
    groups = defaultdict(list)
    for case, report, path in chosen:
        payload = _gz_torch(path, report["experience_sha256"])
        episode = convert_experience(payload, case, evidence["plan_sha256"], report["experience_sha256"])
        groups[case["kind"], case["source"], case["destination"]].append(episode)
    _need(len(groups) == 48 and sum(map(len, groups.values())) >= 144, "successful conversion coverage differs")
    return dict(groups), evidence
