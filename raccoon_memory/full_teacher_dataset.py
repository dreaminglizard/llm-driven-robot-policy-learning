"""Export independently reviewed full-teacher decisions without running a model.

The caller must supply an existing, completed review/selection pair.  This
module never creates a selection from an incomplete report, modifies source
evidence, runs a simulator, or trains a policy.  The training window is a
copy of the verified 8x82 original with only the moving column set to zero.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from raccoon_memory import motion_policy, scene_features
from scripts import plan_scene_full_teacher_collection as planner
from scripts import review_scene_full_teacher_collection as review
from scripts import scene_full_teacher_collection as collection

DATASET_SCHEMA = "raccoon-full-teacher-dataset.v1"
ROW_SCHEMA = "raccoon-full-teacher-decision.v1"
RAW_SCHEMA = "raccoon-full-teacher-original-input.v1"
MOVING_INDEX = 5
HISTORY_LENGTH = 8
FEATURE_COUNT = 82
PLANNED_CASES = 128


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    return _sha_bytes(Path(path).read_bytes())


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return collection.operator._strict_json(raw, str(path)), raw


def _digest_value(value):
    """Digest the canonical UTF-8 JSON representation, not FP32 tensor bytes."""
    return _sha_bytes(_json_bytes(value))


def _unique_ids(value, count, name):
    need(isinstance(value, list) and len(value) == count and
         all(isinstance(x, str) and x for x in value) and len(set(value)) == count,
         f"{name} must contain {count} distinct trial IDs")
    return value


def validate_review_selection(cases, report, selection_file, expected_config, report_sha256):
    """Check a published selection; do not select when the artifact is missing.

    Once the completed artifacts are present, recomputing the fixed rule is a
    read-only consistency check.  Its output is never used to choose samples.
    """
    need(isinstance(expected_config, str) and re.fullmatch(r"[0-9a-f]{64}", expected_config),
         "reviewed configuration SHA-256 is required")
    need(isinstance(cases, list) and len(cases) == PLANNED_CASES, "fixed plan must have 128 cases")
    case_ids = _unique_ids([c.get("trial_id") for c in cases if isinstance(c, dict)], PLANNED_CASES, "plan")
    by_id = dict(zip(case_ids, cases))
    need(isinstance(report, dict) and isinstance(selection_file, dict), "review artifacts must be objects")
    need(report.get("schema") == review.REPORT_SCHEMA and report.get("status") == "complete" and
         report.get("config_sha256") == expected_config and report.get("planned") == PLANNED_CASES and
         isinstance(report.get("initialization"), dict) and report["initialization"].get("status") == "completed" and
         report.get("changed_sources") == [] and report.get("review_index_error") is None and
         report.get("review_source") == "external_json_only" and report.get("automatic_review_generated") is False,
         "completed independent review is required")
    need(selection_file.get("schema") == review.REPORT_SCHEMA and
         selection_file.get("config_sha256") == expected_config and
         selection_file.get("report_sha256") == report_sha256 and
         selection_file.get("source_hashes") == report.get("source_hashes") and
         selection_file.get("selection") == report.get("selection"), "published selection and review differ")
    sources = report.get("source_hashes")
    need(isinstance(sources, dict) and bool(sources) and
         all(isinstance(k, str) and k and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
             for k, v in sources.items()), "review source fingerprints are missing")
    selected = selection_file.get("selection")
    need(isinstance(selected, dict) and selected.get("status") == "complete" and
         selected.get("planned") == PLANNED_CASES, "selection is incomplete")
    ids100 = _unique_ids(selected.get("selected_100"), 100, "100-set")
    ids10 = _unique_ids(selected.get("selected_10"), 10, "10-set")
    need(set(ids10) <= set(ids100) <= set(case_ids), "10-set must be contained in the selected 100-set")
    need(selected.get("remaining_90") == [x for x in ids100 if x not in set(ids10)],
         "remaining 90 does not match the selected subsets")
    need({by_id[x]["split_group"] for x in ids10} == set(planner.GROUPS), "10-set misses a training group")

    outcomes = report.get("outcomes")
    details = report.get("cases")
    need(isinstance(outcomes, list) and isinstance(details, list), "review outcomes or case details missing")
    _unique_ids([x.get("trial_id") for x in outcomes if isinstance(x, dict)], PLANNED_CASES, "outcomes")
    _unique_ids([x.get("trial_id") for x in details if isinstance(x, dict)], PLANNED_CASES, "review details")
    outcome_by_id = {x["trial_id"]: x for x in outcomes}
    detail_by_id = {x["trial_id"]: x for x in details}
    need(set(outcome_by_id) == set(detail_by_id) == set(case_ids), "review cases differ from fixed plan")
    for tid, outcome in outcome_by_id.items():
        need(all(outcome.get(key) is True for key in ("admitted", "started", "returned", "terminal",
                 "source_review_complete", "quality_review_complete")), f"{tid}: outcome or review incomplete")
        need(type(outcome.get("eligible")) is bool and
             outcome.get("physical_status") in ("success", "failure", "not_assessed"),
             f"{tid}: eligibility or physical status is unconfirmed")
        for kind in ("source", "quality"):
            evidence = outcome.get(f"{kind}_review_evidence")
            need(isinstance(evidence, dict) and evidence.get("trial_id") == tid and
                 evidence.get("episode_events_sha256") == outcome.get("episode_events_sha256") and
                 evidence.get("verdict") in ("pass", "fail", "not_applicable") and
                 isinstance(evidence.get("reviewer"), str) and bool(evidence["reviewer"].strip()) and
                 evidence["reviewer"] != by_id[tid]["assigned_selector"] and
                 isinstance(evidence.get("path"), str) and bool(evidence["path"]) and
                 isinstance(evidence.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"]),
                 f"{tid}: independent {kind} review evidence differs")
        if tid in ids100:
            detail = detail_by_id[tid]
            need(outcome["eligible"] is True and outcome["physical_status"] == "success" and
                 all(outcome[f"{kind}_review_evidence"]["verdict"] == "pass" for kind in ("source", "quality")) and
                 detail.get("automatic_quality") is True and detail.get("retreat_status") == "selected_and_observed" and
                 detail.get("assessment_sample_count") == 501, f"{tid}: selected episode is not eligible")
    # This is deliberately after the existing completed artifact was checked.
    need(planner.select_subsets(cases, outcomes) == selected, "published fixed selection does not match its evidence")
    return {"selected_100": list(ids100), "selected_10": list(ids10), "outcomes": outcome_by_id,
            "details": detail_by_id, "source_hashes": dict(sources)}


def zero_moving(original_window, original_mask):
    """Copy a validated original window and hide only feature index 5."""
    need(scene_features.HISTORY_LENGTH == HISTORY_LENGTH and len(scene_features.FEATURE_NAMES) == FEATURE_COUNT and
         scene_features.FEATURE_NAMES[MOVING_INDEX] == "moving", "feature contract differs")
    need(isinstance(original_window, list) and len(original_window) == HISTORY_LENGTH and
         isinstance(original_mask, list) and len(original_mask) == HISTORY_LENGTH and
         all(type(x) is bool for x in original_mask), "window or mask shape differs")
    length = sum(original_mask)
    need(1 <= length <= HISTORY_LENGTH and original_mask == [False] * (HISTORY_LENGTH - length) + [True] * length,
         "mask is not a left-padded history")
    need(all(isinstance(row, list) and len(row) == FEATURE_COUNT and
             all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in original_window),
         "original window contains invalid or nonfinite features")
    need(all(all(x == 0 for x in row) for row, valid in zip(original_window, original_mask) if not valid),
         "left-padded feature rows must be zero")
    result = [list(frame) for frame in original_window]
    for frame in result:
        frame[MOVING_INDEX] = 0.0
    return result


def extract_episode(folder, case, expected_config):
    """Rebuild one selected episode from its own verified returned journal."""
    folder = Path(folder)
    manifest, events = collection.read_saved_session(folder, case, expected_config)
    need(manifest.get("finished") is True and manifest.get("blocked") is False and manifest.get("pending") is None,
         "selected episode is not a completed session")
    history, rows, originals = [], [], []
    pending = None
    for event in events[1:]:
        kind = event.get("event")
        if kind == "teacher_proposal":
            need(pending is None and event.get("decision") == len(rows), "proposal decision order differs")
            input_name = f"decision_{len(rows):02d}_input.json"
            need(event.get("input_file") == input_name, "proposal input path differs")
            input_path = folder / "decisions" / input_name
            item, input_bytes = _read_json(input_path)
            need(_sha_bytes(input_bytes) == event.get("input_sha256") and
                 item.get("trial_id") == case["trial_id"] and item.get("decision") == len(rows) and
                 item.get("requested_kind") == event.get("requested_kind") and
                 item.get("requested_joints") == event.get("requested_joints"), "proposal input differs")
            history.append(scene_features.encode_scene_features(item["policy_observation"],
                                                               case["registered_scene"], case["task"]))
            original, mask = scene_features.history_window(history)
            need(item.get("history_window") == original and item.get("history_mask") == mask and
                 item.get("history_length") == sum(mask), "saved history window or mask differs")
            transformed = zero_moving(original, mask)
            pending = {"proposal": event, "item": item, "input_path": str(input_path.resolve()),
                       "original": original, "mask": mask, "transformed": transformed, "started": None}
        elif kind == "teacher_call_started":
            need(pending is not None and pending["started"] is None and
                 event.get("proposal_sha256") == pending["proposal"].get("event_sha256"),
                 "returned decision has no matching call")
            pending["started"] = event
        elif kind == "teacher_returned":
            need(pending is not None and pending["started"] is not None and event.get("operator_returned") is True and
                 event.get("proposal_sha256") == pending["proposal"].get("event_sha256"),
                 "decision return is missing or mismatched")
            action = pending["proposal"]["requested_kind"]
            need(action in motion_policy.ACTIONS and event.get("outcome") == ("done" if action == "done" else "applied"),
                 "returned action differs")
            started = pending["started"]
            need(started.get("operation") == ("assessment" if action == "done" else "command"),
                 "returned operation differs")
            if action == "move":
                joints = started.get("command", {}).get("joints_deg")
                need(isinstance(joints, list) and len(joints) == 4 and
                     all(type(x) in (int, float) and math.isfinite(x) for x in joints) and
                     joints == pending["proposal"].get("requested_joints"), "applied MOVE target differs")
                target = [float(x) / 180.0 for x in joints]
                need(event.get("move_target_valid") is True and event.get("move_target_normalized") == target,
                     "returned MOVE supervision differs")
            else:
                target = None
                need(event.get("move_target_valid") is False and event.get("move_target_normalized") is None,
                     "non-MOVE has a joint target")
            return_path = folder / "decisions" / event["return_file"]
            need(_sha_file(return_path) == event.get("return_file_sha256"), "actual return fingerprint differs")
            original = pending["original"]
            transformed = pending["transformed"]
            mask = pending["mask"]
            raw_digest = _digest_value({"window": original, "mask": mask})
            transformed_digest = _digest_value({"window": transformed, "mask": mask})
            provenance = {"input_path": pending["input_path"], "input_sha256": pending["proposal"]["input_sha256"],
                "proposal_event_sha256": pending["proposal"]["event_sha256"],
                "call_event_sha256": started["event_sha256"], "return_event_sha256": event["event_sha256"],
                "return_path": str(return_path.resolve()), "return_sha256": event["return_file_sha256"],
                "state_before_sha256": pending["proposal"]["state_before_sha256"],
                "state_after_sha256": event["state_after_sha256"], "original_input_sha256": raw_digest,
                "training_input_sha256": transformed_digest}
            rows.append({"schema": ROW_SCHEMA, "trial_id": case["trial_id"], "case_id": case["case_id"],
                "attempt_number": case["attempt_number"], "split_group": case["split_group"],
                "registered_scene": case["registered_scene"], "task": case["task"],
                "physics_seed": case["physics_seed"], "teacher_agent": case["assigned_selector"],
                "teacher_decision": len(rows), "history_length": sum(mask), "window": transformed, "mask": mask,
                "action": action, "action_index": motion_policy.ACTIONS.index(action),
                "move_target_valid": action == "move", "move_target_normalized": target, "provenance": provenance})
            originals.append({"schema": RAW_SCHEMA, "trial_id": case["trial_id"], "teacher_decision": len(rows) - 1,
                "window": original, "mask": mask, "input_path": pending["input_path"],
                "input_sha256": pending["proposal"]["input_sha256"], "original_input_sha256": raw_digest,
                "training_input_sha256": transformed_digest})
            pending = None
        else:
            raise ValueError("selected episode contains a rejected, unknown or unexpected event")
    need(pending is None and len(rows) == manifest["decision_count"] and rows and
         rows[-1]["action"] == "done" and sum(row["action"] == "done" for row in rows) == 1,
         "selected episode requires exactly one final returned DONE")
    return rows, originals


def _check_sources(sources):
    changed = review._changed_sources(sources)
    need(not changed, "review source evidence is missing or changed: " + ", ".join(changed[:3]))


def _within(path, parent):
    return path == parent or parent in path.parents


def export_dataset(review_directory, expected_config, output, *, paths=None):
    """Publish 100/10 labels only after all existing evidence is revalidated."""
    paths = paths or collection.CollectionPaths()
    review_directory = Path(review_directory).resolve()
    output = Path(output).resolve()
    for protected in (paths.prep.resolve(), paths.out.resolve(), paths.initialization_execution.resolve(), review_directory):
        need(not _within(output, protected), "output overlaps protected source evidence")
    if output.exists():
        raise FileExistsError("output already exists; inspect it instead of retrying")
    need(output.parent.is_dir(), "output parent directory must already exist")
    report_path, selection_path = review_directory / "report.json", review_directory / "selection.json"
    # Missing selection fails here, before any planner or case is consulted.
    report, report_raw = _read_json(report_path)
    selection_file, selection_raw = _read_json(selection_path)
    report_sha, selection_sha = _sha_bytes(report_raw), _sha_bytes(selection_raw)
    module_sha = _sha_file(__file__)
    _, cases, _ = collection.locked(expected_config, paths)
    collection.verify_reference_hashes(paths)
    binding = validate_review_selection(cases, report, selection_file, expected_config, report_sha)
    sources = binding["source_hashes"]
    _check_sources(sources)
    for file in (paths.prep / "config.json", paths.prep / "plan.json"):
        need(sources.get(str(file.resolve())) == _sha_file(file), "fixed plan or configuration is not review-bound")

    init, init_sources = review._initialization(expected_config, paths, cases)
    need(init.get("status") == "completed" and set(init["cases"]) == {x["trial_id"] for x in cases},
         "128-case initialization evidence is incomplete")
    need(all(sources.get(name) == digest for name, digest in init_sources.items()),
         "initialization evidence is not review-bound")
    by_id = {case["trial_id"]: case for case in cases}
    for case in cases:
        tid = case["trial_id"]
        folder = paths.out / tid
        sep = folder / "initial_separation.json"
        initial = init["cases"][tid]
        need(_sha_file(sep) == initial.get("separation_file_sha256") and
             sources.get(str(sep.resolve())) == _sha_file(sep), "initial separation evidence differs")
        separation, _ = _read_json(sep)
        collection.require_clear_separation(separation, initial["binding"]["bound_state_sha256"], PLANNED_CASES)
        saved = binding["outcomes"][tid]
        entry = {kind: {"path": saved[f"{kind}_review_evidence"]["path"],
                        "sha256": saved[f"{kind}_review_evidence"]["sha256"]} for kind in ("source", "quality")}
        actual, detail, evidence = review.inspect_episode(case, folder, expected_config, entry)
        need(actual == saved and detail == binding["details"][tid], f"{tid}: reviewed episode changed")
        need(all(sources.get(name) == digest for name, digest in evidence.items()),
             f"{tid}: episode or review file is not bound to published review")

    rows_by_trial, originals = {}, []
    for tid in binding["selected_100"]:
        rows, raw_rows = extract_episode(paths.out / tid, by_id[tid], expected_config)
        rows_by_trial[tid] = rows
        originals.extend(raw_rows)
    rows100 = [row for tid in binding["selected_100"] for row in rows_by_trial[tid]]
    rows10 = [row for tid in binding["selected_10"] for row in rows_by_trial[tid]]
    raw_jsonl = b"".join(_json_bytes(row) for row in originals)
    train100_jsonl = b"".join(_json_bytes(row) for row in rows100)
    train10_jsonl = b"".join(_json_bytes(row) for row in rows10)
    files = {"original_inputs.jsonl": raw_jsonl, "train_100.jsonl": train100_jsonl,
             "train_10.jsonl": train10_jsonl}
    counts = {str(size): {"episodes": size, "decisions": len(rows),
               "actions": dict(sorted(Counter(r["action"] for r in rows).items())),
               "move_targets": sum(r["move_target_valid"] for r in rows)}
              for size, rows in ((100, rows100), (10, rows10))}
    manifest = {"schema": DATASET_SCHEMA, "status": "complete", "collection_config_sha256": expected_config,
        "review_report": {"path": str(report_path.resolve()), "sha256": report_sha},
        "selection": {"path": str(selection_path.resolve()), "sha256": selection_sha,
                      "selected_100": binding["selected_100"], "selected_10": binding["selected_10"]},
        "source_plan": {"path": str((paths.prep / "plan.json").resolve()),
                        "sha256": _sha_file(paths.prep / "plan.json")},
        "exporter": {"path": str(Path(__file__).resolve()), "sha256": module_sha},
        "feature_schema": scene_features.SCHEMA, "history_length": HISTORY_LENGTH,
        "feature_count": FEATURE_COUNT, "moving_index_zero_based": MOVING_INDEX,
        "transformation": "verify original then copy; set moving column to 0 for every frame; preserve mask and all other features",
        "input_digest": "SHA-256 of canonical UTF-8 JSON with sorted keys and trailing newline",
        "non_move_joint_target": None, "counts": counts,
        "files": {name: {"sha256": _sha_bytes(data), "bytes": len(data)} for name, data in files.items()},
        "scope": "direct teacher returned decisions only; no policy inference, simulation or training",
        "source_reviews": "bound through the reviewed report and unchanged original evidence",
        "automatic_selection": False, "selection_rule_recomputed_for_verification": True}
    # No output directory is created unless all evidence is still unchanged.
    _check_sources(sources)
    need(_sha_file(report_path) == report_sha and _sha_file(selection_path) == selection_sha and
         _sha_file(__file__) == module_sha, "review, selection or exporter changed during conversion")
    collection.locked(expected_config, paths)
    output.mkdir(exist_ok=False)
    # The manifest is written last; partial publication never advertises complete data.
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    _check_sources(sources)
    need(_sha_file(report_path) == report_sha and _sha_file(selection_path) == selection_sha and
         _sha_file(__file__) == module_sha, "review, selection or exporter changed during publication")
    collection.locked(expected_config, paths)
    manifest_bytes = _json_bytes(manifest)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    return {"output": str(output), "manifest_sha256": _sha_bytes(manifest_bytes), "counts": counts}
