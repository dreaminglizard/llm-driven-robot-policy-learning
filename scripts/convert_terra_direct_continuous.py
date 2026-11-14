"""Validate and quarantine a stopped, direct continuous Terra demonstration.

The output is an ordered public record, not a training authorization. Saved
private checkpoints are read only to validate boundary continuity and score
bindings; their contents never enter public inputs or labels. Nothing here runs
a policy, a simulator, a teacher, or a physical device.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raccoon_control.kinematics import Geometry
from raccoon_memory import motion_policy, scene_features, task_catalog

SCHEMA = "terra-continuous-quarantined-public-sequence.v1"
REVIEW_SCHEMA = "terra-continuous-action-review-selection.v1"
# Current collection contract forbids synthetic ID permutations. Historical
# 24-view artifacts remain untouched; active conversion retains original IDs.
ORIGINAL_ID_VIEWS = ((1, 2, 3, 4),)
ROBOT_FIELDS = frozenset(("schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
                          "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh"))
VISIBLE_FIELDS = frozenset(("observation_id", "phase", "observation", "previous_applied_target_deg", "previous_action",
    "nominal_fk_observed_xyz_mm", "nominal_fk_previous_target_xyz_mm", "registered_slot_xy_mm", "expected_scene",
    "physical_confirmation", "request_kind", "task", "window", "mask", "conditional_pose_estimate", "build_goal",
    "registered_initial_scene", "recalled_goal"))
CONTROLS = frozenset(("START_MOVE", "REMEMBER", "RECALL", "FINISH"))
ROBOT_ACTIONS = frozenset(("MOVE", "OPEN", "CLOSE", "DONE"))
VISION_FIELDS = frozenset(("teacher_visual_privilege", "vision_collector_source_sha256",
    "pilot_source_sha256", "student_images_or_image_features", "posthoc_scoring_timing_limit"))
VISION_PRIVILEGE = ("Teacher receives rendered RGB views of the live simulated scene; "
    "the existing student and GRU do not receive images or image-derived features. "
    "This is privileged visual information relative to the existing student.")


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            need(key not in out, "duplicate JSON field")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def load_pinned(path, expected):
    need(isinstance(expected, str) and len(expected) == 64
         and all(c in "0123456789abcdef" for c in expected), "expected a lowercase SHA-256")
    raw = Path(path).read_bytes()
    need(digest(raw) == expected, f"fingerprint mismatch: {Path(path).name}")
    return strict_json(raw)


def finite_vector(value, size, name):
    need(isinstance(value, (list, tuple)) and len(value) == size
         and all(type(x) in (int, float) and math.isfinite(x) for x in value), f"invalid {name}")
    return list(value)


def close(left, right, tolerance=1e-4):
    return len(left) == len(right) and all(abs(a - b) <= tolerance for a, b in zip(left, right, strict=True))


def scene(value):
    return [list(stack) for stack in task_catalog.scene(value)]


def public_observation(value):
    need(isinstance(value, dict) and set(value) == ROBOT_FIELDS, "unexpected or missing robot observation field")
    finite_vector(value["joints_deg"], 4, "observed joints")
    need(value["sensor_valid"] is True and value["gripper_reported_valid"] is True
         and value["gripper_feedback_fresh"] is True and type(value["moving"]) is bool,
         "public robot observation is not valid")
    return copy.deepcopy(value)


def public_visible(value):
    """Allow only fields the pilot explicitly exposes; no physical snapshot is admitted."""
    need(isinstance(value, dict) and set(value) <= VISIBLE_FIELDS, "unexpected visible field")
    required = {"observation_id", "phase", "observation", "previous_applied_target_deg", "previous_action",
                "nominal_fk_observed_xyz_mm", "nominal_fk_previous_target_xyz_mm", "registered_slot_xy_mm",
                "expected_scene", "physical_confirmation", "request_kind", "conditional_pose_estimate"}
    need(required <= value.keys() and value["physical_confirmation"] == "not_inspected", "incomplete public input")
    if value["request_kind"] == "robot_command":
        need(set(value) == required | {"task", "window", "mask"}, "unexpected robot-decision input")
    else:
        goal_field = {"make": "build_goal", "return": "registered_initial_scene", "restore": "recalled_goal"}
        need(value["request_kind"] == "program_control" and value["phase"] in goal_field
             and set(value) == required | {goal_field[value["phase"]]}, "unexpected control-decision input")
    public_observation(value["observation"])
    finite_vector(value["previous_applied_target_deg"], 4, "previous applied target")
    finite_vector(value["nominal_fk_observed_xyz_mm"], 3, "nominal observed FK")
    finite_vector(value["nominal_fk_previous_target_xyz_mm"], 3, "nominal target FK")
    need(close(value["nominal_fk_observed_xyz_mm"], Geometry().forward(value["observation"]["joints_deg"]))
         and close(value["nominal_fk_previous_target_xyz_mm"], Geometry().forward(value["previous_applied_target_deg"])),
         "public nominal FK differs")
    need(len(value["registered_slot_xy_mm"]) == 4, "invalid registered slots")
    for xy in value["registered_slot_xy_mm"]:
        finite_vector(xy, 2, "registered XY")
    scene(value["expected_scene"])
    for key in ("build_goal", "registered_initial_scene", "recalled_goal"):
        if key in value:
            scene(value[key])
    estimate = value["conditional_pose_estimate"]
    need(isinstance(estimate, dict) and set(estimate) <= {"available", "confirmed", "reason", "source", "assumptions",
            "objects", "transfer_source_anchor_mm"} and estimate.get("confirmed") is False
         and estimate.get("source") == "student_public_nominal_belief", "unverified conditional pose source")
    if estimate.get("available") is True:
        expected_ids = {str(x) for stack in value["expected_scene"] for x in stack}
        objects = estimate.get("objects")
        need(isinstance(objects, dict) and set(objects) == expected_ids, "conditional pose identities differ")
        for obj in objects.values():
            need(isinstance(obj, dict) and set(obj) == {"position_mm", "state", "confirmed", "assumption"}
                 and obj["confirmed"] is False, "invalid conditional pose")
            finite_vector(obj["position_mm"], 3, "conditional nominal pose")
        if estimate.get("transfer_source_anchor_mm") is not None:
            finite_vector(estimate["transfer_source_anchor_mm"], 2, "conditional source anchor")
    else:
        need(estimate.get("available") is False and "objects" not in estimate, "invalid unavailable pose")
    return copy.deepcopy(value)


def parse_submission(raw):
    return strict_json(raw)


def vision_metadata(assignment, result, directory):
    """Require matching complete metadata whenever a vision source is marked."""
    present = any(key in item for item in (assignment, result) for key in VISION_FIELDS)
    if not present:
        if directory is not None:
            need(not any((Path(directory) / name).exists() for name in
                         ("teacher_frame_index.jsonl", "teacher_view_acknowledgements.jsonl")),
                 "vision index exists without vision source metadata")
        return None
    need(directory is not None, "vision source requires frame and acknowledgement indices")
    need(all(key in assignment and key in result and assignment[key] == result[key]
             for key in VISION_FIELDS), "vision source metadata missing or differs")
    source = assignment["vision_collector_source_sha256"]
    need(isinstance(source, str) and len(source) == 64 and all(c in "0123456789abcdef" for c in source)
         and source == digest((ROOT / "scripts/collect_terra_vision_continuous.py").read_bytes())
         and assignment["pilot_source_sha256"] == assignment.get("collector_source_sha256")
         and assignment["teacher_visual_privilege"] == VISION_PRIVILEGE
         and assignment["student_images_or_image_features"] is False
         and isinstance(assignment["posthoc_scoring_timing_limit"], str)
         and bool(assignment["posthoc_scoring_timing_limit"]), "vision collector binding differs")
    return {key: copy.deepcopy(assignment[key]) for key in sorted(VISION_FIELDS)}


def vision_evidence(directory, metadata, events):
    """Cross-check source indices and RGB bytes, without asserting that anyone viewed them."""
    from scripts import collect_terra_direct_continuous_pilot as collector

    directory = Path(directory)
    names = ("teacher_frame_index.jsonl", "teacher_view_acknowledgements.jsonl")
    try:
        raw_index, raw_ack = ((directory / name).read_bytes() for name in names)
    except OSError as exc:
        raise ValueError("vision indices missing or unreadable") from exc
    def records(raw):
        lines = raw.decode("utf-8").splitlines()
        need(lines and all(line.strip() for line in lines), "vision index is empty or contains a blank line")
        return [strict_json(line) for line in lines]
    frames, acknowledgements = records(raw_index), records(raw_ack)
    observations = [event for event in events if event.get("event") == "teacher_observation"]
    need(observations and len(frames) == len(acknowledgements) == len(observations),
         "vision index count differs from journal")
    checked, checked_files = [], set()
    frame_dir = (directory / "teacher_frames").resolve()
    for observed, index, ack in zip(observations, frames, acknowledgements, strict=True):
        visible = {key: value for key, value in observed.items() if key != "event"}
        oid = visible["observation_id"]
        need(isinstance(index, dict) and set(index) == {"event", "observation_id", "simulation_time_seconds",
             "observation_sha256", "frames", "teacher_only", "simulated", "camera_basis", "timing"}
             and index["event"] == "teacher_camera_frame" and index["observation_id"] == oid
             and index["observation_sha256"] == collector.single.record_sha(visible)
             and index["teacher_only"] is True and index["simulated"] is True
             and index["camera_basis"] == "fixed overview and public registered slot coordinates"
             and index["timing"] == "same synchronous simulator step as robot observation; no command or step between"
             and type(index["simulation_time_seconds"]) in (int, float)
             and math.isfinite(index["simulation_time_seconds"]) and index["simulation_time_seconds"] >= 0,
             "vision frame index differs from public observation")
        task = visible.get("task")
        slots = range(1, 5) if task is None else sorted({task["source_slot"], task["destination_slot"]})
        need(all(type(slot) is int and 1 <= slot <= 4 for slot in slots), "vision frame slots differ")
        expected_views = ["overview", "top"] + [f"slot-{slot}" for slot in slots]
        entries = index["frames"]
        need(isinstance(entries, list) and all(isinstance(item, dict) and set(item) == {"view", "path", "sha256"}
             for item in entries) and [item["view"] for item in entries] == expected_views,
             "vision frame views missing or differ")
        for item in entries:
            sha, path = item["sha256"], item["path"]
            need(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
                 and isinstance(path, str), "invalid vision frame reference")
            path = Path(path)
            need(path.is_absolute() and path.name == sha + ".png" and path.parent.resolve() == frame_dir
                 and path.resolve().parent == frame_dir, "vision frame path differs")
            if sha not in checked_files:
                try:
                    image = path.read_bytes()
                except OSError as exc:
                    raise ValueError("vision frame missing or unreadable") from exc
                need(digest(image) == sha and len(image) >= 33 and image[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                     and image[16:26] == b"\x00\x00\x03\xc0\x00\x00\x02\xd0\x08\x02",
                     "vision RGB frame fingerprint or format differs")
                checked_files.add(sha)
        need(isinstance(ack, dict) and set(ack) == {"observation_id", "viewed_frame_sha256", "inspection_self_reported"}
             and ack["observation_id"] == oid and ack["inspection_self_reported"] is True
             and ack["viewed_frame_sha256"] in {item["sha256"] for item in entries},
             "vision acknowledgement differs from frame index")
        checked.append({"observation_id": oid, "observation_sha256": index["observation_sha256"],
                        "simulation_time_seconds": index["simulation_time_seconds"],
                        "teacher_only_frames": copy.deepcopy(entries),
                        "self_reported_viewed_frame_sha256": ack["viewed_frame_sha256"]})
    return {**metadata, "frame_index": {"path": str((directory / names[0]).resolve()), "sha256": digest(raw_index)},
            "acknowledgement_index": {"path": str((directory / names[1]).resolve()), "sha256": digest(raw_ack)},
            "index_fingerprints_bound_to_source_result": False,
            "collector_fingerprint": "matches_local_source_not_independently_attested_at_execution",
            "frame_inspection": "self_reported_not_independently_verified", "observations": checked}


def convert_public(assignment, result, trace, events, *, vision_directory=None):
    """Cross-check the complete journal and produce only public, ordered events."""
    from scripts import collect_terra_direct_continuous_pilot as collector

    visual = vision_metadata(assignment, result, vision_directory)
    need(assignment.get("direct_operator_identity_independently_verified") is False
         and result.get("direct_operator_identity_independently_verified") is False,
         "this converter has no independently verified operator identity")
    need(result.get("schema") == "terra-direct-continuous-result.v1"
         and trace.get("schema") == "terra-direct-continuous-public-trace.v1"
         and trace.get("completed") is result.get("completed") is True
         and trace.get("termination") == result.get("termination") == "finished_by_teacher"
         and trace.get("error") is None and result.get("error") is None,
         "only stopped, complete sessions are currently convertible")
    for item in (assignment, result, trace):
        need(item.get("student_actions_used") is False and item.get("automatic_robot_actions") is False
             and item.get("actual_hardware") is False, "not a direct virtual record")
    need(result.get("final_test_used") is assignment.get("final_test_used") is False
         and trace.get("physical_success_during_live") is None
         and trace.get("initial_state_sha256") == result.get("initial_state_sha256") == assignment.get("initial_state_sha256")
         and result.get("plan_sha256") == assignment.get("plan_sha256")
         and result.get("collector_source_sha256") == assignment.get("collector_source_sha256"), "source binding differs")
    need(assignment.get("collector_source_sha256") == collector.COLLECTOR_SHA256
         and assignment.get("conditional_pose_estimator_sha256") == digest(
             (ROOT / "raccoon_memory/nominal_object_belief.py").read_bytes())
         and assignment.get("conditional_pose_center_lateral_on_close") is True,
         "public tracker implementation differs from the recorded collector")
    need(assignment.get("teacher_robot_observation_scope") == result.get("teacher_robot_observation_scope")
         == "student_feature_fields_only", "teacher observation scope differs")
    need(events and events[0].get("event") == "session_start" and events[-1].get("event") == "session_stopped"
         and events[-1].get("completed") is True and events[-1].get("termination") == "finished_by_teacher",
         "journal is not a stopped session")
    start = events[0]
    initial = scene(start["initial_scene"])
    build_goal = scene(start["build_goal"])
    need(initial == scene(assignment["initial_scene"]) and build_goal == scene(assignment["build_goal"])
         and start.get("physical_confirmation") == "not_inspected", "start differs")
    slots = start["registered_slot_xy_mm"]
    expected, phase, active, written, recalled = initial, "make", None, None, None
    tracker = collector.PublicPoseTracker(initial, slots)
    stages, timeline, row_index, stage_index, finished = [], [], 0, 0, False
    previous, previous_action, seen, pending_return = None, None, set(), None
    i = 1
    while i < len(events) - 1:
        need(i + 2 < len(events), "incomplete observation/submission/return")
        observed, submitted, outcome = events[i:i+3]
        need(observed.get("event") == "teacher_observation" and submitted.get("event") == "teacher_submission",
             "observation and submission are not adjacent")
        visible = public_visible({k: v for k, v in observed.items() if k != "event"})
        if pending_return is not None:
            # The collector performs no simulated step between a normal command
            # return and the next public read. DONE's automatic hold is separate.
            need(visible["observation"] == pending_return,
                 "command return differs from next public observation")
            pending_return = None
        if active is None:
            need(visible["request_kind"] == "program_control", "expected public control input")
            expected_control = {"make": build_goal, "return": initial, "restore": recalled}[phase]
            field = {"make": "build_goal", "return": "registered_initial_scene", "restore": "recalled_goal"}[phase]
            need(expected_control is not None and scene(visible[field]) == expected_control, "public control goal differs")
            candidate, tracking_error = None, None
        else:
            need(visible["request_kind"] == "robot_command" and visible["task"] == active["task"],
                 "public robot task differs")
            feature = scene_features.encode_scene_features(visible["observation"], expected, active["task"], slots)
            window, mask = scene_features.history_window(active["history"] + [feature])
            need(window == visible["window"] and mask == visible["mask"], "public stage history differs")
            candidate, tracking_error = tracker.preview(visible["window"])
        need(visible["conditional_pose_estimate"] == tracker.visible(candidate, tracking_error),
             "conditional pose is not the public-history-only estimate")
        oid = visible["observation_id"]
        need(isinstance(oid, str) and oid not in seen and submitted.get("observation_id") == oid
             and outcome.get("observation_id") == oid, "observation ID missing, duplicated or mismatched")
        seen.add(oid)
        need(visible["phase"] == phase and scene(visible["expected_scene"]) == expected
             and visible["registered_slot_xy_mm"] == slots and visible["previous_action"] == previous_action,
             "public sequence context differs")
        if previous is None:
            previous = finite_vector(visible["previous_applied_target_deg"], 4, "initial target")
        need(close(visible["previous_applied_target_deg"], previous), "previous applied target continuity differs")
        raw = submitted.get("raw")
        need(isinstance(raw, str), "submission is not raw text")
        if outcome.get("event") == "teacher_rejection":
            need(outcome.get("robot_command_applied") is False and outcome.get("training_label_created") is False,
                 "rejection may have acted or created a label")
            timeline.append({"kind": "rejected_input", "public_input": visible,
                             "submission_sha256": digest(raw.encode("utf-8")), "training_label": None})
            i += 3
            continue
        request = parse_submission(raw)
        need(isinstance(request, dict) and request.get("observation_id") == oid, "submission ID differs")
        action = request.get("action")
        need(outcome.get("action") == action, "returned action differs")
        if active is None:
            need(visible["request_kind"] == "program_control" and action in CONTROLS, "expected control action")
            if action == "START_MOVE":
                need(set(request) == {"observation_id", "action", "source_slot", "destination_slot"}, "invalid START_MOVE")
                _, task = scene_features.transfer_task(expected, request)
                after = scene(task_catalog.transfer(expected, task["source_slot"], task["destination_slot"]))
                need(outcome.get("event") == "teacher_control_return" and outcome.get("task") == task
                     and scene(outcome.get("expected_after")) == after and stage_index < len(trace["stages"]), "move start differs")
                planned = trace["stages"][stage_index]
                need(planned == {"phase": phase, "before": expected, "after": after, "task": task}, "stage differs")
                tracker.begin(expected, task)
                active = {"before": expected, "after": after, "task": task, "history": []}
                stages.append(copy.deepcopy(planned))
            elif action == "REMEMBER":
                need(set(request) == {"observation_id", "action"} and phase == "make" and expected == build_goal
                     and any(s["phase"] == "make" for s in stages)
                     and outcome.get("event") == "memory_written" and outcome.get("scene_source") == "unconfirmed_expected_scene"
                     and outcome.get("memory_source") == "gru_hidden_state", "invalid memory write")
                written, phase = expected, "return"
            elif action == "RECALL":
                need(set(request) == {"observation_id", "action"} and phase == "return" and expected == initial
                     and any(s["phase"] == "return" for s in stages)
                     and outcome.get("event") == "memory_recalled" and outcome.get("memory_source") == "gru_hidden_state",
                     "invalid memory recall")
                recalled, phase = scene(outcome["decoded_goal"]), "restore"
            else:
                need(set(request) == {"observation_id", "action"} and phase == "restore" and recalled is not None
                     and any(s["phase"] == "restore" for s in stages) and i + 3 == len(events) - 1
                     and expected == recalled and outcome.get("event") == "teacher_control_return"
                     and scene(outcome.get("expected_scene")) == expected, "invalid finish")
                finished = True
            timeline.append({"kind": "teacher_control", "public_input": visible,
                             "direct_action": copy.deepcopy(request), "training_label": None})
            if action == "RECALL":
                timeline.append({"kind": "memory_read_result", "decoded_goal": recalled, "training_label": None})
            previous_action = action
            i += 3
            continue
        need(visible["request_kind"] == "robot_command" and action in ROBOT_ACTIONS
             and outcome.get("event") == "teacher_return" and row_index < len(trace["rows"]), "invalid robot decision")
        row = trace["rows"][row_index]
        need(row["observation_id"] == oid and row["phase"] == phase and row["action"] == action
             and row["submission"] == request and row["visible"] == visible and visible["task"] == active["task"]
             and close(row["previous_applied_target_deg"], previous), "trace and journal decision differ")
        active["history"].append(feature)
        target = None
        guard_input = None
        guard = row["guard"]
        if action == "MOVE":
            need(set(request) == {"observation_id", "action", "target_xyz_mm"}, "MOVE is not direct XYZ")
            xyz = finite_vector(request["target_xyz_mm"], 3, "teacher XYZ")
            ik = finite_vector(row["requested_robot_joints_deg"], 4, "IK joint target")
            guard_input = list(Geometry().inverse(xyz))
            need(close(ik, guard_input), "saved IK differs from teacher XYZ")
            target = finite_vector(row["applied_target_deg"], 4, "applied MOVE target")
            need(isinstance(guard, dict) and guard.get("model_action") == "move"
                 and guard.get("max_allowed_overshoot_deg") == 1.0
                 and type(guard.get("clamped")) is bool
                 and close(finite_vector(outcome["requested_robot_joints_deg"], 4, "returned IK"), ik)
                 and close(finite_vector(guard["original_proposal_joints_deg"], 4, "guard input"), ik)
                 and close(finite_vector(guard["previous_applied_target_deg"], 4, "guard context"), previous)
                 and close(finite_vector(guard["guard_returned_target_deg"], 4, "guarded target"), target)
                 and close(outcome["applied_target_deg"], target), "guard or applied target differs")
            derived = {"requested_xyz_mm": xyz, "ik_joint_target_deg": ik,
                       "guard_clamped": guard["clamped"], "applied_joint_target_deg": target,
                       "provenance": "teacher XYZ; nominal IK and one-degree guard are automatic"}
        else:
            need(set(request) == {"observation_id", "action"} and row.get("requested_robot_joints_deg") is None,
                 "invalid non-MOVE submission")
            derived = None
        guard_target, recalculated_guard = collector.public.guarded_teacher(
            action, guard_input, window, mask, previous)
        need(guard == recalculated_guard, "saved guard differs from public-input recomputation")
        if action == "MOVE":
            need(close(guard_target, target, tolerance=1e-7), "applied MOVE target differs from recomputed guard")
        direct_action = {"action": action, "requested_xyz_mm": request.get("target_xyz_mm")}
        tracker.commit(candidate, tracking_error)
        timeline.append({"kind": "teacher_robot_decision", "public_input": visible,
                         "direct_action": direct_action, "derived_move": derived,
                         "action_label": action, "move_applied_target_deg": target})
        if action == "DONE":
            need(row.get("result") == outcome.get("result") == "done_no_command"
                 and row.get("applied_target_deg") is None and row.get("returned_observation") is None
                 and scene(outcome["expected_after"]) == active["after"], "DONE fabricated a command")
            timeline.append({"kind": "automatic_post_done_hold", "duration_seconds": 1.0,
                             "training_label": None, "physical_state_included": False})
            tracker.finish()
            expected, active = active["after"], None
            stage_index += 1
        else:
            need(row.get("result") == outcome.get("result") == "command_applied", "command was not applied")
            applied = finite_vector(row["applied_target_deg"], 4, "applied target")
            if action != "MOVE":
                need(close(applied, previous) and close(outcome["applied_target_deg"], previous),
                     "gripper action changed the held joint target")
            returned = public_observation(row["returned_observation"])
            need(returned == outcome["returned_observation"], "public return differs")
            timeline.append({"kind": "command_return", "action": action, "applied_joint_target_deg": applied,
                             "returned_observation": returned, "training_label": None})
            pending_return = returned
            previous = applied
        previous_action = action
        row_index += 1
        i += 3
    need(finished and active is None and row_index == len(trace["rows"]) and stage_index == len(trace["stages"])
         and written is not None and recalled is not None and expected == recalled
         and scene(trace["expected_final_scene"]) == expected
         and scene(result["written_expected_scene"]) == written and scene(result["decoded_memory_goal"]) == recalled
         and result.get("recall_matches_written") is (written == recalled),
         "incomplete session or memory binding differs")
    converted = {"schema": SCHEMA, "lineage": {"case_id": f"case-{assignment['case_index']:04d}-goal-{assignment['goal_index']:03d}",
            "original_case_id": assignment["original_case_id"], "initial_state_sha256": assignment["initial_state_sha256"],
            "plan_sha256": assignment["plan_sha256"]}, "stages": stages, "timeline": timeline,
            "provenance": {"direct_commands_recorded": True, "direct_operator_identity": "not_independently_verified",
                "public_display_exposure": "not_independently_verified", "independent_teacher_review": "required",
                "automatic_ik_and_guard": "not_teacher_decisions", "post_done_hold": "automatic_not_a_decision",
                "model_recall": "runtime_output_not_a_teacher_action", "private_snapshots_included": False,
                "guard_recomputation": "current_local_implementation_not_independently_pinned_at_collection",
                "public_robot_joints_vs_checkpoint": "not_independently_decoded"},
            "training_eligibility": "unconfirmed", "split": None, "identity_view": None}
    if visual is not None:
        converted["provenance"]["teacher_visual_evidence"] = vision_evidence(vision_directory, visual, events)
    return converted


def convert_case(directory, result_sha256, assignment_sha256, *, provenance_session_id=None):
    from scripts import collect_terra_direct_continuous_pilot as collector

    directory = Path(directory)
    result = load_pinned(directory / "result.json", result_sha256)
    assignment = load_pinned(directory / "assignment.json", assignment_sha256)
    trace = load_pinned(directory / "public_trace.json", result["live_sha256"])
    checkpoints = load_pinned(directory / "private_checkpoints.json", result["checkpoints_sha256"])
    journal = (directory / "events.jsonl").read_bytes()
    need(digest(journal) == trace["event_journal_sha256"], "event journal fingerprint differs")
    events = [strict_json(line) for line in journal.decode("utf-8").splitlines()]
    converted = convert_public(assignment, result, trace, events, vision_directory=directory)
    need(len(checkpoints) == len(converted["stages"]) == len(result["stages"]), "checkpoint count differs")
    first_reset = checkpoints[0]["before_state"]["reset_and_randomization"]
    case = {"initial_scene": assignment["initial_scene"], "physics_seed": first_reset["seed"],
            "registered_slot_xy_mm": events[0]["registered_slot_xy_mm"]}
    previous = None
    physical = []
    for i, (stage, ck, score) in enumerate(zip(converted["stages"], checkpoints, result["stages"], strict=True), 1):
        collector.validate_stage_checkpoint(case, stage, ck, result["initial_state_sha256"], previous)
        need(score["phase"] == stage["phase"] and score["task"] == stage["task"]
             and scene(score["before"]) == stage["before"] and scene(score["expected_after"]) == stage["after"],
             "physical score stage differs")
        status = score.get("status")
        need(status in ("success", "failure", "unconfirmed"), "invalid physical status")
        if status != "unconfirmed":
            raw = load_pinned(directory / f"physical-stage-{i:03d}" / "assessment.json", score["assessment_sha256"])
            spec = collector.stage_spec(case, stage)
            summary = collector.single.physical_summary(spec, raw, status)
            bound = collector.single.bound_assessment(spec, result["live_sha256"], collector.single.record_sha(ck),
                                                      score["assessment_sha256"], summary)
            need(all(score.get(k) == v for k, v in bound.items()), "physical receipt binding differs")
        else:
            need(score.get("samples") is None, "unconfirmed stage claims a scored sample count")
        physical.append({"stage": i, "status": status, "samples": score.get("samples"),
                         "receipt_sha256": score.get("assessment_sha256"), "source": "stored_posthoc_receipt"})
        previous = ck["after_hold_state"]
    converted["source_fingerprints"] = {"assignment_sha256": assignment_sha256, "result_sha256": result_sha256,
        "public_trace_sha256": result["live_sha256"], "event_journal_sha256": trace["event_journal_sha256"],
        "private_checkpoints_sha256": result["checkpoints_sha256"]}
    converted["posthoc_physical_receipts"] = physical
    converted["provenance"]["physical_hold_matching"] = "existing_receipt_claim_only_no_new_physical_replay"
    converted["provenance"]["checkpoint_continuity"] = "saved_state_vectors_and_reset_metadata_only"
    if provenance_session_id is not None:
        try:
            normalized = str(uuid.UUID(provenance_session_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("session reference must be a UUID") from exc
        converted["provenance"]["caller_supplied_session_reference"] = {
            "session_id": normalized, "verified_by_converter": False}
    return converted


def remap_scene(value, permutation):
    return [[permutation[obj - 1] for obj in stack] for stack in value]


def remap_task(task, permutation):
    result = copy.deepcopy(task)
    result["object_id"] = permutation[result["object_id"] - 1]
    return result


def remap_visible(value, permutation):
    if tuple(permutation) == ORIGINAL_ID_VIEWS[0]:
        return copy.deepcopy(value)
    # Legacy nonidentity helper is not used by active conversion/adapter.
    from scripts import train_terra_bc1020_from_scratch as originals

    result = copy.deepcopy(value)
    for key in ("expected_scene", "build_goal", "registered_initial_scene", "recalled_goal"):
        if key in result:
            result[key] = remap_scene(result[key], permutation)
    if "task" in result:
        result["task"] = remap_task(result["task"], permutation)
    if "window" in result:
        columns = originals.permutation_columns(permutation)
        result["window"] = [[row[col] for col in columns] for row in result["window"]]
    estimate = result["conditional_pose_estimate"]
    if "objects" in estimate:
        estimate["objects"] = {str(permutation[int(k) - 1]): v for k, v in estimate["objects"].items()}
    return result


def validate_original_splits(records, assignments):
    """Check only the supplied original records; do not grant eligibility."""
    need(records and isinstance(assignments, dict), "records and explicit original assignments required")
    ids = [r["lineage"]["case_id"] for r in records]
    need(len(ids) == len(set(ids)) and set(ids) == set(assignments), "assignments must cover each original exactly once")
    need(all(r.get("schema") == SCHEMA and r.get("split") is None and r.get("identity_view") is None
             and r.get("training_eligibility") == "unconfirmed" for r in records), "already split or invalid record")
    groups = {}
    for record in records:
        line = record["lineage"]
        split = assignments[line["case_id"]]
        need(split in ("train", "development", "validation", "holdout"), "invalid split")
        for key in (("parent", line["original_case_id"]), ("initial_state", line["initial_state_sha256"])):
            need(key not in groups or groups[key] == split, "related originals cross partitions")
            groups[key] = split


def split_and_identity_views(records, assignments):
    """Assign original splits and return one original-ID view per full sequence.

    The legacy function name and record fields are retained, but this active
    entrypoint no longer generates 24 permutations. Historical exports are
    not rewritten. This checks the supplied records only. A complete-lineage audit is
    still required; assigning a split does not grant training eligibility.
    """
    validate_original_splits(records, assignments)
    result = []
    for record in records:
        for permutation in ORIGINAL_ID_VIEWS:
            view = copy.deepcopy(record)
            view["split"] = assignments[record["lineage"]["case_id"]]
            view["identity_view"] = list(permutation)
            for stage in view["stages"]:
                stage["before"] = remap_scene(stage["before"], permutation)
                stage["after"] = remap_scene(stage["after"], permutation)
                stage["task"] = remap_task(stage["task"], permutation)
            for event in view["timeline"]:
                if "public_input" in event:
                    event["public_input"] = remap_visible(event["public_input"], permutation)
                if event["kind"] == "memory_read_result":
                    event["decoded_goal"] = remap_scene(event["decoded_goal"], permutation)
            result.append(view)
    return result


def review_entry(record):
    """Produce fingerprints to present to a reviewer, without making a decision."""
    need(record.get("schema") == SCHEMA and record.get("split") is None
         and record.get("identity_view") is None and record.get("training_eligibility") == "unconfirmed",
         "review requires an unsplit, quarantined original")
    source = record["source_fingerprints"]
    entry = {"case_id": record["lineage"]["case_id"],
             "parent_original_case_id": record["lineage"]["original_case_id"],
             "source_assignment_sha256": source["assignment_sha256"],
             "source_result_sha256": source["result_sha256"],
             "source_public_trace_sha256": source["public_trace_sha256"],
             "converted_sequence_sha256": digest(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                                        separators=(",", ":")).encode("utf-8"))}
    for name, value in entry.items():
        if name.endswith("_sha256"):
            need(isinstance(value, str) and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value), "invalid review source fingerprint")
    return entry


def check_review_selection(records, review_bytes, expected_sha256):
    """Require a separately supplied, digest-pinned selection conclusion.

    The review JSON contains ``schema: REVIEW_SCHEMA``, ``conclusion:
    independent_review_selected_direct_action_training``, and
    ``selected_originals`` entries matching ``review_entry(record)``. It must
    be supplied separately; this module does not create a review conclusion.
    The caller is responsible for authenticating the external review and its
    reviewer; a matching digest verifies bytes, not the reviewer's identity.
    """
    need(isinstance(review_bytes, bytes) and isinstance(expected_sha256, str)
         and len(expected_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_sha256)
         and digest(review_bytes) == expected_sha256, "review selection fingerprint differs")
    review = strict_json(review_bytes)
    need(isinstance(review, dict) and set(review) == {"schema", "conclusion", "selected_originals"}
         and review.get("schema") == REVIEW_SCHEMA
         and review.get("conclusion") == "independent_review_selected_direct_action_training"
         and isinstance(review.get("selected_originals"), list), "invalid external review conclusion")
    expected = [review_entry(record) for record in records]
    selected = review["selected_originals"]
    need(len(selected) == len(expected) and all(isinstance(item, dict) for item in selected),
         "review must select exactly the supplied originals")
    selected_by_id = {item.get("case_id"): item for item in selected}
    need(len(selected_by_id) == len(selected)
         and all(selected_by_id.get(item["case_id"]) == item for item in expected),
         "reviewed original ID or source fingerprint differs")


class DirectActionAdapter:
    """Original-ID views of independently selected originals for action CE only.

    Applied MOVE angles are retained as observations, never as raw-student
    regression targets. No dense stand-in target is created for other actions.
    The deployed nominal correction must not be assumed differentiable or
    applied a second time by the training caller. ``count`` counts decision-ID
    combinations (one original-ID view only); ``original_count`` and parent/state counts are separate.
    ``batch(indices)`` lazily gathers views in original, permutation, decision
    order and returns the existing model's 8-by-82 windows and boolean masks.
    """

    def __init__(self, records, assignments, *, review_bytes, review_sha256):
        validate_original_splits(records, assignments)
        need(all(assignments[r["lineage"]["case_id"]] == "train" for r in records),
             "action training adapter requires original-level train assignments")
        check_review_selection(records, review_bytes, review_sha256)
        windows, masks, actions, evidence, case_ids, parents = [], [], [], [], [], []
        self.original_count = len(records)
        self.distinct_parent_original_count = len({r["lineage"]["original_case_id"] for r in records})
        self.distinct_initial_state_count = len({r["lineage"]["initial_state_sha256"] for r in records})
        self._starts, self._sizes, self._view_ends = [], [], []
        views = 0
        for record in records:
            case_id = record["lineage"]["case_id"]
            parent = record["lineage"]["original_case_id"]
            self._starts.append(len(actions))
            for event in record["timeline"]:
                if event["kind"] != "teacher_robot_decision":
                    continue
                action, direct = event["action_label"], event["direct_action"]
                need(action in ROBOT_ACTIONS and direct.get("action") == action, "invalid recorded direct action")
                visible = event["public_input"]
                window, mask = visible["window"], visible["mask"]
                need(isinstance(window, list) and len(window) == 8
                     and all(isinstance(row, list) and len(row) == 82
                             and all(type(v) in (int, float) and math.isfinite(v) for v in row) for row in window)
                     and isinstance(mask, list) and len(mask) == 8 and all(type(v) is bool for v in mask)
                     and 0 < sum(mask) <= 8 and mask == [False] * (8 - sum(mask)) + [True] * sum(mask),
                     "invalid student history or mask")
                if action == "MOVE":
                    applied = finite_vector(event["move_applied_target_deg"], 4, "MOVE applied target")
                    need(direct.get("requested_xyz_mm") is not None and event["derived_move"] is not None,
                         "MOVE lacks direct/automatic provenance")
                else:
                    need(event["move_applied_target_deg"] is None and event["derived_move"] is None
                         and direct.get("requested_xyz_mm") is None, "non-MOVE has a fabricated target")
                    applied = None
                windows.append(window)
                masks.append(mask)
                actions.append(motion_policy.ACTIONS.index(action.lower()))
                evidence.append(applied)
                case_ids.append(case_id)
                parents.append(parent)
            count = len(actions) - self._starts[-1]
            need(count > 0, "reviewed original has no direct decisions")
            self._sizes.append(count)
            views += count * len(ORIGINAL_ID_VIEWS)
            self._view_ends.append(views)
        self.original_decision_count = len(actions)
        self.identity_views_per_original = len(ORIGINAL_ID_VIEWS)
        self.count = views
        self.review_sha256 = review_sha256
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.actions = torch.tensor(actions, dtype=torch.long)
        self._evidence = copy.deepcopy(evidence)
        self._case_ids, self._parents = case_ids, parents
        self._columns = torch.arange(82, dtype=torch.long).unsqueeze(0)

    def batch(self, indices):
        if isinstance(indices, torch.Tensor):
            need(indices.ndim == 1 and indices.dtype in (torch.int32, torch.int64), "invalid batch indices")
            indices = indices.tolist()
        need(isinstance(indices, (list, tuple)) and indices
             and all(type(index) is int and 0 <= index < self.count for index in indices), "invalid batch indices")
        row_ids, permutation_ids = [], []
        for index in indices:
            owner = bisect.bisect_right(self._view_ends, index)
            start = 0 if owner == 0 else self._view_ends[owner - 1]
            permutation, local_row = divmod(index - start, self._sizes[owner])
            row_ids.append(self._starts[owner] + local_row)
            permutation_ids.append(permutation)
        rows = torch.tensor(row_ids, dtype=torch.long)
        permutations = torch.tensor(permutation_ids, dtype=torch.long)
        windows = self.windows[rows].gather(2, self._columns[permutations, None, :].expand(-1, 8, -1))
        return {"windows": windows, "history_masks": self.masks[rows], "action_targets": self.actions[rows],
                "joint_regression_mask": torch.zeros(len(rows), dtype=torch.bool), "joint_regression_targets": None,
                "applied_move_evidence_deg": [copy.deepcopy(self._evidence[i]) for i in row_ids],
                "case_ids": [self._case_ids[i] for i in row_ids],
                "parent_original_case_ids": [self._parents[i] for i in row_ids],
                "identity_permutations": [list(ORIGINAL_ID_VIEWS[i]) for i in permutation_ids],
                "original_count": self.original_count, "original_decision_count": self.original_decision_count,
                "distinct_parent_original_count": self.distinct_parent_original_count,
                "distinct_initial_state_count": self.distinct_initial_state_count,
                "identity_views_per_original": self.identity_views_per_original,
                "review_selection_sha256": self.review_sha256,
                "review_authentication": "caller_supplied_conclusion_reviewer_identity_not_checked",
                "joint_target_contract": "applied_move_is_not_a_raw_552_student_target",
                "development_start_lineage": "pilot_reuses_1000_parent_not_independent_cross_roster_audit_required"}


def direct_action_loss(logits, batch):
    """Cross-entropy only; the caller must supply a trainable action readout.

    The existing 552 wrapper can return detached frozen logits. This helper
    checks that logits require gradients, but the caller must separately check
    they reach the intended parameters. It neither changes the wrapper nor
    assumes gradients through its correction.
    """
    actions, mask = batch["action_targets"], batch["joint_regression_mask"]
    need(isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[1] == len(motion_policy.ACTIONS)
         and isinstance(actions, torch.Tensor) and actions.shape == (len(logits),)
         and actions.dtype == torch.long and isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
         and mask.shape == actions.shape and not bool(mask.any()) and batch["joint_regression_targets"] is None
         and bool(torch.isfinite(logits).all()), "invalid action-only training batch")
    need(logits.requires_grad, "action logits do not connect to a trainable readout")
    return F.cross_entropy(logits, actions.to(logits.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--provenance-session-id", help="optional external reference; not verified here")
    args = parser.parse_args()
    value = convert_case(args.case_directory, args.result_sha256, args.assignment_sha256,
                         provenance_session_id=args.provenance_session_id)
    json.dump(value, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print()


if __name__ == "__main__":
    main()
