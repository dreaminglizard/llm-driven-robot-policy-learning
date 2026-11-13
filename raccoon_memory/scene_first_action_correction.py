"""Pinned correction branches immediately after one applied student command.

This is a separate selection contract from the earlier failure-selected
branches. It reads the previously recorded attempts and never runs a student
or advances simulated physics while preparing the selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_correction_data as old, scene_policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/scene_first_action_corrections.v1.json"
PREPARED = ROOT / "artifacts/architecture_research/20261031_scene_first_action_corrections_prepared_v1"
CONTRACT_SCHEMA = "scene-first-action-correction-selection-contract.v1"
PREPARED_SCHEMA = "scene-first-action-correction-prepared.v1"
BRANCH_SCHEMA = "scene-first-action-student-correction-branch.v1"
ROW_SCHEMA = "scene-first-action-correction-sample.v1"
RULE = "fixed_parent_order_cyclic_seed_after_one_applied_student_command.v1"
SOURCE_FILES = (
    "configs/scene_first_action_corrections.v1.json",
    "raccoon_memory/scene_first_action_correction.py",
    "scripts/prepare_scene_first_action_corrections.py",
    "scripts/operate_scene_first_action_correction.py",
    "tests/test_scene_first_action_correction.py",
    "raccoon_memory/scene_demo_data.py",
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _close_vector(left, right, tolerance=1e-9):
    return (isinstance(left, list) and isinstance(right, list) and len(left) == len(right)
            and all(type(a) in (int, float) and type(b) in (int, float)
                    and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance
                    for a, b in zip(left, right)))


def load_contract():
    contract = read_json(CONTRACT_PATH)
    from scripts import collect_scene_corrections as previous
    upstream = previous.load_prepared()
    if (contract.get("schema") != CONTRACT_SCHEMA or contract.get("selection_rule") != RULE
            or contract.get("decision_index") != 1 or contract.get("applied_student_commands") != 1
            or contract.get("split") != "train" or contract.get("max_direct_attempts_per_group") != 1
            or contract.get("max_direct_actions") != 24 or contract.get("physical_assessment_seconds") != 1.0
            or any(contract.get(key) is not False for key in (
                "success_or_object_state_used_for_selection", "fallback_case_selection", "new_student_execution",
                "new_teacher_action_during_preparation", "actual_hardware", "final_test_used",
                "mix_with_prior_correction_rows"))
            or (ROOT / contract["output"]).resolve() != PREPARED.resolve()
            or (ROOT / contract["original_prepared"]).resolve() != (previous.PREPARED / "prepared.json").resolve()
            or file_hash(ROOT / contract["original_prepared"]) != contract["original_prepared_sha256"]
            or file_hash(ROOT / contract["original_collection_summary"]) != contract["original_collection_summary_sha256"]):
        raise ValueError("fixed first-action selection contract differs")
    cases = contract.get("cases_in_order")
    if not isinstance(cases, list) or len(cases) != 8 or len(upstream["groups_in_order"]) != 8:
        raise ValueError("selection must have eight ordered parent groups")
    for i, case in enumerate(cases):
        seed = (1000, 1001, 1002)[i % 3]
        matches = [c for c in upstream["cases"] if c["parent_group"] == upstream["groups_in_order"][i]
                   and c["physics_seed"] == seed]
        if (len(matches) != 1 or case != {"group_index": i, "parent_group": upstream["groups_in_order"][i],
                                         "physics_seed": seed, "student_run_id": matches[0]["student_run_id"]}):
            raise ValueError("fixed parent order, seed cycle or student run differs")
    summary = read_json(ROOT / contract["original_collection_summary"])
    if (summary.get("planned") != 24 or summary.get("finished") != 24
            or len(summary.get("results", [])) != 24
            or [x.get("student_run_id") for x in summary["results"]]
            != [x["student_run_id"] for x in upstream["cases"]]):
        raise ValueError("original collection is incomplete or reordered")
    return contract, upstream, summary


def validate_student_prefix(first, second, decisions):
    """Validate the recorded two observations and exactly one applied command."""
    old.validate_capture(first)
    old.validate_capture(second)
    if (first["decision_index"] != 0 or first["applied_student_command_count"] != 0
            or second["decision_index"] != 1 or second["applied_student_command_count"] != 1
            or len(first["observations"]) != 1 or len(second["observations"]) != 2
            or second["observations"][0] != first["observations"][0]
            or second["window"][-2:] != [first["window"][-1], second["window"][-1]]
            or second["mask"] != [False] * (len(second["mask"]) - 2) + [True, True]
            or first["registered_scene"] != second["registered_scene"] or first["task"] != second["task"]
            or not isinstance(decisions, list) or len(decisions) < 2):
        raise ValueError("two original observations or applied-command count differ")
    for index, capture in enumerate((first, second)):
        decision = decisions[index]
        observation_keys = ("schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
                            "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")
        expected_observation = {key: capture["observations"][-1].get(key) for key in observation_keys}
        if (decision.get("index") != index or decision.get("policy_features") != capture["window"][-1]
                or decision.get("valid_history") != capture["mask"]
                or decision.get("policy_observation") != expected_observation):
            raise ValueError("student observation or model history differs from its capture")
    earlier, current = decisions[0], decisions[1]
    command = earlier.get("command")
    if (earlier.get("result") != "command_applied" or earlier.get("action") not in ("move", "open", "close")
            or not isinstance(command, dict) or command.get("source") != "shared_student_policy"
            or command.get("seconds") != 1.0
            or not _close_vector(command.get("joints_deg"), second["arm_servo_target_deg"])
            or type(command.get("gripper_opening_m")) not in (int, float)
            or abs(command["gripper_opening_m"] - second["gripper_servo_opening_m"]) > 1e-9
            or earlier.get("simulation_after") != current.get("simulation_before")
            or not isinstance(earlier.get("simulation_after"), dict)):
        raise ValueError("recorded first command or its resulting observation differs")
    start = earlier.get("simulation_before", {}).get("time_seconds")
    end = earlier["simulation_after"].get("time_seconds")
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or abs(end-start-1.0) > 1e-8):
        raise ValueError("first applied command did not advance the recorded one second")
    return True


def _case_evidence(contract, upstream, summary, selected, *, restore):
    """Validate one fixed attempt; later outcome is never used to choose it."""
    from scripts import operate_scene_sim as operator
    matching = [x for x in upstream["cases"] if x["student_run_id"] == selected["student_run_id"]]
    if len(matching) != 1:
        raise ValueError("selected student case is missing")
    case = matching[0]
    index = upstream["cases"].index(case)
    summary_item = summary["results"][index]
    folder = Path(upstream["collection_output"]) / case["student_run_id"]
    attempt_path, capture_path = folder / "result.json", folder / "captures.jsonl"
    if file_hash(attempt_path) != summary_item.get("result_sha256"):
        raise ValueError("student attempt hash differs from the original summary")
    attempt = read_json(attempt_path)
    raw = capture_path.read_bytes()
    restore_info, result = attempt.get("restore"), attempt.get("result")
    if (not isinstance(restore_info, dict) or not isinstance(result, dict)
            or not raw or not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != attempt.get("captures_sha256")
            or attempt.get("case") != case or attempt.get("prepared_sha256") != contract["original_prepared_sha256"]
            or attempt.get("student_only") is not True or attempt.get("actual_hardware") is not False
            or attempt.get("teacher_assistance_count") != 0
            or result.get("teacher_assistance_count") != 0 or result.get("actual_hardware") is not False
            or restore_info.get("entire_state_matches") is not True
            or restore_info.get("parent_post_assessment_state_sha256") != case["parent_state_sha256"]
            or restore_info.get("restored_pre_command_state_sha256") != case["parent_state_sha256"]
            or old.digest(result.get("initial_state")) != case["parent_state_sha256"]
            or case["parent_group"] != selected["parent_group"] or case["physics_seed"] != selected["physics_seed"]):
        raise ValueError("attempt, parent restore or capture file differs")
    lines = raw.splitlines(keepends=True)
    decisions = result.get("decisions", [])
    if len(lines) != len(decisions) or len(lines) < 2 or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("missing, truncated or incomplete student decision captures")
    first = json.loads(lines[0])
    capture = json.loads(lines[1])
    validate_student_prefix(first, capture, decisions)
    initial = result["initial_state"]
    for key in ("state", "state_signature", "generated_model_xml_sha256", "model_parameters_sha256",
                "reset_and_randomization", "committed_layout", "anchor_positions_m"):
        if initial.get(key) != first["simulation_state"].get(key):
            raise ValueError("first capture does not preserve the restored parent physics")
    state = capture["simulation_state"]
    reset = state.get("reset_and_randomization", {})
    if (reset.get("seed") != selected["physics_seed"] or reset.get("randomize") is not True
            or capture["registered_scene"] != case["case"]["registered_scene"]
            or capture["task"] != case["case"]["task"]
            or scene_policy.split_group_for(capture["registered_scene"], capture["task"]) != case["task_group"]):
        raise ValueError("captured task, train group or physical randomization differs")
    plan = read_json(ROOT / "configs/scene_split_plan.v1.json")
    if case["parent_group"] not in plan["train_groups"] or case["task_group"] not in plan["train_groups"]:
        raise ValueError("both parent and successor must be training groups")
    if restore:
        # Creating a separate in-memory model and loading the entire captured
        # state performs no command, mj_step, or object-position editing.
        from raccoon_memory.scene_motion_sim import SceneMotionSim
        sim = SceneMotionSim(reset["initial_layout"], seed=reset["seed"], randomize=reset["randomize"])
        initial_reset = sim.save_state()["reset_and_randomization"]
        sim.load_state(state)
        if sim.save_state()["reset_and_randomization"] != initial_reset:
            raise ValueError("restored physical parameters do not match the original seed")
        operator._verify_branch_first_state(sim, capture)
        if sim.observation() != decisions[0]["simulation_after"]:
            raise ValueError("full restored observation differs from the recorded first-command result")
    source = {"student_run_id": case["student_run_id"], "parent_group": case["parent_group"],
              "task_group": case["task_group"], "physics_seed": case["physics_seed"],
              "model_seed": case["model_seed"], "split": "train", "parent_state_sha256": case["parent_state_sha256"],
              "model_sha256": upstream["models_by_physics_seed"][str(case["physics_seed"])]["sha256"],
              "original_prepared_sha256": contract["original_prepared_sha256"],
              "selection_contract_sha256": file_hash(CONTRACT_PATH),
              "attempt_path": str(attempt_path.resolve()), "attempt_sha256": file_hash(attempt_path),
              "capture_path": str(capture_path.resolve()), "capture_file_sha256": hashlib.sha256(raw).hexdigest(),
              "capture_line_sha256": hashlib.sha256(lines[1]).hexdigest(),
              "source_kind": "recorded_student_pre_inference_after_one_applied_command",
              "simulated": True, "actual_hardware": False}
    envelope = {"schema": BRANCH_SCHEMA, "capture": capture, "student_source": source,
                "selection": {"rule": RULE, "group_index": selected["group_index"],
                              "parent_group": selected["parent_group"], "physics_seed": selected["physics_seed"],
                              "decision_index": 1, "applied_student_commands": 1}}
    envelope["envelope_sha256"] = old.digest(envelope)
    validate_branch_envelope(envelope)
    return envelope


def validate_branch_envelope(envelope):
    if (not isinstance(envelope, dict) or envelope.get("schema") != BRANCH_SCHEMA
            or envelope.get("envelope_sha256") != old.digest({k: v for k, v in envelope.items()
                                                              if k != "envelope_sha256"})):
        raise ValueError("first-action branch schema or hash differs")
    capture = old.validate_capture(envelope.get("capture"))
    source, select = envelope.get("student_source"), envelope.get("selection")
    if (not isinstance(source, dict) or not isinstance(select, dict) or select.get("rule") != RULE
            or type(select.get("group_index")) is not int or not 0 <= select["group_index"] < 8
            or select.get("decision_index") != 1 or select.get("applied_student_commands") != 1
            or capture["decision_index"] != 1 or capture["applied_student_command_count"] != 1
            or select.get("physics_seed") != (1000, 1001, 1002)[select["group_index"] % 3]
            or source.get("physics_seed") != select["physics_seed"]
            or source.get("parent_group") != select.get("parent_group")
            or source.get("task_group") != scene_policy.split_group_for(capture["registered_scene"], capture["task"])
            or source.get("source_kind") != "recorded_student_pre_inference_after_one_applied_command"
            or source.get("split") != "train" or source.get("simulated") is not True
            or source.get("actual_hardware") is not False):
        raise ValueError("first-action branch selection or source differs")
    return envelope


def prepare():
    contract, upstream, summary = load_contract()
    if PREPARED.exists() or PREPARED.is_symlink():
        raise FileExistsError("preparation output must be a new directory")
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    PREPARED.mkdir(parents=True, exist_ok=False)
    selections = PREPARED / "selected_branches"
    selections.mkdir()
    entries = []
    for case in contract["cases_in_order"]:
        entry = dict(case)
        try:
            envelope = _case_evidence(contract, upstream, summary, case, restore=True)
            path = selections / f"group-{case['group_index']:02d}.json"
            write_new(path, envelope)
            entry.update(status="selected", envelope=str(path.resolve()), envelope_file_sha256=file_hash(path),
                         envelope_sha256=envelope["envelope_sha256"],
                         capture_sha256=envelope["capture"]["capture_sha256"],
                         state_sha256=envelope["capture"]["state_sha256"], full_state_roundtrip=True)
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            entry.update(status="excluded", reason=f"{type(exc).__name__}: {exc}")
        entries.append(entry)
    if any(file_hash(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("source code changed during preparation")
    if file_hash(ROOT / contract["original_collection_summary"]) != contract["original_collection_summary_sha256"]:
        raise ValueError("original collection summary changed during preparation")
    spec = {"schema": PREPARED_SCHEMA, "contract_path": str(CONTRACT_PATH.resolve()),
            "contract_sha256": file_hash(CONTRACT_PATH), "source_files_sha256": source_hashes,
            "original_prepared_sha256": contract["original_prepared_sha256"],
            "original_collection_summary_sha256": contract["original_collection_summary_sha256"],
            "selections": entries, "selected": sum(x["status"] == "selected" for x in entries),
            "excluded": sum(x["status"] == "excluded" for x in entries),
            "decision_index": 1, "applied_student_commands": 1, "max_direct_actions": 24,
            "max_direct_attempts_per_group": 1, "physical_assessment_seconds": 1.0,
            "success_or_object_state_used_for_selection": False, "fallback_case_selection": False,
            "new_student_execution": False, "new_teacher_action": False, "physics_steps_executed": 0,
            "actual_hardware": False, "final_test_used": False, "mix_with_prior_correction_rows": False}
    path = PREPARED / "prepared.json"
    write_new(path, spec)
    with (PREPARED / "prepared.sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(file_hash(path) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"prepared": str(path), "sha256": file_hash(path), "selected": spec["selected"],
            "excluded": spec["excluded"], "new_student_execution": False, "physics_steps_executed": 0}


def load_prepared():
    path = PREPARED / "prepared.json"
    if file_hash(path) != (PREPARED / "prepared.sha256").read_text(encoding="ascii").strip():
        raise ValueError("first-action preparation hash differs")
    spec = read_json(path)
    contract, upstream, summary = load_contract()
    if (spec.get("schema") != PREPARED_SCHEMA or spec.get("contract_sha256") != file_hash(CONTRACT_PATH)
            or spec.get("contract_path") != str(CONTRACT_PATH.resolve())
            or len(spec.get("selections", [])) != 8
            or set(spec.get("source_files_sha256", {})) != set(SOURCE_FILES)
            or any(file_hash(ROOT / name) != digest for name, digest in spec["source_files_sha256"].items())):
        raise ValueError("first-action preparation or source code differs")
    entries = spec["selections"]
    if (spec.get("selected") != sum(item.get("status") == "selected" for item in entries)
            or spec.get("excluded") != sum(item.get("status") == "excluded" for item in entries)
            or spec["selected"] + spec["excluded"] != 8
            or spec.get("decision_index") != 1 or spec.get("applied_student_commands") != 1
            or spec.get("max_direct_actions") != 24 or spec.get("max_direct_attempts_per_group") != 1
            or spec.get("physical_assessment_seconds") != 1.0
            or spec.get("physics_steps_executed") != 0
            or any(spec.get(key) is not False for key in (
                "success_or_object_state_used_for_selection", "fallback_case_selection", "new_student_execution",
                "new_teacher_action", "actual_hardware", "final_test_used", "mix_with_prior_correction_rows"))):
        raise ValueError("prepared counts, scope or execution limits differ")
    for expected, item in zip(contract["cases_in_order"], entries):
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("prepared group order or case differs")
        if item["status"] == "selected":
            path = PREPARED / "selected_branches" / f"group-{expected['group_index']:02d}.json"
            if (item.get("envelope") != str(path.resolve()) or item.get("full_state_roundtrip") is not True
                    or file_hash(path) != item.get("envelope_file_sha256")):
                raise ValueError("prepared branch or state verification differs")
        elif not isinstance(item.get("reason"), str) or not item["reason"]:
            raise ValueError("excluded case must have a reason")
    return spec, contract, upstream, summary


def verify_pinned_envelope(envelope):
    validate_branch_envelope(envelope)
    spec, contract, upstream, summary = load_prepared()
    index = envelope["selection"]["group_index"]
    expected_case, item = contract["cases_in_order"][index], spec["selections"][index]
    path = PREPARED / "selected_branches" / f"group-{index:02d}.json"
    if (item.get("status") != "selected" or item.get("group_index") != index
            or item.get("envelope") != str(path.resolve()) or file_hash(path) != item.get("envelope_file_sha256")
            or read_json(path) != envelope or item.get("envelope_sha256") != envelope["envelope_sha256"]):
        raise ValueError("branch is not the pinned first-action selection")
    if _case_evidence(contract, upstream, summary, expected_case, restore=False) != envelope:
        raise ValueError("original student capture or selection lineage changed")
    return envelope


def first_correction_row(envelope, manifest, events):
    """One first direct action plus the original student history; no generated done."""
    validate_branch_envelope(envelope)
    capture = envelope["capture"]
    if (not isinstance(events, list) or not events or events[0].get("event") != "branch"
            or events[0].get("branch_envelope_sha256") != envelope["envelope_sha256"]
            or events[0].get("state_after_sha256") != capture["state_sha256"]):
        raise ValueError("teacher session does not start from the selected student state")
    criterion = events[-1].get("simulation_physical_criterion", {})
    if not manifest.get("finished") or events[-1].get("event") != "finish" or criterion.get("success") is not True:
        return {"eligible": False, "exclusion_reason": "direct_branch_not_confirmed_successful"}
    if (len(events) < 3 or events[1].get("event") != "review"
            or events[1].get("review_source") != "direct_assistant_simulation_observation"
            or events[1].get("scene_review") not in ("consistent", "target_grasped", "target_placed")
            or events[1].get("scene_evidence", {}).get("allowed") is not True
            or events[1].get("state_before_sha256") != capture["state_sha256"]
            or events[1].get("state_after_sha256") != capture["state_sha256"]):
        return {"eligible": False, "exclusion_reason": "initial_scene_not_confirmed_for_direct_correction"}
    first = events[2]
    if first.get("state_before_sha256") != capture["state_sha256"]:
        raise ValueError("first direct decision is not from the captured state")
    if first.get("event") == "finish":
        if first.get("assistant_done_proposal") is not True:
            return {"eligible": False, "exclusion_reason": "no_explicit_first_teacher_action"}
        kind, target = "done", [0.0] * 4
    elif first.get("event") == "action":
        kind, action = first.get("policy_action"), first.get("action")
        if kind not in ("move", "open", "close") or not isinstance(action, dict) or action.get("seconds") != 1.0:
            raise ValueError("first teacher command is not a single explicit policy action")
        angles, finger = action.get("joints_deg"), action.get("gripper_opening_m")
        if (not isinstance(angles, list) or len(angles) != 4
                or any(type(q) not in (int, float) or not math.isfinite(q) or not lo <= q <= hi
                       for q, (lo, hi) in zip(angles, JOINT_LIMITS))
                or type(finger) not in (int, float) or not math.isfinite(finger)):
            raise ValueError("invalid first teacher command")
        if kind == "move":
            if abs(finger - capture["gripper_servo_opening_m"]) > 1e-9:
                raise ValueError("move changed the gripper servo target")
            target = [q / motion_policy.JOINT_SCALE_DEG for q in angles]
        else:
            if (not _close_vector(angles, capture["arm_servo_target_deg"])
                    or abs(finger - (0.04 if kind == "open" else 0.0)) > 1e-9):
                raise ValueError("gripper action did not hold the prior arm servo target")
            target = [0.0] * 4
    else:
        raise ValueError("first teacher event has no direct policy action")
    return {"schema": ROW_SCHEMA, "eligible": True, "window": copy.deepcopy(capture["window"]),
            "mask": list(capture["mask"]), "action": motion_policy.ACTIONS.index(kind),
            "absolute_target": target, "label_origin": "first_direct_assistant_correction_action",
            "selection_rule": RULE, "source": copy.deepcopy(envelope["student_source"]),
            "capture_sha256": capture["capture_sha256"], "branch_envelope_sha256": envelope["envelope_sha256"],
            "teacher_session_id": manifest["session_id"], "teacher_event_sha256": first["event_sha256"],
            "physical_assessment_event_sha256": events[-1]["event_sha256"],
            "simulated": True, "actual_hardware": False}
