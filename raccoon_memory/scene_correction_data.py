"""Student decision snapshots and direct-correction rows; no model or physics execution."""
from __future__ import annotations

import copy
import hashlib
import json
import math

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, scene_policy, task_catalog


CAPTURE_SCHEMA = "scene-student-decision-capture.v1"
BRANCH_SCHEMA = "scene-direct-correction-branch.v1"
ROW_SCHEMA = "scene-direct-correction-sample.v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def make_capture(*, index, registered_scene, task, observations, window, mask, state,
                 arm_servo_target_deg, gripper_servo_opening_m, applied_command_count):
    """Seal exactly the already-computed input window and the pre-inference state."""
    before, task = scene_features.transfer_task(registered_scene, task)
    value = {"schema": CAPTURE_SCHEMA, "decision_index": index,
             "registered_scene": [list(stack) for stack in before], "task": task,
             "observations": copy.deepcopy(observations), "window": copy.deepcopy(window),
             "mask": list(mask), "simulation_state": copy.deepcopy(state),
             "arm_servo_target_deg": list(arm_servo_target_deg),
             "gripper_servo_opening_m": gripper_servo_opening_m,
             "applied_student_command_count": applied_command_count,
             "state_sha256": digest(state), "window_sha256": digest({"window": window, "mask": mask}),
             "history_includes_current_observation": True, "simulated": True, "actual_hardware": False}
    value["capture_sha256"] = digest(value)
    validate_capture(value)
    return value


def validate_capture(capture):
    if not isinstance(capture, dict) or capture.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unsupported student capture")
    body = {key: value for key, value in capture.items() if key != "capture_sha256"}
    if capture.get("capture_sha256") != digest(body):
        raise ValueError("student capture hash differs")
    index, count = capture.get("decision_index"), capture.get("applied_student_command_count")
    if (type(index) is not int or index < 0 or type(count) is not int or not 0 <= count <= index
            or capture.get("simulated") is not True or capture.get("actual_hardware") is not False
            or capture.get("history_includes_current_observation") is not True):
        raise ValueError("invalid student decision provenance")
    before, task = scene_features.transfer_task(capture.get("registered_scene"), capture.get("task"))
    observations = capture.get("observations")
    length = scene_features.HISTORY_LENGTH
    if not isinstance(observations, list) or len(observations) != min(index + 1, length):
        raise ValueError("student history does not contain each captured decision exactly once")
    vectors = [scene_features.encode_scene_features(obs, before, task) for obs in observations]
    window, mask = scene_features.history_window(vectors, length)
    if (capture.get("window") != window or capture.get("mask") != mask
            or capture.get("window_sha256") != digest({"window": window, "mask": mask})):
        raise ValueError("captured student window does not match its observations")
    state = capture.get("simulation_state")
    if not isinstance(state, dict) or capture.get("state_sha256") != digest(state):
        raise ValueError("student state hash differs")
    pending = state.get("pending_transfer")
    expected_after = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
    if (state.get("layout_status") != "in_progress" or state.get("layout_uncertainty_reason") is not None
            or task_catalog.scene(state.get("committed_layout")) != before or not isinstance(pending, dict)
            or task_catalog.scene(pending.get("before_layout")) != before
            or task_catalog.scene(pending.get("after_layout")) != expected_after
            or pending.get("source_slot") != task["source_slot"]
            or pending.get("destination_slot") != task["destination_slot"]
            or pending.get("moved_object_id") != task["object_id"]):
        raise ValueError("captured state is not the matching in-progress transfer")
    arm, opening = capture.get("arm_servo_target_deg"), capture.get("gripper_servo_opening_m")
    if (not isinstance(arm, list) or len(arm) != 4
            or any(type(q) not in (int, float) or not math.isfinite(q) or not lo <= q <= hi
                   for q, (lo, hi) in zip(arm, JOINT_LIMITS))
            or type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= .05):
        raise ValueError("invalid saved servo targets")
    return capture


def validate_branch_envelope(envelope):
    if not isinstance(envelope, dict) or envelope.get("schema") != BRANCH_SCHEMA:
        raise ValueError("unsupported correction branch envelope")
    if envelope.get("envelope_sha256") != digest({k: v for k, v in envelope.items() if k != "envelope_sha256"}):
        raise ValueError("branch envelope hash differs")
    capture = validate_capture(envelope.get("capture"))
    source = envelope.get("student_source")
    selection = envelope.get("selection")
    if (not isinstance(source, dict) or not isinstance(selection, dict)
            or not isinstance(source.get("student_run_id"), str) or not source["student_run_id"]
            or not isinstance(source.get("parent_group"), str)
            or source.get("task_group") != scene_policy.split_group_for(capture["registered_scene"], capture["task"])
            or source.get("split") != "train" or source.get("simulated") is not True
            or source.get("actual_hardware") is not False
            or selection.get("rule") != "fixed_parent_group_seed_priority_first_confirmed_failure"
            or selection.get("termination") not in ("out_of_range_target", "nonfinite_model_output",
                                                      "model_done_physical_failure", "decision_limit")
            or selection.get("decision_index") != capture["decision_index"]):
        raise ValueError("branch source or fixed failure selection is incomplete")
    return envelope


def first_correction_row(envelope, manifest, events):
    """Return one first-direct-action row; do not create an episode or a done label."""
    validate_branch_envelope(envelope)
    capture = envelope["capture"]
    if (not isinstance(events, list) or not events or events[0].get("event") != "branch"
            or events[0].get("branch_envelope_sha256") != envelope["envelope_sha256"]
            or events[0].get("state_after_sha256") != capture["state_sha256"]):
        raise ValueError("direct session does not start at the selected student state")
    if (not manifest.get("finished") or events[-1].get("event") != "finish"
            or events[-1].get("simulation_physical_criterion", {}).get("success") is not True):
        return {"eligible": False, "exclusion_reason": "direct_branch_not_confirmed_successful"}
    if (len(events) < 3 or events[1].get("event") != "review"
            or events[1].get("review_source") != "direct_assistant_simulation_observation"
            or events[1].get("scene_review") not in ("consistent", "target_grasped", "target_placed")
            or events[1].get("scene_evidence", {}).get("allowed") is not True
            or events[1].get("state_before_sha256") != capture["state_sha256"]
            or events[1].get("state_after_sha256") != capture["state_sha256"]):
        return {"eligible": False, "exclusion_reason": "initial_scene_not_confirmed_for_direct_correction"}
    first = events[2]
    arm, opening = capture["arm_servo_target_deg"], capture["gripper_servo_opening_m"]
    if first.get("state_before_sha256") != capture["state_sha256"]:
        raise ValueError("first teacher decision is not from the captured state")
    if first.get("event") == "finish":
        if first.get("assistant_done_proposal") is not True:
            return {"eligible": False, "exclusion_reason": "no_explicit_first_teacher_action"}
        kind, target = "done", [0.0] * 4
    elif first.get("event") == "action":
        kind, action = first.get("policy_action"), first.get("action")
        if kind not in ("move", "open", "close") or not isinstance(action, dict):
            raise ValueError("first action is not an explicitly labelled direct policy action")
        if action.get("seconds") != 1.0:
            raise ValueError("direct action duration must be one second")
        angles, finger = action.get("joints_deg"), action.get("gripper_opening_m")
        if (not isinstance(angles, list) or len(angles) != 4
                or any(type(q) not in (int, float) or not math.isfinite(q) or not lo <= q <= hi
                       for q, (lo, hi) in zip(angles, JOINT_LIMITS))
                or type(finger) not in (int, float) or not math.isfinite(finger)):
            raise ValueError("invalid teacher command")
        if kind == "move":
            if abs(finger - opening) > 1e-9:
                raise ValueError("move changed the finger servo target")
            target = [q / motion_policy.JOINT_SCALE_DEG for q in angles]
        else:
            if any(abs(x-y) > 1e-9 for x, y in zip(angles, arm)) or abs(finger - (0.04 if kind == "open" else 0.0)) > 1e-9:
                raise ValueError("gripper action did not hold the previous arm servo targets")
            target = [0.0] * 4
    else:
        raise ValueError("first teacher event has no policy action")
    if capture["applied_student_command_count"] < 1:
        return {"eligible": False, "exclusion_reason": "no_prior_student_command_applied"}
    return {"schema": ROW_SCHEMA, "eligible": True, "window": copy.deepcopy(capture["window"]),
            "mask": list(capture["mask"]), "action": motion_policy.ACTIONS.index(kind),
            "absolute_target": target, "label_origin": "first_direct_assistant_correction_action",
            "source": copy.deepcopy(envelope["student_source"]),
            "capture_sha256": capture["capture_sha256"], "branch_envelope_sha256": envelope["envelope_sha256"],
            "teacher_session_id": manifest["session_id"], "teacher_event_sha256": first["event_sha256"],
            "physical_assessment_event_sha256": events[-1]["event_sha256"],
            "simulated": True, "actual_hardware": False}
