"""Recorded-observation metrics shared by exclusive and simultaneous outputs."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import torch

from raccoon_memory import motion_policy
from raccoon_memory.scene_simultaneous_adapter import decode_exclusive, decode_simultaneous


def score_recorded(condition, outputs, labels, original_actions):
    """Score proposed commands against original servo targets at saved observations.

    Joint/gripper error uses only true-continue rows where a finite, in-range
    command would be applied. DONE, invalid outputs and extra commands retain
    separate counts rather than being silently filled into that denominator.
    """
    required = ("joint_target_deg", "gripper_target", "done_target", "prior_joint_deg", "prior_opening_m", "prior_gripper")
    if any(k not in labels for k in required):
        raise ValueError("recorded labels lack original or prior servo targets")
    count = len(original_actions)
    if not count or any(len(labels[k]) != count for k in required):
        raise ValueError("recorded label lengths differ")
    if condition == "exclusive":
        if set(outputs) != {"action_logits", "joint_output"}:
            raise ValueError("exclusive output contract differs")
    elif condition == "simultaneous":
        if set(outputs) != {"joint_output", "gripper_logits", "stop_logits"}:
            raise ValueError("simultaneous output contract differs")
    else:
        raise ValueError("unknown policy condition")
    if any(len(x) != count for x in outputs.values()):
        raise ValueError("recorded output lengths differ")
    arrays = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in labels.items()}
    action = original_actions.detach().cpu().tolist() if isinstance(original_actions, torch.Tensor) else list(original_actions)
    counts, rows = Counter(), []
    stop_confusion = {truth: {prediction: 0 for prediction in ("continue", "done", "unknown")}
                      for truth in ("continue", "done")}
    action_confusion = {truth: {prediction: 0 for prediction in (*motion_policy.ACTIONS, "unknown")}
                        for truth in motion_policy.ACTIONS}
    errors, grip_correct = [], 0
    for i in range(count):
        truth_done = int(arrays["done_target"][i])
        if truth_done not in (0, 1) or not 0 <= int(action[i]) < 4 or (int(action[i]) == motion_policy.DONE) != bool(truth_done):
            raise ValueError("stop target and original action disagree")
        previous_q = arrays["prior_joint_deg"][i].tolist()
        previous_open = float(arrays["prior_opening_m"][i])
        args = [outputs[k][i] for k in (("action_logits", "joint_output") if condition == "exclusive"
                                        else ("joint_output", "gripper_logits", "stop_logits"))]
        prior_grip = int(arrays["prior_gripper"][i])
        if prior_grip not in (0, 1):
            raise ValueError("prior gripper command is missing")
        prior_command = "open" if prior_grip == 0 else "closed"
        decoded = (decode_exclusive(*args, previous_q, previous_open, prior_gripper_command=prior_command)
                   if condition == "exclusive" else decode_simultaneous(*args, previous_q, previous_open,
                                                                        prior_gripper_command=prior_command))
        prediction = "unknown" if decoded["done"] is None else "done" if decoded["done"] else "continue"
        truth = "done" if truth_done else "continue"
        stop_confusion[truth][prediction] += 1
        counts["total_rows"] += 1
        counts["true_" + truth] += 1
        counts[decoded["status"]] += 1
        if condition == "exclusive":
            action_confusion[motion_policy.ACTIONS[int(action[i])]][decoded.get("exclusive_action") or "unknown"] += 1
        row = {"index": i, "teacher_done": bool(truth_done), "decoded": decoded,
               "teacher_joint_deg": None, "teacher_gripper": None, "joint_error_deg": None,
               "gripper_correct": None, "teacher_gripper_transition": None}
        if not truth_done:
            target = [float(x) for x in arrays["joint_target_deg"][i]]
            grip = int(arrays["gripper_target"][i])
            previous_grip = prior_grip
            if len(target) != 4 or not all(math.isfinite(x) for x in target) or grip not in (0, 1) or previous_grip not in (0, 1):
                raise ValueError("true-continue row lacks its full recorded targets")
            row["teacher_joint_deg"], row["teacher_gripper"] = target, ("open" if grip == 0 else "closed")
            row["teacher_gripper_transition"] = (grip != previous_grip)
            if decoded["status"] == "command":
                q = decoded["proposed_joint_deg"]
                err = [abs(a - b) for a, b in zip(q, target)]
                if condition == "exclusive" and decoded["exclusive_action"] == "move":
                    predicted_grip = previous_grip
                else:
                    predicted_grip = 0 if decoded["proposed_gripper_opening_m"] == 0.04 else 1
                correct = predicted_grip == grip
                row["joint_error_deg"], row["gripper_correct"] = err, bool(correct)
                errors.append(err)
                grip_correct += int(correct)
                counts["valid_command_with_teacher_target"] += 1
                if decoded["gripper_transition"]:
                    counts["predicted_gripper_transition_with_teacher_target"] += 1
            elif decoded["status"] == "done":
                counts["premature_done"] += 1
            else:
                counts["invalid_or_nonfinite_on_teacher_continue"] += 1
        elif decoded["status"] == "command":
            counts["extra_command_on_teacher_done"] += 1
        if decoded["status"] == "command":
            counts["valid_change_" + decoded["change_category"]] += 1
        rows.append(row)
    error_array = np.asarray(errors, dtype=np.float64)
    metrics = {"count": count, "counts": dict(counts), "stop_confusion": stop_confusion,
        "joint_and_gripper_denominator": len(errors),
        "joint_mae_deg": float(error_array.mean()) if len(errors) else None,
        "joint_axis_mae_deg": error_array.mean(axis=0).tolist() if len(errors) else None,
        "maximum_single_joint_error_deg": float(error_array.max()) if len(errors) else None,
        "gripper_correct": grip_correct,
        "gripper_accuracy_on_valid_applied_commands": grip_correct / len(errors) if len(errors) else None,
        "metric_scope": "saved observations; conditional on true continue and finite in-range proposed command",
        "stop_and_invalid_rows_are_reported_separately": True}
    if condition == "exclusive":
        metrics["exclusive_four_class_confusion_auxiliary"] = action_confusion
    return metrics, rows


def summarize_rollouts(records):
    """Aggregate this adapter's decoded decisions, retaining rejected targets."""
    if not isinstance(records, list) or not records:
        raise ValueError("student rollout records must be nonempty")
    counts, rejected, switches = Counter(), [], []
    count_keys = ("model_calls_started", "model_calls_returned", "command_calls_started", "command_calls_returned",
                  "physical_assessment_calls_started", "physical_assessment_calls_returned")
    actual_calls = Counter({key: 0 for key in count_keys})
    for record in records:
        result = record["result"]
        called, returned = record.get("student_rollout_called"), record.get("student_call_returned")
        if type(called) is not bool or type(returned) is not bool or (returned and not called):
            raise ValueError("student rollout call accounting is missing or inconsistent")
        experience = record.get("experience_counts")
        if (not isinstance(experience, dict) or set(experience) != set(count_keys)
                or any(type(experience[k]) is not int or experience[k] < 0 for k in count_keys)
                or any(experience[started] < experience[finished] for started, finished in
                       (("model_calls_started", "model_calls_returned"), ("command_calls_started", "command_calls_returned"),
                        ("physical_assessment_calls_started", "physical_assessment_calls_returned")))
                or (not called and any(experience.values()))):
            raise ValueError("actual model, command or physical assessment call counts differ")
        actual_calls.update(experience)
        if result.get("teacher_assistance_count") != 0 or result.get("actual_hardware") is not False:
            raise ValueError("rollout lacks student-only simulated provenance")
        success, termination = result.get("success"), result.get("termination")
        if success not in (True, False, None):
            raise ValueError("rollout success is not tri-state")
        if not returned and success is not None:
            raise ValueError("a student call that did not return cannot be a confirmed outcome")
        if success is True and (termination != "model_done" or result.get("assessment", {}).get("success") is not True):
            raise ValueError("success lacks a matching physical assessment")
        if success is None and termination != "evaluation_error":
            raise ValueError("unconfirmed rollout was not identified as an evaluation error")
        counts["success" if success is True else "failure" if success is False else "unknown"] += 1
        counts["student_rollout_called"] += int(called)
        counts["student_call_returned"] += int(returned)
        counts["termination_" + str(termination)] += 1
        for decision in result.get("decisions", []):
            decoded = decision.get("decoded")
            if decoded is None:
                if termination != "evaluation_error":
                    raise ValueError("a completed decision lacks the adapter's decoded record")
                continue
            if decoded.get("status") == "out_of_range_target":
                q = decoded.get("proposed_joint_deg")
                if (not isinstance(q, list) or len(q) != 4 or not all(math.isfinite(v) for v in q)
                        or decoded.get("command") is not None
                        or decision.get("result") not in ("no_command", "cancelled_before_command")
                        or (decision.get("result") == "cancelled_before_command" and termination != "cancelled")
                        or (decision.get("result") == "no_command" and termination != "out_of_range_target")):
                    raise ValueError("rejected out-of-range target is missing or was applied")
                rejected.append({"case_id": record.get("case_id"), "seed": record.get("seed"),
                    "condition": record.get("condition"), "decision_index": decision.get("index"),
                    "proposed_joint_deg": q, "proposed_gripper_opening_m": decoded.get("proposed_gripper_opening_m"),
                    "decision_result": decision["result"], "termination": termination})
                counts["outside_targets"] += 1
            elif decision.get("result") == "command_applied":
                if decoded.get("status") != "command" or not isinstance(decoded.get("command"), dict):
                    raise ValueError("applied command is inconsistent with the decoded result")
                counts["commands"] += 1
                counts["change_" + decoded["change_category"]] += 1
                if decoded.get("gripper_transition") is True:
                    switches.append({"case_id": record.get("case_id"), "seed": record.get("seed"),
                        "condition": record.get("condition"), "decision_index": decision.get("index"),
                        "prior_joint_deg": decoded["prior_joint_deg"], "new_joint_deg": decoded["proposed_joint_deg"],
                        "joint_change_deg": decoded["joint_change_deg"],
                        "teacher_joint_error_deg": None,
                        "teacher_error_scope": "no teacher target at this student-visited observation"})
                    counts["gripper_transitions"] += 1
            elif decision.get("result") == "done_no_command":
                if decoded.get("status") != "done" or decoded.get("command") is not None:
                    raise ValueError("DONE unexpectedly contains an actuator command")
                counts["done_without_command"] += 1
    return {"case_attempts": len(records), "counts": dict(counts), "experience_counts": dict(actual_calls),
            "out_of_range_proposals": rejected,
            "gripper_transitions": switches, "teacher_assistance_count": 0,
            "scope": "saved student-only simulation; no teacher target exists for student-visited states"}
