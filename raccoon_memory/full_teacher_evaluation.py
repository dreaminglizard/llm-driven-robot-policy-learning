"""Bind a full-teacher checkpoint to its dataset and run existing scene rollouts.

This module never creates or resets a simulator, chooses evaluation cases, or
trains a model.  Callers supply an already initialized simulator and explicit
tasks.  OPEN quality and observed post-OPEN motion are separate from the
existing whole-scene physical assessment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from raccoon_memory import full_teacher_training as training
from raccoon_memory import scene_features, task_catalog, teacher_release_quality
from raccoon_memory import motion_sim, scene_motion_sim, sim_data
from scripts import evaluate_scene_policy as evaluator

SCHEMA = "raccoon-full-teacher-student-evaluation.v1"
MAX_DECISIONS = 24
ASSESSMENT_SAMPLES = 501


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    raw = json.dumps(evaluator.safe(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size and
            all(type(x) in (int, float) and math.isfinite(x) for x in value))


def _finite_scalar(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class BoundPolicy:
    """An independently verified model and the exact data used to train it."""

    model: nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_directory: Path
    dataset_manifest_sha256: str
    input_snapshot: dict
    source_snapshot: dict
    review_source_hashes: dict
    training_run_hashes: dict
    evaluation_source_hashes: dict
    model_state_sha256: str
    architecture: str
    episodes: int
    seed: int
    device: torch.device

    def verify(self):
        """Recheck saved inputs, reviewed sources, code and in-memory weights."""
        training._check_input_snapshot(self.dataset_directory, self.input_snapshot)
        training._check_source_snapshot(self.source_snapshot)
        training.exported._check_sources(self.review_source_hashes)
        training._check_source_snapshot(self.training_run_hashes)
        training._check_source_snapshot(self.evaluation_source_hashes)
        _need(training._file_sha(self.checkpoint_path) == self.checkpoint_sha256,
              "checkpoint changed since binding")
        _need(training.state_fingerprint(self.model)["sha256"] == self.model_state_sha256,
              "in-memory model changed since binding")

    def evidence(self):
        return {"checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256,
                "dataset_directory": str(self.dataset_directory),
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "model_state_sha256": self.model_state_sha256, "architecture": self.architecture,
                "episodes": self.episodes, "seed": self.seed, "device": str(self.device),
                "converted_input_sha256": dict(self.input_snapshot),
                "training_source_sha256": dict(self.source_snapshot),
                "training_run_sha256": dict(self.training_run_hashes),
                "review_source_count": len(self.review_source_hashes),
                "review_source_manifest_sha256": hashlib.sha256(
                    training._json_bytes(self.review_source_hashes)).hexdigest(),
                "evaluation_source_sha256": dict(self.evaluation_source_hashes)}


def load_bound_policy(dataset_directory, expected_manifest_sha256, checkpoint_path,
                      expected_checkpoint_sha256, *, device="cpu"):
    """Load only after the completed dataset and checkpoint provenance agree."""
    data = training.load_converted_dataset(dataset_directory, expected_manifest_sha256)
    directory = data["directory"]
    snapshot = training._snapshot_inputs(directory, data["manifest"])
    _need(snapshot["manifest.json"] == expected_manifest_sha256, "dataset manifest changed during binding")
    sources = training._source_snapshot(data)
    checkpoint_path = Path(checkpoint_path).resolve()
    run_hashes_before = _training_run_snapshot(checkpoint_path, expected_manifest_sha256,
                                               expected_checkpoint_sha256)
    model, payload = training.load_checkpoint(checkpoint_path, expected_checkpoint_sha256, device=device)
    _need(payload.get("dataset_manifest_sha256") == expected_manifest_sha256,
          "checkpoint and dataset manifest differ")
    _need(payload.get("review_source_hashes") == data["review_source_hashes"],
          "checkpoint and dataset reviewed sources differ")
    _need(payload.get("source_sha256") == sources, "checkpoint and dataset source fingerprints differ")
    run_hashes = _training_run_snapshot(checkpoint_path, expected_manifest_sha256, expected_checkpoint_sha256)
    _need(run_hashes == run_hashes_before, "training completion evidence changed during binding")
    paths = (Path(__file__), Path(evaluator.__file__), Path(teacher_release_quality.__file__),
             Path(motion_sim.__file__), Path(scene_motion_sim.__file__), Path(sim_data.__file__),
             Path(task_catalog.__file__))
    evaluation_sources = {str(path.resolve()): training._file_sha(path) for path in paths}
    bound = BoundPolicy(model=model, checkpoint_path=checkpoint_path,
        checkpoint_sha256=expected_checkpoint_sha256, dataset_directory=directory,
        dataset_manifest_sha256=expected_manifest_sha256, input_snapshot=dict(snapshot),
        source_snapshot=dict(sources), review_source_hashes=dict(data["review_source_hashes"]),
        training_run_hashes=run_hashes,
        evaluation_source_hashes=evaluation_sources, model_state_sha256=training.state_fingerprint(model)["sha256"],
        architecture=payload["architecture"], episodes=payload["episodes"], seed=payload["seed"],
        device=torch.device(device))
    bound.verify()
    return bound


def _training_run_snapshot(checkpoint_path, expected_manifest_sha256, expected_checkpoint_sha256):
    """Keep the already verified completion files fixed, not just the checkpoint."""
    result_path = checkpoint_path.parent / "result.json"
    train_result, train_result_raw = training._json(result_path)
    _need(isinstance(train_result, dict) and train_result.get("status") == "complete" and
          train_result.get("dataset_manifest_sha256") == expected_manifest_sha256 and
          train_result.get("files", {}).get("checkpoint.pt") == expected_checkpoint_sha256,
          "completed checkpoint evidence differs")
    _need(set(train_result["files"]) == {"checkpoint.pt", "train_config.json", "train_events.jsonl",
                                          "row_exposures.json"}, "completed training files missing")
    run_hashes = {str(result_path): hashlib.sha256(train_result_raw).hexdigest()}
    run_hashes.update({str(checkpoint_path.parent / name): digest
                       for name, digest in train_result["files"].items()})
    training._check_source_snapshot(run_hashes)
    return run_hashes


class _RecordedSimulator:
    """Capture the actual returns that the unchanged evaluator normally discards."""

    def __init__(self, sim):
        self._sim = sim
        self.commands = []
        self.assessments = []

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def command(self, joints_deg, gripper_opening_m, seconds=0.5):
        before = self._sim.observation()
        event = {"command": {"joints_deg": evaluator.safe(joints_deg),
                 "gripper_opening_m": gripper_opening_m, "seconds": seconds},
                 "before_observation": evaluator.safe(before), "returned": False,
                 "return_valid": None, "return_observation": None}
        self.commands.append(event)
        try:
            raw = self._sim.command(joints_deg, gripper_opening_m, seconds=seconds)
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        event["returned"] = True
        event["return_observation"] = evaluator.safe(raw)
        try:
            _need(isinstance(raw, dict) and raw.get("source") == "mujoco_simulation" and
                  raw.get("simulated") is True and raw.get("success") is not False and not raw.get("error") and
                  raw.get("layout_status") == "in_progress" and
                  _finite_scalar(raw.get("time_seconds")) and _finite_scalar(before.get("time_seconds")) and
                  math.isclose(float(seconds), 1.0, rel_tol=0.0, abs_tol=1e-12) and
                  math.isclose(float(raw["time_seconds"]) - float(before["time_seconds"]), 1.0,
                               rel_tol=0.0, abs_tol=1e-6) and
                  _finite_vector(raw.get("joints_deg"), 4) and _finite_vector(raw.get("tcp_mm"), 3) and
                  _finite_scalar(raw.get("gripper_opening_m")) and
                  isinstance(raw.get("privileged_simulation_only"), dict),
                  "command returned invalid or incomplete observation")
        except Exception as exc:
            event["return_valid"] = False
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        event["return_valid"] = True
        return raw

    def evaluate_transfer(self, before_layout, source_slot, destination_slot, stable_seconds=1.0):
        event = {"stable_seconds_requested": stable_seconds, "returned": False,
                 "return_valid": None, "return_value": None}
        self.assessments.append(event)
        try:
            result = self._sim.evaluate_transfer(before_layout, source_slot, destination_slot,
                                                 stable_seconds=stable_seconds)
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        event["returned"] = True
        event["return_value"] = evaluator.safe(result)
        try:
            _need(isinstance(result, dict) and type(result.get("sample_count")) is int and
                  result["sample_count"] == ASSESSMENT_SAMPLES and
                  _finite_scalar(result.get("stable_interval_seconds")) and
                  math.isclose(float(stable_seconds), 1.0, rel_tol=0.0, abs_tol=1e-12) and
                  math.isclose(float(result["stable_interval_seconds"]), 1.0, rel_tol=0.0, abs_tol=1e-6),
                  "assessment did not return one second and 501 samples")
        except Exception as exc:
            event["return_valid"] = False
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        event["return_valid"] = True
        return result


class _RecordedPolicy(nn.Module):
    """Record the very same model invocation without a second forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.history_length = model.history_length
        self.calls = []

    def forward(self, windows, masks):
        event = {"raw_window": evaluator.safe(windows), "mask": evaluator.safe(masks),
                 "returned": False, "transformed_window": None, "outputs": None}
        self.calls.append(event)
        try:
            transformed, _ = training.adapt_inputs(windows, masks)
            event["transformed_window"] = evaluator.safe(transformed)
            output = self.model(windows, masks)
            event["returned"] = True
            event["outputs"] = evaluator.safe(output)
            return output
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise


def _result_evidence(result, calls, command_events, assessment_events):
    """Check same-call results, returned commands, DONE and first history."""
    problems = []
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return ["missing_decisions"]
    for index, call in enumerate(calls):
        if index >= len(decisions):
            problems.append("model_call_without_decision")
            break
        decision = decisions[index]
        if call["returned"] and "action_logits" in decision:
            outputs = call["outputs"]
            if (not isinstance(outputs, list) or len(outputs) != 2 or outputs[0] != decision["action_logits"] or
                    outputs[1] != decision["absolute_joint_output"] or
                    call["mask"][0] != decision.get("valid_history")):
                problems.append("same_call_output_or_mask_mismatch")
            features = decision.get("policy_features")
            if (not isinstance(features, list) or len(features) != 82 or
                    torch.tensor(call["raw_window"][0][-1], dtype=torch.float32).tolist() !=
                    torch.tensor(features, dtype=torch.float32).tolist()):
                problems.append("same_call_current_features_mismatch")
            prior = [row.get("policy_features") for row in decisions[max(0, index - 7):index + 1]]
            if all(isinstance(row, list) and len(row) == 82 for row in prior):
                padding = 8 - len(prior)
                expected_window = [[0.0] * 82 for _ in range(padding)] + prior
                expected_mask = [False] * padding + [True] * len(prior)
                if (call["mask"][0] != expected_mask or
                        torch.tensor(call["raw_window"][0], dtype=torch.float32).tolist() !=
                        torch.tensor(expected_window, dtype=torch.float32).tolist()):
                    problems.append("same_call_history_window_mismatch")
            else:
                problems.append("same_call_history_features_missing")
    if calls and calls[0].get("mask") != [[False] * 7 + [True]]:
        problems.append("first_history_not_one")
    applied = [item for item in decisions if item.get("result") == "command_applied"]
    verified = [event for event in command_events if event.get("return_valid") is True]
    if len(applied) != len(verified):
        problems.append("applied_command_return_count_mismatch")
    for item, event in zip(applied, verified):
        command = item.get("command") or {}
        actual = event["command"]
        if (command.get("joints_deg") != actual["joints_deg"] or
                command.get("gripper_opening_m") != actual["gripper_opening_m"] or
                command.get("seconds") != actual["seconds"] or
                item.get("simulation_before") != event["before_observation"] or
                item.get("simulation_after") != event["return_observation"]):
            problems.append("applied_command_or_return_mismatch")
    if result.get("termination") == "model_done":
        if (not decisions or decisions[-1].get("action") != "done" or
                decisions[-1].get("result") != "done_no_command" or len(assessment_events) != 1 or
                assessment_events[0].get("return_valid") is not True or
                assessment_events[0].get("return_value") != result.get("assessment")):
            problems.append("done_or_assessment_evidence_mismatch")
    elif result.get("assessment") is not None and result.get("termination") != "cancelled":
        problems.append("unexpected_assessment")
    return list(dict.fromkeys(problems))


def _release_and_motion(result):
    """Use applied CLOSE/OPEN boundaries and report later MOVE observations."""
    decisions = result.get("decisions") or []
    scene, task = result.get("registered_scene"), result.get("task")
    holding, last_close, releases = False, None, []
    for position, item in enumerate(decisions):
        if item.get("result") != "command_applied":
            continue
        if item.get("action") == "close":
            holding, last_close = True, item.get("index", position)
        elif item.get("action") == "open" and holding:
            holding = False
            quality = teacher_release_quality.evaluate_release(item.get("simulation_before"),
                item.get("simulation_after"), before_layout=scene,
                source_slot=task["source_slot"], destination_slot=task["destination_slot"])
            moves = []
            for later in decisions[position + 1:]:
                if later.get("result") != "command_applied":
                    continue
                if later.get("action") == "close":
                    break
                if later.get("action") != "move":
                    continue
                before, after = later.get("simulation_before") or {}, later.get("simulation_after") or {}
                record = {"decision_index": later.get("index"), "actual_move_applied": True,
                          "command": later.get("command"),
                          "before_observation_sha256": _digest(before), "after_observation_sha256": _digest(after),
                          "motion_observed": None, "joint_delta_deg": None, "tcp_delta_mm": None,
                          "before_tcp_mm": before.get("tcp_mm"), "after_tcp_mm": after.get("tcp_mm")}
                if (_finite_vector(before.get("joints_deg"), 4) and _finite_vector(after.get("joints_deg"), 4) and
                        _finite_vector(before.get("tcp_mm"), 3) and _finite_vector(after.get("tcp_mm"), 3)):
                    joint_delta = [float(a) - float(b) for a, b in zip(after["joints_deg"], before["joints_deg"])]
                    tcp_delta = [float(a) - float(b) for a, b in zip(after["tcp_mm"], before["tcp_mm"])]
                    record.update(joint_delta_deg=joint_delta, tcp_delta_mm=tcp_delta,
                                  motion_observed=any(value != 0.0 for value in joint_delta + tcp_delta))
                moves.append(record)
            releases.append({"close_decision_index": last_close, "open_decision_index": item.get("index", position),
                             "before_observation_sha256": _digest(item.get("simulation_before")),
                             "after_observation_sha256": _digest(item.get("simulation_after")),
                             "quality": quality, "post_open_moves": moves,
                             "retreat_intent": "not_inferred_from_move"})
    if not releases:
        status = "not_reached"
    elif any(x["quality"]["status"] == "ineligible" for x in releases):
        status = "ineligible"
    elif any(x["quality"]["status"] == "unconfirmed" for x in releases):
        status = "unconfirmed"
    else:
        status = "eligible"
    return {"status": status, "releases": releases,
            "scope": "recorded OPEN boundaries and post-OPEN MOVE only; not physical success or retreat intent"}


def _servo_snapshot(sim):
    arm = [float(sim.data.ctrl[index]) for index in sim._arm_actuators]
    fingers = [float(sim.data.ctrl[index]) for index in sim._finger_actuators]
    _need(_finite_vector(arm, 4) and _finite_vector(fingers, 2), "invalid current servo targets")
    return {"arm_target_rad": arm, "finger_targets_m": fingers}


def _require_virtual_sim(sim):
    """Reject non-simulator objects before any model or command call."""
    _need(isinstance(sim, scene_motion_sim.SceneMotionSim), "a virtual SceneMotionSim is required")
    observation = sim.observation()
    _need(isinstance(observation, dict) and observation.get("source") == "mujoco_simulation" and
          observation.get("simulated") is True, "simulator observation provenance differs")


def _rollout(bound, sim, scene, task, cancel):
    recorded_sim = _RecordedSimulator(sim)
    recorded_model = _RecordedPolicy(bound.model)
    result, initial_servo = None, None
    try:
        initial_servo = _servo_snapshot(sim)
        result = evaluator.run_transfer(recorded_model, recorded_sim, scene, task,
                                        max_decisions=MAX_DECISIONS, device=bound.device, cancel=cancel)
    except Exception as exc:
        return {"outcome": None, "physical_status": "unconfirmed", "error":
                {"type": type(exc).__name__, "message": str(exc)}, "raw_result": None,
                "model_calls": recorded_model.calls, "commands": recorded_sim.commands,
                "assessment_calls": recorded_sim.assessments, "initial_servo": initial_servo,
                "final_servo": None, "release": None}
    problems = []
    try:
        problems = _result_evidence(result, recorded_model.calls, recorded_sim.commands, recorded_sim.assessments)
    except Exception as exc:
        problems.append("evidence_check_error: " + type(exc).__name__ + ": " + str(exc))
    try:
        final_servo = _servo_snapshot(sim)
    except Exception as exc:
        final_servo = None
        problems.append("final_servo_error: " + type(exc).__name__ + ": " + str(exc))
    try:
        release = _release_and_motion(result)
    except Exception as exc:
        release = {"status": "unconfirmed", "releases": [],
                   "error": {"type": type(exc).__name__, "message": str(exc)}}
    termination = result.get("termination")
    physical = result.get("physical_success")
    if problems or result.get("teacher_assistance_count") != 0 or result.get("actual_hardware") is not False:
        outcome, physical_status = None, "unconfirmed"
    elif termination == "model_done" and type(physical) is bool and result.get("success") is physical:
        outcome, physical_status = physical, "success" if physical else "failure"
    elif termination in ("out_of_range_target", "decision_limit") and physical is None:
        outcome, physical_status = False, "not_assessed"
    else:
        outcome, physical_status = None, "unconfirmed"
    return {"outcome": outcome, "physical_status": physical_status, "evidence_problems": problems,
            "raw_result": result, "model_calls": recorded_model.calls,
            "commands": recorded_sim.commands, "assessment_calls": recorded_sim.assessments,
            "initial_servo": initial_servo, "final_servo": final_servo, "release": release}


def _check_after(bound, result):
    try:
        bound.verify()
        result["post_evaluation_integrity"] = "verified"
        return True
    except Exception as exc:
        result["post_evaluation_integrity"] = "unconfirmed"
        result["integrity_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["outcome"] = None
        return False


def evaluate_single(bound, sim, registered_scene, task, *, cancel=None):
    """Evaluate one explicitly supplied task on an already initialized simulator."""
    _need(isinstance(bound, BoundPolicy), "a bound policy is required")
    _require_virtual_sim(sim)
    scene_features.transfer_task(registered_scene, task)
    bound.verify()
    result = _rollout(bound, sim, registered_scene, task, cancel)
    output = {"schema": SCHEMA, "kind": "single", "binding": bound.evidence(), "result": result,
              "outcome": result["outcome"], "physical_execution": "virtual_only", "teacher_assistance": False}
    _check_after(bound, output)
    return output


def evaluate_connection(bound, sim, registered_scene, first_task, second_task, *, cancel=None):
    """Run two transfers on the same state, never resetting between them."""
    _need(isinstance(bound, BoundPolicy), "a bound policy is required")
    _require_virtual_sim(sim)
    before, first = scene_features.transfer_task(registered_scene, first_task)
    middle = task_catalog.transfer(before, first["source_slot"], first["destination_slot"])
    scene_features.transfer_task(middle, second_task)  # Reject invalid second tasks before the first command.
    bound.verify()
    output = {"schema": SCHEMA, "kind": "connection", "binding": bound.evidence(),
              "first": None, "second": None, "second_status": "not_started", "outcome": None,
              "handoff": None, "physical_execution": "virtual_only", "teacher_assistance": False}
    first_result = _rollout(bound, sim, before, first, cancel)
    output["first"] = first_result
    if not _check_after(bound, output):
        output["second_status"] = "not_reached_integrity_unconfirmed"
        return output
    if first_result["outcome"] is not True:
        output["second_status"] = ("not_reached_first_failed" if first_result["outcome"] is False
                                   else "not_reached_first_unconfirmed")
        output["outcome"] = first_result["outcome"]
        return output
    try:
        first_final = first_result["raw_result"]["final_state"]
        actual_state = evaluator.safe(sim.save_state())
        actual_servo = _servo_snapshot(sim)
        current_obs = sim.observation()
        _need(first_final == actual_state and task_catalog.scene(current_obs.get("registered_layout")) == middle and
              current_obs.get("layout_status") == "registered" and
              first_result["final_servo"] == actual_servo, "first final state, layout or servo differs")
        output["handoff"] = {"first_final_state_sha256": _digest(first_final),
                             "second_initial_state_before_call_sha256": _digest(actual_state),
                             "first_final_state": first_final, "second_initial_state_before_call": actual_state,
                             "first_final_servo": first_result["final_servo"],
                             "servo_before_second": actual_servo, "full_state_matches": True,
                             "servo_matches": True}
        second_result = _rollout(bound, sim, middle, second_task, cancel)
        output["second"] = second_result
        raw_second = second_result.get("raw_result")
        _need(isinstance(raw_second, dict) and raw_second.get("initial_state") == first_final and
              second_result["initial_servo"] == first_result["final_servo"],
              "second initial full state or servo differs from the first final state")
        first_mask = ((second_result.get("model_calls") or [{}])[0].get("mask"))
        if first_mask is not None:
            _need(first_mask == [[False] * 7 + [True]], "second transfer history did not restart at H=1")
        output["handoff"].update(second_result_initial_state_sha256=_digest(raw_second["initial_state"]),
                                 second_first_history_mask=first_mask)
        output["second_status"] = "returned"
        output["outcome"] = second_result["outcome"]
    except Exception as exc:
        output["second_status"] = "unconfirmed"
        output["outcome"] = None
        output["handoff_error"] = {"type": type(exc).__name__, "message": str(exc)}
    _check_after(bound, output)
    return output
