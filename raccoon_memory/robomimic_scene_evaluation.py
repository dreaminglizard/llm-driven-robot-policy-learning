"""Read-only preparation helpers for the official core's scene evaluation.

This module owns no model, environment, device, or background process.  The
caller supplies the already-bound completion gate and, for mock checks, the
observation and assessment functions.  Merely importing it cannot run physics.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from raccoon_memory import scene_features, task_catalog


def decode_action(proposed_action, *, valid_target: Callable):
    """Decode an untouched absolute action; never clip a proposed target.

    A zero gripper sample follows the nonnegative/open branch. The raw sample
    remains in the returned record even when the joint target is invalid.
    """
    if not isinstance(proposed_action, torch.Tensor) or tuple(proposed_action.shape) != (1, 5):
        raise ValueError("the official action must have shape [1,5]")
    values = [float(x) for x in proposed_action.detach().cpu()[0].tolist()]
    if not all(math.isfinite(value) for value in values):
        return {"status": "nonfinite_model_output", "raw_action": values, "command": None}
    target = [value * 180.0 for value in values[:4]]
    opening = 0.04 if values[4] >= 0.0 else 0.0
    decoded = {"raw_action": values, "raw_gripper": values[4],
               "gripper_command": "open" if opening else "closed",
               "proposed_joint_deg": target, "proposed_gripper_opening_m": opening}
    if not all(math.isfinite(value) for value in target):
        return {**decoded, "status": "nonfinite_model_output", "command": None}
    if not valid_target(target):
        return {**decoded, "status": "out_of_range_target", "command": None}
    return {**decoded, "status": "command", "command": {"joints_deg": target,
            "gripper_opening_m": opening, "seconds": 1.0}}


def run_transfer(gate, sim, registered_scene, task, *, max_decisions=24, cancel=None,
                 observe_fn=None, encode_fn=None, valid_target_fn=None,
                 read_control_fn=None, validate_assessment_fn=None):
    """Perform one authorized student-only transfer with existing physics rules.

    The real simulator and gate are injected by the evaluation runner. Each
    decision observes once through the existing feature encoder, then calls
    ``gate.decide`` exactly once. The action sample discarded by DONE is never
    sent to a command sink. No teacher or future observation is consulted.
    """
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("positive decision limit required")
    if observe_fn is None:
        from raccoon_memory.sim_data import sim_policy_observation
        observe_fn = sim_policy_observation
    if encode_fn is None:
        encode_fn = scene_features.encode_scene_features
    if read_control_fn is None:
        from raccoon_memory.scene_simultaneous_adapter import read_current_control
        read_control_fn = read_current_control
    if valid_target_fn is None or validate_assessment_fn is None:
        from scripts.evaluate_scene_policy import valid_target, validate_assessment
        valid_target_fn = valid_target if valid_target_fn is None else valid_target_fn
        validate_assessment_fn = validate_assessment if validate_assessment_fn is None else validate_assessment_fn

    before, task = scene_features.transfer_task(registered_scene, task)
    initial_obs = sim.observation()
    if (task_catalog.scene(initial_obs["registered_layout"]) != before
            or initial_obs.get("layout_status") != "registered"):
        raise ValueError("starting scene must be the trusted registered layout")
    control = read_control_fn(sim)
    if control.gripper_opening_m < 0.025 or initial_obs["gripper_opening_m"] < 0.025:
        raise ValueError("a new transfer must start with an open gripper")
    result = {"registered_scene": before, "task": task, "decisions": [], "termination": None,
              "success": None, "physical_success": None, "assessment": None,
              "teacher_assistance_count": 0, "actual_hardware": False,
              "initial_state": sim.save_state(), "initial_observation": initial_obs}
    if cancel is not None and cancel.is_set():
        result.update(termination="cancelled", success=False, final_state=sim.save_state(),
                      layout_status=initial_obs["layout_status"])
        return result

    gate.reset_episode()  # Delegates to the official reset once per episode.
    sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
    gripper = "open"
    last_joints, last_opening = list(control.joints_deg), control.gripper_opening_m
    for index in range(max_decisions):
        if cancel is not None and cancel.is_set():
            result.update(termination="cancelled", success=False)
            break
        item = {"index": index}
        result["decisions"].append(item)
        try:
            actual = read_control_fn(sim)
            if (any(abs(a - b) > 1e-8 for a, b in zip(actual.joints_deg, last_joints))
                    or abs(actual.gripper_opening_m - last_opening) > 1e-8):
                raise ValueError("remembered and current actuator targets differ")
            obs = observe_fn(sim, gripper)
            vector = encode_fn(obs, before, task)
            if len(vector) != 82 or any(not math.isfinite(float(x)) for x in vector):
                raise ValueError("encoded observation must contain 82 finite features")
            item["policy_observation"] = {key: obs[key] for key in (
                "schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
                "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")}
            item["policy_features"] = vector
            item["simulation_before"] = sim.observation()
            decision = gate.decide(torch.tensor([vector], dtype=torch.float32, device="cpu"))
            item["official_decision_count"] = decision.decision_count
            item["completion"] = decision.kind
            if cancel is not None and cancel.is_set():
                result.update(termination="cancelled", success=False)
                item["result"] = "cancelled_before_command"
                break
            if decision.kind == "done":
                if decision.proposed_action is not None:
                    raise ValueError("done must suppress the sampled action")
                result["termination"] = "model_done"
                item["result"] = "done_no_command"
                break
            if decision.kind != "continue":
                raise ValueError("completion gate returned an unknown decision")
            decoded = decode_action(decision.proposed_action, valid_target=valid_target_fn)
            item["decoded"] = decoded
            if decoded["status"] != "command":
                result.update(termination=decoded["status"], success=False)
                item["result"] = "no_command"
                break
            command = decoded["command"]
            sim.command(command["joints_deg"], command["gripper_opening_m"], seconds=command["seconds"])
            gripper = decoded["gripper_command"]
            last_joints, last_opening = list(command["joints_deg"]), command["gripper_opening_m"]
            item["result"] = "command_applied"
            item["simulation_after"] = sim.observation()
        except Exception as exc:
            # CompletionGate rejects a nonfinite continue sample before exposing
            # it. This is the same no-command failure as a decoded nonfinite.
            if isinstance(exc, ValueError) and str(exc) in (
                    "official action proposal is not finite", "completion logits are not finite"):
                result.update(termination="nonfinite_model_output", success=False)
                item["result"] = "no_command"
            else:
                result.update(termination="evaluation_error", success=None,
                              error={"type": type(exc).__name__, "message": str(exc)})
            break
    else:
        result.update(termination="decision_limit", success=False)

    if result["termination"] == "model_done" and cancel is not None and cancel.is_set():
        result.update(termination="cancelled", success=False)
    if result["termination"] == "model_done":
        try:
            assessment = sim.evaluate_transfer(before, task["source_slot"], task["destination_slot"],
                                               stable_seconds=1.0)
            validate_assessment_fn(assessment, before, task)
            result.update(assessment=assessment, physical_success=assessment["success"],
                          success=assessment["success"])
            if cancel is not None and cancel.is_set():
                result.update(termination="cancelled", success=False)
        except Exception as exc:
            result.update(termination="evaluation_error", success=None,
                          error={"type": type(exc).__name__, "message": str(exc)})
    if result["success"] is not True:
        sim.mark_layout_uncertain(result["termination"])
    result["layout_status"] = sim.observation()["layout_status"]
    result["final_state"] = sim.save_state()
    return result
