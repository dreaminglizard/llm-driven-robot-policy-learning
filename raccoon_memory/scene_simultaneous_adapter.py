"""Separate decoders and simulator adapter for the paired output experiment.

This module does not create a simulator or access hardware. Both heads on a
nonterminal simultaneous output are applied together; DONE applies neither.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, sim_data, task_catalog

@dataclass(frozen=True)
class PriorControl:
    joints_deg: tuple[float, float, float, float]
    gripper_opening_m: float


def _values(value, length):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.shape == (1, length):
            value = value[0]
        if value.shape != (length,):
            raise ValueError("model output shape differs")
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("model output shape differs")
    if any(type(x) not in (int, float) for x in value):
        raise ValueError("model output must be numeric")
    return [float(x) for x in value]


def _prior(joints, opening):
    values = _values(joints, 4)
    if (not all(math.isfinite(x) for x in values)
            or any(not lo <= x <= hi for x, (lo, hi) in zip(values, JOINT_LIMITS))
            or type(opening) not in (int, float) or not math.isfinite(opening) or not 0 <= opening <= 0.05):
        raise ValueError("prior actuator controls are missing, nonfinite or outside their limits")
    # A previous opening need not equal one of the new binary output targets.
    return PriorControl(tuple(values), float(opening))


def _finish(q, opening, prior, metadata, prior_gripper_command, predicted_gripper_command):
    if not all(math.isfinite(x) for x in q) or not math.isfinite(opening):
        return {**metadata, "status": "nonfinite_model_output", "done": None, "command": None}
    change = [float(a) - float(b) for a, b in zip(q, prior.joints_deg)]
    grip_change = abs(opening - prior.gripper_opening_m) > 1e-8
    transition = (prior_gripper_command != predicted_gripper_command
                  if prior_gripper_command is not None and predicted_gripper_command is not None else None)
    joint_change = any(abs(x) > 1e-8 for x in change)
    valid = all(lo <= x <= hi for x, (lo, hi) in zip(q, JOINT_LIMITS))
    return {**metadata, "status": "command" if valid else "out_of_range_target", "done": False,
            "proposed_joint_deg": q, "proposed_gripper_opening_m": opening,
            "prior_joint_deg": list(prior.joints_deg), "prior_gripper_opening_m": prior.gripper_opening_m,
            "joint_change_deg": change, "gripper_target_change": grip_change, "gripper_transition": transition,
            "prior_gripper_command": prior_gripper_command, "predicted_gripper_command": predicted_gripper_command,
            "joint_target_change": joint_change,
            "change_category": ("both" if joint_change and grip_change else "joint_only" if joint_change
                                else "gripper_only" if grip_change else "neither"),
            "command": {"joints_deg": q, "gripper_opening_m": opening, "seconds": 1.0} if valid else None}


def _prior_command(prior, value):
    if value is not None and value not in ("open", "closed"):
        raise ValueError("prior gripper command must be open or closed when known")
    if value is not None:
        return value
    if prior.gripper_opening_m == 0.04:
        return "open"
    if prior.gripper_opening_m == 0.0:
        return "closed"
    return None


def decode_exclusive(action_logits, joint_output, prior_joints_deg, prior_opening_m, *, prior_gripper_command=None):
    """Preserve existing MOVE/OPEN/CLOSE/DONE hold-axis behavior."""
    prior = _prior(prior_joints_deg, prior_opening_m)
    previous_command = _prior_command(prior, prior_gripper_command)
    logits, angles = _values(action_logits, 4), _values(joint_output, 4)
    if not all(math.isfinite(x) for x in logits + angles):
        return {"status": "nonfinite_model_output", "done": None, "command": None, "exclusive_action": None}
    choice = max(range(4), key=logits.__getitem__)
    action = motion_policy.ACTIONS[choice]
    if choice == motion_policy.DONE:
        return {"status": "done", "done": True, "command": None, "exclusive_action": action}
    target = [x * 180.0 for x in angles] if choice == motion_policy.MOVE else list(prior.joints_deg)
    opening = prior.gripper_opening_m if choice == motion_policy.MOVE else (0.04 if choice == motion_policy.OPEN else 0.0)
    next_command = previous_command if choice == motion_policy.MOVE else ("open" if choice == motion_policy.OPEN else "closed")
    return _finish(target, opening, prior, {"exclusive_action": action}, previous_command, next_command)


def decode_simultaneous(joint_output, gripper_logits, stop_logits, prior_joints_deg, prior_opening_m, *, prior_gripper_command=None):
    """Apply both independent targets on CONTINUE and no target on DONE."""
    prior = _prior(prior_joints_deg, prior_opening_m)
    previous_command = _prior_command(prior, prior_gripper_command)
    angles, grip, stop = _values(joint_output, 4), _values(gripper_logits, 2), _values(stop_logits, 2)
    if not all(math.isfinite(x) for x in angles + grip + stop):
        return {"status": "nonfinite_model_output", "done": None, "command": None}
    done = max(range(2), key=stop.__getitem__) == 1
    if done:
        return {"status": "done", "done": True, "command": None}
    opening = 0.04 if max(range(2), key=grip.__getitem__) == 0 else 0.0
    return _finish([x * 180.0 for x in angles], opening, prior, {}, previous_command,
                   "open" if opening == 0.04 else "closed")


def read_current_control(sim):
    """Read the simulator's control targets, never the measured joint angles."""
    arm = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
    fingers = [float(sim.data.ctrl[i]) for i in sim._finger_actuators]
    if (len(fingers) != 2 or any(not math.isfinite(x) or not 0 <= x <= 0.025 for x in fingers)
            or abs(fingers[0] - fingers[1]) > 1e-9):
        raise ValueError("finger control targets are missing, out of range or asymmetric")
    return _prior(arm, sum(fingers))


def decode_model(model, condition, windows, masks, prior_joints, prior_opening, prior_gripper_command=None):
    if condition == "exclusive":
        action, joint = model(windows, masks)
        return decode_exclusive(action, joint, prior_joints, prior_opening, prior_gripper_command=prior_gripper_command), {"action_logits": action, "joint_output": joint}
    if condition == "simultaneous":
        joint, grip, stop = model(windows, masks)
        return decode_simultaneous(joint, grip, stop, prior_joints, prior_opening, prior_gripper_command=prior_gripper_command), {
            "joint_output": joint, "gripper_logits": grip, "stop_logits": stop}
    raise ValueError("unknown comparison condition")


@torch.no_grad()
def run_transfer(model, sim, registered_scene, task, condition, assessor, *, max_decisions=24,
                 device=torch.device("cpu"), cancel=None):
    """Execute a separately authorized student-only simulation using the fixed adapter."""
    before, task = scene_features.transfer_task(registered_scene, task)
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    raw = sim.observation()
    if task_catalog.scene(raw["registered_layout"]) != before:
        raise ValueError("registered scene differs from simulator state")
    model.eval()
    result = {"registered_scene": before, "task": task, "condition": condition, "decisions": [],
        "termination": None, "success": None, "physical_success": None, "assessment": None,
        "teacher_assistance_count": 0, "actual_hardware": False,
        "initial_state": sim.save_state(), "initial_observation": raw}
    previous = read_current_control(sim)
    if previous.gripper_opening_m < 0.025 or raw["gripper_opening_m"] < 0.025:
        raise ValueError("a new transfer must start with an open gripper")
    last_joints, opening, gripper = list(previous.joints_deg), previous.gripper_opening_m, "open"
    if cancel is not None and cancel.is_set():
        result.update(termination="cancelled", success=False, final_state=sim.save_state(), layout_status=raw["layout_status"])
        return assessor.safe(result)
    sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
    history = []
    for index in range(max_decisions):
        if cancel is not None and cancel.is_set():
            result.update(termination="cancelled", success=False)
            break
        item = {"index": index}
        result["decisions"].append(item)
        try:
            actual = read_current_control(sim)
            if (any(abs(a - b) > 1e-8 for a, b in zip(actual.joints_deg, last_joints))
                    or abs(actual.gripper_opening_m - opening) > 1e-8):
                raise ValueError("remembered and current actuator targets differ")
            obs = sim_data.sim_policy_observation(sim, gripper)
            vector = scene_features.encode_scene_features(obs, before, task)
            history.append(vector)
            window, mask = scene_features.history_window(history, 8)
            decoded, raw_output = decode_model(model, condition,
                torch.tensor([window], dtype=torch.float32, device=device),
                torch.tensor([mask], dtype=torch.bool, device=device), last_joints, opening, gripper)
            item.update(policy_observation={k: obs[k] for k in ("schema_version", "joints_deg", "sensor_valid", "moving",
                        "gripper_command", "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")},
                policy_features=vector, valid_history=mask, model_output=assessor.safe(raw_output),
                decoded=decoded, simulation_before=sim.observation())
            if decoded["status"] == "nonfinite_model_output":
                result.update(termination="nonfinite_model_output", success=False)
                item["result"] = "no_command"
                break
            if cancel is not None and cancel.is_set():
                result.update(termination="cancelled", success=False)
                item["result"] = "cancelled_before_command"
                break
            if decoded["done"]:
                result["termination"] = "model_done"
                item["result"] = "done_no_command"
                break
            if decoded["status"] == "out_of_range_target":
                result.update(termination="out_of_range_target", success=False)
                item["result"] = "no_command"
                break
            command = decoded["command"]
            sim.command(command["joints_deg"], command["gripper_opening_m"], seconds=command["seconds"])
            item["result"] = "command_applied"
            item["simulation_after"] = sim.observation()
            last_joints, opening = list(command["joints_deg"]), command["gripper_opening_m"]
            if condition != "exclusive" or decoded["exclusive_action"] != "move":
                gripper = "open" if opening == 0.04 else "closed"
        except Exception as exc:
            result.update(termination="evaluation_error", success=None,
                          error={"type": type(exc).__name__, "message": str(exc)})
            break
    else:
        result.update(termination="decision_limit", success=False)
    if result["termination"] == "model_done" and cancel is not None and cancel.is_set():
        result.update(termination="cancelled", success=False)
    if result["termination"] == "model_done":
        try:
            assessed = sim.evaluate_transfer(before, task["source_slot"], task["destination_slot"], stable_seconds=1.0)
            assessor.validate_assessment(assessed, before, task)
            result["assessment"] = assessed
            result["physical_success"] = assessed["success"]
            result["success"] = assessed["success"]
            if cancel is not None and cancel.is_set():
                result.update(termination="cancelled", success=False)
        except Exception as exc:
            result.update(termination="evaluation_error", success=None,
                          error={"type": type(exc).__name__, "message": str(exc)})
    if result["success"] is not True:
        sim.mark_layout_uncertain(result["termination"])
    result["layout_status"] = sim.observation()["layout_status"]
    result["final_state"] = sim.save_state()
    return assessor.safe(result)
