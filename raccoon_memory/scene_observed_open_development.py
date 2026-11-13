"""Isolated simulation development path: OPEN holds the public observed joints.

Copied control flow is pinned to scene_oracle_free_session.py's 2026-11-04
current-runtime fingerprint. Only OPEN target selection and OPEN cancellation
confirmation differ; model, simulator, guard and offline scoring stay separate.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math

import torch
from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, scene_policy, task_catalog
from raccoon_memory.scene_model_session import UnsupportedStep
from raccoon_memory.scene_oracle_free_session import (
    OracleFreeSceneModelSession, _robot_observation, _in_range, _registered_slots_for_sim,
    _require_servo_targets, _hold_one_second,
)


def observed_open_target(observation):
    """Use the same decision observation, accepting only valid public joints."""
    if not isinstance(observation, dict) or any(observation.get(k) is not True for k in (
        "sensor_valid", "gripper_reported_valid", "gripper_feedback_fresh"
    )):
        raise ValueError("OPEN requires a valid fresh public robot observation")
    joints = observation.get("joints_deg")
    if (not isinstance(joints, (list, tuple)) or len(joints) != len(JOINT_LIMITS)
            or any(type(value) not in (int, float) for value in joints)):
        raise ValueError("OPEN requires four numeric public joint angles")
    result = [float(value) for value in joints]
    if not _in_range(result):
        raise ValueError("OPEN observed joint angles must be finite and in range")
    return result


def run_unscored_transfer_observed_open(model, sim, before, task, *, max_decisions, device, cancel,
                          applied_targets, decisions=None, slot_xy_mm=None):
    """Run one transfer without calling a physical-success evaluator."""
    if decisions is None:
        decisions = []
    before, task = scene_features.transfer_task(before, task)
    slots = _registered_slots_for_sim(sim, slot_xy_mm)
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("positive decision limit required")
    if task_catalog.scene(sim._committed_layout) != before or sim.layout_status != "registered":
        raise ValueError("simulator expected scene bookkeeping differs")
    model.eval()
    _require_servo_targets(sim, applied_targets)
    opening = applied_targets["gripper_opening_m"]
    if _robot_observation(sim, "open")["gripper_opening_m"] < 0.025 or opening < 0.025:
        raise ValueError("a new transfer must start with an open gripper")
    last_target = list(applied_targets["joints_deg"])
    if cancel.is_set():
        raise InterruptedError("request cancelled before transfer")
    transfer_started = getattr(model, "on_transfer_started", None)
    if callable(transfer_started):
        transfer_started(before, task, copy.deepcopy(applied_targets))
    sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
    history, gripper = [], "open"
    for index in range(max_decisions):
        if cancel.is_set():
            raise InterruptedError("request cancelled before decision")
        _require_servo_targets(sim, applied_targets)
        observation = _robot_observation(sim, gripper)
        features = scene_features.encode_scene_features(observation, before, task, slots)
        history.append(features)
        window, mask = scene_features.history_window(history, model.history_length)
        windows = torch.tensor([window], dtype=torch.float32, device=device)
        masks = torch.tensor([mask], dtype=torch.bool, device=device)
        # A future unified policy may use this measured controller state as an
        # additional input. The existing BC1020 keeps its unchanged interface.
        contextual = getattr(model, "forward_with_control_state", None)
        guard_events = getattr(model, "events", None)
        guard_count = len(guard_events) if isinstance(guard_events, list) else None
        with torch.inference_mode():
            outputs = (contextual(windows, masks, torch.tensor([last_target], dtype=torch.float32, device=device))
                       if callable(contextual) else model(windows, masks))
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError("student must return action logits and joint targets")
        logits, angles = outputs
        if (not isinstance(logits, torch.Tensor) or not isinstance(angles, torch.Tensor) or
                tuple(logits.shape) != (1, 4) or tuple(angles.shape) != (1, 4) or
                not bool(torch.isfinite(logits).all() and torch.isfinite(angles).all())):
            raise ValueError("student output shape or values are invalid")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        row = {"index": index, "policy_features": features, "valid_history": mask,
               "action_logits": logits.detach().cpu().tolist(), "absolute_joint_output": angles.detach().cpu().tolist(),
               "previous_applied_target_deg": list(last_target), "action": action}
        guard_event = None
        if (guard_count is not None and guard_events is getattr(model, "events", None) and
                len(guard_events) == guard_count + 1 and isinstance(guard_events[-1], dict) and
                {"original_proposal_joints_deg", "guard_returned_target_deg", "actual_applied_joints_deg"}
                <= guard_events[-1].keys()):
            guard_event = guard_events[-1]
            row["guard"] = copy.deepcopy(guard_event)
        decisions.append(row)
        if cancel.is_set():
            row["result"] = "no_command"
            raise InterruptedError("request cancelled before command")
        try:
            _require_servo_targets(sim, applied_targets)
        except BaseException:
            row["result"] = "no_command"
            raise
        if action == "done":
            row["result"] = "done_no_command"
            return {"termination": "model_done", "decisions": decisions, "physical_success": None,
                    "teacher_assistance_count": 0, "actual_hardware": False}
        if action == "move":
            target = [float(value) * 180.0 for value in angles[0].tolist()]
        else:
            if action == "open":
                try:
                    target = observed_open_target(observation)
                except (TypeError, ValueError):
                    row["result"] = "no_command"
                    raise
            else:
                target = list(last_target)
            gripper = "open" if action == "open" else "closed"
            opening = 0.04 if gripper == "open" else 0.0
        if not _in_range(target):
            row["result"] = "no_command"
            return {"termination": "out_of_range_target", "decisions": decisions, "physical_success": None,
                    "teacher_assistance_count": 0, "actual_hardware": False}
        row["attempted_command"] = {"joints_deg": target, "gripper_opening_m": opening, "seconds": 1.0,
                                    "source": "shared_student_policy"}
        row["result"] = "command_unconfirmed"
        started = float(sim.data.time)
        returned = sim.command(target, opening, seconds=1.0)
        if action == "open" and cancel.is_set():
            raise InterruptedError("request cancelled before OPEN command confirmation")
        if (returned.get("source") != "mujoco_simulation" or returned.get("simulated") is not True or
                not math.isclose(float(returned["time_seconds"]) - started, 1.0, abs_tol=1e-6, rel_tol=0)):
            raise ValueError("simulated command did not return normally")
        applied = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
        fingers = [float(sim.data.ctrl[i]) for i in sim._finger_actuators]
        if (len(applied) != len(target) or any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0)
                                              for a, b in zip(applied, target)) or
                len(fingers) != 2 or any(not math.isclose(value, opening / 2, abs_tol=1e-9, rel_tol=0)
                                         for value in fingers)):
            raise ValueError("returned servo targets differ from the applied command")
        row["command"] = row.pop("attempted_command")
        row["result"] = "command_applied"
        if guard_event is not None:
            guard_event["actual_applied_joints_deg"] = list(applied)
            row["guard"]["actual_applied_joints_deg"] = list(applied)
        last_target = target
        applied_targets.update(joints_deg=list(target), gripper_opening_m=opening)
        control_applied = getattr(model, "on_control_applied", None)
        if callable(control_applied):
            control_applied(action, list(target), list(applied), opening)
    return {"termination": "decision_limit", "decisions": decisions, "physical_success": None,
            "teacher_assistance_count": 0, "actual_hardware": False}


class ObservedOpenDevelopmentSession(OracleFreeSceneModelSession):
    """The original v2 session with the explicit isolated transfer routine."""

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        self._ready()
        before, task = scene_features.transfer_task(self.current, action)
        after = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
        group = scene_policy.split_group_for(before, task)
        if group not in self.train_groups:
            raise UnsupportedStep(f"transfer is outside allowed groups: {group}")
        event = {"event": "transfer", "before": before, "planned_after": after, "task": task,
                 "split_group": group, "result": None, "committed": False,
                 "commit_kind": "expected_only", "physical_success": None}
        self.events.append(event)
        before_state = self.sim.save_state()
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError("request invalidated before transfer")
            # Even an interrupted motion invalidates a previous confirmation.
            with self._scene_view_lock:
                self._current_confirmed_scene = None
                self._unresolved_transfer = {"expected_object_id": task["object_id"],
                                             "object_id_source": "expected_scene_source_top",
                                             "physical_object_id": None,
                                             "source_slot": task["source_slot"],
                                             "destination_slot": task["destination_slot"]}
        decisions = []
        try:
            result = run_unscored_transfer_observed_open(self.student, self.sim, before, task, max_decisions=self.max_decisions,
                                           device=self.device, cancel=self.cancel, applied_targets=self._applied_targets,
                                           decisions=decisions, slot_xy_mm=self._registered_slot_xy_mm)
        except BaseException as exc:
            event["result"] = {"termination": "interrupted" if isinstance(exc, InterruptedError) else "error",
                               "decisions": decisions, "physical_success": None,
                               "teacher_assistance_count": 0, "actual_hardware": False,
                               "error_type": type(exc).__name__}
            raise
        event["result"] = result
        if result["termination"] != "model_done":
            raise RuntimeError("student transfer did not return DONE: " + result["termination"])
        done_state = self.sim.save_state()
        _hold_one_second(self.sim, self.cancel)
        _require_servo_targets(self.sim, self._applied_targets)
        if self.cancel.is_set():
            raise InterruptedError("request cancelled after hold")
        hold_state = self.sim.save_state()
        self.memory.observe(after, write=False)
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError("request invalidated before expected-scene update")
            _require_servo_targets(self.sim, self._applied_targets)
            # Only symbolic simulator bookkeeping changes; object state is untouched.
            with self._scene_view_lock:
                self.sim._committed_layout = task_catalog.scene(after)
                self.sim._pending_transfer = None
                self.sim.layout_status = "registered"
                self.sim.layout_uncertainty_reason = None
                self.current = after
                event["committed"] = True
            checkpoint = {"before": before, "task": task, "before_state": before_state,
                          "done_state": done_state, "after_hold_state": hold_state}
            if self._offline_anchor_rebase_required:
                checkpoint["requires_offline_anchor_rebase"] = True
            self._physical_checkpoints.append(checkpoint)
            callback = getattr(self.student, "on_expected_transfer_committed", None)
            if callable(callback):
                callback(before, task, after)
            # This is an expected-scene commit, not a physical confirmation.
            # Publish the transfer boundary while the scene and event commit
            # remain protected by the coordinator's lock.
            if on_transfer_commit is not None:
                on_transfer_commit()
            with self._scene_view_lock:
                self._unresolved_transfer = None

