"""Simulation-only v2 session with explicitly unconfirmed expected scenes.

The ordinary SceneModelSession advances only after physical assessment.  This
separate development session advances on student DONE and a fixed hold instead.
It never feeds simulator object/contact state or assessment into the student.
Saved physical snapshots are available only after the current request releases
the session lock, for offline scoring.  This is not a hardware controller.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
import threading

import mujoco
import torch

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_demo_data, scene_features, scene_policy, scene_recall, task_catalog, task_program
from .scene_model_session import SPLIT_PLAN, DuplicateRequest, SceneModelSession, UnsupportedStep
from .scene_custom_slots_sim import CustomSlotsSceneMotionSim, registered_slot_xy_mm
from .scene_motion_sim import SceneMotionSim
from .scene_program_coordinator import Busy, SceneProgramCoordinator


def _robot_observation(sim, gripper):
    """Read robot state only; no object pose, contact, or outcome is used."""
    joints = [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids]
    velocities = [float(sim.data.qvel[sim.model.jnt_dofadr[j]]) for j in sim._joint_ids]
    tcp = (sim.data.site_xpos[sim._tcp_site] * 1000).tolist()
    opening = float(sum(sim.data.qpos[sim.model.jnt_qposadr[j]] for j in sim._finger_ids))
    now = float(sim.data.time)
    if not all(math.isfinite(x) for x in [*joints, *velocities, *tcp, opening, now]):
        raise ValueError("robot observation is not finite")
    return {"schema_version": "PolicyObservation.phase1", "source": "mujoco_simulation_derived",
            "simulated": True, "observed_monotonic": now, "joints_deg": joints, "tcp_mm": tcp,
            "sensor_valid": True, "moving": max(abs(math.degrees(x)) for x in velocities) > 0.1,
            "gripper_command": gripper, "gripper_reported_state": "open" if opening >= 0.03 else "closed",
            "gripper_reported_valid": True, "gripper_feedback_fresh": True, "gripper_opening_m": opening}


def _in_range(target):
    return len(target) == len(JOINT_LIMITS) and all(math.isfinite(value) and low <= value <= high
        for value, (low, high) in zip(target, JOINT_LIMITS))


def _servo_targets(sim):
    joints = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
    fingers = [float(sim.data.ctrl[i]) for i in sim._finger_actuators]
    if not _in_range(joints):
        raise ValueError("previous applied target is out of range")
    if (len(fingers) != 2 or any(not math.isfinite(x) or not 0 <= x <= 0.025 for x in fingers)
            or abs(fingers[0] - fingers[1]) > 1e-9):
        raise ValueError("finger servo targets must be finite, symmetric and within range")
    return {"joints_deg": joints, "gripper_opening_m": sum(fingers)}


def _require_servo_targets(sim, expected):
    actual = _servo_targets(sim)
    if (any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0)
            for a, b in zip(actual["joints_deg"], expected["joints_deg"])) or
            not math.isclose(actual["gripper_opening_m"], expected["gripper_opening_m"],
                             abs_tol=1e-9, rel_tol=0)):
        raise ValueError("servo targets changed without a confirmed command return")


def _hold_one_second(sim, cancel, *, before_step=None):
    dt = float(sim.model.opt.timestep)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid physics timestep")
    steps = max(1, math.ceil(1.0 / dt - 1e-12))
    start = float(sim.data.time)
    for index in range(steps):
        if cancel.is_set():
            raise InterruptedError("request cancelled during hold")
        if before_step is not None:
            before_step(index, steps)
        mujoco.mj_forward(sim.model, sim.data)
        mujoco.mj_step(sim.model, sim.data)
    mujoco.mj_forward(sim.model, sim.data)
    if not math.isclose(float(sim.data.time) - start, 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("the fixed hold did not last one second")


def _registered_slots_for_sim(sim, requested=None):
    """Bind the student's nominal coordinates to the simulated registration."""
    registered = (sim.slot_xy_mm if isinstance(sim, CustomSlotsSceneMotionSim)
                  else scene_features.NOMINAL_SLOT_XY_MM)
    registered = registered_slot_xy_mm(registered)
    if requested is not None and registered_slot_xy_mm(requested) != registered:
        raise ValueError("student slot coordinates differ from the registered simulation")
    return registered


def run_unscored_transfer(model, sim, before, task, *, max_decisions, device, cancel,
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
        features = scene_features.encode_scene_features(_robot_observation(sim, gripper), before, task, slots)
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


class OracleFreeSceneModelSession(SceneModelSession):
    """Run the existing v2 coordinator with expected and confirmed state separate.

    ``current`` is the expected scene for the inherited planner.  Registration
    is explicit; after an attempted motion the last registered scene remains
    historical and the current physical scene is unconfirmed.  This class is
    restricted to the existing virtual simulator and never confirms a move from
    the student's DONE signal.
    """

    def __init__(self, sim, memory, student, *, device="cpu", max_decisions=24, cancel=None,
                 checkpoint_train_groups=None, development_groups=None, slot_xy_mm=None,
                 unscored_transfer_runner=None):
        if not isinstance(sim, SceneMotionSim):
            raise TypeError("oracle-free expected execution is simulation-only")
        self._registered_slot_xy_mm = _registered_slots_for_sim(sim, slot_xy_mm)
        super().__init__(sim, memory, student, device=device, max_decisions=max_decisions, cancel=cancel,
                         checkpoint_train_groups=checkpoint_train_groups)
        if unscored_transfer_runner is not None and not callable(unscored_transfer_runner):
            raise TypeError("unscored transfer runner must be callable")
        self._unscored_transfer_runner = unscored_transfer_runner
        self.checkpoint_train_groups = self.train_groups
        if development_groups is not None:
            split, _, _ = scene_demo_data.read_split_plan(SPLIT_PLAN)
            groups = frozenset(development_groups)
            allowed = set(split["train_groups"]) | set(split["validation_groups"])
            if not groups or not groups.issubset(allowed) or groups.intersection(split["test_groups"]):
                raise ValueError("development groups must exclude the reserved final split")
            self.train_groups = groups
        self._last_confirmed_scene = None
        self._current_confirmed_scene = None
        self._physical_checkpoints = []
        self._applied_targets = None
        self._offline_anchor_rebase_required = False
        self._unresolved_transfer = None
        self._recovery_scene_snapshot = None
        self._policy_state_suppressed = False
        self._policy_state_epoch = 0
        self._scene_view_lock = threading.RLock()

    def _registered(self, expected):
        # Simulator bookkeeping is deliberately an expectation after a move.
        # Do not inspect object state or turn it into a confirmation.
        if self.sim.layout_status != "registered" or task_catalog.scene(self.sim._committed_layout) != expected:
            raise RuntimeError("expected simulator bookkeeping differs from the session")
        return {"expected_scene": expected, "physical_confirmation": "not_inspected"}

    def _register(self, value):
        # A failed registration hook can leave a partially changed policy.
        with self._scene_view_lock:
            self._policy_state_suppressed = True
            self._policy_state_epoch += 1
        initial_targets = _servo_targets(self.sim)
        super()._register(value)
        _require_servo_targets(self.sim, initial_targets)
        self._applied_targets = initial_targets
        self._last_confirmed_scene = self.current
        self._current_confirmed_scene = self.current
        self.events[-1].update(scene_source="explicit_registration", physical_assessment_performed=False,
                               nominal_slot_xy_mm=[list(xy) for xy in self._registered_slot_xy_mm],
                               control_target_source="validated_explicit_registration")
        callback = getattr(self.student, "on_scene_registered", None)
        if callable(callback):
            callback(self.current, self._registered_slot_xy_mm)
        _require_servo_targets(self.sim, initial_targets)
        if self.cancel.is_set():
            raise InterruptedError("registration cancelled before completion")
        with self._scene_view_lock:
            self._unresolved_transfer = None
            self._policy_state_suppressed = False

    def _halt(self, reason):
        with self._scene_view_lock:
            self._current_confirmed_scene = None
            self._policy_state_suppressed = True
            self._policy_state_epoch += 1
            super()._halt(reason)

    @staticmethod
    def _compose_scene_state(current, current_registered, last_registered, unresolved, status):
        # A declaration is useful for symbolic planning but is not a physical
        # observation.  Never expose it through either confirmed-scene field.
        unresolved = copy.deepcopy(unresolved)
        if unresolved is not None:
            unresolved["status"] = "uncertain" if status == "uncertain" else "in_progress"
        return {"expected_scene": copy.deepcopy(current),
                "current_registered_scene": copy.deepcopy(current_registered),
                "confirmed_current_scene": None,
                "last_confirmed_scene": None,
                "confirmation": ("explicit_registration" if current_registered is not None
                                 else "unconfirmed_after_motion" if last_registered is not None
                                 else "not_registered"),
                "last_registered_scene": copy.deepcopy(last_registered),
                "registration_source": "user_declared" if last_registered is not None else None,
                "physical_confirmation": "not_inspected", "unresolved_transfer": unresolved,
                "physical_success": None, "mode": "simulation_expected_scene"}

    def _symbolic_scene_state(self):
        return self._compose_scene_state(self.current, self._current_confirmed_scene, self._last_confirmed_scene,
                                         self._unresolved_transfer, self.status)

    def scene_state(self):
        # Freeze symbolic output while recovery prepares its new registration.
        with self._scene_view_lock:
            epoch = self._policy_state_epoch
            recovery_snapshot = self._recovery_scene_snapshot
            result = copy.deepcopy(recovery_snapshot) if recovery_snapshot is not None else self._symbolic_scene_state()
            suppressed = recovery_snapshot is not None or self._policy_state_suppressed or self.cancel.is_set()
        callback = getattr(self.student, "conditional_pose_state", None)
        def hide_pose():
            if callable(callback):
                result["conditional_object_pose_beliefs"] = {
                    "available": False, "confirmed": False,
                    "reason": "state_unavailable_after_interruption_or_during_registration"}
            return result
        if suppressed:
            return hide_pose()
        try:
            pose_state = copy.deepcopy(callback()) if callable(callback) else None
        except BaseException:
            with self._scene_view_lock:
                changed = (epoch != self._policy_state_epoch or self._policy_state_suppressed
                           or self._recovery_scene_snapshot is not None or self.cancel.is_set())
            if changed:
                return hide_pose()
            raise
        with self._scene_view_lock:
            changed = (epoch != self._policy_state_epoch or self._policy_state_suppressed
                       or self._recovery_scene_snapshot is not None or self.cancel.is_set())
        if changed:
            return hide_pose()
        if pose_state is not None:
            result["conditional_object_pose_beliefs"] = pose_state
        return result

    def offline_checkpoints(self, *, allow_unrebased=False):
        """Copy privileged snapshots only after a request is no longer running."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("physical checkpoints are unavailable during an active request")
        try:
            if (not allow_unrebased and any(item.get("requires_offline_anchor_rebase")
                                            for item in self._physical_checkpoints)):
                raise RuntimeError("post-recovery checkpoints require an offline physical-anchor rebase")
            return copy.deepcopy(self._physical_checkpoints)
        finally:
            self._lock.release()

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
            runner = self._unscored_transfer_runner or run_unscored_transfer
            result = runner(self.student, self.sim, before, task, max_decisions=self.max_decisions,
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
        hold_observer = getattr(self, "_hold_observation_hook", None)
        if hold_observer is None:
            _hold_one_second(self.sim, self.cancel)
        else:
            _hold_one_second(self.sim, self.cancel, before_step=hold_observer)
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


class OracleFreeSceneProgramCoordinator(SceneProgramCoordinator):
    """Add explicit simulation-only recovery to the unchanged v2 coordinator."""

    def __init__(self, session):
        if not isinstance(session, OracleFreeSceneModelSession):
            raise TypeError("recovery requires an oracle-free simulation session")
        super().__init__(session)

    def _refresh_external_state(self):
        # The recovery itself changes memory; it is not an external mutation.
        if self.session._recovery_scene_snapshot is None:
            super()._refresh_external_state()

    def recover_registration(self, request_id, registration_scene):
        """Register a user-supplied scene after the failed request has stopped.

        This changes symbolic bookkeeping only; it does not reset or move the
        simulated robot or objects and does not assess physical success.

        Recovery-time stop must go through this coordinator's stop API or its
        validated v2 stop path. A directly shared Event can merge a new set
        with an earlier set before the old cancellation is cleared. The legacy
        desktop Session.request_stop and its button/Esc are not wired here.
        """
        if not task_program._valid_id(request_id):
            raise ValueError("invalid recovery request ID")
        with self._lock:
            self._refresh_external_state()
            if self._active is not None or not self.session._lock.acquire(blocking=False):
                raise Busy("recovery requires an idle session")
            try:
                with self.session._request_lock:
                    if request_id in self.session._requests:
                        raise DuplicateRequest("request ID was already used")
                if self.session.status != "uncertain":
                    raise RuntimeError("recovery requires an uncertain session")
                sim = self.session.sim
                registered = scene_recall.scene(registration_scene, sim.active_ids)
                initial_targets = _servo_targets(sim)
                _robot_observation(sim, "open" if initial_targets["gripper_opening_m"] >= 0.025 else "closed")
                with self.session._request_lock:
                    if request_id in self.session._requests:
                        raise DuplicateRequest("request ID was already used")
                    self.session._requests.add(request_id)
                was_cancelled = self.session.cancel.is_set()
                self._generation += 1
                generation = self._generation
                recovery_run = {
                    "kind": "recovery_registration", "request_id": request_id,
                    "program_id": self._new_id("program"), "program_revision": self._new_id("revision"),
                    "generation": generation, "round": 1, "step_index": None, "task_id": None,
                    "task_revision": None, "task_sequence": 0, "start_scene": self.session.current,
                    "task_start_scene": None, "repeat_active": False, "stop_repeat": None,
                    "stop_requested": False, "stop_requests": [], "last_completed": None,
                    "stop_boundary_missed": False, "atomic_inflight": False, "transfer_id": None,
                    "transfer_sequence": 0, "transfer_phase": "between", "last_committed_transfer_id": None,
                    "last_committed_transfer_sequence": 0, "stop_transfer_id": None}
                with self.session._scene_view_lock:
                    self.session._recovery_scene_snapshot = self.session._symbolic_scene_state()
                    self.session._policy_state_suppressed = True
                    self.session._policy_state_epoch += 1
                self._active = recovery_run
                if was_cancelled:
                    self.session.cancel.clear()
                self._bump()
            except BaseException:
                self.session._lock.release()
                raise

        # Keep the session exclusively owned, but let immediate stop acquire
        # the coordinator lock while memory and student callbacks are running.
        def check_cancelled():
            if self.session.cancel.is_set():
                raise InterruptedError("recovery cancelled before registration")

        session_lock_held = True
        try:
            with self.session._memory_binding_lock:
                memory = self.session.memory
                guard = getattr(memory, "synchronized", None)
                with guard() if callable(guard) else nullcontext():
                    check_cancelled()
                    memory.clear()
                    check_cancelled()
                    memory.observe(registered, write=False)
                    check_cancelled()
                    prepared_memory_stamp = self._memory_stamp()
            _require_servo_targets(sim, initial_targets)
            check_cancelled()
            # A failed transfer can leave recurrent or task state pending.
            callback = getattr(self.session.student, "on_scene_registered", None)
            if callable(callback):
                callback(registered, self.session._registered_slot_xy_mm)
            check_cancelled()
            _require_servo_targets(sim, initial_targets)
            pose_callback = getattr(self.session.student, "conditional_pose_state", None)
            pose_state = pose_callback() if callable(pose_callback) else None
            pose_state = copy.deepcopy(pose_state)
            check_cancelled()
            result_state = self.session._compose_scene_state(registered, registered, registered, None, "registered")
            if pose_state is not None:
                result_state["conditional_object_pose_beliefs"] = pose_state
            result = {"request_id": request_id, "status": "registered", "scene_state": result_state,
                      "physical_success": None, "physical_assessment_performed": False,
                      "control_target_source": "validated_explicit_registration"}

            # Follow the same memory-before-coordinator lock order as execution.
            # A changed memory version cannot be accepted as this registration.
            with self.session._memory_binding_lock:
                guard = getattr(memory, "synchronized", None)
                with guard() if callable(guard) else nullcontext():
                    with self._lock:
                        if (self._generation != generation or self.session.cancel.is_set()
                                or self._active is not recovery_run):
                            raise InterruptedError("recovery cancelled before registration")
                        if self._memory_stamp() != prepared_memory_stamp:
                            raise RuntimeError("memory changed during recovery registration")
                        _require_servo_targets(sim, initial_targets)
                        registration_revision = self._new_id("registration")

                        # User registration, not a physical assessment or simulator reset.
                        with self.session._scene_view_lock:
                            sim._committed_layout = registered
                            sim._pending_transfer = None
                            sim.layout_status = "registered"
                            sim.layout_uncertainty_reason = None
                            self.session.current = registered
                            self.session.status = "registered"
                            self.session._last_confirmed_scene = registered
                            self.session._current_confirmed_scene = registered
                            self.session._applied_targets = initial_targets
                            self.session._offline_anchor_rebase_required = True
                            self.session._unresolved_transfer = None
                            self.session.events.append({"event": "registered", "scene": registered, "memory_cleared": True,
                                "scene_source": "explicit_recovery_registration", "request_id": request_id,
                                "control_target_source": "validated_explicit_registration", "physical_assessment_performed": False,
                                "requires_offline_anchor_rebase": True})
                            # Publish only after all symbolic fields are coherent.
                            self.session._recovery_scene_snapshot = None
                            self.session._policy_state_suppressed = False
                        self._generation += 1
                        self._stop_unconfirmed = False
                        self._active = None
                        self._previous_task = self._previous_program = None
                        self._registered_initial = registered
                        self._registration_revision = registration_revision
                        self._memory_revision = None
                        self.events.append({"event": "recovery_registered", "request_id": request_id,
                                            "registration_revision": registration_revision})
                        self._known_session_stamp = ("registered", registered, len(self.session.events),
                                                     prepared_memory_stamp)
                        self._expected_memory_stamp = prepared_memory_stamp
                        self._bump()
                        self.session._lock.release()
                        session_lock_held = False
                        return result
        except BaseException as exc:
            if not session_lock_held:
                raise
            with self._lock:
                if was_cancelled:
                    self.session.cancel.set()
                self.session.events.append({"event": "recovery_registration_failed", "request_id": request_id,
                                            "error_type": type(exc).__name__})
                with self.session._scene_view_lock:
                    self.session._recovery_scene_snapshot = None
                    # The student hook may already have changed its internal state.
                    self.session._policy_state_suppressed = True
                if self._active is recovery_run:
                    self._active = None
                self._memory_revision = None
                self._known_session_stamp = self._session_stamp()
                self._expected_memory_stamp = self._memory_stamp()
                self._bump()
                self.session._lock.release()
            raise
