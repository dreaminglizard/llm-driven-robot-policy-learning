"""Continuous, simulation-only direct velocity demonstrations.

Each command is supplied by the demonstrator.  A completed transfer updates
only the *expected* layout; it does not inspect or move objects.  Physical
checkpoints are private until the session ends and are for later scoring.
"""
from __future__ import annotations

import copy
import hashlib
import math
import uuid

import mujoco

from . import scene_features, task_catalog
from .direct_velocity_demo import DirectVelocityDemo, DirectVelocityExecutionError
from .scene_motion_sim import SceneMotionSim


def _scene_list(scene):
    return [list(stack) for stack in task_catalog.scene(scene)]


class ContinuousDirectVelocity:
    """Keep one simulator and memory across explicitly chosen transfers.

    The memory object must provide ``observe(scene, write=...)`` and
    ``recall()``. The production caller supplies the existing GRU runtime.
    Expected layouts are symbolic and never represent physical confirmation.
    """

    def __init__(self, sim, registered_scene, slot_xy_mm, memory, *, max_chunk_steps=100,
                 max_chunk_seconds=60.0, max_joint_speed_deg_s=30.0):
        if not isinstance(sim, SceneMotionSim) or sim.metadata.get("simulated") is not True:
            raise ValueError("a simulated scene is required")
        initial = task_catalog.scene(registered_scene)
        if sim.layout_status != "registered" or sim._committed_layout != initial:
            raise ValueError("the simulator must start at the registered scene")
        initial_velocity = [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]])) for j in sim._joint_ids]
        if any(not math.isfinite(v) or abs(v) > 1e-9 for v in initial_velocity):
            raise ValueError("the first public moving flag requires a verified stationary start")
        if (not callable(getattr(memory, "observe", None)) or
                not callable(getattr(memory, "recall", None))):
            raise ValueError("a scene memory runtime is required")
        if getattr(memory, "has_memory", False):
            raise ValueError("a new program requires a cleared memory runtime")
        if not isinstance(slot_xy_mm, (tuple, list)) or len(slot_xy_mm) != 4:
            raise ValueError("four registered slot coordinates are required")
        slots = []
        for pair in slot_xy_mm:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)):
                raise ValueError("slot coordinates must be finite pairs")
            slots.append([float(v) for v in pair])
        if hasattr(sim, "slot_xy_mm") and slots != [list(pair) for pair in sim.slot_xy_mm]:
            raise ValueError("registered coordinates differ from the simulated slots")
        self._sim, self._memory = sim, memory
        self.initial, self.expected = initial, initial
        self.slots = slots
        self.phase, self.recalled = "make", None
        self._written = None
        self._id, self._revision = uuid.uuid4().hex, 0
        self._demo, self._active, self._stopped = None, None, False
        self._completed = False
        self._stages, self._boundaries = [], []
        self._options = {"max_chunk_steps": max_chunk_steps, "max_chunk_seconds": max_chunk_seconds,
                         "max_joint_speed_deg_s": max_joint_speed_deg_s}
        self._memory.observe(initial, write=False)

    def _memory_evidence(self):
        hidden = getattr(self._memory, "hidden", None)
        digest = None
        if hidden is not None:
            digest = hashlib.sha256(hidden.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return {"state_version": getattr(self._memory, "state_version", None),
                "hidden_sha256": digest, "has_memory": getattr(self._memory, "has_memory", None)}

    def _require_between(self, observation_id):
        if self._stopped or self._active is not None:
            raise RuntimeError("program controls are available only between active transfers")
        if observation_id != f"{self._id}:program:{self._revision}":
            raise ValueError("the program observation is stale")

    def observation(self):
        if self._active is not None:
            robot = self._demo.observation()
            oid = robot["observation_id"]
        elif self._demo is not None:
            robot = self._demo.observation()
            oid = f"{self._id}:program:{self._revision}"
        else:
            sim = self._sim
            joints = [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids]
            qvel = [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]])) for j in sim._joint_ids]
            opening = sum(float(sim.data.qpos[sim.model.jnt_qposadr[j]]) for j in sim._finger_ids)
            robot = {"simulated": True, "time_seconds": float(sim.data.time), "joints_deg": joints,
                     "applied_target_deg": [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators],
                     "gripper_command": "open" if sum(float(sim.data.ctrl[i]) for i in sim._finger_actuators) >= .025 else "closed",
                     "sensor_valid": True, "moving": max(abs(v) for v in qvel) > .1,
                     "gripper_reported_state": "open" if opening >= .03 else "closed",
                     "gripper_reported_valid": True, "gripper_feedback_fresh": True}
            oid = f"{self._id}:program:{self._revision}"
        # Select robot-only keys: neither snapshots nor simulator object truth
        # are exposed through the teacher's interactive interface.
        keys = ("time_seconds", "joints_deg", "sensor_valid", "moving", "applied_target_deg",
                "gripper_command", "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")
        result = {key: copy.deepcopy(robot[key]) for key in keys}
        result.update(observation_id=oid, phase=self.phase, expected_scene=_scene_list(self.expected),
                      registered_slot_xy_mm=copy.deepcopy(self.slots),
                      current_task=copy.deepcopy(self._active["task"]) if self._active else None,
                      recalled_goal=_scene_list(self.recalled) if self.recalled is not None else None,
                      physical_confirmation="not_inspected", simulated=True, completed=self._completed)
        return result

    def start_transfer(self, observation_id, source_slot, destination_slot):
        self._require_between(observation_id)
        before, task = scene_features.transfer_task(
            self.expected, {"source_slot": source_slot, "destination_slot": destination_slot})
        after = task_catalog.transfer(before, source_slot, destination_slot)
        snapshot = self._sim.save_state()
        if self._stages and snapshot["state"] != self._stages[-1]["after_commit_state"]["state"]:
            self._sim.mark_layout_uncertain("physical state changed between direct velocity transfers")
            raise RuntimeError("physical state changed between transfers")
        if self._demo is None:
            self._demo = DirectVelocityDemo(self._sim, before, task, self.slots, **self._options)
        else:
            if not self._demo.finished or self._demo.stopped:
                raise RuntimeError("previous velocity transfer has not finished")
            self._sim.begin_transfer(before, source_slot, destination_slot)
            self._demo.scene, self._demo.task = before, task
            self._demo.finished = False
            self._demo.revision += 1
        self._active = {"phase": self.phase, "before": before, "after": after, "task": task,
                        "before_state": snapshot, "starting_sample_count": self._demo.sample_count}
        self._revision += 1
        return self.observation()

    def submit(self, observation_id, commands, *, on_progress=None):
        if self._stopped or self._active is None:
            raise RuntimeError("no direct velocity transfer is active")
        def annotate(event, data):
            if event == "sample":
                # The underlying counter excludes passive inter-transfer holds.
                # Preserve both clocks rather than pretending the hold had
                # teacher-generated labels.
                data["program_time_seconds"] = data["next_observation"]["time_seconds"] - data["dt_s"]
                data["program_transfer_index"] = len(self._stages)
                data["program_phase"] = self.phase
            if on_progress is not None:
                on_progress(event, data)
        try:
            return self._demo.submit(observation_id, commands, on_progress=annotate)
        except DirectVelocityExecutionError:
            self._stopped = True
            raise

    def _hold_one_second(self):
        """Passively retain the last actuator targets; do not query object truth."""
        sim = self._sim
        dt = float(sim.model.opt.timestep)
        steps, tail = round(1.0 / dt), round(.02 / dt)
        if (steps < 1 or tail < 1 or not math.isclose(steps * dt, 1., abs_tol=1e-9) or
                not math.isclose(tail * dt, .02, abs_tol=1e-9)):
            raise ValueError("physics timestep cannot represent the one-second hold")
        before_tail = None
        for index in range(steps):
            if index == steps - tail:
                before_tail = [float(sim.data.qpos[sim.model.jnt_qposadr[j]]) for j in sim._joint_ids]
            mujoco.mj_forward(sim.model, sim.data)
            mujoco.mj_step(sim.model, sim.data)
        mujoco.mj_forward(sim.model, sim.data)
        after_tail = [float(sim.data.qpos[sim.model.jnt_qposadr[j]]) for j in sim._joint_ids]
        self._demo.moving = max(abs(math.degrees(a-b)) / .02 for a, b in zip(after_tail, before_tail)) > .1

    def finish_transfer(self, observation_id):
        if self._stopped or self._active is None:
            raise RuntimeError("no direct velocity transfer is active")
        if self._demo.sample_count <= self._active["starting_sample_count"]:
            raise ValueError("a transfer cannot finish without an explicit velocity command")
        terminal = self._demo.finish(observation_id)
        done = self._sim.save_state()
        stage = self._active
        after = stage["after"]
        try:
            self._hold_one_second()
            held = self._sim.save_state()
            self._memory.observe(after, write=False)
            # Match the existing oracle-free session's symbolic commit.  No
            # physical state or object anchor is overwritten here.
            self._sim._committed_layout = task_catalog.scene(after)
            self._sim._pending_transfer = None
            self._sim.layout_status = "registered"
            self._sim.layout_uncertainty_reason = None
            committed = self._sim.save_state()
            if held["state"] != committed["state"]:
                raise RuntimeError("symbolic commit changed the physical integration state")
        except BaseException:
            self._sim.mark_layout_uncertain("continuous velocity symbolic commit failed")
            self._stopped = True
            raise
        self.expected = after
        self._stages.append({**stage, "terminal": terminal, "done_state": done,
                             "after_hold_state": held, "after_commit_state": committed,
                             "passive_hold_seconds": 1.0, "teacher_labels_during_hold": 0,
                             "physical_confirmation": "not_inspected"})
        self._active = None
        self._revision += 1
        return {"terminal": terminal, "expected_after": _scene_list(after),
                "passive_hold_seconds": 1.0, "teacher_labels_during_hold": 0,
                "physical_confirmation": "not_inspected", "observation": self.observation()}

    def remember(self, observation_id):
        self._require_between(observation_id)
        if self.phase != "make" or not any(s["phase"] == "make" for s in self._stages):
            raise ValueError("a make transfer is required before remembering")
        before = self._memory_evidence()
        self._memory.observe(self.expected, write=True)
        self._written, self.phase = self.expected, "return"
        self._boundaries.append({"event": "remember", "expected_scene": _scene_list(self.expected),
                                 "memory_before": before, "memory_after": self._memory_evidence()})
        self._revision += 1
        return self.observation()

    def recall(self, observation_id):
        self._require_between(observation_id)
        if (self.phase != "return" or self.expected != self.initial or
                not any(s["phase"] == "return" for s in self._stages)):
            raise ValueError("return to the initial expected scene before recall")
        before = self._memory_evidence()
        # Only the memory model's decoded output is exposed as the goal.
        self.recalled = task_catalog.scene(self._memory.recall())
        self.phase = "restore"
        self._boundaries.append({"event": "recall", "decoded_goal": _scene_list(self.recalled),
                                 "memory_before": before, "memory_after": self._memory_evidence()})
        self._revision += 1
        return self.observation()

    def finish_program(self, observation_id):
        self._require_between(observation_id)
        if (self.phase != "restore" or self.expected != self.recalled or
                not any(s["phase"] == "restore" for s in self._stages)):
            raise ValueError("restore the decoded goal before finishing")
        self._stopped = self._completed = True
        self._revision += 1
        return {"completed": True, "expected_final_scene": _scene_list(self.expected),
                "memory_recall_matches_written": self.recalled == self._written,
                "transfers": len(self._stages), "physical_success": None,
                "physical_confirmation": "not_inspected", "simulated": True}

    def stop(self):
        if self._stopped:
            raise RuntimeError("the continuous session has already stopped")
        if self._active is not None and not self._demo.stopped:
            self._demo.stop(self._demo.observation()["observation_id"], "continuous_session_stopped")
        self._stopped = True
        return {"completed": False, "transfers": len(self._stages), "physical_success": None}

    def post_session_evidence(self):
        """Return private checkpoints only after the live session has ended."""
        if not self._stopped:
            raise RuntimeError("physical checkpoints are private until the session ends")
        return {"completed": self._completed, "stages": copy.deepcopy(self._stages),
                "incomplete_stage": copy.deepcopy(self._active),
                "memory_boundaries": copy.deepcopy(self._boundaries),
                "physical_success": None, "physical_confirmation": "not_inspected"}
