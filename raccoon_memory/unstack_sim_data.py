"""Offline, provenance-preserving augmentation for the 3-to-1 unstack task.

The teacher may use simulated object poses to adapt the recorded path. Those
poses and contact forces are privileged audit data, never student features.
The original demonstration is accepted only after its committed journal and
an additional continuous physical assessment have both been verified.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from raccoon_control.kinematics import Geometry
from raccoon_memory import sim_data
from raccoon_memory.motion_sim import HOME_JOINTS_DEG, MotionSim, SLOT_XY_M


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = ROOT / "artifacts" / "sim_demonstrations" / "20261030_assistant_unstack_01"
TASK = {"source_slot": 3, "destination_slot": 1, "source_level": 2, "destination_level": 1}
INITIAL_COUNTS = (0, 1, 2, 1)
SPLIT_RANGES = dict(sim_data.SPLIT_RANGES)
SOURCE = "programmatic_physics_augmentation"
TEACHER = "programmatic_feedback_teacher_from_assistant_unstack_demo"
ORIGINAL_SOURCE = "assistant_tool_action_original_simulation"
BASE_CHECKS = {"position_xy", "height", "upright", "linear_speed", "angular_speed", "gripper_open"}
UNSTACK_CHECKS = {
    "moved_cylinder_table_support", "support_cylinder_table_support",
    "support_cylinder_initial_xy", "support_cylinder_initial_height", "support_cylinder_upright",
    "support_cylinder_linear_speed", "support_cylinder_angular_speed",
    "moved_cylinder_free_of_robot_contact", "support_cylinder_free_of_robot_contact",
}


def make_randomized_simulation(seed: int) -> tuple[MotionSim, dict]:
    """Use the same sampled physics and student observations as the stack task."""
    sim, parameters = sim_data.make_randomized_simulation(seed)
    sim.reset(seed=parameters["reset_seed"], counts=INITIAL_COUNTS, xy_jitter_mm=1.5)
    for joint, actuator, angle in zip(sim._joint_ids, sim._arm_actuators, parameters["initial_joints_deg"]):
        radians = math.radians(angle)
        sim.data.qpos[sim.model.jnt_qposadr[joint]] = radians
        sim.data.ctrl[actuator] = radians
    mujoco.mj_forward(sim.model, sim.data)
    initial = sim.observation()
    parameters["slot_offsets_m"] = copy.deepcopy(sim._reset_info["slot_offsets_m"])
    parameters["initial_counts"] = list(INITIAL_COUNTS)
    parameters["initial_object_2_position_m"] = list(initial["objects"][1]["position_m"])
    parameters["initial_object_3_position_m"] = list(initial["objects"][2]["position_m"])
    parameters["task"] = dict(TASK)
    sim.metadata["randomized_parameters"] = copy.deepcopy(parameters)
    return sim, parameters


def _unstack_sample(raw: dict, support_reference: list[float]) -> tuple[dict, dict]:
    """Check contacts and original supporting-cylinder position at one boundary."""
    sim_data._require_finite(raw)
    sim_data._require_finite(support_reference, "support reference")
    if len(support_reference) != 3:
        raise ValueError("support reference must contain three coordinates")
    support, moved = raw["objects"][1], raw["objects"][2]
    if support.get("id") != 2 or moved.get("id") != 3:
        raise ValueError("unexpected cylinder mapping")
    displacement = np.asarray(support["position_m"], dtype=float) - np.asarray(support_reference, dtype=float)
    moved_offset = np.asarray(moved["position_m"], dtype=float) - np.array([*SLOT_XY_M[0], 0.01])
    contacts = raw["contacts"]

    def table_contacts(name):
        return [c for c in contacts if {c["geom1"], c["geom2"]} == {"table", name}
                and c["normal_force_n"] > 0]

    def robot_contacts(name):
        return [c for c in contacts if name in (c["geom1"], c["geom2"])
                and bool({c["geom1"], c["geom2"]} & sim_data.ROBOT_GEOMS)]

    moved_table = table_contacts("object_3_geom")
    support_table = table_contacts("object_2_geom")
    moved_robot = robot_contacts("object_3_geom")
    support_robot = robot_contacts("object_2_geom")
    quat = support["quaternion_wxyz"]
    upright_cos = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    checks = {
        "moved_cylinder_table_support": bool(moved_table),
        "support_cylinder_table_support": bool(support_table),
        "support_cylinder_initial_xy": bool(np.linalg.norm(displacement[:2]) <= 0.008),
        "support_cylinder_initial_height": bool(abs(displacement[2]) <= 0.003),
        "support_cylinder_upright": bool(upright_cos >= 0.9659),
        "support_cylinder_linear_speed": bool(np.linalg.norm(support["linear_velocity_m_s"]) <= 0.01),
        "support_cylinder_angular_speed": bool(np.linalg.norm(support["angular_velocity_rad_s"]) <= 0.3),
        "moved_cylinder_free_of_robot_contact": not moved_robot,
        "support_cylinder_free_of_robot_contact": not support_robot,
    }
    evidence = {"object_2": support, "object_3": moved, "object_3_table_contacts": moved_table,
                "object_2_table_contacts": support_table, "object_3_robot_contacts": moved_robot,
                "object_2_robot_contacts": support_robot,
                "support_displacement_xy_m": float(np.linalg.norm(displacement[:2])),
                "support_displacement_height_m": float(abs(displacement[2])),
                "destination_error_xy_m": float(np.linalg.norm(moved_offset[:2])),
                "destination_error_height_m": float(abs(moved_offset[2]))}
    return checks, evidence


def evaluate_unstack_success(sim, support_reference: list[float]) -> dict:
    """Require every condition at every physics boundary over one second.

    The base criterion checks cylinder 3's nominal destination, height, pose,
    speed and open gripper. Additional checks track support, cylinder 2's
    original position and robot clearance for both cylinders.
    """
    dt = float(sim.model.opt.timestep)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid physics time step")
    steps = max(1, math.ceil(1.0 / dt - 1e-12))
    start = float(sim.data.time)
    extra, evidence = _unstack_sample(sim.observation(), support_reference)
    failures = {key: 0.0 for key, passed in extra.items() if not passed}
    metric_names = ("support_displacement_xy_m", "support_displacement_height_m",
                    "destination_error_xy_m", "destination_error_height_m")
    maximum = {key: evidence[key] for key in metric_names}
    base, combined_base = None, None
    for _ in range(steps):
        current = sim.evaluate_success(object_id=3, destination_slot=1, destination_level=1, stable_seconds=dt)
        checks = current.get("checks")
        if (current.get("source") != "mujoco_simulation" or not isinstance(checks, dict)
                or not BASE_CHECKS.issubset(checks) or any(type(v) is not bool for v in checks.values())
                or current.get("success") is not all(checks.values())):
            raise ValueError("invalid base simulation criterion")
        if combined_base is None:
            base, combined_base = dict(current), dict(checks)
        else:
            combined_base = {key: combined_base[key] and value for key, value in checks.items()}
        sample, evidence = _unstack_sample(sim.observation(), support_reference)
        for key, value in sample.items():
            if not value and key not in failures:
                failures[key] = float(sim.data.time) - start
            extra[key] = extra[key] and value
        for key in metric_names:
            maximum[key] = max(maximum[key], evidence[key])
    elapsed = float(sim.data.time) - start
    if not math.isfinite(elapsed) or elapsed < 1.0 - 1e-9:
        raise ValueError("simulation did not advance for a full second")
    base.update(checks=combined_base, stable_interval_seconds=elapsed, success=all(combined_base.values()))
    return {"success": bool(base["success"] and all(extra.values())), "one_second_stability": base,
            "additional_unstack_checks": extra, "sample_count": steps + 1,
            "checks_scope": "Every physics-step boundary throughout the continuous interval.",
            "support_initial_position_m": list(support_reference),
            "maximum_errors_m": maximum, "first_failed_check_seconds": failures,
            "privileged_simulation_only": evidence}


def _action_plan(actions: list[dict]) -> tuple[int, int, int | None]:
    """Find gripper events without assuming a particular number of actions."""
    if not actions:
        raise ValueError("original demonstration has no actions")
    previous_joints = list(HOME_JOINTS_DEG)
    previous_opening = 0.04
    close_events, open_events, movements = [], [], []
    for index, event in enumerate(actions):
        action = event["action"]
        joints = action["joints_deg"]
        opening = float(action["gripper_opening_m"])
        seconds = float(action["seconds"])
        sim_data._require_finite(action, "original action")
        if len(joints) != 4 or not (0 <= opening <= 0.04) or seconds <= 0:
            raise ValueError("invalid original action")
        changing = not math.isclose(opening, previous_opening, abs_tol=1e-9)
        if changing:
            if not np.allclose(joints, previous_joints, atol=1e-6, rtol=0):
                raise ValueError("original changes arm and gripper targets in one action")
            if opening < sim_data.GRIPPER_OPEN_THRESHOLD_M <= previous_opening:
                close_events.append(index)
            if opening >= sim_data.GRIPPER_OPEN_THRESHOLD_M > previous_opening:
                open_events.append(index)
        else:
            movements.append(index)
        previous_joints, previous_opening = list(joints), opening
    if not close_events or not open_events or open_events[-1] <= close_events[0]:
        raise ValueError("original must close and subsequently release the gripper")
    last_close = max(i for i in close_events if i < open_events[-1])
    drop_moves = [i for i in movements if last_close < i < open_events[-1]]
    return close_events[0], open_events[-1], (drop_moves[-1] if drop_moves else None)


def load_reference_demo(directory: str | Path = DEFAULT_DEMO) -> tuple[list[dict], dict]:
    """Verify the committed original, then check an extra full stability interval."""
    from scripts.operate_motion_sim import _load

    directory = Path(directory).resolve()
    event_path, state_path = directory / "events.jsonl", directory / "state.json"
    before = (event_path.read_bytes(), state_path.read_bytes())
    manifest, events, sim = _load(directory)
    actions = [event for event in events if event.get("event") == "action"]
    if (not manifest.get("finished") or events[-1].get("event") != "finish"
            or len(events) != len(actions) + 2 or events[0].get("action", {}).get("counts") != list(INITIAL_COUNTS)):
        raise ValueError("original is unfinished or does not use the unstack reset")
    _action_plan(actions)
    initial = events[0]["observation_after"]["privileged_simulation_only"]["objects"]
    if (len(initial) != 4 or initial[1].get("id") != 2 or initial[2].get("id") != 3
            or np.linalg.norm(np.asarray(initial[1]["position_m"][:2]) - np.asarray(SLOT_XY_M[2])) > 0.002
            or np.linalg.norm(np.asarray(initial[2]["position_m"][:2]) - np.asarray(SLOT_XY_M[2])) > 0.002
            or abs(initial[1]["position_m"][2] - 0.0105) > 0.001
            or abs(initial[2]["position_m"][2] - 0.0305) > 0.001):
        raise ValueError("original reset has an unexpected cylinder mapping")
    criterion = events[-1].get("simulation_physical_criterion", {})
    checks = criterion.get("checks")
    if (criterion.get("source") != "mujoco_simulation" or criterion.get("success") is not True
            or criterion.get("object_id") != 3 or criterion.get("destination_slot") != 1
            or criterion.get("destination_level") != 1 or criterion.get("stable_interval_seconds", 0) < 1.0 - 1e-9
            or not isinstance(checks, dict) or not BASE_CHECKS.issubset(checks)
            or any(value is not True for value in checks.values())):
        raise ValueError("original lacks the required one-second destination criterion")
    assessment = evaluate_unstack_success(sim, initial[1]["position_m"])
    if not assessment["success"]:
        raise ValueError("original failed the additional one-second unstack criterion")
    if before != (event_path.read_bytes(), state_path.read_bytes()):
        raise ValueError("original demonstration changed during verification")
    source = {"demonstration_directory": str(directory), "session_id": manifest["session_id"],
              "events_sha256": hashlib.sha256(before[0]).hexdigest(),
              "state_sha256": hashlib.sha256(before[1]).hexdigest(),
              "head_event_sha256": manifest["head_event_sha256"], "base_model_xml_sha256": sim.xml_sha256,
              "simulator_code_sha256": manifest["simulator_code_sha256"], "assistant_original_episodes": 1,
              "original_recorded_actions": len(actions), "reset": copy.deepcopy(events[0]["action"]),
              "initial_object_2_position_m": list(initial[1]["position_m"]),
              "initial_object_3_position_m": list(initial[2]["position_m"]),
              "additional_physical_assessment_after_finished_session": assessment}
    return actions, source


def _target(index, event, plan, previous_target, sim, source, parameters):
    first_close, last_open, drop_move = plan
    original = event["action"]
    xyz = np.asarray(Geometry().forward(original["joints_deg"]), dtype=float)
    source_reference = np.asarray(source["initial_object_3_position_m"][:2]) * 1000
    source_actual = np.asarray(parameters["initial_object_3_position_m"][:2]) * 1000
    destination_reference = np.asarray(SLOT_XY_M[0]) * 1000
    destination_actual = destination_reference + np.asarray(parameters["slot_offsets_m"][0]) * 1000
    if index < first_close:
        offset = source_actual - source_reference
    elif index > last_open:
        offset = destination_actual - destination_reference
    else:
        direction = destination_reference - source_reference
        fraction = float(np.clip(np.dot(xyz[:2] - source_reference, direction) / np.dot(direction, direction), 0, 1))
        offset = (1 - fraction) * (source_actual - source_reference) + fraction * (destination_actual - destination_reference)
    xy = xyz[:2] + offset
    if index == drop_move:
        raw = sim.observation()
        # This is privileged programmatic teacher feedback, not a student input.
        xy = np.asarray(raw["tcp_mm"][:2]) + destination_actual - np.asarray(raw["objects"][2]["position_m"][:2]) * 1000
    return list(Geometry().inverse((float(xy[0]), float(xy[1]), float(xyz[2]))))


def _transition(episode_id, index, event, obs, next_obs, target, opening, previous_opening, teacher):
    changing = not math.isclose(opening, previous_opening, abs_tol=1e-9)
    state = "open" if opening >= sim_data.GRIPPER_OPEN_THRESHOLD_M else "closed"
    command = ({"kind": "set_gripper", "state": state} if changing else
               {"kind": "move_joints", "joints_deg": list(target), "speed_pct": sim_data.SPEED_LABEL_PCT})
    if changing and next_obs["gripper_reported_state"] != state:
        raise ValueError("simulated gripper did not report the requested binary state")
    prefix = f"{episode_id}-{index}"
    return {"observation": obs, "action": command, "next_observation": next_obs,
            "applied": True, "status": "gripper_reported" if changing else "completed",
            "provenance": {"command_id": f"{prefix}-command", "input_observation_id": f"{prefix}-before",
                           "result_observation_id": f"{prefix}-after", "decision_index": index,
                           "original_event_sha256": event["event_sha256"], "teacher": teacher}}


def generate_attempt(seed: int, split: str, actions: list[dict], source: dict) -> tuple[dict, dict | None]:
    episode_id = f"sim-unstack-aug-{split}-{seed}"
    attempt = {"episode_id": episode_id, "seed": seed, "split": split, "task": dict(TASK), "source": SOURCE,
               "teacher": TEACHER, "simulated": True, "actual_hardware": False,
               "reference": copy.deepcopy(source), "decisions": [], "success": False}
    transitions = []
    try:
        plan = _action_plan(actions)
        sim, parameters = make_randomized_simulation(seed)
        attempt["parameters"] = parameters
        previous_target = list(parameters["initial_joints_deg"])
        previous_opening, gripper_command = 0.04, "open"
        for index, event in enumerate(actions):
            original = event["action"]
            opening = float(original["gripper_opening_m"])
            changing = not math.isclose(opening, previous_opening, abs_tol=1e-9)
            target = list(previous_target) if changing else _target(index, event, plan, previous_target, sim, source, parameters)
            obs = sim_data.sim_policy_observation(sim, gripper_command)
            before = sim.observation()
            decision = {"input_observation": obs, "simulated_servo_target": {
                "joints_deg": list(target), "gripper_opening_m": opening, "seconds": float(original["seconds"])},
                "original_event_sha256": event["event_sha256"],
                "privileged_simulation_only": {"before": {"objects": before["objects"], "contacts": before["contacts"]}}}
            attempt["decisions"].append(decision)
            sim.command(target, opening, seconds=float(original["seconds"]))
            next_command = "open" if opening >= sim_data.GRIPPER_OPEN_THRESHOLD_M else "closed"
            next_obs = sim_data.sim_policy_observation(sim, next_command)
            after = sim.observation()
            sim_data._require_finite(after)
            transition = _transition(episode_id, index, event, obs, next_obs, target, opening, previous_opening, TEACHER)
            decision["transition"] = transition
            decision["privileged_simulation_only"]["after"] = {"objects": after["objects"], "contacts": after["contacts"]}
            transitions.append(transition)
            previous_target, previous_opening, gripper_command = list(target), opening, next_command
        assessment = evaluate_unstack_success(sim, parameters["initial_object_2_position_m"])
        attempt["physical_assessment"] = assessment
        attempt["success"] = assessment["success"]
        if not assessment["success"]:
            attempt["failure_reason"] = "physical_success_criterion_failed"
            return attempt, None
        quality = {"source": SOURCE, "teacher": TEACHER, "simulated": True, "actual_hardware": False,
                   "physical_success_in_simulation": True, "physical_assessment": copy.deepcopy(assessment),
                   "reference": copy.deepcopy(source), "randomized_parameters": copy.deepcopy(parameters),
                   "gripper_report": "Simulation-derived threshold, not a hardware sensor.",
                   "speed_label": "8 percent is a schema-compatibility label, not measured simulation speed.",
                   "clock": "Elapsed simulation seconds; recurrent history starts at reset."}
        return attempt, {"episode_id": episode_id, "task": dict(TASK), "quality": quality, "transitions": transitions}
    except Exception as exc:
        attempt.update(failure_reason="attempt_exception", exception_type=type(exc).__name__, exception_detail=str(exc))
        return attempt, None


def _assert_arm_match(raw, recorded, label):
    arm = recorded["arm"]
    for key in ("joints_deg", "tcp_mm", "gripper_opening_m", "time_seconds"):
        if not np.allclose(raw[key], arm[key], atol=1e-6, rtol=0):
            raise ValueError(f"original deterministic replay differs at {label}.{key}")


def original_episode(actions: list[dict], source: dict) -> dict:
    """Replay the exact assistant actions, with its provenance separate from augmentation."""
    _action_plan(actions)
    sim = MotionSim()
    reset = source["reset"]
    sim.reset(seed=reset["seed"], counts=reset["counts"], xy_jitter_mm=reset["xy_jitter_mm"])
    episode_id = f"sim-unstack-original-{source['session_id']}"
    transitions = []
    opening, gripper_command = 0.04, "open"
    for index, event in enumerate(actions):
        action = event["action"]
        _assert_arm_match(sim.observation(), event["observation_before"], f"action {index} before")
        obs = sim_data.sim_policy_observation(sim, gripper_command)
        sim.command(action["joints_deg"], action["gripper_opening_m"], seconds=action["seconds"])
        _assert_arm_match(sim.observation(), event["observation_after"], f"action {index} after")
        next_opening = float(action["gripper_opening_m"])
        next_command = "open" if next_opening >= sim_data.GRIPPER_OPEN_THRESHOLD_M else "closed"
        next_obs = sim_data.sim_policy_observation(sim, next_command)
        transitions.append(_transition(episode_id, index, event, obs, next_obs, action["joints_deg"],
                                       next_opening, opening, "assistant_tool_action"))
        opening, gripper_command = next_opening, next_command
    assessment = evaluate_unstack_success(sim, source["initial_object_2_position_m"])
    if not assessment["success"]:
        raise ValueError("exact original replay failed the full unstack criterion")
    return {"episode_id": episode_id, "task": dict(TASK), "transitions": transitions,
            "quality": {"source": ORIGINAL_SOURCE, "teacher": "assistant_tool_action", "simulated": True,
                        "actual_hardware": False, "physical_success_in_simulation": True,
                        "physical_assessment": assessment, "reference": copy.deepcopy(source),
                        "observation_reconstruction": "Deterministic replay of the original commands; arm observations verified against journal.",
                        "speed_label": "8 percent is a schema-compatibility label, not measured simulation speed."}}


parse_seeds = sim_data.parse_seeds


def generate_dataset(output: str | Path, seeds: list[int], split: str, *, demo_dir: str | Path = DEFAULT_DEMO,
                     include_original: bool = False) -> dict:
    """Persist every attempt and positive episodes; keep the optional original separate."""
    requested = Path(output)
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"output path already exists: {requested}")
    if split not in SPLIT_RANGES:
        raise ValueError(f"invalid split: {split}")
    low, high = SPLIT_RANGES[split]
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or not low <= seed <= high for seed in seeds)):
        raise ValueError(f"{split} seeds must be unique integers in [{low}, {high}]")
    if include_original and split != "train":
        raise ValueError("the assistant original may only be exported with the training split")
    actions, source = load_reference_demo(demo_dir)
    original = original_episode(actions, source) if include_original else None
    code_paths = [Path(__file__).resolve(), Path(sim_data.__file__).resolve(),
                  ROOT / "scripts" / "generate_unstack_sim_data.py"]
    target = requested.resolve()
    target.mkdir(parents=False, exist_ok=False)
    failures = Counter()
    success_count = 0
    with (target / "attempts.jsonl").open("x", encoding="utf-8", newline="\n") as attempts, \
         (target / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as episodes:
        for seed in seeds:
            attempt, episode = generate_attempt(seed, split, actions, source)
            attempts.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            attempts.flush()
            if episode is not None:
                episodes.write(json.dumps(episode, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                episodes.flush()
                success_count += 1
            else:
                failures[attempt["failure_reason"]] += 1
    names = ["attempts.jsonl", "episodes.jsonl"]
    if original is not None:
        name = "original_episode.jsonl"
        with (target / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(original, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        names.append(name)
    artifacts = {name: {"sha256": sim_data._sha256(target / name), "bytes": (target / name).stat().st_size}
                 for name in names}
    report = {"schema": "raccoon-unstack-sim-augmentation-report.v1", "task": dict(TASK), "source": SOURCE,
              "teacher": TEACHER, "simulated": True, "actual_hardware": False, "split": split, "seeds": seeds,
              "fixed_splits_inclusive": SPLIT_RANGES, "attempt_count": len(seeds), "success_count": success_count,
              "failure_count": len(seeds) - success_count, "failure_reasons": dict(failures),
              "original_episode_exported_separately": original is not None, "reference": source,
              "generator_code": [{"path": str(path), "sha256": sim_data._sha256(path)} for path in code_paths],
              "artifacts": artifacts,
              "note": "Physics augmentations of one assistant original, not new assistant decisions; privileged feedback is excluded from student inputs."}
    with (target / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
