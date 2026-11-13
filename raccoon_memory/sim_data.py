"""Replay one assistant simulation demonstration with randomized physics.

Every generated row comes from a fresh MuJoCo run. Object poses and contacts
are privileged teacher/evaluation state, not real sensors or policy features.
No robot, SDK, service, network, microphone, or LLM is called here.
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
from raccoon_memory.motion_sim import HOME_JOINTS_DEG, MotionSim


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO = ROOT / "artifacts" / "sim_demonstrations" / "20261030_assistant_stack_01"
SPLIT_RANGES = {"pilot": (900, 902), "train": (1000, 1099), "validation": (1500, 1519), "test": (2000, 2019)}
TASK = {"source_slot": 1, "destination_slot": 3, "source_level": 1, "destination_level": 2}
TEACHER = "programmatic_feedback_teacher_from_assistant_demo"
SOURCE = "programmatic_physics_augmentation"
GRIPPER_OPEN_THRESHOLD_M = 0.03
MOVING_THRESHOLD_DEG_S = 0.1
SPEED_LABEL_PCT = 8.0
ROBOT_GEOMS = {"base_link", "upper_link", "forearm_link", "tool_link", "finger_left_pad", "finger_right_pad"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_finite(value, label="simulation state"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{label}[{index}]")


def _id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"missing model element: {name}")
    return result


def make_randomized_simulation(seed: int) -> tuple[MotionSim, dict]:
    """Return a fresh simulator and its exact sampled, unvalidated parameters.

    No free joint is written after reset. The arm's initial position, controls,
    body mass/inertia, friction and servo parameters are set before stepping.
    Identical seeds and identical code/model versions yield identical conditions.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    streams = np.random.SeedSequence(seed).spawn(5)
    mass_rng, friction_rng, servo_rng, joint_rng = (np.random.default_rng(stream) for stream in streams[:4])
    reset_seed = int(streams[4].generate_state(1, dtype=np.uint64)[0])
    sim = MotionSim()
    objects, friction, servos = [], [], []
    for index, body in enumerate(sim._object_bodies, 1):
        factor = float(mass_rng.uniform(0.8, 1.2))
        sim.model.body_mass[body] *= factor
        sim.model.body_inertia[body] *= factor
        objects.append({"object_id": index, "mass_factor": factor, "mass_kg": float(sim.model.body_mass[body]),
                        "inertia_kg_m2": sim.model.body_inertia[body].tolist()})
    for name in [f"object_{i}_geom" for i in range(1, 5)] + ["finger_left_pad", "finger_right_pad"]:
        geom = _id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        factor = float(friction_rng.uniform(0.8, 1.2))
        sim.model.geom_friction[geom] *= factor
        friction.append({"geom": name, "factor": factor, "friction": sim.model.geom_friction[geom].tolist()})
    for actuator in sim._arm_actuators + sim._finger_actuators:
        factor = float(servo_rng.uniform(0.9, 1.1))
        sim.model.actuator_gainprm[actuator, 0] *= factor
        sim.model.actuator_biasprm[actuator, 1] *= factor
        servos.append({"name": mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                       "factor": factor, "kp": float(sim.model.actuator_gainprm[actuator, 0]),
                       "kv": float(-sim.model.actuator_biasprm[actuator, 2])})
    # Refresh model constants after inertia changes before setting initial state.
    mujoco.mj_setConst(sim.model, sim.data)
    sim.reset(seed=reset_seed, counts=(1, 1, 1, 1), xy_jitter_mm=1.5)
    offsets = joint_rng.uniform(-3.0, 3.0, size=4)
    initial_angles = [float(base + delta) for base, delta in zip(HOME_JOINTS_DEG, offsets)]
    for joint, actuator, angle in zip(sim._joint_ids, sim._arm_actuators, initial_angles):
        radians = math.radians(angle)
        sim.data.qpos[sim.model.jnt_qposadr[joint]] = radians
        sim.data.ctrl[actuator] = radians
    mujoco.mj_forward(sim.model, sim.data)
    parameters = {
        "seed": seed, "reset_seed": reset_seed, "initial_joint_offsets_deg": offsets.tolist(),
        "initial_joints_deg": initial_angles, "slot_offsets_m": copy.deepcopy(sim._reset_info["slot_offsets_m"]),
        "objects": objects, "friction": friction, "servos": servos,
        "ranges": {"slot_xy_jitter_mm": [-1.5, 1.5], "initial_joint_offset_deg": [-3, 3],
                   "object_mass_and_inertia_factor": [0.8, 1.2], "geom_friction_factor": [0.8, 1.2],
                   "servo_stiffness_factor": [0.9, 1.1]},
        "interpretation": "Unvalidated simulation assumptions, not measured hardware variation.",
        "base_model_xml_sha256": sim.xml_sha256,
    }
    sim.metadata["assumed_parameters_basis"] = "Baseline before the separately recorded randomized parameters"
    sim.metadata["randomized_parameters"] = copy.deepcopy(parameters)
    return sim, parameters


def sim_policy_observation(sim: MotionSim, gripper_command: str | None) -> dict:
    """Convert current simulated arm state to the existing policy observation.

    The binary gripper report is derived from simulated opening, not a measured
    device report. Object state, contact, seed and phase are never included.
    """
    if gripper_command not in (None, "open", "closed"):
        raise ValueError("gripper_command must be None, open or closed")
    raw = sim.observation()
    velocity = [float(sim.data.qvel[sim.model.jnt_dofadr[j]]) for j in sim._joint_ids]
    _require_finite(raw)
    _require_finite(velocity, "joint velocity")
    return {
        "schema_version": "PolicyObservation.phase1", "source": "mujoco_simulation_derived", "simulated": True,
        "observed_monotonic": float(raw["time_seconds"]), "joints_deg": list(raw["joints_deg"]),
        "tcp_mm": list(raw["tcp_mm"]), "sensor_valid": True,
        "moving": max(abs(math.degrees(x)) for x in velocity) > MOVING_THRESHOLD_DEG_S,
        "gripper_command": gripper_command,
        "gripper_reported_state": "open" if raw["gripper_opening_m"] >= GRIPPER_OPEN_THRESHOLD_M else "closed",
        "gripper_reported_valid": True, "gripper_feedback_fresh": True,
        "gripper_opening_m": float(raw["gripper_opening_m"]),
    }


def load_reference_demo(directory: str | Path = DEFAULT_DEMO) -> tuple[list[dict], dict]:
    """Load only the committed, physically successful original 12-action demo."""
    # Reuse the original operator's hash-chain and state verifier. It is an
    # offline module: it never connects to the physical controller or SDK.
    from scripts.operate_motion_sim import _load

    directory = Path(directory).resolve()
    event_path, state_path = directory / "events.jsonl", directory / "state.json"
    before = (event_path.read_bytes(), state_path.read_bytes())
    manifest, events, sim = _load(directory)
    if before != (event_path.read_bytes(), state_path.read_bytes()):
        raise ValueError("original demonstration changed during verification")
    actions = [e for e in events if e.get("event") == "action"]
    if len(events) != 14 or len(actions) != 12 or not manifest.get("finished") or events[-1].get("event") != "finish":
        raise ValueError("original demonstration is not a completed 12-action session")
    criterion = events[-1].get("simulation_physical_criterion", {})
    if (criterion.get("success") is not True or criterion.get("object_id") != 1
            or criterion.get("destination_slot") != 3 or criterion.get("destination_level") != 2
            or criterion.get("stable_interval_seconds", 0) < 1.0 - 1e-9
            or not isinstance(criterion.get("checks"), dict) or not all(v is True for v in criterion["checks"].values())):
        raise ValueError("original simulation demonstration lacks the required physical success")
    source = {"demonstration_directory": str(directory),
              "events_sha256": hashlib.sha256(before[0]).hexdigest(),
              "state_sha256": hashlib.sha256(before[1]).hexdigest(),
              "head_event_sha256": manifest["head_event_sha256"],
              "base_model_xml_sha256": sim.xml_sha256,
              "simulator_code_sha256": manifest["simulator_code_sha256"],
              "assistant_original_episodes": 1,
              "original_recorded_actions": 12}
    return actions, source


def _xy_mm(raw, object_id):
    position = raw["objects"][object_id - 1]["position_m"]
    return np.asarray(position[:2], dtype=float) * 1000


def _action_target(index, original_action, previous_target, sim, source_xy, destination_xy):
    """Recreate 12 waypoints, then compensate the carried object's offset."""
    geometry = Geometry()
    original = original_action["joints_deg"]
    z = float(geometry.forward(original)[2])
    if index in (3, 10):
        # The original gripper commands held the previous joint target.
        return list(previous_target)
    if index < 6:
        return list(geometry.inverse((float(source_xy[0]), float(source_xy[1]), z)))
    if index == 6:
        destination = geometry.inverse((float(destination_xy[0]), float(destination_xy[1]), z))
        return [(float(previous_target[0]) + destination[0]) / 2, *previous_target[1:]]
    if index == 7:
        return list(geometry.inverse((float(destination_xy[0]), float(destination_xy[1]), z)))
    if index in (8, 9):
        raw = sim.observation()
        correction = _xy_mm(raw, 3) - _xy_mm(raw, 1)
        xy = np.asarray(raw["tcp_mm"][:2]) + correction
        return list(geometry.inverse((float(xy[0]), float(xy[1]), z)))
    if index == 11:
        xy = geometry.forward(previous_target)[:2]
        return list(geometry.inverse((float(xy[0]), float(xy[1]), z)))
    raise ValueError(f"unknown demonstration action {index}")


def _stack_sample(raw):
    """Evaluate support and absence of robot contact at one simulated instant."""
    _require_finite(raw)
    obj, support = raw["objects"][0], raw["objects"][2]
    delta = np.asarray(obj["position_m"]) - np.asarray(support["position_m"])
    support_contacts = [c for c in raw["contacts"] if {c["geom1"], c["geom2"]} == {"object_1_geom", "object_3_geom"}
                        and c["normal_force_n"] > 0]
    robot_contacts = {name: [c for c in raw["contacts"]
                            if name in (c["geom1"], c["geom2"]) and bool({c["geom1"], c["geom2"]} & ROBOT_GEOMS)]
                      for name in ("object_1_geom", "object_3_geom")}
    support_quat = support["quaternion_wxyz"]
    upright_cos = 1 - 2 * (support_quat[1] ** 2 + support_quat[2] ** 2)
    checks = {"object_to_support_xy": bool(np.linalg.norm(delta[:2]) <= 0.008),
              "object_to_support_height": bool(abs(delta[2] - 0.02) <= 0.003),
              "supporting_cylinder_upright": bool(upright_cos >= 0.9659),
              "supporting_cylinder_linear_speed": bool(np.linalg.norm(support["linear_velocity_m_s"]) <= 0.01),
              "supporting_cylinder_angular_speed": bool(np.linalg.norm(support["angular_velocity_rad_s"]) <= 0.3),
              "actual_object_support_contact": bool(support_contacts),
              "stacked_cylinder_free_of_robot_contact": not robot_contacts["object_1_geom"],
              "supporting_cylinder_free_of_robot_contact": not robot_contacts["object_3_geom"]}
    evidence = {"object_1": obj, "object_3": support, "support_contacts": support_contacts,
                "robot_contacts": robot_contacts}
    return checks, evidence


def _physical_assessment(sim):
    """Check all conditions at every physics step over one continuous second."""
    dt = float(sim.model.opt.timestep)
    steps = max(1, math.ceil(1.0 / dt - 1e-12))
    start = float(sim.data.time)
    stack_checks, evidence = _stack_sample(sim.observation())
    first_failed = {key: 0.0 for key, passed in stack_checks.items() if not passed}
    criterion = None
    stability_checks = None
    for _ in range(steps):
        current = sim.evaluate_success(object_id=1, destination_slot=3, destination_level=2, stable_seconds=dt)
        if stability_checks is None:
            stability_checks = dict(current["checks"])
            criterion = dict(current)
        else:
            for key, value in current["checks"].items():
                stability_checks[key] = stability_checks[key] and value
        sample_checks, evidence = _stack_sample(sim.observation())
        for key, value in sample_checks.items():
            if not value and key not in first_failed:
                first_failed[key] = float(sim.data.time) - start
            stack_checks[key] = stack_checks[key] and value
    elapsed = float(sim.data.time) - start
    criterion["checks"] = stability_checks
    criterion["stable_interval_seconds"] = elapsed
    criterion["success"] = bool(all(stability_checks.values()) and elapsed >= 1.0 - 1e-9)
    return {"success": bool(criterion["success"] and all(stack_checks.values())),
            "one_second_stability": criterion, "additional_stack_checks": stack_checks,
            "stack_checks_scope": "Every physics-step boundary throughout the stability interval.",
            "sample_count": steps + 1, "first_failed_check_seconds": first_failed,
            "privileged_simulation_only": evidence}


def evaluate_stack_success(sim):
    """Apply the shared one-second stack, support, and robot-clearance criterion."""
    return _physical_assessment(sim)


def generate_attempt(seed: int, split: str, actions: list[dict], source: dict) -> tuple[dict, dict | None]:
    """Simulate one complete attempt; return its audit row and positive episode if successful."""
    episode_id = f"sim-aug-{split}-{seed}"
    attempt = {"episode_id": episode_id, "seed": seed, "split": split, "source": SOURCE,
               "teacher": TEACHER, "simulated": True, "actual_hardware": False,
               "reference": copy.deepcopy(source), "decisions": [], "success": False}
    transitions = []
    try:
        if len(actions) != 12:
            raise ValueError("teacher requires exactly 12 original actions")
        sim, parameters = make_randomized_simulation(seed)
        attempt["parameters"] = parameters
        initial = sim.observation()
        source_xy, destination_xy = _xy_mm(initial, 1), _xy_mm(initial, 3)
        previous_target = list(parameters["initial_joints_deg"])
        previous_opening = 0.04
        gripper_command = "open"  # The reset sets an open servo target.
        for index, event in enumerate(actions):
            original = event["action"]
            desired_opening = float(original["gripper_opening_m"])
            target = _action_target(index, original, previous_target, sim, source_xy, destination_xy)
            obs = sim_policy_observation(sim, gripper_command)
            privileged_before = sim.observation()
            next_command = "open" if desired_opening >= GRIPPER_OPEN_THRESHOLD_M else "closed"
            is_gripper = desired_opening != previous_opening
            if is_gripper and target != previous_target:
                raise ValueError("cannot represent simultaneous arm and gripper target changes in policy labels")
            command = {"kind": "set_gripper", "state": next_command} if is_gripper else {
                "kind": "move_joints", "joints_deg": list(target), "speed_pct": SPEED_LABEL_PCT}
            _require_finite(privileged_before)
            decision = {"input_observation": obs, "action": command, "simulated_servo_target": {
                "joints_deg": list(target), "gripper_opening_m": desired_opening, "seconds": float(original["seconds"])},
                "privileged_simulation_only": {
                    "before": {"objects": privileged_before["objects"], "contacts": privileged_before["contacts"]}}}
            attempt["decisions"].append(decision)
            sim.command(target, desired_opening, seconds=float(original["seconds"]))
            next_obs = sim_policy_observation(sim, next_command)
            privileged_after = sim.observation()
            _require_finite(privileged_after)
            prefix = f"{episode_id}-{index}"
            provenance = {"command_id": f"{prefix}-command", "input_observation_id": f"{prefix}-before",
                          "result_observation_id": f"{prefix}-after", "decision_index": index,
                          "original_event_sha256": event["event_sha256"], "teacher": TEACHER}
            transition = {"observation": obs, "action": command, "next_observation": next_obs,
                          "applied": True, "status": "gripper_reported" if is_gripper else "completed",
                          "provenance": provenance}
            transitions.append(transition)
            decision["transition"] = transition
            decision["privileged_simulation_only"]["after"] = {
                "objects": privileged_after["objects"], "contacts": privileged_after["contacts"]}
            if is_gripper and next_obs["gripper_reported_state"] != next_command:
                raise ValueError("simulated gripper report did not reach the commanded binary state")
            previous_target, previous_opening, gripper_command = list(target), desired_opening, next_command
        assessment = evaluate_stack_success(sim)
        attempt["physical_assessment"] = assessment
        attempt["success"] = assessment["success"]
        if not assessment["success"]:
            attempt["failure_reason"] = "physical_success_criterion_failed"
            return attempt, None
        quality = {
            "source": SOURCE, "teacher": TEACHER, "simulated": True, "actual_hardware": False,
            "physical_success_in_simulation": True, "physical_assessment": copy.deepcopy(assessment),
            "reference": copy.deepcopy(source), "randomized_parameters": copy.deepcopy(parameters),
            "gripper_report": f"Simulation-derived opening threshold {GRIPPER_OPEN_THRESHOLD_M} m; not a hardware sensor.",
            "moving_report": f"Simulation-derived joint speed threshold {MOVING_THRESHOLD_DEG_S} degrees/s.",
            "clock": "Elapsed simulation seconds, not wall clock or hardware monotonic time.",
            "speed_label": "8 percent is a schema-compatibility label, not a measured simulation speed.",
            "episode_history": "Reset for each newly initialized simulation.",
        }
        return attempt, {"episode_id": episode_id, "task": dict(TASK), "quality": quality, "transitions": transitions}
    except Exception as exc:
        attempt["failure_reason"] = "attempt_exception"
        attempt["exception_type"] = type(exc).__name__
        attempt["exception_detail"] = str(exc)
        return attempt, None


def parse_seeds(value: str) -> list[int]:
    """Parse inclusive range START:END or comma-separated nonnegative seeds."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("seeds must not be empty")
    seeds = []
    for component in value.split(","):
        component = component.strip().replace("..", ":")
        bounds = component.split(":")
        if len(bounds) == 1:
            seeds.append(int(bounds[0]))
        elif len(bounds) == 2:
            start, end = map(int, bounds)
            if start > end:
                raise ValueError("seed range must be increasing")
            seeds.extend(range(start, end + 1))
        else:
            raise ValueError("invalid seed range")
    if not seeds or any(s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique nonnegative integers")
    return seeds


def generate_dataset(output: str | Path, seeds: list[int], split: str, *, demo_dir: str | Path = DEFAULT_DEMO) -> dict:
    """Save all attempts and only successful episodes into a new directory."""
    requested = Path(output)
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"output path already exists: {requested}")
    if split not in SPLIT_RANGES:
        raise ValueError(f"invalid split: {split}")
    low, high = SPLIT_RANGES[split]
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or not low <= seed <= high for seed in seeds):
        raise ValueError(f"{split} seeds must be unique integers in [{low}, {high}]")
    actions, source = load_reference_demo(demo_dir)
    code_paths = [Path(__file__).resolve(), ROOT / "scripts" / "generate_motion_sim_data.py"]
    code_hashes = [{"path": str(path), "sha256": _sha256(path)} for path in code_paths]
    target = requested.resolve()
    target.mkdir(parents=False, exist_ok=False)
    attempts, episodes = [], []
    for seed in seeds:
        attempt, episode = generate_attempt(seed, split, actions, source)
        attempts.append(attempt)
        if episode is not None:
            episodes.append(episode)
    for name, items in (("attempts.jsonl", attempts), ("episodes.jsonl", episodes)):
        with (target / name).open("xb") as stream:
            for item in items:
                stream.write((json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))
    artifacts = {}
    for name in ("attempts.jsonl", "episodes.jsonl"):
        path = target / name
        artifacts[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    failure_counts = Counter(a["failure_reason"] for a in attempts if not a["success"])
    report = {"schema": "raccoon-sim-augmentation-report.v1", "source": SOURCE, "teacher": TEACHER,
              "simulated": True, "actual_hardware": False, "split": split, "seeds": seeds,
              "attempt_count": len(attempts), "success_count": len(episodes), "failure_count": len(attempts) - len(episodes),
              "failure_reasons": dict(failure_counts), "reference": source, "generator_code": code_hashes,
              "artifacts": artifacts,
              "note": "All attempts are physics simulations derived from one assistant demonstration; no hardware or new LLM decisions."}
    with (target / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
