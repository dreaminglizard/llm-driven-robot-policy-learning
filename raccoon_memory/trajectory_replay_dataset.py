"""Turn stopped direct demonstrations into clearly labelled simulator replays.

The original decisions are coarse position/gripper commands.  This module
replays them in MuJoCo and supervises *next replayed state*, not a recorded
operator speed command.  Simulator qvel and physical object state never enter
the student feature vector.  No hardware or external service is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import mujoco

from . import scene_features
from .scene_custom_slots_sim import CustomSlotsSceneMotionSim

SCHEMA = "raccoon-direct-simulation-replay-trajectories.v1"
SAMPLE_DT_S = 0.02


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _near(left, right, tolerance=1e-7):
    return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(type(a) in (int, float) and type(b) in (int, float)
                    and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance
                    for a, b in zip(left, right, strict=True)))


def source_split(source_group_id):
    """Stable split for the whole original start, independent of row count."""
    _need(isinstance(source_group_id, str) and source_group_id.strip(), "missing original source group")
    bucket = int(hashlib.sha256(("trajectory-v1:" + source_group_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < 20 else "train"


def _source(case_dir):
    folder = Path(case_dir).resolve()
    paths = {name: folder / filename for name, filename in {
        "assignment": "assignment.json", "result": "result.json", "trace": "public_trace.json",
        "checkpoints": "private_checkpoints.json", "events": "events.jsonl"}.items()}
    assignment, result, trace, checkpoints = (_read(paths[name]) for name in
                                               ("assignment", "result", "trace", "checkpoints"))
    _need(result.get("schema") == "terra-direct-continuous-result.v1"
          and trace.get("schema") == "terra-direct-continuous-public-trace.v1"
          and result.get("completed") is trace.get("completed") is True
          and result.get("termination") == trace.get("termination") == "finished_by_teacher"
          and result.get("error") is trace.get("error") is None, "requires a complete stopped session")
    for record in (assignment, result, trace):
        _need(record.get("student_actions_used") is False and record.get("automatic_robot_actions") is False
              and record.get("actual_hardware") is False, "source is not a direct virtual session")
    _need(assignment.get("final_test_used") is result.get("final_test_used") is False,
          "source is not marked as training-origin")
    _need(assignment.get("teacher_robot_observation_scope") == result.get("teacher_robot_observation_scope")
          == "student_feature_fields_only", "unexpected teacher robot observation scope")
    _need(assignment.get("initial_state_sha256") == result.get("initial_state_sha256")
          == trace.get("initial_state_sha256") and assignment.get("plan_sha256") == result.get("plan_sha256"),
          "source bindings differ")
    _need(_sha(paths["trace"]) == result.get("live_sha256")
          and _sha(paths["checkpoints"]) == result.get("checkpoints_sha256")
          and _sha(paths["events"]) == trace.get("event_journal_sha256"), "source file digest differs")
    plan_path = folder.parent.parent / "training_starts.json"
    _need(_sha(plan_path) == assignment["plan_sha256"], "source plan digest differs")
    plan = _read(plan_path)
    index = assignment.get("case_index")
    _need(type(index) is int and 0 <= index < len(plan["starts"]), "invalid source case index")
    case = plan["starts"][index]
    _need(case.get("original_case_id") == assignment.get("original_case_id")
          and case.get("original_initial_state_sha256") == assignment.get("initial_state_sha256"),
          "source original-start binding differs")
    _need(isinstance(trace.get("stages"), list) and isinstance(checkpoints, list)
          and len(trace["stages"]) == len(checkpoints) > 0, "stage and snapshot counts differ")
    assessments = result.get("stages")
    _need(isinstance(assessments, list) and len(assessments) == len(checkpoints),
          "saved stage assessment count differs")
    for stage, assessment in zip(trace["stages"], assessments, strict=True):
        _need(assessment.get("before") == stage["before"] and assessment.get("task") == stage["task"]
              and assessment.get("status") in ("success", "failure", "unconfirmed"),
              "saved stage assessment does not match public task")
    groups, current = [], []
    for row in trace["rows"]:
        current.append(row)
        if row.get("action") == "DONE":
            groups.append(current)
            current = []
    _need(not current and len(groups) == len(checkpoints), "incomplete or ambiguous stage boundary")
    metadata = {"case_directory": str(folder), "source_group_id": assignment["initial_state_sha256"],
                "original_case_id": assignment["original_case_id"],
                "source_initial_state_sha256": assignment["initial_state_sha256"],
                "source_plan_path": str(plan_path.resolve()), "source_plan_sha256": _sha(plan_path),
                "source_files": {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()},
                "source_program_status": result.get("status"),
                "operator_identity_independently_verified": assignment.get("direct_operator_identity_independently_verified"),
                "independent_teacher_review_required": result.get("independent_teacher_review_required"),
                "teacher_visual_privilege": assignment.get("teacher_visual_privilege", "not_marked_in_assignment"),
                "source_training_eligibility": "not_established_by_replay",
                "source_stages_total": len(groups),
                "stage_assessments": [{"stage_number": i, "status": item["status"],
                                       "samples": item.get("samples"),
                                       "assessment_sha256": item.get("assessment_sha256")}
                                      for i, item in enumerate(assessments, 1)]}
    return case, trace["stages"], checkpoints, groups, metadata


def _robot_joints(sim):
    return [math.degrees(float(sim.data.qpos[sim.model.jnt_qposadr[j]])) for j in sim._joint_ids]


def _replay_velocity(sim):
    return [math.degrees(float(sim.data.qvel[sim.model.jnt_dofadr[j]])) for j in sim._joint_ids]


def _public_after(sim, previous_joints, command):
    """Build an allowed observation without reading simulator internal qvel."""
    joints = _robot_joints(sim)
    opening = sum(float(sim.data.qpos[sim.model.jnt_qposadr[j]]) for j in sim._finger_ids)
    moving = max(abs(a - b) / SAMPLE_DT_S for a, b in zip(joints, previous_joints, strict=True)) > 0.1
    _need(all(math.isfinite(x) for x in [*joints, opening]), "nonfinite replay observation")
    return {"schema_version": "PolicyObservation.phase1", "joints_deg": joints,
            "sensor_valid": True, "moving": moving, "gripper_command": command,
            "gripper_reported_state": "open" if opening >= 0.03 else "closed",
            "gripper_reported_valid": True, "gripper_feedback_fresh": True}


def _make_sim(state):
    reset, config = state["reset_and_randomization"], state["custom_slot_configuration"]
    sim = CustomSlotsSceneMotionSim(reset["initial_layout"], seed=reset["seed"], randomize=reset["randomize"],
        slot_xy_mm=config["slot_xy_mm"], ramp_seconds=config["ramp_seconds"],
        threshold_degrees=config["threshold_degrees"], ramp_shape=config["ramp_shape"])
    sim.load_state(state)
    _need(math.isclose(float(sim.model.opt.timestep), 0.002, abs_tol=1e-12), "unexpected physics timestep")
    _need(sim.save_state()["state"] == state["state"], "saved stage start failed to restore")
    return sim


def _episode(stage, checkpoint, rows, metadata, number):
    _need(checkpoint["before"] == stage["before"] and checkpoint["task"] == stage["task"],
          "stage and snapshot task differ")
    _need(rows and rows[-1].get("action") == "DONE" and rows[-1].get("result") == "done_no_command"
          and all(r.get("action") in ("MOVE", "OPEN", "CLOSE") and r.get("result") == "command_applied"
                  for r in rows[:-1]) and len(rows) > 1, "invalid robot decisions")
    sim = _make_sim(checkpoint["before_state"])
    sim.begin_transfer(stage["before"], stage["task"]["source_slot"], stage["task"]["destination_slot"])
    slots = checkpoint["before_state"]["custom_slot_configuration"]["slot_xy_mm"]
    observation = dict(rows[0]["visible"]["observation"])
    _need(_near(_robot_joints(sim), observation["joints_deg"]), "first robot observation differs from replay")
    samples, decisions, max_boundary_error = [], 0, 0.0
    ticks_per_sample = round(SAMPLE_DT_S / float(sim.model.opt.timestep))
    _need(ticks_per_sample == 10, "unexpected sample rate")
    for row_index, row in enumerate(rows[:-1]):
        _need(row["visible"]["task"] == stage["task"] and row["visible"]["expected_scene"] == stage["before"]
              and row["phase"] == stage["phase"], "public task or registration differs")
        current = _robot_joints(sim)
        recorded = row["visible"]["observation"]["joints_deg"]
        _need(_near(current, recorded), "recorded command boundary differs from replay")
        max_boundary_error = max(max_boundary_error, *(abs(a-b) for a,b in zip(current, recorded, strict=True)))
        actual_targets = [math.degrees(float(sim.data.ctrl[a])) for a in sim._arm_actuators]
        _need(_near(actual_targets, row["previous_applied_target_deg"]), "previous command differs from replay")
        action = row["action"]
        # The source saved the transmitted guard output as normalized float32.
        # Its applied target is a later radians/degrees readback, not necessarily
        # bit-identical to the value originally handed to the simulator.
        target = ([float(x) * 180. for x in row["guard"]["guard_returned_normalized_joint_output"][0]]
                  if action == "MOVE" else actual_targets)
        _need(_near(target, row["applied_target_deg"]), "applied target differs")
        grip = ("open" if action == "OPEN" else "closed" if action == "CLOSE"
                else observation["gripper_command"])
        _need(grip in ("open", "closed"), "unsupported gripper command")
        opening = .04 if grip == "open" else 0.
        tick_count = 0
        old_step = mujoco.mj_step

        def record_step(model, data):
            nonlocal tick_count, observation
            old_step(model, data)
            tick_count += 1
            if tick_count % ticks_per_sample:
                return
            before_joints = observation["joints_deg"]
            features = scene_features.encode_scene_features(observation, stage["before"], stage["task"], slots)
            after = _public_after(sim, before_joints, grip)
            velocity = _replay_velocity(sim)
            _need(all(math.isfinite(v) for v in velocity), "nonfinite replay velocity")
            samples.append({"t_s": round(len(samples) * SAMPLE_DT_S, 8), "dt_s": SAMPLE_DT_S,
                "features": features, "target_joint_deg": after["joints_deg"],
                "target_velocity_deg_s": velocity, "target_gripper": grip,
                "terminal": row_index == len(rows) - 2 and tick_count == 500})
            observation = after

        # Recording is local to this isolated process.  Object positions and
        # contact details from MuJoCo are never read into feature construction.
        with patch.object(mujoco, "mj_step", new=record_step), patch.object(sim, "observation", return_value={}):
            sim.command(target, opening, seconds=1.)
        _need(tick_count == 500 and _near(_robot_joints(sim), row["returned_observation"]["joints_deg"]),
              "recorded command return differs from replay")
        decisions += 1
    _need(_near(_robot_joints(sim), rows[-1]["visible"]["observation"]["joints_deg"]),
          "recorded DONE boundary differs from replay")
    replay_done = sim.save_state()["state"]
    expected_done = checkpoint["done_state"]["state"]
    # A restored MuJoCo state can differ very slightly from the live run's
    # contact solver history.  Record the mixed-unit state-vector delta and
    # reject large divergences instead of claiming bit-exact physical replay.
    _need(_near(replay_done, expected_done, tolerance=1e-5), "saved DONE state differs from replay")
    _need(len(samples) == 50 * decisions and sum(x["terminal"] for x in samples) == 1,
          "sample count or terminal boundary differs")
    directory = Path(metadata["case_directory"])
    episode_id = f"{directory.parent.parent.name}/{directory.name}:stage-{number:03d}"
    return {"episode_id": episode_id, "source_group_id": metadata["source_group_id"],
            "split": source_split(metadata["source_group_id"]), "source_case_directory": metadata["case_directory"],
            "source_stage_number": number, "source_phase": stage["phase"],
            "source_observation_ids": [r["observation_id"] for r in rows],
            "recorded_commands": decisions, "max_boundary_joint_error_deg": max_boundary_error,
            "done_state_mixed_unit_vector_max_abs_error": max(abs(a-b) for a,b in zip(replay_done, expected_done, strict=True)),
            "samples": samples}


def build_dataset(case_dirs, *, max_stages_per_case=None, success_only=False, skip_unreplayable=False):
    """Replay explicit source folders; stages of an original start never split."""
    _need(case_dirs and type(success_only) is bool and type(skip_unreplayable) is bool and (max_stages_per_case is None or
          (type(max_stages_per_case) is int and max_stages_per_case > 0)), "invalid sources or stage limit")
    paths = [Path(p).resolve() for p in case_dirs]
    _need(len(paths) == len(set(paths)), "duplicate source folder")
    source_records, episodes, original_bindings = [], [], {}
    for path in paths:
        _, stages, checkpoints, decisions, metadata = _source(path)
        group = metadata["source_group_id"]
        original = metadata["original_case_id"]
        _need(original not in original_bindings or original_bindings[original] == group,
              "one original ID has different saved initial states")
        original_bindings[original] = group
        metadata["source_stages_included"] = []
        metadata["source_stages_excluded"] = []
        for i in range(len(stages)):
            assessment = metadata["stage_assessments"][i]
            if success_only and not (assessment["status"] == "success" and assessment["samples"] == 501):
                metadata["source_stages_excluded"].append({"stage_number": i + 1,
                    "reason": "no_recorded_501_sample_success", "status": assessment["status"]})
                continue
            if max_stages_per_case is not None and len(metadata["source_stages_included"]) >= max_stages_per_case:
                metadata["source_stages_excluded"].append({"stage_number": i + 1,
                    "reason": "per_case_limit", "status": assessment["status"]})
                continue
            try:
                episode = _episode(stages[i], checkpoints[i], decisions[i], metadata, i + 1)
            except (ValueError, RuntimeError, KeyError) as exc:
                if not skip_unreplayable:
                    raise
                metadata["source_stages_excluded"].append({"stage_number": i + 1,
                    "reason": "replay_verification_failed", "status": assessment["status"],
                    "detail": f"{type(exc).__name__}: {exc}"})
                continue
            episode["source_stage_posthoc_status"] = assessment["status"]
            episode["source_stage_posthoc_sample_count"] = assessment["samples"]
            metadata["source_stages_included"].append(i + 1)
            episodes.append(episode)
        source_records.append(metadata)
    return {"schema": SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
        "metadata": {"source_kind": "simulator_replay_of_direct_position_and_gripper_decisions",
            "student_inputs": "current permitted robot observation and registered scene/task/slot features; moving within the replay is estimated from observed angles, except the original public first observation",
            "sample_dt_s": SAMPLE_DT_S, "physics_dt_s": 0.002,
            "position_target": "simulated observed joint position at t + dt, degrees; not original operator command",
            "velocity_target": "simulated joint qvel at t + dt, signed degrees/second; not an original operator speed command and not the derivative of an interpolated target",
            "gripper_target": "applied open/closed target during [t,t+dt]",
            "terminal_target": "true only when t+dt reaches recorded transfer DONE boundary",
            "time_origin": "elapsed virtual time since start of this individual transfer; one-second commands replayed without gaps",
            "split": "sha256('trajectory-v1:' + initial_state_sha256) first 32 bits modulo 100 < 20 means validation; otherwise train; same initial-state hash always stays together; conflicting states for one original ID are rejected; internal split of training-origin starts, not independent final holdout",
            "actual_hardware": False, "new_operator_demonstration": False,
            "student_object_positions_contacts_or_simulator_qvel_as_input": False,
            "physical_success_assessed": False, "identity_augmentation": False,
            "training_eligibility": "not_established_by_replay; preserve source review restrictions",
            "mujoco_version": mujoco.__version__, "max_stages_per_case": max_stages_per_case,
            "selection": "recorded 501-sample successful stage only" if success_only else "all requested stages; no success filtering",
            "unreplayable_stages_recorded_and_excluded": skip_unreplayable,
            "episodes": len(episodes), "samples": sum(len(e["samples"]) for e in episodes),
            "source_groups": len(set(original_bindings.values())), "sources": source_records}, "episodes": episodes}

