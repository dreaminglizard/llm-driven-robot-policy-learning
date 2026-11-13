"""Programmatic scene-transfer demonstrations with auditable source lineage.

This is a new teacher variant inspired by two original assistant-operated
simulation demonstrations, not an exact replay and not new LLM decisions.
Object poses and contacts are privileged teacher/evaluator information.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from raccoon_control.kinematics import Geometry, check_joints
from raccoon_memory import motion_policy, scene_features, sim_data, task_catalog, unstack_sim_data
from raccoon_memory.scene_motion_sim import SceneMotionSim


ROOT = Path(__file__).resolve().parents[1]
SPLIT_RANGES = {"pilot": (900, 902), "train": (1000, 1099),
                "validation": (1500, 1519), "test": (2000, 2019)}
TEACHER = "programmatic_scene_feedback_variant_from_two_assistant_demonstrations"
SOURCE = "programmatic_scene_physics_demonstration"
STAGES = ("source_approach", "source_pregrasp", "grasp_descent", "gripper_close",
          "grasp_lift", "clearance_lift", "transit_turn", "destination_above",
          "destination_preposition", "placement", "gripper_open", "retreat")
COMMAND_SECONDS = 1.0
MAX_TRANSIT_ROTATION_DEG = 30.0


def _teacher_variant(grasp_tcp_base_mm: float, placement_pitch_deg: float = 0.0) -> str:
    if type(grasp_tcp_base_mm) not in (int, float) or float(grasp_tcp_base_mm) not in (15.0, 20.0):
        raise ValueError("grasp TCP base height must be 15 or 20 mm")
    if type(placement_pitch_deg) not in (int, float) or float(placement_pitch_deg) not in (0.0, -5.0):
        raise ValueError("placement pitch must be 0 or -5 degrees")
    if float(placement_pitch_deg) == -5.0:
        if float(grasp_tcp_base_mm) != 15.0:
            raise ValueError("the -5 degree placement variant requires the 15 mm grasp base")
        return "segmented_transit_grasp15_placepitch_minus5_v4"
    return "segmented_transit_grasp15_v3" if float(grasp_tcp_base_mm) == 15.0 else "segmented_transit_v2"


class TeacherWaypointError(ValueError):
    """Keep the intended waypoint and original error when inverse kinematics fails."""

    def __init__(self, evidence: dict, cause: Exception):
        super().__init__(str(cause))
        self.evidence = copy.deepcopy(evidence)
        self.cause_type = type(cause).__name__


class GraspLost(ValueError):
    """The object is far from the TCP with no positive finger-pad contact."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def split_group(case: dict) -> str:
    """Group equivalent count/height patterns regardless of slot or object IDs."""
    counts = tuple(case["counts_before"])
    task = case["task"]
    source, destination = task["source_slot"], task["destination_slot"]
    if (len(counts) != 4 or type(source) is not int or type(destination) is not int
            or not 1 <= source <= 4 or not 1 <= destination <= 4 or source == destination):
        raise ValueError("invalid case for split grouping")
    other = sorted(counts[i] for i in range(4) if i not in (source - 1, destination - 1))
    source_level, destination_level = counts[source - 1], counts[destination - 1] + 1
    if task["source_level"] != source_level or task["destination_level"] != destination_level:
        raise ValueError("case levels contradict its starting counts")
    return f"n{sum(counts)}_s{source_level}_d{destination_level}_other{other[0]}{other[1]}"


def read_split_plan(path: str | Path) -> tuple[dict, str, bytes]:
    """Validate the same disjoint group plan for training and validation."""
    raw = Path(path).read_bytes()
    plan = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("split plan must be an object with three group arrays")
    known = {split_group(case) for case in task_catalog.one_step_cases()}
    groups = {}
    for name in ("train", "validation", "test"):
        values = plan.get(f"{name}_groups")
        if not isinstance(values, list) or any(type(value) is not str for value in values):
            raise ValueError(f"split plan {name} must be an array of group names")
        if len(values) != len(set(values)) or not set(values).issubset(known):
            raise ValueError(f"split plan {name} contains duplicate or unknown groups")
        groups[name] = set(values)
    if (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]):
        raise ValueError("training, validation and final-test split groups must be disjoint")
    return plan, _sha(raw), raw


def _select_cases(case_ids: list[str] | None, all_cases: bool, id_map: tuple[int, ...]) -> list[dict]:
    if len(id_map) != 4 or any(type(obj) is not int for obj in id_map) or set(id_map) != set(task_catalog.OBJECT_IDS):
        raise ValueError("id-map must be a permutation of 1, 2, 3 and 4")
    catalogue = task_catalog.one_step_cases()
    by_id = {case["case_id"]: case for case in catalogue}
    if len(by_id) != len(catalogue):
        raise ValueError("duplicate case IDs in task catalogue")
    if bool(all_cases) == bool(case_ids):
        raise ValueError("select explicit case IDs or all cases, exactly one")
    ids = list(by_id) if all_cases else list(case_ids)
    if len(ids) != len(set(ids)) or any(case_id not in by_id for case_id in ids):
        raise ValueError("case IDs must be known and unique")
    selected = []
    for case_id in ids:
        case = copy.deepcopy(by_id[case_id])
        before = tuple(tuple(id_map[obj - 1] for obj in stack) for stack in case["canonical_scene_before"])
        before, task = scene_features.transfer_task(before, case["task"])
        case["registered_scene"] = [list(stack) for stack in before]
        case["task"] = task
        case["moved_object_id"] = task["object_id"]
        case["split_group"] = split_group(case)
        case["id_map"] = list(id_map)
        selected.append(case)
    return selected


def _reference_summary(source: dict) -> dict:
    fields = ("demonstration_directory", "events_sha256", "state_sha256", "head_event_sha256",
              "base_model_xml_sha256", "simulator_code_sha256", "assistant_original_episodes",
              "original_recorded_actions")
    result = {key: source[key] for key in fields}
    if result["assistant_original_episodes"] != 1 or result["original_recorded_actions"] != len(STAGES):
        raise ValueError("each reference must be one verified 12-action assistant demonstration")
    return result


def load_source_references() -> tuple[dict, dict]:
    stack_actions, stack = sim_data.load_reference_demo()
    unstack_actions, unstack = unstack_sim_data.load_reference_demo()
    templates = {"stack": stack_actions, "unstack": unstack_actions}
    for name, events in templates.items():
        if len(events) != len(STAGES):
            raise ValueError(f"{name} reference must have 12 actions")
        openings = [float(event["action"]["gripper_opening_m"]) for event in events]
        if (any(not math.isclose(openings[i], 0.04, abs_tol=1e-9) for i in (0, 1, 2, 10, 11))
                or any(not math.isclose(openings[i], 0.0, abs_tol=1e-9) for i in range(3, 10))):
            raise ValueError(f"{name} reference gripper events differ from the expected 12-stage structure")
    return {"stack": _reference_summary(stack), "unstack": _reference_summary(unstack)}, templates


def _objects(raw: dict) -> dict:
    privileged = raw["privileged_simulation_only"]
    values = privileged["objects"]
    result = {obj["id"]: obj for obj in values}
    if len(result) != len(values):
        raise ValueError("duplicate object ID in simulator observation")
    return result


def _object_xyz_mm(raw: dict, object_id: int) -> list[float]:
    position = _objects(raw)[object_id]["position_m"]
    if len(position) != 3 or any(not math.isfinite(float(v)) for v in position):
        raise ValueError("invalid privileged teacher object position")
    return [float(value) * 1000 for value in position]


def _destination_xy(raw: dict, scene, task: dict) -> tuple[list[float], str]:
    stack = scene[task["destination_slot"] - 1]
    if stack:
        return _object_xyz_mm(raw, stack[-1])[:2], "observed_support_position"
    return list(scene_features.NOMINAL_SLOT_XY_MM[task["destination_slot"] - 1]), "registered_nominal_slot"


def target_for_stage(index: int, raw: dict, scene, task: dict, previous_target,
                     grasp_tcp_base_mm: float = 20.0, placement_pitch_deg: float = 0.0) -> tuple[list[float], dict]:
    """Construct one new teacher waypoint; diagnostics stay outside student input."""
    if type(index) is not int or not 0 <= index < len(STAGES):
        raise ValueError("invalid teacher stage")
    _teacher_variant(grasp_tcp_base_mm, placement_pitch_deg)
    geometry = Geometry()
    source_xyz = _object_xyz_mm(raw, task["object_id"])
    destination_xy, destination_basis = _destination_xy(raw, scene, task)
    level = task["source_level"]
    evidence = {"stage": STAGES[index], "source_object_xyz_mm": source_xyz,
                "destination_xy_mm": destination_xy, "destination_xy_basis": destination_basis}
    if index in (3, 10):
        return list(previous_target), evidence
    if index in (0, 1, 2, 4, 5):
        z = {0: 100.0, 1: 40.0 + 20.0 * (level - 1), 2: float(grasp_tcp_base_mm) + 20.0 * (level - 1),
             4: 60.0 + 20.0 * (level - 1), 5: 130.0}[index]
        xyz = [*source_xyz[:2], z]
    elif index == 6:
        raise ValueError("transit rotation must use the segmented waypoint planner")
    elif index == 7:
        xyz = [*destination_xy, 130.0]
    elif index in (8, 9):
        tcp = [float(value) for value in raw["tcp_mm"]]
        if len(tcp) != 3 or any(not math.isfinite(value) for value in tcp):
            raise ValueError("invalid teacher TCP observation")
        offset = [tcp[i] - source_xyz[i] for i in range(3)]
        desired_center = [*destination_xy, 10.0 + 20.0 * (task["destination_level"] - 1)]
        desired_tcp = [desired_center[i] + offset[i] for i in range(3)]
        xyz = list(desired_tcp)
        if index == 8:
            xyz[2] = max(60.0, desired_tcp[2] + 20.0)
        evidence.update(observed_tcp_mm=tcp, tcp_minus_object_mm=offset,
                        desired_object_center_mm=desired_center, compensated_tcp_mm=desired_tcp)
    else:
        last_xyz = geometry.forward(previous_target)
        xyz = [last_xyz[0], last_xyz[1], 100.0]
    evidence["requested_tcp_mm"] = list(xyz)
    pitch = float(placement_pitch_deg) if index in (8, 9) else 0.0
    evidence["requested_pitch_deg"] = pitch
    try:
        return list(geometry.inverse(xyz, pitch_deg=pitch)), evidence
    except Exception as exc:
        raise TeacherWaypointError(evidence, exc) from exc


def transit_waypoints(raw: dict, scene, task: dict, previous_target) -> tuple[list[list[float]], dict]:
    """Turn at clearance in commanded increments no greater than 30 degrees."""
    destination_xy, basis = _destination_xy(raw, scene, task)
    evidence = {"stage": "transit_turn", "destination_xy_mm": destination_xy,
                "destination_xy_basis": basis, "transit_destination_tcp_mm": [*destination_xy, 130.0],
                "max_rotation_per_segment_deg": MAX_TRANSIT_ROTATION_DEG}
    try:
        destination = Geometry().inverse(evidence["transit_destination_tcp_mm"])
    except Exception as exc:
        raise TeacherWaypointError(evidence, exc) from exc
    start, end = float(previous_target[0]), float(destination[0])
    segments = max(1, math.ceil(abs(end - start) / MAX_TRANSIT_ROTATION_DEG))
    targets = [[start + (end - start) * number / segments, *previous_target[1:]]
               for number in range(1, segments + 1)]
    evidence.update(start_joint_1_deg=start, destination_joint_1_deg=end, segments=segments)
    return targets, evidence


def grasp_evidence(raw: dict, object_id: int) -> dict:
    """Log measured object/TCP separation and positive contacts at a command boundary."""
    obj_xyz = _object_xyz_mm(raw, object_id)
    tcp = [float(value) for value in raw["tcp_mm"]]
    if len(tcp) != 3 or any(not math.isfinite(value) for value in tcp):
        raise ValueError("invalid observed TCP")
    contacts = raw["privileged_simulation_only"]["contacts"]
    moved_geom = f"object_{object_id}_geom"
    other_geoms = {f"object_{obj}_geom" for obj in _objects(raw) if obj != object_id}
    positive = [contact for contact in contacts if float(contact["normal_force_n"]) > 0]
    left = any({moved_geom, "finger_left_pad"} == {c["geom1"], c["geom2"]} for c in positive)
    right = any({moved_geom, "finger_right_pad"} == {c["geom1"], c["geom2"]} for c in positive)
    other_robot = [c for c in positive if {c["geom1"], c["geom2"]} & other_geoms
                   and {c["geom1"], c["geom2"]} & sim_data.ROBOT_GEOMS]
    moved_other = [c for c in positive if moved_geom in (c["geom1"], c["geom2"])
                   and {c["geom1"], c["geom2"]} & other_geoms]
    distance = math.dist(obj_xyz, tcp)
    return {"object_id": object_id, "object_xyz_mm": obj_xyz, "tcp_mm": tcp, "tcp_object_distance_mm": distance,
            "left_pad_positive_contact": left, "right_pad_positive_contact": right,
            "no_pad_contact_and_distance_over_25mm": not left and not right and distance > 25.0,
            "other_cylinder_robot_contacts": other_robot, "moved_other_cylinder_contacts": moved_other,
            "scope": "Positive contacts at this command endpoint only; intermediate contacts are not sampled here."}


def _check_assessment(assessment: dict, before, task: dict) -> bool:
    """Require whole-scene support, stability and preservation in the result."""
    if not isinstance(assessment, dict) or type(assessment.get("success")) is not bool:
        raise ValueError("invalid transfer assessment")
    after = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
    if (assessment.get("source") != "mujoco_simulation" or assessment.get("simulated") is not True
            or assessment.get("actual_hardware") is not False
            or task_catalog.scene(assessment.get("before_layout")) != task_catalog.scene(before)
            or task_catalog.scene(assessment.get("after_layout")) != after
            or assessment.get("moved_object_id") != task["object_id"]
            or assessment.get("stable_interval_seconds", 0) < 1.0 - 1e-9
            or type(assessment.get("sample_count")) is not int or assessment["sample_count"] < 2):
        raise ValueError("transfer assessment does not cover the requested scene and stability interval")
    checks = assessment.get("checks")
    required = {"gripper_open", "no_unexpected_cylinder_contact"}
    for obj in (item for stack in before for item in stack):
        required.update(f"object_{obj}.{name}" for name in (
            "position_xy", "height", "upright", "linear_speed", "angular_speed",
            "required_support_contact", "free_of_robot_contact"))
        if obj != task["object_id"]:
            required.update((f"object_{obj}.registered_slot_xy", f"object_{obj}.registered_level_height"))
    if task["destination_level"] == 2:
        required.update((f"object_{task['object_id']}.aligned_with_support_xy",
                         f"object_{task['object_id']}.aligned_with_support_height"))
    if (not isinstance(checks, dict) or not required.issubset(checks)
            or any(type(value) is not bool for value in checks.values())
            or assessment["success"] != all(checks.values())):
        raise ValueError("transfer assessment has missing or inconsistent scene-preservation checks")
    return assessment["success"]


def generate_attempt(case: dict, seed: int, split: str, references: dict, templates: dict,
                     *, randomize: bool, split_plan_sha256: str | None = None,
                     grasp_tcp_base_mm: float = 20.0, placement_pitch_deg: float = 0.0,
                     _current_sim: SceneMotionSim | None = None, _continuation: dict | None = None) -> tuple[dict, dict | None]:
    teacher_variant = _teacher_variant(grasp_tcp_base_mm, placement_pitch_deg)
    id_suffix = "".join(str(obj) for obj in case["id_map"])
    episode_id = f"sim-scene-{split}-{case['case_id']}-ids{id_suffix}-seed{seed}"
    if _current_sim is not None:
        if not isinstance(_continuation, dict) or not isinstance(_continuation.get("parent_episode_id"), str):
            raise ValueError("a continued simulator needs explicit parent provenance")
        episode_id += "-continued-after-" + _continuation["parent_episode_id"]
    before = task_catalog.scene(case["registered_scene"])
    task = dict(case["task"])
    template_name = "unstack" if task["source_level"] == 2 else "stack"
    attempt = {"episode_id": episode_id, "seed": seed, "split": split, "case_id": case["case_id"],
               "split_group": case["split_group"], "split_plan_sha256": split_plan_sha256,
               "registered_scene": [list(stack) for stack in before], "task": task,
               "id_map": list(case["id_map"]), "nominal_slot_xy_mm": [list(xy) for xy in scene_features.NOMINAL_SLOT_XY_MM],
               "source": SOURCE, "teacher": TEACHER, "source_references": copy.deepcopy(references),
               "template": template_name, "teacher_variant": teacher_variant,
               "grasp_tcp_base_mm": float(grasp_tcp_base_mm), "placement_pitch_deg": float(placement_pitch_deg),
               "source_replay": False, "new_llm_decisions": False,
               "simulated": True, "actual_hardware": False, "randomize": randomize,
               "decisions": [], "success": False}
    transitions = []
    sim = None
    try:
        sim = SceneMotionSim(before, seed=seed, randomize=randomize) if _current_sim is None else _current_sim
        if _current_sim is not None:
            if not isinstance(sim, SceneMotionSim):
                raise ValueError("continued teacher requires SceneMotionSim")
            if (sim.layout_status != "registered" or sim.pending_transfer is not None
                    or task_catalog.scene(sim.observation()["registered_layout"]) != before):
                raise ValueError("continued scene must be registered, idle and match the next task")
        initial = sim.observation()
        attempt["randomized_parameters"] = copy.deepcopy(initial["privileged_simulation_only"]["reset_and_randomization"])
        attempt["simulator_metadata"] = copy.deepcopy(initial["metadata"])
        attempt["initial_state"] = sim.save_state()
        if _current_sim is None:
            previous_target = list(initial["joints_deg"])
            opening, gripper_command = 0.04, "open"
        else:
            previous_target = [math.degrees(float(sim.data.ctrl[actuator])) for actuator in sim._arm_actuators]
            check_joints(previous_target)
            finger_targets = [float(sim.data.ctrl[actuator]) for actuator in sim._finger_actuators]
            if (any(not math.isfinite(value) or not 0 <= value <= 0.025 for value in finger_targets)
                    or not math.isclose(finger_targets[0], finger_targets[1], abs_tol=1e-9)
                    or sum(finger_targets) < sim_data.GRIPPER_OPEN_THRESHOLD_M
                    or initial["gripper_opening_m"] < sim_data.GRIPPER_OPEN_THRESHOLD_M):
                raise ValueError("continued gripper must be open with valid symmetric servo targets")
            opening, gripper_command = sum(finger_targets), "open"
            attempt["continuation"] = copy.deepcopy(_continuation)
            attempt["continuation"]["initial_observed_joints_deg"] = list(initial["joints_deg"])
            attempt["continuation"]["initial_servo_target_joints_deg"] = list(previous_target)
            attempt["continuation"]["initial_finger_servo_targets_m"] = finger_targets
            attempt["continuation"]["student_history"] = "empty at task start; maximum eight observations"
        attempt["transfer_contract"] = sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
        for index, stage in enumerate(STAGES):
            transit_info = None
            if index == 6:
                planning_observation = sim.observation()
                try:
                    planned_targets, transit_info = transit_waypoints(planning_observation, before, task, previous_target)
                except TeacherWaypointError as exc:
                    attempt["decisions"].append({"stage": stage,
                        "source_template_event_sha256": templates[template_name][index]["event_sha256"],
                        "input_observation": sim_data.sim_policy_observation(sim, gripper_command),
                        "privileged_simulation_only": {"before": planning_observation["privileged_simulation_only"],
                                                       "teacher_target_evidence": exc.evidence},
                        "waypoint_error": {"cause_type": exc.cause_type, "message": str(exc)}})
                    raise
            else:
                planned_targets = [None]
            for segment_number, planned_target in enumerate(planned_targets, 1):
                raw_before = sim.observation()
                obs = sim_data.sim_policy_observation(sim, gripper_command)
                scene_features.encode_scene_features(obs, before, task, scene_features.NOMINAL_SLOT_XY_MM)
                decision = {"stage": stage, "source_template_event_sha256": templates[template_name][index]["event_sha256"],
                            "input_observation": obs,
                            "privileged_simulation_only": {"before": raw_before["privileged_simulation_only"]}}
                attempt["decisions"].append(decision)
                try:
                    if planned_target is not None:
                        target = list(planned_target)
                        evidence = {**transit_info, "segment_number": segment_number,
                                    "requested_joints_deg": list(target),
                                    "rotation_from_previous_target_deg": abs(target[0] - previous_target[0])}
                    else:
                        target, evidence = target_for_stage(index, raw_before, before, task, previous_target,
                                                            grasp_tcp_base_mm=grasp_tcp_base_mm,
                                                            placement_pitch_deg=placement_pitch_deg)
                    if index in (6, 7) and abs(target[0] - previous_target[0]) > MAX_TRANSIT_ROTATION_DEG + 1e-7:
                        raise ValueError("transit rotation exceeds the maximum commanded segment")
                except TeacherWaypointError as exc:
                    decision["privileged_simulation_only"]["teacher_target_evidence"] = exc.evidence
                    decision["waypoint_error"] = {"cause_type": exc.cause_type, "message": str(exc)}
                    raise
                decision["privileged_simulation_only"]["teacher_target_evidence"] = evidence
                if _current_sim is not None and not transitions:
                    decision["initial_approach_delta_deg"] = [float(target[i]) - float(initial["joints_deg"][i]) for i in range(4)]
                    attempt["continuation"]["initial_approach_delta_deg"] = list(decision["initial_approach_delta_deg"])
                    attempt["continuation"]["initial_approach_delta_norm_deg"] = math.sqrt(sum(x*x for x in decision["initial_approach_delta_deg"]))
                next_opening = 0.0 if index == 3 else (0.04 if index == 10 else opening)
                changing = not math.isclose(next_opening, opening, abs_tol=1e-9)
                next_command = "open" if next_opening >= sim_data.GRIPPER_OPEN_THRESHOLD_M else "closed"
                decision["simulated_servo_target"] = {"joints_deg": list(target),
                                                       "gripper_opening_m": next_opening, "seconds": COMMAND_SECONDS}
                sim.command(target, next_opening, seconds=COMMAND_SECONDS)
                next_obs = sim_data.sim_policy_observation(sim, next_command)
                scene_features.encode_scene_features(next_obs, before, task, scene_features.NOMINAL_SLOT_XY_MM)
                raw_after = sim.observation()
                decision["privileged_simulation_only"]["after"] = raw_after["privileged_simulation_only"]
                decision["privileged_simulation_only"]["grasp_and_contact_check"] = grasp_evidence(raw_after, task["object_id"])
                decision["simulation_arm_after"] = {key: raw_after[key] for key in
                                                    ("time_seconds", "joints_deg", "tcp_mm", "gripper_opening_m")}
                if index in (4, 5, 6, 7, 8) and decision["privileged_simulation_only"]["grasp_and_contact_check"]["no_pad_contact_and_distance_over_25mm"]:
                    raise GraspLost("object is farther than 25 mm from TCP and has no positive finger-pad contact")
                if changing and next_obs["gripper_reported_state"] != next_command:
                    raise ValueError("simulated gripper did not report the requested state")
                command = ({"kind": "set_gripper", "state": next_command} if changing else
                           {"kind": "move_joints", "joints_deg": list(target), "speed_pct": sim_data.SPEED_LABEL_PCT})
                prefix = f"{episode_id}-{len(transitions)}"
                transition = {"observation": obs, "action": command, "next_observation": next_obs,
                              "applied": True, "status": "gripper_reported" if changing else "completed",
                              "provenance": {"command_id": f"{prefix}-command",
                                             "input_observation_id": f"{prefix}-before",
                                             "result_observation_id": f"{prefix}-after",
                                             "teacher": TEACHER}}
                if _current_sim is not None:
                    transition["provenance"]["teacher_stage"] = stage
                    transition["provenance"]["teacher_stage_segment"] = segment_number
                transitions.append(transition)
                previous_target, opening, gripper_command = list(target), next_opening, next_command
        attempt["pre_assessment_state"] = sim.save_state()
        assessment = sim.evaluate_transfer(before, task["source_slot"], task["destination_slot"], stable_seconds=1.0)
        attempt["physical_assessment"] = assessment
        passed = _check_assessment(assessment, before, task)
        attempt["post_assessment_state"] = sim.save_state()
        if not passed:
            attempt["failure_reason"] = "physical_success_criterion_failed"
            return attempt, None
        quality = {"source": SOURCE, "teacher": TEACHER, "teacher_variant": teacher_variant,
                   "grasp_tcp_base_mm": float(grasp_tcp_base_mm), "placement_pitch_deg": float(placement_pitch_deg),
                   "simulated": True, "actual_hardware": False,
                   "physical_success_in_simulation": True, "physical_assessment_summary": {
                       "success": True, "stable_interval_seconds": assessment["stable_interval_seconds"],
                       "sample_count": assessment["sample_count"], "checks": copy.deepcopy(assessment["checks"]),
                       "detailed_evidence": "attempts.jsonl: matching episode_id"},
                   "split": split, "split_plan_sha256": split_plan_sha256, "source_replay": False,
                   "new_llm_decisions": False, "teacher_target_information": "privileged; excluded from student encoding",
                   "gripper_report": "simulation-derived opening threshold, not a hardware sensor",
                   "speed_label": "8 percent is a compatibility label, not a measured simulation speed"}
        if _current_sim is not None:
            quality["start_state"] = "restored_successful_parent_post_assessment_state"
            quality["continuation"] = copy.deepcopy(attempt["continuation"])
        episode = {"episode_id": episode_id, "task": task, "registered_scene": [list(stack) for stack in before],
                   "nominal_slot_xy_mm": [list(xy) for xy in scene_features.NOMINAL_SLOT_XY_MM],
                   "split_group": case["split_group"], "source_references": copy.deepcopy(references),
                   "quality": quality, "transitions": transitions}
        # Validate the existing episode schema and every new common-policy input.
        motion_policy.build_episodes([episode])
        for transition in transitions:
            for key in ("observation", "next_observation"):
                scene_features.encode_scene_features(transition[key], before, task, scene_features.NOMINAL_SLOT_XY_MM)
        attempt["success"] = True
        return attempt, episode
    except Exception as exc:
        reason = "grasp_lost_during_transport" if isinstance(exc, GraspLost) else "attempt_exception"
        attempt.update(failure_reason=reason, exception_type=type(exc).__name__, exception_detail=str(exc))
        if attempt["decisions"]:
            attempt["failure_stage"] = attempt["decisions"][-1]["stage"]
        if sim is not None:
            try:
                reason = f"teacher_{reason}: {type(exc).__name__}: {exc}"
                attempt["layout_uncertain_result"] = sim.mark_layout_uncertain(reason)
            except Exception as mark_exc:
                attempt["layout_uncertain_error"] = {"type": type(mark_exc).__name__, "message": str(mark_exc)}
            try:
                attempt["failure_state"] = sim.save_state()
                attempt["failure_observation"] = sim.observation()
            except Exception as snapshot_exc:
                attempt["failure_snapshot_error"] = {"type": type(snapshot_exc).__name__, "message": str(snapshot_exc)}
        return attempt, None


def generate_attempt_from_state(sim: SceneMotionSim, case: dict, seed: int, references: dict, templates: dict,
                                *, split_plan_sha256: str, continuation: dict) -> tuple[dict, dict | None]:
    """Continue a verified idle scene using the saved servo targets, never resetting it."""
    if type(seed) is not int or not 1000 <= seed <= 1099:
        raise ValueError("continued teacher collection accepts only training seeds")
    if not isinstance(sim, SceneMotionSim) or not isinstance(continuation, dict):
        raise ValueError("continued teacher requires a verified scene and parent evidence")
    saved_state = sim.save_state()
    reset_info = saved_state.get("reset_and_randomization")
    if (not isinstance(reset_info, dict) or type(reset_info.get("seed")) is not int
            or reset_info["seed"] != seed or reset_info.get("randomize") is not True):
        raise ValueError("restored physical seed or randomization differs from the continuation contract")
    state_hash = _sha(json.dumps(saved_state, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if (continuation.get("entire_state_matches") is not True
            or continuation.get("parent_post_assessment_state_sha256") != state_hash
            or continuation.get("restored_pre_command_state_sha256") != state_hash):
        raise ValueError("current state does not match the verified parent snapshot")
    plan, actual_hash, _ = read_split_plan(ROOT / "configs" / "scene_split_plan.v1.json")
    before, task = scene_features.transfer_task(case.get("registered_scene"), case.get("task"))
    derived_group = split_group({"counts_before": [len(stack) for stack in before], "task": task})
    if (actual_hash != split_plan_sha256 or case.get("split_group") != derived_group
            or derived_group not in plan["train_groups"] or case.get("task") != task):
        raise ValueError("continued task does not belong to the pinned training split")
    return generate_attempt(case, seed, "train", references, templates, randomize=True,
                            split_plan_sha256=split_plan_sha256, grasp_tcp_base_mm=15.0,
                            placement_pitch_deg=-5.0, _current_sim=sim, _continuation=continuation)


def generate_dataset(output: str | Path, seeds: list[int], split: str, *, case_ids: list[str] | None = None,
                     all_cases: bool = False, id_map=(1, 2, 3, 4), randomize: bool = False,
                     split_plan: str | Path | None = None, grasp_tcp_base_mm: float = 20.0,
                     placement_pitch_deg: float = 0.0) -> dict:
    """Save every selected case/seed attempt and only physically verified positives."""
    requested = Path(output).absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"output must be a new directory: {requested}")
    if split not in SPLIT_RANGES:
        raise ValueError("unknown data split")
    if split == "test":
        raise ValueError("final test seeds are reserved; generation is currently disabled")
    low, high = SPLIT_RANGES[split]
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or not low <= seed <= high for seed in seeds)):
        raise ValueError(f"{split} seeds must be unique integers in [{low}, {high}]")
    if type(randomize) is not bool:
        raise ValueError("randomize must be boolean")
    teacher_variant = _teacher_variant(grasp_tcp_base_mm, placement_pitch_deg)
    cases = _select_cases(case_ids, all_cases, tuple(id_map))
    requested_case_count = len(cases)
    plan_hash, plan_raw = None, None
    if split_plan is None:
        raise ValueError("pilot, train and validation require the same explicit disjoint split plan")
    if split_plan is not None:
        plan, plan_hash, plan_raw = read_split_plan(split_plan)
        plan_split = "train" if split == "pilot" else split
        permitted = set(plan[f"{plan_split}_groups"])
        refused = sorted({case["split_group"] for case in cases} - permitted)
        if refused and not all_cases:
            raise ValueError(f"selected groups are not assigned to {plan_split}: {refused}")
        if all_cases:
            cases = [case for case in cases if case["split_group"] in permitted]
            if not cases:
                raise ValueError(f"split plan has no cases assigned to {plan_split}")
    references, templates = load_source_references()
    source_paths = [Path(__file__).resolve(), ROOT / "scripts" / "generate_scene_demonstrations.py",
                    ROOT / "raccoon_memory" / "scene_motion_sim.py", Path(scene_features.__file__).resolve(),
                    Path(task_catalog.__file__).resolve(), Path(motion_policy.__file__).resolve(),
                    Path(sim_data.__file__).resolve(), Path(unstack_sim_data.__file__).resolve(),
                    ROOT / "raccoon_memory" / "motion_sim.py", ROOT / "scripts" / "operate_motion_sim.py",
                    ROOT / "raccoon_control" / "kinematics.py", ROOT / "raccoon_control" / "validation.py",
                    ROOT / "raccoon_control" / "errors.py"]
    source_data = {str(path): path.read_bytes() for path in source_paths}
    requested.mkdir(parents=True, exist_ok=False)
    source_dir = requested / "source"
    source_dir.mkdir()
    for name, raw in source_data.items():
        with (source_dir / Path(name).name).open("xb") as stream:
            stream.write(raw)
    if plan_raw is not None:
        with (requested / "split_plan.json").open("xb") as stream:
            stream.write(plan_raw)
    results, failures = Counter(), Counter()
    with (requested / "attempts.jsonl").open("x", encoding="utf-8", newline="\n") as attempts, \
         (requested / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as episodes:
        for case in cases:
            for seed in seeds:
                attempt, episode = generate_attempt(case, seed, split, references, templates,
                                                    randomize=randomize, split_plan_sha256=plan_hash,
                                                    grasp_tcp_base_mm=grasp_tcp_base_mm,
                                                    placement_pitch_deg=placement_pitch_deg)
                attempts.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                attempts.flush()
                os.fsync(attempts.fileno())
                results["attempt_count"] += 1
                if episode is None:
                    results["failure_count"] += 1
                    failures[attempt["failure_reason"]] += 1
                else:
                    episodes.write(json.dumps(episode, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                    episodes.flush()
                    os.fsync(episodes.fileno())
                    results["success_count"] += 1
    code_unchanged = all(Path(name).read_bytes() == raw for name, raw in source_data.items())
    artifacts = {name: {"sha256": _sha((requested / name).read_bytes()), "bytes": (requested / name).stat().st_size}
                 for name in ("attempts.jsonl", "episodes.jsonl")}
    report = {"schema": "raccoon-scene-demonstration-report.v1", "source": SOURCE, "teacher": TEACHER,
              "teacher_variant": teacher_variant, "grasp_tcp_base_mm": float(grasp_tcp_base_mm),
              "placement_pitch_deg": float(placement_pitch_deg),
              "max_transit_rotation_per_command_deg": MAX_TRANSIT_ROTATION_DEG,
              "source_replay": False, "new_llm_decisions": False, "simulated": True, "actual_hardware": False,
              "split": split, "seeds": seeds, "fixed_split_ranges_inclusive": SPLIT_RANGES,
              "case_ids": [case["case_id"] for case in cases], "id_map": list(id_map),
              "requested_case_count": requested_case_count,
              "excluded_cases_outside_split_plan": requested_case_count - len(cases),
              "split_groups": sorted({case["split_group"] for case in cases}), "split_plan_sha256": plan_hash,
              "same_split_plan_required_across_train_and_validation": True,
              "pilot_uses_training_groups_only": True,
              "pilot_episodes_excluded_from_model_train_validation_and_test": split == "pilot",
              "randomize": randomize, "attempt_count": results["attempt_count"],
              "success_count": results["success_count"], "failure_count": results["failure_count"],
              "failure_reasons": dict(failures), "source_references": references,
              "code_sha256": {name: _sha(raw) for name, raw in source_data.items()},
              "runtime_versions": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "torch")},
              "source_code_unchanged_during_generation": code_unchanged, "artifacts": artifacts,
              "note": "New programmatic teacher variant; group separation is not independent teacher-origin separation. Detailed privileged information remains in attempts, not policy features."}
    with (requested / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
