"""On-policy REINFORCE for the simulated recurrent manipulation student.

This module uses MuJoCo only. Object poses and contacts are used by the reward,
never as policy features. No teacher action, waypoint, or phase is read.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical, Normal

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy as policy
from raccoon_memory import sim_data


TASK = dict(sim_data.TASK)
OBSERVATION_FIELDS = (
    "schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
    "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh",
)
COMMAND_SECONDS = 1.0
OPENING_OPEN_M = 0.04
OPENING_CLOSED_M = 0.0
TRAIN_SEEDS = (1000, 1099)
VALIDATION_SEEDS = (1500, 1519)
TEST_SEEDS = (2000, 2019)
SUCCESS_REWARD = 10.0
FAILURE_REWARD = -1.0
INVALID_TARGET_REWARD = -2.0
DECISION_COST = -0.02
REWARD_DESCRIPTION = (
    "Each decision costs 0.02. Potential phi = -2 * distance(TCP, object 1) "
    "- 5 * distance(object 1, [object 3 x,y,z+0.02m]) + 0.25 when both "
    "finger pads contact object 1 with positive normal force. The shaping term "
    "is gamma*phi(after)-phi(before), with terminal phi=0. At done or the "
    "decision limit, the independent one-second physical stack assessment adds "
    "+10 for success or -1 for failure. Invalid targets add -2 and end the episode. "
    "For move, both the joint-head mean and sampled target must be in range. "
    "Reward-only object poses and contacts are not policy inputs."
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_train_seeds(first: int, last: int) -> list[int]:
    if (type(first) is not int or type(last) is not int or first > last
            or first < TRAIN_SEEDS[0] or last > TRAIN_SEEDS[1]):
        raise ValueError("training environment seeds must be within 1000..1099")
    return list(range(first, last + 1))


def load_bc_checkpoint(path: Path, device: torch.device):
    raw = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("BC checkpoint needs a state_dict")
    config = checkpoint.get("config", {}).get("model", {})
    scales = config.get("feature_scale")
    if (type(config.get("hidden_size")) is not int or config["hidden_size"] < 1
            or config.get("feature_names") != list(policy.FEATURE_NAMES)
            or config.get("actions") != list(policy.ACTIONS)
            or not isinstance(scales, dict)
            or scales.get("joint_degrees") != policy.JOINT_SCALE_DEG
            or scales.get("slot_and_level") != policy.TASK_SCALE
            or config.get("output_delta_scale_degrees") != policy.JOINT_SCALE_DEG
            or config.get("output_parameterization") != "absolute"):
        raise ValueError("BC checkpoint is not a compatible absolute-angle policy")
    model = policy.MotionPolicy(config["hidden_size"], output_parameterization="absolute").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("BC checkpoint contains non-finite weights")
    return model, checkpoint, sha256(raw)


def valid_target(values: Any) -> bool:
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                    for value, (low, high) in zip(values, JOINT_LIMITS)))


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return not (type(value) is float and not math.isfinite(value))


def safe_json(value: Any) -> Any:
    """Keep invalid output visible without writing non-standard JSON numbers."""
    if isinstance(value, torch.Tensor):
        return safe_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def student_observation(sim, gripper_command: str) -> dict:
    raw = sim_data.sim_policy_observation(sim, gripper_command)
    return {key: raw[key] for key in OBSERVATION_FIELDS}


def reward_state(raw: dict) -> dict:
    """Compute an auditable potential using only current physical state."""
    if not finite_tree(raw):
        raise ValueError("non-finite simulator observation")
    objects = {obj["id"]: obj for obj in raw["objects"]}
    obj, support = objects[1], objects[3]
    xyz, support_xyz = obj["position_m"], support["position_m"]
    tcp = [float(value) / 1000 for value in raw["tcp_mm"]]
    goal = [support_xyz[0], support_xyz[1], support_xyz[2] + 0.02]
    distance_to_object = math.dist(tcp, xyz)
    distance_to_goal = math.dist(xyz, goal)
    relevant_contacts = [c for c in raw["contacts"]
                         if "object_1_geom" in (c["geom1"], c["geom2"])]
    pads = {name: any(name in (c["geom1"], c["geom2"]) and c["normal_force_n"] > 0
                      for c in relevant_contacts)
            for name in ("finger_left_pad", "finger_right_pad")}
    both = all(pads.values())
    potential = -2.0 * distance_to_object - 5.0 * distance_to_goal + (0.25 if both else 0.0)
    return {"potential": potential, "tcp_m": tcp, "object_1": obj, "support_object_3": support,
            "target_relative_to_support_m": goal, "tcp_object_distance_m": distance_to_object,
            "object_goal_distance_m": distance_to_goal, "finger_contacts": pads,
            "object_1_contacts": relevant_contacts, "gripper_opening_m": raw["gripper_opening_m"],
            "time_seconds": raw["time_seconds"], "privileged_simulation_only": True}


def assess_stack(sim) -> dict:
    """Require the complete shared one-second MuJoCo stack criterion."""
    result = sim_data.evaluate_stack_success(sim)
    if not isinstance(result, dict) or not finite_tree(result) or type(result.get("success")) is not bool:
        raise ValueError("invalid physical stack assessment")
    base = result.get("one_second_stability")
    stack = result.get("additional_stack_checks")
    checks = base.get("checks") if isinstance(base, dict) else None
    required_base = {"position_xy", "height", "upright", "linear_speed", "angular_speed", "gripper_open"}
    required_stack = {"object_to_support_xy", "object_to_support_height", "supporting_cylinder_upright",
                      "supporting_cylinder_linear_speed", "supporting_cylinder_angular_speed",
                      "actual_object_support_contact", "stacked_cylinder_free_of_robot_contact",
                      "supporting_cylinder_free_of_robot_contact"}
    duration = base.get("stable_interval_seconds") if isinstance(base, dict) else None
    if (not isinstance(base, dict) or base.get("source") != "mujoco_simulation"
            or type(base.get("success")) is not bool
            or not isinstance(checks, dict) or not required_base.issubset(checks)
            or any(type(value) is not bool for value in checks.values())
            or not isinstance(stack, dict) or not required_stack.issubset(stack)
            or any(type(value) is not bool for value in stack.values())
            or type(duration) not in (int, float) or not math.isfinite(duration) or duration < 1.0 - 1e-9
            or type(result.get("sample_count")) is not int or result["sample_count"] < 2
            or base["success"] != all(checks.values())
            or result["success"] != (base["success"] and all(stack.values()))):
        raise ValueError("incomplete or inconsistent physical stack assessment")
    return result


def fingerprint_state(state: dict[str, torch.Tensor]) -> dict:
    """Hash tensor names, types, dimensions and exact contiguous bytes."""
    groups = {"all": hashlib.sha256(), "gru": hashlib.sha256(),
              "action_head": hashlib.sha256(), "joint_head": hashlib.sha256()}
    for name, tensor in sorted(state.items()):
        data = tensor.detach().cpu().contiguous()
        header = f"{name}|{data.dtype}|{tuple(data.shape)}\n".encode("utf-8")
        content = data.numpy().tobytes()
        group = "joint_head" if name.startswith("delta_head.") else name.split(".", 1)[0]
        for key in ("all", group):
            groups[key].update(header)
            groups[key].update(content)
    return {name: digest.hexdigest() for name, digest in groups.items()}


def weight_changes(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict:
    result = {}
    for label, prefix in (("gru", "gru."), ("action_head", "action_head."), ("joint_head", "delta_head.")):
        differences = [after[name].detach().cpu().double() - tensor.detach().cpu().double()
                       for name, tensor in before.items() if name.startswith(prefix)]
        square_sum = sum(float((value * value).sum()) for value in differences)
        result[label] = {"l2": math.sqrt(square_sum),
                         "max_abs": max(float(value.abs().max()) for value in differences)}
    return result


def _as_error(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def _finish_reward(decision: dict, before: dict, after: dict | None, gamma: float,
                   terminal_reward: float | None) -> None:
    terminal = terminal_reward is not None
    shaping = (0.0 if terminal else gamma * after["potential"]) - before["potential"]
    value = DECISION_COST + shaping + (terminal_reward if terminal else 0.0)
    decision["reward"] = {"decision_cost": DECISION_COST, "potential_before": before["potential"],
                          "potential_after": None if terminal else after["potential"],
                          "shaping": shaping, "terminal": terminal_reward, "total": value}


@torch.no_grad()
def rollout(model: policy.MotionPolicy, sim, seed: int, parameters: dict, *, gamma: float,
            std_deg: float, max_decisions: int, device: torch.device) -> dict:
    """Collect one trajectory; all commands are the sampled student's commands."""
    model.eval()
    episode = {"seed": seed, "randomized_parameters": parameters, "task": dict(TASK),
               "teacher_assistance_count": 0, "simulated": True, "actual_hardware": False,
               "decisions": [], "termination": None, "physical_success": None,
               "physical_command_count": 0, "assessment": None}
    hidden, gripper, opening = None, "open", OPENING_OPEN_M
    try:
        obs = student_observation(sim, gripper)
        last_target = list(obs["joints_deg"])
        if not valid_target(last_target):
            raise ValueError("initial joint state is outside joint limits")
        initial_state = sim.save_state()
        if not finite_tree(initial_state):
            raise ValueError("non-finite initial simulation state")
        episode["initial_integration_state"] = initial_state
        episode["initial_time_seconds"] = float(sim.data.time)
    except Exception as exc:
        episode.update(termination="simulation_error", error=_as_error(exc))
        return safe_json(episode)

    for index in range(max_decisions):
        decision = {"index": index, "command_applied": False}
        episode["decisions"].append(decision)
        try:
            before = reward_state(sim.observation())
            obs = student_observation(sim, gripper)
            features = policy.encode_features(obs, TASK)
            decision.update(policy_observation=obs, policy_features=features, reward_state_before=before)
            encoded = torch.tensor(features, dtype=torch.float32, device=device).reshape(1, 1, -1)
            logits, delta, hidden, mean = model.step(encoded, hidden, return_head_output=True)
            decision["model_output"] = {"action_logits": safe_json(logits),
                                        "joint_mean_normalized": safe_json(mean),
                                        "joint_delta_normalized": safe_json(delta)}
            if (logits.shape != (1, 1, len(policy.ACTIONS)) or mean.shape != (1, 1, 4)
                    or not all(bool(torch.isfinite(value).all()) for value in (logits, delta, mean, hidden))):
                episode.update(termination="nonfinite_or_invalid_model_output", physical_success=False)
                decision["result"] = "invalid_output_no_command"
                break
            logits, mean = logits[0, 0], mean[0, 0]
            mean_degrees = [float(value) * policy.JOINT_SCALE_DEG for value in mean.cpu().tolist()]
            decision["mean_joint_deg"] = mean_degrees
            discrete = Categorical(logits=logits)
            choice = int(discrete.sample().item())
            logp_discrete = discrete.log_prob(torch.tensor(choice, device=device))
            decision["action"] = policy.ACTIONS[choice]
            decision["action_index"] = choice
            decision["action_log_probability"] = float(logp_discrete.item())
            decision["joint_log_probability"] = None
            if choice == policy.MOVE:
                distribution = Normal(mean, torch.full_like(mean, std_deg / policy.JOINT_SCALE_DEG))
                sample = distribution.sample()
                if not bool(torch.isfinite(sample).all()):
                    decision["sampled_joint_normalized"] = safe_json(sample)
                    decision["result"] = "nonfinite_sample_no_command"
                    episode.update(termination="nonfinite_joint_sample", physical_success=False)
                    break
                target = [float(value) * policy.JOINT_SCALE_DEG for value in sample.cpu().tolist()]
                joint_logp = distribution.log_prob(sample).sum()
                decision.update(sampled_joint_normalized=sample.cpu().tolist(), sampled_joint_deg=target,
                                joint_log_probability=float(joint_logp.item()))
                logp = logp_discrete + joint_logp
            else:
                target, logp = list(last_target), logp_discrete
            if not bool(torch.isfinite(logp)):
                decision["result"] = "nonfinite_log_probability_no_command"
                episode.update(termination="nonfinite_log_probability", physical_success=False)
                break
            decision["policy_log_probability"] = float(logp.item())
            if choice == policy.MOVE and (not valid_target(mean_degrees) or not valid_target(target)):
                decision["result"] = "out_of_range_target_no_command"
                _finish_reward(decision, before, None, gamma, INVALID_TARGET_REWARD)
                episode.update(termination="out_of_range_target", physical_success=False)
                break
            if choice == policy.DONE:
                criterion = assess_stack(sim)
                episode["assessment"] = criterion
                episode["physical_success"] = criterion["success"]
                episode["termination"] = "model_done"
                decision["reward_state_after"] = reward_state(sim.observation())
                decision["result"] = "done_selected_no_command"
                _finish_reward(decision, before, None, gamma,
                               SUCCESS_REWARD if criterion["success"] else FAILURE_REWARD)
                break
            next_gripper = gripper
            if choice == policy.OPEN:
                next_gripper, opening = "open", OPENING_OPEN_M
            elif choice == policy.CLOSE:
                next_gripper, opening = "closed", OPENING_CLOSED_M
            decision["command"] = {"joints_deg": target, "gripper_opening_m": opening,
                                   "seconds": COMMAND_SECONDS, "source": "stochastic_student"}
            response = sim.command(joints_deg=target, gripper_opening_m=opening, seconds=COMMAND_SECONDS)
            if not finite_tree(response):
                raise ValueError("non-finite simulator command response")
            episode["physical_command_count"] += 1
            decision["command_applied"] = True
            decision["result"] = "command_applied_in_simulation"
            after = reward_state(response)
            decision["reward_state_after"] = after
            last_target, gripper = list(target), next_gripper
            if index == max_decisions - 1:
                criterion = assess_stack(sim)
                episode["assessment"] = criterion
                episode["physical_success"] = criterion["success"]
                episode["termination"] = "decision_limit"
                decision["assessment_state_after"] = reward_state(sim.observation())
                _finish_reward(decision, before, None, gamma,
                               SUCCESS_REWARD if criterion["success"] else FAILURE_REWARD)
            else:
                _finish_reward(decision, before, after, gamma, None)
        except Exception as exc:
            decision["result"] = "simulation_or_policy_error"
            decision["error"] = _as_error(exc)
            episode.update(termination="simulation_error", error=_as_error(exc))
            break
    episode["final_time_seconds"] = float(sim.data.time)
    # Non-finite outputs cannot define a valid stochastic likelihood; halt the
    # run after preserving the failure rather than manufacture a training row.
    if episode["termination"] in ("nonfinite_or_invalid_model_output", "nonfinite_joint_sample",
                                  "nonfinite_log_probability", "simulation_error"):
        return safe_json(episode)
    discounted_return = 0.0
    for decision in reversed(episode["decisions"]):
        discounted_return = decision["reward"]["total"] + gamma * discounted_return
        decision["return_to_go"] = discounted_return
    episode["undiscounted_reward"] = sum(d["reward"]["total"] for d in episode["decisions"])
    episode["discounted_return"] = discounted_return
    return safe_json(episode)


def reinforce_loss(model: policy.MotionPolicy, episodes: list[dict], *, gamma: float,
                   std_deg: float, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Recompute sampled log-probs with recurrence; use unbiased reward-to-go.

    Each time-t score is weighted by gamma**t * G_t. The sum is divided by the
    number of episodes, not decisions. No teacher targets or detached hidden
    state are introduced; gradients flow through each complete GRU history.
    """
    if not episodes:
        raise ValueError("at least one trajectory is required")
    terms, action_terms, joint_terms = [], 0, 0
    for episode in episodes:
        hidden = None
        for index, decision in enumerate(episode["decisions"]):
            if "return_to_go" not in decision:
                raise ValueError("incomplete or invalid trajectory cannot be optimized")
            features = torch.tensor(decision["policy_features"], dtype=torch.float32, device=device).reshape(1, 1, -1)
            logits, _, hidden, mean = model.step(features, hidden, return_head_output=True)
            logits, mean = logits[0, 0], mean[0, 0]
            chosen = torch.tensor(decision["action_index"], dtype=torch.long, device=device)
            logp = Categorical(logits=logits).log_prob(chosen)
            action_terms += 1
            if decision["action_index"] == policy.MOVE:
                sampled = torch.tensor(decision["sampled_joint_normalized"], dtype=torch.float32, device=device)
                logp = logp + Normal(mean, torch.full_like(mean, std_deg / policy.JOINT_SCALE_DEG)).log_prob(sampled).sum()
                joint_terms += 1
            if not torch.isfinite(logp) or not math.isclose(float(logp.detach()), decision["policy_log_probability"],
                                                            rel_tol=1e-5, abs_tol=2e-4):
                raise ValueError("recomputed log probability differs from the collected policy")
            terms.append(logp * (gamma ** index * decision["return_to_go"]))
    if not terms:
        raise ValueError("no likelihood-bearing decisions")
    loss = -torch.stack(terms).sum() / len(episodes)
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite policy-gradient loss")
    return loss, {"action_likelihood_terms": action_terms, "joint_likelihood_terms": joint_terms}
