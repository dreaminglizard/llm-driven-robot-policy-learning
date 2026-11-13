"""On-policy REINFORCE for the shared scene policy in simulated physics.

Only registered scene/goal and the existing arm/gripper features enter the
student. Physical object state and contacts are used exclusively for reward
and independent terminal assessment. There is no teacher action or fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random
from typing import Any

import torch
from torch.distributions import Categorical, Normal
import torch.nn.functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_features, sim_data, task_catalog
from scripts.evaluate_scene_policy import validate_assessment

GAMMA = 0.99
STD_DEG = 0.5
COMMAND_SECONDS = 1.0
STEP_COST = -0.02
SUCCESS_REWARD = 10.0
FAILURE_REWARD = -1.0
INVALID_TARGET_REWARD = -2.0
OTHER_DISPLACEMENT_REWARD = -1.0
OBSERVATION_FIELDS = ("schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
                      "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not (type(value) is float and not math.isfinite(value))


def safe_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return safe_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def valid_target(values: Any) -> bool:
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and lo <= v <= hi
                    for v, (lo, hi) in zip(values, JOINT_LIMITS)))


def student_observation(sim, gripper_command: str) -> dict:
    observation = sim_data.sim_policy_observation(sim, gripper_command)
    return {key: observation[key] for key in OBSERVATION_FIELDS}


def reward_state(raw: dict, task: dict, anchors: dict[int, list[float]]) -> dict:
    """Measure potential and other-object drift from privileged simulation state."""
    if not finite_tree(raw):
        raise ValueError("non-finite simulation observation")
    privileged = raw["privileged_simulation_only"]
    objects = {item["id"]: item for item in privileged["objects"]}
    if len(objects) != len(privileged["objects"]) or set(objects) != set(anchors):
        raise ValueError("active objects do not match initial anchors")
    obj_id = task["object_id"]
    obj = objects[obj_id]
    xyz = [float(x) for x in obj["position_m"]]
    tcp = [float(x) / 1000.0 for x in raw["tcp_mm"]]
    slot_xy = scene_features.NOMINAL_SLOT_XY_MM[task["destination_slot"] - 1]
    goal = [slot_xy[0] / 1000.0, slot_xy[1] / 1000.0, 0.02 * (task["destination_level"] - 0.5)]
    tcp_distance, goal_distance = math.dist(tcp, xyz), math.dist(xyz, goal)
    contacts = [c for c in privileged["contacts"] if f"object_{obj_id}_geom" in (c["geom1"], c["geom2"])]
    pads = {name: any(name in (c["geom1"], c["geom2"]) and c["normal_force_n"] > 0 for c in contacts)
            for name in ("finger_left_pad", "finger_right_pad")}
    potential = -0.25 * min(tcp_distance / 0.30, 1.0) - 0.25 * min(goal_distance / 0.30, 1.0)
    potential += 0.10 if all(pads.values()) else 0.0
    other = {}
    for object_id, item in objects.items():
        if object_id == obj_id:
            continue
        point, anchor = item["position_m"], anchors[object_id]
        xy = math.dist(point[:2], anchor[:2])
        height = abs(point[2] - anchor[2])
        other[str(object_id)] = {"xy_m": xy, "height_m": height,
                                 "displaced": xy > 0.008 or height > 0.003}
    result = {"potential": potential, "tcp_object_distance_m": tcp_distance,
              "object_goal_distance_m": goal_distance, "goal_nominal_m": goal,
              "target_object_position_m": xyz, "pad_contacts": pads,
              "other_object_displacement": other, "privileged_simulation_only": True}
    if not finite_tree(result):
        raise ValueError("non-finite reward state")
    return result


def reward_components(before: dict, after: dict | None, *, gamma: float,
                      terminal: float | None, other_penalty: float = 0.0) -> dict:
    if terminal is None and after is None:
        raise ValueError("a nonterminal transition needs an after state")
    if other_penalty not in (0.0, OTHER_DISPLACEMENT_REWARD):
        raise ValueError("invalid other-object penalty")
    shaping = (0.0 if terminal is not None else gamma * after["potential"]) - before["potential"]
    total = STEP_COST + shaping + (terminal or 0.0) + other_penalty
    result = {"decision_cost": STEP_COST, "potential_before": before["potential"],
              "potential_after": None if terminal is not None else after["potential"],
              "shaping": shaping, "terminal": terminal, "other_object_penalty": other_penalty,
              "total": total}
    if not finite_tree(result):
        raise ValueError("non-finite reward")
    return result


def sample_policy(logits: torch.Tensor, mean: torch.Tensor, *, std_deg: float,
                  generator: torch.Generator) -> dict:
    if logits.shape != (4,) or mean.shape != (4,) or not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all()):
        raise ValueError("non-finite or invalid model output")
    if not math.isfinite(std_deg) or std_deg <= 0:
        raise ValueError("invalid joint standard deviation")
    categorical = Categorical(logits=logits)
    action = int(torch.multinomial(categorical.probs, 1, generator=generator).item())
    action_logp = categorical.log_prob(torch.tensor(action, device=logits.device))
    sampled, joint_logp = None, None
    if action == motion_policy.MOVE:
        std = torch.full_like(mean, std_deg / motion_policy.JOINT_SCALE_DEG)
        sampled = mean + std * torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        joint_logp = Normal(mean, std).log_prob(sampled).sum()
    total = action_logp if joint_logp is None else action_logp + joint_logp
    if not bool(torch.isfinite(total)) or (sampled is not None and not bool(torch.isfinite(sampled).all())):
        raise ValueError("non-finite sampled action or log probability")
    return {"action_index": action, "action_log_probability": float(action_logp),
            "joint_log_probability": None if joint_logp is None else float(joint_logp),
            "policy_log_probability": float(total), "action_entropy": float(categorical.entropy()),
            "sampled_joint_normalized": None if sampled is None else sampled.cpu().tolist()}


def _elapsed_steps(sim, before_time: float) -> int:
    dt = float(sim.model.opt.timestep)
    elapsed = float(sim.data.time) - before_time
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(elapsed) or elapsed < -1e-9:
        raise ValueError("invalid simulated elapsed time")
    steps = round(elapsed / dt)
    if not math.isclose(elapsed, steps * dt, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("simulation time is not an integer number of physics steps")
    return steps


def _cancelled(cancel) -> bool:
    return cancel is not None and cancel.is_set()


@torch.no_grad()
def rollout(model, sim, registered_scene, task, *, generator: torch.Generator,
            gamma: float = GAMMA, std_deg: float = STD_DEG, max_decisions: int = 24,
            device: torch.device = torch.device("cpu"), cancel=None) -> dict:
    """Collect one physical episode; incomplete/error trajectories cannot train."""
    if not (math.isfinite(gamma) and 0 < gamma <= 1 and math.isfinite(std_deg) and std_deg > 0
            and type(max_decisions) is int and max_decisions > 0):
        raise ValueError("invalid rollout parameters")
    before, goal = scene_features.transfer_task(registered_scene, task)
    episode = {"registered_scene": [list(stack) for stack in before], "task": goal,
               "teacher_assistance_count": 0, "simulated": True, "actual_hardware": False,
               "decisions": [], "termination": None, "success": None, "physical_success": None,
               "assessment": None, "physical_command_count": 0, "physical_command_attempts": 0,
               "command_physics_steps": 0, "assessment_physics_steps": 0,
               "other_object_displacement_penalty_applied": False, "done_before_any_command": False}
    started = False
    try:
        model.eval()
        raw = sim.observation()
        if task_catalog.scene(raw["registered_layout"]) != before:
            raise ValueError("registered scene differs from simulator")
        episode["initial_state"] = sim.save_state()
        if not finite_tree(episode["initial_state"]):
            raise ValueError("non-finite initial simulation state")
        anchors = {obj["id"]: list(obj["position_m"]) for obj in raw["privileged_simulation_only"]["objects"]}
        if set(anchors) != {obj for stack in before for obj in stack}:
            raise ValueError("initial object IDs differ from registered scene")
        episode["initial_anchor_positions_m"] = anchors
        finger = [float(sim.data.ctrl[i]) for i in sim._finger_actuators]
        if (len(finger) != 2 or any(not math.isfinite(x) or not 0 <= x <= 0.025 for x in finger)
                or abs(finger[0] - finger[1]) > 1e-9):
            raise ValueError("invalid finger servo targets")
        opening = sum(finger)
        last_target = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
        if not valid_target(last_target) or raw["gripper_opening_m"] < .025 or opening < .025:
            raise ValueError("new transfer requires valid arm targets and an open gripper")
        if _cancelled(cancel):
            episode["termination"] = "cancelled"
        else:
            sim.begin_transfer(before, goal["source_slot"], goal["destination_slot"])
            started = True
            history, gripper = [], "open"
            for index in range(max_decisions):
                if _cancelled(cancel):
                    episode["termination"] = "cancelled"
                    break
                decision = {"index": index, "command_applied": False}
                episode["decisions"].append(decision)
                try:
                    before_reward = reward_state(sim.observation(), goal, anchors)
                    obs = student_observation(sim, gripper)
                    vector = scene_features.encode_scene_features(obs, before, goal)
                    history.append(vector)
                    window, mask = scene_features.history_window(history, scene_features.HISTORY_LENGTH)
                    logits, mean = model(torch.tensor([window], dtype=torch.float32, device=device),
                                         torch.tensor([mask], dtype=torch.bool, device=device))
                    decision.update(policy_observation=obs, policy_features=vector, window=window, mask=mask,
                                    reward_state_before=before_reward,
                                    model_output={"action_logits": safe_json(logits), "joint_mean_normalized": safe_json(mean)})
                    if logits.shape != (1, 4) or mean.shape != (1, 4):
                        raise ValueError("invalid model head shapes")
                    if not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all()):
                        raise ValueError("non-finite model output")
                    if _cancelled(cancel):
                        episode["termination"] = "cancelled"
                        decision["result"] = "cancelled_before_sampling"
                        break
                    sampled = sample_policy(logits[0], mean[0], std_deg=std_deg, generator=generator)
                    decision.update(sampled)
                    choice = sampled["action_index"]
                    decision["action"] = motion_policy.ACTIONS[choice]
                    if choice == motion_policy.MOVE:
                        mean_deg = [v * motion_policy.JOINT_SCALE_DEG for v in mean[0].cpu().tolist()]
                        target = [v * motion_policy.JOINT_SCALE_DEG for v in sampled["sampled_joint_normalized"]]
                        decision.update(mean_joint_deg=mean_deg, mean_out_of_range=not valid_target(mean_deg),
                                        sampled_joint_deg=target)
                    else:
                        target = list(last_target)
                    if _cancelled(cancel):
                        episode["termination"] = "cancelled"
                        decision["result"] = "cancelled_before_command"
                        break
                    if choice == motion_policy.DONE:
                        episode["done_before_any_command"] = episode["physical_command_count"] == 0
                        decision["result"] = "done_no_command"
                        assessment_start = float(sim.data.time)
                        try:
                            assessed = sim.evaluate_transfer(before, goal["source_slot"], goal["destination_slot"],
                                                             stable_seconds=1.0)
                        finally:
                            episode["assessment_physics_steps"] += _elapsed_steps(sim, assessment_start)
                        validate_assessment(assessed, before, goal)
                        if assessed.get("sample_count") != episode["assessment_physics_steps"] + 1:
                            raise ValueError("assessment sample count differs from actual physics steps")
                        episode.update(assessment=assessed, physical_success=assessed["success"])
                        if _cancelled(cancel):
                            episode["termination"] = "cancelled"
                            break
                        episode.update(termination="model_done", success=assessed["success"])
                        decision["reward"] = reward_components(before_reward, None, gamma=gamma,
                                                               terminal=SUCCESS_REWARD if assessed["success"] else FAILURE_REWARD)
                        break
                    if choice == motion_policy.MOVE and not valid_target(target):
                        episode.update(termination="out_of_range_target", success=False)
                        decision["result"] = "invalid_sample_no_command"
                        decision["reward"] = reward_components(before_reward, None, gamma=gamma, terminal=INVALID_TARGET_REWARD)
                        break
                    next_gripper = gripper if choice == motion_policy.MOVE else ("open" if choice == motion_policy.OPEN else "closed")
                    next_opening = opening if choice == motion_policy.MOVE else (0.04 if next_gripper == "open" else 0.0)
                    decision["command"] = {"joints_deg": target, "gripper_opening_m": next_opening,
                                           "seconds": COMMAND_SECONDS, "source": "shared_student_policy"}
                    episode["physical_command_attempts"] += 1
                    start_time = float(sim.data.time)
                    try:
                        sim.command(target, next_opening, seconds=COMMAND_SECONDS)
                    finally:
                        steps = _elapsed_steps(sim, start_time)
                        episode["command_physics_steps"] += steps
                        decision["command_physics_steps"] = steps
                    if steps < 1:
                        raise ValueError("simulation command advanced no physics steps")
                    decision["command_applied"] = True
                    decision["result"] = "command_applied"
                    episode["physical_command_count"] += 1
                    last_target, opening, gripper = target, next_opening, next_gripper
                    after_reward = reward_state(sim.observation(), goal, anchors)
                    decision["reward_state_after"] = after_reward
                    displaced = any(v["displaced"] for v in after_reward["other_object_displacement"].values())
                    penalty = 0.0
                    if displaced and not episode["other_object_displacement_penalty_applied"]:
                        penalty = OTHER_DISPLACEMENT_REWARD
                        episode["other_object_displacement_penalty_applied"] = True
                    if _cancelled(cancel):
                        episode["termination"] = "cancelled"
                        break
                    if index + 1 == max_decisions:
                        episode.update(termination="decision_limit", success=False)
                        decision["reward"] = reward_components(before_reward, None, gamma=gamma,
                                                               terminal=FAILURE_REWARD, other_penalty=penalty)
                        break
                    decision["reward"] = reward_components(before_reward, after_reward, gamma=gamma,
                                                           terminal=None, other_penalty=penalty)
                except Exception as exc:
                    decision["result"] = "error"
                    decision["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    episode.update(termination="error", success=None,
                                   error={"type": type(exc).__name__, "message": str(exc)})
                    break
        if episode["termination"] is None:
            raise RuntimeError("rollout ended without a termination reason")
        if started and episode["success"] is not True:
            sim.mark_layout_uncertain(episode["termination"])
        episode["final_state"] = sim.save_state()
    except Exception as exc:
        episode.update(termination="error", success=None, error={"type": type(exc).__name__, "message": str(exc)})
        if started:
            try:
                sim.mark_layout_uncertain("rollout_error")
                episode["final_state"] = sim.save_state()
            except Exception as state_exc:
                episode["final_state_error"] = {"type": type(state_exc).__name__, "message": str(state_exc)}
    if episode["termination"] in ("model_done", "decision_limit", "out_of_range_target"):
        value = 0.0
        for decision in reversed(episode["decisions"]):
            value = decision["reward"]["total"] + gamma * value
            decision["return_to_go"] = value
        episode["discounted_return"] = value
        episode["undiscounted_reward"] = sum(d["reward"]["total"] for d in episode["decisions"])
    return safe_json(episode)


def reinforce_loss(model, episodes: list[dict], *, gamma: float = GAMMA, std_deg: float = STD_DEG,
                   device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, dict]:
    """Recompute on-policy likelihoods and use discounted reward-to-go."""
    if not episodes or not math.isfinite(gamma) or not 0 < gamma <= 1 or not math.isfinite(std_deg) or std_deg <= 0:
        raise ValueError("valid complete episodes, discount and standard deviation are required")
    terms, action_terms, joint_terms = [], 0, 0
    for episode in episodes:
        if episode.get("termination") not in ("model_done", "decision_limit", "out_of_range_target"):
            raise ValueError("incomplete or invalid trajectory cannot be optimized")
        for index, decision in enumerate(episode["decisions"]):
            if decision.get("index") != index or "return_to_go" not in decision:
                raise ValueError("missing or inconsistent return-to-go")
            windows = torch.tensor([decision["window"]], dtype=torch.float32, device=device)
            masks = torch.tensor([decision["mask"]], dtype=torch.bool, device=device)
            logits, mean = model(windows, masks)
            if logits.shape != (1, 4) or mean.shape != (1, 4) or not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all()):
                raise ValueError("non-finite or invalid recomputed model output")
            recorded = decision["model_output"]
            if (not torch.equal(logits.detach().cpu(), torch.tensor(recorded["action_logits"], dtype=torch.float32))
                    or not torch.equal(mean.detach().cpu(), torch.tensor(recorded["joint_mean_normalized"], dtype=torch.float32))):
                raise ValueError("recomputed distribution parameters differ from collection policy")
            action = torch.tensor(decision["action_index"], dtype=torch.long, device=device)
            action_logp = Categorical(logits=logits[0]).log_prob(action)
            logp, action_terms = action_logp, action_terms + 1
            joint_logp = None
            if decision["action_index"] == motion_policy.MOVE:
                sample = torch.tensor(decision["sampled_joint_normalized"], dtype=torch.float32, device=device)
                if sample.shape != (4,) or not bool(torch.isfinite(sample).all()):
                    raise ValueError("invalid recorded joint sample")
                joint_logp = Normal(mean[0], torch.full_like(mean[0], std_deg / motion_policy.JOINT_SCALE_DEG)).log_prob(sample).sum()
                logp, joint_terms = logp + joint_logp, joint_terms + 1
            if (not bool(torch.isfinite(logp))
                    or not math.isclose(float(action_logp.detach()), decision["action_log_probability"], rel_tol=1e-5, abs_tol=2e-4)
                    or (joint_logp is not None and not math.isclose(float(joint_logp.detach()), decision["joint_log_probability"], rel_tol=1e-5, abs_tol=2e-4))
                    or not math.isclose(float(logp.detach()), decision["policy_log_probability"], rel_tol=1e-5, abs_tol=2e-4)):
                raise ValueError("recomputed likelihood differs from collection policy")
            weight = gamma ** index * decision["return_to_go"]
            if not math.isfinite(weight):
                raise ValueError("non-finite return")
            terms.append(logp * weight)
    if not terms:
        raise ValueError("no likelihood-bearing decisions")
    loss = -torch.stack(terms).sum() / len(episodes)
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite policy-gradient loss")
    return loss, {"action_likelihood_terms": action_terms, "joint_likelihood_terms": joint_terms}


def derived_seed(seed: int, label: str) -> int:
    if type(seed) is not int or seed < 0 or not isinstance(label, str) or not label:
        raise ValueError("a nonnegative seed and nonempty stream label are required")
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8], "big") % (2**63)


def make_episode_schedule(plan: dict, seed: int) -> dict:
    """Fix group-balanced tasks, ID permutations and training physics seeds."""
    groups = plan.get("train_groups")
    if not isinstance(groups, list) or len(groups) != 8 or len(set(groups)) != 8:
        raise ValueError("the fixed comparison requires eight training groups")
    if set(groups) & (set(plan["validation_groups"]) | set(plan["test_groups"])):
        raise ValueError("training and held-out groups overlap")
    grouped = defaultdict(list)
    for case in task_catalog.one_step_cases():
        group = scene_policy_group(case["canonical_scene_before"], case["task"])
        if group in groups:
            grouped[group].append(case)
    if sum(map(len, grouped.values())) != 156 or any(len(grouped[g]) not in (12, 24) for g in groups):
        raise ValueError("training task catalogue differs from fixed split")
    per_group, id_orders = {}, {}
    for group in groups:
        items = list(grouped[group])
        rng = random.Random(derived_seed(seed, f"case:{group}"))
        rng.shuffle(items)
        sequence = list(items)
        if len(sequence) < 20:
            second = list(items)
            rng.shuffle(second)
            sequence.extend(second)
        per_group[group] = sequence[:20]
        ids = [1, 2, 3, 4] * 5
        random.Random(derived_seed(seed, f"target_ids:{group}")).shuffle(ids)
        id_orders[group] = ids
    physical_seeds = list(range(1000, 1080)) * 2
    random.Random(derived_seed(seed, "physics_seed_order")).shuffle(physical_seeds)
    map_rng = random.Random(derived_seed(seed, "identity_mapping"))
    episodes = []
    for update in range(20):
        for group in groups:
            case = per_group[group][update]
            original_target, desired_target = case["moved_object_id"], id_orders[group][update]
            old = [value for value in range(1, 5) if value != original_target]
            new = [value for value in range(1, 5) if value != desired_target]
            map_rng.shuffle(new)
            mapping = dict(zip(old, new))
            mapping[original_target] = desired_target
            layout = [[mapping[obj] for obj in stack] for stack in case["canonical_scene_before"]]
            before, task = scene_features.transfer_task(layout, case["task"])
            if task["object_id"] != desired_target or scene_policy_group(before, task) != group:
                raise ValueError("identity mapping changed the task group or target")
            episodes.append({"update": update + 1, "episode_in_update": len(episodes) % len(groups),
                             "case_id": case["case_id"], "split_group": group, "kind": case["kind"],
                             "registered_scene": [list(stack) for stack in before], "task": task,
                             "source_to_new_object_id": {str(k): mapping[k] for k in sorted(mapping)},
                             "physics_seed": physical_seeds[len(episodes)]})
    if (len(episodes) != 160 or len({e["case_id"] for e in episodes}) != 136
            or Counter(e["physics_seed"] for e in episodes) != Counter({s: 2 for s in range(1000, 1080)})
            or any(Counter(e["task"]["object_id"] for e in episodes if e["split_group"] == group)
                   != Counter({1: 5, 2: 5, 3: 5, 4: 5}) for group in groups)):
        raise ValueError("experience schedule does not satisfy the fixed coverage")
    return {"root_seed": seed, "group_order": groups, "episodes": episodes,
            "physical_seed_range": [1000, 1079], "unique_canonical_cases": 136,
            "stream_labels": ["case:<group>", "target_ids:<group>", "identity_mapping", "physics_seed_order"]}


def scene_policy_group(registered_scene, task) -> str:
    # Import here to keep the policy feature/label implementation authoritative.
    from raccoon_memory.scene_policy import split_group_for
    return split_group_for(registered_scene, task)


def make_bc_schedule(event_count: int, seed: int, *, updates: int = 20, batch_size: int = 256) -> list[list[int]]:
    if type(event_count) is not int or event_count < 1 or type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer event, update and batch counts required")
    generator = torch.Generator(device="cpu").manual_seed(derived_seed(seed, "bc_minibatches"))
    order, position, batches = [], 0, []
    for _ in range(updates):
        batch = []
        while len(batch) < batch_size:
            if position == len(order):
                order, position = torch.randperm(event_count, generator=generator).tolist(), 0
            take = min(batch_size - len(batch), len(order) - position)
            batch.extend(order[position:position + take])
            position += take
        batches.append(batch)
    return batches


def _group_gradient_norms(named_parameters, gradients) -> dict:
    totals = {"encoder": 0.0, "action_head": 0.0, "joint_head": 0.0}
    for (name, _), gradient in zip(named_parameters, gradients):
        group = name.split(".", 1)[0]
        if group not in totals:
            raise ValueError(f"unexpected policy parameter group: {name}")
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("non-finite gradient")
            totals[group] += float(gradient.detach().double().square().sum())
    return {key: math.sqrt(value) for key, value in totals.items()}


def optimizer_update(model, optimizer, bc_batch: dict, *, episodes: list[dict] | None,
                     gamma: float = GAMMA, std_deg: float = STD_DEG, rl_weight: float = 0.001,
                     device: torch.device = torch.device("cpu")) -> dict:
    """One paired BC or BC+RL update with separately measured gradient terms."""
    if not math.isfinite(rl_weight) or rl_weight < 0:
        raise ValueError("RL loss weight must be finite and nonnegative")
    model.train()
    windows, masks = bc_batch["windows"].to(device), bc_batch["masks"].to(device)
    labels, targets = bc_batch["actions"].to(device), bc_batch["absolute_targets"].to(device)
    logits, joints = model(windows, masks)
    ce = F.cross_entropy(logits, labels)
    moves = labels == motion_policy.MOVE
    mse = F.mse_loss(joints[moves], targets[moves]) if bool(moves.any()) else joints.sum() * 0.0
    bc_loss = ce + 100.0 * mse
    if not bool(torch.isfinite(bc_loss)):
        raise ValueError("non-finite BC loss")
    named = list(model.named_parameters())
    params = [p for _, p in named]
    bc_grads = torch.autograd.grad(bc_loss, params, retain_graph=True, allow_unused=True)
    bc_norms = _group_gradient_norms(named, bc_grads)
    rl_loss, rl_norms, counts = None, None, None
    if episodes is not None:
        if not episodes or not any(e.get("command_physics_steps", 0) > 0 for e in episodes):
            raise ValueError("RL update requires completed episodes with actual command physics steps")
        rl_loss, counts = reinforce_loss(model, episodes, gamma=gamma, std_deg=std_deg, device=device)
        rl_grads = torch.autograd.grad(rl_loss, params, retain_graph=True, allow_unused=True)
        rl_norms = _group_gradient_norms(named, rl_grads)
    total = bc_loss if rl_loss is None else bc_loss + rl_weight * rl_loss
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite combined loss")
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    total_norms = _group_gradient_norms(named, [p.grad for p in params])
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise ValueError("optimizer produced non-finite weights")
    return {"bc_loss": float(bc_loss.detach()), "bc_action_cross_entropy": float(ce.detach()),
            "bc_move_mse_normalized": float(mse.detach()), "bc_move_count": int(moves.sum()),
            "rl_loss": None if rl_loss is None else float(rl_loss.detach()), "rl_weight": rl_weight,
            "total_loss": float(total.detach()), "bc_gradient_l2": bc_norms, "rl_gradient_l2": rl_norms,
            "weighted_rl_gradient_l2": None if rl_norms is None else {k: rl_weight*v for k, v in rl_norms.items()},
            "total_gradient_l2": total_norms, "likelihood_terms": counts}
