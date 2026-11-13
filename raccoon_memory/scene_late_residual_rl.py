"""Experience and REINFORCE for the frozen policy's late joint head.

Only a predicted MOVE with at least four valid observations is sampled.
The action kind and earlier joint targets are deterministic.  Object positions
and contacts are used by the existing simulated reward, never as policy input.
This module neither constructs a simulator nor performs an optimizer update.
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Normal

from raccoon_memory import motion_policy, scene_features, scene_motion_rl as common, task_catalog
from scripts.evaluate_scene_policy import validate_assessment

GAMMA = common.GAMMA
STD_DEG = common.STD_DEG
MIN_LATE_HISTORY = 4
COMPLETED_TERMINATIONS = ("model_done", "decision_limit", "out_of_range_target")


def _validate_parameters(gamma: float, std_deg: float) -> None:
    if not (math.isfinite(gamma) and 0 < gamma <= 1 and math.isfinite(std_deg) and std_deg > 0):
        raise ValueError("invalid discount or joint standard deviation")


def validate_trainable_head(model) -> None:
    """Require the existing policy to be frozen and only late-head parameters trainable."""
    if not hasattr(model, "frozen_policy") or not hasattr(model, "late_head"):
        raise ValueError("a frozen policy and late head are required")
    head = list(model.late_head.parameters())
    if not head or any(not p.requires_grad for p in head):
        raise ValueError("all late-head parameters must be trainable")
    if any(p.requires_grad for p in model.frozen_policy.parameters()):
        raise ValueError("the existing policy must remain frozen")
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in head}:
        raise ValueError("only the late head may be trainable")


def _history_length(mask) -> int:
    if (not isinstance(mask, (list, tuple)) or len(mask) != scene_features.HISTORY_LENGTH
            or any(type(item) is not bool for item in mask)):
        raise ValueError("invalid history mask")
    length = sum(mask)
    if length < 1 or list(mask) != [False] * (len(mask) - length) + [True] * length:
        raise ValueError("history mask must be nonempty and left padded")
    return length


@torch.no_grad()
def sample_policy(logits: torch.Tensor, mean: torch.Tensor, mask, *,
                  generator: torch.Generator, std_deg: float = STD_DEG) -> dict:
    """Select an action and sample only a late MOVE in normalized joint units."""
    _validate_parameters(GAMMA, std_deg)
    length = _history_length(mask)
    if (logits.shape != (4,) or mean.shape != (4,)
            or not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all())):
        raise ValueError("invalid model output")
    action = int(torch.argmax(logits).item())
    sampled = action == motion_policy.MOVE and length >= MIN_LATE_HISTORY
    target, logp = None, None
    if sampled:
        std = torch.full_like(mean, std_deg / motion_policy.JOINT_SCALE_DEG)
        target = mean + std * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        logp = Normal(mean, std).log_prob(target).sum()
        if not bool(torch.isfinite(target).all() and torch.isfinite(logp)):
            raise ValueError("non-finite joint sample or likelihood")
    return {"action_index": action, "action": motion_policy.ACTIONS[action],
            "history_length": length, "joint_sampled": sampled,
            "sampled_joint_normalized": None if target is None else target.cpu().tolist(),
            "joint_log_probability": None if logp is None else float(logp.cpu()),
            "action_log_probability": None}


@torch.no_grad()
def rollout(model, sim, registered_scene, task, *, generator: torch.Generator,
            gamma: float = GAMMA, std_deg: float = STD_DEG, max_decisions: int = 24,
            device: torch.device = torch.device("cpu"), cancel=None) -> dict:
    """Collect one student-only simulated transfer; incomplete records cannot train."""
    _validate_parameters(gamma, std_deg)
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("invalid decision limit")
    validate_trainable_head(model)
    before, goal = scene_features.transfer_task(registered_scene, task)
    ep = {"registered_scene": [list(s) for s in before], "task": goal,
          "teacher_assistance_count": 0, "simulated": True, "actual_hardware": False,
          "sampling": {"action": "argmax", "joint": "normal_only_late_move",
                       "std_deg": std_deg, "gamma": gamma},
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
        ep["initial_state"] = sim.save_state()
        if not common.finite_tree(ep["initial_state"]):
            raise ValueError("non-finite initial simulation state")
        objects = raw["privileged_simulation_only"]["objects"]
        anchors = {obj["id"]: list(obj["position_m"]) for obj in objects}
        if len(anchors) != len(objects) or set(anchors) != {obj for s in before for obj in s}:
            raise ValueError("initial object IDs differ from registered scene")
        ep["initial_anchor_positions_m"] = anchors
        finger = [float(sim.data.ctrl[i]) for i in sim._finger_actuators]
        if (len(finger) != 2 or any(not math.isfinite(x) or not 0 <= x <= .025 for x in finger)
                or abs(finger[0] - finger[1]) > 1e-9):
            raise ValueError("invalid finger servo targets")
        opening = sum(finger)
        last_target = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
        if not common.valid_target(last_target) or raw["gripper_opening_m"] < .025 or opening < .025:
            raise ValueError("new transfer requires valid arm targets and an open gripper")
        if common._cancelled(cancel):
            ep["termination"] = "cancelled"
        else:
            sim.begin_transfer(before, goal["source_slot"], goal["destination_slot"])
            started = True
            history, gripper = [], "open"
            for index in range(max_decisions):
                if common._cancelled(cancel):
                    ep["termination"] = "cancelled"
                    break
                d = {"index": index, "command_applied": False}
                ep["decisions"].append(d)
                try:
                    reward_before = common.reward_state(sim.observation(), goal, anchors)
                    obs = common.student_observation(sim, gripper)
                    vector = scene_features.encode_scene_features(obs, before, goal)
                    history.append(vector)
                    window, mask = scene_features.history_window(history, scene_features.HISTORY_LENGTH)
                    logits, mean = model(torch.tensor([window], dtype=torch.float32, device=device),
                                         torch.tensor([mask], dtype=torch.bool, device=device))
                    d.update(policy_observation=obs, policy_features=vector, window=window, mask=mask,
                             reward_state_before=reward_before,
                             model_output={"action_logits": common.safe_json(logits),
                                           "joint_mean_normalized": common.safe_json(mean)})
                    if (logits.shape != (1, 4) or mean.shape != (1, 4)
                            or not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all())):
                        raise ValueError("non-finite or invalid model output")
                    if common._cancelled(cancel):
                        ep["termination"], d["result"] = "cancelled", "cancelled_before_sampling"
                        break
                    selected = sample_policy(logits[0], mean[0], mask, generator=generator, std_deg=std_deg)
                    d.update(selected)
                    choice = selected["action_index"]
                    if choice == motion_policy.MOVE:
                        mean_deg = [x * motion_policy.JOINT_SCALE_DEG for x in mean[0].cpu().tolist()]
                        target = ([x * motion_policy.JOINT_SCALE_DEG for x in selected["sampled_joint_normalized"]]
                                  if selected["joint_sampled"] else list(mean_deg))
                        d.update(mean_joint_deg=mean_deg, mean_out_of_range=not common.valid_target(mean_deg),
                                 sampled_joint_deg=target if selected["joint_sampled"] else None,
                                 proposed_joint_deg=target)
                    else:
                        target = list(last_target)
                    if common._cancelled(cancel):
                        ep["termination"], d["result"] = "cancelled", "cancelled_before_command"
                        break
                    if choice == motion_policy.DONE:
                        ep["done_before_any_command"] = ep["physical_command_count"] == 0
                        d["result"] = "done_no_command"
                        start = float(sim.data.time)
                        try:
                            assessed = sim.evaluate_transfer(before, goal["source_slot"], goal["destination_slot"],
                                                             stable_seconds=1.0)
                        finally:
                            ep["assessment_physics_steps"] += common._elapsed_steps(sim, start)
                        validate_assessment(assessed, before, goal)
                        if assessed.get("sample_count") != ep["assessment_physics_steps"] + 1:
                            raise ValueError("assessment sample count differs from actual physics steps")
                        ep.update(assessment=assessed, physical_success=assessed["success"])
                        if common._cancelled(cancel):
                            ep["termination"] = "cancelled"
                            break
                        ep.update(termination="model_done", success=assessed["success"])
                        d["reward"] = common.reward_components(
                            reward_before, None, gamma=gamma,
                            terminal=common.SUCCESS_REWARD if assessed["success"] else common.FAILURE_REWARD)
                        break
                    if choice == motion_policy.MOVE and not common.valid_target(target):
                        ep.update(termination="out_of_range_target", success=False)
                        d["result"] = "invalid_sample_no_command" if selected["joint_sampled"] else "invalid_mean_no_command"
                        d["reward"] = common.reward_components(reward_before, None, gamma=gamma,
                                                               terminal=common.INVALID_TARGET_REWARD)
                        break
                    next_gripper = gripper if choice == motion_policy.MOVE else ("open" if choice == motion_policy.OPEN else "closed")
                    next_opening = opening if choice == motion_policy.MOVE else (.04 if next_gripper == "open" else 0.)
                    d["command"] = {"joints_deg": target, "gripper_opening_m": next_opening,
                                    "seconds": common.COMMAND_SECONDS, "source": "late_residual_student"}
                    ep["physical_command_attempts"] += 1
                    start = float(sim.data.time)
                    try:
                        sim.command(target, next_opening, seconds=common.COMMAND_SECONDS)
                        d["command_returned"] = True
                    finally:
                        steps = common._elapsed_steps(sim, start)
                        ep["command_physics_steps"] += steps
                        d["command_physics_steps"] = steps
                    if steps < 1:
                        raise ValueError("simulation command advanced no physics steps")
                    d.update(command_applied=True, result="command_applied")
                    ep["physical_command_count"] += 1
                    last_target, opening, gripper = target, next_opening, next_gripper
                    reward_after = common.reward_state(sim.observation(), goal, anchors)
                    d["reward_state_after"] = reward_after
                    displaced = any(v["displaced"] for v in reward_after["other_object_displacement"].values())
                    penalty = 0.0
                    if displaced and not ep["other_object_displacement_penalty_applied"]:
                        penalty = common.OTHER_DISPLACEMENT_REWARD
                        ep["other_object_displacement_penalty_applied"] = True
                    if common._cancelled(cancel):
                        ep["termination"] = "cancelled"
                        break
                    if index + 1 == max_decisions:
                        ep.update(termination="decision_limit", success=False)
                        d["reward"] = common.reward_components(reward_before, None, gamma=gamma,
                                                               terminal=common.FAILURE_REWARD, other_penalty=penalty)
                        break
                    d["reward"] = common.reward_components(reward_before, reward_after, gamma=gamma,
                                                           terminal=None, other_penalty=penalty)
                except Exception as exc:
                    d["result"] = "error_after_command" if d.get("command_applied") else "error"
                    d["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    ep.update(termination="error", success=None, error=d["error"])
                    break
        if ep["termination"] is None:
            raise RuntimeError("rollout ended without a termination reason")
        if started and ep["success"] is not True:
            sim.mark_layout_uncertain(ep["termination"])
        ep["final_state"] = sim.save_state()
    except Exception as exc:
        ep.update(termination="error", success=None, error={"type": type(exc).__name__, "message": str(exc)})
        if started:
            try:
                sim.mark_layout_uncertain("rollout_error")
                ep["final_state"] = sim.save_state()
            except Exception as state_exc:
                ep["final_state_error"] = {"type": type(state_exc).__name__, "message": str(state_exc)}
    if ep["termination"] in COMPLETED_TERMINATIONS:
        value = 0.0
        for d in reversed(ep["decisions"]):
            value = d["reward"]["total"] + gamma * value
            d["return_to_go"] = value
        ep["discounted_return"] = value
        ep["undiscounted_reward"] = sum(d["reward"]["total"] for d in ep["decisions"])
    return common.safe_json(ep)


def validate_experience_batch(episodes, *, expected_count: int = 8) -> dict:
    """Guard the fixed collection batch, separately from gradient computation."""
    if type(expected_count) is not int or expected_count < 1 or len(episodes) != expected_count:
        raise ValueError("experience batch has the wrong number of episodes")
    if any(ep.get("termination") not in COMPLETED_TERMINATIONS for ep in episodes):
        raise ValueError("incomplete or invalid trajectory cannot be optimized")
    steps = sum(ep.get("command_physics_steps", 0) for ep in episodes)
    terms = sum(bool(d.get("joint_sampled")) for ep in episodes for d in ep["decisions"])
    if steps < 1:
        raise ValueError("experience batch contains no command physics steps")
    if terms < 1:
        raise ValueError("experience batch contains no sampled late MOVE")
    return {"episode_count": len(episodes), "command_physics_steps": steps, "joint_likelihood_terms": terms}


def reinforce_loss(model, episodes: list[dict], *, gamma: float = GAMMA, std_deg: float = STD_DEG,
                   device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, dict]:
    """Recompute on-policy joint log likelihood from detached, stored samples.

    Every decision contributes to returns, but only sampled late MOVE decisions
    have likelihood terms. The caller must separately enforce the experiment's
    batch-size/physical-step guard and reject a zero or non-finite head gradient.
    """
    _validate_parameters(gamma, std_deg)
    validate_trainable_head(model)
    if not episodes:
        raise ValueError("at least one completed episode is required")
    terms, deterministic = [], 0
    for ep in episodes:
        if ep.get("termination") not in COMPLETED_TERMINATIONS:
            raise ValueError("incomplete or invalid trajectory cannot be optimized")
        if ep.get("sampling") != {"action": "argmax", "joint": "normal_only_late_move",
                                  "std_deg": std_deg, "gamma": gamma}:
            raise ValueError("collection distribution or discount differs")
        decisions = ep["decisions"]
        value = 0.0
        for d in reversed(decisions):
            value = d["reward"]["total"] + gamma * value
            if not math.isfinite(value) or not math.isclose(value, d["return_to_go"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("recorded return-to-go differs from the rewards")
        for index, d in enumerate(decisions):
            if d.get("index") != index:
                raise ValueError("decision index differs")
            length = _history_length(d["mask"])
            if d.get("history_length") != length:
                raise ValueError("recorded history length differs")
            windows = torch.tensor([d["window"]], dtype=torch.float32, device=device)
            masks = torch.tensor([d["mask"]], dtype=torch.bool, device=device)
            logits, mean = model(windows, masks)
            if (logits.shape != (1, 4) or mean.shape != (1, 4)
                    or not bool(torch.isfinite(logits).all() and torch.isfinite(mean).all())):
                raise ValueError("non-finite or invalid recomputed output")
            recorded = d["model_output"]
            if (not torch.equal(logits.detach().cpu(), torch.tensor(recorded["action_logits"], dtype=torch.float32))
                    or not torch.equal(mean.detach().cpu(), torch.tensor(recorded["joint_mean_normalized"], dtype=torch.float32))):
                raise ValueError("recomputed distribution parameters differ from collection")
            choice = int(logits.argmax(dim=1).item())
            sampled = choice == motion_policy.MOVE and length >= MIN_LATE_HISTORY
            if (d.get("action_index") != choice or d.get("action") != motion_policy.ACTIONS[choice]
                    or d.get("joint_sampled") is not sampled or d.get("action_log_probability") is not None):
                raise ValueError("recorded action or sampling gate differs")
            if not sampled:
                if d.get("sampled_joint_normalized") is not None or d.get("joint_log_probability") is not None:
                    raise ValueError("deterministic action cannot have a sampled likelihood")
                deterministic += 1
                continue
            sample = torch.tensor(d["sampled_joint_normalized"], dtype=torch.float32, device=device).detach()
            if sample.shape != (4,) or not bool(torch.isfinite(sample).all()):
                raise ValueError("invalid recorded joint sample")
            logp = Normal(mean[0], torch.full_like(mean[0], std_deg / motion_policy.JOINT_SCALE_DEG)).log_prob(sample).sum()
            if (not bool(torch.isfinite(logp)) or not math.isclose(float(logp.detach()), d["joint_log_probability"],
                                                                   rel_tol=1e-5, abs_tol=2e-4)):
                raise ValueError("recomputed joint likelihood differs from collection")
            terms.append(logp * (gamma ** index * d["return_to_go"]))
    if not terms:
        raise ValueError("no sampled late MOVE likelihoods")
    loss = -torch.stack(terms).sum() / len(episodes)
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite policy-gradient loss")
    return loss, {"episode_count": len(episodes), "joint_likelihood_terms": len(terms),
                  "deterministic_decisions": deterministic, "action_likelihood_terms": 0}
