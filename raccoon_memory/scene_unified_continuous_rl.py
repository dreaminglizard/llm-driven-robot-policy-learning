"""On-policy, continuous-transfer RL for the unified observable policy.

The sampler only receives the robot's existing observation and last normally
returned control target.  RewardProbe is a separate simulator-only observer;
its measurements are never supplied to the model or used to choose a task.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from raccoon_memory import (full_teacher_training, motion_policy, scene_features, scene_fk_loss,
                            scene_motion_rl as old_rl)


GAMMA = 0.99
STD_DEG = 0.5
SEQUENCE_SUCCESS_BONUS = 10.0
UNFINISHED_STAGE_PENALTY = -10.0


class StochasticUnifiedPolicy(nn.Module):
    """Sample one action while preserving the actual distribution separately.

    Returned scores encode the sampled action for the existing argmax executor;
    they are *not* model logits.  The existing one-degree guard wraps this
    adapter, so it operates on the sampled target, not the distribution mean.
    """

    def __init__(self, model, generator, *, std_deg=STD_DEG, reward_probe=None):
        super().__init__()
        if not isinstance(generator, torch.Generator) or not math.isfinite(std_deg) or std_deg <= 0:
            raise ValueError("a generator and positive finite standard deviation are required")
        self.model, self.generator, self.std_deg = model, generator, float(std_deg)
        self.history_length, self.feature_count = model.history_length, model.feature_count
        self.reward_probe = reward_probe
        self.collection_model_sha256 = full_teacher_training.state_fingerprint(model)["sha256"]
        self.samples = []

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self.reward_probe is not None:
            self.reward_probe.before_decision()
        outputs = self.model.training_outputs(windows, masks, previous_target_degrees)
        if outputs.action_logits.shape != (1, 4) or outputs.ungated_move_joints.shape != (1, 4):
            raise ValueError("one four-action policy output is required")
        sample = old_rl.sample_policy(outputs.action_logits[0], outputs.ungated_move_joints[0],
                                      std_deg=self.std_deg, generator=self.generator)
        sample.update(window=windows[0].detach().cpu().tolist(), mask=masks[0].detach().cpu().tolist(),
                      previous_target_degrees=previous_target_degrees[0].detach().cpu().tolist(),
                      model_logits=outputs.action_logits[0].detach().cpu().tolist(),
                      move_mean_normalized=outputs.ungated_move_joints[0].detach().cpu().tolist(),
                      collection_model_sha256=self.collection_model_sha256, sampling_std_deg=self.std_deg)
        self.samples.append(sample)
        scores = torch.zeros_like(outputs.action_logits)
        scores[0, sample["action_index"]] = 1.0
        angles = (torch.tensor([sample["sampled_joint_normalized"]], dtype=windows.dtype, device=windows.device)
                  if sample["action_index"] == motion_policy.MOVE else outputs.absolute_joints)
        return scores, angles


def _soft_distance(value, scale):
    """Bounded, continuous scale; it is not an acceptance threshold."""
    value = float(value)
    if not math.isfinite(value) or value < 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid reward distance")
    return value / (value + scale)


def reward_measurement(raw, before, task, anchors):
    """Compute simulator-only continuous auxiliaries for one decision state."""
    base = old_rl.reward_state(raw, task, anchors)
    objects = {obj["id"]: obj for obj in raw["privileged_simulation_only"]["objects"]}
    moved = objects[task["object_id"]]["position_m"]
    destination = before[task["destination_slot"] - 1]
    if task["destination_level"] == 2:
        if len(destination) != 1:
            raise ValueError("stacking support does not match the registered task")
        support_id = destination[-1]
        support = objects[support_id]["position_m"]
        support_xy = math.dist(moved[:2], support[:2])
        support_height = abs(moved[2] - support[2] - .02)
        support_geom = f"object_{support_id}_geom"
    else:
        support_xy = 0.0
        support_height = abs(moved[2] - .01)
        support_geom = "table"
    geom = f"object_{task['object_id']}_geom"
    force = sum(max(0.0, float(c["normal_force_n"])) for c in raw["privileged_simulation_only"]["contacts"]
                if {c["geom1"], c["geom2"]} == {geom, support_geom})
    other = list(base["other_object_displacement"].values())
    other_distance = (sum((_soft_distance(v["xy_m"], .03) + _soft_distance(v["height_m"], .02)) / 2
                          for v in other) / len(other) if other else 0.0)
    components = {"tcp": -.10 * _soft_distance(base["tcp_object_distance_m"], .30),
                  "goal": -.20 * _soft_distance(base["object_goal_distance_m"], .30),
                  "support_alignment": -.05 * (_soft_distance(support_xy, .08)
                                                  + _soft_distance(support_height, .04)) / 2,
                  "support_force": .02 * _soft_distance(force, .10),
                  "other_objects": -.05 * other_distance}
    return {"potential": sum(components.values()), "components": components,
            "support_xy_m": support_xy, "support_height_error_m": support_height,
            "support_force_n": force, "other_object_displacement": base["other_object_displacement"],
            "privileged_simulation_only": True}


class RewardProbe:
    """Observe rewards alongside a simulator; never return them to the actor."""

    def __init__(self, sim):
        self.sim = sim
        self.records = []
        self._contract = None
        self._before = self._task = self._anchors = None

    def before_decision(self):
        pending = self.sim.pending_transfer
        if not isinstance(pending, dict):
            raise ValueError("the simulator has no active transfer")
        before, task = scene_features.transfer_task(pending["before_layout"], {
            "source_slot": pending["source_slot"], "destination_slot": pending["destination_slot"]})
        contract = (before, task["source_slot"], task["destination_slot"])
        raw = self.sim.observation()
        if contract != self._contract:
            self._contract, self._before, self._task = contract, before, task
            self._anchors = {obj["id"]: list(obj["position_m"])
                             for obj in raw["privileged_simulation_only"]["objects"]}
        self.records.append({"before": reward_measurement(raw, self._before, self._task, self._anchors),
                             "after": None})

    def after_command(self):
        if not self.records or self.records[-1]["after"] is not None:
            raise ValueError("a command must follow exactly one observed decision")
        self.records[-1]["after"] = reward_measurement(
            self.sim.observation(), self._before, self._task, self._anchors)


def attach_rewards(stages, samples, measurements, *, planned_stages, gamma=GAMMA,
                   sequence_bonus=SEQUENCE_SUCCESS_BONUS):
    """Attach post-hoc stage scores and continuous potentials to one trajectory.

    The first DONE is nonterminal if another planned stage was reached.  Its
    potential is followed by that stage's initial potential, so discounted
    potential differences telescope across task boundaries.  Error/unknown
    episodes are rejected rather than assigned fabricated failure rewards. A
    specifically observed closed-gripper start after DONE is an incomplete
    sequence, not a physical failure or an unconfirmed command.
    """
    if (not stages or type(planned_stages) is not int or planned_stages < len(stages) or
            not math.isfinite(gamma) or not 0 < gamma <= 1 or
            not math.isfinite(sequence_bonus) or sequence_bonus < 0):
        raise ValueError("invalid trajectory or reward parameters")
    if len(samples) != len(measurements):
        raise ValueError("sample and reward observation counts differ")
    if (not samples or len({row.get("collection_model_sha256") for row in samples}) != 1 or
            len({row.get("sampling_std_deg") for row in samples}) != 1):
        raise ValueError("trajectory has missing or mixed collection policy identities")
    if stages[-1].get("termination") == "model_done" and len(stages) != planned_stages:
        raise ValueError("a completed intermediate transfer is missing its planned continuation")
    flat, cursor, stage_successes, blocked_start = [], 0, [], False
    for number, stage in enumerate(stages):
        decisions = stage.get("decisions", [])
        termination = stage.get("termination")
        error = stage.get("error") or {}
        if (termination == "error" and not decisions and number > 0 and number == len(stages) - 1 and
                error.get("type") == "ValueError" and
                error.get("message") == "a new transfer must start with an open gripper"):
            if (stage.get("row_start") != cursor or stage.get("row_end") != cursor or
                    (stage.get("offline_score") or {}).get("status") != "not_assessed"):
                raise ValueError("blocked transfer must not contain a sampled decision")
            stage_successes.append(None)
            blocked_start = True
            continue
        if termination not in ("model_done", "decision_limit", "out_of_range_target") or not decisions:
            raise ValueError("incomplete or invalid trajectory cannot train")
        if number < len(stages) - 1 and termination != "model_done":
            raise ValueError("only DONE can be followed by another transfer")
        if stage.get("row_start") != cursor or stage.get("row_end") != cursor + len(decisions):
            raise ValueError("stage decision ranges differ from captured samples")
        if termination == "model_done":
            status = (stage.get("offline_score") or {}).get("status")
            if status not in ("success", "failure"):
                raise ValueError("DONE is missing a valid offline physical assessment")
            stage_successes.append(status == "success")
        else:
            # No physical assessment happened for a limit or invalid command.
            stage_successes.append(None)
        for index, decision in enumerate(decisions):
            sample, observed = samples[cursor], measurements[cursor]
            if (decision.get("index") != index or decision.get("action") != motion_policy.ACTIONS[sample["action_index"]]
                    or len(sample["previous_target_degrees"]) != 4):
                raise ValueError("sample and executor decisions differ")
            outcome = decision.get("result")
            if outcome not in ("command_applied", "done_no_command", "no_command"):
                raise ValueError("unconfirmed command cannot train")
            if outcome == "command_applied" and observed["after"] is None:
                raise ValueError("normal command return lacks reward measurement")
            if outcome != "command_applied" and observed["after"] is not None:
                raise ValueError("a no-command decision has an after-command measurement")
            if (sample["action_index"] == motion_policy.MOVE and outcome == "command_applied"):
                guard = decision.get("guard") or {}
                actual = decision.get("command", {}).get("joints_deg")
                if actual != guard.get("guard_returned_target_deg"):
                    raise ValueError("guarded and normally returned MOVE targets differ")
            elif outcome == "command_applied":
                actual = decision.get("command", {}).get("joints_deg")
                previous = sample["previous_target_degrees"]
                if (not isinstance(actual, list) or len(actual) != 4 or
                        any(not math.isclose(a, b, abs_tol=1e-5, rel_tol=0) for a, b in zip(actual, previous))):
                    raise ValueError("gripper command did not retain the prior applied arm target")
            flat.append({**sample, "stage": number, "stage_index": index,
                         "result": outcome, "reward_before": observed["before"],
                         "reward_after": observed["after"]})
            cursor += 1
    if cursor != len(samples):
        raise ValueError("unmatched policy samples")
    sequence_success = (len(stages) == planned_stages and all(value is True for value in stage_successes)
                        and stages[-1]["termination"] == "model_done")
    unfinished_stages = planned_stages - sum(stage["termination"] == "model_done" for stage in stages)
    for index, row in enumerate(flat):
        stage = stages[row["stage"]]
        last = row["stage_index"] == len(stage["decisions"]) - 1
        final = index == len(flat) - 1
        outcome = stage["termination"] if last else None
        if (last and ((outcome == "model_done" and row["result"] != "done_no_command") or
                      (outcome == "decision_limit" and row["result"] != "command_applied") or
                      (outcome == "out_of_range_target" and row["result"] != "no_command"))):
            raise ValueError("terminal decision and executor result differ")
        before = row["reward_before"]["potential"]
        if not last:
            if row["result"] != "command_applied":
                raise ValueError("nonfinal decision lacks an applied command")
            after = row["reward_after"]["potential"]
            after_measurement = row["reward_after"]
            next_before = flat[index + 1]["reward_before"]["potential"]
            if not math.isclose(after, next_before, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("successive reward observations do not share the same physical state")
        elif not final:
            if outcome != "model_done" or row["result"] != "done_no_command":
                raise ValueError("middle boundary is not a normally returned DONE")
            after = flat[index + 1]["reward_before"]["potential"]
            after_measurement = flat[index + 1]["reward_before"]
        else:
            after = None
            after_measurement = None
        milestone = ((old_rl.SUCCESS_REWARD if stage_successes[row["stage"]] else old_rl.FAILURE_REWARD)
                     if outcome == "model_done" else 0.0)
        terminal = (old_rl.FAILURE_REWARD if outcome == "decision_limit" else
                    old_rl.INVALID_TARGET_REWARD if outcome == "out_of_range_target" else 0.0)
        completion = sequence_bonus if final and sequence_success else 0.0
        incomplete = (UNFINISHED_STAGE_PENALTY * unfinished_stages
                      if final and (outcome in ("decision_limit", "out_of_range_target") or blocked_start) else 0.0)
        shaping = (gamma * after if after is not None else 0.0) - before
        before_parts = row["reward_before"].get("components")
        shaping_parts = None
        if before_parts is not None:
            after_parts = None if after_measurement is None else after_measurement.get("components")
            if after_measurement is not None and (after_parts is None or set(after_parts) != set(before_parts)):
                raise ValueError("potential component names differ across a transition")
            shaping_parts = {name: (gamma * after_parts[name] if after_parts is not None else 0.0) - value
                             for name, value in before_parts.items()}
            if not math.isclose(sum(shaping_parts.values()), shaping, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("potential components do not sum to shaping reward")
        total = old_rl.STEP_COST + shaping + milestone + terminal + completion + incomplete
        row["reward"] = {"decision_cost": old_rl.STEP_COST, "potential_before": before,
                         "potential_after": after, "shaping": shaping, "stage_success_reward": milestone,
                         "terminal_failure_reward": terminal, "sequence_success_reward": completion,
                         "incomplete_sequence_reward": incomplete, "shaping_components": shaping_parts,
                         "total": total}
        if not old_rl.finite_tree(row["reward"]):
            raise ValueError("nonfinite reward")
    value = 0.0
    for row in reversed(flat):
        value = row["reward"]["total"] + gamma * value
        row["return_to_go"] = value
    return {"decisions": flat, "sequence_success": sequence_success, "stage_successes": stage_successes,
            "unfinished_stages": unfinished_stages, "blocked_start_open_gripper": blocked_start,
            "planned_stages": planned_stages, "gamma": gamma}


def reinforce_loss(model, episodes, *, gamma=GAMMA, std_deg=STD_DEG, device="cpu"):
    """Recompute logits and sampled MOVE likelihood under the collection model."""
    if not episodes or not math.isfinite(gamma) or not 0 < gamma <= 1 or not math.isfinite(std_deg) or std_deg <= 0:
        raise ValueError("valid trajectories and sampling parameters are required")
    terms, move_terms = [], 0
    model_sha = full_teacher_training.state_fingerprint(model)["sha256"]
    for episode in episodes:
        if episode.get("gamma") != gamma or not episode.get("decisions"):
            raise ValueError("trajectory discount or decisions differ")
        value = 0.0
        for row in reversed(episode["decisions"]):
            value = row["reward"]["total"] + gamma * value
            if not math.isclose(value, row["return_to_go"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("recorded return differs from rewards")
        for index, row in enumerate(episode["decisions"]):
            if row.get("collection_model_sha256") != model_sha or row.get("sampling_std_deg") != std_deg:
                raise ValueError("collection policy fingerprint or sampling standard deviation changed")
            windows = torch.tensor([row["window"]], dtype=torch.float32, device=device)
            masks = torch.tensor([row["mask"]], dtype=torch.bool, device=device)
            previous = torch.tensor([row["previous_target_degrees"]], dtype=torch.float32, device=device)
            outputs = model.training_outputs(windows, masks, previous)
            logits, mean = outputs.action_logits[0], outputs.ungated_move_joints[0]
            if (not torch.equal(logits.detach().cpu(), torch.tensor(row["model_logits"], dtype=torch.float32)) or
                    not torch.equal(mean.detach().cpu(), torch.tensor(row["move_mean_normalized"], dtype=torch.float32))):
                raise ValueError("policy distribution changed since collection")
            action = torch.tensor(row["action_index"], dtype=torch.long, device=device)
            logp = Categorical(logits=logits).log_prob(action)
            if row["action_index"] == motion_policy.MOVE:
                sample = torch.tensor(row["sampled_joint_normalized"], dtype=torch.float32, device=device).detach()
                if sample.shape != (4,) or not bool(torch.isfinite(sample).all()):
                    raise ValueError("invalid sampled MOVE target")
                logp = logp + Normal(mean, torch.full_like(mean, std_deg / 180.)).log_prob(sample).sum()
                move_terms += 1
            elif row["sampled_joint_normalized"] is not None:
                raise ValueError("non-MOVE decision has a sampled joint target")
            if (not bool(torch.isfinite(logp)) or not math.isclose(float(logp.detach()),
                    row["policy_log_probability"], rel_tol=1e-5, abs_tol=2e-4)):
                raise ValueError("sample likelihood differs from collection")
            terms.append(logp * (gamma ** index * row["return_to_go"]))
    loss = -torch.stack(terms).sum() / len(episodes)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite policy-gradient loss")
    return loss, {"episodes": len(episodes), "action_terms": len(terms), "move_terms": move_terms}


def distillation_terms(model, batch):
    """BC action distribution and one observable MOVE target, kept separate."""
    outputs = model.training_outputs(batch["windows"], batch["masks"], batch["previous"])
    reference, target = batch["bc_logits"], batch["joint_target"]
    if reference.shape != outputs.action_logits.shape or target.shape != outputs.ungated_move_joints.shape:
        raise ValueError("distillation targets have incompatible shapes")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(target).all()):
        raise ValueError("distillation targets must be finite")
    log_reference = F.log_softmax(reference, dim=1)
    action = (log_reference.exp() * (log_reference - F.log_softmax(outputs.action_logits, dim=1))).sum(1).mean()
    moves = reference.argmax(1) == motion_policy.MOVE
    joints = (F.mse_loss(outputs.ungated_move_joints[moves], target[moves]) if bool(moves.any())
              else outputs.ungated_move_joints.sum() * 0.)
    fk, _, _ = scene_fk_loss.move_fk_loss(outputs.ungated_move_joints, target, moves)
    return {"action_kl": action, "joint_mse": joints, "nominal_fk": fk, "move_count": int(moves.sum())}


def _gradient_groups(model, loss):
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    sums = {}
    for (name, _), grad in zip(named, gradients):
        group = ("base.encoder" if name.startswith("base.encoder.") else
                 "base.action_head" if name.startswith("base.action_head.") else
                 "base.joint_head" if name.startswith("base.joint_head.") else name.split(".", 1)[0])
        sums.setdefault(group, 0.0)
        if grad is not None:
            if not bool(torch.isfinite(grad).all()):
                raise ValueError("nonfinite gradient")
            sums[group] += float(grad.detach().double().square().sum())
    return {name: math.sqrt(value) for name, value in sums.items()}


def _encoder_alignment(model, first, second):
    params = [p for name, p in model.named_parameters() if name.startswith("base.encoder.")]
    first_grads = torch.autograd.grad(first, params, retain_graph=True, allow_unused=True)
    second_grads = torch.autograd.grad(second, params, retain_graph=True, allow_unused=True)
    dot = a2 = b2 = 0.0
    for a, b in zip(first_grads, second_grads):
        if a is not None:
            a2 += float(a.detach().double().square().sum())
        if b is not None:
            b2 += float(b.detach().double().square().sum())
        if a is not None and b is not None:
            dot += float((a.detach().double() * b.detach().double()).sum())
    return {"dot": dot, "cosine": dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None}


def paired_update(control, control_optimizer, candidate, candidate_optimizer, batch, episodes, *,
                  action_weight, joint_weight, fk_weight, rl_weight, gamma=GAMMA, std_deg=STD_DEG,
                  diagnostic_batch=None):
    """One same-batch BC-only/BC+RL update of all parameters of two models."""
    weights = {"action": action_weight, "joint": joint_weight, "fk": fk_weight, "rl": rl_weight}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("loss weights must be finite nonnegative numbers")
    for model, optimizer in ((control, control_optimizer), (candidate, candidate_optimizer)):
        params = list(model.parameters())
        optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
        if (not params or any(not p.requires_grad for p in params) or len(params) != len(optimizer_params)
                or {id(p) for p in params} != {id(p) for p in optimizer_params}):
            raise ValueError("every model parameter must be trainable and in the optimizer exactly once")
    if {id(p) for p in control.parameters()} & {id(p) for p in candidate.parameters()}:
        raise ValueError("paired models must not share parameter storage")
    control.eval()
    candidate.eval()
    bc_control, bc_candidate = distillation_terms(control, batch), distillation_terms(candidate, batch)
    rl_loss, counts = reinforce_loss(candidate, episodes, gamma=gamma, std_deg=std_deg)
    groups = {"action_kl": _gradient_groups(candidate, bc_candidate["action_kl"]),
              "joint_mse": _gradient_groups(candidate, bc_candidate["joint_mse"]),
              "nominal_fk": _gradient_groups(candidate, bc_candidate["nominal_fk"]),
              "rl": _gradient_groups(candidate, rl_loss)}
    diagnostic = None
    if diagnostic_batch is not None:
        observed = distillation_terms(candidate, diagnostic_batch)
        diagnostic = {"rows": len(diagnostic_batch["windows"]), "move_count": observed["move_count"],
                      "action_kl": float(observed["action_kl"].detach()),
                      "joint_mse": float(observed["joint_mse"].detach()),
                      "nominal_fk": float(observed["nominal_fk"].detach()),
                      "action_gradient_l2": _gradient_groups(candidate, observed["action_kl"]),
                      "joint_gradient_l2": _gradient_groups(candidate, observed["joint_mse"]),
                      "fk_gradient_l2": _gradient_groups(candidate, observed["nominal_fk"]),
                      "action_vs_rl_encoder": _encoder_alignment(candidate, observed["action_kl"], rl_loss),
                      "joint_vs_rl_encoder": _encoder_alignment(candidate, observed["joint_mse"], rl_loss)}
    def total(terms):
        return (action_weight * terms["action_kl"] + joint_weight * terms["joint_mse"]
                + fk_weight * terms["nominal_fk"])
    control_loss = total(bc_control)
    candidate_loss = total(bc_candidate) + rl_weight * rl_loss
    if not bool(torch.isfinite(control_loss) and torch.isfinite(candidate_loss)):
        raise ValueError("nonfinite paired objective")
    control_optimizer.zero_grad(set_to_none=True)
    candidate_optimizer.zero_grad(set_to_none=True)
    control_loss.backward()
    candidate_loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
           for model in (control, candidate) for p in model.parameters()):
        raise ValueError("nonfinite paired gradient")
    control_optimizer.step()
    candidate_optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for model in (control, candidate) for p in model.parameters()):
        raise ValueError("nonfinite paired parameter")
    return {"weights": weights, "bc_action_kl": float(bc_candidate["action_kl"].detach()),
            "bc_joint_mse": float(bc_candidate["joint_mse"].detach()),
            "bc_nominal_fk": float(bc_candidate["nominal_fk"].detach()),
            "bc_move_count": bc_candidate["move_count"], "rl_loss": float(rl_loss.detach()),
            "candidate_total": float(candidate_loss.detach()), "control_total": float(control_loss.detach()),
            "gradient_l2_by_component_and_group": groups, "same_observation_diagnostic": diagnostic,
            "likelihood_terms": counts}
