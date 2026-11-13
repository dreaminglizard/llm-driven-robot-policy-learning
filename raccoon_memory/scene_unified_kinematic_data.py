"""Align full recorded episodes with their previous applied servo targets.

The caller verifies the original files and split before using these functions.
Legacy command targets remain legacy teacher labels: this module does not
claim that their original generator used only nonprivileged observations.
No target is inferred from an observed joint position or a label placeholder.
"""
from __future__ import annotations

import math

import torch

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy, scene_policy, scene_simultaneous_data

TRAIN_SCOPES = frozenset(("base", "extra", "teacher", "direct", "diverse_continuation"))
PRIOR_TOLERANCE_DEG = 1e-7


def _prior_valid(value):
    if (not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 4 or
            not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError("previous servo targets must be a finite [rows,4] tensor")
    bounds = value.new_tensor(JOINT_LIMITS)
    if bool(((value < bounds[:, 0] - PRIOR_TOLERANCE_DEG) |
             (value > bounds[:, 1] + PRIOR_TOLERANCE_DEG)).any()):
        raise ValueError("previous servo target outside joint limits")


def align_saved_pool(pool, controls, *, scope):
    """Attach a saved control array to the same original windows and labels.

    Source hashes, episode lineage and train/development separation must have
    been checked by the caller. The recorded initial actuator setting is not
    called a returned command. Later prior targets are reconstructed from the
    preceding recorded command and are not direct servo snapshots.
    """
    if scope not in TRAIN_SCOPES:
        raise ValueError("scope is not an allowed training source")
    if not isinstance(pool, dict) or not isinstance(controls, dict):
        raise ValueError("original pool and recovered controls are required")
    keys = ("windows", "masks", "actions", "absolute_targets")
    values = [pool.get(k) for k in keys]
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError("original windows, mask and teacher labels are required")
    windows, masks, actions, targets = values
    n = len(actions) if actions.ndim else 0
    if (n == 0 or windows.shape != (n, 8, 82) or not windows.is_floating_point() or
            masks.shape != (n, 8) or masks.dtype != torch.bool or actions.shape != (n,) or
            actions.dtype != torch.long or targets.shape != (n, 4) or not targets.is_floating_point() or
            any(value.device.type != "cpu" for value in values) or
            not bool(torch.isfinite(windows).all()) or not bool(masks[:, -1].all()) or
            bool((masks[:, :-1] & ~masks[:, 1:]).any()) or
            bool(((actions < 0) | (actions >= len(motion_policy.ACTIONS))).any())):
        raise ValueError("original training pool violates the causal input or label contract")
    prior = controls.get("prior_joint_deg")
    _prior_valid(prior)
    if prior.shape != (n, 4) or prior.device.type != "cpu":
        raise ValueError("previous servo count or device differs")
    joint = controls.get("joint_target_deg")
    done = controls.get("done_target")
    gripper = controls.get("gripper_target")
    episode = controls.get("episode_ref")
    decision = controls.get("decision_index")
    terminal = controls.get("terminal_origin")
    if (not isinstance(joint, torch.Tensor) or joint.shape != (n, 4) or not joint.is_floating_point() or
            any(not isinstance(item, torch.Tensor) or item.shape != (n,) or item.device.type != "cpu"
                for item in (done, gripper, episode, decision, terminal)) or joint.device.type != "cpu" or
            any(item.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                for item in (done, gripper, episode, decision, terminal)) or
            bool(((episode < 0) | (decision < 0)).any())):
        raise ValueError("recovered teacher command or episode lineage is incomplete")
    move = actions == motion_policy.MOVE
    stopping = actions == motion_policy.DONE
    if (not torch.equal(done.to(torch.long), stopping.to(torch.long)) or
            bool((terminal[stopping] <= 0).any()) or bool((terminal[~stopping] != 0).any()) or
            bool((gripper[stopping] != -1).any()) or
            bool(((gripper[~stopping] != scene_simultaneous_data.OPEN) &
                  (gripper[~stopping] != scene_simultaneous_data.CLOSED)).any()) or
            bool((gripper[actions == motion_policy.OPEN] != scene_simultaneous_data.OPEN).any()) or
            bool((gripper[actions == motion_policy.CLOSE] != scene_simultaneous_data.CLOSED).any()) or
            not bool(torch.isfinite(joint[move]).all()) or not bool(torch.isfinite(targets[move]).all()) or
            not torch.equal((joint[move] / motion_policy.JOINT_SCALE_DEG).to(targets.dtype), targets[move])):
        raise ValueError("recovered servo command disagrees with the original teacher labels")
    return {"windows": windows, "masks": masks, "actions": actions, "absolute_targets": targets,
            "previous_target_degrees": prior.to(dtype=windows.dtype), "episode_ref": episode,
            "decision_index": decision, "terminal_origin": terminal, "scope": scope,
            "initial_control_origin": "saved_initial_actuator_setting_not_a_returned_command",
            "later_control_origin": "preceding_recorded_applied_servo_command",
            "teacher_label_origin": "legacy_recorded_teacher_may_have_used_privileged_geometry"}


def recover_recorded_episode(episode, attempt, initial_servo):
    """Reconstruct a later episode's prior targets from its saved command chain.

    ``initial_servo`` must be decoded by the caller from the episode's saved
    initial actuator state; this function checks any independently recorded
    initial arm and finger targets against it. It never examines object poses.
    """
    if (not isinstance(episode, dict) or not isinstance(attempt, dict) or
            episode.get("episode_id") != attempt.get("episode_id") or attempt.get("success") is not True or
            attempt.get("simulated") is not True or attempt.get("actual_hardware") is not False or
            episode.get("registered_scene") != attempt.get("registered_scene") or
            episode.get("task") != attempt.get("task") or
            not isinstance(initial_servo, scene_simultaneous_data.ServoTargets)):
        raise ValueError("episode, successful attempt or initial actuator state differs")
    continuation = attempt.get("continuation")
    if continuation is not None:
        if not isinstance(continuation, dict):
            raise ValueError("continuation record is invalid")
        arm, fingers = continuation.get("initial_servo_target_joints_deg"), continuation.get("initial_finger_servo_targets_m")
        if (not isinstance(arm, list) or len(arm) != 4 or not isinstance(fingers, list) or len(fingers) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in (*arm, *fingers)) or
                any(abs(a - b) > 1e-9 for a, b in zip(arm, initial_servo.joints_deg)) or
                abs(fingers[0] - fingers[1]) > 1e-9 or
                abs(sum(fingers) - initial_servo.gripper_opening_m) > 1e-9):
            raise ValueError("initial servo record and decoded actuator control differ")
    transitions, decisions = episode.get("transitions"), attempt.get("decisions")
    if not isinstance(transitions, list) or not transitions or not isinstance(decisions, list) or len(decisions) != len(transitions):
        raise ValueError("episode transitions and recorded commands differ")
    previous, previous_rows, actual_actions = initial_servo, [], []
    for transition, decision in zip(transitions, decisions):
        label, target = scene_simultaneous_data.teacher_command(transition, decision)
        if label in (motion_policy.OPEN, motion_policy.CLOSE) and any(
                abs(a - b) > 1e-8 for a, b in zip(target.joints_deg, previous.joints_deg)):
            raise ValueError("gripper command did not hold the previous arm target")
        previous_rows.append(previous.joints_deg)
        actual_actions.append(label)
        previous = target
    previous_rows.append(previous.joints_deg)
    actual_actions.append(motion_policy.DONE)
    built = scene_policy.build_episodes([episode])
    result = scene_policy.collate_windows(built)
    if result["actions"].tolist() != actual_actions:
        raise ValueError("saved episode actions differ from full recorded commands")
    prior = torch.tensor(previous_rows, dtype=torch.float32)
    _prior_valid(prior)
    result["previous_target_degrees"] = prior
    result["scope"] = "diverse_continuation"
    result["initial_control_origin"] = "saved_initial_actuator_setting_not_a_returned_command"
    result["later_control_origin"] = "preceding_recorded_applied_servo_command"
    result["teacher_label_origin"] = "legacy_recorded_teacher_may_have_used_privileged_geometry"
    return result
