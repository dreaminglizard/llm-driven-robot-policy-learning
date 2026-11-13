"""Build registration histories from causal, confirmed public decisions.

This module does not open physical-state or assessment files.  Callers must
verify provenance and whole-registration eligibility before supplying rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import motion_policy
from .scene_registration_distilled_policy import START, ControlEvent, encode_public_row


@dataclass(frozen=True)
class Registration:
    features: torch.Tensor
    previous: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor  # actual MOVE targets / 180; non-MOVE cells are ignored
    labels: torch.Tensor
    evidence: dict  # not a policy input


def build_registration(*, rows, previous, actions, targets, labels, instruction_start,
                       applied_actions, applied_targets_deg, committed, return_joints, evidence):
    """Encode in chronological order; labels never determine prior events.

    ``applied_actions`` and ``applied_targets_deg`` are independently confirmed
    control records, including for context-only rows.  END is not a control and
    becomes a mode only after its expected-scene commit.  ``return_joints`` is
    an optional actual callback-time sample, not the next decision observation.
    """
    rows = torch.as_tensor(rows, dtype=torch.float32).clone()
    previous = torch.as_tensor(previous, dtype=torch.float32).clone()
    actions = torch.as_tensor(actions, dtype=torch.long).clone()
    targets = torch.as_tensor(targets, dtype=torch.float32).clone()
    labels = torch.as_tensor(labels, dtype=torch.bool).clone()
    starts = torch.as_tensor(instruction_start, dtype=torch.bool).clone()
    n = len(rows)
    if (n < 1 or rows.shape != (n, 82) or previous.shape != (n, 4)
            or actions.shape != (n,) or targets.shape != (n, 4) or labels.shape != (n,)
            or starts.shape != (n,) or not bool(starts[0])
            or len(applied_actions) != n or len(applied_targets_deg) != n
            or len(committed) != n or len(return_joints) != n
            or not bool(torch.isfinite(rows).all() and torch.isfinite(previous).all())
            or bool(((actions[labels] < 0) | (actions[labels] >= 4)).any())):
        raise ValueError("invalid registration decision and event arrays")
    mode, event, features = START, None, []
    for i in range(n):
        if i:
            last = applied_actions[i-1]
            if last not in range(4):
                raise ValueError("every preceding context action must be confirmed")
            if last == motion_policy.DONE:
                if committed[i-1] is not True or not bool(starts[i]):
                    raise ValueError("a new instruction requires a committed END")
                if applied_targets_deg[i-1] is not None:
                    raise ValueError("END is not an applied physical command")
            else:
                if bool(starts[i]) or committed[i-1] is not False:
                    raise ValueError("only a committed END starts a new instruction")
                applied = torch.as_tensor(applied_targets_deg[i-1], dtype=torch.float32)
                if applied.shape != (4,) or not bool(torch.isfinite(applied).all()) \
                        or not torch.allclose(applied, previous[i], atol=4e-5, rtol=0):
                    raise ValueError("next input does not retain the actual applied target")
                if last != motion_policy.MOVE and not torch.allclose(applied, previous[i-1], atol=4e-5, rtol=0):
                    raise ValueError("a gripper command changed the held joint target")
                after = return_joints[i-1]
                event = ControlEvent(last, tuple(rows[i-1, 26:30].tolist()),
                                     tuple((rows[i-1, :4] * 180).tolist()),
                                     None if after is None else tuple(after))
            if last == motion_policy.DONE and not torch.allclose(previous[i-1], previous[i], atol=4e-5, rtol=0):
                raise ValueError("END must retain the last applied target")
            mode = last
        features.append(encode_public_row(rows[i], previous[i], previous_mode=mode,
                                          new_instruction=bool(starts[i]), control_event=event))
        if bool(labels[i]):
            if int(actions[i]) != applied_actions[i]:
                raise ValueError("label does not match the confirmed decision")
            if int(actions[i]) == motion_policy.MOVE:
                actual = torch.as_tensor(applied_targets_deg[i], dtype=torch.float32)
                if (actual.shape != (4,) or not bool(torch.isfinite(actual).all())
                        or not torch.allclose(targets[i]*180, actual, atol=4e-5, rtol=0)):
                    raise ValueError("MOVE labels must be the actual applied joint targets")
    if applied_actions[-1] != motion_policy.DONE or committed[-1] is not True:
        raise ValueError("a completed registration must end with a committed END")
    return Registration(torch.stack(features), previous, actions, targets, labels, dict(evidence))


def from_original_94(data, episodes, groups):
    """Convert the original 87 registrations without inventing return samples."""
    result = []
    for number, group in enumerate(groups):
        indices, starts = [], []
        for episode_index in group:
            episode = episodes[episode_index]
            indices.extend(range(episode["start"], episode["start"] + episode["length"]))
            starts.extend([True] + [False] * (episode["length"] - 1))
        rows = data["windows"][indices, -1]
        actions = data["actions"][indices]
        previous = data["previous"][indices]
        targets = data["targets"][indices]
        applied_targets = [(targets[i] * 180).tolist() if int(action) == motion_policy.MOVE
                           else None if int(action) == motion_policy.DONE else previous[i].tolist()
                           for i, action in enumerate(actions)]
        result.append(build_registration(rows=rows, previous=previous, actions=actions, targets=targets,
            labels=data["label_mask"][indices], instruction_start=starts,
            applied_actions=actions.tolist(), applied_targets_deg=applied_targets,
            committed=[int(action) == motion_policy.DONE for action in actions],
            return_joints=[None] * len(indices),
            evidence={"source": "direct_operator", "registration_index": number,
                      "trajectory_paths": [episodes[i]["trajectory"] for i in group],
                      "callback_time_joint_samples_recorded": False}))
    return tuple(result)


def batch_registrations(registrations):
    if not registrations:
        raise ValueError("at least one registration is required")
    n, length = len(registrations), max(len(item.actions) for item in registrations)
    count = registrations[0].features.shape[-1]
    result = {"features": torch.zeros(n, length, count), "previous": torch.zeros(n, length, 4),
              "actions": torch.full((n, length), -1, dtype=torch.long), "targets": torch.zeros(n, length, 4),
              "valid": torch.zeros(n, length, dtype=torch.bool), "labels": torch.zeros(n, length, dtype=torch.bool)}
    for i, item in enumerate(registrations):
        k = len(item.actions)
        if item.features.shape != (k, count) or item.previous.shape != (k, 4):
            raise ValueError("registration shapes differ")
        for name in ("features", "previous", "actions", "targets", "labels"):
            result[name][i, :k] = getattr(item, name)
        result["valid"][i, :k] = True
    return result
