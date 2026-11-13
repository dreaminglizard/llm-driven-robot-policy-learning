"""Causal, destination-relative nominal TCP targets from verified train data.

The input is exactly the existing 92-feature registered-scene/robot history.
The two ID views share every registration, time step, action and target.  In
the ID-free view only object-identity columns are zeroed; the slot-level empty
bits still encode occupancy and hence object count.  Source/destination,
levels, slot XY, robot joints and confirmed prior commands remain available.

For each confirmed applied MOVE, ``Geometry.forward(q)`` gives nominal TCP in
the robot base frame, with unwrapped pitch ``90 + q2 + q3 + q4`` degrees.
For the registered destination slot XY and destination level L the fixed
per-transfer reference is ``g = (slot_x, slot_y, 15 + 20*(L-1))`` mm.  The
target is ``((xyz-g)/300, pitch/180)``.  The 15/35 mm heights are an existing
nominal grip-height assumption, not measured or true object positions.  They
do not claim that a placement or a physical rollout succeeded.

Non-MOVEs and unlabelled context have NaN targets.  A MOVE whose target cannot
be validated is also NaN with a false target mask and a reason in ``issues``.
The source action mask and evidence remain separate and unchanged.  In
particular, an IK failure is a limitation of the existing inverse branch, not
proof that the original applied command was impossible.  No simulator step,
model training, held-out trajectory or device access occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import torch

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import motion_policy, scene_applied_action_dataset as experience
from . import scene_applied_action_student as student, scene_direct_applied_dataset as direct


FORMAT = "raccoon-scene-target-relative-tcp-pitch-dataset.v1"
POSITION_SCALE_MM = 300.0
PITCH_SCALE_DEG = 180.0
BASE_GRIP_HEIGHT_MM = 15.0
LEVEL_SPACING_MM = 20.0
FEATURE_NAMES = student.FEATURE_NAMES
FEATURE_COUNT = student.FEATURE_COUNT


def _indices(prefix, count):
    return tuple(FEATURE_NAMES.index(f"{prefix}{i}") for i in range(1, count + 1))


SOURCE_SLOT_INDICES = _indices("source_slot_", 4)
DESTINATION_SLOT_INDICES = _indices("destination_slot_", 4)
SOURCE_LEVEL_INDICES = _indices("source_level_", 2)
DESTINATION_LEVEL_INDICES = _indices("destination_level_", 2)
EMPTY_INDICES = tuple(FEATURE_NAMES.index(f"registered_slot_{slot}_level_{level}_id_0")
                      for slot in range(1, 5) for level in range(1, 3))
OBJECT_ID_INDICES = (
    *_indices("moved_object_id_", 4),
    *(FEATURE_NAMES.index(f"registered_slot_{slot}_level_{level}_id_{obj}")
      for slot in range(1, 5) for level in range(1, 3) for obj in range(1, 5)),
    *_indices("active_object_id_", 4),
)
SLOT_XY_INDICES = tuple(FEATURE_NAMES.index(f"registered_slot_{slot}_{axis}_mm/300")
                        for slot in range(1, 5) for axis in ("x", "y"))


@dataclass(frozen=True)
class TargetIssue:
    """A row omitted from TCP supervision, kept outside all model inputs."""

    row_index: int
    reason: str
    detail: str


@dataclass(frozen=True)
class TargetTcpEpisode:
    features_with_ids: torch.Tensor       # [time, 92]
    features_without_ids: torch.Tensor    # [time, 92], same rows and order
    actions: torch.Tensor                 # [time], -1 for unlabelled context
    relative_tcp_pitch: torch.Tensor      # [time, 4], NaN outside target mask
    valid_mask: torch.Tensor              # [time]
    label_mask: torch.Tensor              # [time]
    applied_move_mask: torch.Tensor       # [time], verified by source loader
    tcp_target_mask: torch.Tensor         # [time], derived and IK verified
    evidence: dict                        # audit only, never model input
    issues: tuple[TargetIssue, ...]       # audit only, never model input


@dataclass(frozen=True)
class TargetTcpDataset:
    episodes: tuple[TargetTcpEpisode, ...]  # experience, then direct; source order preserved
    experience_excluded: tuple[experience.Exclusion, ...]
    direct_excluded: tuple[experience.Exclusion, ...]
    source_fingerprints: dict               # audit only, never model input


def _features(value):
    if (not isinstance(value, torch.Tensor) or value.ndim < 1 or
            value.shape[-1] != FEATURE_COUNT or not value.is_floating_point() or
            not bool(torch.isfinite(value).all())):
        raise ValueError("finite floating registered-scene features with last dimension 92 required")
    return value


def mask_object_ids(features):
    """Return an independent, same-width ID-free view of all supplied rows.

    Slot-level ``id_0`` is the existing *empty* indicator, not an identity.
    Eight preserved empty indicators specify each occupancy and their sum also
    determines the number of objects.  No row or registration is sampled.
    """
    result = _features(features).clone()
    result[..., list(OBJECT_ID_INDICES)] = 0
    return result


def registered_occupancy(features):
    """Return four slots by two levels, using only the preserved empty bits."""
    value = _features(features)
    empty = value[..., list(EMPTY_INDICES)]
    if bool(((empty != 0) & (empty != 1)).any()):
        raise ValueError("slot-level empty indicators must be binary")
    return (1 - empty).reshape(*value.shape[:-1], 4, 2)


def _one_hot_index(features, indices, label):
    values = features[..., list(indices)]
    if bool(((values != 0) & (values != 1)).any()) or bool((values.sum(dim=-1) != 1).any()):
        raise ValueError(f"registered {label} must be one-hot")
    return values.argmax(dim=-1)


def registered_goal_mm(features):
    """Extract destination reference in base-frame millimetres, shape [...,3].

    The caller supplies the current, registered (possibly ID-masked) feature
    row.  This is a static nominal reference for the current transfer, not an
    object measurement.  No fallback coordinates or inferred levels are used.
    """
    value = _features(features).to(dtype=torch.float64)
    source = _one_hot_index(value, SOURCE_SLOT_INDICES, "source slot")
    destination = _one_hot_index(value, DESTINATION_SLOT_INDICES, "destination slot")
    _one_hot_index(value, SOURCE_LEVEL_INDICES, "source level")
    level = _one_hot_index(value, DESTINATION_LEVEL_INDICES, "destination level")
    if bool((source == destination).any()):
        raise ValueError("source and destination slots must be distinct")
    slot_xy = value[..., list(SLOT_XY_INDICES)].reshape(*value.shape[:-1], 4, 2) * POSITION_SCALE_MM
    destination_xy = torch.gather(slot_xy, -2, destination[..., None, None].expand(*destination.shape, 1, 2)).squeeze(-2)
    height = BASE_GRIP_HEIGHT_MM + LEVEL_SPACING_MM * level.to(dtype=value.dtype)
    return torch.cat((destination_xy, height.unsqueeze(-1)), dim=-1)


def _pose_and_goal(value, goal, label):
    if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != 4 or
            not value.is_floating_point() or not bool(torch.isfinite(value).all()) or
            not isinstance(goal, torch.Tensor) or goal.shape != (*value.shape[:-1], 3) or
            not goal.is_floating_point() or goal.device != value.device or
            not bool(torch.isfinite(goal).all())):
        raise ValueError(f"finite {label} [...,4] and matching goal [...,3] required")
    return value.to(dtype=torch.float64), goal.to(dtype=torch.float64)


def encode_relative_pose(pose_mm_deg, goal_mm):
    """Encode base XYZ mm and unwrapped pitch degrees relative to goal."""
    pose, goal = _pose_and_goal(pose_mm_deg, goal_mm, "pose")
    return torch.cat(((pose[..., :3] - goal) / POSITION_SCALE_MM,
                      pose[..., 3:] / PITCH_SCALE_DEG), dim=-1)


def decode_relative_pose(relative, goal_mm):
    """Invert the encoding without clipping, IK or a physical command."""
    relative, goal = _pose_and_goal(relative, goal_mm, "relative pose")
    return torch.cat((relative[..., :3] * POSITION_SCALE_MM + goal,
                      relative[..., 3:] * PITCH_SCALE_DEG), dim=-1)


def nominal_pose(joints_deg):
    """Existing public nominal FK plus unwrapped pitch; no device access."""
    if (not isinstance(joints_deg, torch.Tensor) or joints_deg.shape != (4,) or
            not joints_deg.is_floating_point() or not bool(torch.isfinite(joints_deg).all())):
        raise ValueError("four finite applied joint angles required")
    joints = joints_deg.to(dtype=torch.float64, device="cpu").tolist()
    if any(q < lo - 1e-4 or q > hi + 1e-4 for q, (lo, hi) in zip(joints, JOINT_LIMITS)):
        raise ValueError("applied joint angles exceed source loader joint limits")
    xyz = Geometry().forward(joints)
    return torch.tensor([*xyz, 90.0 + sum(joints[1:])], dtype=torch.float64)


def _verify_same_pitch_ik(pose):
    """Require the existing IK branch to reproduce XYZ and unwrapped pitch."""
    xyz, pitch = pose[:3].tolist(), float(pose[3])
    geometry = Geometry()
    try:
        solved = geometry.inverse(xyz, pitch_deg=pitch)
    except ControlError as exc:
        raise ValueError(f"{exc.code}: {exc.message}") from exc
    actual_xyz = geometry.forward(solved)
    if (math.dist(actual_xyz, xyz) > 1e-5 or
            abs(90.0 + sum(solved[1:]) - pitch) > 1e-6):
        raise ValueError("same-pitch IK returned a different nominal pose")


def _convert_episode(source, *, dataset_source=None):
    """Convert a loader episode; this helper does not establish provenance."""
    if not isinstance(source, experience.TrainingEpisode):
        raise ValueError("a verified source TrainingEpisode is required")
    features = _features(source.features)
    if features.ndim != 2 or features.device.type != "cpu" or features.dtype != torch.float32:
        raise ValueError("one CPU float32 registration history is required")
    n = features.shape[0]
    if n < 1:
        raise ValueError("registration cannot be empty")
    for name in ("valid_mask", "label_mask", "applied_move_mask"):
        mask = getattr(source, name)
        if not isinstance(mask, torch.Tensor) or mask.shape != (n,) or mask.dtype != torch.bool or mask.device.type != "cpu":
            raise ValueError(f"invalid {name}")
    if (not isinstance(source.actions, torch.Tensor) or source.actions.shape != (n,) or
            source.actions.dtype != torch.long or source.actions.device.type != "cpu" or
            not isinstance(source.applied_move_targets_deg, torch.Tensor) or
            source.applied_move_targets_deg.shape != (n, 4) or
            source.applied_move_targets_deg.dtype != torch.float32 or
            source.applied_move_targets_deg.device.type != "cpu"):
        raise ValueError("invalid source action or applied target shape")
    valid, labeled, applied = source.valid_mask, source.label_mask, source.applied_move_mask
    if (not bool(valid.all()) or bool((labeled & ~valid).any()) or bool((applied & ~labeled).any()) or
            bool((source.actions[~labeled] != -1).any()) or
            bool(((source.actions[labeled] < 0) | (source.actions[labeled] >= len(motion_policy.ACTIONS))).any()) or
            bool((source.actions[applied] != motion_policy.MOVE).any()) or
            not bool(torch.isnan(source.applied_move_targets_deg[~applied]).all())):
        raise ValueError("inconsistent validity, labels, or applied MOVE masks")
    registration_start = features[:, student.REGISTRATION_START_INDEX]
    transfer_start = features[:, student.TRANSFER_START_INDEX]
    if (bool(((registration_start != 0) & (registration_start != 1)).any()) or
            bool(((transfer_start != 0) & (transfer_start != 1)).any()) or
            registration_start[0] != 1 or transfer_start[0] != 1 or
            int(registration_start.sum()) != 1):
        raise ValueError("each episode must be one complete registration history")

    # Goal comes exclusively from each row's registered task/slots.  This also
    # validates malformed goals on context and non-MOVE rows, not just labels.
    goal = registered_goal_mm(features)
    if not torch.equal(features[:, list(SLOT_XY_INDICES)],
                       features[0, list(SLOT_XY_INDICES)].expand(n, -1)):
        raise ValueError("registered slot coordinates changed during registration")
    task_indices = (*SOURCE_SLOT_INDICES, *DESTINATION_SLOT_INDICES,
                    *SOURCE_LEVEL_INDICES, *DESTINATION_LEVEL_INDICES)
    same_transfer = transfer_start[1:] == 0
    if (not torch.equal(features[1:, list(task_indices)][same_transfer],
                        features[:-1, list(task_indices)][same_transfer]) or
            not torch.equal(goal[1:][same_transfer], goal[:-1][same_transfer])):
        raise ValueError("registered task or reference changed without a transfer boundary")
    with_ids = features.clone()
    without_ids = mask_object_ids(features)
    target = torch.full((n, 4), float("nan"), dtype=torch.float32)
    target_mask = torch.zeros(n, dtype=torch.bool)
    issues = [TargetIssue(index, "move_application_unconfirmed", "no verified applied MOVE target")
              for index in (labeled & (source.actions == motion_policy.MOVE) & ~applied).nonzero().flatten().tolist()]
    for index in applied.nonzero().flatten().tolist():
        try:
            pose = nominal_pose(source.applied_move_targets_deg[index])
        except (ValueError, ControlError) as exc:
            issues.append(TargetIssue(index, "invalid_applied_joint_or_fk", str(exc)))
            continue
        try:
            _verify_same_pitch_ik(pose)
        except (ValueError, ControlError) as exc:
            issues.append(TargetIssue(index, "same_pitch_ik_failed", str(exc)))
            continue
        try:
            encoded = encode_relative_pose(pose, goal[index]).to(dtype=torch.float32)
            if not bool(torch.isfinite(encoded).all()):
                raise ValueError("relative pose is not finite")
        except (ValueError, ControlError) as exc:
            issues.append(TargetIssue(index, "relative_target_invalid", str(exc)))
            continue
        # A representable float32 target can cross a hard IK boundary on
        # decoding, even when the original double-precision FK is feasible.
        try:
            _verify_same_pitch_ik(decode_relative_pose(encoded, goal[index]))
        except (ValueError, ControlError) as exc:
            issues.append(TargetIssue(index, "same_pitch_ik_failed_after_float32", str(exc)))
            continue
        target[index] = encoded
        target_mask[index] = True
    evidence = dict(source.evidence)
    if dataset_source is not None:
        evidence["dataset_source"] = dataset_source
    return TargetTcpEpisode(with_ids, without_ids, source.actions.clone(), target,
                            valid.clone(), labeled.clone(), applied.clone(), target_mask,
                            evidence, tuple(sorted(issues, key=lambda issue: issue.row_index)))


def load_training_dataset(*, summary_path, manifest_path, split_plan_path, fingerprints):
    """Load both pinned train-only sources using their existing verifiers.

    ``fingerprints`` must be externally fixed TrustedFingerprints, not hashes
    computed from the files being accepted.  Source failures/exclusions and the
    three identity-limited takeovers remain in the respective audit evidence.
    Validation/test or final hold-out trajectories are neither loaded nor
    used.  A malformed verified episode fails the load rather than being
    silently discarded; per-MOVE FK/IK failures are retained as ``issues``.
    """
    corrected = experience.load_training_episodes(summary_path, manifest_path=manifest_path,
                                                  split_plan_path=split_plan_path, fingerprints=fingerprints)
    demonstrated = direct.load_direct_training_episodes()
    if hashlib.sha256(Path(split_plan_path).read_bytes()).hexdigest() != demonstrated.split_sha256:
        raise ValueError("experience and direct data must use the same verified split plan")
    episodes = (tuple(_convert_episode(e, dataset_source="student_experience") for e in corrected.episodes) +
                tuple(_convert_episode(e, dataset_source="direct_demonstration") for e in demonstrated.episodes))
    fingerprints_out = {
        "experience_manifest_sha256": corrected.manifest_sha256,
        "experience_summary_sha256": corrected.summary_sha256,
        "experience_collector_sha256": fingerprints.collector_sha256,
        "direct_record_sha256": demonstrated.source_record_sha256,
        "split_sha256": demonstrated.split_sha256,
        "experience_registration_count": len(corrected.episodes),
        "direct_registration_count": len(demonstrated.episodes),
    }
    return TargetTcpDataset(episodes, corrected.excluded, demonstrated.excluded, fingerprints_out)
