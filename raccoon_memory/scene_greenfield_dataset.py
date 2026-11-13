"""Convert the verified train-only direct registrations to cylinder-policy v1.

The 82 recorded public features contain nominal registration/expected scene,
public joints and the gripper command, but no timestamps or joint velocities.
The audited one-second schedule permits a public finite-difference average;
however, a per-axis physical speed limit for v1 normalization is not known.
Until such limits are established, actor velocity fields are zero with a
separate false validity flag; the original actor has no validity input.
No simulator state, later physical assessment or model/source identity is used
as an actor input.  This module does not execute the robot or train a model.

``encode_step`` is also the shared inference encoding.  A caller may supply a
publicly derived velocity only with an independently specified per-axis speed
limit, and is responsible for verifying the underlying observation interval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
import io
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import motion_policy, scene_applied_action_student as old_student
from . import scene_direct_applied_dataset as direct, scene_features, task_catalog


ROOT = Path(__file__).resolve().parents[1]
_IO_PATH = (ROOT / "artifacts/architecture_research/"
            "20261103_shared_greenfield_architecture_review_v1/cylinder_policy_v1/public_io.py")
_spec = importlib.util.spec_from_file_location("_cylinder_policy_v1_public_io", _IO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("the reviewed cylinder-policy public input module is unavailable")
public_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(public_io)

MOVE, OPEN, CLOSE, END, START = public_io.MOVE, public_io.OPEN, public_io.CLOSE, public_io.END, public_io.START
Q_MIN_RAD = np.deg2rad(np.asarray(JOINT_LIMITS, dtype=np.float64)[:, 0])
Q_MAX_RAD = np.deg2rad(np.asarray(JOINT_LIMITS, dtype=np.float64)[:, 1])
# Manufacturer's nominal workspace distance with the standard gripper.  It is
# a normalization length, not a verified workspace at every height or a
# hardware calibration.  The old 82 features use a different 300 mm scale.
# https://github.com/RobomationLAB/RaccoonBot_Guide_KR/wiki/로봇-사양
REACH_M = 0.295
TABLE_Z_M = 0.0
CYLINDER_RADIUS_M = 0.010
CYLINDER_HEIGHT_M = 0.020
MAX_LEVEL = task_catalog.MAX_HEIGHT
MAX_TOKENS = task_catalog.SLOT_COUNT + len(task_catalog.OBJECT_IDS)
FORMAT = "raccoon-scene-greenfield-direct-dataset.v1"


class EncodedStep(NamedTuple):
    tokens: np.ndarray       # [MAX_TOKENS, 11], zero padded
    mask: np.ndarray         # [MAX_TOKENS] bool
    src: int
    dst: int
    robot: np.ndarray        # [30] for four joints
    velocity_valid: bool     # metadata; not an input to the original actor


@dataclass(frozen=True)
class GreenfieldRegistration:
    # Original train_bc.py reads its seven keys and safely ignores extra arrays.
    arrays: dict[str, np.ndarray]
    evidence: dict


@dataclass(frozen=True)
class GreenfieldDataset:
    registrations: tuple[GreenfieldRegistration, ...]
    excluded: tuple
    counts: dict
    source_record_sha256: str
    split_sha256: str


def _one_hot(values, name):
    if not np.isin(values, (0.0, 1.0)).all() or float(np.sum(values)) != 1.0:
        raise ValueError(f"{name} must be one-hot")
    return int(np.argmax(values))


def _legacy_row(features82):
    if isinstance(features82, torch.Tensor):
        features82 = features82.detach().cpu().numpy()
    row = np.asarray(features82, dtype=np.float64)
    if row.shape != (len(scene_features.FEATURE_NAMES),) or not np.isfinite(row).all():
        raise ValueError("features82 must be one finite public 82-feature row")
    if row[4] != 1:
        raise ValueError("joint sensor must be valid")
    source = _one_hot(row[14:18], "source slot") + 1
    destination = _one_hot(row[18:22], "destination slot") + 1
    source_level = _one_hot(row[22:24], "source level") + 1
    destination_level = _one_hot(row[24:26], "destination level") + 1
    moved = _one_hot(row[26:30], "moved object") + 1
    scene = []
    for slot in range(4):
        values = [_one_hot(row[30 + slot * 10 + level * 5:35 + slot * 10 + level * 5],
                           "registered slot level") for level in range(2)]
        if values[0] == 0 and values[1] != 0:
            raise ValueError("registered scene has an object above an empty level")
        scene.append(tuple(v for v in values if v))
    scene, task = scene_features.transfer_task(scene, {
        "source_slot": source, "destination_slot": destination,
        "source_level": source_level, "destination_level": destination_level, "object_id": moved,
    })
    actual_active = np.asarray([float(any(obj in stack for stack in scene)) for obj in range(1, 5)])
    if not np.array_equal(row[70:74], actual_active):
        raise ValueError("registered active objects disagree with the scene")
    command = _one_hot(row[6:9], "gripper command")
    if command == 0:
        raise ValueError("an actual initial gripper command is required")
    return row, scene, task, row[74:82].reshape(4, 2) * scene_features.SLOT_SCALE_MM, command == 2


def _legal_target_rad(target_deg):
    values = np.asarray(target_deg, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("the applied MOVE target must contain four finite degrees")
    bounds = np.asarray(JOINT_LIMITS, dtype=np.float64)
    # Source targets are float32; snap only representational error at an endpoint.
    if np.any(values < bounds[:, 0] - 3e-5) or np.any(values > bounds[:, 1] + 3e-5):
        raise ValueError("the applied MOVE target exceeds physical joint limits")
    return np.deg2rad(np.clip(values, bounds[:, 0], bounds[:, 1]))


def normalize_target_deg(target_deg):
    """Convert recorded degrees to the actor's per-joint [-1, 1] target."""
    return public_io.normalize_joint_target(_legal_target_rad(target_deg), Q_MIN_RAD, Q_MAX_RAD)


def decode_target_deg(target_norm):
    """Decode actor per-joint [-1, 1] targets to robot degrees."""
    return np.rad2deg(public_io.decode_joint_target(target_norm, Q_MIN_RAD, Q_MAX_RAD))


def encode_step(features82, *, previous_mode, last_move_target_deg, new_instruction,
                velocity_deg_s=None, dq_limit_deg_s=None):
    """Encode one public observation for both direct training and inference.

    ``features82`` is the current (not a padded history) row produced by
    ``scene_features.encode_scene_features``.  Previous mode is START at
    registration, END immediately after a completed instruction, or the last
    normally applied MOVE/OPEN/CLOSE.  The last MOVE target persists through
    all other modes, including END.  ``None`` means no MOVE has been applied
    since registration, even if an initial servo target was recorded.

    A supplied velocity must come from public observations and a verified
    interval, with independently known positive axis limits.  With no velocity,
    the zero-filled actor fields are explicitly marked invalid in metadata.
    """
    row, scene, task, xy_mm, closed = _legacy_row(features82)
    if type(new_instruction) is not bool:
        raise ValueError("new_instruction must be boolean")
    if type(previous_mode) is not int or previous_mode not in (MOVE, OPEN, CLOSE, END, START):
        raise ValueError("previous_mode must be a v1 mode")
    if previous_mode == START and not new_instruction:
        raise ValueError("registration START must begin an instruction")
    q_deg = row[:4] * motion_policy.JOINT_SCALE_DEG
    q_rad = np.deg2rad(q_deg)
    target = None if last_move_target_deg is None else _legal_target_rad(last_move_target_deg)
    if (velocity_deg_s is None) != (dq_limit_deg_s is None):
        raise ValueError("velocity and verified per-axis speed limits must be supplied together")
    if velocity_deg_s is None:
        velocity_rad_s, limit_rad_s, velocity_valid = np.zeros(4), np.ones(4), False
    else:
        velocity = np.asarray(velocity_deg_s, dtype=np.float64)
        limit = np.asarray(dq_limit_deg_s, dtype=np.float64)
        if (velocity.shape != (4,) or limit.shape != (4,) or not np.isfinite(velocity).all()
                or not np.isfinite(limit).all() or np.any(limit <= 0)):
            raise ValueError("velocity and positive speed limits must each have four finite values")
        velocity_rad_s, limit_rad_s, velocity_valid = np.deg2rad(velocity), np.deg2rad(limit), True

    tcp_mm = np.asarray(Geometry().forward(q_deg.tolist()), dtype=np.float64)
    yaw, pitch = q_rad[0], q_rad[1:].sum() + math.pi / 2
    axis = np.asarray([-math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    tcp_m = tcp_mm / 1000.0

    # Every object occurs once; a table row occurs only at an empty slot.
    # Only the exposed top row of a stack is a support surface.
    tops, radius, height, levels, objects, supports = [], [], [], [], [], []
    source_index = destination_index = None
    for slot, stack in enumerate(scene, 1):
        if not stack:
            if slot == task["destination_slot"]:
                destination_index = len(tops)
            tops.append([xy_mm[slot-1, 0] / 1000, xy_mm[slot-1, 1] / 1000, TABLE_Z_M])
            radius.append(0.0); height.append(0.0); levels.append(0); objects.append(0); supports.append(1)
        else:
            for level, _object_id in enumerate(stack, 1):
                is_top = level == len(stack)
                if slot == task["source_slot"] and is_top:
                    source_index = len(tops)
                if slot == task["destination_slot"] and is_top:
                    destination_index = len(tops)
                tops.append([xy_mm[slot-1, 0] / 1000, xy_mm[slot-1, 1] / 1000,
                             TABLE_Z_M + level * CYLINDER_HEIGHT_M])
                radius.append(CYLINDER_RADIUS_M); height.append(CYLINDER_HEIGHT_M); levels.append(level)
                objects.append(1); supports.append(int(is_top))
    if source_index is None or destination_index is None or len(tops) > MAX_TOKENS:
        raise ValueError("nominal scene roles cannot be represented")
    active = public_io.build_tokens(tops, radius, height, levels, objects, supports, tcp_m, REACH_M, MAX_LEVEL)
    tokens = np.zeros((MAX_TOKENS, 11), dtype=np.float32)
    mask = np.zeros(MAX_TOKENS, dtype=np.bool_)
    tokens[:len(active)], mask[:len(active)] = active, True
    robot = public_io.build_robot(q_rad, velocity_rad_s, target, Q_MIN_RAD, Q_MAX_RAD, limit_rad_s,
                                  axis, previous_mode, int(closed), int(new_instruction), tcp_m, REACH_M)
    return EncodedStep(tokens, mask, source_index, destination_index, robot, velocity_valid)


def _origins(episode, hashes):
    count = len(episode.actions)
    if episode.evidence["source_kind"] == "direct_operator":
        return np.full(count, "direct_operator", dtype="<U18")
    if episode.evidence["source_kind"] != "student_prefix_direct_takeover":
        raise ValueError("unsupported registration provenance")
    paths = episode.evidence["source_trajectories"]
    if len(paths) != 1:
        raise ValueError("takeover must have exactly one verified source trajectory")
    raw = direct._read_pinned(Path(paths[0]), hashes)
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("takeover row provenance is incomplete")
    result = []
    for i, row in enumerate(rows):
        phase = row.get("phase")
        if phase not in ("student", "teacher"):
            raise ValueError("unknown takeover row provenance")
        if phase == "student" and bool(episode.label_mask[i]):
            raise ValueError("student experience must not be a demonstration label")
        result.append("student_experience" if phase == "student" else "direct_operator")
    return np.asarray(result, dtype="<U18")


def convert_registration(episode, *, row_origin=None):
    """Convert a verified direct registration, preserving labels and history."""
    length = len(episode.actions)
    if (length < 1 or episode.features.shape != (length, old_student.FEATURE_COUNT)
            or not bool(episode.valid_mask.all())):
        raise ValueError("a nonempty, unpadded verified registration is required")
    features = episode.features.detach().cpu().numpy()
    actions = episode.actions.detach().cpu().numpy().astype(np.int64, copy=True)
    labels = episode.label_mask.detach().cpu().numpy().astype(np.bool_, copy=True)
    moves = episode.applied_move_mask.detach().cpu().numpy().astype(np.bool_, copy=True)
    targets_deg = episode.applied_move_targets_deg.detach().cpu().numpy()
    registration_start = features[:, old_student.REGISTRATION_START_INDEX] == 1
    transfer_start = features[:, old_student.TRANSFER_START_INDEX] == 1
    if (not registration_start[0] or registration_start.sum() != 1 or not transfer_start[0]
            or not np.array_equal(actions == -1, ~labels)
            or not np.array_equal((actions == MOVE) & labels, moves)):
        raise ValueError("registration boundaries or supervision masks differ")
    if np.any((actions < -1) | (actions > END)) or actions[-1] != END:
        raise ValueError("the verified registration must end with an explicit END")
    if not np.array_equal(transfer_start[1:], actions[:-1] == END):
        raise ValueError("instruction boundaries must follow the preceding explicit END")
    if row_origin is None and episode.evidence.get("source_kind") == "student_prefix_direct_takeover":
        raise ValueError("takeover row provenance must be explicitly supplied")
    if row_origin is None:
        row_origin = np.full(length, "direct_operator", dtype="<U18")
    row_origin = np.asarray(row_origin)
    if (row_origin.shape != (length,) or not np.isin(row_origin, ("direct_operator", "student_experience")).all()
            or np.any(labels & (row_origin == "student_experience"))):
        raise ValueError("row provenance disagrees with supervision")
    tokens, mask, src, dst, robot, velocity_valid = [], [], [], [], [], []
    targets = np.full((length, 4), np.nan, dtype=np.float32)
    last_move, mode = None, START
    for t in range(length):
        if t:
            prior = int(actions[t-1])
            physical_index = _one_hot(features[t, old_student.PREVIOUS_ACTION_SLICE], "prior applied action")
            if prior == END:
                mode = END
            else:
                if physical_index == 0:
                    raise ValueError("a nonterminal decision lacks a normally applied action")
                mode = (MOVE, OPEN, CLOSE)[physical_index-1]
                if prior != -1 and prior != mode:
                    raise ValueError("the next observation contradicts the applied action")
                if mode == MOVE:
                    recorded = features[t, old_student.PREVIOUS_TARGET_SLICE] * motion_policy.JOINT_SCALE_DEG
                    if prior == MOVE and not np.allclose(recorded, targets_deg[t-1], atol=3e-5, rtol=0):
                        raise ValueError("the next observation contradicts the applied MOVE target")
                    # Preserve the original MOVE return for labels, and the verified
                    # next public applied-target record for unlabelled context.
                    last_move = targets_deg[t-1] if prior == MOVE else recorded
        step = encode_step(features[t, :82], previous_mode=mode, last_move_target_deg=last_move,
                           new_instruction=bool(transfer_start[t]))
        tokens.append(step.tokens); mask.append(step.mask); src.append(step.src); dst.append(step.dst)
        robot.append(step.robot); velocity_valid.append(step.velocity_valid)
        if moves[t]:
            targets[t] = normalize_target_deg(targets_deg[t])
    arrays = {"tokens": np.stack(tokens), "mask": np.stack(mask),
              "src": np.asarray(src, dtype=np.int64), "dst": np.asarray(dst, dtype=np.int64),
              "robot": np.stack(robot), "actions": actions, "targets": targets,
              "velocity_valid": np.asarray(velocity_valid, dtype=np.bool_),
              "registration_start": registration_start, "instruction_start": transfer_start,
              "label_mask": labels, "row_origin": row_origin.astype("<U18")}
    return GreenfieldRegistration(arrays, dict(episode.evidence))


def load_greenfield_training_episodes():
    """Verify and convert the pinned 106-registration direct training corpus."""
    source = direct.load_direct_training_episodes()
    raw = direct.CORPUS_RECORD.read_bytes()
    if direct._digest(raw) != source.source_record_sha256:
        raise ValueError("direct corpus record changed after verification")
    record = json.loads(raw)["teacher_source"]
    hashes = {direct._key(path): digest for path, digest in record["source_sha256"].items()}
    registrations = tuple(convert_registration(e, row_origin=_origins(e, hashes)) for e in source.episodes)
    counts = Counter(source.counts)
    counts["velocity_valid_observations"] = sum(int(r.arrays["velocity_valid"].sum()) for r in registrations)
    counts["student_experience_rows"] = sum(int((r.arrays["row_origin"] == "student_experience").sum())
                                            for r in registrations)
    counts["direct_operator_rows"] = sum(int((r.arrays["row_origin"] == "direct_operator").sum())
                                         for r in registrations)
    return GreenfieldDataset(registrations, source.excluded, dict(counts),
                             source.source_record_sha256, source.split_sha256)
