"""Prepare separately sourced, explicit velocity journals for training.

One manifest entry names one original journal and its externally evidenced
start group. No replay source, concatenation, ID permutation or resampling is
accepted. Sequences remain compact: history windows are formed per batch.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from . import direct_velocity_training_data as direct, scene_features, scene_trajectory_policy as policy
from scripts import create_direct_velocity_manifest as manifest_tools


MANIFEST_SCHEMA = "raccoon-direct-velocity-source-manifest.v1"
POSITION_LABELS = ("next_observed_joint", "applied_position_servo_target", "current_applied_position_servo_target")


def prepare_direct_manifest(payload: dict, manifest_dir: str | Path, *, position_label: str | None = None) -> dict:
    if (not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA
            or not isinstance(payload.get("entries"), list) or not payload["entries"]):
        raise ValueError("a nonempty direct-command manifest is required")
    if position_label not in POSITION_LABELS:
        raise ValueError("direct position label must be explicitly selected")
    base = Path(manifest_dir).resolve()
    if (payload.get("actual_hardware") is not False or payload.get("identity_augmentation") is not False
            or payload.get("episodes_joined") is not False or not isinstance(payload.get("source_plan_path"), str)):
        raise ValueError("direct manifest must declare simulated, unaugmented and separate episodes")
    plan_path = Path(payload["source_plan_path"])
    plan_path = (base / plan_path).resolve() if not plan_path.is_absolute() else plan_path.resolve()
    plan_raw = plan_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != payload.get("source_plan_sha256"):
        raise ValueError("direct collection plan fingerprint differs")
    plan = json.loads(plan_raw)
    if plan.get("schema") != "raccoon-direct-velocity-collection-plan.v3" or not isinstance(plan.get("cases"), list):
        raise ValueError("unsupported direct collection plan")
    cases = {case["case_id"]: case for case in plan["cases"]}
    if len(cases) != len(plan["cases"]):
        raise ValueError("duplicate direct collection case IDs")
    parts = {"train": [], "validation": []}
    groups = {"train": set(), "validation": set()}
    episode_spans = {"train": [], "validation": []}
    next_offset = {"train": 0, "validation": 0}
    integrity_entries, sources, episode_summaries = [], [], []
    seen_journal_digests = set()
    incomplete_unconfirmed = 0
    excluded_first_samples = excluded_single_sample_episodes = 0
    for entry in payload["entries"]:
        if not isinstance(entry, dict) or entry.get("split") not in ("train", "validation", "excluded"):
            raise ValueError("direct manifest entry requires a valid split")
        group = entry.get("source_group_id")
        path = entry.get("journal_path")
        if not isinstance(group, str) or not group.strip() or not isinstance(path, str) or not path.strip():
            raise ValueError("direct manifest entry requires a source group and journal path")
        journal = Path(path)
        journal = (base / journal).resolve() if not journal.is_absolute() else journal.resolve()
        journal_raw = journal.read_bytes()
        journal_sha = hashlib.sha256(journal_raw).hexdigest()
        if journal_sha != entry.get("journal_sha256") or journal_sha in seen_journal_digests:
            raise ValueError("direct journal fingerprint differs or source is duplicated")
        seen_journal_digests.add(journal_sha)
        case = cases.get(entry.get("case_id"))
        if (not isinstance(case, dict) or manifest_tools.start_recipe(case) != entry.get("source_group_basis")
                or manifest_tools.start_group(case) != group):
            raise ValueError("direct source group or plan case differs")
        source_lines = journal_raw.decode("utf-8").splitlines()
        if not source_lines:
            raise ValueError("direct journal is empty")
        ready = json.loads(source_lines[0])
        observation, simulation = ready.get("observation"), ready.get("simulation")
        if (ready.get("event") != "ready" or not isinstance(observation, dict) or not isinstance(simulation, dict)
                or observation.get("registered_scene") != case["registered_scene"]
                or observation.get("registered_slot_xy_mm") != case["registered_slot_xy_mm"]
                or observation.get("task") != case["task"] or simulation.get("seed") != case["physics_seed"]
                or simulation.get("randomize") is not case["randomize"]):
            raise ValueError("direct journal start or task differs from the collection plan")
        if entry.get("journal_complete") is False:
            last_event = json.loads(source_lines[-1]).get("event")
            if (entry["split"] != "excluded" or entry.get("posthoc_status") != "unconfirmed"
                    or entry.get("samples") is not None or last_event == "finished"
                    or entry.get("last_recorded_event") != last_event):
                raise ValueError("incomplete journal must remain excluded and unconfirmed")
            sources.append({"episode_id": None, "source_group_id": group, "split": "excluded",
                            "journal_path": str(journal), "journal_sha256": journal_sha,
                            "posthoc_status": "unconfirmed", "recorded_sample_count": None,
                            "initial_rest_evidence": "unconfirmed", "initial_state_sha256": None})
            incomplete_unconfirmed += 1
            continue
        source = direct.read_event_journal(journal, source_group_id=group)
        metadata, episode = source["metadata"], source["episodes"][0]
        samples = episode["samples"]
        expected_split = manifest_tools.internal_split(group) if metadata["posthoc_status"] == "success" else "excluded"
        if (metadata["source_sha256"] != entry.get("journal_sha256")
                or metadata["posthoc_status"] != entry.get("posthoc_status")
                or entry["split"] != expected_split or entry.get("samples") != len(samples)
                or entry.get("submitted_decisions") != metadata.get("submitted_decisions")
                or entry.get("first_moving_valid") != samples[0]["input_moving_valid"]):
            raise ValueError("direct journal fingerprint, status, split or sample count differs")
        compact = {"schema": source["schema"], "metadata": dict(metadata),
                   "episodes": [{"source_group_id": group,
                                 "samples": [{"sample_id": row["sample_id"]} for row in samples]}]}
        integrity_entry = {"data": compact, "split": entry["split"]}
        if "initial_state_sha256" in entry:
            integrity_entry["initial_state_sha256"] = entry["initial_state_sha256"]
        integrity_entries.append(integrity_entry)
        summary = {"episode_id": episode["episode_id"], "source_group_id": group, "split": entry["split"],
                   "journal_path": str(journal), "journal_sha256": metadata["source_sha256"],
                   "posthoc_status": metadata["posthoc_status"], "recorded_sample_count": len(samples),
                   "submitted_decisions": metadata["submitted_decisions"],
                   "initial_rest_evidence": metadata["initial_rest_evidence"],
                   "initial_state_sha256": entry.get("initial_state_sha256")}
        sources.append(summary)
        if entry["split"] == "excluded":
            continue
        if metadata["posthoc_status"] != "success":
            raise ValueError("failed or unconfirmed episode cannot enter a success fit")
        if any(type(row.get("input_moving_valid")) is not bool for row in samples):
            raise ValueError("each direct sample requires moving validity")
        invalid = [i for i, row in enumerate(samples) if not row["input_moving_valid"]]
        if invalid not in ([], [0]):
            raise ValueError("only an unsupported first moving observation may be excluded")
        if invalid:
            excluded_first_samples += 1
            samples = samples[1:]
        if not samples:
            excluded_single_sample_episodes += 1
            continue
        split = entry["split"]
        groups[split].add(group)
        count = len(samples)
        feature_rows = [row["features"] for row in samples]
        if any(len(row) != len(scene_features.FEATURE_NAMES) for row in feature_rows):
            raise ValueError("invalid direct feature size")
        features = torch.tensor(feature_rows, dtype=torch.float32)
        next_observed = torch.tensor([row["target_joint_deg"] for row in samples], dtype=torch.float32)
        applied_servo = torch.tensor([row["applied_position_servo_target_deg"] for row in samples], dtype=torch.float32)
        current_servo = torch.tensor([row["current_applied_position_servo_target_deg"] for row in samples], dtype=torch.float32)
        positions = {"next_observed_joint": next_observed, "applied_position_servo_target": applied_servo,
                     "current_applied_position_servo_target": current_servo}[position_label]
        velocities = torch.tensor([row["target_velocity_deg_s"] for row in samples], dtype=torch.float32)
        if not bool(torch.isfinite(features).all() and torch.isfinite(next_observed).all()
                    and torch.isfinite(applied_servo).all() and torch.isfinite(current_servo).all()
                    and torch.isfinite(velocities).all()):
            raise ValueError("direct observations and labels must be finite")
        grippers = torch.tensor([policy.GRIPPER_STATES.index(row["target_gripper"]) for row in samples], dtype=torch.long)
        terminals = torch.tensor([row["terminal"] for row in samples], dtype=torch.float32)
        start = next_offset[split]
        parts[split].append({"features": features, "episode_starts": torch.full((count,), start, dtype=torch.long),
                             "positions": positions / policy.JOINT_SCALE_DEG,
                             "next_observed_positions": next_observed / policy.JOINT_SCALE_DEG,
                             "applied_servo_positions": applied_servo / policy.JOINT_SCALE_DEG,
                             "current_servo_positions": current_servo / policy.JOINT_SCALE_DEG,
                             "velocities": velocities / policy.VELOCITY_SCALE_DEG_S,
                             "grippers": grippers, "terminals": terminals})
        episode_spans[split].append({"episode_id": episode["episode_id"], "offset": start, "count": count})
        next_offset[split] += count
        episode_summaries.append({"episode_id": episode["episode_id"], "source_group_id": group, "split": split,
                                  "sample_count": count, "first_moving_sample_excluded": bool(invalid)})
    integrity = direct.validate_partition_integrity(integrity_entries)
    if any(not parts[split] for split in parts):
        raise ValueError("both training and internal validation need usable successful direct journals")
    result = {split: {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
              for split, pieces in parts.items()}
    source_counts = {split: {"journals": sum(source["split"] == split for source in sources),
                             "submitted_decisions": sum(source.get("submitted_decisions") or 0 for source in sources
                                                        if source["split"] == split),
                             "recorded_samples": sum(source.get("recorded_sample_count") or 0 for source in sources
                                                     if source["split"] == split)}
                     for split in ("train", "validation", "excluded")}
    result.update(step_seconds=direct.SAMPLE_SECONDS, groups={key: sorted(value) for key, value in groups.items()},
                  episode_spans=episode_spans, episode_summaries=episode_summaries,
                  source_metadata={"source_kind": "explicitly_submitted_velocity_commands_actor_unverified",
                    "position_label": position_label,
                    "position_target": {"next_observed_joint": "next simulated observed angle, not an independently submitted angle",
                        "applied_position_servo_target": "end-of-interval servo target integrated from submitted velocity, not an independently submitted angle",
                        "current_applied_position_servo_target": "start-of-interval servo target, not an independently submitted angle"}[position_label],
                    "velocity_target": "explicitly submitted signed joint velocity",
                    "applied_position_servo_target_deg": "separately preserved in original journal; integrated submitted velocity",
                    "resampled": False, "identity_augmentation": False, "journal_concatenation": False,
                    "initial_unknown_moving_samples_excluded": excluded_first_samples,
                    "unusable_single_sample_episodes": excluded_single_sample_episodes,
                    "incomplete_unconfirmed_journals": incomplete_unconfirmed,
                    "partition_integrity": integrity, "source_groups_and_state_digests": "caller supplied; evidence must be verified",
                    "source_counts": source_counts,
                    "sources": sources})
    return result


def gripper_sample_strata(data: dict, episode_spans: list[dict], *, radius_ticks: int = 25) -> dict:
    """Partition recorded rows by a real command transition, without crossing episodes.

    The neighborhood is used only for choosing training rows; no future feature
    or altered command is supplied to the model. The three sets do not overlap.
    """
    if type(radius_ticks) is not int or radius_ticks < 1:
        raise ValueError("radius_ticks must be a positive integer")
    features, truth = data["features"], data["grippers"]
    count = len(truth)
    if (not isinstance(features, torch.Tensor) or features.ndim != 2 or len(features) != count
            or features.shape[1] != len(scene_features.FEATURE_NAMES) or count == 0
            or not isinstance(truth, torch.Tensor) or truth.ndim != 1 or truth.dtype != torch.long
            or not bool(((truth == 0) | (truth == 1)).all())):
        raise ValueError("valid compact direct gripper observations and labels are required")
    open_index = scene_features.FEATURE_NAMES.index("gripper_command_open")
    closed_index = scene_features.FEATURE_NAMES.index("gripper_command_closed")
    previous_open, previous_closed = features[:, open_index] == 1, features[:, closed_index] == 1
    if not bool((previous_open ^ previous_closed).all()):
        raise ValueError("recorded previous gripper command must be known")
    previous = previous_closed.long()
    transition = truth != previous
    nearby = torch.zeros(count, dtype=torch.bool)
    next_offset = 0
    if not isinstance(episode_spans, list) or not episode_spans:
        raise ValueError("episode spans are required")
    for span in episode_spans:
        if not isinstance(span, dict):
            raise ValueError("invalid episode span")
        start, length = span.get("offset"), span.get("count")
        if (type(start) is not int or type(length) is not int or start != next_offset
                or length < 1 or start + length > count):
            raise ValueError("episode spans must cover the data contiguously")
        end = start + length
        for index in (torch.nonzero(transition[start:end], as_tuple=False).flatten() + start).tolist():
            nearby[max(start, index-radius_ticks):min(end, index+radius_ticks+1)] = True
        next_offset = end
    if next_offset != count:
        raise ValueError("episode spans must cover the data contiguously")
    nearby &= ~transition
    ordinary = ~(transition | nearby)
    return {"transition": torch.nonzero(transition, as_tuple=False).flatten(),
            "nearby_hold": torch.nonzero(nearby, as_tuple=False).flatten(),
            "ordinary_hold": torch.nonzero(ordinary, as_tuple=False).flatten(),
            "previous": previous}


def _sample_gripper_indices(strata: dict, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw an equal share from each unmodified row category, with replacement."""
    if type(batch_size) is not int or batch_size < 3 or batch_size % 3:
        raise ValueError("gripper batch size must be positive and divisible by three")
    chosen = []
    for name in ("transition", "nearby_hold", "ordinary_hold"):
        pool = strata[name]
        if not len(pool):
            raise ValueError(f"training data requires at least one {name} row")
        chosen.append(pool[torch.randint(len(pool), (batch_size // 3,), generator=generator)])
    return torch.cat(chosen)


def _sample_directional_gripper_indices(strata: dict, truth: torch.Tensor,
                                        batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw the specified 4 close, 4 open, 32 nearby and 88 ordinary rows."""
    if type(batch_size) is not int or batch_size != 128:
        raise ValueError("directional gripper batch size must be 128")
    transition = strata["transition"]
    pools = (("close_transition", transition[truth[transition] == 1], 4),
             ("open_transition", transition[truth[transition] == 0], 4),
             ("nearby_hold", strata["nearby_hold"], 32),
             ("ordinary_hold", strata["ordinary_hold"], 88))
    chosen = []
    for name, pool, count in pools:
        if not len(pool):
            raise ValueError(f"training data requires at least one {name} row")
        chosen.append(pool[torch.randint(len(pool), (count,), generator=generator)])
    return torch.cat(chosen)


def recorded_gripper_transition_metrics(model, data: dict, episode_spans: list[dict], *, radius_ticks: int) -> dict:
    """Score every recorded observation, including separate close/open misses and extra commands."""
    from . import scene_trajectory_training as training

    strata = gripper_sample_strata(data, episode_spans, radius_ticks=radius_ticks)
    truth, previous = data["grippers"], strata["previous"]
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(truth), 512):
            windows, masks = training._batch_windows(model, data, slice(start, min(start+512, len(truth))))
            predictions.append(model(windows, masks)[2].argmax(-1))
    predicted = torch.cat(predictions)
    expected_change, proposed_change = truth != previous, predicted != previous
    tp = int((expected_change & proposed_change).sum())
    fp = int((~expected_change & proposed_change).sum())
    fn = int((expected_change & ~proposed_change).sum())
    directions = {}
    for state, target in (("close", 1), ("open", 0)):
        actual = expected_change & (truth == target)
        extra = ~expected_change & proposed_change & (predicted == target)
        directions[state] = {"actual_transitions": int(actual.sum()),
                             "detected_at_recorded_tick": int((actual & proposed_change).sum()),
                             "missed_at_recorded_tick": int((actual & ~proposed_change).sum()),
                             "unnecessary_transitions": int(extra.sum())}
    hold_metrics = {}
    for name in ("nearby_hold", "ordinary_hold"):
        indices = strata[name]
        hold_metrics[name] = {"samples": len(indices),
                              "unnecessary_transitions": int(proposed_change[indices].sum())}
    return {"samples": len(truth), "correct": int((truth == predicted).sum()),
            "accuracy": float((truth == predicted).float().mean()),
            "actual_transitions": int(expected_change.sum()), "detected_at_recorded_tick": tp,
            "missed_at_recorded_tick": fn, "unnecessary_transitions": fp,
            "transition_precision": tp / (tp + fp) if tp + fp else None,
            "transition_recall": tp / (tp + fn) if tp + fn else None,
            "directions": directions, "hold_rows": hold_metrics}


def fit_gripper_head(model, data: dict, episode_spans: list[dict], *, seed: int, updates: int,
                     batch_size: int, learning_rate: float, radius_ticks: int,
                     sampling: str = "equal_three") -> dict:
    """Fit only the existing gripper head on selected, unmodified real rows."""
    from . import scene_trajectory_training as training

    if type(seed) is not int or seed < 0 or type(updates) is not int or updates < 1:
        raise ValueError("seed must be nonnegative and updates must be positive")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if sampling not in ("equal_three", "directional_4_4_32_88"):
        raise ValueError("unsupported gripper sampling")
    strata = gripper_sample_strata(data, episode_spans, radius_ticks=radius_ticks)
    generator = torch.Generator().manual_seed(seed)
    def sample(source_generator):
        if sampling == "directional_4_4_32_88":
            return _sample_directional_gripper_indices(strata, data["grippers"], batch_size, source_generator)
        return _sample_gripper_indices(strata, batch_size, source_generator)
    # Validate both the pool and batch size before changing any model parameter.
    sample(torch.Generator().manual_seed(seed))
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("gripper_head."))
    # Freeze the shared encoder and cache the identical causal representations.
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(data["grippers"]), 512):
            windows, masks = training._batch_windows(model, data, slice(start, min(start+512, len(data["grippers"]))))
            masked = windows.masked_fill(~masks.unsqueeze(-1), 0.0)
            encoded.append(model.encoder(torch.cat((masked, masks.unsqueeze(-1).to(windows.dtype)), dim=-1).flatten(1)))
    encoded = torch.cat(encoded)
    optimizer = torch.optim.Adam(model.gripper_head.parameters(), lr=float(learning_rate))
    last_loss = None
    for _ in range(updates):
        indices = sample(generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.gripper_head(encoded[indices]), data["grippers"][indices])
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite gripper training loss")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    frozen = [name for name in original if not name.startswith("gripper_head.")]
    changed_frozen = [name for name in frozen if not torch.equal(original[name], model.state_dict()[name])]
    changed_head = [name for name in original if name.startswith("gripper_head.")
                    and not torch.equal(original[name], model.state_dict()[name])]
    if changed_frozen or not changed_head or any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("head-only training did not preserve finite frozen weights")
    allocation = ({"close_transition": 4, "open_transition": 4, "nearby_hold": 32, "ordinary_hold": 88}
                  if sampling == "directional_4_4_32_88" else
                  {name: batch_size // 3 for name in ("transition", "nearby_hold", "ordinary_hold")})
    return {"train_stratum_counts": {name: len(strata[name]) for name in ("transition", "nearby_hold", "ordinary_hold")},
            "train_transition_direction_counts": {
                "close": int((data["grippers"][strata["transition"]] == 1).sum()),
                "open": int((data["grippers"][strata["transition"]] == 0).sum())},
            "samples_per_stratum_per_batch": batch_size // 3 if sampling == "equal_three" else None,
            "batch_allocation": allocation, "final_sampled_batch_cross_entropy": last_loss,
            "changed_head_parameters": changed_head, "frozen_parameters_unchanged": frozen}


def train_gripper_head(manifest_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path, *,
                       seed: int = 20261030, updates: int = 3000, batch_size: int = 126,
                       learning_rate: float = .001, radius_ticks: int = 25,
                       sampling: str = "equal_three") -> dict:
    """Create a separate, provenance-bound simulation candidate; never change execution semantics."""
    from . import (direct_velocity_training_data, motion_policy, robot_observation_features,
                   scene_trajectory_training as training, task_catalog)

    manifest_path, checkpoint_path, output_dir = map(lambda p: Path(p).resolve(),
                                                      (manifest_path, checkpoint_path, output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    manifest_raw, checkpoint_raw = manifest_path.read_bytes(), checkpoint_path.read_bytes()
    manifest_sha, checkpoint_sha = hashlib.sha256(manifest_raw).hexdigest(), hashlib.sha256(checkpoint_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = policy.load_checkpoint(checkpoint)
    if checkpoint["provenance"].get("dataset_sha256") != manifest_sha:
        raise ValueError("base checkpoint and direct manifest must use the identical recorded split")
    if (model.actuation_semantics != "integrated_velocity_with_position_target"
            or model.direct_position_label != "current_applied_position_servo_target"):
        raise ValueError("base checkpoint must have the integrated direct-velocity contract")
    prepared = prepare_direct_manifest(manifest, manifest_path.parent, position_label=model.direct_position_label)
    if model.step_seconds != prepared["step_seconds"]:
        raise ValueError("base checkpoint and direct manifest step periods differ")
    code_paths = [Path(path).resolve() for path in (__file__, policy.__file__, training.__file__,
                  direct_velocity_training_data.__file__, scene_features.__file__, motion_policy.__file__,
                  robot_observation_features.__file__,
                  task_catalog.__file__, manifest_tools.__file__, Path(__file__).resolve().parents[1] /
                  "raccoon_control" / "kinematics.py", Path(__file__).resolve().parents[1] /
                  "scripts" / "train_scene_trajectory_gripper_head.py")]
    code_snapshots = {path.name: path.read_bytes() for path in code_paths}
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        initial = {name: recorded_gripper_transition_metrics(model, prepared[name], prepared["episode_spans"][name],
                                                              radius_ticks=radius_ticks)
                   for name in ("train", "validation")}
        fit = fit_gripper_head(model, prepared["train"], prepared["episode_spans"]["train"], seed=seed,
                               updates=updates, batch_size=batch_size, learning_rate=learning_rate,
                               radius_ticks=radius_ticks, sampling=sampling)
        final = {name: recorded_gripper_transition_metrics(model, prepared[name], prepared["episode_spans"][name],
                                                            radius_ticks=radius_ticks)
                 for name in ("train", "validation")}
        base_model = policy.load_checkpoint(checkpoint)
        differences = {name: {"joint_target": 0.0, "joint_velocity": 0.0, "done_logit": 0.0}
                       for name in ("train", "validation")}
        with torch.no_grad():
            for name in differences:
                data = prepared[name]
                for start in range(0, len(data["grippers"]), 512):
                    windows, masks = training._batch_windows(model, data, slice(start, min(start+512, len(data["grippers"]))))
                    before, after = base_model(windows, masks), model(windows, masks)
                    for key, i in (("joint_target", 0), ("joint_velocity", 1), ("done_logit", 3)):
                        differences[name][key] = max(differences[name][key], float((before[i]-after[i]).abs().max()))
        if any(value != 0.0 for group in differences.values() for value in group.values()):
            raise ValueError("frozen non-gripper outputs changed on recorded observations")
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    if training._file_sha256(manifest_path) != manifest_sha or training._file_sha256(checkpoint_path) != checkpoint_sha:
        raise ValueError("manifest or base checkpoint changed during training")
    plan_path = Path(manifest["source_plan_path"])
    if not plan_path.is_absolute():
        plan_path = manifest_path.parent / plan_path
    if (training._file_sha256(plan_path) != manifest["source_plan_sha256"]
            or any(training._file_sha256(source["journal_path"]) != source["journal_sha256"]
                   for source in prepared["source_metadata"]["sources"])):
        raise ValueError("a direct journal or collection plan changed during training")
    if any(path.read_bytes() != code_snapshots[path.name] for path in code_paths):
        raise ValueError("training code changed during training")
    config = {"seed": seed, "updates": updates, "batch_size": batch_size, "learning_rate": float(learning_rate),
              "optimizer": "adam", "weight_decay": 0.0, "radius_ticks": radius_ticks,
              "radius_seconds": radius_ticks * model.step_seconds,
              "sampling": ("4 close, 4 open, 32 nearby hold, 88 ordinary hold; seeded with replacement"
                           if sampling == "directional_4_4_32_88" else
                           "equal transition, nearby hold, ordinary hold; seeded with replacement"),
              "sampling_mode": sampling, "batch_allocation": fit["batch_allocation"],
              "loss": "cross entropy on recorded gripper targets only", "labels_modified": False,
              "frozen": "encoder, joint head, velocity head and done head",
              "runtime_or_command_changes": False}
    provenance = dict(checkpoint["provenance"])
    provenance["gripper_head_retraining"] = {"base_checkpoint_path": str(checkpoint_path),
                                             "base_checkpoint_sha256": checkpoint_sha,
                                             "manifest_sha256": manifest_sha, "configuration": config}
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for name, content in code_snapshots.items():
        (source_dir / name).write_bytes(content)
    candidate_path = output_dir / "trajectory_student.pt"
    with candidate_path.open("xb") as stream:
        torch.save(policy.checkpoint_payload(model, provenance=provenance), stream)
    report = {"schema": "raccoon-direct-trajectory-gripper-head-training-report.v1",
              "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
              "base_checkpoint_path": str(checkpoint_path), "base_checkpoint_sha256": checkpoint_sha,
              "checkpoint_path": str(candidate_path), "checkpoint_sha256": training._file_sha256(candidate_path),
              "configuration": config, "groups": prepared["groups"],
              "episode_counts": {name: len(prepared["episode_spans"][name]) for name in ("train", "validation")},
              "fit": fit, "initial_recorded_metrics": initial, "final_recorded_metrics": final,
              "frozen_recorded_output_max_abs_differences": differences,
              "source_integrity_rechecked": True,
              "code_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in code_snapshots.items()},
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "limitations": ["actor identity is unverified", "validation is internal to the original direct collection",
                              "row-level metrics use teacher-recorded observations, not closed-loop success",
                              "oversampling means head scores are not calibrated to natural transition frequency",
                              "training samples may use proximity to future transitions only for selection, never as model input",
                              "no physical device or microphone was used"]}
    with (output_dir / "training_report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report


def fit_all_heads(model, data: dict, episode_spans: list[dict], *, seed: int, updates: int,
                  batch_size: int, learning_rate: float, radius_ticks: int) -> dict:
    """Jointly fit the existing encoder and every original action head."""
    from . import scene_trajectory_training as training

    if type(seed) is not int or seed < 0 or type(updates) is not int or updates < 1:
        raise ValueError("seed must be nonnegative and updates must be positive")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    strata = gripper_sample_strata(data, episode_spans, radius_ticks=radius_ticks)
    _sample_directional_gripper_indices(strata, data["grippers"], batch_size,
                                        torch.Generator().manual_seed(seed))
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(seed)
    terminal_occurrences, terminal_rows = 0, set()
    last_loss = None
    for _ in range(updates):
        indices = _sample_directional_gripper_indices(strata, data["grippers"], batch_size, generator)
        terminal = indices[data["terminals"][indices].bool()]
        terminal_occurrences += len(terminal)
        terminal_rows.update(terminal.tolist())
        optimizer.zero_grad(set_to_none=True)
        # Exact original multi-head loss, with original labels for each sampled row.
        loss = training._loss(model, data, indices)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite joint training loss")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    changed = [name for name in original if not torch.equal(original[name], model.state_dict()[name])]
    if not changed or any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("joint training did not preserve finite model weights")
    return {"train_stratum_counts": {name: len(strata[name]) for name in ("transition", "nearby_hold", "ordinary_hold")},
            "train_transition_direction_counts": {
                "close": int((data["grippers"][strata["transition"]] == 1).sum()),
                "open": int((data["grippers"][strata["transition"]] == 0).sum())},
            "batch_allocation": {"close_transition": 4, "open_transition": 4,
                                 "nearby_hold": 32, "ordinary_hold": 88},
            "sampled_terminal_occurrences": terminal_occurrences,
            "unique_terminal_rows_sampled": len(terminal_rows),
            "available_terminal_rows": int(data["terminals"].bool().sum()),
            "changed_parameter_tensors": changed, "final_sampled_batch_loss": last_loss}


def train_all_heads(manifest_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path, *,
                    seed: int = 20261030, updates: int = 200, batch_size: int = 128,
                    learning_rate: float = .001, radius_ticks: int = 5) -> dict:
    """Create one direct-data multi-head candidate from an identified checkpoint."""
    from . import (direct_velocity_training_data, motion_policy, robot_observation_features,
                   scene_trajectory_training as training, task_catalog)

    manifest_path, checkpoint_path, output_dir = map(lambda p: Path(p).resolve(),
                                                      (manifest_path, checkpoint_path, output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    manifest_raw, checkpoint_raw = manifest_path.read_bytes(), checkpoint_path.read_bytes()
    manifest_sha, checkpoint_sha = hashlib.sha256(manifest_raw).hexdigest(), hashlib.sha256(checkpoint_raw).hexdigest()
    manifest = json.loads(manifest_raw)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = policy.load_checkpoint(checkpoint)
    if checkpoint["provenance"].get("dataset_sha256") != manifest_sha:
        raise ValueError("base checkpoint and direct manifest must use the identical recorded split")
    if (model.actuation_semantics != "integrated_velocity_with_position_target"
            or model.direct_position_label != "current_applied_position_servo_target"):
        raise ValueError("base checkpoint must have the integrated direct-velocity contract")
    prepared = prepare_direct_manifest(manifest, manifest_path.parent, position_label=model.direct_position_label)
    if model.step_seconds != prepared["step_seconds"]:
        raise ValueError("base checkpoint and direct manifest step periods differ")
    code_paths = [Path(path).resolve() for path in (__file__, policy.__file__, training.__file__,
                  direct_velocity_training_data.__file__, scene_features.__file__, motion_policy.__file__,
                  robot_observation_features.__file__, task_catalog.__file__, manifest_tools.__file__,
                  Path(__file__).resolve().parents[1] / "raccoon_control" / "kinematics.py",
                  Path(__file__).resolve().parents[1] / "scripts" / "train_scene_trajectory_all_heads.py")]
    code_snapshots = {path.name: path.read_bytes() for path in code_paths}
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        def metrics():
            return {name: {"all_outputs": training.recorded_metrics(model, prepared[name],
                                            episode_spans=prepared["episode_spans"][name]),
                           "gripper_transitions": recorded_gripper_transition_metrics(
                               model, prepared[name], prepared["episode_spans"][name], radius_ticks=radius_ticks)}
                    for name in ("train", "validation")}
        initial = metrics()
        fit = fit_all_heads(model, prepared["train"], prepared["episode_spans"]["train"], seed=seed,
                            updates=updates, batch_size=batch_size, learning_rate=learning_rate, radius_ticks=radius_ticks)
        final = metrics()
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    if training._file_sha256(manifest_path) != manifest_sha or training._file_sha256(checkpoint_path) != checkpoint_sha:
        raise ValueError("manifest or base checkpoint changed during training")
    plan_path = Path(manifest["source_plan_path"])
    if not plan_path.is_absolute():
        plan_path = manifest_path.parent / plan_path
    if (training._file_sha256(plan_path) != manifest["source_plan_sha256"]
            or any(training._file_sha256(source["journal_path"]) != source["journal_sha256"]
                   for source in prepared["source_metadata"]["sources"])):
        raise ValueError("a direct journal or collection plan changed during training")
    if any(path.read_bytes() != code_snapshots[path.name] for path in code_paths):
        raise ValueError("training code changed during training")
    config = {"seed": seed, "updates": updates, "batch_size": batch_size, "learning_rate": float(learning_rate),
              "optimizer": "adam", "weight_decay": 0.0, "gradient_clip": None,
              "radius_ticks": radius_ticks, "radius_seconds": radius_ticks * model.step_seconds,
              "sampling": "4 close, 4 open, 32 nearby hold, 88 ordinary hold; seeded with replacement",
              "batch_allocation": fit["batch_allocation"],
              "terminal_sampling": "no reserved terminal rows; sampled occurrences reported",
              "loss": "joint MSE + velocity MSE + 0.05 gripper CE + 0.05 terminal BCE",
              "labels_modified": False, "trainable": "encoder and joint, velocity, gripper, done heads",
              "new_input_features": False, "runtime_or_command_changes": False}
    provenance = dict(checkpoint["provenance"])
    provenance["all_head_finetuning"] = {"base_checkpoint_path": str(checkpoint_path),
                                         "base_checkpoint_sha256": checkpoint_sha,
                                         "manifest_sha256": manifest_sha, "configuration": config}
    metric_keys = ("joint_mae_deg_per_joint", "velocity_mae_deg_s_per_joint", "gripper_accuracy",
                   "terminal_true_positive", "terminal_false_positive", "terminal_false_negative",
                   "early_terminal_episode_count")
    deltas = {name: {key: final[name]["all_outputs"][key] - initial[name]["all_outputs"][key] for key in metric_keys}
              for name in ("train", "validation")}
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for name, content in code_snapshots.items():
        (source_dir / name).write_bytes(content)
    candidate_path = output_dir / "trajectory_student.pt"
    with candidate_path.open("xb") as stream:
        torch.save(policy.checkpoint_payload(model, provenance=provenance), stream)
    report = {"schema": "raccoon-direct-trajectory-all-heads-training-report.v1",
              "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
              "base_checkpoint_path": str(checkpoint_path), "base_checkpoint_sha256": checkpoint_sha,
              "checkpoint_path": str(candidate_path), "checkpoint_sha256": training._file_sha256(candidate_path),
              "configuration": config, "groups": prepared["groups"],
              "episode_counts": {name: len(prepared["episode_spans"][name]) for name in ("train", "validation")},
              "fit": fit, "initial_recorded_metrics": initial, "final_recorded_metrics": final,
              "recorded_metric_deltas": deltas, "source_integrity_rechecked": True,
              "code_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in code_snapshots.items()},
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "limitations": ["actor identity is unverified", "validation is internal to the original direct collection",
                              "row-level metrics use teacher-recorded observations, not closed-loop success",
                              "oversampled action labels and terminal sampling affect calibration",
                              "training samples may use proximity to future transitions only for selection, never as model input",
                              "no physical device or microphone was used"]}
    with (output_dir / "training_report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
