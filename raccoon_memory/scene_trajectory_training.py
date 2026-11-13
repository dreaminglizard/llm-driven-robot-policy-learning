"""Reproducible, offline behavior cloning for the direct trajectory student.

Replay-derived positions and velocities are labels, never student inputs.
Validation is grouped by original source and is not an independent holdout.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from . import motion_policy, scene_features, scene_trajectory_policy as policy, task_catalog

REPLAY_SCHEMA = "raccoon-direct-simulation-replay-trajectories.v1"
REPLAY_SOURCE_KIND = "simulator_replay_of_direct_position_and_gripper_decisions"
REPLAY_STEP_SECONDS = 0.02


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return [_number(x, name) for x in value]


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _direct_alignment(path: str | Path, source_hashes: set[str]) -> dict:
    """Bind a stopped, scored simulation alignment to journals in this dataset."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    summary = json.loads(raw)
    if (summary.get("schema") != "raccoon-integrated-direct-velocity-replay.v1"
            or summary.get("phase") != "posthoc_scored" or summary.get("simulated") is not True
            or summary.get("actual_hardware") is not False or summary.get("student_performance_assessment") is not False
            or not isinstance(summary.get("rows"), list) or not summary["rows"]):
        raise ValueError("direct control alignment is not a completed simulated reference")
    plan_path = path.parent / "plan.json"
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    if hashlib.sha256(plan_raw).hexdigest() != summary.get("plan_sha256"):
        raise ValueError("direct control alignment plan fingerprint differs")
    sources = {row["name"]: row for row in plan.get("sources", [])}
    if len(sources) != len(plan.get("sources", [])) or set(sources) != {row.get("name") for row in summary["rows"]}:
        raise ValueError("direct control alignment cases differ from its plan")
    for row in summary["rows"]:
        details = row.get("trajectory_difference", {})
        if (row.get("status") != "success" or row.get("collector_status") != "success"
                or row.get("same_checks_as_collector") is not True
                or details.get("same_sample_clock") is not True or details.get("same_gripper_targets") is not True
                or sources[row["name"]].get("source_sha256") not in source_hashes):
            raise ValueError("direct control alignment failed or is not bound to these source journals")
        score_path = path.parent / row["score_file"]
        if _file_sha256(score_path) != row.get("score_file_sha256"):
            raise ValueError("direct control alignment score fingerprint differs")
        score = json.loads(score_path.read_bytes())
        assessment = score.get("integrated", {}).get("assessment", {})
        if assessment.get("sample_count") != 501 or assessment.get("success") is not True:
            raise ValueError("direct control alignment does not contain a successful 501-sample assessment")
    return {"summary_path": str(path), "summary_sha256": hashlib.sha256(raw).hexdigest(),
            "plan_path": str(plan_path), "plan_sha256": summary["plan_sha256"],
            "cases": len(summary["rows"]), "scope": "simulation reference replay, not a student performance score"}


def prepare_dataset(payload: dict, *, history_length: int = scene_features.HISTORY_LENGTH,
                    history_offsets: list[int] | tuple[int, ...] | None = None) -> dict:
    """Validate grouped replay episodes and construct strictly causal windows."""
    if not isinstance(payload, dict):
        raise ValueError("replay dataset must be an object")
    metadata = payload.get("metadata")
    if (payload.get("schema") != REPLAY_SCHEMA or not isinstance(metadata, dict)
            or metadata.get("source_kind") != REPLAY_SOURCE_KIND
            or metadata.get("sample_dt_s") != REPLAY_STEP_SECONDS
            or metadata.get("student_object_positions_contacts_or_simulator_qvel_as_input") is not False
            or metadata.get("actual_hardware") is not False):
        raise ValueError("unsupported replay dataset schema or source contract")
    if (payload.get("feature_names") != list(scene_features.FEATURE_NAMES)
            or not isinstance(payload.get("episodes"), list) or not payload["episodes"]):
        raise ValueError("dataset must declare the exact feature names and contain episodes")
    if type(history_length) is not int or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    offsets = policy._history_offsets(history_offsets, history_length)
    pieces = {"train": [], "validation": []}
    groups = {"train": set(), "validation": set()}
    episode_spans = {"train": [], "validation": []}
    next_offset = {"train": 0, "validation": 0}
    seen_ids, dt, summaries = set(), None, []
    for episode in payload["episodes"]:
        if not isinstance(episode, dict):
            raise ValueError("each episode must be an object")
        episode_id = _text(episode.get("episode_id"), "episode_id")
        group = _text(episode.get("source_group_id"), "source_group_id")
        if episode_id in seen_ids:
            raise ValueError("episode IDs must be unique")
        seen_ids.add(episode_id)
        split = episode.get("split")
        if split not in pieces:
            raise ValueError("only train and internal validation episodes are permitted")
        groups[split].add(group)
        samples = episode.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("each episode must contain samples")
        features, positions, velocities, grippers, terminals = [], [], [], [], []
        last_time = None
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError("each sample must be an object")
            time = _number(sample.get("t_s"), "t_s")
            tick = _number(sample.get("dt_s"), "dt_s")
            if time < 0 or tick <= 0:
                raise ValueError("sample time must be nonnegative and dt must be positive")
            if dt is None:
                dt = tick
            if (not math.isclose(tick, dt, rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(tick, REPLAY_STEP_SECONDS, rel_tol=0, abs_tol=1e-9)):
                raise ValueError("all samples must use the same dt")
            if last_time is not None and not math.isclose(time-last_time, dt, rel_tol=0, abs_tol=1e-7):
                raise ValueError("episode samples must be contiguous and chronological")
            last_time = time
            features.append(_vector(sample.get("features"), len(scene_features.FEATURE_NAMES), "features"))
            positions.append(_vector(sample.get("target_joint_deg"), 4, "target_joint_deg"))
            velocities.append(_vector(sample.get("target_velocity_deg_s"), 4, "target_velocity_deg_s"))
            gripper = sample.get("target_gripper")
            if gripper not in policy.GRIPPER_STATES:
                raise ValueError("target_gripper must be open or closed")
            grippers.append(policy.GRIPPER_STATES.index(gripper))
            terminal = sample.get("terminal")
            if type(terminal) is not bool or terminal != (index == len(samples)-1):
                raise ValueError("only the final sample of each episode must be terminal")
            terminals.append(terminal)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        count = len(features)
        windows = torch.zeros((count, history_length, len(scene_features.FEATURE_NAMES)), dtype=torch.float32)
        masks = torch.zeros((count, history_length), dtype=torch.bool)
        for index, offset in enumerate(offsets):
            if offset < count:
                windows[offset:, index] = feature_tensor[:count-offset]
                masks[offset:, index] = True
        pieces[split].append({"windows": windows, "masks": masks,
                              "positions": torch.tensor(positions, dtype=torch.float32) / policy.JOINT_SCALE_DEG,
                              "velocities": torch.tensor(velocities, dtype=torch.float32) / policy.VELOCITY_SCALE_DEG_S,
                              "grippers": torch.tensor(grippers, dtype=torch.long),
                              "terminals": torch.tensor(terminals, dtype=torch.float32)})
        episode_spans[split].append({"episode_id": episode_id, "offset": next_offset[split], "count": count})
        next_offset[split] += count
        summaries.append({"episode_id": episode_id, "source_group_id": group, "split": split,
                          "sample_count": count})
    if any(not pieces[split] for split in pieces):
        raise ValueError("both training and internal validation episodes are required")
    if groups["train"] & groups["validation"]:
        raise ValueError("source groups overlap between training and validation")
    result = {}
    for split in pieces:
        result[split] = {key: torch.cat([piece[key] for piece in pieces[split]]) for key in pieces[split][0]}
    result.update(step_seconds=dt, episode_summaries=summaries,
                  groups={name: sorted(values) for name, values in groups.items()}, episode_spans=episode_spans)
    return result


def _loss(model, data, indices):
    windows, masks = _batch_windows(model, data, indices)
    positions, velocities, grippers, terminals = model(windows, masks)
    position_loss = F.mse_loss(positions, data["positions"][indices])
    velocity_loss = F.mse_loss(velocities, data["velocities"][indices])
    gripper_loss = F.cross_entropy(grippers, data["grippers"][indices])
    terminal_loss = F.binary_cross_entropy_with_logits(terminals, data["terminals"][indices])
    return position_loss + velocity_loss + 0.05 * gripper_loss + 0.05 * terminal_loss


def _batch_windows(model: policy.TrajectoryPolicy, data: dict, indices):
    """Gather past-only windows, optionally lazily from compact episodes."""
    if "windows" in data:
        return data["windows"][indices], data["masks"][indices]
    count = len(data["features"])
    if isinstance(indices, slice):
        start, stop, step = indices.indices(count)
        indices = torch.arange(start, stop, step, dtype=torch.long)
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError("lazy batch indices must be a one-dimensional integer tensor")
    source_indices = indices[:, None] - torch.tensor(model.history_offsets, dtype=torch.long)[None, :]
    masks = source_indices >= data["episode_starts"][indices, None]
    windows = data["features"][source_indices.clamp(min=0)]
    return windows.masked_fill(~masks.unsqueeze(-1), 0.0), masks


def replay_label_bounds(data: dict) -> dict:
    """Quantify recorded replay labels outside the candidate's research bounds."""
    joints = data["positions"] * policy.JOINT_SCALE_DEG
    speeds = data["velocities"] * policy.VELOCITY_SCALE_DEG_S
    limits = torch.tensor(policy.JOINT_LIMITS, dtype=joints.dtype)
    position_excess = torch.maximum(limits[:, 0] - joints, joints - limits[:, 1]).clamp_min(0)
    speed_excess = (speeds.abs() - policy.SIMULATED_VELOCITY_LIMIT_DEG_S).clamp_min(0)
    return {"position_samples_outside_joint_limits": int((position_excess > 0).any(1).sum()),
            "position_joint_values_outside_joint_limits": int((position_excess > 0).sum()),
            "position_maximum_excess_deg": float(position_excess.max()),
            "velocity_samples_outside_simulation_limit": int((speed_excess > 0).any(1).sum()),
            "velocity_joint_values_outside_simulation_limit": int((speed_excess > 0).sum()),
            "velocity_maximum_excess_deg_s": float(speed_excess.max())}


def recorded_metrics(model: policy.TrajectoryPolicy, data: dict, *, batch_size: int = 512,
                     episode_spans: list[dict] | None = None) -> dict:
    """Teacher-observation next-tick metrics, not closed-loop or device success."""
    model.eval()
    count = len(data["positions"])
    joint_error = velocity_error = 0.0
    gripper_correct = tp = fp = fn = 0
    outside_position_samples = outside_position_values = outside_velocity_samples = outside_velocity_values = 0
    max_position_excess = max_velocity_excess = 0.0
    terminal_predictions = []
    gripper_counts = {state: {"actual": 0, "predicted": 0, "correct": 0}
                      for state in policy.GRIPPER_STATES}
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            rows = slice(start, min(start+batch_size, count))
            windows, masks = _batch_windows(model, data, rows)
            positions, velocities, grippers, terminals = model(windows, masks)
            limits = positions.new_tensor(policy.JOINT_LIMITS)
            joint_deg = positions * policy.JOINT_SCALE_DEG
            speed_deg_s = velocities * policy.VELOCITY_SCALE_DEG_S
            position_excess = torch.maximum(limits[:, 0] - joint_deg, joint_deg - limits[:, 1]).clamp_min(0)
            velocity_excess = (speed_deg_s.abs() - policy.SIMULATED_VELOCITY_LIMIT_DEG_S).clamp_min(0)
            outside_position_samples += int((position_excess > 0).any(1).sum())
            outside_position_values += int((position_excess > 0).sum())
            outside_velocity_samples += int((velocity_excess > 0).any(1).sum())
            outside_velocity_values += int((velocity_excess > 0).sum())
            max_position_excess = max(max_position_excess, float(position_excess.max()))
            max_velocity_excess = max(max_velocity_excess, float(velocity_excess.max()))
            joint_error += float((positions-data["positions"][rows]).abs().sum()) * policy.JOINT_SCALE_DEG
            velocity_error += float((velocities-data["velocities"][rows]).abs().sum()) * policy.VELOCITY_SCALE_DEG_S
            predicted_gripper, true_gripper = grippers.argmax(-1), data["grippers"][rows]
            gripper_correct += int((predicted_gripper == true_gripper).sum())
            for index, state in enumerate(policy.GRIPPER_STATES):
                gripper_counts[state]["actual"] += int((true_gripper == index).sum())
                gripper_counts[state]["predicted"] += int((predicted_gripper == index).sum())
                gripper_counts[state]["correct"] += int(((predicted_gripper == index) & (true_gripper == index)).sum())
            predicted, truth = terminals >= 0, data["terminals"][rows].bool()
            terminal_predictions.extend(predicted.tolist())
            tp += int((predicted & truth).sum())
            fp += int((predicted & ~truth).sum())
            fn += int((~predicted & truth).sum())
    result = {"samples": count, "joint_mae_deg_per_joint": joint_error / (count*4),
            "velocity_mae_deg_s_per_joint": velocity_error / (count*4),
            "joint_range_outside_samples": outside_position_samples,
            "joint_range_outside_values": outside_position_values,
            "joint_range_maximum_excess_deg": max_position_excess,
            "velocity_range_outside_samples": outside_velocity_samples,
            "velocity_range_outside_values": outside_velocity_values,
            "velocity_range_maximum_excess_deg_s": max_velocity_excess,
            "gripper_correct": gripper_correct, "gripper_accuracy": gripper_correct / count,
            "gripper_classes": gripper_counts,
            "terminal_true_positive": tp, "terminal_false_positive": fp, "terminal_false_negative": fn,
            "terminal_true_negative": count-tp-fn-fp,
            "terminal_positive_samples": tp+fn, "terminal_negative_samples": count-tp-fn,
            "terminal_predicted_positive": tp+fp}
    if episode_spans is not None:
        episodes = []
        for span in episode_spans:
            offset, length = span["offset"], span["count"]
            predictions = terminal_predictions[offset:offset+length]
            if len(predictions) != length:
                raise ValueError("episode spans do not match metrics data")
            first = next((i for i, value in enumerate(predictions) if value), None)
            episodes.append({"episode_id": span["episode_id"], "sample_count": length,
                             "early_terminal_on_recorded_observation": first is not None and first < length-1,
                             "first_predicted_terminal_index": first,
                             "recorded_terminal_index": length-1})
        result["episode_count"] = len(episodes)
        result["early_terminal_episode_count"] = sum(e["early_terminal_on_recorded_observation"] for e in episodes)
        result["episode_terminal_diagnostics"] = episodes
    return result


def train(dataset_path: str | Path, output_dir: str | Path, *, seed: int = 20261107,
          updates: int = 1000, batch_size: int = 256, learning_rate: float = 0.001,
          hidden_size: int = 64, history_length: int = scene_features.HISTORY_LENGTH,
          position_residual: bool = False, bounded_outputs: bool = False,
          optimizer_name: str = "adam", weight_decay: float = 0.0,
          gradient_clip: float | None = None,
          history_offsets: list[int] | tuple[int, ...] | None = None,
          direct_position_label: str | None = None, actuation_semantics: str | None = None,
          control_alignment: str | Path | None = None) -> dict:
    """Train a new candidate and persist its checkpoint and recorded metrics."""
    if type(seed) is not int or seed < 0 or type(updates) is not int or updates < 1:
        raise ValueError("seed must be nonnegative and updates must be positive integers")
    if type(batch_size) is not int or batch_size < 16:
        raise ValueError("batch_size must be at least 16")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("optimizer_name must be adam or adamw")
    if type(weight_decay) not in (int, float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if gradient_clip is not None and (type(gradient_clip) not in (int, float)
                                      or not math.isfinite(gradient_clip) or gradient_clip <= 0):
        raise ValueError("gradient_clip must be finite and positive when set")
    dataset_path, output_dir = Path(dataset_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    raw = dataset_path.read_bytes()
    payload = json.loads(raw)
    direct_mode = payload.get("schema") == "raccoon-direct-velocity-source-manifest.v1"
    if direct_mode:
        if actuation_semantics not in policy.ACTUATION_SEMANTICS:
            raise ValueError("direct training requires explicit valid actuation semantics")
        if (actuation_semantics == "integrated_velocity_with_position_target"
                and direct_position_label != "current_applied_position_servo_target"):
            raise ValueError("integrated actuation requires the start-of-interval position label")
        from . import direct_velocity_training_data, scene_trajectory_direct_training
        policy._history_offsets(history_offsets, history_length)
        prepared = scene_trajectory_direct_training.prepare_direct_manifest(payload, dataset_path.parent,
                                                                             position_label=direct_position_label)
    else:
        if direct_position_label is not None:
            raise ValueError("direct position label applies only to direct-command manifests")
        if control_alignment is not None:
            raise ValueError("direct control alignment applies only to direct-command manifests")
        if actuation_semantics not in (None, "position_target_with_velocity_force"):
            raise ValueError("replay training uses the position-and-velocity-force research contract")
        prepared = prepare_dataset(payload, history_length=history_length, history_offsets=history_offsets)
        actuation_semantics = "position_target_with_velocity_force"
    alignment = (_direct_alignment(control_alignment,
                 {source["journal_sha256"] for source in prepared["source_metadata"]["sources"]})
                 if control_alignment is not None else None)
    code_paths = [Path(path).resolve() for path in (__file__, policy.__file__, scene_features.__file__,
                  motion_policy.__file__, task_catalog.__file__, Path(__file__).resolve().parents[1] /
                  "scripts" / "train_scene_trajectory_policy.py", Path(__file__).resolve().parents[1] /
                  "raccoon_control" / "kinematics.py")]
    if direct_mode:
        code_paths.extend([Path(direct_velocity_training_data.__file__).resolve(),
                           Path(scene_trajectory_direct_training.__file__).resolve(),
                           Path(__file__).resolve().parents[1] / "scripts" / "create_direct_velocity_manifest.py"])
    code_snapshots = {path.name: path.read_bytes() for path in code_paths}
    train_data = prepared["train"]
    terminal_indices = torch.nonzero(train_data["terminals"].bool(), as_tuple=False).flatten()
    nonterminal_indices = torch.nonzero(~train_data["terminals"].bool(), as_tuple=False).flatten()
    if not len(terminal_indices) or not len(nonterminal_indices):
        raise ValueError("training data must contain both terminal and nonterminal samples")
    terminal_per_batch = max(1, batch_size // 16)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = policy.TrajectoryPolicy(step_seconds=prepared["step_seconds"], hidden_size=hidden_size,
                                            history_length=history_length, position_residual=position_residual,
                                            bounded_outputs=bounded_outputs, history_offsets=history_offsets)
            initial = {name: recorded_metrics(model, prepared[name], episode_spans=prepared["episode_spans"][name])
                       for name in ("train", "validation")}
            model.train()
            optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
            optimizer = optimizer_cls(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
            generator = torch.Generator().manual_seed(seed)
            last_loss = None
            for _ in range(updates):
                ordinary = nonterminal_indices[torch.randint(len(nonterminal_indices),
                    (batch_size-terminal_per_batch,), generator=generator)]
                terminal = terminal_indices[torch.randint(len(terminal_indices),
                    (terminal_per_batch,), generator=generator)]
                indices = torch.cat((ordinary, terminal))
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, train_data, indices)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip), error_if_nonfinite=True)
                optimizer.step()
                last_loss = float(loss.detach())
            final = {name: recorded_metrics(model, prepared[name], episode_spans=prepared["episode_spans"][name])
                     for name in ("train", "validation")}
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    dataset_sha256 = hashlib.sha256(raw).hexdigest()
    if _file_sha256(dataset_path) != dataset_sha256:
        raise ValueError("dataset or manifest changed during training")
    journals_unchanged, plan_unchanged = None, None
    if direct_mode:
        journals_unchanged = all(_file_sha256(source["journal_path"]) == source["journal_sha256"]
                                 for source in prepared["source_metadata"]["sources"])
        plan_path = Path(payload["source_plan_path"])
        if not plan_path.is_absolute():
            plan_path = dataset_path.parent / plan_path
        plan_unchanged = _file_sha256(plan_path) == payload["source_plan_sha256"]
        if not journals_unchanged or not plan_unchanged:
            raise ValueError("a direct journal or collection plan changed during training")
    config = {"seed": seed, "updates": updates, "batch_size": batch_size,
              "learning_rate": float(learning_rate), "hidden_size": hidden_size,
              "history_length": history_length, "step_seconds": prepared["step_seconds"],
              "direct_position_label": direct_position_label,
              "actuation_semantics": actuation_semantics,
              "position_residual": position_residual, "bounded_outputs": bounded_outputs,
              "output_bounds": ({"joint_limits_deg": [list(pair) for pair in policy.JOINT_LIMITS],
                                 "numerical_position_margin_deg": policy.NUMERICAL_POSITION_MARGIN_DEG,
                                 "simulation_velocity_abs_deg_s": policy.SIMULATED_VELOCITY_LIMIT_DEG_S,
                                 "scope": "simulation research limits, not verified physical limits"}
                                if bounded_outputs else None),
              "history_offsets": list(model.history_offsets),
              "maximum_history_span_seconds": model.history_offsets[0]*prepared["step_seconds"],
              "input_contract": "current and past public features, including the observed gripper command; "
                                "no previous applied joint target or discrete robot action history",
              "terminal_samples_per_batch": terminal_per_batch,
              "sampling": "seeded with replacement; terminal rows reserved as reported",
              "optimizer": optimizer_name, "weight_decay": float(weight_decay), "gradient_clip": gradient_clip,
              "loss": "joint MSE + velocity MSE + 0.05 gripper CE + 0.05 terminal BCE",
              "terminal_semantics": "apply_then_finish",
              "terminal_semantics_description": "finish after applying the current model output for step_seconds"}
    dataset_metadata = ({"manifest_metadata": {key: value for key, value in payload.items() if key != "entries"},
                         **prepared["source_metadata"]} if direct_mode else
                        {key: value for key, value in payload.items() if key != "episodes"})
    source_scope = ("explicitly submitted velocities and explicitly selected simulated position labels; internal source-group validation; "
                    "not physical success" if direct_mode else
                    "offline replay-derived targets; internal source-group validation; not physical success")
    provenance = {"dataset_path": str(dataset_path), "dataset_sha256": dataset_sha256,
                  "dataset_metadata": dataset_metadata,
                  "control_alignment": alignment,
                  "scope": source_scope,
                  "configuration": config}
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    source_dir.mkdir()
    for name, content in code_snapshots.items():
        (source_dir / name).write_bytes(content)
    checkpoint_path = output_dir / "trajectory_student.pt"
    with checkpoint_path.open("xb") as stream:
        checkpoint = policy.checkpoint_payload(model, provenance=provenance)
        checkpoint["terminal_semantics"] = config["terminal_semantics"]
        checkpoint["actuation_semantics"] = config["actuation_semantics"]
        checkpoint["direct_position_label"] = config["direct_position_label"]
        torch.save(checkpoint, stream)
    report = {"schema": "raccoon-direct-trajectory-training-report.v1", "configuration": config,
              "dataset_path": str(dataset_path), "dataset_sha256": dataset_sha256,
              "control_alignment": alignment,
              "dataset_metadata": provenance["dataset_metadata"], "groups": prepared["groups"],
              "episodes": prepared["episode_summaries"],
              "sample_counts": {name: len(prepared[name]["positions"]) for name in ("train", "validation")},
              "label_bounds": {name: replay_label_bounds(prepared[name]) for name in ("train", "validation")},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "code_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in code_snapshots.items()},
              "source_unchanged_during_training": all(path.read_bytes() == code_snapshots[path.name] for path in code_paths),
              "direct_journals_unchanged_during_training": journals_unchanged,
              "direct_plan_unchanged_during_training": plan_unchanged,
              "python_version": platform.python_version(), "torch_version": str(torch.__version__),
              "initial_recorded_metrics": initial, "final_recorded_metrics": final,
              "final_sampled_batch_loss": last_loss,
              "limitations": (["velocity targets are explicitly submitted commands; actor identity is unverified",
                               "position labels are observed angles or integrated servo targets, not independently submitted angle commands",
                               "initial moving status requires the matching verified-rest runtime contract"] if direct_mode else
                              ["targets are replay-derived, not direct teacher joint-velocity demonstrations"]) + [
                              "validation shares the original training source collection and is not a final holdout",
                              "recorded-observation errors do not establish closed-loop or physical success",
                              "terminal label occurs after this interval; an executor that stops before applying it has a one-tick mismatch",
                              "terminal rows are oversampled in training; raw logits are not calibrated to their recorded frequency",
                              "short observation window may not distinguish distant task phases or repeated poses",
                              "raw outputs require independent limits and stopping guards"]}
    with (output_dir / "training_report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
