"""Train the small full-teacher policies from an already completed export.

The same adapter is used by training, recorded evaluation and loaded-policy
inference.  This module does not select demonstrations or run a simulator.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control import kinematics
from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import full_teacher_dataset as exported
from raccoon_memory import motion_policy, scene_features, scene_policy

CHECKPOINT_SCHEMA = "raccoon-full-teacher-policy.v1"
RUN_SCHEMA = "raccoon-full-teacher-training.v1"
RECORD_SCHEMA = "raccoon-full-teacher-recorded-evaluation.v1"
ARCHITECTURES = ("mlp", "gru")
FEATURE_COUNT, HISTORY_LENGTH, MOVING_INDEX = 82, 8, 5
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    return _sha(Path(path).read_bytes())


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _json(path):
    value, raw = exported._read_json(path)
    return value, raw


def _jsonl(path):
    path = Path(path)
    raw = path.read_bytes()
    need(bool(raw) and raw.endswith(b"\n"), f"{path.name}: missing or truncated JSONL")
    lines = raw.splitlines()
    need(all(line.strip() for line in lines), f"{path.name}: empty JSONL row")
    rows = [exported.collection.operator._strict_json(line, f"{path.name}:{i + 1}")
            for i, line in enumerate(lines)]
    need(all(isinstance(row, dict) for row in rows), f"{path.name}: row must be an object")
    return rows, raw


def _source(reference, name):
    need(isinstance(reference, dict) and isinstance(reference.get("path"), str) and reference["path"] and
         isinstance(reference.get("sha256"), str) and _SHA.fullmatch(reference["sha256"]),
         f"{name}: source reference missing")
    path = Path(reference["path"])
    value, raw = _json(path)
    need(_sha(raw) == reference["sha256"], f"{name}: source fingerprint differs")
    return value, path.resolve()


def _validate_rows(rows100, rows10, originals, cases, selected):
    """Check actual decision order, explicit DONE, MOVE masks and both inputs."""
    by_case = {c["trial_id"]: c for c in cases}
    ids100, ids10 = selected["selected_100"], selected["selected_10"]
    need(len(originals) == len(rows100) and bool(rows100) and bool(rows10), "missing original or training rows")
    raw_by_key, rows_by_trial = {}, {tid: [] for tid in ids100}
    for raw in originals:
        need(raw.get("schema") == exported.RAW_SCHEMA, "original input schema differs")
        key = (raw.get("trial_id"), raw.get("teacher_decision"))
        need(key not in raw_by_key, "duplicate original decision")
        raw_by_key[key] = raw
    row_keys = {"schema", "trial_id", "case_id", "attempt_number", "split_group", "registered_scene", "task",
                "physics_seed", "teacher_agent", "teacher_decision", "history_length", "window", "mask",
                "action", "action_index", "move_target_valid", "move_target_normalized", "provenance"}
    raw_keys = {"schema", "trial_id", "teacher_decision", "window", "mask", "input_path", "input_sha256",
                "original_input_sha256", "training_input_sha256"}
    for row in rows100:
        need(set(row) == row_keys and row.get("schema") == exported.ROW_SCHEMA, "training row schema differs")
        tid, decision = row.get("trial_id"), row.get("teacher_decision")
        need(tid in rows_by_trial and type(decision) is int and decision == len(rows_by_trial[tid]) and decision < 24,
             "episode decision order differs")
        case = by_case[tid]
        for key in ("case_id", "attempt_number", "split_group", "registered_scene", "task", "physics_seed"):
            need(row.get(key) == case[key], f"{tid}: case metadata differs")
        need(row.get("teacher_agent") == case["assigned_selector"], "teacher origin differs")
        raw = raw_by_key.get((tid, decision))
        need(isinstance(raw, dict) and set(raw) == raw_keys and raw.get("mask") == row.get("mask"),
             "original input or mask differs")
        converted = exported.zero_moving(raw["window"], raw["mask"])
        need(converted == row.get("window") and row.get("history_length") == min(decision + 1, HISTORY_LENGTH)
             and sum(row["mask"]) == row["history_length"], "input transformation or history differs")
        original_digest = exported._digest_value({"window": raw["window"], "mask": raw["mask"]})
        training_digest = exported._digest_value({"window": converted, "mask": row["mask"]})
        provenance = row.get("provenance")
        need(isinstance(provenance, dict) and raw.get("original_input_sha256") == original_digest and
             raw.get("training_input_sha256") == training_digest and
             provenance.get("original_input_sha256") == original_digest and
             provenance.get("training_input_sha256") == training_digest and
             provenance.get("input_path") == raw.get("input_path") and
             provenance.get("input_sha256") == raw.get("input_sha256"), "original and transformed provenance differs")
        for key in ("input_sha256", "proposal_event_sha256", "call_event_sha256", "return_event_sha256",
                    "return_sha256", "state_before_sha256", "state_after_sha256"):
            need(isinstance(provenance.get(key), str) and _SHA.fullmatch(provenance[key]),
                 f"returned decision provenance missing: {key}")
        need(all(isinstance(provenance.get(key), str) and provenance[key] for key in ("input_path", "return_path")),
             "returned input or output path missing")
        action = row.get("action")
        need(action in motion_policy.ACTIONS and type(row.get("action_index")) is int and
             row["action_index"] == motion_policy.ACTIONS.index(action), "action index differs")
        if action == "move":
            target = row.get("move_target_normalized")
            need(row.get("move_target_valid") is True and isinstance(target, list) and len(target) == 4 and
                 all(type(x) in (int, float) and math.isfinite(x) and lo - 1e-7 <= x * 180.0 <= hi + 1e-7
                     for x, (lo, hi) in zip(target, JOINT_LIMITS)), "MOVE target is missing or invalid")
        else:
            need(row.get("move_target_valid") is False and row.get("move_target_normalized") is None,
                 "non-MOVE must not have a joint target")
        rows_by_trial[tid].append(row)
    need(len(raw_by_key) == len(rows100), "orphan original inputs")
    for tid, episode in rows_by_trial.items():
        need(bool(episode) and episode[-1]["action"] == "done" and
             sum(x["action"] == "done" for x in episode) == 1, f"{tid}: one explicit final DONE is required")
    need(rows100 == [r for tid in ids100 for r in rows_by_trial[tid]], "100-set episode order differs")
    need(rows10 == [r for tid in ids10 for r in rows_by_trial[tid]], "10-set is not the same selected rows")
    return rows_by_trial


def load_converted_dataset(directory, expected_manifest_sha256):
    """Read a complete 100/10 export; never generate or change a selection."""
    directory = Path(directory).resolve()
    need(isinstance(expected_manifest_sha256, str) and _SHA.fullmatch(expected_manifest_sha256),
         "reviewed dataset manifest SHA-256 is required")
    manifest, raw_manifest = _json(directory / "manifest.json")
    need(_sha(raw_manifest) == expected_manifest_sha256 and isinstance(manifest, dict) and
         manifest.get("schema") == exported.DATASET_SCHEMA and manifest.get("status") == "complete" and
         manifest.get("feature_schema") == scene_features.SCHEMA and manifest.get("history_length") == HISTORY_LENGTH and
         manifest.get("feature_count") == FEATURE_COUNT and manifest.get("moving_index_zero_based") == MOVING_INDEX and
         manifest.get("non_move_joint_target") is None and manifest.get("automatic_selection") is False and
         manifest.get("selection_rule_recomputed_for_verification") is True,
         "completed converted dataset manifest differs")
    report, report_path = _source(manifest.get("review_report"), "review report")
    selection_file, selection_path = _source(manifest.get("selection"), "selection")
    plan, plan_path = _source(manifest.get("source_plan"), "source plan")
    need(isinstance(plan, dict) and isinstance(plan.get("cases"), list), "source plan cases missing")
    config_sha = manifest.get("collection_config_sha256")
    selection = exported.validate_review_selection(plan["cases"], report, selection_file, config_sha,
                                                    _file_sha(report_path))
    exported._check_sources(selection["source_hashes"])
    need(selection["source_hashes"].get(str(plan_path)) == _file_sha(plan_path), "source plan was not review-bound")
    need(manifest["selection"].get("selected_100") == selection["selected_100"] and
         manifest["selection"].get("selected_10") == selection["selected_10"], "exported selection differs")
    exporter_ref = manifest.get("exporter")
    need(isinstance(exporter_ref, dict) and isinstance(exporter_ref.get("path"), str) and
         isinstance(exporter_ref.get("sha256"), str) and _SHA.fullmatch(exporter_ref["sha256"]) and
         _file_sha(exporter_ref["path"]) == exporter_ref["sha256"], "exporter source differs")
    files = manifest.get("files")
    expected_names = {"original_inputs.jsonl", "train_100.jsonl", "train_10.jsonl"}
    need(isinstance(files, dict) and set(files) == expected_names, "converted file manifest differs")
    loaded = {}
    for name in sorted(expected_names):
        rows, raw = _jsonl(directory / name)
        info = files[name]
        need(isinstance(info, dict) and info.get("sha256") == _sha(raw) and info.get("bytes") == len(raw),
             f"{name}: exported file fingerprint differs")
        loaded[name] = rows
    rows100, rows10, originals = loaded["train_100.jsonl"], loaded["train_10.jsonl"], loaded["original_inputs.jsonl"]
    _validate_rows(rows100, rows10, originals, plan["cases"], selection)
    for size, rows in ((100, rows100), (10, rows10)):
        counts = {"episodes": size, "decisions": len(rows),
                  "actions": dict(sorted(Counter(r["action"] for r in rows).items())),
                  "move_targets": sum(r["move_target_valid"] for r in rows)}
        need(manifest.get("counts", {}).get(str(size)) == counts, f"{size}-episode counts differ")
    exported._check_sources(selection["source_hashes"])
    return {"directory": directory, "manifest": manifest, "manifest_sha256": expected_manifest_sha256,
            "rows": {10: rows10, 100: rows100}, "source_paths": (report_path, selection_path, plan_path),
            "review_source_hashes": selection["source_hashes"]}


def adapt_inputs(windows, masks):
    """Validate then copy and zero moving for training, recording and inference."""
    need(isinstance(windows, torch.Tensor) and windows.is_floating_point() and windows.ndim == 3 and
         tuple(windows.shape[1:]) == (HISTORY_LENGTH, FEATURE_COUNT) and windows.shape[0] > 0,
         "windows must be a nonempty batch of 8x82 floating inputs")
    need(isinstance(masks, torch.Tensor) and masks.dtype == torch.bool and masks.shape == windows.shape[:2] and
         masks.device == windows.device, "mask shape, type or device differs")
    need(bool(torch.isfinite(windows).all()) and bool(masks[:, -1].all()) and
         not bool((masks[:, :-1] & ~masks[:, 1:]).any()), "nonfinite or non-left-padded input")
    need(not bool((windows.masked_select(~masks.unsqueeze(-1)) != 0).any()), "padded feature rows must be zero")
    transformed = windows.clone()
    transformed[:, :, MOVING_INDEX] = 0.0
    return transformed, masks


class FullTeacherPolicy(nn.Module):
    """Existing end-to-end policy with the common input adapter at every call."""

    def __init__(self, architecture):
        super().__init__()
        need(architecture in ARCHITECTURES, "architecture must be mlp or gru")
        self.architecture = architecture
        self.history_length, self.feature_count = HISTORY_LENGTH, FEATURE_COUNT
        self.policy = scene_policy.ScenePolicy(architecture=architecture, hidden_size=64,
                                               history_length=HISTORY_LENGTH)

    def forward(self, windows, masks):
        windows, masks = adapt_inputs(windows, masks)
        return self.policy(windows, masks)


def initialize_policy(architecture, seed, learning_rate=0.002, device="cpu"):
    """Create a fresh policy and empty Adam without consuming the caller RNG."""
    need(architecture in ARCHITECTURES and type(seed) is int and 0 <= seed < 2**32,
         "invalid architecture or seed")
    need(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
         "invalid Adam learning rate")
    need(torch.get_default_dtype() == torch.float32, "default dtype must be float32")
    device = torch.device(device)
    need(device.type in ("cpu", "cuda") and (device.type != "cuda" or torch.cuda.is_available()),
         "requested device is unavailable")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model = FullTeacherPolicy(architecture)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    need(not optimizer.state and all(p.requires_grad for p in model.parameters()), "model or Adam is not fresh")
    return model, optimizer


def state_fingerprint(model):
    """Stable SHA-256 over parameter names, shapes, dtypes and tensor bytes."""
    full = hashlib.sha256()
    values = {}
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        need(value.is_floating_point() and bool(torch.isfinite(value).all()), f"invalid tensor: {name}")
        raw = value.numpy().tobytes()
        descriptor = _json_bytes({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                  "sha256": _sha(raw)})
        full.update(len(descriptor).to_bytes(8, "big")); full.update(descriptor); full.update(raw)
        values[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _sha(raw)}
    return {"sha256": full.hexdigest(), "tensors": values}


def verify_adam(model, optimizer, updates, learning_rate):
    """Check the saved Adam owns the whole policy and has the expected steps."""
    need(type(updates) is int and updates > 0 and len(optimizer.param_groups) == 1, "Adam update contract differs")
    group = optimizer.param_groups[0]
    parameters = list(model.parameters())
    need(len(group["params"]) == len(parameters) and all(a is b for a, b in zip(group["params"], parameters)) and
         float(group["lr"]) == float(learning_rate) and tuple(group["betas"]) == (0.9, 0.999) and
         float(group["eps"]) == 1e-8 and float(group["weight_decay"]) == 0.0 and not group["amsgrad"],
         "Adam parameter order or options differ")
    need(len(optimizer.state) == len(parameters), "Adam state is incomplete")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        need(isinstance(state, dict) and set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Adam state fields differ")
        step, first, second = state["step"], state["exp_avg"], state["exp_avg_sq"]
        need(isinstance(step, torch.Tensor) and step.numel() == 1 and bool(torch.isfinite(step).all()) and
             float(step.item()) == updates and isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor) and
             first.shape == second.shape == parameter.shape and first.dtype == second.dtype == parameter.dtype and
             bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()) and bool((second >= 0).all()),
             "Adam step or moments differ")
    return True


def rows_to_batch(rows, device="cpu"):
    need(isinstance(rows, list) and bool(rows), "training rows are empty")
    windows = torch.tensor([r["window"] for r in rows], dtype=torch.float32, device=device)
    masks = torch.tensor([r["mask"] for r in rows], dtype=torch.bool, device=device)
    actions = torch.tensor([r["action_index"] for r in rows], dtype=torch.long, device=device)
    valid = torch.tensor([r["move_target_valid"] for r in rows], dtype=torch.bool, device=device)
    targets = torch.tensor([r["move_target_normalized"] if r["move_target_valid"] else [0.0] * 4 for r in rows],
                           dtype=torch.float32, device=device)
    need(torch.equal(valid, actions == motion_policy.MOVE), "MOVE supervision mask differs")
    adapt_inputs(windows, masks)
    return {"windows": windows, "masks": masks, "actions": actions, "targets": targets, "move_valid": valid}


def supervised_loss(logits, joints, actions, targets, move_valid, joint_weight=100.0):
    need(isinstance(logits, torch.Tensor) and logits.ndim == 2 and logits.shape[1] == len(motion_policy.ACTIONS) and
         joints.shape == logits.shape and actions.shape == logits.shape[:1] and targets.shape == joints.shape and
         move_valid.dtype == torch.bool and move_valid.shape == actions.shape and
         bool(torch.isfinite(logits).all()) and bool(torch.isfinite(joints).all()) and bool(torch.isfinite(targets).all()),
         "invalid or nonfinite model outputs and targets")
    need(actions.dtype == torch.long and bool(((0 <= actions) & (actions < len(motion_policy.ACTIONS))).all()) and
         torch.equal(move_valid, actions == motion_policy.MOVE), "invalid action or MOVE supervision mask")
    need(type(joint_weight) in (int, float) and math.isfinite(joint_weight) and joint_weight > 0,
         "invalid joint loss weight")
    ce = F.cross_entropy(logits, actions)
    mse = F.mse_loss(joints[move_valid], targets[move_valid]) if bool(move_valid.any()) else joints.sum() * 0.0
    return ce + float(joint_weight) * mse, ce, mse


class BatchStream:
    """A private RNG shuffles whole passes and fills batches across boundaries."""

    def __init__(self, row_count, batch_size, seed):
        need(type(row_count) is int and row_count > 0 and type(batch_size) is int and batch_size > 0 and
             type(seed) is int and 0 <= seed < 2**32, "invalid sampling settings")
        self.row_count, self.batch_size = row_count, batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.order, self.position, self.passes_started = None, 0, 0

    def next(self):
        pieces, remaining = [], self.batch_size
        while remaining:
            if self.order is None or self.position == self.row_count:
                self.order = torch.randperm(self.row_count, generator=self.generator)
                self.position = 0
                self.passes_started += 1
            take = min(remaining, self.row_count - self.position)
            pieces.append(self.order[self.position:self.position + take])
            self.position += take
            remaining -= take
        return torch.cat(pieces)


@dataclass(frozen=True)
class Settings:
    architecture: str
    episodes: int
    seed: int
    updates: int = 3000
    batch_size: int = 256
    learning_rate: float = 0.002
    joint_weight: float = 100.0
    device: str = "cpu"

    def validate(self):
        need(self.architecture in ARCHITECTURES and self.episodes in (10, 100) and
             type(self.seed) is int and 0 <= self.seed < 2**32 and
             type(self.updates) is int and self.updates > 0 and type(self.batch_size) is int and self.batch_size > 0 and
             type(self.learning_rate) in (int, float) and math.isfinite(self.learning_rate) and self.learning_rate > 0 and
             type(self.joint_weight) in (int, float) and math.isfinite(self.joint_weight) and self.joint_weight > 0,
             "invalid training settings")


def _snapshot_inputs(directory, manifest):
    expected = {"manifest.json": _file_sha(directory / "manifest.json")}
    for name, spec in manifest["files"].items():
        expected[name] = spec["sha256"]
        need(_file_sha(directory / name) == spec["sha256"], "converted dataset changed")
    return expected


def _check_input_snapshot(directory, expected):
    need(all((directory / name).is_file() and _file_sha(directory / name) == digest
             for name, digest in expected.items()), "converted dataset changed during execution")


def _source_snapshot(data):
    references = [data["manifest"][name] for name in ("review_report", "selection", "source_plan", "exporter")]
    snapshot = {}
    for reference in references:
        name, digest = str(Path(reference["path"]).resolve()), reference["sha256"]
        need(name not in snapshot or snapshot[name] == digest, "source path has conflicting fixed fingerprints")
        snapshot[name] = digest
    files = [Path(__file__), Path(scene_policy.__file__), Path(scene_features.__file__), Path(motion_policy.__file__),
             Path(kinematics.__file__), Path(exported.planner.__file__), Path(exported.review.__file__),
             Path(exported.collection.__file__)]
    for path in files:
        name, digest = str(path.resolve()), _file_sha(path)
        need(name not in snapshot or snapshot[name] == digest, "source code differs from fixed fingerprint")
        snapshot[name] = digest
    _check_source_snapshot(snapshot)
    return snapshot


def _check_source_snapshot(snapshot):
    need(all(Path(name).is_file() and _file_sha(name) == digest for name, digest in snapshot.items()),
         "review, selection, plan or source code changed during execution")


def _write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value))


def train(dataset_directory, expected_manifest_sha256, output, settings, *, start_guard=None):
    """Run exactly the specified fixed budget; no retries or physical execution."""
    settings.validate()
    data = load_converted_dataset(dataset_directory, expected_manifest_sha256)
    directory, output = data["directory"], Path(output).resolve()
    need(output != directory and directory not in output.parents, "output overlaps converted dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists; inspect it instead of retrying")
    need(output.parent.is_dir(), "output parent directory must already exist")
    snapshot = _snapshot_inputs(directory, data["manifest"])
    need(snapshot["manifest.json"] == expected_manifest_sha256, "dataset manifest changed")
    sources = _source_snapshot(data)
    rows = data["rows"][settings.episodes]
    model, optimizer = initialize_policy(settings.architecture, settings.seed, settings.learning_rate, settings.device)
    initial = state_fingerprint(model)
    batch = rows_to_batch(rows, device=settings.device)
    stream = BatchStream(len(rows), settings.batch_size, settings.seed)
    first_indices, guard_evidence = None, None
    if start_guard is not None:
        need(callable(start_guard), "training start guard must be callable")
        first_indices = stream.next()
        guard_evidence = start_guard(data, settings, model, optimizer, stream, first_indices)
        need(isinstance(guard_evidence, dict), "training start guard did not return evidence")
    exposure = torch.zeros(len(rows), dtype=torch.int64)
    config = {"schema": RUN_SCHEMA, "settings": asdict(settings), "dataset_manifest_sha256": expected_manifest_sha256,
              "training_file_sha256": data["manifest"]["files"][f"train_{settings.episodes}.jsonl"]["sha256"],
              "initial_state": initial, "initial_adam_state_empty": not bool(optimizer.state),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "batch_rng": {"kind": "independent CPU torch.Generator", "seed": settings.seed,
                            "rule": "shuffle each full pass and fill each batch across pass boundaries"},
              "input_adapter": "copy 8x82 window; set moving column 5 to zero; preserve mask",
              "loss": "all-action cross entropy + joint_weight * mean MOVE-only normalized joint MSE",
              "source_sha256": sources, "review_source_count": len(data["review_source_hashes"]),
              "review_source_manifest_sha256": _sha(_json_bytes(data["review_source_hashes"])),
              "non_move_target": None, "automatic_retry": False}
    if guard_evidence is not None:
        config["start_guard"] = guard_evidence
    _check_input_snapshot(directory, snapshot)
    _check_source_snapshot(sources)
    exported._check_sources(data["review_source_hashes"])
    output.mkdir(exist_ok=False)
    _write_new(output / "train_config.json", config)
    events_path = output / "train_events.jsonl"
    completed, active, step_started, step_returned = 0, None, False, False
    order_hash = hashlib.sha256()
    start = time.perf_counter()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model.train()
        with events_path.open("x", encoding="utf-8", newline="\n") as events:
            for update in range(1, settings.updates + 1):
                active, step_started, step_returned = update, False, False
                indices = first_indices if update == 1 and first_indices is not None else stream.next()
                indices_list = indices.tolist()
                event = {"event": "update_started", "update": update, "row_indices": indices_list}
                events.write(_json_bytes(event).decode("utf-8")); events.flush()
                order_hash.update(_json_bytes(indices_list))
                selected = indices.to(device=batch["actions"].device)
                optimizer.zero_grad(set_to_none=True)
                logits, joints = model(batch["windows"][selected], batch["masks"][selected])
                total, ce, mse = supervised_loss(logits, joints, batch["actions"][selected],
                                                batch["targets"][selected], batch["move_valid"][selected],
                                                settings.joint_weight)
                need(bool(torch.isfinite(total)), "nonfinite loss")
                total.backward()
                need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                     "nonfinite gradient")
                step_started = True
                optimizer.step()
                step_returned = True
                exposure += torch.bincount(indices, minlength=len(rows))
                completed = update
                event = {"event": "update_returned", "update": update, "loss": float(total.detach()),
                         "action_ce": float(ce.detach()), "move_mse": float(mse.detach()),
                         "move_count": int(batch["move_valid"][selected].sum().item())}
                events.write(_json_bytes(event).decode("utf-8")); events.flush()
        _check_input_snapshot(directory, snapshot)
        _check_source_snapshot(sources)
        exported._check_sources(data["review_source_hashes"])
        final = state_fingerprint(model)
        verify_adam(model, optimizer, completed, settings.learning_rate)
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "architecture": settings.architecture, "seed": settings.seed,
                      "updates": completed, "dataset_manifest_sha256": expected_manifest_sha256,
                      "episodes": settings.episodes, "settings": asdict(settings),
                      "initial_state_sha256": initial["sha256"],
                      "input_adapter": config["input_adapter"], "model_state": model.state_dict(),
                      "source_sha256": sources, "review_source_hashes": data["review_source_hashes"],
                      "optimizer_state": optimizer.state_dict()}
        with (output / "checkpoint.pt").open("xb") as target:
            torch.save(checkpoint, target)
        exposure_rows = [{"row_index": i, "trial_id": row["trial_id"], "teacher_decision": row["teacher_decision"],
                          "exposure_count": int(exposure[i])} for i, row in enumerate(rows)]
        episode_counts = Counter()
        for item in exposure_rows:
            episode_counts[item["trial_id"]] += item["exposure_count"]
        exposure_report = {"rows": exposure_rows, "per_episode": dict(sorted(episode_counts.items())),
                           "total": int(exposure.sum().item()), "expected_total": completed * settings.batch_size}
        need(exposure_report["total"] == exposure_report["expected_total"], "actual exposure accounting differs")
        _write_new(output / "row_exposures.json", exposure_report)
        _check_input_snapshot(directory, snapshot)
        _check_source_snapshot(sources)
        exported._check_sources(data["review_source_hashes"])
        result = {"schema": RUN_SCHEMA, "status": "complete", "updates": completed,
                  "dataset_manifest_sha256": expected_manifest_sha256, "settings": asdict(settings),
                  "initial_state_sha256": initial["sha256"], "final_state_sha256": final["sha256"],
                  "batch_order_sha256": order_hash.hexdigest(), "passes_started": stream.passes_started,
                  "elapsed_seconds": time.perf_counter() - start,
                  "files": {name: _file_sha(output / name) for name in
                            ("train_config.json", "train_events.jsonl", "checkpoint.pt", "row_exposures.json")},
                  "physical_execution": False, "automatic_retry": False}
        _write_new(output / "result.json", result)
        return {"output": str(output), "result_sha256": _file_sha(output / "result.json"),
                "checkpoint_sha256": result["files"]["checkpoint.pt"], "updates": completed}
    except BaseException as exc:
        try:
            _write_new(output / "training_interrupted.json", {"schema": RUN_SCHEMA, "status": "interrupted",
                "last_returned_update": completed, "active_update": active, "optimizer_step_started": step_started,
                "optimizer_step_returned": step_returned, "exception_type": type(exc).__name__,
                "message": str(exc), "automatic_retry": False})
        except Exception:
            pass
        raise
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)


def load_checkpoint(path, expected_sha256, *, device="cpu"):
    path = Path(path)
    need(isinstance(expected_sha256, str) and _SHA.fullmatch(expected_sha256) and _file_sha(path) == expected_sha256,
         "checkpoint fingerprint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(isinstance(payload, dict) and payload.get("schema") == CHECKPOINT_SCHEMA and
         payload.get("architecture") in ARCHITECTURES and type(payload.get("seed")) is int and
         type(payload.get("updates")) is int and payload["updates"] > 0 and payload.get("episodes") in (10, 100) and
         payload.get("input_adapter") == "copy 8x82 window; set moving column 5 to zero; preserve mask",
         "checkpoint contract differs")
    result, _ = _json(path.parent / "result.json")
    need(isinstance(result, dict) and result.get("schema") == RUN_SCHEMA and result.get("status") == "complete" and
         result.get("updates") == payload["updates"] and result.get("dataset_manifest_sha256") ==
         payload.get("dataset_manifest_sha256") and result.get("files", {}).get("checkpoint.pt") == expected_sha256,
         "completed training evidence differs")
    for name, digest in result.get("files", {}).items():
        need(name in ("train_config.json", "train_events.jsonl", "checkpoint.pt", "row_exposures.json") and
             _file_sha(path.parent / name) == digest, "completed training file fingerprint differs")
    need(set(result.get("files", {})) == {"train_config.json", "train_events.jsonl", "checkpoint.pt", "row_exposures.json"},
         "completed training files missing")
    config, _ = _json(path.parent / "train_config.json")
    need(isinstance(config, dict) and config.get("schema") == RUN_SCHEMA and
         config.get("settings") == payload.get("settings") == result.get("settings") and
         config.get("dataset_manifest_sha256") == payload.get("dataset_manifest_sha256") and
         config.get("initial_adam_state_empty") is True and isinstance(config.get("initial_state"), dict) and
         config.get("initial_state", {}).get("sha256") == payload.get("initial_state_sha256") ==
         result.get("initial_state_sha256") and config.get("source_sha256") == payload.get("source_sha256"),
         "training configuration and checkpoint provenance differ")
    review_sources = payload.get("review_source_hashes")
    need(isinstance(review_sources, dict) and bool(review_sources) and
         all(isinstance(k, str) and k and isinstance(v, str) and _SHA.fullmatch(v) for k, v in review_sources.items()) and
         config.get("review_source_count") == len(review_sources) and
         config.get("review_source_manifest_sha256") == _sha(_json_bytes(review_sources)),
         "checkpoint review source fingerprints differ")
    exported._check_sources(review_sources)
    need(isinstance(payload.get("source_sha256"), dict) and bool(payload["source_sha256"]),
         "checkpoint source fingerprints missing")
    _check_source_snapshot(payload["source_sha256"])
    need(isinstance(payload.get("settings"), dict), "checkpoint training settings missing")
    try:
        settings = Settings(**payload["settings"])
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint training settings differ") from exc
    settings.validate()
    need(settings.architecture == payload["architecture"] and settings.seed == payload["seed"] and
         settings.episodes == payload["episodes"] and settings.updates == payload["updates"],
         "checkpoint settings and completed updates differ")
    model, optimizer = initialize_policy(payload["architecture"], payload["seed"], settings.learning_rate, device="cpu")
    state = payload.get("model_state")
    expected = model.state_dict()
    need(isinstance(state, dict) and set(state) == set(expected), "checkpoint tensor names differ")
    for name, tensor in state.items():
        need(isinstance(tensor, torch.Tensor) and tensor.dtype == expected[name].dtype and
             tensor.shape == expected[name].shape and bool(torch.isfinite(tensor).all()),
             f"checkpoint tensor differs: {name}")
    model.load_state_dict(state, strict=True)
    need(state_fingerprint(model)["sha256"] == result.get("final_state_sha256"), "final model fingerprint differs")
    try:
        optimizer.load_state_dict(payload["optimizer_state"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("checkpoint Adam state differs") from exc
    verify_adam(model, optimizer, payload["updates"], settings.learning_rate)
    model.to(device); model.eval()
    return model, payload


@torch.no_grad()
def recorded_metrics(model, batch, batch_size=256):
    """Recorded-row metrics through the same model adapter, not physical success."""
    need(type(batch_size) is int and batch_size > 0, "batch size must be positive")
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    total_abs = torch.zeros(4, dtype=torch.float64)
    squared, action_sum, moves = 0.0, 0.0, 0
    count = len(batch["actions"])
    need(count > 0, "recorded data is empty")
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        logits, joints = model(batch["windows"][start:end], batch["masks"][start:end])
        actual = batch["actions"][start:end]
        valid = batch["move_valid"][start:end]
        need(torch.equal(valid, actual == motion_policy.MOVE) and
             bool(torch.isfinite(logits).all()) and bool(torch.isfinite(joints).all()), "recorded output is invalid")
        action_sum += float(F.cross_entropy(logits, actual, reduction="sum"))
        confusion += torch.bincount((actual * 4 + logits.argmax(-1)).cpu(), minlength=16).reshape(4, 4)
        if bool(valid.any()):
            error = (joints[valid] - batch["targets"][start:end][valid]).detach().cpu().double()
            total_abs += error.abs().sum(0) * 180.0
            squared += float(error.square().sum())
            moves += int(valid.sum())
    return {"rows": count, "move_rows": moves, "explicit_done_rows": int(confusion[motion_policy.DONE].sum()),
            "action_accuracy": float(confusion.diagonal().sum().item() / count),
            "action_ce": action_sum / count, "action_confusion": confusion.tolist(),
            "move_mse_normalized": squared / (4 * moves) if moves else None,
            "move_mae_per_joint_deg": (total_abs / moves).tolist() if moves else None,
            "scope": "selected training teacher rows only; not held-out evaluation or student physical success"}


def record(dataset_directory, expected_manifest_sha256, checkpoint_path, expected_checkpoint_sha256, output,
           *, device="cpu", batch_size=256):
    data = load_converted_dataset(dataset_directory, expected_manifest_sha256)
    directory, output = data["directory"], Path(output).resolve()
    need(output != directory and directory not in output.parents, "output overlaps converted dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists; inspect it instead of retrying")
    need(output.parent.is_dir(), "output parent directory must already exist")
    snapshot = _snapshot_inputs(directory, data["manifest"])
    need(snapshot["manifest.json"] == expected_manifest_sha256, "dataset manifest changed")
    sources = _source_snapshot(data)
    model, payload = load_checkpoint(checkpoint_path, expected_checkpoint_sha256, device=device)
    need(payload.get("dataset_manifest_sha256") == expected_manifest_sha256, "checkpoint dataset differs")
    rows = data["rows"][payload["episodes"]]
    exported._check_sources(data["review_source_hashes"])
    metrics = recorded_metrics(model, rows_to_batch(rows, device=device), batch_size)
    _check_input_snapshot(directory, snapshot)
    _check_source_snapshot(sources)
    exported._check_sources(data["review_source_hashes"])
    need(_file_sha(checkpoint_path) == expected_checkpoint_sha256, "checkpoint changed during recorded evaluation")
    output.mkdir(exist_ok=False)
    report = {"schema": RECORD_SCHEMA, "status": "complete", "dataset_manifest_sha256": expected_manifest_sha256,
              "checkpoint_sha256": expected_checkpoint_sha256, "episodes": payload["episodes"],
              "architecture": payload["architecture"], "seed": payload["seed"], "metrics": metrics,
              "evaluation_subset": "selected_training_rows", "input_adapter": payload["input_adapter"],
              "physical_execution": False}
    _write_new(output / "record.json", report)
    return {"output": str(output), "record_sha256": _file_sha(output / "record.json"), "metrics": metrics}
