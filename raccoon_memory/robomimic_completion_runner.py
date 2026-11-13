"""Fixed completion-head experiment helpers; importing performs no work.

The only model input is the existing 82 current features plus the post-observation
official LSTM h. Provenance and sampling identities are stored separately.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from raccoon_memory.robomimic_completion import (
    CompletionHead, FrozenOfficialCore, completion_checkpoint, restore_completion_head,
)
from raccoon_memory.robomimic_completion_training import (
    completion_loss, extract_episode, make_head_optimizer, train_head_step,
)
from raccoon_memory.robomimic_scene_sequences import sha256_file
from raccoon_memory.robomimic_scene_training import RESUME_SCHEMA, model_weight_sha256, state_dict_equal


CACHE_SCHEMA = "raccoon-robomimic-completion-cache.v1"
RECOVERY_SCHEMA = "raccoon-robomimic-completion-resume.v1"
FINAL_SCHEMA = "raccoon-robomimic-completion-final.v1"
DRAW_DTYPE = np.dtype([("unit", "<u2"), ("variant", "u1"), ("row", "<u4")])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _publish(path, writer):
    """Publish a fully flushed new file without replacing another result."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"already exists: {path}")
    temporary = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _publish(path, lambda stream: stream.write(data))


def write_torch(path, value):
    _publish(path, lambda stream: torch.save(value, stream))


def _same_json_or_write(path, value):
    if Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"existing result differs: {path}")
    else:
        write_json(path, value)


def verify_final_core(seed_dir, *, seed, data_sha256, spec_sha256, expected_identity,
                      expected_updates=20000, expected_samples=2000000, verify_counts=None):
    """Verify final publication and its recovery point before extracting anything."""
    seed_dir = Path(seed_dir)
    manifest_path = seed_dir / "final_manifest.json"
    manifest = read_json(manifest_path)
    final_path = seed_dir / "final_scene_core.pt"
    recovery_path = seed_dir / "checkpoints" / f"step_{expected_updates:06d}.pt"
    if (manifest.get("seed") != seed or manifest.get("updates") != expected_updates
            or manifest.get("checkpoint") != final_path.name
            or manifest.get("selection") != "fixed final update, no performance selection"
            or type(manifest.get("valid_actions")) is not int or manifest["valid_actions"] <= 0
            or manifest.get("checkpoint_sha256") != sha256_file(final_path)):
        raise ValueError("final action-core manifest is missing, incomplete, or mismatched")
    recovery_sha = sha256_file(recovery_path)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    cumulative = recovery.get("cumulative", {})
    if (recovery.get("schema") != RESUME_SCHEMA or recovery.get("identity") != expected_identity
            or recovery.get("updates_completed") != expected_updates
            or cumulative.get("updates") != expected_updates
            or cumulative.get("sampled_chunks") != expected_samples
            or cumulative.get("valid_actions") != manifest["valid_actions"]
            or not state_dict_equal(recovery.get("model"), final.get("model"))
            or recovery.get("config") != final.get("config")):
        raise ValueError("final model or Adam state differs from its 20000-update recovery point")
    if verify_counts is not None:
        verify_counts(cumulative)
    core = FrozenOfficialCore.from_trained_scene_checkpoint(
        final_path, expected_spec_sha256=spec_sha256, expected_dataset_sha256=data_sha256, device="cpu")
    if (core.checkpoint_provenance.get("training_seed") != seed
            or core.checkpoint_provenance.get("optimizer_updates") != expected_updates
            or core.checkpoint_file_sha256 != manifest["checkpoint_sha256"]
            or model_weight_sha256(core.algorithm) != recovery.get("model_weight_sha256")
            or sha256_file(recovery_path) != recovery_sha):
        raise ValueError("restored action core differs from final recovery metadata")
    core.assert_frozen()
    return core, {"seed": seed, "final_manifest_sha256": sha256_file(manifest_path),
                  "final_checkpoint_sha256": core.checkpoint_file_sha256,
                  "recovery_checkpoint_sha256": recovery_sha, "core_weights_sha256": core.weights_sha256,
                  "updates": expected_updates, "valid_actions": manifest["valid_actions"]}


def unit_lookup(units):
    """Reuse exactly the action runner's parent -> representation mapping."""
    if not isinstance(units, list) or not units:
        raise ValueError("balanced units are required")
    lookup = {}
    for unit_index, unit in enumerate(units):
        variants = unit.get("variants")
        if not isinstance(variants, list) or not variants or len(variants) > 255:
            raise ValueError("unit has invalid variants")
        for variant_index, variant in enumerate(variants):
            eid = variant.get("episode_id")
            if not isinstance(eid, str) or not eid or eid in lookup:
                raise ValueError("duplicate or missing episode in action index")
            lookup[eid] = (unit_index, variant_index)
    return lookup


def check_episode_identity(episode, units, lookup):
    """Check the current episode against the already-audited action index."""
    from scripts.prepare_robomimic_scene_training_index import identity

    eid = episode["episode_id"]
    if eid not in lookup:
        raise ValueError("training episode is absent from the action index")
    u, v = lookup[eid]
    unit, variant = units[u], units[u]["variants"][v]
    kind = "bundle_train" if unit["kind"] == "physical" else unit["kind"]
    if (episode["source_kind"] != kind or episode["group"] != variant["group"]
            or episode["object_count"] != variant["object_count"]
            or episode.get("source_episode_line_sha256") != variant.get("source_episode_line_sha256")):
        raise ValueError("episode identity or source lineage differs from the action index")
    if kind == "bundle_train":
        if (episode.get("source_lineage", {}).get("original_physical_episode_id") != unit["unit_id"]
                or list(identity(episode, unit["unit_id"])) != variant["identity_permutation"]):
            raise ValueError("physical parent or identity permutation differs")
    elif eid != unit["unit_id"]:
        raise ValueError("non-physical unit identity differs")
    return u, v


def _episode_seed(seed, split, eid):
    message = f"completion-feature-v1|{seed}|{split}|{eid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") & ((1 << 63) - 1)


def _flush(arrays, directory):
    for name, array in arrays.items():
        array.flush()
        with (directory / f"{name}.npy").open("r+b") as stream:
            os.fsync(stream.fileno())


def _content_hash(features, labels, mask):
    digest = hashlib.sha256()
    for value in (features, labels, mask):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def build_split_cache(directory, *, core, seed, split, episodes, expected_episodes, expected_rows,
                      source_identity, units=None, resume=False, progress_interval=100):
    """Extract episode-ordered features with restartable, verified boundaries.

    `episodes` is a fresh iterable on every call. Each episode has its own
    deterministic sampling RNG, so replay after a committed boundary is exact.
    """
    directory = Path(directory)
    if split not in ("train", "validation") or progress_interval <= 0:
        raise ValueError("invalid cache split or progress interval")
    if directory.exists() and not resume:
        raise FileExistsError("cache exists; explicit resume required")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"seed": seed, "split": split, "source_identity": source_identity,
                "core_weights_sha256": core.weights_sha256,
                "core_checkpoint_sha256": core.checkpoint_file_sha256,
                "core_synthetic_mode": core.synthetic_test,
                "expected_episodes": expected_episodes, "expected_rows": expected_rows}
    identity_path = directory / "identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("cache initialization identity differs")
    else:
        if any(directory.glob("*.npy")) or any(directory.glob("progress_*.json")) or (directory / "manifest.json").exists():
            raise ValueError("cache files exist without a committed initialization identity")
        write_json(identity_path, identity)
    final_path = directory / "manifest.json"
    if final_path.exists():
        manifest, arrays, entries = load_cache(directory, identity=identity)
        return manifest
    lookup = unit_lookup(units) if split == "train" else None
    paths = {"features": (np.float32, (expected_rows, 482)),
             "labels": (np.int8, (expected_rows,)), "mask": (np.bool_, (expected_rows,))}
    progress = sorted(directory.glob("progress_*.json"))
    if progress:
        saved = read_json(progress[-1])
        if saved.get("identity") != identity or not isinstance(saved.get("entries"), list):
            raise ValueError("cache progress identity differs")
        entries = saved["entries"]
        if progress[-1].name != f"progress_{len(entries):06d}.json":
            raise ValueError("cache progress boundary differs")
        arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r+", allow_pickle=False) for name in paths}
    else:
        # No episode has been committed. Matching initialization identity permits
        # safely recreating even a truncated array left during initial allocation.
        arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
                  for name, (dtype, shape) in paths.items()}
        _flush(arrays, directory)
        entries = []
        write_json(directory / "progress_000000.json", {"identity": identity, "entries": entries})
    for name, (dtype, shape) in paths.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ValueError("cache array dtype or shape differs")
    cursor, seen = 0, set()
    old_count = len(entries)
    for number, episode in enumerate(episodes):
        eid, records = episode["episode_id"], episode["records"]
        if episode["split"] != split or eid in seen:
            raise ValueError("duplicate episode or wrong cache split")
        seen.add(eid)
        u, v = check_episode_identity(episode, units, lookup) if split == "train" else (None, None)
        n, start = len(records), cursor
        cursor += n
        if cursor > expected_rows:
            raise ValueError("too many source observations")
        eligible = [i for i, record in enumerate(records) if record["terminal_supervision_valid"]]
        labels = [int(record["terminal_label"]) if record["terminal_supervision_valid"] else -1 for record in records]
        if not eligible or sum(label == 1 for label in labels) != 1 or labels[-1] != 1:
            raise ValueError("episode completion supervision differs")
        terminal_only = bool((n - 1) % 10 == 0)
        basic = {"episode_id": eid, "offset": start, "length": n, "unit": u, "variant": v,
                 "eligible": eligible, "terminal_only_chunk": terminal_only}
        if number < old_count:
            item = entries[number]
            if any(item.get(key) != value for key, value in basic.items()):
                raise ValueError("committed cache episode differs from source order or lineage")
            cached_labels = arrays["labels"][start:cursor]
            cached_mask = arrays["mask"][start:cursor]
            if (not np.array_equal(cached_labels, np.asarray(labels, dtype=np.int8))
                    or not np.array_equal(cached_mask, np.asarray([x >= 0 for x in labels], dtype=np.bool_))
                    or item.get("content_sha256") != _content_hash(arrays["features"][start:cursor], cached_labels, cached_mask)
                    or not np.array_equal(arrays["features"][start:cursor, :82],
                                          np.asarray([r["features"] for r in records], dtype=np.float32))):
                raise ValueError("committed cache data differs from source or recorded fingerprint")
            continue
        torch.manual_seed(_episode_seed(seed, split, eid))
        extracted = extract_episode(core, episode)
        if (extracted["split"] != split or extracted["episode_id"] != eid
                or extracted["core_weights_sha256"] != core.weights_sha256
                or extracted["core_checkpoint_sha256"] != core.checkpoint_file_sha256
                or tuple(extracted["features"].shape) != (n, 482)
                or extracted["decision_counts"] != list(range(1, n + 1))):
            raise ValueError("extracted feature identity or observation count differs")
        x = extracted["features"].numpy()
        y = extracted["terminal_label"].numpy().astype(np.int8)
        mask = extracted["terminal_mask"].numpy()
        if (not np.isfinite(x).all() or not np.array_equal(y, labels)
                or not np.array_equal(mask, np.asarray([x >= 0 for x in labels], dtype=np.bool_))
                or not np.array_equal(x[:, :82], np.asarray([r["features"] for r in records], dtype=np.float32))):
            raise ValueError("extracted values or completion supervision differ from source")
        arrays["features"][start:cursor] = x
        arrays["labels"][start:cursor] = y
        arrays["mask"][start:cursor] = mask
        entries.append({**basic, "content_sha256": _content_hash(x, y, mask)})
        if len(entries) % progress_interval == 0:
            _flush(arrays, directory)
            write_json(directory / f"progress_{len(entries):06d}.json", {"identity": identity, "entries": entries})
    if len(entries) != expected_episodes or cursor != expected_rows or len(seen) != expected_episodes:
        raise ValueError("cache episode or source-observation count differs")
    if split == "train" and set(seen) != set(lookup):
        raise ValueError("cached training episodes do not cover the fixed action units")
    core.assert_frozen()
    _flush(arrays, directory)
    _same_json_or_write(directory / "episodes.json", entries)
    counts = {"observations": cursor, "supervised": int(arrays["mask"].sum()),
              "continue": int((arrays["labels"] == 0).sum()), "done": int((arrays["labels"] == 1).sum()),
              "prefix_only": int((~arrays["mask"]).sum()),
              "terminal_only_chunks": sum(item["terminal_only_chunk"] for item in entries)}
    files = {f"{name}.npy": sha256_file(directory / f"{name}.npy") for name in paths}
    files["episodes.json"] = sha256_file(directory / "episodes.json")
    files["identity.json"] = sha256_file(identity_path)
    manifest = {"schema": CACHE_SCHEMA, "identity": identity, "counts": counts, "files": files,
                "observation": "current scene[82] plus post-observation LSTM top h[400]",
                "episode_rng": "sha256(seed,split,episode_id); sampled actions discarded"}
    _same_json_or_write(final_path, manifest)
    return manifest


def load_cache(directory, *, identity):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("identity") != identity:
        raise ValueError("completed cache identity differs")
    expected_files = {"features.npy", "labels.npy", "mask.npy", "episodes.json", "identity.json"}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("cache file inventory differs")
    for filename, digest in manifest["files"].items():
        if sha256_file(directory / filename) != digest:
            raise ValueError(f"cache file fingerprint differs: {filename}")
    if read_json(directory / "identity.json") != identity:
        raise ValueError("cache initialization identity differs")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("features", "labels", "mask")}
    entries = read_json(directory / "episodes.json")
    rows = identity["expected_rows"]
    if (arrays["features"].shape != (rows, 482) or arrays["features"].dtype != np.float32
            or arrays["labels"].shape != (rows,) or arrays["labels"].dtype != np.int8
            or arrays["mask"].shape != (rows,) or arrays["mask"].dtype != np.bool_
            or len(entries) != identity["expected_episodes"]):
        raise ValueError("cache arrays or episode count differ")
    return manifest, arrays, entries


class CompletionSampler:
    """Uniform original unit, then ID representation, then supervised observation."""

    def __init__(self, entries, units, *, seed):
        self.units = units
        self.choices = [[None] * len(unit["variants"]) for unit in units]
        for episode in entries:
            u, v = episode["unit"], episode["variant"]
            if (type(u) is not int or type(v) is not int or not (0 <= u < len(self.choices))
                    or not (0 <= v < len(self.choices[u])) or self.choices[u][v] is not None
                    or episode["episode_id"] != units[u]["variants"][v]["episode_id"]):
                raise ValueError("training cache unit mapping differs")
            rows = np.asarray(episode["eligible"], dtype=np.int64) + int(episode["offset"])
            if not len(rows) or np.any(rows < 0) or np.any(rows >= 2 ** 32):
                raise ValueError("unit has no valid completion observation")
            self.choices[u][v] = rows.astype(np.uint32)
        if any(any(rows is None for rows in variants) for variants in self.choices):
            raise ValueError("training cache does not cover all unit representations")
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def draw(self, count):
        if type(count) is not int or count <= 0:
            raise ValueError("sample count must be positive")
        result = np.empty(count, dtype=DRAW_DTYPE)
        unit = self.rng.integers(0, len(self.choices), size=count, dtype=np.uint16)
        variants = np.asarray([len(self.choices[int(u)]) for u in unit], dtype=np.uint16)
        variant = self.rng.integers(0, variants, size=count, dtype=np.uint16)
        sizes = np.asarray([len(self.choices[int(u)][int(v)]) for u, v in zip(unit, variant)], dtype=np.uint32)
        choice = self.rng.integers(0, sizes, size=count, dtype=np.uint32)
        rows = np.asarray([self.choices[int(u)][int(v)][int(i)] for u, v, i in zip(unit, variant, choice)], dtype=np.uint32)
        result["unit"], result["variant"], result["row"] = unit, variant, rows
        return result

    def validate(self, draws, mask):
        draws = np.asarray(draws)
        if draws.dtype != DRAW_DTYPE:
            raise ValueError("completion draw dtype differs")
        for item in draws.reshape(-1):
            u, v, row = int(item["unit"]), int(item["variant"]), int(item["row"])
            if (u >= len(self.choices) or v >= len(self.choices[u]) or row >= len(mask)
                    or not mask[row] or row not in self.choices[u][v]):
                raise ValueError("draw selects an invalid or unsupervised observation")

    def state_dict(self):
        import copy
        return copy.deepcopy(self.rng.bit_generator.state)

    def load_state_dict(self, state):
        import copy
        if state.get("bit_generator") != "PCG64":
            raise ValueError("completion sampler RNG differs")
        self.rng.bit_generator.state = copy.deepcopy(state)


def _head_hash(head):
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_steps(optimizer):
    steps = set()
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None or not torch.isfinite(value).all() or value.numel() != 1:
            raise ValueError("completion Adam state has an invalid step")
        value = float(value.item())
        if value != int(value):
            raise ValueError("completion Adam state has a nonintegral step")
        steps.add(int(value))
    return steps


def _validate_optimizer(optimizer, head, completed, learning_rate):
    parameters = list(head.parameters())
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
        raise ValueError("completion optimizer is not the fixed Adam")
    group = optimizer.param_groups[0]
    if (len(group["params"]) != len(parameters)
            or any(left is not right for left, right in zip(group["params"], parameters))
            or group["lr"] != learning_rate or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8 or group["weight_decay"] != 0
            or group["amsgrad"] or group["maximize"]):
        raise ValueError("completion Adam parameters or settings differ")
    if _optimizer_steps(optimizer) != (set() if completed == 0 else {completed}):
        raise ValueError("completion Adam state and update count differ")
    if completed and len(optimizer.state) != len(parameters):
        raise ValueError("completion Adam state is incomplete")
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError("completion Adam moment is invalid")


def _draw_hash(draws, count):
    return hashlib.sha256(np.ascontiguousarray(draws[:count]).tobytes()).hexdigest()


def evaluate_cache(head, arrays, *, batch_size=4096):
    """Recorded-observation argmax confusion; does not choose a checkpoint."""
    head.eval()
    confusion = np.zeros((2, 2), dtype=np.int64)
    total_loss, total = 0.0, 0
    for start in range(0, len(arrays["mask"]), batch_size):
        end = min(start + batch_size, len(arrays["mask"]))
        selected = np.flatnonzero(arrays["mask"][start:end]) + start
        if not len(selected):
            continue
        features = torch.from_numpy(np.array(arrays["features"][selected], copy=True))
        labels = torch.from_numpy(np.array(arrays["labels"][selected], dtype=np.int64, copy=True))
        with torch.no_grad():
            logits = head(features)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            predicted = logits.argmax(dim=1).cpu().numpy()
        if not torch.isfinite(loss):
            raise ValueError("recorded evaluation loss is not finite")
        confusion += np.bincount(labels.numpy() * 2 + predicted, minlength=4).reshape(2, 2)
        total_loss += float(loss)
        total += len(selected)
    if total == 0:
        raise ValueError("evaluation has no supervised observations")
    return {"supervised_observations": total, "cross_entropy": total_loss / total,
            "accuracy": float(np.trace(confusion) / total),
            "confusion_true_rows_predicted_columns_continue_done": confusion.tolist()}


def _ensure_cache_binding(manifest, core, seed, split):
    identity = manifest.get("identity", {})
    if (manifest.get("schema") != CACHE_SCHEMA or identity.get("seed") != seed or identity.get("split") != split
            or identity.get("core_weights_sha256") != core.weights_sha256
            or identity.get("core_checkpoint_sha256") != core.checkpoint_file_sha256
            or identity.get("core_synthetic_mode") is not core.synthetic_test):
        raise ValueError("cache split, seed or action-core binding differs")


def _save_head_resume(path, *, core, head, optimizer, sampler, identity, completed,
                      draws, initial_sha, class_exposure):
    _validate_optimizer(optimizer, head, completed, identity["learning_rate"])
    core.assert_frozen()
    payload = {"schema": RECOVERY_SCHEMA, "identity": identity, "updates_completed": completed,
               "completion_head": completion_checkpoint(core, head), "head_sha256": _head_hash(head),
               "optimizer": optimizer.state_dict(), "sampler_rng": sampler.state_dict(),
               "torch_rng": torch.get_rng_state(), "draws_sha256": _draw_hash(draws, completed),
               "initial_head_sha256": initial_sha, "class_exposure": class_exposure}
    write_torch(path, payload)


def _restore_head_resume(path, *, core, identity, sampler, draws, labels, expected_batch):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    completed = payload.get("updates_completed")
    if (payload.get("schema") != RECOVERY_SCHEMA or payload.get("identity") != identity
            or type(completed) is not int or not 0 <= completed <= len(draws)
            or payload.get("draws_sha256") != _draw_hash(draws, completed)):
        raise ValueError("completion recovery identity, progress or recorded draws differ")
    head = restore_completion_head(core, payload["completion_head"])
    if _head_hash(head) != payload.get("head_sha256"):
        raise ValueError("restored completion head weights differ")
    optimizer = make_head_optimizer(core, head, learning_rate=identity["learning_rate"])
    optimizer.load_state_dict(payload["optimizer"])
    _validate_optimizer(optimizer, head, completed, identity["learning_rate"])
    counts = [0, 0]
    for step in range(completed):
        regenerated = sampler.draw(expected_batch)
        if not np.array_equal(regenerated, draws[step]):
            raise ValueError("recorded completion draws differ from deterministic sampler")
        chosen = np.asarray(labels[regenerated["row"]], dtype=np.int64)
        if np.any((chosen < 0) | (chosen > 1)):
            raise ValueError("recorded draw has no completion label")
        counts = [counts[i] + int((chosen == i).sum()) for i in (0, 1)]
    if sampler.state_dict() != payload.get("sampler_rng") or counts != payload.get("class_exposure"):
        raise ValueError("completion sample RNG or class exposure differs from committed draws")
    torch.set_rng_state(payload["torch_rng"])
    core.assert_frozen()
    return head, optimizer, completed, payload["initial_head_sha256"], counts


def _training_statistics(draws, labels, units):
    flat = draws.reshape(-1)
    unit_counts = np.bincount(flat["unit"], minlength=len(units))
    pair = flat["unit"].astype(np.int64) * 256 + flat["variant"].astype(np.int64)
    variants = np.bincount(pair, minlength=len(units) * 256).reshape(len(units), 256)
    row_counts = np.bincount(flat["row"].astype(np.int64), minlength=len(labels))
    classes = np.bincount(np.asarray(labels[flat["row"]], dtype=np.int64), minlength=2)
    kinds = {}
    for index, unit in enumerate(units):
        kinds[unit["kind"]] = kinds.get(unit["kind"], 0) + int(unit_counts[index])
    total = len(flat)
    return {"total_supervised_draws": total,
            "class_exposure": {"continue": int(classes[0]), "done": int(classes[1])},
            "class_proportions": {"continue": float(classes[0] / total), "done": float(classes[1] / total)},
            "source_exposure": kinds, "source_proportions": {key: value / total for key, value in kinds.items()},
            "unit_exposure": [{"unit_id": unit["unit_id"], "kind": unit["kind"], "count": int(unit_counts[i]),
                               "variant_counts": [int(x) for x in variants[i, :len(unit["variants"])]]}
                              for i, unit in enumerate(units)],
            "observations_exposed_at_least_once": int((row_counts > 0).sum())}, row_counts


def train_seed(directory, *, core, seed, units, train_manifest, train_arrays, train_entries,
               validation_manifest, validation_arrays, config_sha256, resume=False,
               updates=5000, batch_size=256, learning_rate=0.001, checkpoint_interval=500):
    """Train the last head only; recover optimizer, draws and both RNG states."""
    _ensure_cache_binding(train_manifest, core, seed, "train")
    _ensure_cache_binding(validation_manifest, core, seed, "validation")
    if train_manifest["identity"]["source_identity"] != validation_manifest["identity"]["source_identity"]:
        raise ValueError("training and validation cache source identities differ")
    if (type(updates) is not int or updates <= 0 or type(batch_size) is not int or batch_size <= 0
            or not math.isfinite(learning_rate) or learning_rate <= 0 or checkpoint_interval <= 0):
        raise ValueError("invalid completion training settings")
    directory = Path(directory)
    if directory.exists() and not resume:
        raise FileExistsError("training directory exists; explicit resume required")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    identity = {"seed": seed, "config_sha256": config_sha256, "core_weights_sha256": core.weights_sha256,
                "core_checkpoint_sha256": core.checkpoint_file_sha256,
                "train_cache_files": train_manifest["files"], "validation_cache_files": validation_manifest["files"],
                "source_identity": train_manifest["identity"]["source_identity"],
                "updates": updates, "batch_size": batch_size, "learning_rate": learning_rate,
                "checkpoint_interval": checkpoint_interval}
    identity_path = directory / "identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("completion training initialization identity differs")
    else:
        if (any(checkpoints.glob("step_*.pt")) or (directory / "draws.npy").exists()
                or (directory / "final_completion_head.pt").exists()):
            raise ValueError("training files exist without a committed initialization identity")
        write_json(identity_path, identity)
    sampler_seed = seed + 160000000
    sampler = CompletionSampler(train_entries, units, seed=sampler_seed)
    draw_path = directory / "draws.npy"
    committed = sorted(checkpoints.glob("step_*.pt"))
    if committed:
        draws = np.load(draw_path, mmap_mode="r+", allow_pickle=False)
        if draws.dtype != DRAW_DTYPE or draws.shape != (updates, batch_size):
            raise ValueError("recorded completion draw array differs")
        head, optimizer, start, initial_sha, class_exposure = _restore_head_resume(
            committed[-1], core=core, identity=identity, sampler=sampler, draws=draws,
            labels=train_arrays["labels"], expected_batch=batch_size)
        if (committed[-1].name != f"step_{start:06d}.pt"
                or (start != updates and start % checkpoint_interval != 0)):
            raise ValueError("completion recovery boundary differs")
    else:
        if (directory / "final_completion_head.pt").exists() or (directory / "final_manifest.json").exists():
            raise ValueError("completion final exists without a committed recovery point")
        # Without a checkpoint none of these draws were committed.
        draws = np.lib.format.open_memmap(draw_path, mode="w+", dtype=DRAW_DTYPE, shape=(updates, batch_size))
        torch.manual_seed(seed)
        head = CompletionHead()
        optimizer = make_head_optimizer(core, head, learning_rate=learning_rate)
        start, initial_sha, class_exposure = 0, _head_hash(head), [0, 0]
        draws.flush()
        _save_head_resume(checkpoints / "step_000000.pt", core=core, head=head, optimizer=optimizer,
                          sampler=sampler, identity=identity, completed=0, draws=draws,
                          initial_sha=initial_sha, class_exposure=class_exposure)
    attempt = directory / f"attempt_{uuid.uuid4().hex}.jsonl"
    with attempt.open("x", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"event": "attempt_start", "resume_from_update": start}) + "\n")
        try:
            for step in range(start, updates):
                selected = sampler.draw(batch_size)
                rows = selected["row"]
                if not np.all(train_arrays["mask"][rows]):
                    raise ValueError("sampler selected a prefix or invalid completion record")
                x = torch.from_numpy(np.array(train_arrays["features"][rows], copy=True))
                y = torch.from_numpy(np.array(train_arrays["labels"][rows], dtype=np.int64, copy=True))
                batch = {"split": "train", "features": x, "terminal_label": y,
                         "terminal_mask": torch.ones(batch_size, dtype=torch.bool),
                         "core_weights_sha256": core.weights_sha256,
                         "core_checkpoint_sha256": core.checkpoint_file_sha256,
                         "core_synthetic_mode": core.synthetic_test}
                result = train_head_step(core, head, optimizer, batch)
                if not result["updated"] or result["supervised_rows"] != batch_size or not math.isfinite(result["loss"]):
                    raise ValueError("completion update failed or has nonfinite loss")
                draws[step] = selected
                exposure = [int((y == i).sum().item()) for i in (0, 1)]
                class_exposure = [class_exposure[i] + exposure[i] for i in (0, 1)]
                completed = step + 1
                log.write(json.dumps({"event": "update", "update": completed, "loss": result["loss"],
                                      "continue": exposure[0], "done": exposure[1]}) + "\n")
                if completed % checkpoint_interval == 0 or completed == updates:
                    draws.flush()
                    with draw_path.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    path = checkpoints / f"step_{completed:06d}.pt"
                    _save_head_resume(path, core=core, head=head, optimizer=optimizer, sampler=sampler,
                                      identity=identity, completed=completed, draws=draws,
                                      initial_sha=initial_sha, class_exposure=class_exposure)
                    log.write(json.dumps({"event": "checkpoint_committed", "update": completed,
                                          "sha256": sha256_file(path)}) + "\n")
            if updates > 0 and _head_hash(head) == initial_sha:
                raise ValueError("completion head weights did not change")
            stats, row_counts = _training_statistics(draws, train_arrays["labels"], units)
            if [stats["class_exposure"]["continue"], stats["class_exposure"]["done"]] != class_exposure:
                raise ValueError("final sample exposure differs from committed counters")
            row_path = directory / "row_exposure.npy"
            if row_path.exists():
                if not np.array_equal(np.load(row_path, allow_pickle=False), row_counts):
                    raise ValueError("existing per-observation exposure differs")
            else:
                _publish(row_path, lambda stream: np.save(stream, row_counts, allow_pickle=False))
            final_payload = {"schema": FINAL_SCHEMA, "identity": identity, "updates_completed": updates,
                             "head": completion_checkpoint(core, head), "head_sha256": _head_hash(head),
                             "initial_head_sha256": initial_sha, "draws_sha256": _draw_hash(draws, updates),
                             "final_recovery_sha256": sha256_file(checkpoints / f"step_{updates:06d}.pt")}
            final_path = directory / "final_completion_head.pt"
            if final_path.exists():
                existing = torch.load(final_path, map_location="cpu", weights_only=True)
                if not state_dict_equal(existing, final_payload):
                    raise ValueError("existing final completion head differs")
            else:
                write_torch(final_path, final_payload)
            evaluation = {"train_recorded_observations": evaluate_cache(head, train_arrays),
                          "validation_recorded_observations": evaluate_cache(head, validation_arrays),
                          "scope": "recorded observations only; not closed-loop or physical success"}
            report = {"schema": "raccoon-robomimic-completion-report.v1", "seed": seed,
                      "updates_completed": updates, "batch_size": batch_size, "learning_rate": learning_rate,
                      "sampling": "uniform original unit, uniform ID variant, uniform supervised observation, with replacement",
                      "class_weighting": "natural unweighted cross entropy", "prediction": "argmax",
                      "selection": "fixed final update only", "exposure": stats, "evaluation": evaluation,
                      "row_exposure_sha256": sha256_file(row_path), "final_head_sha256": sha256_file(final_path),
                      "core_weights_sha256": core.weights_sha256,
                      "core_checkpoint_sha256": core.checkpoint_file_sha256,
                      "completion_config_sha256": config_sha256, "actual_hardware": False,
                      "new_physics_rollouts": 0, "final_test_used": False}
            _same_json_or_write(directory / "final_manifest.json", report)
            log.write(json.dumps({"event": "final_published", "updates": updates}) + "\n")
        except BaseException as exc:
            log.write(json.dumps({"event": "attempt_error", "error": repr(exc)}) + "\n")
            raise
    core.assert_frozen()
    return report


def load_final_completion_head(directory, *, core, seed, config_sha256,
                               train_cache_dir, validation_cache_dir, expected_updates=5000):
    """Restore a published last head only after verifying its complete lineage."""
    directory = Path(directory)
    report = read_json(directory / "final_manifest.json")
    final_path = directory / "final_completion_head.pt"
    if (report.get("schema") != "raccoon-robomimic-completion-report.v1"
            or report.get("seed") != seed or report.get("updates_completed") != expected_updates
            or report.get("selection") != "fixed final update only"
            or report.get("core_weights_sha256") != core.weights_sha256
            or report.get("core_checkpoint_sha256") != core.checkpoint_file_sha256
            or report.get("completion_config_sha256") != config_sha256
            or report.get("final_head_sha256") != sha256_file(final_path)
            or report.get("row_exposure_sha256") != sha256_file(directory / "row_exposure.npy")):
        raise ValueError("final completion manifest or binding differs")
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    identity = final.get("identity", {})
    if (final.get("schema") != FINAL_SCHEMA or final.get("updates_completed") != expected_updates
            or identity.get("seed") != seed or identity.get("config_sha256") != config_sha256
            or identity.get("updates") != expected_updates
            or identity.get("core_weights_sha256") != core.weights_sha256
            or identity.get("core_checkpoint_sha256") != core.checkpoint_file_sha256):
        raise ValueError("final completion checkpoint identity differs")
    for split, path in (("train", train_cache_dir), ("validation", validation_cache_dir)):
        cache_info = read_json(Path(path) / "manifest.json")
        _ensure_cache_binding(cache_info, core, seed, split)
        if (cache_info["identity"]["source_identity"] != identity["source_identity"]
                or cache_info["files"] != identity[f"{split}_cache_files"]):
            raise ValueError("final completion cache lineage differs")
        # Verify contents, then close mappings: the inference gate only needs the head.
        _, arrays, _ = load_cache(path, identity=cache_info["identity"])
        for array in arrays.values():
            array._mmap.close()
    recovery_path = directory / "checkpoints" / f"step_{expected_updates:06d}.pt"
    if sha256_file(recovery_path) != final.get("final_recovery_sha256"):
        raise ValueError("final completion recovery fingerprint differs")
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
    if (recovery.get("schema") != RECOVERY_SCHEMA or recovery.get("identity") != identity
            or recovery.get("updates_completed") != expected_updates
            or recovery.get("head_sha256") != final.get("head_sha256")
            or recovery.get("draws_sha256") != final.get("draws_sha256")
            or not state_dict_equal(recovery.get("completion_head"), final.get("head"))):
        raise ValueError("published completion head differs from final recovery state")
    draws = np.load(directory / "draws.npy", mmap_mode="r", allow_pickle=False)
    if (draws.dtype != DRAW_DTYPE or draws.shape != (expected_updates, identity["batch_size"])
            or _draw_hash(draws, expected_updates) != final["draws_sha256"]):
        raise ValueError("published completion sample history differs")
    draws._mmap.close()
    head = restore_completion_head(core, final["head"])
    if _head_hash(head) != final["head_sha256"]:
        raise ValueError("restored completion weights differ")
    optimizer = make_head_optimizer(core, head, learning_rate=identity["learning_rate"])
    optimizer.load_state_dict(recovery["optimizer"])
    _validate_optimizer(optimizer, head, expected_updates, identity["learning_rate"])
    core.assert_frozen()
    head.eval()
    return head, {"completion_manifest_sha256": sha256_file(directory / "final_manifest.json"),
                  "completion_checkpoint_sha256": report["final_head_sha256"],
                  "action_core_checkpoint_sha256": core.checkpoint_file_sha256,
                  "action_core_weights_sha256": core.weights_sha256, "updates": expected_updates}
