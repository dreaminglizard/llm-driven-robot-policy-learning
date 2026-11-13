"""Build offline command demonstrations from explicitly referenced saved records.

Manifest v1 has ``base_dir`` (relative to the manifest's directory) and an
``episodes`` list. Each episode specifies ``episode_id``, ``trial``, ``journal``,
``confirmation``, ``registration`` and a ``task`` with source/destination slots
and levels. An optional ``execution`` is the stage file referenced by a wrapper
trial; ``prior_trial`` is the bridge from an indirect confirmation to a trial.

Only pre-command sensor observations are inputs. IDs, stage labels and event
timestamps are retained in provenance; neither symbolic layouts nor action
history are represented as measured observations. The separate execution
field is result/dispatch evidence, never an input or requested-action label.
No controller is imported.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "raccoon-motion-demonstrations-manifest.v1"
REPORT_SCHEMA = "raccoon-motion-export-report.v2"
TASK_FIELDS = ("source_slot", "destination_slot", "source_level", "destination_level")
OBSERVATION_FIELDS = (
    "schema_version", "observed_monotonic", "simulated", "state", "joints_deg",
    "tcp_mm", "sensor_sequence", "sensor_age_seconds", "sensor_valid", "moving",
    # This is the preceding command, not a sensor reading.
    "gripper_command", "gripper_reported_state", "gripper_reported_raw",
    "gripper_module_type", "gripper_feedback_age_seconds", "gripper_reported_valid",
    "gripper_feedback_fresh",
)


class MotionDataError(ValueError):
    """An input cannot safely be represented as a complete demonstration."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> None:
    raise MotionDataError(code, detail)


def _strict_json(data: bytes, label: str) -> Any:
    def reject_constant(value: str) -> None:
        _fail("non_finite", f"{label}: non-finite JSON number {value}")

    try:
        return json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("invalid_json", f"{label}: {exc}")


def _finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non_finite", label)
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite(item, f"{label}[{i}]")


def _dict(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        _fail("invalid_structure", f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid_structure", f"{label} must be a nonempty string")
    return value


def _utc(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_string(value, label).replace("Z", "+00:00"))
    except ValueError:
        _fail("invalid_timestamp", label)
    if parsed.utcoffset() != timedelta(0):
        _fail("invalid_timestamp", f"{label} is not explicitly UTC")
    return parsed


class _Sources:
    """Read immutable byte snapshots and record exactly what was read."""

    def __init__(self) -> None:
        self.cache: dict[Path, bytes] = {}
        self.missing: dict[str, str] = {}

    def read(self, path: Path) -> bytes:
        path = path.resolve()
        if path not in self.cache:
            try:
                self.cache[path] = path.read_bytes()
            except OSError as exc:
                self.missing[str(path)] = str(exc)
                _fail("missing_source", f"{path}: {exc}")
        return self.cache[path]

    def json(self, path: Path) -> dict:
        return _dict(_strict_json(self.read(path), str(path)), str(path))

    def report(self) -> list[dict]:
        return [{"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                 "bytes": len(data)} for path, data in sorted(self.cache.items(), key=lambda x: str(x[0]))]


def _resolve(base: Path, value: Any, label: str) -> Path:
    path = Path(_string(value, label))
    return (path if path.is_absolute() else base / path).resolve()


def _same_ref(base: Path, actual: Any, expected: Path, label: str) -> None:
    if _resolve(base, actual, label) != expected:
        _fail("reference_mismatch", f"{label} does not reference {expected}")


def _has_ref(base: Path, doc: dict, field: str, expected: Path) -> bool:
    return isinstance(doc.get(field), str) and _resolve(base, doc[field], field) == expected


def _verify_recorded_hashes(base: Path, doc: dict, sources: _Sources) -> set[Path]:
    recorded = doc.get("evidence_sha256", {})
    if not isinstance(recorded, dict):
        _fail("invalid_confirmation", "evidence_sha256 must be an object")
    matched = set()
    for name, digest in recorded.items():
        path = _resolve(base, name, "evidence_sha256 path")
        # Hash the specified evidence if it is one of the files this extraction
        # actually consumes. Unrelated recorded evidence remains a reference.
        if path in sources.cache:
            if not isinstance(digest, str) or hashlib.sha256(sources.read(path)).hexdigest() != digest.lower():
                _fail("evidence_hash_mismatch", f"recorded evidence hash differs: {path}")
            matched.add(path)
    return matched


def _confirmation_and_execution(entry: dict, base: Path, sources: _Sources) -> tuple[dict, dict, dict, dict, dict]:
    trial_path = _resolve(base, entry.get("trial"), "trial")
    confirmation_path = _resolve(base, entry.get("confirmation"), "confirmation")
    registration_path = _resolve(base, entry.get("registration"), "registration")
    trial = sources.json(trial_path)
    confirmation = sources.json(confirmation_path)
    registration = sources.json(registration_path)
    bridge = None
    execution_path = trial_path
    if entry.get("execution") is not None:
        execution_path = _resolve(base, entry["execution"], "execution")
        _same_ref(base, _dict(trial.get("result"), "trial.result").get("execution_reference"),
                  execution_path, "trial.result.execution_reference")
        execution = sources.json(execution_path)
    elif "execution" in trial and isinstance(trial["execution"], dict):
        execution = trial["execution"]
    else:
        execution = trial

    if entry.get("prior_trial") is not None:
        bridge_path = _resolve(base, entry["prior_trial"], "prior_trial")
        bridge = sources.json(bridge_path)
        _same_ref(base, confirmation.get("prior_trial"), bridge_path, "confirmation.prior_trial")
        candidates = bridge.get("trials")
        if not isinstance(candidates, list):
            _fail("reference_mismatch", "prior_trial.trials is missing")
        matching = [t for t in candidates if isinstance(t, dict)
                    and isinstance(t.get("result"), dict)
                    and _has_ref(base, t["result"], "execution_reference", execution_path)]
        if len(matching) != 1 or matching[0]["result"].get("event") != "completed":
            _fail("reference_mismatch", "prior_trial does not identify one completed target execution")
        if confirmation.get("transfer_completion_user_confirmed") is not True:
            _fail("unconfirmed_success", "confirmation does not explicitly confirm transfer completion")
        _string(confirmation.get("user_statement"), "confirmation.user_statement")
        task = _dict(entry.get("task"), "task")
        if confirmation.get("confirmed_transfer") != f"{task.get('source_slot')}_to_{task.get('destination_slot')}":
            _fail("reference_mismatch", "confirmation direction differs from task")
    else:
        if confirmation.get("physical_success") is not True:
            _fail("unconfirmed_success", "physical success is not explicitly user-confirmed")
        if confirmation.get("source") != "user_message" or not isinstance(confirmation.get("user_message"), str) or not confirmation["user_message"].strip():
            _fail("unconfirmed_success", "confirmation does not identify a user message")
        _same_ref(base, confirmation.get("trial_reference"), trial_path, "confirmation.trial_reference")
        if execution_path != trial_path:
            _same_ref(base, confirmation.get("execution_reference"), execution_path,
                      "confirmation.execution_reference")

    matches = _verify_recorded_hashes(base, confirmation, sources)
    registration_linked = (
        _has_ref(base, confirmation, "registration_reference", registration_path)
        or _has_ref(base, trial, "registration_reference", registration_path)
        or (bridge is not None and _has_ref(base, bridge, "registration_reference", registration_path))
        or registration_path in matches
    )
    if not registration_linked:
        _fail("reference_mismatch", "registered layout is not linked to trial or confirmation")
    if execution.get("mode") != "hardware_expected" or execution.get("status") != "command_sequence_completed":
        _fail("failed_execution", "execution did not complete its hardware command sequence")
    quality = {
        "action_source": "TransferRunner_saved_commands",
        "physical_transfer_success": "user_confirmed",
        "symbolic_task_source": "registered_layout",
        "coordinate_source": execution.get("coordinate_source"),
        "physical_calibration_verified": execution.get("physical_calibration_verified"),
        "units": {
            "joints_deg": "degrees",
            "tcp_mm": "nominal millimetres (computed from joint state)",
            "gripper_reported_state": "discrete sensor-reported state",
            "gripper_reported_raw": "device-reported code",
            "gripper_command": "discrete command state",
        },
        "device_verified_object_success": confirmation.get("device_verified_object_success"),
        "placement_offset_reported": confirmation.get("placement_offset_reported"),
        "precise_placement_validated": confirmation.get("precise_placement_validated"),
        "precision_training_positive": confirmation.get("precision_training_positive", False),
        "segment_annotation": confirmation.get("segment_annotation"),
        "latest_confirmation_reference": str(confirmation_path),
        "physical_roundtrip_success": confirmation.get("physical_roundtrip_success"),
    }
    return trial, execution, confirmation, registration, quality


def _validate_task(entry: dict, execution: dict, registration: dict) -> dict:
    task = _dict(entry.get("task"), "task")
    if set(task) != set(TASK_FIELDS) or any(type(task[k]) is not int or task[k] < 1 for k in TASK_FIELDS):
        _fail("invalid_task", "task must contain only four positive integer slot/level fields")
    counts = registration.get("registered_counts", registration.get("counts"))
    compiled = _dict(execution.get("compiled"), "execution.compiled")
    initial = compiled.get("initial")
    final = compiled.get("expected_final")
    if (not isinstance(counts, list) or not isinstance(initial, list) or len(counts) != len(initial)
            or any(type(v) is not int or v < 0 for v in counts)
            or any(not isinstance(slot, list) for slot in initial)
            or [len(slot) for slot in initial] != counts):
        _fail("registration_mismatch", "registered counts differ from symbolic execution start")
    source, destination = task["source_slot"] - 1, task["destination_slot"] - 1
    if source >= len(counts) or destination >= len(counts) or source == destination:
        _fail("invalid_task", "invalid source or destination slot")
    if task["source_level"] != counts[source] or task["destination_level"] != counts[destination] + 1:
        _fail("registration_mismatch", "task levels differ from registered counts")
    expected = [list(slot) for slot in initial]
    if not expected[source]:
        _fail("registration_mismatch", "source slot was empty")
    expected[destination].append(expected[source].pop())
    if final != expected:
        _fail("registration_mismatch", "compiled symbolic result is not this one transfer")
    return {key: task[key] for key in TASK_FIELDS}


def _journal(path: Path, sources: _Sources) -> dict[str, list[tuple[int, dict]]]:
    data = sources.read(path)
    if data and not data.endswith(b"\n"):
        _fail("truncated_journal", f"{path}: missing final newline")
    commands: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = _dict(_strict_json(line, f"{path}:{number}"), f"{path}:{number}")
        except MotionDataError as exc:
            _fail("truncated_journal" if exc.code == "invalid_json" else exc.code, exc.detail)
        command = event.get("command_id")
        if isinstance(command, str) and event.get("event") in ("command_admitted", "command_dispatched", "command_result"):
            commands[command].append((number, event))
    return commands


def _observation(value: Any, label: str) -> dict:
    obs = _dict(value, label)
    for field in ("observation_id", "instance_id"):
        _string(obs.get(field), f"{label}.{field}")
    if obs.get("simulated") is not False or obs.get("sensor_valid") is not True or obs.get("moving") is not False:
        _fail("invalid_observation", f"{label} is simulated, invalid, moving, or missing a flag")
    for field, length in (("joints_deg", 4), ("tcp_mm", 3)):
        vector = obs.get(field)
        if not isinstance(vector, list) or len(vector) != length or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            _fail("invalid_observation", f"{label}.{field} is not a finite {length}-vector")
    for field in ("observed_monotonic", "sensor_age_seconds"):
        value = obs.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            _fail("invalid_observation", f"{label}.{field} must be finite and nonnegative")
    for field in ("revision", "sensor_sequence"):
        if type(obs.get(field)) is not int or obs[field] < 0:
            _fail("invalid_observation", f"{label}.{field} must be a nonnegative integer")
    for field in ("gripper_reported_valid", "gripper_feedback_fresh"):
        if type(obs.get(field)) is not bool:
            _fail("invalid_observation", f"{label}.{field} must be boolean")
    _finite(obs, label)
    # Exclude recent_actions, references, perception and symbolic state to avoid
    # leaking command history or representing inferred object state as a sensor.
    return {key: obs[key] for key in OBSERVATION_FIELDS if key in obs}


def _transitions(entry: dict, execution: dict, journal: dict, used_commands: set[str]) -> tuple[list[dict], set[str]]:
    stages = execution.get("stages")
    operations = _dict(execution.get("compiled"), "compiled").get("operations")
    if not isinstance(stages, list) or not stages or not isinstance(operations, list) or len(operations) != len(stages):
        _fail("incomplete_execution", "stage and compiled operation counts do not match")
    ids = []
    for stage in stages:
        outcome = _dict(_dict(stage, "stage").get("outcome"), "stage.outcome")
        ids.append(_string(outcome.get("command_id"), "stage.outcome.command_id"))
    if len(set(ids)) != len(ids) or set(ids) & used_commands:
        _fail("duplicate_command", "a stage command is repeated in this or another episode")
    try:
        prefix, _ = ids[0].rsplit("-", 1)
    except ValueError:
        _fail("invalid_command_id", "command IDs do not carry ordered stage indices")
    if not prefix or ids != [f"{prefix}-{i}" for i in range(len(ids))]:
        _fail("invalid_command_id", "command ID sequence is incomplete or out of order")
    extras = [key for key in journal if key.startswith(prefix + "-") and key not in set(ids)]
    if extras:
        _fail("incomplete_execution", f"journal contains commands absent from stages: {extras[:3]}")

    transitions = []
    seen_observations: set[str] = set()
    instance = None
    previous_line = 0
    previous_end = None
    previous_utc = None
    previous_revision = -1
    for index, (stage, operation, command_id) in enumerate(zip(stages, operations, ids)):
        operation = _dict(operation, f"compiled.operations[{index}]")
        for key in ("transfer", "stage", "segment", "segments", "transport_phase"):
            if stage.get(key) != operation.get(key):
                _fail("stage_mismatch", f"stage {index}: {key} differs from compiled operation")
        events = journal.get(command_id, [])
        admissions = [(line, event) for line, event in events if event["event"] == "command_admitted"]
        dispatches = [(line, event) for line, event in events if event["event"] == "command_dispatched"]
        results = [(line, event) for line, event in events if event["event"] == "command_result"]
        if len(admissions) != 1 or len(results) != 1 or len(dispatches) > 1:
            code = "duplicate_command" if len(admissions) > 1 or len(results) > 1 or len(dispatches) > 1 else "missing_command_event"
            _fail(code, f"{command_id}: found {len(admissions)} admissions, {len(dispatches)} dispatches and {len(results)} results")
        admit_line, admitted = admissions[0]
        result_line, result_event = results[0]
        if not previous_line < admit_line < result_line:
            _fail("event_order", f"{command_id}: journal order is inconsistent")
        admit_time = _utc(admitted.get("timestamp_utc"), "command_admitted.timestamp_utc")
        result_time = _utc(result_event.get("timestamp_utc"), "command_result.timestamp_utc")
        if result_time < admit_time or (previous_utc is not None and admit_time < previous_utc):
            _fail("event_order", f"{command_id}: UTC event times are inconsistent")
        if admitted.get("simulated") is not False or result_event.get("simulated") is not False:
            _fail("simulated", f"{command_id}: journal event is not real hardware")
        request = _dict(admitted.get("request"), "request")
        raw_obs = _dict(admitted.get("input_observation"), "input_observation")
        result = _dict(result_event.get("result"), "result")
        raw_next = _dict(result.get("observation"), "result.observation")
        if request.get("command_id") != command_id or result.get("command_id") != command_id:
            _fail("invalid_command_id", f"{command_id}: nested command ID differs")
        if request.get("observation_id") != raw_obs.get("observation_id"):
            _fail("observation_id_mismatch", f"{command_id}: admitted observation ID differs")
        if result != stage["outcome"]:
            _fail("stage_mismatch", f"{command_id}: journal result differs from saved stage outcome")
        action = _dict(request.get("action"), "request.action")
        if action != operation.get("action"):
            _fail("stage_mismatch", f"{command_id}: admitted action differs from compiled action")
        _finite(action, "action")
        kind, status, applied = action.get("kind"), result.get("status"), result.get("applied")
        if result.get("simulated") is not False or result.get("replayed") is not False:
            _fail("failed_command", f"{command_id}: result is simulated or replayed")
        effective_target = None
        steps = None
        if kind == "move_joints":
            target = action.get("joints_deg")
            if not isinstance(target, list) or len(target) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in target):
                _fail("invalid_action", f"{command_id}: invalid joint target")
            if not ((status == "completed" and applied is True) or (status == "already_at_target" and applied is False)):
                _fail("failed_command", f"{command_id}: invalid motion result {status}/{applied}")
            result_target = _dict(result.get("target"), "result.target")
            effective_target, steps = result_target.get("effective_target_deg"), result_target.get("steps")
            if (result_target.get("kind") != "move_joints" or result_target.get("requested_joints_deg") != target
                    or not isinstance(effective_target, list) or len(effective_target) != 4
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in effective_target)
                    or not isinstance(steps, list) or len(steps) != 4 or any(type(x) is not int for x in steps)):
                _fail("target_mismatch", f"{command_id}: requested or effective joint target is invalid")
            if applied is True:
                receipt = _dict(result.get("receipt"), "result.receipt")
                if receipt.get("effective_target_deg") != effective_target or receipt.get("steps") != steps:
                    _fail("target_mismatch", f"{command_id}: result receipt differs from target")
        elif kind == "set_gripper":
            if action.get("state") not in ("open", "closed"):
                _fail("invalid_action", f"{command_id}: invalid gripper command")
            if (status != "gripper_reported" or applied is not True
                    or result.get("gripper_reported_confirmed") is not True
                    or result.get("gripper_command") != action["state"]):
                _fail("failed_command", f"{command_id}: gripper command did not report completion")
        else:
            _fail("invalid_action", f"{command_id}: unsupported action {kind}")
        if applied is True and len(dispatches) != 1:
            _fail("missing_dispatch", f"{command_id}: applied result has no dispatch event")
        if applied is False and dispatches:
            _fail("unexpected_dispatch", f"{command_id}: non-applied result has a dispatch event")
        dispatch = {"sent": False, "event": None, "journal_line": None, "timestamp_utc": None,
                    "command_id": command_id, "evidence": "no_dispatch_in_journal_snapshot"}
        if dispatches:
            dispatch_line, dispatched = dispatches[0]
            dispatch_time = _utc(dispatched.get("timestamp_utc"), "command_dispatched.timestamp_utc")
            if (not admit_line < dispatch_line < result_line or not admit_time <= dispatch_time <= result_time
                    or dispatched.get("simulated") is not False):
                _fail("event_order", f"{command_id}: dispatch is out of order or simulated")
            if kind == "move_joints":
                if dispatched.get("target") != result_target or dispatched.get("receipt") != result.get("receipt"):
                    _fail("target_mismatch", f"{command_id}: dispatched target/receipt differ from result")
            elif dispatched.get("gripper_command") != action["state"]:
                _fail("target_mismatch", f"{command_id}: dispatched gripper command differs")
            dispatch = {"sent": True, "event": dispatched["event"], "journal_line": dispatch_line,
                        "timestamp_utc": dispatched["timestamp_utc"], "command_id": command_id,
                        "evidence": "journal_event"}
        before = _observation(raw_obs, "input_observation")
        after = _observation(raw_next, "result.observation")
        before_id, after_id = raw_obs["observation_id"], raw_next["observation_id"]
        if before_id == after_id or before_id in seen_observations or after_id in seen_observations:
            _fail("duplicate_observation", f"{command_id}: observation IDs are not unique")
        seen_observations.update((before_id, after_id))
        if instance is None:
            instance = raw_obs["instance_id"]
        if raw_obs["instance_id"] != instance or raw_next["instance_id"] != instance:
            _fail("instance_mismatch", f"{command_id}: observations have different instances")
        if not (raw_obs["observed_monotonic"] < raw_next["observed_monotonic"]
                and raw_obs["revision"] < raw_next["revision"]
                and raw_obs["revision"] >= previous_revision
                and (previous_end is None or raw_obs["observed_monotonic"] >= previous_end)):
            _fail("observation_order", f"{command_id}: observations are not ordered before and after the action")
        provenance = {
            "command_id": command_id, "instance_id": instance,
            "input_observation_id": before_id, "result_observation_id": after_id,
            "stage_index": index, "stage": stage.get("stage"), "segment": stage.get("segment"),
            "admitted_journal_line": admit_line, "result_journal_line": result_line,
            "admitted_at_utc": admitted["timestamp_utc"], "result_at_utc": result_event["timestamp_utc"],
        }
        execution = {"effective_target_deg": effective_target, "steps": steps,
                     "target_source": "command_result.target" if kind == "move_joints" else None,
                     "dispatch": dispatch}
        transitions.append({"observation": before, "action": action, "next_observation": after,
                            "applied": applied, "status": status, "execution": execution,
                            "provenance": provenance})
        previous_line, previous_end, previous_revision, previous_utc = result_line, raw_next["observed_monotonic"], raw_next["revision"], result_time
    return transitions, set(ids)


def extract_manifest(manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Return complete episodes and an auditable report without writing files."""
    path = Path(manifest_path).resolve()
    sources = _Sources()
    manifest = sources.json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        _fail("invalid_manifest", f"manifest.schema must be {MANIFEST_SCHEMA}")
    base = _resolve(path.parent, manifest.get("base_dir", "."), "base_dir")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        _fail("invalid_manifest", "manifest.episodes must be a nonempty list")
    # Capture every explicitly named source, including those belonging to an
    # excluded episode, so the report identifies the exact snapshot inspected.
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in ("trial", "journal", "confirmation", "registration", "execution", "prior_trial"):
            if field in entry:
                try:
                    sources.read(_resolve(base, entry[field], field))
                except MotionDataError:
                    pass  # The per-episode pass records the corresponding exclusion.
    episodes = []
    excluded = []
    used_commands: set[str] = set()
    episode_ids: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        episode_id = entry.get("episode_id")
        try:
            _string(episode_id, "episode_id")
            if episode_id in episode_ids:
                _fail("duplicate_episode", f"episode_id repeated: {episode_id}")
            episode_ids.add(episode_id)
            _, execution, _, registration, quality = _confirmation_and_execution(entry, base, sources)
            task = _validate_task(entry, execution, registration)
            journal_path = _resolve(base, entry.get("journal"), "journal")
            transitions, commands = _transitions(entry, execution, _journal(journal_path, sources), used_commands)
            used_commands.update(commands)
            episodes.append({"episode_id": episode_id, "task": task, "quality": quality, "transitions": transitions})
        except MotionDataError as exc:
            excluded.append({"index": index, "episode_id": episode_id, "reason": exc.code, "detail": exc.detail})
    status_counts = Counter(transition["status"] for episode in episodes for transition in episode["transitions"])
    dispatch_counts = Counter("sent" if transition["execution"]["dispatch"]["sent"] else "not_sent"
                              for episode in episodes for transition in episode["transitions"])
    report = {"schema": REPORT_SCHEMA, "manifest": str(path), "sources": sources.report(),
              "missing_sources": [{"path": k, "detail": v} for k, v in sorted(sources.missing.items())],
              "excluded_episodes": excluded, "independent_episode_count": len(episodes),
              "transition_count": sum(len(e["transitions"]) for e in episodes),
              "status_counts": dict(sorted(status_counts.items())),
              "dispatch_counts": dict(sorted(dispatch_counts.items())),
              "note": "Full episodes only; event UTC and observation monotonic time are distinct clocks."}
    return episodes, report


def export_manifest(manifest_path: str | Path, output: str | Path) -> dict:
    """Write episodes.jsonl and report.json to a directory that does not exist."""
    requested = Path(output)
    if requested.is_symlink():
        raise FileExistsError(f"output path is a symlink: {requested}")
    target = requested.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    code_paths = [Path(__file__).resolve(), Path(__file__).resolve().parent.parent / "scripts" / "export_motion_demonstrations.py"]
    code_hashes = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in code_paths]
    episodes, report = extract_manifest(manifest_path)
    target.mkdir(parents=False, exist_ok=False)
    episode_path = target / "episodes.jsonl"
    with episode_path.open("xb") as stream:
        for episode in episodes:
            stream.write((json.dumps(episode, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))
    episode_bytes = episode_path.read_bytes()
    report["extractor_code"] = code_hashes
    report["artifacts"] = {"episodes_jsonl": {"path": "episodes.jsonl", "sha256": hashlib.sha256(episode_bytes).hexdigest(),
                                              "bytes": len(episode_bytes)}}
    with (target / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return report
