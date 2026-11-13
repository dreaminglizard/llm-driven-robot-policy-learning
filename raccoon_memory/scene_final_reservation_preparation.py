"""Prepare, but do not execute, a future final evaluation with explicit seeds.

This module performs no file I/O, creates no scenes on import, and has no CLI.
Callers supply byte snapshots and independently reviewed evidence.  Only after
the candidate, sources, callbacks and reviews pass validation is the deferred
reservation reader called.  The reservation's actual schema is intentionally
not guessed: its fixed, fingerprinted adapter must extract the ordered list.

The returned unit dictionaries have the shape consumed by the existing
``evaluate_frozen_student_final.live_unit``.  The final session supports exact
seed lists as well as legacy ranges; callers must still bind the actual
candidate, runtime adapter and execution declaration.  Preparation alone does
not establish readiness for a real final run.  In particular, review receipts
are attestations, not independent proof of their assertions or of a complete
dependency inventory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import marshal
from pathlib import PurePath


KINDS = ("move_to_empty", "stack", "unstack", "transfer_between_stacks")
RUNTIME_ROLES = frozenset((
    "student_loader", "student_adapter", "public_control", "observation_callback",
    "joint_guard", "memory_loader", "session_factory", "program_coordinator",
    "simulation_factory", "scorer", "execution_driver", "reservation_adapter", "case_factory",
    "case_validator", "loaded_candidate_verifier",
))
INTEGRATION_CHECKS = frozenset((
    "multiple_training_seeds", "new_development_starts", "continuous_two_transfers",
    "memory_ab", "same_candidate_and_runtime",
))
_PHASES = frozenset(("training", "development", "selection"))


def _sha(raw):
    if type(raw) is not bytes:
        raise ValueError("a byte snapshot is required")
    return hashlib.sha256(raw).hexdigest()


def _encoded(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("declaration is not canonical JSON") from exc


def _json(raw, name):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key in {name}")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (AttributeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    return value


def _sha_string(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _nonempty_names(value, name):
    if (type(value) is not list or not value or any(type(v) is not str or not v for v in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must be a nonempty list of distinct names")
    return value


def _valid_seeds(value, name, *, empty=False):
    if (type(value) is not list or (not empty and not value)
            or any(type(seed) is not int or seed < 0 for seed in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must be a list of distinct nonnegative integers")
    return value


def runtime_code_sha256(function):
    """Fingerprint a loaded Python function; this does not inspect any file.

    Factories must be ordinary Python functions.  Their loaded code and their
    separately declared source snapshot are bound; closure/global state and
    native-library behavior need the caller's independent runtime validation.
    """
    if not inspect.isfunction(function) or function.__closure__:
        raise ValueError("runtime bindings must be Python functions without closures")
    def default_value(value):
        # Preserve types: changing a tuple default into a list can change the
        # runtime despite identical JSON.  Paths are common loader defaults.
        if value is None or type(value) in (bool, int, float, str):
            return {"type": type(value).__name__, "value": value}
        if isinstance(value, PurePath):
            return {"type": type(value).__name__, "value": value.as_posix()}
        if type(value) is bytes:
            return {"type": "bytes", "value": value.hex()}
        if type(value) in (tuple, list):
            return {"type": type(value).__name__, "items": [default_value(item) for item in value]}
        if type(value) is dict and all(type(key) is str for key in value):
            return {"type": "dict", "items": {key: default_value(item) for key, item in value.items()}}
        raise ValueError("runtime function default is not fingerprintable; bind an explicit wrapper")

    defaults = {"positional": default_value(function.__defaults__),
                "keyword": default_value(function.__kwdefaults__)}
    return _sha(marshal.dumps(function.__code__) + b"\0" + _encoded(defaults))


def _function_name(function):
    return f"{function.__module__}.{function.__qualname__}"


def _binding(function, record, sources, role):
    if (type(record) is not dict or set(record) != {"source", "name", "code_sha256"}
            or record["source"] not in sources or not _sha_string(record["code_sha256"])):
        raise ValueError(f"invalid runtime binding for {role}")
    actual_code = runtime_code_sha256(function)
    if _function_name(function) != record["name"] or actual_code != record["code_sha256"]:
        raise ValueError(f"loaded runtime binding differs for {role}")
    # Compare the loaded code's filename without reading it.  Source snapshots
    # have already been verified; no claim is made about unrecorded imports.
    loaded = function.__code__.co_filename.replace("\\", "/").lower()
    declared = record["source"].replace("\\", "/").lower().lstrip("/")
    if not loaded.endswith("/" + declared) and loaded != declared:
        raise ValueError(f"loaded source path differs for {role}")


def _check_candidate(candidate, snapshots, callables):
    if type(candidate) is not dict or candidate.get("format") != "raccoon-final-candidate.v1" or candidate.get("status") != "frozen":
        raise ValueError("a frozen candidate declaration is required")
    if type(snapshots) is not dict or type(callables) is not dict or set(callables) != RUNTIME_ROLES:
        raise ValueError("all fixed runtime function bindings are required")
    for name in ("policy_checkpoint", "memory_checkpoint", "training_evidence", "split_plan",
                 "public_control", "evaluation_protocol"):
        digest = candidate.get(name + "_sha256")
        if not _sha_string(digest) or _sha(snapshots.get(name)) != digest:
            raise ValueError(f"frozen {name} fingerprint differs")
    source_hashes = candidate.get("source_sha256")
    source_bytes = snapshots.get("runtime_sources")
    if (type(source_hashes) is not dict or not source_hashes or type(source_bytes) is not dict
            or set(source_hashes) != set(source_bytes)):
        raise ValueError("complete declared runtime source snapshots are required")
    for name, digest in source_hashes.items():
        if (type(name) is not str or not name or name.startswith(("/", "\\"))
                or any(part in ("", ".", "..") or ":" in part for part in name.replace("\\", "/").split("/"))
                or not _sha_string(digest)
                or _sha(source_bytes[name]) != digest):
            raise ValueError("frozen runtime source fingerprint differs")
    bindings = candidate.get("runtime_bindings")
    if type(bindings) is not dict or set(bindings) != RUNTIME_ROLES:
        raise ValueError("complete loader, control, observation and runtime bindings are required")
    for role in RUNTIME_ROLES:
        _binding(callables[role], bindings[role], source_hashes, role)
    return _sha(_encoded(candidate))


def _check_evidence(candidate, candidate_sha, snapshots, reviews):
    evidence = _json(snapshots["training_evidence"], "training evidence")
    provenance = evidence.get("final_evaluation_provenance")
    if (type(provenance) is not dict or provenance.get("training_sources_complete") is not True
            or provenance.get("checkpoint_sha256") != candidate["policy_checkpoint_sha256"]
            or provenance.get("split_plan_sha256") != candidate["split_plan_sha256"]):
        raise ValueError("training evidence does not bind the frozen candidate and split")
    ids = _nonempty_names(provenance.get("training_source_ids"), "training source IDs")
    sources = provenance.get("training_sources")
    if type(sources) is not dict or set(ids) != set(sources):
        raise ValueError("training source inventory is incomplete")
    groups = set()
    for name in ids:
        source = sources[name]
        if type(source) is not dict or not _sha_string(source.get("sha256")):
            raise ValueError("training source fingerprint is invalid")
        values = source.get("groups")
        if type(values) is dict:
            if not values or any(type(n) is not int or n <= 0 for n in values.values()):
                raise ValueError("training source group counts must be positive")
            values = list(values)
        groups.update(_nonempty_names(values, "training source groups"))

    if type(reviews) is not dict or set(reviews) != {"direct_demonstration", "integration"}:
        raise ValueError("direct demonstration and integration review snapshots are required")
    direct = _json(reviews["direct_demonstration"], "direct demonstration review")
    if (direct.get("format") != "raccoon-direct-demonstration-review.v1"
            or direct.get("status") != "satisfied" or direct.get("candidate_sha256") != candidate_sha
            or direct.get("training_evidence_sha256") != candidate["training_evidence_sha256"]):
        raise ValueError("direct demonstration review is incomplete or belongs to another candidate")
    reviewed = direct.get("reviewed_training_sources")
    if (type(reviewed) is not dict or not reviewed or not set(reviewed).issubset(sources)
            or any(reviewed[name] != sources[name]["sha256"] for name in reviewed)):
        raise ValueError("direct demonstration review does not bind actual training sources")
    integration = _json(reviews["integration"], "integration review")
    checks = integration.get("checks")
    if (integration.get("format") != "raccoon-final-integration-review.v1"
            or integration.get("status") != "passed" or integration.get("candidate_sha256") != candidate_sha
            or type(checks) is not dict or set(checks) != INTEGRATION_CHECKS
            or any(checks[name] is not True for name in INTEGRATION_CHECKS)):
        raise ValueError("candidate-specific integration review is incomplete")

    physical = provenance.get("physical_seed_provenance")
    if (type(physical) is not dict or physical.get("complete") is not True
            or physical.get("policy_checkpoint_sha256") != candidate["policy_checkpoint_sha256"]
            or physical.get("memory_checkpoint_sha256") != candidate["memory_checkpoint_sha256"]
            or physical.get("split_plan_sha256") != candidate["split_plan_sha256"]):
        raise ValueError("complete candidate-bound physical seed provenance is required")
    phases = physical.get("phase_complete")
    if type(phases) is not dict or set(phases) != _PHASES or any(phases[p] is not True for p in _PHASES):
        raise ValueError("seed provenance must cover training, development and selection")
    physical_ids = _nonempty_names(physical.get("source_ids"), "physical seed source IDs")
    physical_sources, raw_sources = physical.get("sources"), snapshots.get("physical_seed_sources")
    if (type(physical_sources) is not dict or set(physical_sources) != set(physical_ids)
            or type(raw_sources) is not dict or set(raw_sources) != set(physical_ids)):
        raise ValueError("all physical seed source snapshots are required")
    covered, used = set(), set()
    for name in physical_ids:
        source = physical_sources[name]
        if type(source) is not dict or not _sha_string(source.get("sha256")) or _sha(raw_sources[name]) != source["sha256"]:
            raise ValueError("physical seed source fingerprint differs")
        source_phases = _nonempty_names(source.get("phases"), "physical seed source phases")
        if not set(source_phases).issubset(_PHASES):
            raise ValueError("physical seed source phase is invalid")
        covered.update(source_phases)
        used.update(_valid_seeds(source.get("seeds"), "previous physical seeds", empty=True))
    if covered != _PHASES:
        raise ValueError("physical seed source inventory lacks a required phase")
    for name in ids:
        source = physical_sources.get(name)
        if (type(source) is not dict or "training" not in source["phases"]
                or source["sha256"] != sources[name]["sha256"]):
            raise ValueError("physical seed inventory does not bind every training source")
    return groups, frozenset(used)


def _check_protocol(candidate, snapshots, groups):
    protocol = _json(snapshots["evaluation_protocol"], "evaluation protocol")
    plan = _json(snapshots["split_plan"], "split plan")
    train, validation, test = (set(_nonempty_names(plan.get(k), k)) for k in
                               ("train_groups", "validation_groups", "test_groups"))
    if train & validation or train & test or validation & test:
        raise ValueError("split groups overlap")
    checkpoint = set(_nonempty_names(protocol.get("checkpoint_train_groups"), "checkpoint groups"))
    allowed = set(_nonempty_names(protocol.get("allowed_execution_groups"), "execution groups"))
    if checkpoint != groups or not checkpoint.issubset(train):
        raise ValueError("checkpoint training groups differ from source evidence or split")
    layer, runtime = protocol.get("evaluation_layer"), protocol.get("runtime")
    if (type(runtime) is not dict or runtime.get("evaluation_layer") != layer
            or runtime.get("physical_assessment_during_session") is not False):
        raise ValueError("evaluation runtime or assessment boundary is incomplete")
    if layer == "structural_holdout":
        if not allowed.issubset(test) or allowed & checkpoint:
            raise ValueError("structural holdout must use unseen test groups")
    elif layer == "physical_holdout":
        if not allowed.issubset(checkpoint) or runtime.get("randomize") is not True:
            raise ValueError("physical holdout must use randomized recorded training groups")
    else:
        raise ValueError("evaluation layer must be explicit")
    count = protocol.get("units_per_kind")
    reserved_count = candidate.get("reserved_seed_count")
    if (type(count) is not int or count < 20 or type(reserved_count) is not int or reserved_count < 20
            or count < reserved_count + 1 or protocol.get("transfers_per_unit") != 3):
        raise ValueError("schedule must cover every reserved seed and the same-seed memory pair")
    acceptance, assessment = protocol.get("acceptance"), protocol.get("assessment")
    if (type(acceptance) is not dict or acceptance.get("minimum_kind_success_rate") != 0.95
            or acceptance.get("minimum_kind_complete_program_rate") != 0.95
            or acceptance.get("non_target_escape_allowed") is not False
            or type(assessment) is not dict or type(assessment.get("stable_seconds")) not in (int, float)
            or assessment.get("stable_seconds") != 1.0
            or type(assessment.get("samples")) is not int or assessment["samples"] != 501):
        raise ValueError("final assessment and acceptance criteria are incomplete")
    return protocol


@dataclass(frozen=True, slots=True)
class PreparedReservation:
    """Immutable in-memory snapshot; it is not a final-run authorization."""
    candidate_sha256: str
    review_sha256: tuple
    reservation_sha256: str
    reserved_seeds: tuple
    units_per_kind: int
    evaluation_layer: str
    allowed_execution_groups: tuple
    case_binding: tuple
    validator_binding: tuple


def prepare_reservation(candidate, snapshots, reviews, runtime_functions, runtime_objects, *,
                        reservation_reader, reservation_adapter):
    """Check all readiness gates before calling the deferred reservation reader.

    All inputs, including external review receipts, are supplied in memory. The
    adapter is the exact frozen function declared in ``runtime_functions`` and
    returns the ordered seed list from the reservation snapshot.  Completeness
    of extracted seed histories and validity of reviews require separate audit.
    """
    candidate_sha = _check_candidate(candidate, snapshots, runtime_functions)
    groups, used = _check_evidence(candidate, candidate_sha, snapshots, reviews)
    protocol = _check_protocol(candidate, snapshots, groups)
    review_hashes = tuple(sorted((k, _sha(v)) for k, v in reviews.items()))
    if not callable(reservation_reader) or reservation_adapter is not runtime_functions["reservation_adapter"]:
        raise ValueError("the frozen reservation adapter and deferred reader are required")
    if (type(runtime_objects) is not dict
            or set(runtime_objects) != {"student", "memory", "public_control", "observation_callback"}
            or runtime_objects["observation_callback"] is not runtime_functions["observation_callback"]):
        raise ValueError("the loaded candidate, control and fixed observation callback are required")
    object_bindings = dict(runtime_objects)

    def check_unchanged():
        if (_check_candidate(candidate, snapshots, runtime_functions) != candidate_sha
                or tuple(sorted((k, _sha(v)) for k, v in reviews.items())) != review_hashes
                or type(runtime_objects) is not dict or set(runtime_objects) != set(object_bindings)
                or any(runtime_objects[k] is not v for k, v in object_bindings.items())
                or runtime_objects["observation_callback"] is not runtime_functions["observation_callback"]
                or reservation_adapter is not runtime_functions["reservation_adapter"]):
            raise ValueError("a frozen input changed during reservation binding")
        later_groups, later_used = _check_evidence(candidate, candidate_sha, snapshots, reviews)
        if later_groups != groups or later_used != used or _check_protocol(candidate, snapshots, groups) != protocol:
            raise ValueError("a reviewed input changed during reservation binding")

    # This is a candidate-specific, independently reviewed function whose code
    # is pinned above.  It must compare actual loaded weights, control settings
    # and callback wiring against these exact snapshots.  We cannot establish the
    # correctness of that future verifier from a receipt or a hash alone.
    if runtime_functions["loaded_candidate_verifier"](runtime_objects, snapshots, candidate) is not True:
        raise ValueError("loaded candidate or runtime verification failed")
    check_unchanged()
    raw = reservation_reader()
    digest = _sha(raw)
    seeds = _valid_seeds(reservation_adapter(raw), "reserved seeds")
    # External callbacks must not be allowed to silently replace a reviewed
    # candidate, a source snapshot, a runtime function or its review receipt.
    check_unchanged()
    if len(seeds) != candidate["reserved_seed_count"]:
        raise ValueError("reserved seed count differs from the frozen declaration")
    if set(seeds) & used:
        raise ValueError("a reserved seed appears in training, development or selection evidence")
    case = candidate["runtime_bindings"]["case_factory"]
    validator = candidate["runtime_bindings"]["case_validator"]
    return PreparedReservation(candidate_sha, review_hashes,
        digest, tuple(seeds), protocol["units_per_kind"], protocol["evaluation_layer"],
        tuple(protocol["allowed_execution_groups"]), (case["name"], case["source"], case["code_sha256"]),
        (validator["name"], validator["source"], validator["code_sha256"]))


def verify_prepared_inputs(prepared, candidate, snapshots, reviews, runtime_functions, runtime_objects, *,
                           reservation_snapshot, reservation_adapter):
    """Recheck the bound bytes and runtime before handing units to a driver.

    The caller must obtain current snapshots from the relevant sources; this
    pure function does not reread files or attest that snapshots are current.
    """
    if type(prepared) is not PreparedReservation:
        raise ValueError("a prepared reservation is required")
    again = prepare_reservation(candidate, snapshots, reviews, runtime_functions, runtime_objects,
        reservation_reader=lambda: reservation_snapshot, reservation_adapter=reservation_adapter)
    if again != prepared:
        raise ValueError("a previously bound final input changed")


def prepare_runner_units(prepared, case_factory, case_validator):
    """Build runner-shaped units only after binding; the factory owns the scenes.

    Each kind receives every reserved seed in its original order, plus cyclic
    repetitions to meet the declared denominator.  The first two stack cases
    use the same seed to support an A/B memory pair without dropping any seed.
    The entire batch is validated before it is returned to an execution driver.
    """
    if type(prepared) is not PreparedReservation:
        raise ValueError("a prepared reservation is required")
    name, source, code_sha = prepared.case_binding
    _binding(case_factory, {"name": name, "source": source, "code_sha256": code_sha}, {source: True}, "case_factory")
    name, source, code_sha = prepared.validator_binding
    _binding(case_validator, {"name": name, "source": source, "code_sha256": code_sha},
             {source: True}, "case_validator")
    seeds, count = prepared.reserved_seeds, prepared.units_per_kind
    if len(seeds) < 20 or count < len(seeds) + 1:
        raise ValueError("invalid prepared seed schedule")
    result, ids = [], set()
    for kind in KINDS:
        schedule = [seeds[i % len(seeds)] for i in range(count)]
        if kind == "stack":
            # Add the pair before the ordered inventory rather than replacing a
            # seed or turning the inventory into an inclusive numeric range.
            schedule = [seeds[0]] + [seeds[i % len(seeds)] for i in range(count - 1)]
        for index, seed in enumerate(schedule):
            unit = case_factory(kind, index, seed)
            if (type(unit) is not dict or unit.get("kind") != kind or type(unit.get("physics_seed")) is not int
                    or unit["physics_seed"] != seed or type(unit.get("unit_id")) is not str or not unit["unit_id"]
                    or unit["unit_id"] in ids or "scene" not in unit
                    or type(unit.get("source_slot")) is not int or type(unit.get("destination_slot")) is not int
                    or unit["source_slot"] not in (1, 2, 3, 4) or unit["destination_slot"] not in (1, 2, 3, 4)
                    or unit["source_slot"] == unit["destination_slot"]):
                raise ValueError("case factory returned an invalid runner unit or changed its seed")
            # Copy through the runner's JSON representation.  A validator may
            # inspect or even mutate its argument without altering returned units.
            unit = _json(_encoded(unit), "case unit")
            case_for_validation = _json(_encoded(unit), "case unit")
            actual_groups = case_validator(case_for_validation)
            if type(actual_groups) is tuple:
                actual_groups = list(actual_groups)
            actual_groups = _nonempty_names(actual_groups, "validated case groups")
            declared_groups = _nonempty_names(unit.get("groups"), "declared case groups")
            if (set(actual_groups) != set(declared_groups)
                    or not set(actual_groups).issubset(prepared.allowed_execution_groups)):
                raise ValueError("case groups differ from the frozen evaluation split")
            ids.add(unit["unit_id"])
            result.append(dict(unit))
        if set(schedule) != set(seeds):
            raise ValueError("a reserved seed is missing from the schedule")
        if kind == "stack":
            first, second = result[-count], result[-count + 1]
            if (first["source_slot"] != second["source_slot"]
                    or first["destination_slot"] == second["destination_slot"]
                    or _encoded(first["scene"]) != _encoded(second["scene"])):
                raise ValueError("memory A/B units must share a scene and source but use different destinations")
    return result
