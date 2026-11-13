"""Explicit first-generation holdout with an incomplete historical seed ledger.

This is a different claim from the existing complete-history certification.
No reservation is opened and no simulator is constructed by this module.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import math
import re
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
import secrets
import os
import tempfile

import numpy as np


MODE = "prospective_sealed_v2"
SPECIFICATION_SCHEMA = "raccoon-prospective-final-specification.v1"
COMMITMENT_IDS = ("@specification", "@reservation", "@consumption")
PHASES = frozenset(("training", "development", "selection"))
RESERVATION_V3_FORMAT = "raccoon-independent-final-holdout-reservation.v3"


class ProspectiveProtocolError(ValueError):
    """A frozen-boundary violation, distinct from student performance failure."""


def verify_zero_world_predecessor(plan, *, reader=None):
    """Bind the original schema rejection and the unchanged scientific plan."""
    prior = plan.get("prior_attempt")
    if (type(prior) is not dict or prior.get("worlds_created") != 0
            or prior.get("student_decisions") != 0 or prior.get("physical_scores") != 0
            or prior.get("closed_not_retried") is not True):
        raise ValueError("a disclosed zero-world predecessor is required")
    snapshots = {}
    for key in ("status_record", "specification"):
        ref = prior[key]
        raw = reader(ref["path"]) if reader is not None else Path(ref["path"]).read_bytes()
        if sha(raw) != ref["sha256"]:
            raise ValueError("an original predecessor record changed")
        snapshots[key] = document(raw)
    status, original = snapshots["status_record"], snapshots["specification"]
    if (status.get("status") != "protocol_invalid_or_infrastructure_error"
            or status.get("error") != {"type": "ValueError", "message": "sealed JSON must contain exactly one candidate integer seed array"}
            or status.get("retry_or_reservation_replacement_allowed") is not False
            or status.get("specification_sha256") != prior["specification"]["sha256"]
            or (Path(prior["status_record"]["path"]).parent / "evaluation").exists()):
        raise ValueError("the predecessor is not the bound zero-world metadata rejection")
    for key in ("policy_checkpoint_sha256", "memory_checkpoint_sha256", "split_plan_sha256",
                "evaluation_protocol", "training_evidence", "known_physical_history", "known_initial_state_keys"):
        if encoded(plan.get(key)) != encoded(original.get(key)):
            raise ValueError("the followup changed a frozen candidate, scientific rule or known-history scope")
    permitted_changes = {"raccoon_memory/scene_prospective_evaluation.py", "scripts/evaluate_prospective_frozen_student.py"}
    if any(plan["source_sha256"].get(name) != digest for name, digest in original["source_sha256"].items()
           if name not in permitted_changes):
        raise ValueError("the followup changed frozen policy, control or physical assessment code")
    if plan.get("campaign_stop_after_this_followup") is not True:
        raise ValueError("the followup campaign must have a fixed stopping rule")
    return prior


def reserve_followup_v3(path, *, study_plan_path):
    """Explicitly reserve a whole new study after a zero-world schema failure.

    The predecessor remains invalid and consumed. This never retries it, reads
    its reserved values, creates a world, or consults any performance result.
    Returned metadata omits all twenty new seed values.
    """
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    plan_raw = Path(study_plan_path).read_bytes()
    plan = document(plan_raw)
    if plan.get("status") != "preregistered_before_reservation":
        raise ValueError("the study plan must precede new seed generation")
    verify_v3_descriptor(plan["reservation"])
    prior_attempt = verify_zero_world_predecessor(plan)
    study_plan_sha256 = sha(plan_raw)
    policy_checkpoint_sha256 = plan["policy_checkpoint_sha256"]
    source = Path(__file__).resolve()
    metadata = {"format": RESERVATION_V3_FORMAT, "status": "reserved_not_generated_not_executed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "physics_seeds": secrets.SystemRandom().sample(range(2**40, 2**48), 20),
        "producer": {"source": "raccoon_memory/scene_prospective_evaluation.py", "sha256": sha(source.read_bytes()),
            "method": "secrets.SystemRandom.sample without replacement", "integer_domain_half_open": [2**40, 2**48],
            "performance_or_world_consulted": False, "seed_values_printed": False},
        "unchanged_policy_checkpoint_sha256": policy_checkpoint_sha256, "prior_attempt": prior_attempt,
        "study_plan_sha256": study_plan_sha256,
        "scope": "new preregistered prospective study; no repair, continuation or successful relabelling of the predecessor"}
    raw = encoded(metadata) + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".reservation-v3-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if sha(temporary.read_bytes()) != sha(raw):
            raise OSError("reservation temporary file was incomplete")
        os.link(temporary, path)  # Atomic publication; fails if target already exists.
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {"path": str(Path(path).resolve()), "sha256": sha(raw), "count": 20,
            "format": RESERVATION_V3_FORMAT, "worlds_created": 0, "seed_values_printed": False}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(raw):
    if type(raw) is not bytes:
        raise ValueError("a byte snapshot is required")
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("a JSON object is required")
    return value


@lru_cache(maxsize=4096)
def effective_randomization_key(seed):
    """Mirror SceneMotionSim's actual NumPy initializer, without a world.

    Full Python integer entropy enters SeedSequence. Four child generators
    supply mass/friction/servo/joints; the fifth supplies the uint64 reset key.
    No raw integer truncation or float conversion is used here or by the sim.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("a nonnegative integer physics seed is required")
    streams = np.random.SeedSequence(seed).spawn(5)
    reset = int(streams[4].generate_state(1, dtype=np.uint64)[0])
    return sha(encoded({"primary_states": [np.random.default_rng(s).bit_generator.state for s in streams[:4]],
                        "slot_state": np.random.default_rng(reset).bit_generator.state}))


def physical_snapshot_key(snapshot):
    """Exact physical-state key, excluding seed, paths, labels and source code.

    Comparison is across snapshots with the same XML/model representation;
    it is not a geometric-nearness or cross-simulator equivalence test.
    """
    keys = ("generated_model_xml_sha256", "model_parameters", "active_object_ids", "state_signature", "state")
    if type(snapshot) is not dict or any(key not in snapshot for key in keys):
        raise ValueError("an actual complete physical snapshot is required")
    def normalized(value):
        if type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError("nonfinite physical snapshot")
            return 0.0 if value == 0 else float(value)
        if type(value) is dict:
            return {key: normalized(item) for key, item in value.items()}
        if type(value) in (list, tuple):
            return [normalized(item) for item in value]
        if type(value) is str:
            return value
        raise ValueError("unsupported physical snapshot value")
    return sha(encoded({key: normalized(snapshot[key]) for key in keys}))


def recorded_initial_keys(raw, format):
    """Recompute comparable keys from explicitly bound development sources."""
    if format == "jsonl_first_stage_before.v1":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif format == "json_records_first_stage_before.v1":
        records = document(raw)["records"]
    else:
        raise ValueError("unsupported recorded physical state source format")
    states = []
    for record in records:
        stages = record.get("stages", [])
        if stages and "before_state" in stages[0]:
            states.append(stages[0]["before_state"])
    if not states:
        raise ValueError("recorded source contains no comparable initial states")
    return {physical_snapshot_key(state) for state in states}


@lru_cache(maxsize=64)
def recorded_seed_exclusions(raw, format):
    """Extract only explicit physics fields or labelled conservative exclusions."""
    if format == "json_direct_case_id_exclusions.v1":
        episodes = document(raw)["teacher_source"]["episodes"]
        return tuple(sorted({int(match.group(1)) for episode in episodes
            if (match := re.search(r"(?:^|-)p(\d+)$", episode["case_id"]))}))
    found = set()
    def visit(value):
        if type(value) is dict:
            seed = value.get("physics_seed")
            if type(seed) is int and seed >= 0:
                found.add(seed)
            reset = value.get("reset_and_randomization")
            if type(reset) is dict and type(reset.get("seed")) is int and reset["seed"] >= 0:
                found.add(reset["seed"])
            for key, child in value.items():
                if key not in ("decisions", "trace", "features", "samples", "state", "windows", "targets"):
                    visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)
    if format == "gzip_jsonl_physics_fields.v1":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            for line in stream:
                if line.strip():
                    visit(json.loads(line))
    elif format == "jsonl_physics_fields.v1":
        for line in raw.splitlines():
            if line.strip():
                visit(json.loads(line))
    elif format == "json_physics_fields.v1":
        visit(document(raw))
    else:
        raise ValueError("unsupported known physical-seed extraction method")
    return tuple(sorted(found))


def decode_reservation(raw, descriptor):
    """Apply the exact preopening adapter; never guess a missing field name.

    If producer field names are unavailable, a typed adapter requires exactly
    one 20-integer array anywhere in the sealed JSON. Ambiguity is an error,
    not an opportunity to choose or replace a reservation after opening.
    """
    value = document(raw)
    if descriptor.get("adapter") == "unique_20_integer_array.v1":
        candidates = []
        def visit(item):
            if type(item) is list:
                if len(item) == 20 and all(type(seed) is int and seed >= 0 for seed in item):
                    candidates.append(item)
                else:
                    for child in item:
                        visit(child)
            elif type(item) is dict:
                for child in item.values():
                    visit(child)
        visit(value)
        if len(candidates) != 1:
            raise ValueError("sealed JSON must contain exactly one candidate integer seed array")
        seeds = candidates[0]
    elif descriptor.get("adapter") == "named_producer_fields.v1":
        schema_key, seeds_key = descriptor["schema_field"], descriptor["seeds_field"]
        if value.get(schema_key) != descriptor["schema_value"]:
            raise ValueError("reserved metadata schema differs from the frozen adapter")
        seeds = value.get(seeds_key)
    else:
        raise ValueError("unsupported frozen reservation adapter")
    if (type(seeds) is not list or len(seeds) != descriptor["count"] or len(seeds) != 20
            or any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds))):
        raise ValueError("reserved seeds differ from the declared count or type")
    if descriptor.get("schema_value") == RESERVATION_V3_FORMAT and any(not 2**40 <= seed < 2**48 for seed in seeds):
        raise ValueError("v3 seeds differ from the implemented producer domain")
    return tuple(seeds)


def verify_v3_descriptor(descriptor):
    if ({key: descriptor.get(key) for key in ("adapter", "schema_field", "schema_value", "seeds_field", "count")} != {
            "adapter": "named_producer_fields.v1", "schema_field": "format", "count": 20,
            "schema_value": RESERVATION_V3_FORMAT, "seeds_field": "physics_seeds"}):
        raise ValueError("v3 requires the exact implemented producer field contract")


def verify_issued_study_plan(spec, plan_raw):
    """The only post-draw changes are issuing status, reservation hash, linkage."""
    plan = document(plan_raw)
    projected = json.loads(encoded(spec))
    binding = projected.pop("study_plan", None)
    if type(binding) is not dict or binding.get("sha256") != sha(plan_raw):
        raise ValueError("the issued study plan fingerprint differs")
    projected["status"] = "preregistered_before_reservation"
    projected["reservation"].pop("sha256", None)
    if encoded(projected) != encoded(plan):
        raise ValueError("the study plan changed after new seed generation")


def verify_known_history(record, provenance, snapshots, reserved_seeds=()):
    """Seed-independent validation is shared with the preopening check."""
    ids, sources = record.get("source_ids"), record.get("sources")
    if (type(ids) is not list or not ids or any(type(name) is not str or not name or name.startswith("@") for name in ids)
            or len(ids) != len(set(ids)) or type(sources) is not dict or set(ids) != set(sources)
            or type(snapshots) is not dict or set(snapshots) != set(ids)):
        raise ValueError("known-history source byte snapshots are required")
    covered = set()
    known_keys = set()
    for name in ids:
        source = sources[name]
        if type(source) is not dict or sha(snapshots[name]) != source.get("sha256"):
            raise ValueError("known-history source bytes differ")
        phases, seeds = source.get("phases"), source.get("seeds")
        if (type(phases) is not list or not phases or len(phases) != len(set(phases))
                or any(phase not in PHASES for phase in phases)):
            raise ValueError("known-history phases are invalid")
        covered.update(phases)
        if seeds is None:
            if type(source.get("unknown_reason")) is not str or not source["unknown_reason"]:
                raise ValueError("unknown source seeds need an explicit reason")
            continue
        if (type(seeds) is not list or len(seeds) != len(set(seeds))
                or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise ValueError("known physical seeds must be explicit distinct integers or unknown")
        if source.get("seed_extractor") is not None and tuple(sorted(seeds)) != recorded_seed_exclusions(
                snapshots[name], source["seed_extractor"]):
            raise ValueError("known seed exclusions differ from their bound original source")
        if set(seeds) & set(reserved_seeds):
            raise ValueError("a known physical seed overlaps the prospective reservation")
        known_keys.update(effective_randomization_key(seed) for seed in seeds)
    if covered != PHASES:
        raise ValueError("known-history sources must cover all three phases without claiming completeness")
    for name in provenance["training_source_ids"]:
        if (name not in sources or "training" not in sources[name]["phases"]
                or sources[name]["sha256"] != provenance["training_sources"][name]["sha256"]):
            raise ValueError("known history must bind every normalized training source")
    return known_keys


def verify_prospective_provenance(provenance, contract, protocol, snapshots):
    """Verify the frozen prospective claim; never synthesize complete history."""
    record = provenance.get("prospective_seed_provenance")
    if (type(record) is not dict or record.get("mode") != MODE
            or record.get("historical_disjointness_certified") is not False
            or record.get("ancestral_ledger_complete") is not False
            or not record.get("limitations")):
        raise ValueError("explicit prospective limitations and uncertified historical disjointness are required")
    ids, sources = record.get("source_ids"), record.get("sources")
    commitment_ids = set(COMMITMENT_IDS)
    if type(record.get("commitments")) is dict and "@study_plan" in record["commitments"]:
        commitment_ids.add("@study_plan")
    if type(ids) is not list or type(snapshots) is not dict or set(snapshots) != set(ids) | commitment_ids:
        raise ValueError("known-history sources and commitment byte snapshots are required")
    known_keys = verify_known_history(record, provenance, {name: snapshots[name] for name in ids}, contract.final_seeds)
    final_keys = [effective_randomization_key(seed) for seed in contract.final_seeds]
    if len(set(final_keys)) != len(final_keys) or set(final_keys) & known_keys:
        raise ValueError("an effective RNG initialization key collides")

    commitments = record.get("commitments")
    if type(commitments) is not dict or set(commitments) != commitment_ids:
        raise ValueError("prospective commitments are incomplete")
    if any(sha(snapshots[name]) != commitments[name] for name in commitment_ids):
        raise ValueError("a prospective commitment changed")
    spec, consumed = document(snapshots["@specification"]), document(snapshots["@consumption"])
    if (spec.get("schema") != SPECIFICATION_SCHEMA or spec.get("status") != "frozen"
            or consumed.get("status") != "consumed" or not consumed.get("run_id")
            or consumed.get("specification_sha256") != commitments["@specification"]
            or consumed.get("reservation_sha256") != commitments["@reservation"]
            or spec.get("reservation", {}).get("sha256") != commitments["@reservation"]):
        raise ValueError("reservation consumption is not bound to the frozen specification")
    if decode_reservation(snapshots["@reservation"], spec["reservation"]) != contract.final_seeds:
        raise ValueError("final seed order differs from the original sealed reservation")
    if spec["reservation"].get("schema_value") == RESERVATION_V3_FORMAT:
        if "@study_plan" not in commitment_ids:
            raise ValueError("v3 requires the plan frozen before seed generation")
        verify_issued_study_plan(spec, snapshots["@study_plan"])
        reservation = document(snapshots["@reservation"])
        if (reservation.get("study_plan_sha256") != commitments["@study_plan"]
                or reservation.get("unchanged_policy_checkpoint_sha256") != contract.policy_checkpoint_sha256
                or reservation.get("producer", {}).get("sha256") != spec["source_sha256"].get(
                    "raccoon_memory/scene_prospective_evaluation.py")):
            raise ValueError("new reservation does not bind the pre-draw plan and unchanged candidate")
    for key, expected in (("policy_checkpoint_sha256", contract.policy_checkpoint_sha256),
                          ("memory_checkpoint_sha256", contract.memory_checkpoint_sha256),
                          ("split_plan_sha256", contract.split_plan_sha256)):
        if spec.get(key) != expected:
            raise ValueError("prospective specification binds different model inputs")
    if encoded(spec.get("known_physical_history")) != encoded({"source_ids": ids, "sources": sources}):
        raise ValueError("known-history scope changed after specification freeze")
    base_protocol = dict(protocol)
    base_protocol.pop("final_seeds", None)
    if "final_seed_range" in protocol or encoded(spec.get("evaluation_protocol")) != encoded(base_protocol):
        raise ValueError("prospective protocol changed after specification freeze")
    base_provenance = dict(provenance)
    base_provenance.pop("prospective_seed_provenance")
    if encoded(spec.get("training_evidence", {}).get("final_evaluation_provenance")) != encoded(base_provenance):
        raise ValueError("training evidence changed after specification freeze")

