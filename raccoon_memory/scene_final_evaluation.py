"""Explicit, simulation-only entry to a frozen final evaluation.

This module does not select a policy, create test scenes, or run an evaluation
at import time.  The caller must supply the exact checkpoint and existing
training-evidence bytes; no lineage is inferred from a baseline policy.  A
contract may be embedded directly in the ordinary evaluation result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math

import torch

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import scene_recall
from .scene_motion_sim import SceneMotionSim
from .scene_oracle_free_session import OracleFreeSceneModelSession
from .scene_ramped_motion_sim import RampedSceneMotionSim


STRUCTURAL_HOLDOUT = "structural_holdout"
PHYSICAL_HOLDOUT = "physical_holdout"
PROSPECTIVE_PHYSICAL_HOLDOUT = "prospective_sealed_v2"
_SEED_PHASES = frozenset(("training", "development", "selection"))


def _digest(raw):
    if type(raw) is not bytes:
        raise ValueError("artifact must be a byte snapshot")
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _valid_sha(value):
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _groups(values, name):
    if (type(values) is not tuple or not values or any(type(item) is not str or not item for item in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{name} must be a nonempty tuple of distinct group names")
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class FinalEvaluationSettings:
    """The portions of the fixed settings applied by this entry point."""

    max_decisions: int
    randomize: bool
    policy_class: str
    memory_class: str
    evaluation_protocol_sha256: str
    simulator: str = "immediate"
    ramp_seconds: float = 0.0
    threshold_degrees: float = 0.0
    device: str = "cpu"
    evaluation_layer: str = STRUCTURAL_HOLDOUT

    def __post_init__(self):
        if type(self.max_decisions) is not int or self.max_decisions < 1:
            raise ValueError("max_decisions must be a positive integer")
        if type(self.randomize) is not bool or type(self.device) is not str or self.device != "cpu":
            raise ValueError("randomization and CPU device must be explicit")
        if type(self.evaluation_layer) is not str or self.evaluation_layer not in (
                STRUCTURAL_HOLDOUT, PHYSICAL_HOLDOUT, PROSPECTIVE_PHYSICAL_HOLDOUT):
            raise ValueError("unsupported final evaluation layer")
        if self.evaluation_layer != STRUCTURAL_HOLDOUT and not self.randomize:
            raise ValueError("physical holdout requires randomized physics")
        if any(type(value) is not str or not value for value in (self.policy_class, self.memory_class)):
            raise ValueError("policy and memory classes must be explicit")
        if not _valid_sha(self.evaluation_protocol_sha256):
            raise ValueError("evaluation protocol fingerprint is required")
        if type(self.simulator) is not str or self.simulator not in ("immediate", "linear", "minimum_jerk"):
            raise ValueError("unsupported simulator profile")
        for value in (self.ramp_seconds, self.threshold_degrees):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("ramp settings must be finite and nonnegative")
        if ((self.simulator == "immediate" and (self.ramp_seconds != 0 or self.threshold_degrees != 0))
                or (self.simulator != "immediate" and self.ramp_seconds <= 0)):
            raise ValueError("ramp settings disagree with the simulator profile")

    def result_record(self):
        return {**asdict(self), "guard": "one_degree_near_joint_limit",
                "command_seconds": 1.0, "post_done_hold_seconds": 1.0,
                "physical_assessment_during_session": False,
                "independence_basis": ({STRUCTURAL_HOLDOUT: "unseen_task_groups",
                    PHYSICAL_HOLDOUT: "heldout_physics_seeds_on_training_groups",
                    PROSPECTIVE_PHYSICAL_HOLDOUT: "prospective_first_generation_incomplete_ancestral_ledger"}
                    [self.evaluation_layer])}

    def sha256(self):
        return _digest(_json_bytes(self.result_record()))

    def protocol_runtime(self):
        result = self.result_record()
        del result["evaluation_protocol_sha256"]
        return result


@dataclass(frozen=True, slots=True)
class FinalEvaluationContract:
    policy_checkpoint_sha256: str
    memory_checkpoint_sha256: str
    training_evidence_sha256: str
    split_plan_sha256: str
    evaluation_settings_sha256: str
    checkpoint_train_groups: tuple
    allowed_execution_groups: tuple
    final_seed_range: tuple | None
    evaluation_settings: FinalEvaluationSettings
    final_seeds: tuple | None = None

    def __post_init__(self):
        for value in (self.policy_checkpoint_sha256, self.memory_checkpoint_sha256,
                      self.training_evidence_sha256, self.split_plan_sha256, self.evaluation_settings_sha256):
            if not _valid_sha(value):
                raise ValueError("contract fingerprints must be lowercase SHA-256 strings")
        trained = _groups(self.checkpoint_train_groups, "checkpoint_train_groups")
        allowed = _groups(self.allowed_execution_groups, "allowed_execution_groups")
        if self.final_seeds is None:
            if (type(self.final_seed_range) is not tuple or len(self.final_seed_range) != 2
                    or any(type(seed) is not int or seed < 0 for seed in self.final_seed_range)
                    or self.final_seed_range[0] > self.final_seed_range[1]):
                raise ValueError("final seed range must be an inclusive integer pair")
        elif (self.final_seed_range is not None or type(self.final_seeds) is not tuple
                or not self.final_seeds or any(type(seed) is not int or seed < 0 for seed in self.final_seeds)
                or len(self.final_seeds) != len(set(self.final_seeds))):
            raise ValueError("explicit final seeds must be distinct nonnegative integers without a range")
        if (type(self.evaluation_settings) is not FinalEvaluationSettings
                or self.evaluation_settings.sha256() != self.evaluation_settings_sha256):
            raise ValueError("frozen evaluation settings differ from their fingerprint")
        if self.evaluation_settings.evaluation_layer == STRUCTURAL_HOLDOUT and trained & allowed:
            raise ValueError("final execution groups overlap the training groups")
        if self.evaluation_settings.evaluation_layer != STRUCTURAL_HOLDOUT and not allowed.issubset(trained):
            raise ValueError("physical holdout groups must be recorded checkpoint training groups")
        if self.evaluation_settings.evaluation_layer == PROSPECTIVE_PHYSICAL_HOLDOUT and self.final_seeds is None:
            raise ValueError("prospective evaluation requires an exact reservation list")

    def seed_specification(self):
        """Bind an exact ordered list or the legacy inclusive range, never both."""
        return ({"final_seed_range": list(self.final_seed_range)} if self.final_seeds is None
                else {"final_seeds": list(self.final_seeds)})

    def contains_seed(self, seed):
        if type(seed) is not int:
            return False
        return (self.final_seed_range[0] <= seed <= self.final_seed_range[1] if self.final_seeds is None
                else seed in self.final_seeds)

    def first_seed(self):
        return self.final_seed_range[0] if self.final_seeds is None else self.final_seeds[0]

    def result_record(self):
        """Return a JSON-compatible copy for the existing evaluation result."""
        value = json.loads(_json_bytes(asdict(self)))
        # Keep the existing range-mode record and fingerprint unchanged.
        value.pop("final_seeds" if self.final_seeds is None else "final_seed_range")
        value["evaluation_settings"] = self.evaluation_settings.result_record()
        return {"contract": value, "contract_sha256": _digest(_json_bytes(value))}


def _document(raw, name):
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid {name} JSON") from exc
    if not isinstance(value, (dict, list)):
        raise ValueError(f"invalid {name} document")
    return value


def _plan_groups(plan, key):
    values = plan.get(key) if isinstance(plan, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"split plan lacks {key}")
    return _groups(tuple(values), key)


def _verify_loaded_weights(raw, model, name):
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"invalid {name} checkpoint") from exc
    expected = payload.get("state_dict") if isinstance(payload, dict) else None
    if name == "policy" and isinstance(payload, dict):
        from . import scene_placement_belief_policy as placement
        from . import scene_applied_action_policy as applied
        loader = configuration = None
        if payload.get("format") in (placement.FORMAT, placement.GRASP_FORMAT):
            loader, configuration = placement.load_checkpoint, _placement_configuration
        elif payload.get("format") == "raccoon-feasible-task-frame-pair-checkpoint.v1":
            from . import scene_feasible_task_frame_student as feasible
            loader, configuration = feasible.load_policy_checkpoint, _feasible_task_frame_configuration
        elif payload.get("format") == applied.TASK_FRAME_CHECKPOINT_FORMAT:
            loader, configuration = applied.load_checkpoint, _task_frame_configuration
        elif payload.get("format") in applied.CHECKPOINT_FORMATS:
            loader, configuration = applied.load_checkpoint, _applied_configuration
        if loader is not None:
            # Reconstruct the exact declared execution wrapper, including its
            # static controller or recurrent-model configuration. No scene runs.
            try:
                packaged = loader(payload)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("invalid packaged policy checkpoint") from exc
            if (type(model) is not type(packaged) or
                    _json_bytes(configuration(model)) != _json_bytes(configuration(packaged))):
                raise ValueError("policy checkpoint configuration differs from the loaded model")
            expected_buffers, actual_buffers = dict(packaged.named_buffers()), dict(model.named_buffers())
            if (set(expected_buffers) != set(actual_buffers) or any(
                    not _same_tensor_bits(value, actual_buffers[key], floating=False)
                    for key, value in expected_buffers.items())):
                raise ValueError("policy checkpoint buffers differ from the loaded model")
            expected = packaged.state_dict()
    actual = model.state_dict() if isinstance(model, torch.nn.Module) else None
    if not isinstance(expected, dict) or not expected or not isinstance(actual, dict) or set(expected) != set(actual):
        raise ValueError(f"{name} checkpoint does not match the loaded model")
    for key, value in expected.items():
        loaded = actual[key]
        if not _same_tensor_bits(value, loaded, floating=True):
            raise ValueError(f"{name} checkpoint does not match the loaded model")


def _same_tensor_bits(expected, actual, *, floating):
    return (isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
            and expected.device.type == actual.device.type == "cpu"
            and expected.dtype == actual.dtype and expected.shape == actual.shape
            and (not floating or expected.is_floating_point())
            and bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
            and torch.equal(expected.detach().reshape(-1).contiguous().view(torch.uint8),
                            actual.detach().reshape(-1).contiguous().view(torch.uint8)))


def _placement_configuration(model):
    # Conditional position/task/history states legitimately evolve in a session.
    # These static choices must continue to match the frozen checkpoint instead.
    value = {"joint_limited_pitch": model.joint_limited_pitch,
             "object_height_mm": model.belief.object_height_mm,
             "center_lateral_on_close": model.belief.center_lateral_on_close,
             "history_length": model.history_length, "feature_count": model.feature_count,
             "reference_class": _class_name(model.reference),
             "reference_architecture": model.reference.architecture,
             "reference_history_length": model.reference.history_length,
             "reference_feature_count": model.reference.feature_count}
    if hasattr(model, "grasp_tcp_above_center_mm"):
        value["grasp_tcp_above_center_mm"] = model.grasp_tcp_above_center_mm
    return value


def _applied_configuration(model):
    # Runtime hidden state, expected scenes and pending commands legitimately
    # evolve. The public-input interface and learned recurrence stay frozen.
    learned, gru = model.model, model.model.gru
    return {"history_length": model.history_length, "feature_count": model.feature_count,
            "model_class": _class_name(learned), "model_training": learned.training,
            "gru_class": _class_name(gru), "gru_input_size": gru.input_size,
            "gru_hidden_size": gru.hidden_size, "gru_layers": gru.num_layers,
            "gru_bias": gru.bias, "gru_batch_first": gru.batch_first,
            "gru_dropout": gru.dropout, "gru_bidirectional": gru.bidirectional,
            "action_head_class": _class_name(learned.action_head),
            "action_head_inputs": learned.action_head.in_features,
            "action_head_outputs": learned.action_head.out_features,
            "joint_head_class": _class_name(learned.joint_head),
            "joint_head_inputs": learned.joint_head.in_features,
            "joint_head_outputs": learned.joint_head.out_features}


def _task_frame_configuration(model):
    from .scene_task_frame_student import model_contract
    learned = model.model
    modules = {}
    for name, part in learned.named_modules():
        value = {"class": _class_name(part), "training": part.training}
        if isinstance(part, torch.nn.Linear):
            value.update(inputs=part.in_features, outputs=part.out_features, bias=part.bias is not None)
        elif isinstance(part, torch.nn.LSTM):
            value.update(input_size=part.input_size, hidden_size=part.hidden_size, layers=part.num_layers,
                         bias=part.bias, batch_first=part.batch_first, dropout=part.dropout,
                         bidirectional=part.bidirectional, proj_size=part.proj_size)
        elif isinstance(part, torch.nn.SiLU):
            value["inplace"] = part.inplace
        modules[name] = value
    return {"history_length": model.history_length, "feature_count": model.feature_count,
            "model_contract": model_contract(learned.include_object_ids), "modules": modules}


def _feasible_task_frame_configuration(model):
    from . import scene_feasible_task_frame_student as feasible
    branch = "bounded_fk" if isinstance(model.model, feasible.FeasibleTaskFrameStudent) else "cartesian"
    return {"body": _task_frame_configuration(model), "execution_contract": feasible.model_contract(branch)}


def _class_name(model):
    return f"{type(model).__module__}.{type(model).__qualname__}"


def _verify_physical_seed_provenance(provenance, contract, source_snapshots):
    """Check an explicitly complete seed inventory, with bytes for every source.

    ``physical_seed_provenance`` binds both checkpoints and the split. It names
    every source and records its training/development/selection phases and the
    exact physical seeds used in that source. An explicit empty list means a
    reviewed source used no physical seeds; missing/unknown seeds are rejected.
    ``phase_complete`` maps all three phase names to True. ``source_ids`` and
    ``sources`` identify each source; each source has ``sha256``, ``phases`` and
    ``seeds`` fields, and ``source_snapshots`` supplies its original bytes.
    The inventory must include every normalized training source with the same
    digest. Completeness and extraction of seed lists must still be established
    from the underlying records by the producer; a hash is not that proof.
    """
    record = provenance.get("physical_seed_provenance")
    if not isinstance(record, dict) or record.get("complete") is not True:
        raise ValueError("complete physical seed provenance is required")
    if (record.get("policy_checkpoint_sha256") != contract.policy_checkpoint_sha256
            or record.get("memory_checkpoint_sha256") != contract.memory_checkpoint_sha256
            or record.get("split_plan_sha256") != contract.split_plan_sha256):
        raise ValueError("physical seed provenance does not bind both checkpoints and split")
    phase_complete = record.get("phase_complete")
    if (not isinstance(phase_complete, dict) or set(phase_complete) != _SEED_PHASES
            or any(phase_complete[phase] is not True for phase in _SEED_PHASES)):
        raise ValueError("physical seed sources must be complete for training, development and selection")
    ids, sources = record.get("source_ids"), record.get("sources")
    if (not isinstance(ids, list) or not ids or any(type(name) is not str or not name for name in ids)
            or len(ids) != len(set(ids)) or not isinstance(sources, dict) or set(ids) != set(sources)):
        raise ValueError("physical seed source inventory is incomplete")
    if type(source_snapshots) is not dict or set(source_snapshots) != set(ids):
        raise ValueError("a byte snapshot of every physical seed source is required")
    covered = set()
    for name in ids:
        source = sources[name]
        if not isinstance(source, dict) or not _valid_sha(source.get("sha256")):
            raise ValueError("physical seed source fingerprint is missing")
        if _digest(source_snapshots[name]) != source["sha256"]:
            raise ValueError("physical seed source snapshot fingerprint differs")
        phases, seeds = source.get("phases"), source.get("seeds")
        if (not isinstance(phases, list) or not phases
                or any(type(phase) is not str or phase not in _SEED_PHASES for phase in phases)
                or len(phases) != len(set(phases))):
            raise ValueError("physical seed source phases are invalid")
        if (not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(seeds) != len(set(seeds))):
            raise ValueError("physical seed source seeds must be an explicit list of distinct nonnegative integers")
        if any(contract.contains_seed(seed) for seed in seeds):
            raise ValueError("a previously used physical seed overlaps the reserved final seeds")
        covered.update(phases)
    if covered != _SEED_PHASES:
        raise ValueError("physical seed source inventory lacks a required phase")
    for name in provenance["training_source_ids"]:
        source = sources.get(name)
        if (not isinstance(source, dict) or "training" not in source["phases"]
                or source["sha256"] != provenance["training_sources"][name]["sha256"]):
            raise ValueError("physical seed inventory must include every recorded training source")


def _verify_training_source_inventory(provenance, policy_sha256, split_sha256, train_groups, *, prospective=False):
    """Shared seed-independent checks, also required before opening a seal."""
    if not isinstance(provenance, dict):
        raise ValueError("training result lacks normalized complete provenance")
    if (provenance.get("checkpoint_sha256") != policy_sha256
            or provenance.get("split_plan_sha256") != split_sha256):
        raise ValueError("training evidence does not bind this policy and split")
    source_ids, sources = provenance.get("training_source_ids"), provenance.get("training_sources")
    source_completeness = provenance.get("training_sources_complete") is True
    if prospective:
        source_completeness = (provenance.get("training_sources_complete") is False
            and provenance.get("current_update_sources_complete") is True
            and provenance.get("ancestral_training_sources_complete") is False)
    if (not source_completeness or not isinstance(source_ids, list) or not source_ids
            or any(type(name) is not str or not name for name in source_ids)
            or len(source_ids) != len(set(source_ids)) or not isinstance(sources, dict)
            or set(source_ids) != set(sources)):
        raise ValueError("training result lacks a complete source inventory")
    recorded = set()
    for name in source_ids:
        source = sources[name]
        if not isinstance(source, dict) or not _valid_sha(source.get("sha256")):
            raise ValueError("training source fingerprint is missing")
        values = source.get("groups")
        if isinstance(values, dict):
            if any(type(count) is not int or count <= 0 for count in values.values()):
                raise ValueError("training source group counts must be positive")
            values = list(values)
        if not isinstance(values, list):
            raise ValueError("training source does not record its groups")
        recorded.update(_groups(tuple(values), "recorded training groups"))
    if recorded != set(train_groups):
        raise ValueError("actual training groups differ from their evidence")


def _validate_inputs(contract, settings, *, policy_checkpoint, memory_checkpoint,
                     training_evidence, split_plan, evaluation_protocol, student, memory, physics_seed,
                     physical_seed_source_snapshots=None):
    if type(contract) is not FinalEvaluationContract or type(settings) is not FinalEvaluationSettings:
        raise ValueError("a fixed final contract and settings are required")
    for raw, expected, name in ((policy_checkpoint, contract.policy_checkpoint_sha256, "policy"),
                                (memory_checkpoint, contract.memory_checkpoint_sha256, "memory"),
                                (training_evidence, contract.training_evidence_sha256, "training evidence"),
                                (split_plan, contract.split_plan_sha256, "split plan")):
        if _digest(raw) != expected:
            raise ValueError(f"{name} fingerprint differs from the contract")
    if settings != contract.evaluation_settings or settings.sha256() != contract.evaluation_settings_sha256:
        raise ValueError("evaluation settings fingerprint differs from the contract")
    if _digest(evaluation_protocol) != settings.evaluation_protocol_sha256:
        raise ValueError("evaluation protocol fingerprint differs from the settings")
    protocol = _document(evaluation_protocol, "evaluation protocol")
    expected_protocol = {"runtime": settings.protocol_runtime(),
                         "checkpoint_train_groups": list(contract.checkpoint_train_groups),
                         "allowed_execution_groups": list(contract.allowed_execution_groups),
                         **contract.seed_specification()}
    alternate_seed_key = "final_seeds" if contract.final_seeds is None else "final_seed_range"
    if (not isinstance(protocol, dict) or any(
            _json_bytes(protocol.get(key)) != _json_bytes(value) for key, value in expected_protocol.items())
            or alternate_seed_key in protocol):
        raise ValueError("evaluation protocol settings or boundaries differ from the contract")
    if not contract.contains_seed(physics_seed):
        raise ValueError("physics seed is outside the final contract")

    plan = _document(split_plan, "split plan")
    train = _plan_groups(plan, "train_groups")
    validation = _plan_groups(plan, "validation_groups")
    test = _plan_groups(plan, "test_groups")
    if train & validation or train & test or validation & test:
        raise ValueError("split groups overlap")
    if not frozenset(contract.checkpoint_train_groups).issubset(train):
        raise ValueError("checkpoint training groups differ from the split")
    if settings.evaluation_layer == STRUCTURAL_HOLDOUT:
        if not frozenset(contract.allowed_execution_groups).issubset(test):
            raise ValueError("final execution groups must be in the reserved split")
    elif not frozenset(contract.allowed_execution_groups).issubset(train):
        raise ValueError("physical holdout groups must be in the training split")
    ranges = plan.get("seed_ranges_inclusive") if isinstance(plan, dict) else None
    if not isinstance(ranges, dict):
        raise ValueError("split plan lacks seed ranges")
    seed_ranges = []
    for name in ("train", "validation", "test"):
        bounds = ranges.get(name)
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(value) is not int or value < 0 for value in bounds) or bounds[0] > bounds[1]):
            raise ValueError("split plan has an invalid seed range")
        seed_ranges.append(bounds)
    if any(max(a[0], b[0]) <= min(a[1], b[1])
           for index, a in enumerate(seed_ranges) for b in seed_ranges[index + 1:]):
        raise ValueError("split seed ranges overlap")
    test_range = seed_ranges[-1]
    if contract.final_seeds is None:
        if not test_range[0] <= contract.final_seed_range[0] <= contract.final_seed_range[1] <= test_range[1]:
            raise ValueError("final seed range differs from the split")
        if settings.evaluation_layer == PHYSICAL_HOLDOUT and contract.final_seed_range != tuple(test_range):
            raise ValueError("physical holdout requires the entire reserved final seed range")
    elif any(lo <= seed <= hi for seed in contract.final_seeds for lo, hi in seed_ranges[:2]):
        raise ValueError("explicit final seeds overlap the split training or validation ranges")
    # A replacement reservation is bound by the exact contract/protocol list,
    # not by the historical test interval. Its actual source inventory below
    # is mandatory even for a structural holdout.

    evidence = _document(training_evidence, "training evidence")
    provenance = evidence.get("final_evaluation_provenance") if isinstance(evidence, dict) else None
    _verify_training_source_inventory(provenance, contract.policy_checkpoint_sha256,
        contract.split_plan_sha256, contract.checkpoint_train_groups,
        prospective=settings.evaluation_layer == PROSPECTIVE_PHYSICAL_HOLDOUT)
    if settings.evaluation_layer == PROSPECTIVE_PHYSICAL_HOLDOUT:
        from .scene_prospective_evaluation import verify_prospective_provenance
        verify_prospective_provenance(provenance, contract, protocol, physical_seed_source_snapshots)
    elif settings.evaluation_layer == PHYSICAL_HOLDOUT or contract.final_seeds is not None:
        _verify_physical_seed_provenance(provenance, contract, physical_seed_source_snapshots)
    if not isinstance(student, GuardedPolicy) or student.training or student.policy.training:
        raise ValueError("the loaded one-degree guarded student must be in evaluation mode")
    if not isinstance(memory, scene_recall.SceneRecall) or memory.model.training:
        raise ValueError("the loaded scene memory must be in evaluation mode")
    if (_class_name(student.policy) != settings.policy_class
            or _class_name(memory.model) != settings.memory_class):
        raise ValueError("loaded model classes differ from the final settings")
    _verify_loaded_weights(policy_checkpoint, student.policy, "policy")
    _verify_loaded_weights(memory_checkpoint, memory.model, "memory")


class _DeferredScene(SceneMotionSim):
    """Type-compatible placeholder; constructing it does not build a scene."""

    def __init__(self):
        pass


def _create_simulator(initial_scene, physics_seed, settings):
    if settings.simulator == "immediate":
        return SceneMotionSim(initial_scene, seed=physics_seed, randomize=settings.randomize)
    return RampedSceneMotionSim(initial_scene, seed=physics_seed, randomize=settings.randomize,
                                ramp_seconds=settings.ramp_seconds, threshold_degrees=settings.threshold_degrees,
                                ramp_shape=settings.simulator)


class _FinalSceneSession(OracleFreeSceneModelSession):
    """Use legacy ``train_groups`` only as the frozen execution-group alias."""

    _LOCKED = frozenset(("train_groups", "checkpoint_train_groups", "sim", "student", "memory",
                         "device", "max_decisions", "_contract", "_settings", "_physics_seed",
                         "_policy_checkpoint", "_memory_checkpoint", "_final_bindings_locked"))

    def __setattr__(self, name, value):
        if getattr(self, "_final_bindings_locked", False) and name in self._LOCKED:
            raise AttributeError("final evaluation bindings cannot be reassigned")
        super().__setattr__(name, value)

    def verify_frozen(self):
        if (self.plan_sha256 != self._contract.split_plan_sha256
                or self.train_groups != frozenset(self._contract.allowed_execution_groups)
                or self.checkpoint_train_groups != frozenset(self._contract.checkpoint_train_groups)
                or self.device != self._settings.device or self.max_decisions != self._settings.max_decisions):
            raise ValueError("final session bindings changed")
        reset = getattr(self.sim, "_reset_info", None)
        if (not isinstance(reset, dict) or reset.get("seed") != self._physics_seed
                or type(reset.get("seed")) is not int or reset.get("randomize") is not self._settings.randomize):
            raise ValueError("simulator seed or randomization differs from the final contract")
        if self._settings.simulator == "immediate":
            if type(self.sim) is not SceneMotionSim:
                raise ValueError("simulator profile differs from the final contract")
        elif (type(self.sim) is not RampedSceneMotionSim or self.sim.ramp_shape != self._settings.simulator
              or self.sim.ramp_seconds != self._settings.ramp_seconds
              or self.sim.threshold_degrees != self._settings.threshold_degrees):
            raise ValueError("simulator profile differs from the final contract")
        if (not isinstance(self.student, GuardedPolicy) or self.student.training or self.student.policy.training
                or not isinstance(self.memory, scene_recall.SceneRecall) or self.memory.model.training
                or _class_name(self.student.policy) != self._settings.policy_class
                or _class_name(self.memory.model) != self._settings.memory_class):
            raise ValueError("loaded model configuration changed")
        _verify_loaded_weights(self._policy_checkpoint, self.student.policy, "policy")
        _verify_loaded_weights(self._memory_checkpoint, self.memory.model, "memory")

    def _register(self, value):
        self.verify_frozen()
        return super()._register(value)

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        self.verify_frozen()
        return super()._move(action, commit_lock=commit_lock, is_current=is_current,
                             on_transfer_commit=on_transfer_commit)

    def _execute_validated_step(self, step, **kwargs):
        self.verify_frozen()
        return super()._execute_validated_step(step, **kwargs)


def create_final_session(contract, settings, *, policy_checkpoint, memory_checkpoint,
                         training_evidence, split_plan, evaluation_protocol, student, memory, physics_seed,
                         scene_factory, cancel=None, physical_seed_source_snapshots=None):
    """Validate frozen inputs before constructing any actual simulator scene.

    The existing training result must include a normalized, explicitly complete
    ``final_evaluation_provenance`` source inventory. Every source must record a
    fingerprint and its actual training groups; unsupported or incomplete prior
    results are rejected rather than supplemented from another checkpoint. The
    optional physical holdout additionally requires a complete normalized seed
    inventory and the byte snapshots it names. These are attestations backed by
    source identities, not automatic verification that every seed was extracted.
    The caller must verify the wider schedule and call ``session.verify_frozen()``
    before accepting results.
    """
    if not callable(scene_factory):
        raise ValueError("an explicit deferred scene factory is required")
    _validate_inputs(contract, settings, policy_checkpoint=policy_checkpoint, memory_checkpoint=memory_checkpoint,
                     training_evidence=training_evidence, split_plan=split_plan,
                     evaluation_protocol=evaluation_protocol, student=student, memory=memory,
                     physics_seed=physics_seed, physical_seed_source_snapshots=physical_seed_source_snapshots)
    session = _FinalSceneSession(_DeferredScene(), memory, student, device=settings.device,
                                 max_decisions=settings.max_decisions, cancel=cancel,
                                 checkpoint_train_groups=contract.checkpoint_train_groups)
    # The inherited constructor checks the live split file without building a
    # scene.  Reject a changed file before invoking the simulator factory.
    if (session.plan_sha256 != contract.split_plan_sha256
            or session.checkpoint_train_groups != frozenset(contract.checkpoint_train_groups)):
        raise ValueError("live split or checkpoint training groups changed")
    session.train_groups = frozenset(contract.allowed_execution_groups)
    session._contract, session._settings, session._physics_seed = contract, settings, physics_seed
    session._policy_checkpoint, session._memory_checkpoint = policy_checkpoint, memory_checkpoint
    initial_scene = scene_factory(physics_seed)
    session.sim = _create_simulator(initial_scene, physics_seed, settings)
    session.verify_frozen()
    session._final_bindings_locked = True
    return session
