"""Bind the current Terra student and corrected GRU to one development simulation.

The 1,000-original training roster and the allowed logical transitions are
checked independently. Only already-exposed 552-development starts may be
bound. This is not a final evaluation, a hardware controller, or a new score.
Loading and binding never run a policy or advance the simulator.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import copy
from functools import partial
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch
import weakref

import torch

from raccoon_control import near_joint_limit_guard, near_joint_limit_policy
from raccoon_control.near_joint_limit_policy import NearJointLimitPolicy
from . import full_teacher_training, scene_features, scene_policy, scene_recall, task_catalog
from . import scene_belief_canonical_policy, scene_current_late_nonlinear, scene_placement_belief_policy
from . import scene_current_observation_correction, scene_no_moving_policy
from . import terra552_historical_provenance as historical_provenance
from . import terra_kinematic_correction as traced, terra_registered_nominal_correction as registered
from .scene_custom_slots_sim import CustomSlotsSceneMotionSim
from .scene_model_session import UnsupportedStep
from .scene_oracle_free_session import OracleFreeSceneModelSession, OracleFreeSceneProgramCoordinator
from scripts import evaluate_guarded_student_v2_memory_session as public_boundary
from scripts import evaluate_terra_kinematic_correction_development as development
from scripts import evaluate_terra_v21_memory_session as evidence
from scripts import experiment_terra_encoder_multiseed as starts
from scripts import experiment_terra_registered_joint_heads as original
from scripts import train_terra_bc1020_from_scratch as direct

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/architecture_research'
STUDENT = (ARTIFACTS / 'astra_terra_weight_repair_v1/short_move_lock_direct_continuation_v1'
           / 'original_only/unlocked/policy/candidate.pt')
STUDENT_SHA256 = 'b799075076f27970f22bdff1ed08ae9c72468519a94e184c5ff8e828409df132'
STUDENT_STATE_SHA256 = 'b6175a790de929046d583559a74f8610461ccf8dd56a15a6a953354d8a6b81c6'
FIT = STUDENT.parent.parent / 'fit.json'
FIT_SHA256 = '36f49275874c83dd4a9db43b1f894411ca5530d85a8df83da537a8f9b5577c32'
PARENT_SHA256 = 'e53b435797ce5c53a6205e8c04973ce8151db175e03e9e5257b423d7df9c510b'
BASE_REPORT = ARTIFACTS / 'terra_encoder_joint_training_seed20261101_v1/training/result.json'
BASE_REPORT_SHA256 = '6446d4f970022be4eb7a5a1020b76f550a594587c30cf4ccaa77a4ab1255031c'
MEMORY = ARTIFACTS / 'astra_terra_weight_repair_v1/gru128_decoder_margin_repair_v1/model.pt'
MEMORY_SHA256 = 'e8fdf96cb266c094afb90a02bc8953e18f68ca13f8aa03bea2709dea0ac06682'
MEMORY_STATE_SHA256 = '4f49ee6a5e54871bcb545e3c097d64cc31dcebc24985ad66e780513a9ccdd30b'
SCHEDULE_SHA256 = '5e20d8c2962cd1cdc6ede34c92e40647a41d1aa37dc2e157b7515cb46fa64ff5'
MAX_DECISIONS = 60
_GUARD_FORWARD = NearJointLimitPolicy.forward
_TRACED_CALL = traced.TracedGuard._call
_REGISTERED_FORWARD = registered.RegisteredGraspPlacement.forward_with_control_state
_REGISTERED_FRAME = registered.RegisteredGraspPlacement._forward_in_frame
_GRASP_FORWARD = scene_placement_belief_policy.GraspPlacementBeliefCanonicalPolicy.forward_with_control_state
_PLACEMENT_FORWARD = scene_placement_belief_policy.PlacementBeliefCanonicalPolicy.forward_with_control_state
_BELIEF_FORWARD = scene_belief_canonical_policy.BeliefCanonicalPolicy.forward_with_control_state
_STUDENT_FORWARD = scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy.forward
_STUDENT_DIAGNOSTICS = scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy.diagnostics
_FROZEN_OUTPUTS = scene_current_observation_correction.SceneCurrentObservationCorrectionPolicy._frozen_outputs
_ENCODE_LENGTHS = scene_no_moving_policy.SceneNoMovingPolicy._encode_with_lengths
_RECALL_OBSERVE = scene_recall.SceneRecall.observe
_RECALL_READ = scene_recall.SceneRecall.recall
_RECALL_CLEAR = scene_recall.SceneRecall.clear
_GRU_ADVANCE = scene_recall.SceneMemory.advance
_GRU_DECODE = scene_recall.SceneMemory.decode
_LAYER_FORWARDS = {torch.nn.Sequential: torch.nn.Sequential.forward, torch.nn.Linear: torch.nn.Linear.forward,
                   torch.nn.ReLU: torch.nn.ReLU.forward, torch.nn.GRUCell: torch.nn.GRUCell.forward}
_SIMULATOR_OWNERS = weakref.WeakKeyDictionary()
_SIMULATOR_OWNERS_LOCK = threading.Lock()

_EXECUTION_FILES = tuple(ROOT / path for path in (
    'raccoon_memory/terra552_simulation_runtime.py',
    'raccoon_memory/terra552_historical_provenance.py',
    'raccoon_memory/terra_registered_nominal_correction.py',
    'raccoon_memory/terra_kinematic_correction.py',
    'raccoon_memory/scene_placement_belief_policy.py',
    'raccoon_memory/scene_belief_canonical_policy.py',
    'raccoon_memory/scene_canonical_task_policy.py',
    'raccoon_memory/scene_current_late_nonlinear.py',
    'raccoon_memory/scene_current_observation_correction.py',
    'raccoon_memory/scene_no_moving_policy.py', 'raccoon_memory/scene_early_history_correction.py',
    'raccoon_memory/nominal_bounded_ik.py',
    'raccoon_memory/nominal_object_belief.py',
    'raccoon_memory/scene_oracle_free_session.py',
    'raccoon_memory/scene_model_session.py',
    'raccoon_memory/scene_program_coordinator.py',
    'raccoon_memory/task_program.py', 'raccoon_memory/task_program_v2.py',
    'raccoon_memory/task_catalog.py', 'raccoon_memory/scene_policy.py',
    'raccoon_memory/scene_features.py', 'raccoon_memory/scene_recall.py',
    'raccoon_memory/scene_causal_frame_policy.py', 'raccoon_memory/scene_unified_kinematic_policy.py',
    'raccoon_memory/scene_demo_data.py', 'raccoon_memory/motion_sim.py',
    'raccoon_memory/scene_custom_slots_sim.py', 'raccoon_memory/scene_ramped_motion_sim.py',
    'raccoon_memory/scene_motion_sim.py', 'raccoon_memory/motion_policy.py',
    'raccoon_control/near_joint_limit_guard.py', 'raccoon_control/near_joint_limit_policy.py',
    'raccoon_control/kinematics.py', 'configs/sim/raccoon_pick_place.xml',
    'configs/scene_split_plan.v1.json', 'configs/terra_mixed_coverage.v2.json',
    'configs/task_program.v2.schema.json', 'configs/task_program.v2_1.schema.json'))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _case_id(before, source, destination):
    before = task_catalog.scene(before)
    task_catalog.transfer(before, source, destination)
    return 'layout_' + ''.join(str(len(stack)) for stack in before) + f'_s{source}_d{destination}'


@dataclass(frozen=True, slots=True)
class Terra552Contract:
    student_checkpoint_sha256: str
    student_state_sha256: str
    memory_checkpoint_sha256: str
    memory_state_sha256: str
    training_report_sha256: str
    training_index_sha256: str
    checkpoint_train_groups: tuple[str, ...]
    development_transition_ids: tuple[str, ...]
    development_schedule_sha256: str
    training_originals: int = 1000
    scope: str = 'existing_development_starts_only'
    qualification: str = 'integration_only_no_prior_score_inherited'
    actual_hardware: bool = False


@dataclass(frozen=True, slots=True)
class _SimulatorBinding:
    model: object
    data: object
    reset_info_sha256: str
    model_parameters_sha256: str
    slot_xy_mm: tuple
    initial_state_sha256: str
    initial_physical_state_sha256: str
    development_index: int


def _expected_contract():
    catalog = task_catalog.one_step_cases()
    groups = tuple(sorted({scene_policy.split_group_for(case['canonical_scene_before'], case['task'])
                           for case in catalog}))
    transitions = tuple(sorted(case['case_id'] for case in catalog))
    return Terra552Contract(STUDENT_SHA256, STUDENT_STATE_SHA256, MEMORY_SHA256, MEMORY_STATE_SHA256,
                            original.PARENT_REPORT_SHA, original.INDEX_SHA, groups, transitions, SCHEDULE_SHA256)


def _physical_start_sha(state):
    # The uncertain marker and expected-scene bookkeeping can change on a
    # stop even when no physical command was issued. Compare all physical and
    # model data separately, without mistaking that marker for a reset.
    symbolic = {'committed_layout', 'layout_status', 'layout_uncertainty_reason',
                'pending_transfer', 'layout_safety_sha256'}
    return development.evaluator.record_sha({key: value for key, value in state.items() if key not in symbolic})


def _training_and_transition_scope(student_binding):
    """Verify recorded training exposure separately from the logical allowlist."""
    base = direct.pinned_json(BASE_REPORT, BASE_REPORT_SHA256)
    parent = base['metadata']['parent_binding']
    _need(base['status'] == 'completed' and base['metadata']['index_sha256'] == original.INDEX_SHA
          and base['metadata']['originals'] == 1000
          and student_binding['physical_originals'] == 1000
          and student_binding['training_index_sha256'] == original.INDEX_SHA,
          'student and original training evidence differ')
    # The original audit must inspect historical training-source bytes, not
    # today's changed session. Run it in a fresh process so its narrowly scoped
    # historical source resolver cannot affect a concurrent live collector.
    try:
        checked = subprocess.run(
            [sys.executable, str(Path(historical_provenance.__file__).resolve())],
            input=json.dumps(parent), text=True, capture_output=True, cwd=ROOT,
            timeout=180, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('historical training provenance verification timed out') from exc
    if checked.returncode:
        raise ValueError('historical training provenance verification failed: '
                         + checked.stderr.strip()[-1500:])
    try:
        provenance = json.loads(checked.stdout)
        recorded_groups = provenance['groups']
        _need(provenance['schema'] == 'terra552-historical-training-provenance.v1'
              and isinstance(recorded_groups, list) and len(recorded_groups) == len(set(recorded_groups))
              and all(isinstance(group, str) and group for group in recorded_groups)
              and provenance['preregistration_sha256'] == '3902a36c09bf374a3e6cd6ee4f47b1966aa91a4b095f62e0380eaddcf93b9b7e'
              and provenance['historical_session_sha256'] == historical_provenance.HISTORICAL_SESSION_SHA256
              and Path(provenance['historical_session_path']).resolve() == historical_provenance.HISTORICAL_SESSION.resolve()
              and provenance['current_session_sha256'] == direct.digest(ROOT / historical_provenance.SESSION_SOURCE)
              and direct.digest(historical_provenance.HISTORICAL_SESSION) == historical_provenance.HISTORICAL_SESSION_SHA256,
              'historical and current training source identities differ')
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid historical training provenance response') from exc
    roster = direct.pinned_json(original.PARENT / 'result.json', original.PARENT_REPORT_SHA)['roster']
    payload = torch.load(io.BytesIO(STUDENT.read_bytes()), map_location='cpu', weights_only=True)
    ids = [row['case_id'] for row in roster]
    _need(direct.digest(STUDENT) == STUDENT_SHA256 and len(ids) == len(set(ids)) == 1000
          and payload['terra_training']['selected_case_ids'] == ids,
          'the student and the 1,000-original roster differ')
    catalog = task_catalog.one_step_cases()
    by_id = {case['case_id']: case for case in catalog}
    _need(len(catalog) == len(by_id) == 276, 'the legal transition catalog differs')
    derived = set()
    for row in roster:
        start = row['training_start']
        case = by_id.get(start['layout_id'])
        _need(case is not None, 'roster contains a transition outside the catalog')
        group = scene_policy.split_group_for(case['canonical_scene_before'], case['task'])
        _need(group == start['recomputed_task_group'], 'training group differs from its recorded transition')
        derived.add(group)
    catalog_groups = {scene_policy.split_group_for(case['canonical_scene_before'], case['task']) for case in catalog}
    _need(len(derived) == 15 and derived == set(recorded_groups) == catalog_groups
          and {row['training_start']['layout_id'] for row in roster} == set(by_id),
          'training roster or independently defined execution transitions are incomplete')
    return tuple(sorted(derived)), tuple(sorted(by_id))


def _development_starts():
    """Read only the already registered development schedule and start hashes."""
    specs, state_hashes, schedule_sha = starts.specs('repeated552')
    _need(schedule_sha == SCHEDULE_SHA256 and len(specs) == len(state_hashes) == 552
          and all(spec['partition'] == 'development' and spec['index'] == i
                  for i, spec in enumerate(specs))
          and all(direct.checked_digest(value) for value in state_hashes),
          'the registered development starts differ')
    return specs, state_hashes


class _VerifiedSceneRecall(scene_recall.SceneRecall):
    """Check the same pinned model before and after a latent memory call."""

    def __init__(self, model, owner):
        self._owner = owner
        super().__init__(model)

    def _verified_call(self, function, *args, **kwargs):
        with self.synchronized():
            self._owner.verify()
            if self._owner._sim is not None:
                self._owner._verify_simulator()
            try:
                result = function(*args, **kwargs)
            except BaseException:
                try:
                    self._owner.verify()
                    if self._owner._sim is not None:
                        self._owner._verify_simulator()
                except BaseException:
                    _RECALL_CLEAR(self)
                raise
            try:
                self._owner.verify()
                if self._owner._sim is not None:
                    self._owner._verify_simulator()
            except BaseException:
                # Never retain a latent produced under unverified execution.
                _RECALL_CLEAR(self)
                raise
            return result

    def observe(self, current, *, write=False):
        return _VerifiedSceneRecall._verified_call(self, _RECALL_OBSERVE, self, current, write=write)

    def recall(self):
        return _VerifiedSceneRecall._verified_call(self, _RECALL_READ, self)


_VERIFIED_MEMORY_CALL = _VerifiedSceneRecall._verified_call


class _VerifiedCommitBoundary:
    """Verify both sides of the existing atomic commit before publishing it."""

    def __init__(self, session, lock, notify, before, task, event_index, checkpoint_count):
        self.session, self.lock, self.notify = session, lock, notify
        self.before, self.task = before, task
        self.event_index, self.checkpoint_count = event_index, checkpoint_count
        self.pending_notification = False

    def mark_commit(self):
        self.pending_notification = True

    def __enter__(self):
        if self.lock is not None:
            self.lock.acquire()
        try:
            self.session.verify_frozen()
        except BaseException:
            if self.lock is not None:
                self.lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:
                if any(event.get('event') == 'transfer' and event.get('committed')
                       for event in self.session.events[self.event_index:]):
                    self._withdraw_unverified_commit()
                return False
            if exc_type is None:
                try:
                    self.session.verify_frozen()
                except BaseException:
                    if any(event.get('event') == 'transfer' and event.get('committed')
                           for event in self.session.events[self.event_index:]):
                        self._withdraw_unverified_commit()
                    raise
                if self.pending_notification:
                    if self.notify is not None:
                        self.notify()
                    self.pending_notification = False
        finally:
            if self.lock is not None:
                self.lock.release()
        return False

    def _withdraw_unverified_commit(self):
        # Physical commands, if any, cannot be undone. Withdraw only the
        # tentative symbolic commit and leave recovery explicitly required.
        session = self.session
        with session._scene_view_lock:
            session.current = self.before
            session.sim._committed_layout = self.before
            session._unresolved_transfer = {
                'expected_object_id': self.task['object_id'],
                'object_id_source': 'expected_scene_source_top',
                'physical_object_id': None,
                'source_slot': self.task['source_slot'],
                'destination_slot': self.task['destination_slot']}
            for event in session.events[self.event_index:]:
                if event.get('event') == 'transfer' and event.get('committed'):
                    event['committed'] = False
                    event['integrity_failure'] = 'commit_not_verified'
            del session._physical_checkpoints[self.checkpoint_count:]
        session._halt('model or runtime changed before commit verification')


class _Terra552Session(OracleFreeSceneModelSession):
    """Existing expected-scene execution, with an explicit development allowlist."""

    _LOCKED = frozenset({'sim', 'student', 'memory', 'device', 'max_decisions',
                         'train_groups', 'checkpoint_train_groups', '_owner', '_scope_locked'})

    def __setattr__(self, name, value):
        if getattr(self, '_scope_locked', False) and name in self._LOCKED:
            raise AttributeError('the Terra simulation binding is immutable')
        super().__setattr__(name, value)

    def __init__(self, sim, owner, *, cancel=None):
        # The legacy constructor only accepts the old eight-group split. It is
        # not evidence about this student. Bind the verified roster and the
        # independently defined 276-transition development scope afterward.
        super().__init__(sim, owner.memory, owner.guard, device='cpu', max_decisions=MAX_DECISIONS,
                         cancel=cancel, slot_xy_mm=sim.slot_xy_mm)
        self.checkpoint_train_groups = frozenset(owner.contract.checkpoint_train_groups)
        self.train_groups = frozenset(owner._execution_groups)
        self._owner = owner
        self._scope_locked = True
        self.verify_frozen()

    def verify_frozen(self):
        owner = self._owner
        owner.verify()
        owner._verify_simulator()
        _need(self.student is owner.guard and self.memory is owner.memory and self.sim is owner._sim
              and self.checkpoint_train_groups == frozenset(owner.contract.checkpoint_train_groups)
              and self.train_groups == frozenset(owner._execution_groups)
              and self.device == 'cpu' and self.max_decisions == MAX_DECISIONS,
              'session and development scope differ')

    def _execute_validated_step(self, step, **kwargs):
        self.verify_frozen()
        return super()._execute_validated_step(step, **kwargs)

    def _move(self, action, **kwargs):
        self.verify_frozen()
        before, task = scene_features.transfer_task(self.current, action)
        case_id = _case_id(before, task['source_slot'], task['destination_slot'])
        if case_id not in self._owner._transition_ids:
            raise UnsupportedStep('transfer is outside the explicitly allowed development catalog')
        commit = _VerifiedCommitBoundary(self, kwargs.pop('commit_lock', None),
                                         kwargs.pop('on_transfer_commit', None), before, task,
                                         len(self.events), len(self._physical_checkpoints))
        raw_command = self.sim.command
        raw_save = self.sim.save_state
        raw_begin = self.sim.begin_transfer

        def checked_begin(*args, **options):
            self.verify_frozen()
            result = raw_begin(*args, **options)
            object.__setattr__(self._owner, '_transfer_started', True)
            self.verify_frozen()
            return result

        def checked_command(*args, **options):
            self.verify_frozen()
            result = raw_command(*args, **options)
            self.verify_frozen()
            return result

        def checked_save(*args, **options):
            self.verify_frozen()
            result = raw_save(*args, **options)
            self.verify_frozen()
            return result

        # Mirror the evaluated live boundary: the student sees robot state and
        # expected scene only, never object/contact observations or assessment.
        with ExitStack() as boundary:
            for name in ('_objects', '_contacts', '_assessment_sample', 'evaluate_transfer', 'reset', 'load_state'):
                boundary.enter_context(patch.object(self.sim, name, new=public_boundary.forbidden))
            boundary.enter_context(patch.object(self.sim, 'observation',
                                                new=partial(public_boundary.public_command_return, self.sim)))
            boundary.enter_context(patch.object(self.sim, 'command', new=checked_command))
            boundary.enter_context(patch.object(self.sim, 'save_state', new=checked_save))
            boundary.enter_context(patch.object(self.sim, 'begin_transfer', new=checked_begin))
            return super()._move(action, commit_lock=commit, on_transfer_commit=commit.mark_commit, **kwargs)


class _Terra552Coordinator(OracleFreeSceneProgramCoordinator):
    def recover_registration(self, request_id, registration_scene):
        self.session.verify_frozen()
        return super().recover_registration(request_id, registration_scene)


class Terra552SimulationRuntime:
    """One frozen student, one pinned memory and at most one virtual session."""

    _LOCKED = frozenset({'contract', '_student', '_memory_model', 'memory', 'guard',
                         '_execution_groups', '_transition_ids', '_binding_locked',
                         '_sim', '_sim_binding', '_bound_methods', '_source_hashes',
                         '_original_guard_forward', '_checked_guard_forward', '_transfer_started',
                         'session', 'coordinator'})

    def __setattr__(self, name, value):
        if getattr(self, '_binding_locked', False) and name in self._LOCKED:
            raise AttributeError('the Terra runtime binding is immutable')
        super().__setattr__(name, value)

    def __init__(self, student, memory_model, contract):
        _need(type(contract) is Terra552Contract and contract == self._selected_contract(),
              'runtime contract metadata or execution scope differs')
        self._student, self._memory_model, self.contract = student, memory_model, contract
        self.guard = registered.make_guard(student)
        self.memory = self._make_memory(memory_model)
        self._source_hashes = tuple((path, direct.digest(path)) for path in self._execution_files())
        self._original_guard_forward = self.guard.forward_with_control_state

        def checked_guard_forward(*args, **options):
            self.verify()
            if self._sim is not None:
                self._verify_simulator()
            result = self._original_guard_forward(*args, **options)
            self.verify()
            if self._sim is not None:
                self._verify_simulator()
            return result

        self._checked_guard_forward = checked_guard_forward
        self.guard.forward_with_control_state = checked_guard_forward
        catalog = task_catalog.one_step_cases()
        self._transition_ids = frozenset(contract.development_transition_ids)
        self._execution_groups = frozenset(scene_policy.split_group_for(c['canonical_scene_before'], c['task'])
                                           for c in catalog if c['case_id'] in self._transition_ids)
        self._bind_lock = threading.Lock()
        self._sim = self._sim_binding = self._bound_methods = self.session = self.coordinator = None
        self._transfer_started = False
        self._binding_locked = True
        self.verify()

    def _selected_contract(self):
        return _expected_contract()

    def _make_memory(self, model):
        return _VerifiedSceneRecall(model, self)

    def _execution_files(self):
        return _EXECUTION_FILES

    def _memory_checkpoint(self):
        return MEMORY, MEMORY_SHA256

    def _expected_memory_modules(self):
        return (('', scene_recall.SceneMemory), ('encoder', torch.nn.GRUCell), ('decoder', torch.nn.Linear))

    def _verify_memory(self):
        _need(type(self._memory_model) is scene_recall.SceneMemory and self._memory_model.hidden_size == 128
              and type(self.memory) is _VerifiedSceneRecall and self.memory._owner is self
              and getattr(self.memory.observe, '__func__', None) is _VerifiedSceneRecall.observe
              and getattr(self.memory.recall, '__func__', None) is _VerifiedSceneRecall.recall
              and getattr(self._memory_model.advance, '__func__', None) is _GRU_ADVANCE
              and getattr(self._memory_model.decode, '__func__', None) is _GRU_DECODE,
              'the selected GRU128 memory is required')

    def _make_session(self, sim, *, cancel):
        return _Terra552Session(sim, self, cancel=cancel)

    def verify(self):
        """Check pinned artifacts, loaded tensors and the one correction/guard chain."""
        _need(self.contract == self._selected_contract(), 'runtime contract metadata or execution scope differs')
        for path, expected in ((STUDENT, STUDENT_SHA256), (FIT, FIT_SHA256), self._memory_checkpoint(),
                               (BASE_REPORT, BASE_REPORT_SHA256),
                               (original.PARENT / 'result.json', original.PARENT_REPORT_SHA),
                               (original.INDEX, original.INDEX_SHA)):
            _need(direct.digest(path) == expected, f'runtime input fingerprint differs: {path}')
        for path, expected in self._source_hashes:
            _need(direct.digest(path) == expected, f'execution source fingerprint differs: {path}')
        _need(direct.digest(historical_provenance.HISTORICAL_SESSION)
              == historical_provenance.HISTORICAL_SESSION_SHA256,
              'historical training session source fingerprint differs')
        guard = self.guard
        _need(type(guard) is traced.TracedGuard and type(guard.policy) is registered.RegisteredGraspPlacement
              and type(guard.policy.reference) is traced.StudentProbe
              and guard.policy.reference.student is self._student
              and type(self._student) is scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy
              and type(self._student).forward is _STUDENT_FORWARD
              and getattr(self._student.forward, '__func__', None) is _STUDENT_FORWARD
              and type(guard._guard) is NearJointLimitPolicy and guard._guard.model is guard._dispatch
              and type(guard)._call is _TRACED_CALL and type(guard._guard).forward is _GUARD_FORWARD
              and getattr(guard._guard.forward, '__func__', None) is _GUARD_FORWARD
              and guard.forward_with_control_state is self._checked_guard_forward
              and near_joint_limit_policy.MAX_OVERSHOOT_DEG == near_joint_limit_guard.MAX_OVERSHOOT_DEG == 1.0
              and self.memory.model is self._memory_model,
              'one registered correction, one one-degree guard, and the selected memory are required')
        _need(_VerifiedSceneRecall._verified_call is _VERIFIED_MEMORY_CALL
              and getattr(self.memory._verified_call, '__func__', None) is _VERIFIED_MEMORY_CALL,
              'memory verification call differs from the bound implementation')
        self._verify_memory()
        torch_module = torch.nn.modules.module
        _need(scene_recall.SceneRecall.observe is _RECALL_OBSERVE
              and scene_recall.SceneRecall.recall is _RECALL_READ
              and scene_recall.SceneRecall.clear is _RECALL_CLEAR
              and not torch_module._global_forward_hooks and not torch_module._global_forward_pre_hooks
              and all(not module._forward_hooks and not module._forward_pre_hooks
                      and not module._state_dict_hooks and not module._state_dict_pre_hooks
                      for model in (guard, self._memory_model) for module in model.modules()),
              'inference or state-dictionary hooks are not permitted on bound models')
        policy = guard.policy
        _need(type(policy).forward_with_control_state is _REGISTERED_FORWARD
              and type(policy)._forward_in_frame is _REGISTERED_FRAME
              and getattr(policy.forward_with_control_state, '__func__', None) is _REGISTERED_FORWARD
              and getattr(policy._forward_in_frame, '__func__', None) is _REGISTERED_FRAME
              and scene_placement_belief_policy.GraspPlacementBeliefCanonicalPolicy.forward_with_control_state
                  is _GRASP_FORWARD
              and scene_placement_belief_policy.PlacementBeliefCanonicalPolicy.forward_with_control_state
                  is _PLACEMENT_FORWARD
              and scene_belief_canonical_policy.BeliefCanonicalPolicy.forward_with_control_state is _BELIEF_FORWARD
              and traced.GRASP_TCP_ABOVE_CENTER_MM == policy.grasp_tcp_above_center_mm == 5.0
              and policy.grasp_placement_enabled is True and policy.joint_limited_pitch is True
              and policy.belief.center_lateral_on_close is True and policy.belief.object_height_mm == 20.0
              and torch.equal(policy.reference_slots,
                              torch.tensor(((150., 150.), (150., 0.), (-150., 150.), (-150., 0.)),
                                           dtype=torch.float64)), 'registered correction configuration differs')
        frozen = self._student.frozen_policy
        base = frozen.base
        expected_student_modules = (
            ('', scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy),
            ('frozen_policy', scene_current_observation_correction.SceneCurrentObservationCorrectionPolicy),
            ('frozen_policy.base', scene_no_moving_policy.SceneNoMovingPolicy),
            ('frozen_policy.base.encoder', torch.nn.Sequential),
            ('frozen_policy.base.encoder.0', torch.nn.Linear),
            ('frozen_policy.base.encoder.1', torch.nn.ReLU),
            ('frozen_policy.base.encoder.2', torch.nn.Linear),
            ('frozen_policy.base.encoder.3', torch.nn.ReLU),
            ('frozen_policy.base.action_head', torch.nn.Linear),
            ('frozen_policy.base.joint_head', torch.nn.Linear),
            ('frozen_policy.correction_head', torch.nn.Linear),
            ('late_head', torch.nn.Sequential), ('late_head.0', torch.nn.Linear),
            ('late_head.1', torch.nn.ReLU), ('late_head.2', torch.nn.Linear))
        expected_memory_modules = self._expected_memory_modules()
        student_modules = tuple(self._student.named_modules())
        memory_modules = tuple(self._memory_model.named_modules())
        _need(tuple((name, type(module)) for name, module in student_modules) == expected_student_modules
              and tuple((name, type(module)) for name, module in memory_modules) == expected_memory_modules
              and all(type(module).forward is _LAYER_FORWARDS[type(module)]
                      and getattr(module.forward, '__func__', None) is _LAYER_FORWARDS[type(module)]
                      for _, module in student_modules + memory_modules if type(module) in _LAYER_FORWARDS),
              'student or memory layer types or inference methods differ')
        current_columns = tuple(index for index in range(82) if index != 5)
        kept_columns = tuple(index for index in range(8 * 83)
                             if index not in {frame * 83 + 5 for frame in range(8)})
        _need(type(frozen) is scene_current_observation_correction.SceneCurrentObservationCorrectionPolicy
              and type(base) is scene_no_moving_policy.SceneNoMovingPolicy
              and type(self._student).diagnostics is _STUDENT_DIAGNOSTICS
              and getattr(self._student.diagnostics, '__func__', None) is _STUDENT_DIAGNOSTICS
              and type(frozen)._frozen_outputs is _FROZEN_OUTPUTS
              and getattr(frozen._frozen_outputs, '__func__', None) is _FROZEN_OUTPUTS
              and type(base)._encode_with_lengths is _ENCODE_LENGTHS
              and getattr(base._encode_with_lengths, '__func__', None) is _ENCODE_LENGTHS
              and scene_current_late_nonlinear.MIN_LATE_HISTORY == 4
              and torch.equal(frozen._current_columns, torch.tensor(current_columns, dtype=torch.long))
              and torch.equal(base._kept_columns, torch.tensor(kept_columns, dtype=torch.long)),
              'student inference components or input projection differ')
        _need(all(not module.training for model in (guard, self._memory_model) for module in model.modules())
              and all(not parameter.requires_grad for model in (guard, self._memory_model)
                      for parameter in model.parameters())
              and direct.bc._weights_fingerprint(self._student)['all'] == STUDENT_STATE_SHA256
              and full_teacher_training.state_fingerprint(self._memory_model)['sha256']
                  == self.contract.memory_state_sha256,
              'loaded student or memory tensors differ')
        _need(self._transition_ids == frozenset(self.contract.development_transition_ids)
              == frozenset(case['case_id'] for case in task_catalog.one_step_cases())
              and self._execution_groups == frozenset(self.contract.checkpoint_train_groups)
              and len(self._transition_ids) == 276 and len(self._execution_groups) == 15,
              'development transition scope changed')

    def _verify_simulator(self):
        sim, fixed = self._sim, self._sim_binding
        _need(type(sim) is CustomSlotsSceneMotionSim and fixed is not None
              and sim.model is fixed.model and sim.data is fixed.data
              and sim.slot_xy_mm == fixed.slot_xy_mm
              and development.evaluator.record_sha(sim._reset_info) == fixed.reset_info_sha256
              and sim.model_parameters_sha256 == fixed.model_parameters_sha256
              and development.evaluator.record_sha(sim._model_parameter_snapshot()) == fixed.model_parameters_sha256
              and (sim.ramp_seconds, sim.threshold_degrees, sim.ramp_shape, sim.model.opt.timestep)
                  == (.5, 0., 'minimum_jerk', .002),
              'bound development simulator or control profile changed')
        if not self._transfer_started:
            _need(_physical_start_sha(CustomSlotsSceneMotionSim.save_state(sim))
                  == fixed.initial_physical_state_sha256,
                  'bound development initial state changed before first transfer')

    def describe(self):
        return {**asdict(self.contract),
                'student_path': str(STUDENT), 'memory_path': str(MEMORY),
                'training_provenance': '1,000 direct originals and subsequent derived-weight lineage',
                'historical_training_session_source': {
                    'path': str(historical_provenance.HISTORICAL_SESSION),
                    'sha256': historical_provenance.HISTORICAL_SESSION_SHA256,
                    'scope': 'recorded_training_provenance_only'},
                'execution_source_sha256': {path.relative_to(ROOT).as_posix(): digest
                                            for path, digest in self._source_hashes},
                'memory_calibration': 'development-repaired GRU128',
                'registered_nominal_correction_count': 1, 'one_degree_guard_count': 1,
                'model_hook_policy': 'forward and state-dictionary hooks are not permitted',
                'final_evaluation': 'requires separately unused physical starts, coordinates and memory histories',
                'physical_confirmation': 'not_inspected'}

    def development_spec(self, index):
        if type(index) is not int or not 0 <= index < 552:
            raise ValueError('select one registered development index from 0 through 551')
        specs, _ = _development_starts()
        return copy.deepcopy(specs[index])

    def bind(self, sim, *, development_index, cancel=None):
        """Attach to one unchanged registered development simulator, without advancing it."""
        with self._bind_lock:
            self.verify()
            if self.session is not None or self.memory.has_memory:
                raise RuntimeError('load a fresh runtime for each session')
            if type(development_index) is not int or not 0 <= development_index < 552:
                raise ValueError('a registered development index from 0 through 551 is required')
            if type(sim) is not CustomSlotsSceneMotionSim:
                raise TypeError('only the custom-slot virtual simulator is accepted')
            specs, hashes = _development_starts()
            spec = specs[development_index]
            slots = tuple(map(tuple, spec['registered_slot_xy_mm']))
            _need(sim.slot_xy_mm == slots and sim._reset_info['seed'] == spec['physics_seed']
                  and sim._reset_info['randomize'] is True
                  and task_catalog.scene(sim._reset_info['initial_layout']) == task_catalog.scene(spec['initial_scene'])
                  and (sim.ramp_seconds, sim.threshold_degrees, sim.ramp_shape, sim.model.opt.timestep)
                      == (.5, 0., 'minimum_jerk', .002), 'simulator is not the selected development start')
            initial_state = sim.save_state()
            initial_sha = development.evaluator.record_sha(initial_state)
            _need(initial_sha == hashes[development_index], 'simulator state differs from the registered fresh start')
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            with _SIMULATOR_OWNERS_LOCK:
                current_owner = _SIMULATOR_OWNERS.get(sim)
                if current_owner is not None and current_owner() is not None:
                    raise RuntimeError('the simulator already belongs to another runtime')
                _SIMULATOR_OWNERS[sim] = weakref.ref(self)
            object.__setattr__(self, '_sim', sim)
            object.__setattr__(self, '_sim_binding', _SimulatorBinding(
                sim.model, sim.data, development.evaluator.record_sha(sim._reset_info),
                sim.model_parameters_sha256, sim.slot_xy_mm, initial_sha, _physical_start_sha(initial_state),
                development_index))
            methods = ExitStack()
            try:
                for name in ('reset', 'load_state'):
                    methods.enter_context(patch.object(sim, name, new=public_boundary.forbidden))
                session = self._make_session(sim, cancel=cancel)
                coordinator = _Terra552Coordinator(session)
                _need(development.evaluator.record_sha(sim.save_state()) == initial_sha,
                      'binding unexpectedly changed the simulation state')
            except BaseException:
                methods.close()
                object.__setattr__(self, '_sim', None)
                object.__setattr__(self, '_sim_binding', None)
                with _SIMULATOR_OWNERS_LOCK:
                    if (current_owner := _SIMULATOR_OWNERS.get(sim)) is not None and current_owner() is self:
                        del _SIMULATOR_OWNERS[sim]
                raise
            object.__setattr__(self, '_bound_methods', methods)
            object.__setattr__(self, 'session', session)
            object.__setattr__(self, 'coordinator', coordinator)
            return coordinator


def _load_student_and_contract():
    """Load the same pinned student and independently validate its training scope."""
    fit = direct.pinned_json(FIT, FIT_SHA256)
    student, binding = development.load_terra_candidate(STUDENT, STUDENT_SHA256)
    _need(binding['student_state_sha256'] == fit['binding']['state_sha256'] == STUDENT_STATE_SHA256
          and fit['binding']['checkpoint_sha256'] == STUDENT_SHA256
          and fit['binding']['parent_sha256'] == PARENT_SHA256,
          'student checkpoint and development fit differ')
    groups, transitions = _training_and_transition_scope(binding)
    contract = Terra552Contract(STUDENT_SHA256, STUDENT_STATE_SHA256, MEMORY_SHA256, MEMORY_STATE_SHA256,
                                original.PARENT_REPORT_SHA, original.INDEX_SHA, groups, transitions, SCHEDULE_SHA256)
    return student, contract


def load_terra552_simulation_runtime():
    """Load the exact frozen checkpoints without executing a model or simulator."""
    student, contract = _load_student_and_contract()
    memory_bytes = MEMORY.read_bytes()
    _need(direct.sha_bytes(memory_bytes) == MEMORY_SHA256, 'corrected GRU checkpoint differs')
    memory = scene_recall.load_scene_memory(io.BytesIO(memory_bytes), device='cpu').eval().requires_grad_(False)
    return Terra552SimulationRuntime(student, memory, contract)
