"""Optional development-only shared-slot memory for the fixed Terra 552 session.

This does not select or replace the default repaired-GRU runtime. The student,
start conditions, nominal correction, guard, single owner and interruption
semantics are inherited from the same simulation-only runtime.
"""
from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path

import torch

from . import scene_recall
from . import terra552_simulation_runtime as shared
from scripts import astra_memory_slot_write_hold as slots
from scripts import astra_memory_write_hold as write_hold

ROOT = shared.ROOT
MEMORY_ROOT = (shared.ARTIFACTS / 'astra_terra_weight_repair_v1/learned_write_hold_memory_552_v1')
CHECKPOINT = MEMORY_ROOT / 'shared_slots32/model.pt'
CHECKPOINT_SHA256 = '8ca4648569a4c6feb801a1eb981edd99e8179dad112045e5f79e4554df5057d0'
STATE_SHA256 = '7237666e7abc40a6c171385743f114a415e4f70812a11931c37568bc194145eb'
TRAINING_REPORT = MEMORY_ROOT / 'shared_slots32/training.json'
TRAINING_REPORT_SHA256 = '17e082eb608a1965687c7a612f03a3bbd74fee577630070c0dcbf221d9cb9898'
CORPUS = MEMORY_ROOT / 'corpus.json'
CORPUS_SHA256 = '7947b4e01e8ff837a09c371efef18e1622230eb2521f21bded1c6b7313517e8c'
SLOT_SOURCE = Path(slots.__file__).resolve()
SLOT_SOURCE_SHA256 = 'a22a568c0b3bbbc497dc424be5a864cb81753af084dee31f411fd8273f3222ce'
HOLD_SOURCE = Path(write_hold.__file__).resolve()
HOLD_SOURCE_SHA256 = '8927e61de335679b925ce3c8ec37b00a59b4c06c76912e05f026107ca569cede'
MEMORY_ARCHITECTURE = 'learned_shared_slot_write_hold_32'
RECALL_SOURCE = 'decoded_shared_slot_latent'
_ADVANCE = write_hold.WriteHoldMemory.advance
_DECODE = write_hold.WriteHoldMemory.decode
_ENCODER_FORWARD = slots.EncodeStacks.forward
_DECODER_FORWARD = slots.DecodeStacks.forward


def _verified_provenance():
    report = shared.direct.pinned_json(TRAINING_REPORT, TRAINING_REPORT_SHA256)
    corpus = shared.direct.pinned_json(CORPUS, CORPUS_SHA256)
    shared._need(report['architecture'] == 'shared_slots32' and report['model_sha256'] == CHECKPOINT_SHA256
                 and report['data_sha256'] == CORPUS_SHA256 and report['fixed_motion_sha256'] == shared.STUDENT_SHA256
                 and report['source_sha256'] == SLOT_SOURCE_SHA256 and report['prior_runtime'] == HOLD_SOURCE_SHA256
                 and report['parameter_count'] == 1746 and report['state_dimension'] == 32
                 and report['updates'] == 1200 and report['batch_size'] == 64 and report['rate'] == .001
                 and report['actual_hardware'] is False and report['final_holdout'] is False
                 and corpus['training_histories'] == 76800 and corpus['observations'] == 2515590
                 and corpus['write_events'] == 115262 and corpus['overwrite_events'] == 38462
                 and corpus['final_holdout'] is False and corpus['physical_robot'] is False,
                 'shared-slot memory training provenance differs')
    return report, corpus


class _SlotSceneRecall(shared._VerifiedSceneRecall):
    """Use the shared checked recall with a distinct, exact candidate type."""


class _SlotEvents(list):
    """Name the actual learned reader at the moment a recall event is recorded."""

    def append(self, event):
        if isinstance(event, dict) and event.get('event') == 'recall':
            if event.get('goal_source') != 'decoded_gru_hidden_state':
                raise ValueError('the inherited recall event contract changed')
            event['goal_source'] = RECALL_SOURCE
            event['memory_architecture'] = MEMORY_ARCHITECTURE
            event['memory_checkpoint_sha256'] = CHECKPOINT_SHA256
        super().append(event)


class _SlotSession(shared._Terra552Session):
    _LOCKED = shared._Terra552Session._LOCKED | frozenset({'events', '_slot_events_ready'})

    def __init__(self, sim, owner, *, cancel=None):
        super().__init__(sim, owner, cancel=cancel)
        object.__setattr__(self, 'events', _SlotEvents(self.events))
        object.__setattr__(self, '_slot_events_ready', True)
        self.verify_frozen()

    def verify_frozen(self):
        super().verify_frozen()
        if getattr(self, '_slot_events_ready', False):
            shared._need(type(self.events) is _SlotEvents, 'shared-slot recall event labeling changed')


class Terra552SlotMemoryRuntime(shared.Terra552SimulationRuntime):
    """The existing guarded session with exactly the learned shared-slot memory."""

    def _selected_contract(self):
        return replace(shared._expected_contract(), memory_checkpoint_sha256=CHECKPOINT_SHA256,
                       memory_state_sha256=STATE_SHA256)

    def _make_memory(self, model):
        return _SlotSceneRecall(model, self)

    def _execution_files(self):
        return shared._EXECUTION_FILES + (Path(__file__).resolve(), SLOT_SOURCE, HOLD_SOURCE)

    def _memory_checkpoint(self):
        return CHECKPOINT, CHECKPOINT_SHA256

    def _expected_memory_modules(self):
        return (('', slots.SharedStackMemory), ('encoder', slots.EncodeStacks),
                ('encoder.intermediate', torch.nn.Linear), ('encoder.local', torch.nn.Linear),
                ('encoder.pooled', torch.nn.Linear), ('decoder', slots.DecodeStacks),
                ('decoder.local', torch.nn.Linear), ('decoder.pooled', torch.nn.Linear),
                ('decoder.readout', torch.nn.Linear))

    def _verify_memory(self):
        model = self._memory_model
        shared._need(type(model) is slots.SharedStackMemory and model.hidden_size == 32
                     and type(self.memory) is _SlotSceneRecall and self.memory._owner is self
                     and getattr(self.memory.observe, '__func__', None) is _SlotSceneRecall.observe
                     and getattr(self.memory.recall, '__func__', None) is _SlotSceneRecall.recall
                     and type(model.encoder) is slots.EncodeStacks and type(model.decoder) is slots.DecodeStacks
                     and getattr(model.advance, '__func__', None) is _ADVANCE
                     and getattr(model.decode, '__func__', None) is _DECODE
                     and type(model.encoder).forward is _ENCODER_FORWARD
                     and getattr(model.encoder.forward, '__func__', None) is _ENCODER_FORWARD
                     and type(model.decoder).forward is _DECODER_FORWARD
                     and getattr(model.decoder.forward, '__func__', None) is _DECODER_FORWARD
                     and write_hold.DIM == 32 and scene_recall.INPUT_SIZE == 41
                     and scene_recall.TOKEN_COUNT == 8 and scene_recall.VOCAB_SIZE == 5,
                     'the selected learned shared-slot memory is required')
        for path, expected in ((TRAINING_REPORT, TRAINING_REPORT_SHA256), (CORPUS, CORPUS_SHA256),
                               (SLOT_SOURCE, SLOT_SOURCE_SHA256), (HOLD_SOURCE, HOLD_SOURCE_SHA256)):
            shared._need(shared.direct.digest(path) == expected, f'shared-slot source fingerprint differs: {path}')

    def _make_session(self, sim, *, cancel):
        return _SlotSession(sim, self, cancel=cancel)

    def describe(self):
        result = super().describe()
        result.update(memory_path=str(CHECKPOINT), memory_architecture=MEMORY_ARCHITECTURE,
                      memory_calibration='development-selected learned shared-slot write/hold model',
                      memory_training_report=str(TRAINING_REPORT), memory_training_report_sha256=TRAINING_REPORT_SHA256,
                      memory_training_corpus=str(CORPUS), memory_training_corpus_sha256=CORPUS_SHA256,
                      memory_code_sha256={str(SLOT_SOURCE): SLOT_SOURCE_SHA256,
                                          str(HOLD_SOURCE): HOLD_SOURCE_SHA256},
                      memory_training_scope=('76,800 historical symbolic training histories; '
                                             'not physical demonstrations or joined robot trajectories'),
                      recall_event_source=RECALL_SOURCE,
                      legacy_recall_log_compatibility=('legacy scripts filtering the GRU-specific label '
                                                       'must explicitly accept this candidate label'),
                      qualification='optional_development_candidate_no_prior_score_inherited')
        return result


def load_terra552_slot_memory_runtime():
    """Load the pinned student and learned slot memory without advancing a scene."""
    _verified_provenance()
    student, contract = shared._load_student_and_contract()
    raw = CHECKPOINT.read_bytes()
    shared._need(shared.direct.sha_bytes(raw) == CHECKPOINT_SHA256, 'shared-slot checkpoint differs')
    model = slots.load(io.BytesIO(raw))
    contract = replace(contract, memory_checkpoint_sha256=CHECKPOINT_SHA256, memory_state_sha256=STATE_SHA256)
    return Terra552SlotMemoryRuntime(student, model, contract)
