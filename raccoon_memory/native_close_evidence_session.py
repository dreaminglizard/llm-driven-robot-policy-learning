"""Optional CLOSE-evidence research candidate in the existing virtual v2 session.

The standard source-approach loader remains unchanged. This loader uses the
same public simulator, applied-control history, one guard, GRU and executor.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import weakref

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import native_approach_session as base
from . import scene_recall, terra_native_close_evidence as candidate

STUDENT = (base.ROOT / 'artifacts/architecture_research/astra_native_close_evidence_20261106_v1'
           / 'distance_and_action/candidate.pt')
STUDENT_SHA256 = '071c6d1bbdf6cd0128ad7586c82f91e9bbaf8163c18ce3a2829d91db05f3ba7c'
BEST_DEVELOPMENT_STUDENT = (base.ROOT / 'artifacts/architecture_research/astra_close_reproduction_three_seeds_20261106_v1'
                            / 'seed_20261030/candidate.pt')
BEST_DEVELOPMENT_STUDENT_SHA256 = '48170fc8aa8e05b231115367fb2960f9a8f69920c359cf3019472d5c911d3977'
MEMORY = base.MEMORY
MEMORY_SHA256 = base.MEMORY_SHA256
PublicCustomSlotsSceneMotionSim = base.PublicCustomSlotsSceneMotionSim


def load_session(sim, *, cancel=None, student_path=STUDENT, memory_path=MEMORY):
    """Bind this exact candidate and the existing GRU to one virtual session."""
    if not isinstance(sim, PublicCustomSlotsSceneMotionSim):
        raise TypeError('a public-only virtual simulator is required')
    with base._SIMULATOR_OWNERS_LOCK:
        owner = base._SIMULATOR_OWNERS.get(sim)
        if owner is not None and owner() is not None:
            raise RuntimeError('this simulator is already bound to a live session')
    student_bytes, memory_bytes = Path(student_path).read_bytes(), Path(memory_path).read_bytes()
    base._need(hashlib.sha256(student_bytes).hexdigest() in
               (STUDENT_SHA256, BEST_DEVELOPMENT_STUDENT_SHA256), 'student checkpoint differs')
    base._need(hashlib.sha256(memory_bytes).hexdigest() == MEMORY_SHA256, 'GRU checkpoint differs')
    student = candidate.load_student_checkpoint(io.BytesIO(student_bytes), device='cpu').requires_grad_(False)
    memory_model = scene_recall.load_scene_memory(io.BytesIO(memory_bytes), device='cpu').requires_grad_(False)
    base._need(memory_model.hidden_size == 128, 'expected the GRU128 memory')
    guard = GuardedPolicy(base.AppliedHistoryPolicy(student)).eval()
    split, _, _ = base.runtime.scene_demo_data.read_split_plan(base.runtime.SPLIT_PLAN)
    allowed = set(split['train_groups']) | set(split['validation_groups'])
    session = base.runtime.OracleFreeSceneModelSession(sim, scene_recall.SceneRecall(memory_model), guard,
        max_decisions=60, development_groups=allowed, slot_xy_mm=sim.slot_xy_mm, cancel=cancel,
        unscored_transfer_runner=base.run_native_transfer)
    session.checkpoint_train_groups = None
    coordinator = base.runtime.OracleFreeSceneProgramCoordinator(session)
    with base._SIMULATOR_OWNERS_LOCK:
        owner = base._SIMULATOR_OWNERS.get(sim)
        if owner is not None and owner() is not None:
            raise RuntimeError('this simulator is already bound to a live session')
        base._SIMULATOR_OWNERS[sim] = weakref.ref(session)
    return coordinator
