"""Inject the official completion gate into an existing simulation session.

No checkpoint, simulator, GUI or device is created on import.  The bridge
preserves the evaluation runner's result and only owns its concurrency guard.
"""
from __future__ import annotations

import threading
import weakref

import torch

from raccoon_memory import robomimic_scene_evaluation
from raccoon_memory.robomimic_completion import CompletionGate


_registry_lock = threading.Lock()
_algorithm_guards = weakref.WeakKeyDictionary()


def _guard_for(algorithm):
    # Different CompletionGate instances may still own the same LSTM object.
    with _registry_lock:
        guard = _algorithm_guards.get(algorithm)
        if guard is None:
            guard = threading.Lock()
            _algorithm_guards[algorithm] = guard
        return guard


def run_official_transfer(gate, sim, before, task, *, max_decisions=24, device="cpu", cancel=None):
    """Match SceneModelSession's runner contract using the existing evaluator.

    The caller must separately verify the frozen checkpoints and create the
    session, gate and simulation. A concurrent call using the same underlying
    action core fails before the evaluator observes or moves either scene.
    """
    if not isinstance(gate, CompletionGate):
        raise TypeError("a bound official CompletionGate is required")
    if torch.device(device).type != "cpu":
        raise ValueError("the official scene bridge requires the CPU device")
    algorithm = gate.core.algorithm
    guard = _guard_for(algorithm)
    if not guard.acquire(blocking=False):
        raise RuntimeError("the official action core is already executing another transfer")
    try:
        if (torch.device(algorithm.device).type != "cpu"
                or any(parameter.device.type != "cpu" for parameter in algorithm.nets.parameters())
                or any(buffer.device.type != "cpu" for buffer in algorithm.nets.buffers())
                or any(parameter.device.type != "cpu" for parameter in gate.head.parameters())
                or any(buffer.device.type != "cpu" for buffer in gate.head.buffers())):
            raise ValueError("the official gate must be on the CPU")
        gate.core.assert_frozen()
        if gate.head.training:
            raise RuntimeError("the completion head must be in evaluation mode")
        return robomimic_scene_evaluation.run_transfer(
            gate, sim, before, task, max_decisions=max_decisions, cancel=cancel)
    finally:
        guard.release()
