"""Explicit 15-group adapter for 1103 with the reviewed current v2 overlay.

This module must be loaded alongside the code extracted from the pinned
development ZIP. It changes the execution allowlist only: the checkpoint's
eight recorded training groups, policy, original OPEN, guard and memory remain
those of that package. Current v2 command sources are bound separately.
Binding does not move a simulator.
This is a candidate backend, not a statement that an evaluation has passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import torch

from . import scene_development_runtime as preserved, scene_program_coordinator, task_program, task_program_v2
from .scene_oracle_free_session import OracleFreeSceneModelSession, OracleFreeSceneProgramCoordinator
from .scene_ramped_motion_sim import RampedSceneMotionSim

ARCHIVE_SHA256 = "a7db4dee2b2d5d1fbc06a823bbe01caafb17d649a5ad2a19557857004abb510d"
SPLIT_SHA256 = "13b568c881b042bdf85c94c952db77cb02f145ba6e89056cc046c65f97a27aa7"
CANDIDATE_SHA256 = "92041ff58ac4465b7dee95472ea7d482c9d1bb445e69d1cae66f6f9322eb012d"
CURRENT_COMMAND_OVERLAY = {
    "raccoon_memory/task_program.py": "b8f2741ecbb1da72e91ee8701b817916131153b49e774627d65700e5a9b4c811",
    "raccoon_memory/task_program_v2.py": "af7129307814594a77bfdd15bc67870c8e8c4b44bb711ffe6785b3974dfac4c1",
    "raccoon_memory/scene_program_coordinator.py": "82e702b7b76c5ecc833351f4c3c2e9174bc0f16caa4297ad83da5ab5d8511637",
    "configs/task_program.v2.prompt_detailed.v1.txt": "2261d263e1adb3b21a2d81a61535227d43bc8c94027409d219461e8f9e580196",
}


@dataclass(frozen=True, slots=True)
class FullScopeContract:
    checkpoint_train_groups: tuple[str, ...]
    execution_groups: tuple[str, ...]
    split_sha256: str
    archive_sha256: str = ARCHIVE_SHA256
    candidate_sha256: str = CANDIDATE_SHA256


def scope_contract(split, trained):
    """Bind the actual training groups separately from all public groups."""
    if not isinstance(split, dict):
        raise ValueError("split must be a mapping")
    parts = [split.get(k) for k in ("train_groups", "validation_groups", "test_groups")]
    if any(not isinstance(p, list) or any(type(x) is not str or not x for x in p)
           or len(set(p)) != len(p) for p in parts):
        raise ValueError("split groups must be distinct nonempty names")
    sets = [set(p) for p in parts]
    if tuple(map(len, parts)) != (8, 3, 4) or any(
        left & right for i, left in enumerate(sets) for right in sets[i + 1:]
    ):
        raise ValueError("the expected disjoint 8/3/4 groups are required")
    if not isinstance(trained, (list, tuple, set, frozenset)) or set(trained) != sets[0]:
        raise ValueError("checkpoint training groups must remain the eight recorded groups")
    return FullScopeContract(tuple(parts[0]), tuple(sum(parts, [])), SPLIT_SHA256)


class _FullScopeSession(OracleFreeSceneModelSession):
    """Preserved session with an explicit, separately verified execution scope."""

    _LOCKED = frozenset({"train_groups", "checkpoint_train_groups", "sim", "student", "memory",
                         "max_decisions", "device", "_fullscope_owner", "_scope_locked"})

    def __setattr__(self, name, value):
        if getattr(self, "_scope_locked", False) and name in self._LOCKED:
            raise AttributeError("full-scope session binding is immutable")
        super().__setattr__(name, value)

    def __init__(self, sim, owner, *, cancel=None):
        contract = owner.contract
        super().__init__(sim, owner._base.memory, owner._base.student, device="cpu", max_decisions=60,
                         checkpoint_train_groups=contract.checkpoint_train_groups, cancel=cancel)
        if self.checkpoint_train_groups != frozenset(contract.checkpoint_train_groups):
            raise ValueError("actual checkpoint training scope differs")
        self.train_groups = frozenset(contract.execution_groups)
        self._fullscope_owner = owner
        self._scope_locked = True
        self.verify_frozen()

    def verify_frozen(self):
        owner = self._fullscope_owner
        owner.verify()
        if (self.train_groups != frozenset(owner.contract.execution_groups)
                or self.checkpoint_train_groups != frozenset(owner.contract.checkpoint_train_groups)
                or self.plan_sha256 != owner.contract.split_sha256
                or self.student is not owner._base.student or self.memory is not owner._base.memory
                or self.max_decisions != 60 or self.device != "cpu"):
            raise ValueError("full-scope session and contract differ")

    def _register(self, value):
        self.verify_frozen()
        return super()._register(value)

    def _move(self, action, **kwargs):
        self.verify_frozen()
        return super()._move(action, **kwargs)

    def _execute_validated_step(self, step, **kwargs):
        self.verify_frozen()
        return super()._execute_validated_step(step, **kwargs)


class FullScopeRuntime:
    """One preserved v2 coordinator and one explicitly scoped virtual session."""

    def __setattr__(self, name, value):
        if getattr(self, "_binding_locked", False) and name in ("_base", "contract", "_binding_locked"):
            raise AttributeError("full-scope runtime contract is immutable")
        super().__setattr__(name, value)

    def __init__(self, base, contract):
        self._base, self.contract = base, contract
        self.session = self.coordinator = None
        self._binding_locked = True
        self.verify()

    def verify(self):
        import json
        self._base.verify()
        spec = self._base._spec
        split = json.loads((self._base._root / spec["split"]["path"]).read_bytes())
        if (spec["student"]["sha256"] != self.contract.candidate_sha256
                or spec["split"]["sha256"] != self.contract.split_sha256
                or tuple(spec["checkpoint_train_groups"]) != self.contract.checkpoint_train_groups
                or self._base._trained != frozenset(self.contract.checkpoint_train_groups)
                or scope_contract(split, spec["checkpoint_train_groups"]) != self.contract):
            raise ValueError("full-scope contract differs from preserved profile")
        for relative, expected in CURRENT_COMMAND_OVERLAY.items():
            if hashlib.sha256((self._base._root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("current v2 command overlay changed")
        for module, relative in ((task_program, "raccoon_memory/task_program.py"),
                                 (task_program_v2, "raccoon_memory/task_program_v2.py"),
                                 (scene_program_coordinator, "raccoon_memory/scene_program_coordinator.py")):
            if Path(module.__file__).resolve() != (self._base._root / relative).resolve():
                raise ValueError("current v2 command module was imported outside the bound package")
        if self._base.identity.command_source_files_match_qualified_version is not False:
            raise ValueError("the historical qualified command version must remain separately identified")

    def describe(self):
        return {"preserved_runtime": self._base.describe(), "full_scope_contract": asdict(self.contract),
                "current_command_overlay_sha256": dict(CURRENT_COMMAND_OVERLAY),
                "preserved_qualified_command_sources_match": False,
                "qualification": "candidate_only_no_prior_score_inherited"}

    def bind(self, sim, *, cancel=None):
        """Use this same method for prospective execution and later adoption."""
        with self._base._bind_lock:
            self.verify()
            if self.session is not None or self._base.session is not None or self._base.memory.has_memory:
                raise RuntimeError("load a fresh runtime for each session")
            if not isinstance(sim, RampedSceneMotionSim):
                raise TypeError("the preserved virtual ramped simulator is required")
            execution = self._base._spec["execution"]
            values = ((sim.ramp_seconds, execution["ramp_seconds"]),
                      (sim.threshold_degrees, execution["ramp_threshold_degrees"]),
                      (sim.model.opt.timestep, execution["physics_timestep_seconds"]))
            if (sim.ramp_shape != execution["ramp_shape"] or any(
                    not math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12) for a, b in values)):
                raise ValueError("simulator profile differs from the preserved runtime")
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            session = _FullScopeSession(sim, self, cancel=cancel)
            coordinator = OracleFreeSceneProgramCoordinator(session)
            self.session, self.coordinator = session, coordinator
            self._base.session, self._base.coordinator = session, coordinator
            return coordinator


def load_fullscope_runtime(*, package_root):
    """Load only from an isolated extracted copy of the pinned package."""
    root = Path(package_root).resolve(strict=True)
    if Path(preserved.__file__).resolve() != root / "raccoon_memory/scene_development_runtime.py":
        raise ValueError("preserved runtime must be imported from the isolated package")
    import json
    base = preserved.load_development_runtime(root=root)
    split = json.loads((root / base._spec["split"]["path"]).read_bytes())
    return FullScopeRuntime(base, scope_contract(split, base._spec["checkpoint_train_groups"]))
