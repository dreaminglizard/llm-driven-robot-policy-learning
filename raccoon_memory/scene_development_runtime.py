"""Load the qualified simulation student before constructing a virtual scene.

``load_development_runtime().bind(sim)`` returns the simulation v2 coordinator
with explicit user-declared registration recovery after an interrupted request.
Each loaded runtime owns one student, one GRU memory and at most one session.
The fixed profile identifies the public-development candidate, not a final or
hardware qualification. Loading and binding do not run a policy or a movement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import threading

import torch

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import full_teacher_training, scene_placement_belief_policy, scene_recall
from .scene_oracle_free_session import OracleFreeSceneModelSession, OracleFreeSceneProgramCoordinator
from .scene_ramped_motion_sim import RampedSceneMotionSim

ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path("configs/scene_development_runtime.v1.json")
PROFILE_SHA256 = "f4dbdc02c28af1ba219554e0e4cb843c9330788cbeb271bbeaa9745add14c83e"
_EXPOSED_GROUPS = frozenset({"n4_s1_d2_other11", "n4_s2_d1_other11"})


def _checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"runtime input fingerprint differs: {path}")
    return data


def _artifact(root, relative):
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("runtime artifacts must be inside the selected project")
    return resolved


def _state_sha(model):
    return full_teacher_training.state_fingerprint(model)["sha256"]


class _DevelopmentSession(OracleFreeSceneModelSession):
    def __init__(self, sim, memory, student, *, trained, allowed, cancel):
        super().__init__(sim, memory, student, device="cpu", max_decisions=60,
                         checkpoint_train_groups=trained, cancel=cancel)
        # These two catalog groups were already exposed in development. They
        # are execution scope, not a claim that they belonged to training.
        self.train_groups = frozenset(allowed)


@dataclass(frozen=True)
class RuntimeIdentity:
    candidate_id: str
    profile_sha256: str
    student_path: str
    student_checkpoint_sha256: str
    student_loaded_state_sha256: str
    memory_path: str
    memory_checkpoint_sha256: str
    memory_loaded_state_sha256: str
    command_source_files_match_qualified_version: bool
    qualification: str = "public_development_only"
    actual_hardware: bool = False


class DevelopmentRuntime:
    """An explicitly selected, single-session simulation runtime."""

    def __init__(self, root, spec, split, student, memory_model):
        self._root, self._spec = root, spec
        self._student = student
        self.student = GuardedPolicy(student)
        self.memory = scene_recall.SceneRecall(memory_model)
        self._memory_model = memory_model
        self._trained = frozenset(spec["checkpoint_train_groups"])
        self._allowed = frozenset(split["train_groups"] + split["validation_groups"]) | _EXPOSED_GROUPS
        self._bind_lock = threading.Lock()
        self.session = self.coordinator = None
        self.identity = RuntimeIdentity(
            candidate_id=spec["candidate_id"], profile_sha256=PROFILE_SHA256,
            student_path=str(_artifact(root, spec["student"]["path"])),
            student_checkpoint_sha256=spec["student"]["sha256"],
            student_loaded_state_sha256=spec["student"]["loaded_state_sha256"],
            memory_path=str(_artifact(root, spec["memory"]["path"])),
            memory_checkpoint_sha256=spec["memory"]["sha256"],
            memory_loaded_state_sha256=spec["memory"]["loaded_state_sha256"],
            command_source_files_match_qualified_version=all(
                hashlib.sha256(_artifact(root, name).read_bytes()).hexdigest() == digest
                for name, digest in spec["qualified_command_sources"].items()))

    def describe(self):
        return asdict(self.identity)

    def verify(self):
        """Check the selected files and the actual loaded tensors."""
        _checked_bytes(self._root / PROFILE, PROFILE_SHA256)
        for name in ("student", "memory", "split"):
            item = self._spec[name]
            _checked_bytes(_artifact(self._root, item["path"]), item["sha256"])
        for name, expected in self._spec["required_assets"].items():
            _checked_bytes(_artifact(self._root, name), expected)
        if (type(self.student) is not GuardedPolicy or self.student.policy is not self._student or
                type(self._student) is not scene_placement_belief_policy.GraspPlacementBeliefCanonicalPolicy or
                self.memory.model is not self._memory_model or self.student.training or
                any(m.training for m in self._student.modules()) or
                any(m.training for m in self._memory_model.modules()) or
                any(p.requires_grad for m in (self._student, self._memory_model) for p in m.parameters())):
            raise ValueError("runtime models must remain the selected frozen evaluation models")
        for name, model in (("student", self._student), ("memory", self._memory_model)):
            if _state_sha(model) != self._spec[name]["loaded_state_sha256"]:
                raise ValueError(f"loaded {name} state differs from the selected profile")

    def bind(self, sim, *, cancel=None):
        """Attach once to an existing matching simulator without moving it."""
        with self._bind_lock:
            return self._bind_once(sim, cancel=cancel)

    def _bind_once(self, sim, *, cancel):
        self.verify()
        if self.session is not None or self.memory.has_memory:
            raise RuntimeError("load a fresh runtime for a new session")
        if not isinstance(sim, RampedSceneMotionSim):
            raise TypeError("the development runtime requires the virtual ramped scene")
        execution = self._spec["execution"]
        values = ((sim.ramp_seconds, execution["ramp_seconds"]),
                  (sim.threshold_degrees, execution["ramp_threshold_degrees"]),
                  (sim.model.opt.timestep, execution["physics_timestep_seconds"]))
        if (sim.ramp_shape != execution["ramp_shape"] or
                any(not math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12) for a, b in values)):
            raise ValueError("simulator control profile differs from the selected runtime")
        # Match the CPU execution mode used by the qualified development runs.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        session = _DevelopmentSession(sim, self.memory, self.student, trained=self._trained,
                                      allowed=self._allowed, cancel=cancel)
        coordinator = OracleFreeSceneProgramCoordinator(session)
        self.session, self.coordinator = session, coordinator
        return coordinator


def load_development_runtime(*, root=ROOT):
    """Load exactly the pinned 03 student and GRU128; never read score files.

    ``root`` only locates the same checked-in profile and artifacts in a project
    copy. Selecting another candidate requires a new explicit profile version.
    """
    root = Path(root).resolve(strict=True)
    spec = json.loads(_checked_bytes(root / PROFILE, PROFILE_SHA256))
    split = json.loads(_checked_bytes(_artifact(root, spec["split"]["path"]), spec["split"]["sha256"]))
    trained = frozenset(spec["checkpoint_train_groups"])
    if (not trained or not trained.issubset(split["train_groups"]) or
            frozenset(spec["previously_exposed_development_groups"]) != _EXPOSED_GROUPS or
            not _EXPOSED_GROUPS.issubset(split["test_groups"])):
        raise ValueError("the declared training and exposed development scope differs")
    student_item, memory_item = spec["student"], spec["memory"]
    student_bytes = _checked_bytes(_artifact(root, student_item["path"]), student_item["sha256"])
    memory_bytes = _checked_bytes(_artifact(root, memory_item["path"]), memory_item["sha256"])
    for name, expected in spec["required_assets"].items():
        json.loads(_checked_bytes(_artifact(root, name), expected))
    # Deserialize the exact checked bytes; a later path change cannot substitute
    # a different checkpoint between the digest check and torch.load.
    payload = torch.load(io.BytesIO(student_bytes), map_location="cpu", weights_only=True)
    student = scene_placement_belief_policy.load_checkpoint(payload, device="cpu").requires_grad_(False)
    memory_model = scene_recall.load_scene_memory(io.BytesIO(memory_bytes), device="cpu").requires_grad_(False)
    if memory_model.hidden_size != memory_item["hidden_size"]:
        raise ValueError("loaded memory width differs from the selected profile")
    runtime = DevelopmentRuntime(root, spec, split, student, memory_model)
    runtime.verify()
    return runtime
