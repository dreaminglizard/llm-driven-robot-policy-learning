"""Completion boundary for the separately checkpointed deterministic BC-RNN.

The action core returns one deterministic proposal per observation.  A done
decision discards that proposal and never issues a command.  Importing this
module does not load data, train a model, or access a simulator or device.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re

import torch

from raccoon_memory.robomimic_completion import (
    CompletionDecision, CompletionHead, DONE, FrozenOfficialCore, bind_head,
    check_official_source, core_fingerprint,
)
from raccoon_memory.robomimic_deterministic_checkpoint import RESUME_SCHEMA, restore_final
from raccoon_memory.robomimic_deterministic_core import (
    DeterministicRNNDecision, DeterministicRNNObserver, _require_deterministic_core,
)
from raccoon_memory.robomimic_scene_sequences import sha256_file
from raccoon_memory.robomimic_scene_training import model_weight_sha256, state_dict_equal


class _CompletionObserver(DeterministicRNNObserver):
    """Observe one official call without validating an action that may be discarded."""

    def step(self, observation):
        if self._needs_reset:
            raise RuntimeError("reset the deterministic recurrent core before a new episode or after an error")
        if self.model.nets.training or self.model._rnn_is_open_loop is not False:
            raise RuntimeError("deterministic recurrent core must be eval and closed-loop")
        try:
            if (not isinstance(observation, torch.Tensor) or not observation.is_floating_point()
                    or tuple(observation.shape) != (1, 82) or not torch.isfinite(observation).all()):
                raise ValueError("current observation must be a finite [1,82] tensor")
            if self.model._rnn_counter != self._counter:
                raise RuntimeError("deterministic recurrent core was advanced outside this observer")
            current = observation.detach().to(device=self.model.device, dtype=torch.float32)
            with torch.no_grad():
                action = self.model.get_action({"scene": current})
                state = self.model._rnn_hidden_state
            if (self.model._rnn_counter != self._counter + 1 or not isinstance(action, torch.Tensor)
                    or tuple(action.shape) != (1, 5) or not isinstance(state, tuple) or len(state) != 2
                    or any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (2, 1, 400)
                           or not torch.isfinite(value).all() for value in state)):
                raise RuntimeError("official deterministic proposal or post-observation state differs")
            self._counter += 1
            return DeterministicRNNDecision(action.detach().clone(), state[0].detach().clone(),
                                            state[1].detach().clone(), self._counter)
        except Exception:
            self._needs_reset = True
            raise


class FrozenDeterministicCore(FrozenOfficialCore):
    """Reuse frozen-core training interfaces with a deterministic observer."""

    def __init__(self, algorithm, *, synthetic_test=False):
        if synthetic_test is not True:
            raise ValueError("real deterministic cores require a verified final checkpoint")
        self._init_deterministic(algorithm, synthetic=True, provenance=None, checkpoint_sha=None)

    def _init_deterministic(self, algorithm, *, synthetic, provenance, checkpoint_sha):
        _require_deterministic_core(algorithm)
        if torch.device(algorithm.device).type != "cpu":
            raise ValueError("deterministic completion uses the fixed CPU core")
        self.synthetic_test = synthetic
        self.checkpoint_provenance = copy.deepcopy(provenance)
        self.checkpoint_file_sha256 = checkpoint_sha
        self.source_hashes = check_official_source()
        self.algorithm = algorithm
        self.observer = _CompletionObserver(algorithm)
        algorithm.set_eval()
        for parameter in algorithm.nets.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.weights_sha256 = core_fingerprint(algorithm)
        self._needs_reset = True

    @classmethod
    def from_final_checkpoint(cls, path, *, seed, training_spec_sha256, dataset_sha256):
        """Verify a published final and matching recovery before creating a real core.

        The training runner also verifies the fixed sample schedule and initial
        checkpoint before using this core for any project-data extraction.
        """
        import json

        if (type(seed) is not int or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                                         for value in (training_spec_sha256, dataset_sha256))):
            raise ValueError("seed and fixed training/data fingerprints are required")
        final_path = Path(path)
        if final_path.name != "final_deterministic_core.pt":
            raise ValueError("the published deterministic final checkpoint is required")
        manifest_path = final_path.parent / "final_manifest.json"
        recovery_path = final_path.parent / "checkpoints" / "step_020000.pt"
        manifest_sha = sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_sha, recovery_sha = sha256_file(final_path), sha256_file(recovery_path)
        if (manifest.get("seed") != seed or manifest.get("updates") != 20000
                or manifest.get("checkpoint") != final_path.name
                or manifest.get("checkpoint_sha256") != final_sha
                or manifest.get("recovery_checkpoint_sha256") != recovery_sha
                or manifest.get("training_config_sha256") != training_spec_sha256
                or manifest.get("selection") != "fixed final update, no performance selection"
                or type(manifest.get("valid_actions")) is not int or manifest["valid_actions"] <= 0):
            raise ValueError("deterministic final manifest or file fingerprint differs")
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        identity, cumulative = recovery.get("identity", {}), recovery.get("cumulative", {})
        if (recovery.get("schema") != RESUME_SCHEMA or recovery.get("updates_completed") != 20000
                or identity.get("seed") != seed or identity.get("training_config_sha256") != training_spec_sha256
                or identity.get("source_data_sha256") != dataset_sha256
                or cumulative.get("updates") != 20000 or cumulative.get("sampled_chunks") != 2000000
                or cumulative.get("valid_actions") != manifest["valid_actions"]
                or not state_dict_equal(recovery.get("model"), final.get("model"))
                or recovery.get("config") != final.get("config")):
            raise ValueError("deterministic final and recovery model, Adam or provenance differ")
        algorithm, _config, provenance = restore_final(
            final_path, training_spec_sha256=training_spec_sha256, dataset_sha256=dataset_sha256,
            seed=seed, updates=20000, device="cpu")
        if (provenance["checkpoint_sha256"] != final_sha
                or model_weight_sha256(algorithm) != recovery.get("model_weight_sha256")
                or sha256_file(final_path) != final_sha or sha256_file(recovery_path) != recovery_sha
                or sha256_file(manifest_path) != manifest_sha):
            raise ValueError("deterministic checkpoint changed or differs from the recovery point")
        before = core_fingerprint(algorithm)
        core = cls.__new__(cls)
        core._init_deterministic(algorithm, synthetic=False, provenance=provenance, checkpoint_sha=final_sha)
        if core.weights_sha256 != before:
            raise RuntimeError("deterministic core changed during freezing")
        core.assert_frozen()
        return core, {"seed": seed, "updates": 20000, "valid_actions": manifest["valid_actions"],
                      "final_manifest_sha256": manifest_sha, "final_checkpoint_sha256": final_sha,
                      "recovery_checkpoint_sha256": recovery_sha, "core_weights_sha256": core.weights_sha256}

    def _observe_once(self, observation):
        if self._needs_reset:
            raise RuntimeError("reset the deterministic core before a new episode or after an error")
        self.assert_frozen(check_weights=False)
        try:
            if (not isinstance(observation, torch.Tensor) or not observation.is_floating_point()
                    or tuple(observation.shape) != (1, 82) or not torch.isfinite(observation).all()):
                raise ValueError("current observation must be a finite [1,82] tensor")
            current = observation.detach().to(device=self.algorithm.device, dtype=torch.float32).clone()
            result = self.observer.step(current)
            if (tuple(result.hidden.shape) != (2, 1, 400) or tuple(result.cell.shape) != (2, 1, 400)
                    or tuple(result.action.shape) != (1, 5) or not torch.isfinite(result.hidden).all()
                    or not torch.isfinite(result.cell).all()):
                raise RuntimeError("deterministic proposal or post-observation state differs")
            features = torch.cat((current, result.hidden[-1].to(current.device)), dim=1).detach()
            return features, result.action.detach().clone(), result.decision_count
        except Exception:
            self._needs_reset = True
            raise


class DeterministicCompletionGate:
    """Use one core call; a done decision returns no action to a caller."""

    def __init__(self, core, head):
        if not isinstance(core, FrozenDeterministicCore) or not isinstance(head, CompletionHead):
            raise TypeError("a deterministic frozen core and completion head are required")
        bind_head(core, head)
        self.core, self.head = core, head

    def reset_episode(self):
        self.core.reset_episode()

    def decide(self, observation):
        if self.head.training:
            raise RuntimeError("completion head must be in evaluation mode")
        try:
            features, proposal, count = self.core._observe_once(observation)
            with torch.no_grad():
                logits = self.head(features.to(device=next(self.head.parameters()).device))
            if int(logits.argmax(dim=1).item()) == DONE:
                self.core._needs_reset = True
                return CompletionDecision("done", None, count)
            if not torch.isfinite(proposal).all():
                raise ValueError("deterministic action proposal is not finite")
            return CompletionDecision("continue", proposal.clone(), count)
        except Exception:
            self.core._needs_reset = True
            raise
