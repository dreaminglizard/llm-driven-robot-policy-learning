"""Project-specific completion head over a frozen official BC-RNN/GMM core.

No device access, model loading, simulation, or training occurs on import.
The official action is sampled once per observation; completion discards it.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
import re

import torch
from torch import nn

from raccoon_memory.robomimic_state_observer import OfficialRNNStateObserver


OBS_KEY = "scene"
OBS_SIZE = 82
HIDDEN_SIZE = 400
INPUT_SIZE = OBS_SIZE + HIDDEN_SIZE
CONTINUE, DONE = 0, 1
COMPLETION_SCHEMA = "raccoon-robomimic-completion-head.v1"
OFFICIAL_COMMIT = "d309eaecc18acf4152a830a895a6984b8ac71b05"
OFFICIAL_SOURCE_HASHES = {
    "robomimic.algo.bc": "aad11dedf0a564fd11e57c488e8d013957f0c84caf9668f553babec6a04e066c",
    "robomimic.models.policy_nets": "1ae6d327ec3b6ab1e2dc7d6f7fabfd34913006a93701358f24852f5223e7be88",
    "robomimic.models.obs_nets": "5c785067e06cebe25af0db850fbf501f9045a1188a0199068230bf1567392c7d",
    "robomimic.models.base_nets": "50ff90dbf4742a6bcc7b0a90e7058269d9e346e25df7e11524580eb04a1f5a3f",
    "robomimic.algo.algo": "9c058ac36d1b068895daf4156f36115b110c1035ec16b8167ab4ef97a46c2c39",
    "robomimic.config.bc_config": "bd5c5416ca711cb229bcf4392e24232c9188736e8fda0faca141b7b5e350d919",
    "robomimic.utils.obs_utils": "a7e14c05f46d751fd8c0f0a93801191548e90a19e3909b91322f61bbadf9b4a7",
}


def check_official_source() -> dict[str, str]:
    """Reject a different version of the official private-state contract."""
    from robomimic.algo import algo, bc
    from robomimic.config import bc_config
    from robomimic.models import base_nets, obs_nets, policy_nets
    from robomimic.utils import obs_utils

    found = {}
    for module in (bc, policy_nets, obs_nets, base_nets, algo, bc_config, obs_utils):
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != OFFICIAL_SOURCE_HASHES[module.__name__]:
            raise ValueError(f"official source fingerprint differs: {module.__name__}")
        found[module.__name__] = digest
    return found


def core_fingerprint(algorithm) -> str:
    """Hash all core parameters and buffers, including their types and names."""
    digest = hashlib.sha256()
    for name, value in sorted(algorithm.nets.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_fingerprint(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompletionHead(nn.Module):
    """Use only current 82 features and the post-observation top-layer h."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUT_SIZE, 64), nn.ReLU(), nn.Linear(64, 2))
        self._bound_core_sha256 = None
        self._bound_core_checkpoint_sha256 = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                or features.ndim != 2 or features.shape[1] != INPUT_SIZE
                or not torch.isfinite(features).all()):
            raise ValueError("completion input must be a finite [batch,482] tensor")
        # Never allow completion loss to backpropagate into the action core.
        logits = self.net(features.detach())
        if not torch.isfinite(logits).all():
            raise ValueError("completion logits are not finite")
        return logits


class FrozenOfficialCore:
    """Own the observer and the reset boundary for one frozen core instance."""

    def __init__(self, algorithm, *, checkpoint_provenance: dict | None = None, synthetic_test: bool = False):
        if type(synthetic_test) is not bool:
            raise TypeError("synthetic_test must be a boolean")
        if not synthetic_test:
            raise ValueError("real action cores require a verified trained scene-core checkpoint")
        if checkpoint_provenance is not None:
            raise ValueError("synthetic checks must not claim trained-checkpoint provenance")
        self._initialize(algorithm, checkpoint_provenance=None, synthetic_test=True, checkpoint_file_sha256=None)

    def _initialize(self, algorithm, *, checkpoint_provenance, synthetic_test, checkpoint_file_sha256):
        if not synthetic_test and (not isinstance(checkpoint_provenance, dict)
              or checkpoint_provenance.get("schema") != "raccoon-robomimic-scene-core.v1"
              or checkpoint_provenance.get("official_commit") != OFFICIAL_COMMIT
              or type(checkpoint_provenance.get("training_seed")) is not int
              or type(checkpoint_provenance.get("optimizer_updates")) is not int
              or checkpoint_provenance["optimizer_updates"] <= 0
              or any(not isinstance(checkpoint_provenance.get(k), str)
                     or re.fullmatch(r"[0-9a-f]{64}", checkpoint_provenance[k]) is None
                     for k in ("dataset_sha256", "training_spec_sha256"))):
            raise ValueError("a trained scene-core checkpoint provenance with completed updates is required")
        self.synthetic_test = synthetic_test
        self.checkpoint_provenance = copy.deepcopy(checkpoint_provenance)
        self.checkpoint_file_sha256 = checkpoint_file_sha256
        self.source_hashes = check_official_source()
        self.algorithm = algorithm
        # This verifies the exact official BC_RNN_GMM type and closed-loop LSTM.
        self.observer = OfficialRNNStateObserver(algorithm)
        shapes = algorithm.obs_key_shapes
        lstms = [m for m in algorithm.nets.modules() if isinstance(m, nn.LSTM)]
        if (set(shapes) != {OBS_KEY} or tuple(shapes[OBS_KEY]) != (OBS_SIZE,) or algorithm.ac_dim != 5
                or algorithm._rnn_horizon != 10 or len(lstms) != 1
                or lstms[0].hidden_size != HIDDEN_SIZE or lstms[0].num_layers != 2 or lstms[0].bidirectional):
            raise ValueError("official core must use scene[82], action[5], LSTM 400x2 and horizon 10")
        algorithm.set_eval()
        for parameter in algorithm.nets.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.weights_sha256 = core_fingerprint(algorithm)
        self._needs_reset = True

    @classmethod
    def from_trained_scene_checkpoint(cls, path, *, expected_spec_sha256: str,
                                      expected_dataset_sha256: str, device="cpu"):
        """Bind to the other adapter's official checkpoint and require training."""
        from raccoon_memory.robomimic_scene_checkpoint import load_scene_core_checkpoint

        if any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
               for v in (expected_spec_sha256, expected_dataset_sha256)):
            raise ValueError("expected action-core dataset and training specification hashes are required")
        checkpoint_file_sha256 = _file_fingerprint(path)
        algorithm, _config, provenance = load_scene_core_checkpoint(
            path, device=device, load_optimizers=True, expected_spec_sha256=expected_spec_sha256,
            expected_dataset_sha256=expected_dataset_sha256)
        if _file_fingerprint(path) != checkpoint_file_sha256:
            raise ValueError("scene-core checkpoint changed while it was being restored")
        expected_updates = provenance.get("optimizer_updates")
        if type(expected_updates) is not int or expected_updates <= 0:
            raise ValueError("a trained scene-core checkpoint provenance with completed updates is required")
        optimizer = algorithm.optimizers.get("policy")
        if not isinstance(optimizer, torch.optim.Adam):
            raise ValueError("official action-core Adam state is missing")
        steps = set()
        for state in optimizer.state.values():
            value = state.get("step")
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError("official action-core Adam step is invalid")
            step = float(value.item())
            if step != int(step):
                raise ValueError("official action-core Adam step is not integral")
            steps.add(int(step))
        if steps != {expected_updates}:
            raise ValueError("checkpoint update count differs from official Adam state")
        restored_weights = core_fingerprint(algorithm)
        core = cls.__new__(cls)
        core._initialize(algorithm, checkpoint_provenance=provenance, synthetic_test=False,
                         checkpoint_file_sha256=checkpoint_file_sha256)
        if core.weights_sha256 != restored_weights:
            raise RuntimeError("restored action-core weights changed during initialization")
        return core

    def assert_frozen(self, *, check_weights: bool = True) -> None:
        if (self.algorithm.nets.training or self.algorithm._rnn_is_open_loop is not False
                or any(p.requires_grad or p.grad is not None for p in self.algorithm.nets.parameters())):
            raise RuntimeError("the official action core must remain eval, frozen and without gradients")
        if check_weights and core_fingerprint(self.algorithm) != self.weights_sha256:
            raise RuntimeError("frozen official core weights or buffers changed")

    def reset_episode(self) -> None:
        self.assert_frozen()
        self._needs_reset = True
        self.observer.reset()  # Official reset, once at each caller-owned boundary.
        self._needs_reset = False

    def _observe_once(self, observation: torch.Tensor):
        if self._needs_reset:
            raise RuntimeError("reset the official core before a new episode or after an error")
        self.assert_frozen(check_weights=False)
        try:
            if (not isinstance(observation, torch.Tensor) or not observation.is_floating_point()
                    or tuple(observation.shape) != (1, OBS_SIZE) or not torch.isfinite(observation).all()):
                raise ValueError("current observation must be a finite [1,82] tensor")
            current = observation.detach().to(device=self.algorithm.device, dtype=torch.float32).clone()
            result = self.observer.step({OBS_KEY: current})
            if (tuple(result.hidden.shape) != (2, 1, HIDDEN_SIZE)
                    or tuple(result.cell.shape) != (2, 1, HIDDEN_SIZE)
                    or tuple(result.sampled_action.shape) != (1, 5)
                    or not torch.isfinite(result.hidden).all() or not torch.isfinite(result.cell).all()):
                raise RuntimeError("official post-observation state or action shape differs")
            features = torch.cat((current, result.hidden[-1].to(current.device)), dim=1).detach()
            return features, result.sampled_action.detach(), result.decision_count
        except Exception:
            # Official get_action may already have advanced hidden state or RNG.
            self._needs_reset = True
            raise


@dataclass(frozen=True)
class CompletionDecision:
    """An unapplied action proposal, present only for continue."""

    kind: str
    proposed_action: torch.Tensor | None
    decision_count: int


def bind_head(core: FrozenOfficialCore, head: CompletionHead) -> None:
    """Prevent reusing a completion head with different action-core weights."""
    if not isinstance(core, FrozenOfficialCore) or not isinstance(head, CompletionHead):
        raise TypeError("a frozen core and project completion head are required")
    core.assert_frozen()
    if head._bound_core_sha256 is not None and (
            head._bound_core_sha256 != core.weights_sha256
            or head._bound_core_checkpoint_sha256 != core.checkpoint_file_sha256):
        raise ValueError("completion head belongs to different action-core weights or checkpoint")
    head._bound_core_sha256 = core.weights_sha256
    head._bound_core_checkpoint_sha256 = core.checkpoint_file_sha256


def completion_checkpoint(core: FrozenOfficialCore, head: CompletionHead) -> dict:
    """Build a portable in-memory payload, bound to the exact frozen core."""
    bind_head(core, head)
    state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("completion head weights are not finite")
    return {"schema": COMPLETION_SCHEMA, "core_weights_sha256": core.weights_sha256,
            "core_checkpoint_sha256": core.checkpoint_file_sha256,
            "official_source_hashes": dict(core.source_hashes), "class_names": ["continue", "done"],
            "input_size": INPUT_SIZE, "hidden_size": 64, "head_state": state}


def restore_completion_head(core: FrozenOfficialCore, payload: dict) -> CompletionHead:
    """Reject a mismatched core, architecture, source, or nonfinite weights."""
    core.assert_frozen()
    if (not isinstance(payload, dict) or payload.get("schema") != COMPLETION_SCHEMA
            or payload.get("core_weights_sha256") != core.weights_sha256
            or "core_checkpoint_sha256" not in payload
            or payload["core_checkpoint_sha256"] != core.checkpoint_file_sha256
            or payload.get("official_source_hashes") != core.source_hashes
            or payload.get("class_names") != ["continue", "done"]
            or payload.get("input_size") != INPUT_SIZE or payload.get("hidden_size") != 64
            or not isinstance(payload.get("head_state"), dict)):
        raise ValueError("completion checkpoint contract or action core differs")
    head = CompletionHead()
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in payload["head_state"].values()):
        raise ValueError("completion checkpoint contains invalid weights")
    head.load_state_dict(payload["head_state"], strict=True)
    bind_head(core, head)
    head.eval()
    return head


class CompletionGate:
    """Run the official core once and suppress its sample when done is chosen."""

    def __init__(self, core: FrozenOfficialCore, head: CompletionHead):
        if not isinstance(core, FrozenOfficialCore) or not isinstance(head, CompletionHead):
            raise TypeError("completion gate requires its frozen core and project head")
        bind_head(core, head)
        self.core, self.head = core, head

    def reset_episode(self) -> None:
        self.core.reset_episode()

    def decide(self, observation: torch.Tensor) -> CompletionDecision:
        if self.head.training:
            raise RuntimeError("completion head must be in evaluation mode")
        try:
            features, sampled_action, count = self.core._observe_once(observation)
            device = next(self.head.parameters()).device
            with torch.no_grad():
                logits = self.head(features.to(device=device))
            choice = int(logits.argmax(dim=1).item())
            if choice == DONE:
                # No sampled action is returned or passed to a command sink.
                self.core._needs_reset = True
                return CompletionDecision("done", None, count)
            if not torch.isfinite(sampled_action).all():
                raise ValueError("official action proposal is not finite")
            return CompletionDecision("continue", sampled_action.clone(), count)
        except Exception:
            self.core._needs_reset = True
            raise
