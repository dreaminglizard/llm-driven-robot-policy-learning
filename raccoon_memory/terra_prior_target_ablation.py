"""Unconnected, simulation-research ablation of a prior servo target.

The caller must supply a full, from-scratch Terra BC1020 candidate and the
binding returned by the existing ``verify_candidate``. This module checks the
binding's contract and the supplied weights' fingerprint, but cannot establish
that an arbitrary caller-created binding was independently verified.

Only the existing causal 8x82 student window, its mask, and (where required)
the student's own confirmed *pre-decision* joint control are accepted. Source
labels assert the caller's verification; they are not proof of it. In
particular, a current label, unconfirmed proposal, future observation, actual
object pose, and post-hoc assessment are not valid inputs. The shape checks do
not establish the provenance of a window or of a supplied control tensor.

This is a small, freshly initialized adapter on a frozen independent copy of
the baseline, not an independently trained-from-scratch whole model. It neither
loads data, trains, evaluates, nor connects to an executor or a device. An
external experiment must record its own additional optimization budget.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re
from typing import NamedTuple

import torch
from torch import nn

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import scene_current_late_nonlinear, scene_fk_loss


CONDITIONS = ("zero", "prior_target", "prior_target_fk")
CONTROL_SOURCES = ("validated_initial_servo", "confirmed_command_return")
HISTORY_LENGTH = 8
FEATURE_COUNT = 82
JOINT_SCALE_DEG = 180.0
FK_SCALE_MM = 300.0
CONTEXT_COUNT = 7
ADAPTER_INPUT_COUNT = 19
ADAPTER_HIDDEN_COUNT = 16
ADAPTER_OUTPUT_COUNT = 8
ADAPTER_PARAMETER_COUNT = 456
FORMAT = "terra-prior-target-ablation.v1"


@dataclass(frozen=True, slots=True)
class PriorControl:
    """Caller-verified pre-decision targets and their per-row source labels.

    ``joints_deg`` is float32 [batch,4], before division by 180. Each source
    must be either a validated initial servo setting or a confirmed command
    return. This label does not itself verify the target; the caller must bind
    it to the current student's own control history. Unknown targets are None,
    never a fabricated zero vector.
    """

    joints_deg: torch.Tensor
    sources: tuple[str, ...]


class AblationOutputs(NamedTuple):
    logits: torch.Tensor
    joints: torch.Tensor
    baseline_logits: torch.Tensor
    baseline_joints: torch.Tensor
    context: torch.Tensor
    correction: torch.Tensor


def _weights_fingerprint(model: nn.Module) -> str:
    """Match the existing Terra evaluator's all-state fingerprint."""
    state = model.state_dict()
    if not state:
        raise ValueError("baseline must have a nonempty state")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError("baseline state must be float32 tensors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("baseline state must be finite")
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _binding_summary(binding: dict, baseline: nn.Module) -> dict:
    if not isinstance(binding, dict):
        raise ValueError("a caller-verified Terra evaluation binding is required")
    if (binding.get("schema") != "terra-bc1020-evaluation-binding.v1"
            or binding.get("training_recipe") != "terra_from_scratch_bc1020"
            or type(binding.get("originals")) is not int or binding["originals"] != 1000
            or "selection_seed" not in binding or binding["selection_seed"] is not None):
        raise ValueError("binding must describe the full 1000-original from-scratch baseline")
    seed = binding.get("training_seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("baseline training seed is invalid")
    keys = ("report_sha256", "index_sha256", "checkpoint_sha256", "weights_sha256")
    if any(not isinstance(binding.get(key), str)
           or not re.fullmatch(r"[0-9a-f]{64}", binding[key]) for key in keys):
        raise ValueError("baseline binding must contain lowercase SHA-256 digests")
    if _weights_fingerprint(baseline) != binding["weights_sha256"]:
        raise ValueError("baseline weights differ from the supplied binding")
    return {key: binding[key] for key in ("schema", "training_recipe", "originals",
                                           "selection_seed", "training_seed", *keys)}


class TerraPriorTargetAblation(nn.Module):
    """One 19-to-16-to-8 adapter with one of three fixed input conditions.

    The 12 shared inputs are last observed joints / 180, baseline action
    logits, and baseline joint outputs / 180. Seven additional inputs are zero,
    previous applied joints / 180 plus three zeros, or the same joints plus
    nominal FK XYZ / 300 mm. FK is recomputed from the float32 degree tensor
    with the fixed public geometry; it is never supplied by the caller. Thus
    training preprocessing must cast degrees to float32 *before* both the
    division and FK calculation. This may differ slightly from the teacher's
    double-precision displayed FK.

    The eight outputs add four logits and four normalized absolute joint
    corrections. The baseline's own output gate is evaluated before this
    adapter; it is not rerun if the adapter changes the chosen action. All
    conditions use this same order and architecture. No joint guard is included.
    """

    def __init__(self, baseline: scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy,
                 binding: dict, *, condition: str, initialization_seed: int):
        super().__init__()
        if type(baseline) is not scene_current_late_nonlinear.SceneCurrentLateNonlinearPolicy:
            raise ValueError("the existing BC1020 late-correction baseline is required")
        if condition not in CONDITIONS:
            raise ValueError("unknown prior-target ablation condition")
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**32:
            raise ValueError("adapter initialization seed must be an integer in [0, 2**32)")
        parameters = tuple(baseline.parameters())
        if (not parameters or any(p.dtype != torch.float32 for p in parameters)
                or len({p.device for p in parameters}) != 1):
            raise ValueError("baseline parameters must be float32 on one device")
        self._baseline_binding = _binding_summary(binding, baseline)
        self._condition = condition
        self.initialization_seed = initialization_seed
        self.history_length, self.feature_count = HISTORY_LENGTH, FEATURE_COUNT
        self.baseline = copy.deepcopy(baseline).requires_grad_(False).eval()
        self.assert_baseline_unchanged()
        # Preserve the caller's RNG and give every condition the same fresh
        # first layer and a zero-output second layer for a shared start.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.adapter = nn.Sequential(
                nn.Linear(ADAPTER_INPUT_COUNT, ADAPTER_HIDDEN_COUNT, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(ADAPTER_HIDDEN_COUNT, ADAPTER_OUTPUT_COUNT, dtype=torch.float32),
            )
            nn.init.zeros_(self.adapter[2].weight)
            nn.init.zeros_(self.adapter[2].bias)
        self.adapter.to(device=parameters[0].device)
        if sum(p.numel() for p in self.adapter.parameters()) != ADAPTER_PARAMETER_COUNT:
            raise ValueError("adapter size differs from the comparison contract")

    @property
    def condition(self) -> str:
        return self._condition

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self.baseline.requires_grad_(False)
        return self

    def assert_baseline_unchanged(self) -> None:
        """Recheck the frozen copy against the caller-supplied binding.

        Call before and after an external training run. This verifies a weight
        fingerprint, not the authenticity of the original training evidence.
        """
        if any(parameter.requires_grad for parameter in self.baseline.parameters()):
            raise ValueError("baseline parameters must remain frozen")
        if _weights_fingerprint(self.baseline) != self._baseline_binding["weights_sha256"]:
            raise ValueError("frozen baseline weights differ from the supplied binding")

    def experiment_contract(self) -> dict:
        """Describe the separate adapter; optimization progress is not tracked."""
        self.assert_baseline_unchanged()
        return {
            "format": FORMAT,
            "condition": self.condition,
            "baseline_binding": dict(self._baseline_binding),
            "baseline_provenance": "caller_must_use_independently_verified_binding",
            "baseline_weights_frozen": True,
            "adapter_initialization_seed": self.initialization_seed,
            "adapter_initialization": "fresh_seeded_first_linear_zero_output_linear",
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
            "additional_optimizer_updates": None,
            "additional_training_record": "not_tracked_caller_must_record",
            "fk_preprocessing": "float32_degrees_then_nominal_fk_xyz_div300",
        }

    def _validate_window(self, windows: torch.Tensor, masks: torch.Tensor) -> None:
        parameter = next(self.adapter.parameters())
        if self.condition not in CONDITIONS:
            raise ValueError("unknown prior-target ablation condition")
        if (not isinstance(windows, torch.Tensor) or windows.dtype != torch.float32
                or windows.ndim != 3 or tuple(windows.shape[1:]) != (HISTORY_LENGTH, FEATURE_COUNT)
                or windows.shape[0] < 1 or windows.device != parameter.device):
            raise ValueError("windows must be float32 [batch,8,82] on the model device")
        if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                or tuple(masks.shape) != tuple(windows.shape[:2]) or masks.device != windows.device):
            raise ValueError("masks must be bool [batch,8] on the window device")
        if not bool(torch.isfinite(windows).all()):
            raise ValueError("window values must be finite")
        if not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any()):
            raise ValueError("masks must be left-padded and contain the current observation")
        if any(p.dtype != torch.float32 or p.device != windows.device for p in self.baseline.parameters()):
            raise ValueError("baseline must remain float32 on the model device")
        if any(p.dtype != torch.float32 or p.device != windows.device for p in self.adapter.parameters()):
            raise ValueError("adapter must remain float32 on the model device")
        if any(p.requires_grad for p in self.baseline.parameters()):
            raise ValueError("baseline parameters must remain frozen")
        if any(m.training for m in self.baseline.modules()):
            raise ValueError("baseline must remain in evaluation mode")

    def _context(self, windows: torch.Tensor, prior: PriorControl | None) -> torch.Tensor:
        batch = windows.shape[0]
        if prior is None:
            if self.condition != "zero":
                raise ValueError("confirmed prior control is required for this condition")
            return windows.new_zeros((batch, CONTEXT_COUNT))
        if not isinstance(prior, PriorControl):
            raise ValueError("prior control must carry caller-verified source labels")
        joints = prior.joints_deg
        if (not isinstance(joints, torch.Tensor) or joints.dtype != torch.float32
                or tuple(joints.shape) != (batch, 4) or joints.device != windows.device):
            raise ValueError("prior target must be float32 [batch,4] on the window device")
        if (not isinstance(prior.sources, tuple) or len(prior.sources) != batch
                or any(source not in CONTROL_SOURCES for source in prior.sources)):
            raise ValueError("each prior target needs a known confirmed control source")
        if not bool(torch.isfinite(joints).all()):
            raise ValueError("prior target must be finite")
        low = joints.new_tensor([limits[0] for limits in JOINT_LIMITS])
        high = joints.new_tensor([limits[1] for limits in JOINT_LIMITS])
        if bool(((joints < low) | (joints > high)).any()):
            raise ValueError("prior target is outside the joint limits")
        if self.condition == "zero":
            return windows.new_zeros((batch, CONTEXT_COUNT))
        joints = joints.detach()
        normalized = joints / JOINT_SCALE_DEG
        if self.condition == "prior_target":
            return torch.cat((normalized, windows.new_zeros((batch, 3))), dim=1)
        xyz, _ = scene_fk_loss.nominal_forward(joints, Geometry())
        return torch.cat((normalized, xyz / FK_SCALE_MM), dim=1)

    def diagnostics(self, windows: torch.Tensor, masks: torch.Tensor,
                    prior: PriorControl | None = None) -> AblationOutputs:
        self._validate_window(windows, masks)
        context = self._context(windows, prior)
        with torch.no_grad():
            baseline_logits, baseline_joints = self.baseline(windows, masks)
        batch = windows.shape[0]
        if any(not isinstance(output, torch.Tensor) or tuple(output.shape) != (batch, 4)
               or output.dtype != torch.float32 or output.device != windows.device
               or not bool(torch.isfinite(output).all()) for output in (baseline_logits, baseline_joints)):
            raise ValueError("baseline outputs must be finite float32 [batch,4]")
        inputs = torch.cat((windows[:, -1, :4], baseline_logits, baseline_joints, context), dim=1)
        correction = self.adapter(inputs)
        logits, joints = baseline_logits + correction[:, :4], baseline_joints + correction[:, 4:]
        if not bool(torch.isfinite(correction).all() and torch.isfinite(logits).all()
                    and torch.isfinite(joints).all()):
            raise ValueError("adapter outputs must be finite")
        return AblationOutputs(logits, joints, baseline_logits, baseline_joints, context, correction)

    def forward(self, windows: torch.Tensor, masks: torch.Tensor,
                prior: PriorControl | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.diagnostics(windows, masks, prior)
        return output.logits, output.joints
