"""Offline prototype: compare MOVE labels at the guarded-return boundary.

Each evaluation copies one already-established public controller state.  This
module never advances a simulator, changes model weights, or infers an absent
historical correction peak.  A local numerical Jacobian bridges the existing
Python/IK implementation to autograd only when conservative branch and step-size
checks pass.  The result is a numerical local estimate, not a differentiability
proof or a deployed controller.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_control.near_joint_limit_guard import GuardedPolicy
from raccoon_memory import motion_policy
from raccoon_memory.scene_placement_belief_policy import GraspPlacementBeliefCanonicalPolicy


class UnknownPeak:
    """Fail if a correction attempts to consume an unavailable old peak."""

    def __init__(self, field):
        self.field = field

    def _raise(self, *_args):
        raise ValueError(f"unconfirmed peak read: {self.field}")

    __lt__ = __le__ = __gt__ = __ge__ = __float__ = __bool__ = _raise

    def __deepcopy__(self, _memo):
        return self


class DerivativeUnconfirmed(ValueError):
    """The return or a local derivative is not established at this point."""


class _FixedProposal(nn.Module):
    """Inject one normalized proposal and an explicitly selected action."""

    history_length, feature_count = 8, 82

    def __init__(self, q_degrees, action):
        super().__init__()
        # Normalize in float64 before the model's float32 boundary. This also
        # exactly recovers a float32 model output previously multiplied by 180.
        self.q = (q_degrees.detach().to(dtype=torch.float64).reshape(1, 4) / 180.0).to(dtype=torch.float32)
        self.action = int(action)

    def forward_with_control_state(self, windows, masks, previous):
        scores = self.q.new_zeros((1, 4))
        scores[0, self.action] = 1.0
        return scores, self.q


def _peak(value):
    if isinstance(value, UnknownPeak):
        return {"unconfirmed": value.field}
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        return {"unconfirmed": "invalid_or_unavailable_peak"}
    return float(value)


def _state(policy):
    return {"registered": policy._registered, "task": copy.deepcopy(policy._task),
            "source_anchor": copy.deepcopy(policy._source_anchor), "tracking_error": policy._tracking_error,
            "nominal_slots": copy.deepcopy(policy._nominal_slots), "last_action": policy._last_action,
            "approach_peak_mm": _peak(policy._approach_peak_mm), "closed_peak_mm": _peak(policy._closed_peak_mm),
            "grasp_started": policy._grasp_started, "belief": copy.deepcopy(policy.belief.__dict__)}


@dataclass(frozen=True)
class Step:
    status: str
    action: int
    returned_degrees: tuple[float, ...] | None
    guard_event: dict | None
    diagnostic: dict | None
    updated_state: dict
    detail: str | None = None


class AppliedTargetBoundary:
    """One saved input and one explicit pre-step correction state.

    ``origin`` must say where the state came from.  When it is reconstructed by
    replaying a fixed model on saved observations it is that replay's state,
    never an assertion about an earlier teacher's hidden peak.
    """

    def __init__(self, policy, window, mask, previous_target_degrees, *, origin):
        if not isinstance(policy, GraspPlacementBeliefCanonicalPolicy):
            raise ValueError("the existing grasp/placement policy is required")
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("an explicit state origin is required")
        if (not isinstance(window, torch.Tensor) or window.shape != (8, 82) or window.dtype != torch.float32 or
                not isinstance(mask, torch.Tensor) or mask.shape != (8,) or mask.dtype != torch.bool or
                not isinstance(previous_target_degrees, torch.Tensor) or previous_target_degrees.shape != (4,) or
                previous_target_degrees.dtype != torch.float32 or window.device.type != "cpu" or
                mask.device.type != "cpu" or previous_target_degrees.device.type != "cpu" or
                not bool(torch.isfinite(window).all() and torch.isfinite(previous_target_degrees).all())):
            raise ValueError("one finite CPU observation, mask, and previous target are required")
        self._snapshot = copy.deepcopy(policy)
        # A caller may carry a different unavailable-peak sentinel.  Preserve
        # that uncertainty without rejecting branches that never read it.
        for field in ("_approach_peak_mm", "_closed_peak_mm"):
            value = getattr(self._snapshot, field)
            if (value is not None and not isinstance(value, UnknownPeak) and
                    not (type(value) in (int, float) and math.isfinite(value))):
                setattr(self._snapshot, field, UnknownPeak("invalid_or_unavailable_peak"))
        self.window = window.detach().clone()
        self.mask = mask.detach().clone()
        self.previous = previous_target_degrees.detach().clone()
        self.origin = origin

    def state_before(self):
        return _state(self._snapshot)

    def evaluate(self, q_degrees, *, action):
        """Return value/rejection and updated *copy*; no command is applied."""
        if not isinstance(q_degrees, torch.Tensor) or q_degrees.shape != (4,) or q_degrees.device.type != "cpu":
            raise ValueError("one CPU four-joint proposal is required")
        if not bool(torch.isfinite(q_degrees).all()) or type(action) is not int or action not in range(4):
            raise ValueError("finite proposal and explicit action are required")
        policy = copy.deepcopy(self._snapshot)
        policy.reference = _FixedProposal(q_degrees, action)
        guard = GuardedPolicy(policy)
        try:
            with torch.inference_mode():
                logits, values = guard.forward_with_control_state(self.window[None], self.mask[None], self.previous[None])
            event = copy.deepcopy(guard.events[-1])
            diagnostic = copy.deepcopy(policy.last_diagnostic)
            if (logits.shape != (1, 4) or int(logits.argmax(1)) != action or values.shape != (1, 4) or
                    not bool(torch.isfinite(values).all())):
                return Step("rejected", action, None, event, diagnostic, _state(policy), "invalid guard output")
            if action != motion_policy.MOVE:
                return Step("non_move", action, None, event, diagnostic, _state(policy))
            if event.get("status") != "within_limits" or event.get("guard_returned_target_deg") is None:
                return Step("rejected", action, None, event, diagnostic, _state(policy), event.get("status"))
            result = tuple(float(v) * 180.0 for v in values[0].tolist())
            if any(not lo <= q <= hi for q, (lo, hi) in zip(result, JOINT_LIMITS)):
                return Step("rejected", action, None, event, diagnostic, _state(policy), "guard out of range")
            return Step("returned", action, result, event, diagnostic, _state(policy))
        except (TypeError, ValueError, RuntimeError) as exc:
            # Keep the copy's state even on a rejection; this is not a confirmed
            # command return. Unknown peaks are distinguished from IK failure.
            message = str(exc)
            kind = "state_unconfirmed" if "unconfirmed peak read:" in message else "rejected"
            return Step(kind, action, None, copy.deepcopy(guard.events[-1]) if guard.events else None,
                        copy.deepcopy(policy.last_diagnostic), _state(policy), f"{type(exc).__name__}: {message}")

    def _branch(self, result, *, step_degrees):
        if result.status != "returned":
            raise DerivativeUnconfirmed(f"return_{result.status}: {result.detail}")
        event, diag = result.guard_event or {}, result.diagnostic or {}
        if event.get("clamped"):
            raise DerivativeUnconfirmed("guard_clamp_or_boundary")
        q = result.returned_degrees
        margin = min(min(qi-lo, hi-qi) for qi, (lo, hi) in zip(q, JOINT_LIMITS))
        if margin <= max(.25, 4*step_degrees):
            raise DerivativeUnconfirmed("guard_or_joint_limit_boundary")
        if diag.get("output_rejection") or diag.get("joint_limit_pose_adjustments"):
            raise DerivativeUnconfirmed("frame_ik_adjustment_or_rejection")
        active = diag.get("placement_attraction") or diag.get("grasp_attraction")
        if active is None:
            # An inactive release branch is locally independent of peak values;
            # first-observation peak initialization is classified separately.
            if not result.updated_state["grasp_started"]:
                raise DerivativeUnconfirmed("unknown_correction_branch")
            return ("inactive_after_release", bool(diag.get("identity_registration")))
        if active.get("rejected"):
            raise DerivativeUnconfirmed("correction_ik_rejection")
        weight = float(active["weight"])
        if "grasp_attraction" in diag:
            proposed, goal = active["proposed_grasp_reference_mm"], active["registered_source_goal_mm"]
            prior = self._snapshot._approach_peak_mm
            if isinstance(prior, UnknownPeak):
                raise DerivativeUnconfirmed("unknown_approach_peak")
            if prior is not None and abs(proposed[2] - float(prior)) < .1:
                raise DerivativeUnconfirmed("approach_peak_boundary")
            distance = math.dist(result.updated_state["source_anchor"],
                                 result.updated_state["nominal_slots"][result.updated_state["task"]["destination_slot"]-1])
            # If the proposal itself initializes or strictly exceeds the old
            # peak, height/peak is identically one in this local branch.  The
            # zero attraction is stable, including at the first observation.
            if (weight <= 1e-12 and (prior is None or proposed[2] > float(prior)+.1) and
                    proposed[2] > goal[2]+.1 and distance > 1.1):
                return ("grasp_first_peak_identity" if prior is None else "grasp_new_peak_identity",
                        bool(diag.get("identity_registration")))
        if not .0001 < weight < .9999:
            raise DerivativeUnconfirmed("saturation_or_peak_boundary")
        if abs(float(active.get("pitch_change_deg", 0.0))) > .001:
            raise DerivativeUnconfirmed("correction_ik_adjustment")
        if "placement_attraction" in diag:
            proposed, goal = active["proposed_conditional_object_mm"], active["registered_goal_mm"]
            peak = active["closed_peak_mm"]
            observed = result.updated_state["belief"]["points"][result.updated_state["belief"]["held"]][2]
            prior = self._snapshot._closed_peak_mm
            if isinstance(prior, UnknownPeak):
                raise DerivativeUnconfirmed("unknown_closed_peak")
            floor = max(observed, observed if prior is None else float(prior))
            peak_source = "proposal" if proposed[2] > floor else "history"
            if abs(proposed[2] - floor) < .1:
                raise DerivativeUnconfirmed("closed_peak_boundary")
            source = result.updated_state["source_anchor"]
            delta = [goal[i]-source[i] for i in range(2)]
            length2 = sum(x*x for x in delta)
            if length2 < 1 or peak <= goal[2]:
                raise DerivativeUnconfirmed("placement_geometry_boundary")
            progress = sum((proposed[i]-source[i])*delta[i] for i in range(2))/length2
            height = (proposed[2]-goal[2])/(peak-goal[2])
            if not .005 < progress < .995 or not .005 < height < .995:
                raise DerivativeUnconfirmed("placement_saturation_boundary")
            return ("placement", peak_source, bool(diag.get("identity_registration")))
        proposed, goal = active["proposed_grasp_reference_mm"], active["registered_source_goal_mm"]
        peak = active["proposed_approach_peak_mm"]
        prior = self._snapshot._approach_peak_mm
        if isinstance(prior, UnknownPeak):
            raise DerivativeUnconfirmed("unknown_approach_peak")
        if prior is None or abs(proposed[2] - float(prior)) < .1:
            raise DerivativeUnconfirmed("approach_peak_boundary")
        distance = math.dist(result.updated_state["source_anchor"],
                             result.updated_state["nominal_slots"][result.updated_state["task"]["destination_slot"]-1])
        if distance < 1 or peak <= goal[2]:
            raise DerivativeUnconfirmed("grasp_geometry_boundary")
        proximity = 1-math.dist(proposed[:2], goal[:2])/distance
        height = (proposed[2]-goal[2])/(peak-goal[2])
        if not .005 < proximity < .995 or not .005 < height < .995:
            raise DerivativeUnconfirmed("grasp_saturation_boundary")
        return ("grasp", "proposal" if proposed[2] > float(prior) else "history",
                bool(diag.get("identity_registration")))

    def numerical_jacobian(self, q_degrees, *, step_degrees=.05):
        """Central Jacobian in degrees, guarded by branch and h/2 agreement."""
        if not isinstance(q_degrees, torch.Tensor) or q_degrees.shape != (4,) or not q_degrees.is_floating_point():
            raise ValueError("four floating proposal degrees are required")
        if not .005 <= step_degrees <= .25:
            raise ValueError("numerical step must be between .005 and .25 degrees")
        q = q_degrees.detach().to(dtype=torch.float64, device="cpu")
        base = self.evaluate(q, action=motion_policy.MOVE)
        branch = self._branch(base, step_degrees=step_degrees)
        jacobians = []
        for h in (step_degrees, step_degrees/2):
            columns = []
            for i in range(4):
                delta = torch.zeros(4, dtype=torch.float64); delta[i] = h
                plus, minus = (self.evaluate(q+delta, action=motion_policy.MOVE),
                               self.evaluate(q-delta, action=motion_policy.MOVE))
                if self._branch(plus, step_degrees=step_degrees) != branch or self._branch(minus, step_degrees=step_degrees) != branch:
                    raise DerivativeUnconfirmed("correction_branch_changed_within_probe")
                columns.append((torch.tensor(plus.returned_degrees, dtype=torch.float64)-
                                torch.tensor(minus.returned_degrees, dtype=torch.float64))/(2*h))
            jacobians.append(torch.stack(columns, dim=1))
        difference = float((jacobians[0]-jacobians[1]).abs().max())
        scale = float(jacobians[1].abs().max())
        if difference > .005 + .02*scale:
            raise DerivativeUnconfirmed(f"step_size_unstable: {difference:.6g}")
        return torch.tensor(base.returned_degrees, dtype=torch.float64), jacobians[1], {
            "branch": branch, "steps_degrees": [step_degrees, step_degrees/2],
            "max_jacobian_difference": difference, "status": "locally_numerically_checked"}


class _NumericalApplied(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, boundary, step_degrees):
        output, jacobian, _report = boundary.numerical_jacobian(q, step_degrees=step_degrees)
        ctx.save_for_backward(jacobian.to(dtype=q.dtype, device=q.device))
        return output.to(dtype=q.dtype, device=q.device)

    @staticmethod
    def backward(ctx, grad_output):
        (jacobian,) = ctx.saved_tensors
        return jacobian.transpose(0, 1).matmul(grad_output), None, None


def applied_target_loss(boundary, logits, q_normalized, target_degrees, *, label_action, label_mask, joint_weight=100.,
                        step_degrees=.05):
    """Teacher action CE plus gated MOVE loss on the guarded returned proposal.

    q_normalized is the model's original normalized output.  It is promoted
    before conversion to degrees, preserving the exact float32 value and its
    gradient. A non-MOVE label never consumes or fabricates a joint label.
    """
    if not isinstance(boundary, AppliedTargetBoundary) or type(label_mask) is not bool:
        raise ValueError("boundary and an explicit boolean label mask are required")
    if not label_mask:
        if label_action is not None or target_degrees is not None:
            raise ValueError("context-only observations cannot provide a direct label")
        return {"loss": None, "action_ce": None, "joint_mse": None, "guarded_return_degrees": None}
    if (not isinstance(logits, torch.Tensor) or logits.shape != (4,) or not logits.is_floating_point() or
            not bool(torch.isfinite(logits).all())):
        raise ValueError("four finite floating action logits are required")
    if type(label_action) is not int or label_action not in range(4):
        raise ValueError("an explicit recorded action label is required")
    if (not isinstance(q_normalized, torch.Tensor) or q_normalized.shape != (4,) or
            not q_normalized.is_floating_point() or not bool(torch.isfinite(q_normalized).all())):
        raise ValueError("four finite original normalized joint proposals are required")
    if not math.isfinite(joint_weight) or joint_weight < 0:
        raise ValueError("finite nonnegative joint weight required")
    action_loss = F.cross_entropy(logits.reshape(1, 4), torch.tensor([label_action], device=logits.device))
    if label_action != motion_policy.MOVE:
        if target_degrees is not None:
            raise ValueError("non-MOVE labels do not have a joint target")
        return {"loss": action_loss, "action_ce": action_loss, "joint_mse": None, "guarded_return_degrees": None}
    if (not isinstance(target_degrees, torch.Tensor) or target_degrees.shape != (4,) or
            not bool(torch.isfinite(target_degrees).all())):
        raise ValueError("recorded MOVE target requires four finite degrees")
    q_degrees = q_normalized.to(dtype=torch.float64) * 180.0
    output = _NumericalApplied.apply(q_degrees, boundary, float(step_degrees))
    joint_loss = ((output/180.0 - target_degrees.to(output)/180.0)**2).mean()
    return {"loss": action_loss + float(joint_weight)*joint_loss, "action_ce": action_loss,
            "joint_mse": joint_loss, "guarded_return_degrees": output}


def finite_proposal_check(boundary, target_degrees, proposals, *, tolerance_degrees=.001):
    """Check only the supplied finite grid; a miss is explicitly unconfirmed."""
    target = torch.as_tensor(target_degrees, dtype=torch.float64)
    if target.shape != (4,) or not bool(torch.isfinite(target).all()) or not math.isfinite(tolerance_degrees) or tolerance_degrees < 0:
        raise ValueError("finite target and nonnegative explicit tolerance required")
    impossible = any(float(q) < lo-tolerance_degrees or float(q) > hi+tolerance_degrees
                     for q, (lo, hi) in zip(target, JOINT_LIMITS))
    if impossible:
        return {"status": "analytically_impossible", "reason": "target_outside_accepted_guard_range",
                "proposals_checked": 0, "tolerance_degrees": tolerance_degrees}
    best = None
    statuses = []
    for values in proposals:
        q = torch.as_tensor(values, dtype=torch.float64)
        step = boundary.evaluate(q, action=motion_policy.MOVE)
        statuses.append(step.status)
        if step.status == "returned":
            error = float((torch.tensor(step.returned_degrees, dtype=torch.float64)-target).abs().max())
            if best is None or error < best["max_error_deg"]:
                best = {"proposal_deg": q.tolist(), "return_deg": list(step.returned_degrees), "max_error_deg": error}
    return {"status": "solution_found" if best is not None and best["max_error_deg"] <= tolerance_degrees else "unconfirmed",
            "proposals_checked": len(statuses), "return_statuses": statuses,
            "best": best, "tolerance_degrees": tolerance_degrees}
