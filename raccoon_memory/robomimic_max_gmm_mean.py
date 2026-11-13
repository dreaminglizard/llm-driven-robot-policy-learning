"""Evaluation-only maximum-probability GMM component mean adapter.

The existing completion gate performs exactly one unmodified official action
call and keeps its normal RNG consumption. This adapter observes the GMM that
that call created and replaces the action only after a CONTINUE decision.
It neither owns an environment nor applies an action.
"""
from __future__ import annotations

from unittest.mock import patch

import torch
import torch.distributions as distributions

from raccoon_memory.robomimic_completion import CompletionDecision, CompletionGate


class MaximumGMMMeanGate:
    """Wrap the bound official core and completion head for one experiment.

    The installed method observers are instance-local and last only for one
    synchronous ``decide``. They do not resample or change the input, hidden
    state, completion features, original sample, or random-generator state.
    ``trace`` retains the unmodified sample and all component parameters for
    each call, including DONE and exceptions where they were available.
    """

    def __init__(self, gate: CompletionGate):
        if not isinstance(gate, CompletionGate):
            raise TypeError("a verified official completion gate is required")
        if gate.core.algorithm.nets["policy"].use_tanh:
            raise ValueError("the experiment requires the untanh-wrapped official GMM")
        if not gate.core.algorithm.algo_config.gmm.low_noise_eval:
            raise ValueError("the official low-noise evaluation setting must be preserved")
        self.gate = gate
        self.trace = []

    def clear_trace(self) -> None:
        """Clear only the experiment's audit records before an attempt."""
        self.trace = []

    def reset_episode(self) -> None:
        self.clear_trace()
        self.gate.reset_episode()

    def decide(self, observation: torch.Tensor) -> CompletionDecision:
        core = self.gate.core
        policy = core.algorithm.nets["policy"]
        original_forward = policy.forward_train
        original_observe = core.observer.step
        item = {"call_index": len(self.trace), "original_sample": None,
                "component_means": None, "component_scales": None,
                "component_probabilities": None, "highest_probability_index": None,
                "highest_probability_mean": None, "completion": None,
                "proposed_action": None, "decision_count": None, "status": "started"}
        self.trace.append(item)
        distributions_seen, samples_seen = [], []

        def observe_distribution(*args, **kwargs):
            if args or kwargs.get("return_state") is not True:
                raise RuntimeError("unexpected official GMM forward path")
            value = original_forward(*args, **kwargs)
            dist, _state = value
            if (not isinstance(dist, distributions.MixtureSameFamily)
                    or not isinstance(dist.component_distribution, distributions.Independent)
                    or not isinstance(dist.component_distribution.base_dist, distributions.Normal)):
                raise RuntimeError("official GMM distribution contract changed")
            means = dist.component_distribution.base_dist.loc.detach().clone()
            scales = dist.component_distribution.base_dist.scale.detach().clone()
            probabilities = dist.mixture_distribution.probs.detach().clone()
            if tuple(means.shape) != (1, 1, 5, 5) or tuple(scales.shape) != (1, 1, 5, 5) or tuple(probabilities.shape) != (1, 1, 5):
                raise RuntimeError("official GMM component dimensions differ")
            distributions_seen.append((means[0, 0], scales[0, 0], probabilities[0, 0]))
            item["component_means"] = means[0, 0].cpu().tolist()
            item["component_scales"] = scales[0, 0].cpu().tolist()
            item["component_probabilities"] = probabilities[0, 0].cpu().tolist()
            return value

        def observe_sample(*args, **kwargs):
            result = original_observe(*args, **kwargs)
            sample = result.sampled_action.detach().clone()
            samples_seen.append(sample)
            item["original_sample"] = sample.cpu()[0].tolist()
            return result

        try:
            with patch.object(policy, "forward_train", new=observe_distribution), \
                    patch.object(core.observer, "step", new=observe_sample):
                decision = self.gate.decide(observation)
            if len(distributions_seen) != 1 or len(samples_seen) != 1:
                raise RuntimeError("expected exactly one official observation and GMM sample")
            means, _scales, probabilities = distributions_seen[0]
            item["completion"] = decision.kind
            item["decision_count"] = decision.decision_count
            valid_distribution = bool(torch.isfinite(means).all() and torch.isfinite(probabilities).all())
            if valid_distribution:
                selected = int(torch.argmax(probabilities).item())
                selected_mean = means[selected].detach().reshape(1, 5).clone()
                item["highest_probability_index"] = selected
                item["highest_probability_mean"] = selected_mean.cpu()[0].tolist()
            else:
                selected_mean = None
            if decision.kind == "done":
                if decision.proposed_action is not None:
                    raise RuntimeError("the existing completion gate exposed a done action")
                item["status"] = "done_action_suppressed"
                return CompletionDecision("done", None, decision.decision_count)
            if decision.kind != "continue" or decision.proposed_action is None:
                raise RuntimeError("the existing completion gate returned an invalid decision")
            if selected_mean is None or not bool(torch.isfinite(selected_mean).all()):
                raise ValueError("official action proposal is not finite")
            item["proposed_action"] = selected_mean.cpu()[0].tolist()
            item["status"] = "continue_proposal"
            return CompletionDecision("continue", selected_mean, decision.decision_count)
        except Exception as exc:
            # The official call may already have consumed RNG and hidden state.
            core._needs_reset = True
            item["status"] = "error"
            item["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
