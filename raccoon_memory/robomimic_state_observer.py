"""Expose a snapshot of the state produced by an official BC-RNN/GMM call.

The caller owns the official algorithm, observations, random state, and episode
boundaries. This module does not install, modify, train, or execute a robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservedRNNDecision:
    """The already-sampled action and copied post-observation LSTM state.

    ``sampled_action`` is a proposal, not an applied servo command. A later
    completion decision may discard it. The state tensors cannot alias the
    algorithm's live recurrent state and are detached from the action core.
    """

    sampled_action: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor
    decision_count: int


class OfficialRNNStateObserver:
    """Read the official closed-loop LSTM state after one ``get_action`` call.

    This relies on the private ``_rnn_hidden_state`` and ``_rnn_counter`` of
    robomimic BC_RNN_GMM at the pinned source revision documented with the
    tests. No hidden-state reset, counter update, sampling, observation
    preprocessing, or control command is implemented here.
    """

    def __init__(self, algorithm):
        # Import lazily so unrelated project modules do not require robomimic.
        from robomimic.algo.bc import BC_RNN_GMM

        if type(algorithm) is not BC_RNN_GMM:
            raise TypeError("expected the official BC_RNN_GMM instance")
        if algorithm.algo_config.rnn.rnn_type != "LSTM":
            raise ValueError("the observer requires an LSTM recurrent state")
        if algorithm._rnn_is_open_loop is not False:
            raise ValueError("the observer requires current-observation closed-loop mode")
        self._algorithm = algorithm

    def reset(self):
        """Delegate episode reset to the original algorithm."""
        return self._algorithm.reset()

    def step(self, obs_dict, goal_dict=None):
        """Call the official algorithm exactly once and copy its new state."""
        if self._algorithm.nets.training:
            raise RuntimeError("the official policy must be in evaluation mode")
        if self._algorithm._rnn_is_open_loop is not False:
            raise RuntimeError("the policy no longer uses current-observation closed-loop mode")

        # This observer is only for inference / frozen-core feature extraction.
        # The stop head, if added later, is evaluated outside this context.
        with torch.no_grad():
            action = self._algorithm.get_action(obs_dict, goal_dict=goal_dict)
            state = self._algorithm._rnn_hidden_state
            if (not isinstance(action, torch.Tensor) or not isinstance(state, tuple)
                    or len(state) != 2 or not all(isinstance(x, torch.Tensor) for x in state)):
                raise RuntimeError("official action or post-observation LSTM state contract changed")
            hidden, cell = (x.detach().clone() for x in state)

        return ObservedRNNDecision(
            sampled_action=action,
            hidden=hidden,
            cell=cell,
            decision_count=self._algorithm._rnn_counter,
        )
