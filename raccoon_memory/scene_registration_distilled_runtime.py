"""Simulation session connecting the distilled student to public callbacks."""
from __future__ import annotations

from pathlib import Path

import torch

from raccoon_control.near_joint_limit_guard import GuardedPolicy
from . import scene_oracle_free_session as session
from . import scene_registration_distilled_policy as policy


def load_guarded_checkpoint(path):
    """Return the one learned student behind the existing joint-limit guard."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return GuardedPolicy(policy.RegistrationAdapter(policy.load_checkpoint(payload)))


class RegistrationSceneSession(session.OracleFreeSceneModelSession):
    """Observe public joints at the already-confirmed control callback.

    No callback is simulated when the executor rejects or cannot confirm a
    command.  This session is restricted to the original virtual simulator by
    its superclass and does not perform any teacher or IK correction.
    """
    def __init__(self, sim, memory, student, **kwargs):
        if not isinstance(student, GuardedPolicy) or not isinstance(student.policy, policy.RegistrationAdapter):
            raise ValueError("the distilled student and existing joint guard are required")
        super().__init__(sim, memory, student, **kwargs)
        student.policy.return_joint_observer = lambda: session._robot_observation(sim, None)["joints_deg"]
