"""Learned public-pose contributions to the CLOSE action logit of one student."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from . import terra_native_approach_policy as approach
from .scene_fk_loss import nominal_forward


def public_pose(windows, targets):
    frame=windows[:,-1]
    center=approach.original.slot_centers(frame)[:,0]
    measured=nominal_forward(frame[:,:4]*180.)[0]-center
    applied=nominal_forward(targets[:,-1]*180.)[0]-center
    return torch.stack((applied[:,:2].norm(dim=1),measured[:,:2].norm(dim=1),applied[:,2],measured[:,2]),dim=1)


def features(pose, previous_action, limits):
    xy=(pose[:,:2]-limits[:2]).clamp_min(0.)/20.
    z=pose[:,2:]
    above=(z-limits[4:6]).clamp_min(0.)
    below=(limits[2:4]-z).clamp_min(0.)
    return torch.cat((xy,(above+below)/limits[6:],(previous_action!=0).to(z.dtype)[:,None]),dim=1)


class CloseEvidencePolicy(approach.NativeApproachPolicy):
    def __init__(self):
        super().__init__()
        self.register_buffer("demonstrated_close_pose",torch.tensor([0.,0.,0.,0.,0.,0.,1.,1.]))
        self.close_output_weight=nn.Parameter(torch.zeros(5))

    def outputs(self,windows,masks,previous_targets,previous_actions):
        out=super().outputs(windows,masks,previous_targets,previous_actions)
        x=features(public_pose(windows,previous_targets),previous_actions[:,-1],self.demonstrated_close_pose)
        correction=(x*self.close_output_weight).sum(dim=1)
        # The same four-way policy argmax still decides all actions.
        basis=F.one_hot(torch.tensor(2,device=x.device),4).to(x.dtype)
        return out._replace(action_logits=out.action_logits-correction[:,None]*basis)


def contract():
    return approach.contract() | {
        "format":"terra-native-source-distance-close-evidence.v1",
        "output":"same original four action logits, with an additive learned contribution to CLOSE only; joint proposals unchanged",
        "public_close_features":["previous applied FK xy excess","observed joint FK xy excess","previous applied FK z outside demonstrated grasp heights",
                                 "observed joint FK z outside demonstrated grasp heights","previous applied action differs from MOVE"],
        "learned_statistics":"original CLOSE examples only: XY maximum; each height min, max and interquartile range; distances in millimetres",
        "no_runtime_phase_or_case_rules":True,
    }


def load_student_checkpoint(path,device="cpu"):
    record=torch.load(path,map_location="cpu",weights_only=True)
    model=CloseEvidencePolicy()
    expected=model.state_dict()
    state=record["state_dict"]
    if (record.get("contract")!=contract() or set(state)!=set(expected) or
        any(v.shape!=state[k].shape or v.dtype!=state[k].dtype or not torch.isfinite(state[k]).all() for k,v in expected.items()) or
        any(not torch.equal(expected[k],state[k]) for k in ("joint_low","joint_high"))):
        raise ValueError("invalid learned CLOSE evidence checkpoint")
    limits=state["demonstrated_close_pose"]
    if not bool((limits[:2]>=0).all() and (limits[2:4]<=limits[4:6]).all() and (limits[6:]>0).all()):
        raise ValueError("invalid demonstrated CLOSE geometry")
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()
