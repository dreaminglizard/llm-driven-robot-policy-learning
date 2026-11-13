"""Registration-relative source approach term for the native geometry student."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from raccoon_memory import terra_native_geometry_policy as original
from raccoon_memory.scene_fk_loss import nominal_forward


def approach_distance(windows,previous_targets):
    """Public FK distance from the last confirmed setpoint to registered source XY."""
    previous_xyz,_=nominal_forward(previous_targets[:,-1]*180.)
    source_xy=original.slot_centers(windows[:,-1])[:,0,:2]
    return torch.linalg.vector_norm(previous_xyz[:,:2]-source_xy,dim=-1)


class NativeApproachPolicy(original.NativeGeometryPolicy):
    def __init__(self):
        super().__init__()
        self.source_distance_weight=torch.nn.Parameter(torch.zeros(()))

    def outputs(self,windows,masks,previous_targets,previous_actions):
        out=super().outputs(windows,masks,previous_targets,previous_actions)
        distance=(approach_distance(windows,previous_targets)/20.).clamp(max=8.)
        selected_source=F.one_hot(torch.tensor(original.SRC_CONTACT,device=distance.device),len(original.MODES)).to(distance.dtype)
        scores=out.mode_logits+(self.source_distance_weight*distance)[:,None]*selected_source
        choice=scores.masked_fill(~out.valid_modes,float('-inf')).argmax(1)
        joints=out.candidate_joints_deg[torch.arange(len(choice),device=choice.device),choice]
        return out._replace(mode_logits=scores,mode=choice,joints_deg=joints)


def contract():
    value=original.contract()
    value.update({'format':'terra-native-geometry-source-approach.v1','base_geometry_contract':original.FORMAT,
                  'learned_source_contact_extra_logit':'scalar times min(8, registered-source XY distance of the previous applied joints nominal FK in mm / 20)',
                  'extra_private_inputs':False})
    return value


def load_student_checkpoint(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('contract')!=contract():raise ValueError('source approach checkpoint contract differs')
    model=NativeApproachPolicy();expected=model.state_dict();actual=payload['state_dict']
    if set(expected)!=set(actual) or any(actual[n].shape!=v.shape or actual[n].dtype!=v.dtype or not torch.isfinite(actual[n]).all() for n,v in expected.items()):
        raise ValueError('invalid source approach checkpoint tensor')
    if any(not torch.equal(actual[n],expected[n]) for n in ('joint_low','joint_high')):raise ValueError('joint limits changed')
    model.load_state_dict(actual,strict=True)
    return model.to(device).eval()
