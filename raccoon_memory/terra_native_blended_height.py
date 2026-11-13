"""A single geometry student with one learned continuous source-height blend."""
from __future__ import annotations

import torch
from torch import nn
from raccoon_memory import terra_native_approach_policy as approach
from raccoon_memory import terra_native_geometry_policy as geometry
from raccoon_memory.scene_fk_loss import nominal_forward


def blend_features(out,w,p):
    f=w[:,-1];source=geometry.slot_centers(f)[:,0];observed=nominal_forward(180.*f[:,:4])[0];prev=nominal_forward(180.*p[:,-1])[0]
    return torch.stack(((out.mode_logits[:,3]-out.mode_logits[:,1])/4.,
        torch.linalg.vector_norm(prev[:,:2]-source[:,:2],dim=1)/100.,torch.linalg.vector_norm(observed[:,:2]-source[:,:2],dim=1)/100.,
        f[:,7],f[:,10],(out.poses[:,0,2]-source[:,2])/100.,(prev[:,2]-source[:,2])/100.,(observed[:,2]-source[:,2])/100.),dim=1)


class BlendedHeightPolicy(approach.NativeApproachPolicy):
    def __init__(self):
        super().__init__();self.source_height_gate=nn.Linear(8,1)

    def outputs(self,w,m,p,a):
        out=super().outputs(w,m,p,a)
        alpha=self.source_height_gate(blend_features(out,w,p)).sigmoid().squeeze(1)
        source=out.poses[:,0];z=source[:,2]+alpha*(out.poses[:,2,2]-source[:,2])
        pose=torch.cat((source[:,:2],z[:,None],source[:,3:]),1)
        q,feasible=geometry.inverse_registered(pose,self.joint_low+1e-4,self.joint_high-1e-4)
        posed=torch.cat((pose[:,None],out.poses[:,1:]),1)
        qs=torch.cat((out.candidate_joints_deg[:,:1],q[:,None],out.candidate_joints_deg[:,2:]),1)
        valid=torch.cat((out.valid_modes[:,:1],feasible[:,None],out.valid_modes[:,2:]),1)
        choice=out.mode_logits.masked_fill(~valid,float('-inf')).argmax(1)
        return out._replace(poses=posed,candidate_joints_deg=qs,valid_modes=valid,mode=choice,joints_deg=qs[torch.arange(len(q),device=q.device),choice])


def contract():
    return approach.contract()|{'format':'terra-native-geometry-learned-height-blend.v1','base_approach_contract':'terra-native-geometry-source-approach.v1',
        'source_column_height':'continuous sigmoid-affine 8-input interpolation of two learned source-frame heights; no manual action, phase switch, or alternate policy',
        'gate_inputs':['source contact minus column mode logit /4','previous applied TCP distance to source XY /100mm','observed TCP distance to source XY /100mm',
            'gripper commanded open','gripper reported open','predicted source column height relative to object center /100mm','previous applied relative height /100mm','observed relative height /100mm']}


def load_student_checkpoint(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True);m=BlendedHeightPolicy();actual=payload['state_dict'];exp=m.state_dict()
    if payload.get('contract')!=contract() or set(actual)!=set(exp) or any(actual[k].shape!=t.shape or actual[k].dtype!=t.dtype or not bool(torch.isfinite(actual[k]).all()) for k,t in exp.items()):
        raise ValueError('invalid height blend checkpoint')
    if any(not torch.equal(actual[k],exp[k])for k in ('joint_low','joint_high')):raise ValueError('joint limits changed')
    m.load_state_dict(actual,strict=True);return m.to(device).eval()
