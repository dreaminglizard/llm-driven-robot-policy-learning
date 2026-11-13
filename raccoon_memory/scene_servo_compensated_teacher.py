"""Observable servo-compensated nominal teacher, for offline training only.

The candidate body is the exact code used for the 192-start teacher diagnostic.
Use teacher_reached_rule() around calls: it supplies the same observed-FK
arrival rule without changing the student policy or using object state.
"""
from collections import Counter
import math
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from raccoon_memory import motion_policy, scene_nominal_kinematic_teacher as teacher

# BEGIN VERIFIED CANDIDATE SOURCE
geometry=Geometry()
def observed_reached(row,prior,goal):
 observed=[float(v)*motion_policy.JOINT_SCALE_DEG for v in row[:4]]
 if len(observed)!=4 or any(not math.isfinite(x) for x in observed):raise ValueError("invalid observed joints")
 target=geometry.forward(goal);actual=geometry.forward(observed)
 return (max(abs(a-b) for a,b in zip(observed,goal))<=1.0 and
         math.hypot(actual[0]-target[0],actual[1]-target[1])<=1.0 and
         abs(actual[2]-target[2])<=1.5 and
         abs(sum(observed[1:])-sum(goal[1:]))<=.5)
class Candidate(nn.Module):
 history_length=8;feature_count=82
 def __init__(self):
  super().__init__();self.stats=Counter();self.max_correction=0.;self.max_projection_correction=0.;self.min_margin=math.inf;self.min_cap=math.inf
 def forward_with_control_state(self,w,m,p):
  window=w[0].tolist();mask=m[0].tolist();prior=p[0].tolist()
  action,goal=teacher.decide(window,mask,prior);self.stats["decisions"]+=1
  if action==motion_policy.MOVE:
   rows=[r for r,valid in zip(window,mask) if valid];row=rows[-1]
   source,closed,retreat=teacher._waypoints(row)
   command=teacher._one_hot(row,6,3)
   released=any(teacher._one_hot(old,6,3)==2 for old in rows[:-1])
   path=([retreat] if released else source) if command==1 else [source[-1],*closed]
   ids=[i for i,g in enumerate(path) if teacher._distance(g,goal)<1e-9]
   assert len(ids)==1,(ids,"goal does not belong uniquely to active path")
   i=ids[0];others=[g for j,g in enumerate(path) if j!=i]
   spacing=min((teacher._distance(goal,g) for g in others),default=math.inf)
   cap=min(1.25,spacing/4);self.min_cap=min(self.min_cap,cap)
   observed=[float(x)*motion_policy.JOINT_SCALE_DEG for x in row[:4]]
   delta=np.array(prior)-np.array(observed);norm=float(np.linalg.norm(delta))
   if norm>cap:
    delta*=cap/norm;self.stats["norm_capped"]+=1
   self.max_correction=max(self.max_correction,float(np.linalg.norm(delta)))
   corrected=[max(lo,min(hi,float(g)+float(d))) for g,d,(lo,hi) in zip(goal,delta,JOINT_LIMITS)]
   actual_norm=teacher._distance(corrected,goal);self.max_projection_correction=max(self.max_projection_correction,actual_norm)
   assert actual_norm<=cap+1e-10 and all(lo<=q<=hi for q,(lo,hi) in zip(corrected,JOINT_LIMITS))
   if others:
    margin=min(teacher._distance(corrected,g) for g in others)-actual_norm
    assert margin>0
    self.min_margin=min(self.min_margin,margin)
   if any(abs(float(q)-float(g)-float(z))>1e-10 for q,g,z in zip(corrected,goal,delta)):
    self.stats["projected"]+=1
   self.stats["move"]+=1;goal=corrected
  elif action==motion_policy.CLOSE:
   source,closed,_=teacher._waypoints(window[-1]);path=[source[-1],*closed]
   assert min(range(len(path)),key=lambda i:teacher._distance(prior,path[i]))==0
   self.stats["close_phase_transition_nearest_ok"]+=1
  logits=w.new_full((1,4),-1.);logits[0,action]=1.
  return logits,w.new_tensor([[float(q)/180. for q in goal]])
# END VERIFIED CANDIDATE SOURCE

def teacher_reached_rule():
    """Apply the candidate's observed-only arrival rule during collection."""
    return patch.object(teacher, "_reached", observed_reached)

