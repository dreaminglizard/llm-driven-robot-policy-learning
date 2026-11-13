"""One causal manipulation policy with learned, registration-relative MOVE goals.

Four learned coordinate choices express a target in the source or destination
column, at transport or contact height. A fifth, unrestricted in-range joint
choice covers every other motion. No stage, future command, object pose, or
image enters inference. There is no nominal post-output correction.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from raccoon_memory.scene_fk_loss import nominal_forward


FORMAT = 'terra-native-geometry-slot-gru.v1'
MODES = ('free', 'source_column', 'destination_column', 'source_contact', 'destination_contact')
FREE, SRC_COLUMN, DST_COLUMN, SRC_CONTACT, DST_CONTACT = range(5)
LIMITS = tuple(JOINT_LIMITS)
GEOMETRY = Geometry()


class Outputs(NamedTuple):
    action_logits: torch.Tensor
    mode_logits: torch.Tensor
    poses: torch.Tensor
    candidate_joints_deg: torch.Tensor
    valid_modes: torch.Tensor
    mode: torch.Tensor
    joints_deg: torch.Tensor


def slot_centers(frames):
    """Return source and destination nominal cylinder centers in millimetres."""
    xy = frames[..., 74:82].reshape(*frames.shape[:-1], 4, 2) * 300.
    centers = []
    for slot, upper in ((frames[..., 14:18], frames[..., 23:24]), (frames[..., 18:22], frames[..., 25:26])):
        centers.append(torch.cat(((slot.unsqueeze(-1) * xy).sum(-2), 10. + 20. * upper), dim=-1))
    return torch.stack(centers, dim=-2)


def inverse_registered(pose, low, high):
    """Positive-radius, negative-elbow nominal solution and its validity.

    Unreachable proposals are masked at mode selection. Clamping the acos
    argument only protects the numerical evaluation of rejected proposals.
    """
    x, y, z, pitch_deg = pose.unbind(-1)
    pitch = torch.deg2rad(pitch_deg)
    radius = (x.square() + y.square()).clamp_min(1e-10).sqrt()
    r = radius - GEOMETRY.tool_forward_mm * pitch.cos() + GEOMETRY.tool_vertical_mm * pitch.sin()
    h = z - GEOMETRY.base_mm - GEOMETRY.tool_forward_mm * pitch.sin() - GEOMETRY.tool_vertical_mm * pitch.cos()
    upper, fore = GEOMETRY.upper_mm, GEOMETRY.forearm_mm
    cosine = (r.square()+h.square()-upper**2-fore**2)/(2.*upper*fore)
    elbow = -cosine.clamp(-1.+1e-6, 1.-1e-6).acos()
    shoulder = torch.atan2(-r, h) - torch.atan2(fore * elbow.sin(), upper + fore * elbow.cos())
    # A registered center exactly at the base axis has no unique yaw.
    yaw = torch.atan2(-x, torch.where(radius < 1e-4, torch.ones_like(y), y))
    joints = torch.stack((torch.rad2deg(yaw), torch.rad2deg(shoulder), torch.rad2deg(elbow),
                          pitch_deg-90.-torch.rad2deg(shoulder+elbow)), dim=-1)
    valid = ((radius > 1e-4) & (cosine.abs() < 1.-1e-6) & ((joints >= low) & (joints <= high)).all(-1))
    return joints, valid


class NativeGeometryPolicy(nn.Module):
    history_length, feature_count = 8, 82

    def __init__(self):
        super().__init__()
        self.slot_encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        # Two registered-relative TCP vectors for the observed state and two for
        # the previous applied setpoint. Both use nominal FK of public joints.
        self.gru = nn.GRU(96+14+4+5+12, 64, batch_first=True)
        self.action_head = nn.Linear(64, 4)
        self.mode_head = nn.Linear(64, len(MODES))
        self.free_head = nn.Linear(64, 4)
        self.pose_head = nn.Linear(64, 4*4)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        self.register_buffer('joint_low', torch.tensor([v[0] for v in LIMITS], dtype=torch.float32))
        self.register_buffer('joint_high', torch.tensor([v[1] for v in LIMITS], dtype=torch.float32))

    def encode(self, windows, masks, previous_targets, previous_actions):
        if (windows.ndim != 3 or windows.shape[1:] != (8,82) or previous_targets.shape != (*windows.shape[:2],4)
                or masks.shape != windows.shape[:2] or masks.dtype != torch.bool
                or previous_actions.shape != masks.shape or previous_actions.dtype != torch.long):
            raise ValueError('expected public 8x82 windows, a left mask, prior applied joints and action history')
        if not bool(masks[:, -1].all()) or bool((masks[:, :-1] & ~masks[:, 1:]).any()):
            raise ValueError('invalid left-padded history')
        lengths = masks.sum(1)
        time = torch.arange(8, device=windows.device)[None, :]
        index = (time+8-lengths[:,None]).clamp(max=7)
        frames = windows.gather(1, index[..., None].expand(-1,-1,82))
        prior = previous_targets.gather(1, index[..., None].expand(-1,-1,4))
        acts = previous_actions.gather(1,index)
        valid = time < lengths[:,None]
        if bool((((acts < 0)|(acts > 4)) & valid).any()):
            raise ValueError('invalid confirmed previous action')
        levels = frames[..., 30:70].reshape(-1,8,4,2,5)
        target = frames[..., 26:30].unsqueeze(-2).unsqueeze(-2)
        matches = (levels[...,1:] * target).sum(-1)
        src, dst = frames[...,14:18], frames[...,18:22]
        xy = frames[...,74:82].reshape(-1,8,4,2)
        token = torch.cat((xy, 1.-levels[...,0], matches, src[...,None], dst[...,None]), dim=-1)
        embedded = self.slot_encoder(token)
        scene = torch.cat((embedded.sum(-2), (embedded*src[...,None]).sum(-2), (embedded*dst[...,None]).sum(-2)), dim=-1)
        centers = slot_centers(frames)
        tcp = nominal_forward(frames[...,:4]*180.)[0]
        applied_tcp = nominal_forward(prior*180.)[0]
        relative = torch.cat(((centers - tcp[...,None,:]).flatten(-2), (centers - applied_tcp[...,None,:]).flatten(-2)), dim=-1) / 100.
        obs = torch.cat((scene, frames[...,:14], prior, F.one_hot(acts.clamp(0,4),5).to(prior.dtype), relative), dim=-1)
        seq, _ = self.gru(obs.masked_fill(~valid[...,None], 0.))
        return seq[torch.arange(len(windows), device=windows.device), lengths-1]

    def outputs(self, windows, masks, previous_targets, previous_actions):
        hidden = self.encode(windows, masks, previous_targets, previous_actions)
        mode_logits = self.mode_head(hidden)
        low, high = self.joint_low, self.joint_high
        # A tiny inward margin also accommodates float32 normalization by 180.
        free = (low+high)/2. + ((high-low)/2. - 1e-4) * self.free_head(hidden).tanh()
        center = slot_centers(windows[:, -1]).repeat(1,2,1)
        delta = self.pose_head(hidden).reshape(-1,4,4).tanh()
        xy = center[..., :2] + 12. * delta[..., :2]
        z = torch.cat((85. + 90.*delta[:, :2, 2], center[:,2:,2] + 5. + 15.*delta[:,2:,2]), dim=1)
        poses = torch.cat((xy, z[...,None], (8.*delta[...,3])[...,None]), dim=-1)
        angled, feasible = inverse_registered(poses, low+1e-4, high-1e-4)
        candidates = torch.cat((free[:,None], angled), dim=1)
        valid = F.pad(feasible, (1,0), value=True)
        choice = mode_logits.masked_fill(~valid, float('-inf')).argmax(1)
        joints = candidates[torch.arange(len(windows), device=windows.device), choice]
        return Outputs(self.action_head(hidden), mode_logits, poses, candidates, valid, choice, joints)

    def forward(self, windows, masks, previous_targets, previous_actions):
        out = self.outputs(windows, masks, previous_targets, previous_actions)
        return out.action_logits, out.joints_deg/180.


def contract():
    return {'format': FORMAT, 'history_length':8, 'feature_count':82, 'mode_order':list(MODES),
            'actions':['move','open','close','done'], 'input':'unchanged public 8x82; confirmed preceding joint targets and actions',
            'features':'role-shared slots and causal relative FK of observed/applied joints to registered source/destination',
            'geometry':'positive-radius, negative-elbow, analytic nominal IK; unreachable modes masked; free bounded joint mode always enabled',
            'pose_parameterization_mm_deg':{'xy_residual':12.,'column_z_center':85.,'column_z_half_range':90.,
                'contact_z_above_registered_center':5.,'contact_z_residual':15.,'pitch_residual':8.},
            'joint_limits_deg':[list(v) for v in LIMITS], 'output':'four action logits, bounded absolute applied joints / 180; outer range guard only',
            'runtime_image_or_object_truth':False, 'future_or_explicit_stage_input':False, 'nominal_output_postcorrection':False,
            'stored_original_ids_only':True, 'id_permutation':False, 'multi_view_distillation':False}


def load_student_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('contract') != contract():
        raise ValueError('native geometry checkpoint contract differs')
    model = NativeGeometryPolicy()
    params = model.state_dict()
    loaded = payload['state_dict']
    if set(loaded) != set(params) or any(p.shape != loaded[k].shape or p.dtype != loaded[k].dtype
            or not bool(torch.isfinite(loaded[k]).all()) for k,p in params.items()):
        raise ValueError('native geometry state contains invalid parameters')
    if any(not torch.equal(loaded[k],params[k]) for k in ('joint_low','joint_high')):
        raise ValueError('native geometry joint bounds changed')
    model.load_state_dict(loaded, strict=True)
    return model.to(device).eval()
