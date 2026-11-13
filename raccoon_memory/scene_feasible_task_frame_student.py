"""Bounded-joint nominal-FK alternative to the ID-masked TaskFrame student.

The learned body is exactly the ordinary shared-slot/LSTM96/4-action body.
Its four pose-head values are mapped with tanh to JOINT_LIMITS in degrees,
then passed through differentiable nominal FK and the same destination frame.
This describes a nominal pose, not collision, contact, path or task success.

The candidate exposes both its joint proposal and FK pose. Its dedicated policy
passes the joint proposal through the normal joint guard, without an IK
roundtrip. The Cartesian control uses TaskFramePolicy and its exact IK. Thus the
comparison includes both output representation and interpretation path.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import NamedTuple

import torch

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from . import motion_policy, scene_task_frame_student as base, scene_task_frame_training as packed
from .scene_applied_action_policy import AppliedActionPolicy
from .scene_fk_loss import nominal_forward


FORMAT = "raccoon-scene-feasible-task-frame-student.v1"
CHECKPOINT_FORMAT = "raccoon-feasible-task-frame-pair-checkpoint.v1"
BRANCHES = ("cartesian", "bounded_fk")
PAIR_SETTINGS = {
    "pair_comparison": "id_masked_unrestricted_cartesian_vs_bounded_joints_nominal_fk",
    "training_forward": packed.FORMAT,
    "input_view": "features_without_ids_for_both_branches",
    "candidate_transform": "joint_limits_center_plus_half_range_times_tanh_then_nominal_fk",
    "runtime": "cartesian_exact_IK_then_guard__bounded_fk_direct_selected_joints_then_same_guard",
    "comparison_scope": "output_representation_and_interpretation_path",
}
_CONFIG_LITERALS = {
    "source_sampling": "each_registration_once_per_epoch_no_replacement_shared_permutation",
    "source_weight_normalization": "weighted_row_loss_sum_divided_by_selected_unweighted_row_count",
    "sequence": "whole_registration_no_transfer_reset_context_advances_state",
    "position_loss": "mean_xyz_squared_error_in_mm2_only_confirmed_applied_MOVE",
    "pitch_loss": "unwrapped_squared_error_in_deg2_only_confirmed_applied_MOVE",
    "action_loss": "cross_entropy_only_labeled_valid_rows",
    "pose_target": "destination_post_transfer_level_relative_world_xyz_div300_and_unwrapped_pitch_div180",
    "checkpoint": "inference_weights_and_provenance_not_optimizer_or_rng_state",
    "optimizer": "fresh_Adam_default_options_per_branch", "device": "cpu", "torch_threads": 1,
    "deterministic_algorithms": True, **PAIR_SETTINGS,
}


def validate_training_config(config):
    """Validate a complete saved comparison config without importing a trainer."""
    numeric = ("learning_rate", "action_weight", "position_weight_per_mm2", "pitch_weight_per_deg2")
    keys = set(_CONFIG_LITERALS) | set(numeric) | {"seed", "epochs", "batch_size", "source_weights"}
    if (not isinstance(config, dict) or set(config) != keys
            or any(type(config[k]) is not type(v) or config[k] != v for k, v in _CONFIG_LITERALS.items())
            or type(config["seed"]) is not int or not 0 <= config["seed"] < 2**63
            or type(config["epochs"]) is not int or config["epochs"] < 1
            or type(config["batch_size"]) is not int or not 1 <= config["batch_size"] <= 64):
        raise ValueError("unsupported feasible pair training configuration")
    weights = config["source_weights"]
    if not isinstance(weights, dict) or set(weights) != {"student_experience", "direct_demonstration"}:
        raise ValueError("unsupported feasible pair source weights")
    if any(type(x) is not float or not math.isfinite(x) or x <= 0
           for x in [*(config[k] for k in numeric), *weights.values()]):
        raise ValueError("finite positive feasible pair loss and source weights are required")


class FeasibleTaskFrameOutputs(NamedTuple):
    action_logits: torch.Tensor
    move_pose: torch.Tensor
    state: tuple[torch.Tensor, torch.Tensor]
    move_joints_deg: torch.Tensor


def latent_to_joints(latent):
    """Smoothly map finite, dimensionless head values to joint degrees.

    Float rounding can reach an endpoint after tanh saturation, so the promise
    is the closed joint interval. This is not an IK call or a hardware target.
    """
    if (not isinstance(latent, torch.Tensor) or latent.ndim < 1 or latent.shape[-1] != 4
            or not latent.is_floating_point() or not bool(torch.isfinite(latent).all())):
        raise ValueError("finite floating latent joint values with last dimension 4 are required")
    limits = latent.new_tensor(JOINT_LIMITS)
    center = (limits[:, 0] + limits[:, 1]) / 2.0
    half = (limits[:, 1] - limits[:, 0]) / 2.0
    return center + half * torch.tanh(latent)


def latent_to_joints_and_pose(features, latent, valid_mask, constants=None):
    """Convert valid rows to joint degrees and the destination-relative pose.

    Padding may contain NaNs: select valid rows before FK or frame validation.
    Invalid output rows are zero; no padded row can update a loss or gradient.
    """
    base._inputs(features)
    constants = base._constants(features, constants)
    if (not isinstance(latent, torch.Tensor) or latent.shape != (*features.shape[:-1], 4)
            or latent.dtype != features.dtype or latent.device != features.device
            or not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
            or valid_mask.shape != features.shape[:-1] or valid_mask.device != features.device):
        raise ValueError("matching latent poses and boolean valid mask are required")
    # An empty selection gives a graph-connected zero without reading padding
    # or summing large finite valid values that could overflow before * 0.
    result = torch.zeros_like(latent) + latent.reshape(-1)[:0].sum()
    joints = torch.zeros_like(result)
    if bool(valid_mask.any()):
        q = latent_to_joints(latent[valid_mask])
        geometry = Geometry(*constants[:5].detach().cpu().tolist())
        xyz, _ = nominal_forward(q, geometry)
        origin = base.frame_origin(features[valid_mask], constants)
        pitch = q[..., 1:].sum(-1, keepdim=True) + 90.0
        pose = torch.cat(((xyz - origin) / constants[7], pitch / 180.0), dim=-1)
        if not bool(torch.isfinite(pose).all()):
            raise ValueError("bounded FK pose overflows nominal coordinates")
        result[valid_mask] = pose
        joints[valid_mask] = q
    return joints, result


def latent_to_pose(features, latent, valid_mask, constants=None):
    """Return only the destination-relative pose for the shared loss."""
    return latent_to_joints_and_pose(features, latent, valid_mask, constants)[1]


def convert_outputs(features, valid_mask, outputs, constants=None):
    """Apply the candidate head transform to a raw TaskFrameOutputs value."""
    if not isinstance(outputs, base.TaskFrameOutputs):
        raise ValueError("raw TaskFrameOutputs are required")
    return outputs._replace(move_pose=latent_to_pose(features, outputs.move_pose, valid_mask, constants))


class FeasibleTaskFrameStudent(base.TaskFrameStudent):
    """ID-masked streaming policy with the same parameter names as the body."""

    def __init__(self):
        super().__init__(include_object_ids=False)

    def forward(self, features, valid_mask, state=None):
        output = super().forward(features, valid_mask, state)
        joints, pose = latent_to_joints_and_pose(features, output.move_pose, valid_mask, self.nominal_constants)
        return FeasibleTaskFrameOutputs(output.action_logits, pose, output.state, joints)


def candidate_from_raw(model):
    """Copy a trained raw ID-masked body into the streaming candidate wrapper."""
    if type(model) is not base.TaskFrameStudent or model.include_object_ids is not False:
        raise ValueError("a raw ID-masked TaskFrameStudent body is required")
    with torch.random.fork_rng(devices=[]):
        result = FeasibleTaskFrameStudent().to(device=model.action_head.weight.device,
                                               dtype=model.action_head.weight.dtype)
    result.load_state_dict(model.state_dict(), strict=True)
    return result.train(model.training)


def model_contract(branch):
    if branch not in BRANCHES:
        raise ValueError("unknown paired head representation")
    return {
        "model_format": FORMAT, "branch": branch,
        "body": base.model_contract(include_object_ids=False),
        "constructor": ("TaskFrameStudent(include_object_ids=False)" if branch == "cartesian"
                        else "FeasibleTaskFrameStudent()"),
        "head": ("unconstrained_relative_cartesian_xyz_and_unwrapped_pitch" if branch == "cartesian"
                 else "joint_center_plus_half_range_times_tanh_then_nominal_fk"),
        "joint_limits_deg": [list(item) for item in JOINT_LIMITS],
        "move_output": "destination_nominal_grasp_relative_world_xyz_mm/300_and_unwrapped_pitch_deg/180",
        "execution": ("TaskFramePolicy_exact_IK_then_normal_joint_guard" if branch == "cartesian"
                      else "FeasibleTaskFramePolicy_direct_joint_proposal_then_normal_joint_guard"),
        "comparison_scope": "output_representation_and_interpretation_path",
        "ik_limitations": "cartesian_zero_radius_rejected;_branch_and_finite_tolerance_can_reject",
        "limitations": "nominal_fk_is_not_contact_task_success_or_path_safety",
    }


def load_checkpoint(payload):
    """Strictly restore a single inference artifact; provenance is external."""
    from .scene_applied_action_policy import _encoded, _state_digest

    keys = {"format", "model", "config", "config_sha256", "provenance", "training", "state_dict"}
    if not isinstance(payload, dict) or set(payload) != keys or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported feasible pair checkpoint format")
    contract = payload.get("model")
    branch = contract.get("branch") if isinstance(contract, dict) else None
    if (branch not in BRANCHES or contract != model_contract(branch)
            or not isinstance(payload.get("config"), dict)
            or payload.get("config_sha256") != hashlib.sha256(_encoded(payload["config"])).hexdigest()
            or not isinstance(payload.get("provenance"), dict) or not isinstance(payload.get("training"), dict)
            or payload["training"].get("branch") != branch or not isinstance(payload.get("state_dict"), dict)):
        raise ValueError("unsupported feasible pair checkpoint contract")
    validate_training_config(payload["config"])
    training, provenance, config = payload["training"], payload["provenance"], payload["config"]
    training_keys = {"branch", "initial_state_sha256", "final_state_sha256", "ordered_episode_sha256",
                     "episode_order_sha256", "epochs_completed", "optimizer_updates", "training_forward"}
    order = training.get("episode_order_sha256")
    if (set(training) != training_keys or training.get("training_forward") != packed.FORMAT
            or type(training.get("epochs_completed")) is not int or training["epochs_completed"] != config["epochs"]
            or type(training.get("optimizer_updates")) is not int or training["optimizer_updates"] < 1
            or not isinstance(order, list) or len(order) != config["epochs"]
            or training.get("ordered_episode_sha256") != provenance.get("ordered_episode_sha256")
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in (training.get("initial_state_sha256"), training.get("final_state_sha256"),
                                 training.get("ordered_episode_sha256"), *order))):
        raise ValueError("unsupported feasible pair checkpoint training evidence")
    with torch.random.fork_rng(devices=[]):
        model = (base.TaskFrameStudent(False) if branch == "cartesian" else FeasibleTaskFrameStudent()).cpu()
    expected, state = model.state_dict(), payload["state_dict"]
    if (set(state) != set(expected) or any(not isinstance(value, torch.Tensor)
            or value.device.type != "cpu" or value.dtype != expected[name].dtype or value.shape != expected[name].shape
            or not bool(torch.isfinite(value).all()) for name, value in state.items())
            or any(not torch.equal(state[name], value) for name, value in model.named_buffers())):
        raise ValueError("feasible pair tensors or nominal geometry differ")
    model.load_state_dict(state, strict=True)
    if _state_digest(model) != payload["training"].get("final_state_sha256"):
        raise ValueError("feasible pair weights fingerprint differs")
    return model.eval()


class FeasibleTaskFramePolicy(AppliedActionPolicy):
    """Confirmed-control adapter for direct candidate joints; use with guard.

    The model selects all actions and MOVE joints. The pose is a diagnostic and
    the training target representation, never an IK input or scene attraction.
    """

    def __init__(self, model):
        torch.nn.Module.__init__(self)
        if not isinstance(model, FeasibleTaskFrameStudent):
            raise TypeError("a FeasibleTaskFrameStudent is required")
        self.model = model
        self._scene = self._slots = None
        self._reset_runtime()
        self.eval()

    def _reset_runtime(self):
        super()._reset_runtime()
        self.pose_events = []

    def _nominal_error(self, joints, world_pose):
        geometry = Geometry(*self.model.nominal_constants[:5].cpu().tolist())
        return {"position_error_mm": math.dist(geometry.forward(joints), world_pose[:3]),
                "pitch_error_deg": sum(joints[1:]) + 90.0 - world_pose[3],
                "joints_deg": list(joints), "physical_measurement": False}

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        super().on_control_applied(action, transmitted_target_deg, applied_target_deg, gripper_opening_m)
        if action == "move":
            event = self.pose_events[-1]
            event["confirmed_command_nominal_error"] = self._nominal_error(
                applied_target_deg, event["world_pose_mm_deg"])

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self._task is None or self._previous_target is None:
            raise ValueError("confirmed transfer-start control is required")
        if self._pending_action is not None or self._done:
            raise ValueError("the previous decision still requires command return or expected commit")
        row = self._row(windows, masks, previous_target_degrees)
        output = self.model(row, masks[:, -1:], self._state)
        if not isinstance(output, FeasibleTaskFrameOutputs):
            raise ValueError("feasible student output contract differs")
        logits, pose, joints = output.action_logits[:, 0], output.move_pose[:, 0], output.move_joints_deg[:, 0]
        limits = joints.new_tensor(JOINT_LIMITS)
        if (logits.shape != (1, 4) or pose.shape != (1, 4) or joints.shape != (1, 4)
                or not bool(torch.isfinite(logits).all() and torch.isfinite(pose).all() and torch.isfinite(joints).all())
                or bool(((joints < limits[:, 0]) | (joints > limits[:, 1])).any())):
            raise ValueError("feasible student returned invalid action, pose or joint proposal")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        event = {"action": action, "relative_pose_normalized": pose[0].cpu().tolist(), "ik": None,
                 "physical_measurement": False}
        if action == "move":
            target = joints
            constants = self.model.nominal_constants.double()
            world = base.pose_to_world(row[0, 0].double(), pose[0].double(), constants).cpu().tolist()
            proposed = joints[0].cpu().tolist()
            event.update(status="direct_joint_proposal", joint_proposal_deg=proposed, world_pose_mm_deg=world,
                         proposal_nominal_fk_error=self._nominal_error(proposed, world),
                         confirmed_command_nominal_error=None)
        else:
            target = joints.new_tensor([self._previous_target])
            event["status"] = "non_move_pose_ignored"
        normalized = target / motion_policy.JOINT_SCALE_DEG
        self.pose_events.append(event)
        self._state = tuple(value.detach() for value in output.state)
        self._first_registration_row = self._first_transfer_row = False
        if action == "done":
            self._done = True
        else:
            self._pending_action = action
        return logits, normalized


def load_policy_checkpoint(payload):
    """Restore the branch-specific adapter; caller must install joint guard."""
    from .scene_task_frame_policy import TaskFramePolicy
    model = load_checkpoint(payload)
    return (FeasibleTaskFramePolicy(model) if isinstance(model, FeasibleTaskFrameStudent) else TaskFramePolicy(model))


def load_guarded_policy(payload):
    """Restore the appropriate branch adapter wrapped in the existing guard."""
    from raccoon_control.near_joint_limit_guard import GuardedPolicy
    return GuardedPolicy(load_policy_checkpoint(payload))


def diagnose_exact_ik(features, move_pose, *, constants=None, reference_joints_deg=None):
    """Record the existing solver's exact result without commanding a robot.

    The solver does not distinguish a failed branch from numerical tolerance;
    preserve that uncertainty and its original message instead of inventing a
    cause. A successful signed branch and solver residuals remain available.
    """
    from .scene_task_frame_policy import PoseIKError, exact_pose_joints

    if (not isinstance(features, torch.Tensor) or features.ndim != 1
            or not isinstance(move_pose, torch.Tensor) or move_pose.ndim != 1):
        raise ValueError("one feature row and one pose are required")
    constants = base._constants(features, constants)
    world = base.pose_to_world(features, move_pose, constants).detach().cpu().tolist()
    radius = math.hypot(world[0], world[1])
    geometry = Geometry(*constants[:5].detach().cpu().tolist())
    try:
        joints, detail = exact_pose_joints(world, geometry=geometry, reference_joints_deg=reference_joints_deg)
    except PoseIKError as exc:
        reason = ("zero_radius_singularity" if radius < 1e-6 else
                  "ik_branch_or_numerical_tolerance_or_input_unresolved")
        return {"status": "ik_rejected_no_command", "reason": reason, "radius_mm": radius,
                "world_pose_mm_deg": world, "solver_error": str(exc),
                "branch_failure_separately_identified": None,
                "numerical_tolerance_failure_separately_identified": None, "physical_measurement": False}
    return {"status": "exact_pose_proposal", "world_pose_mm_deg": world, "radius_mm": radius,
            "joints_deg": joints, "solver": detail, "physical_measurement": False}
