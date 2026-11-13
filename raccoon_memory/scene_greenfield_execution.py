"""Run the shared greenfield Actor through the oracle-free policy interface.

Only the current public 82-feature row and validated registration/control
callbacks enter this adapter.  The common dataset encoder constructs the
nominal scene and robot vectors; no simulator object, contact or qvel is read.
The executor does not expose observation timestamps to the policy, and joint
speed limits have not been established.  Both training and execution therefore
use zero velocity with a separately recorded invalid flag, even though the
nominal command/hold intervals have been verified.  The original Actor has no
velocity-validity input.

This is a simulation policy adapter, not a device controller.  An injected
Actor implements the original six-input, three-output, single-step contract.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_features, task_catalog
from . import scene_greenfield_dataset as greenfield
from .scene_custom_slots_sim import registered_slot_xy_mm


class GreenfieldExecutionPolicy(nn.Module):
    """Stateful bridge from an injected Actor(4) to the shared executor.

    ``on_scene_registered`` is the only reset boundary.  The executor's
    confirmed-control and expected-scene callbacks advance history; merely
    predicting an action never records a MOVE as applied.
    """

    history_length = scene_features.HISTORY_LENGTH
    feature_count = len(scene_features.FEATURE_NAMES)

    def __init__(self, actor):
        super().__init__()
        if not callable(actor) or not callable(getattr(actor, "initial_hidden", None)):
            raise TypeError("an Actor with initial_hidden(batch_size) is required")
        self.actor = actor
        self._scene = self._slots = None
        self._reset_runtime()
        self.eval()

    def _reset_runtime(self):
        self._hidden = None
        self._task = None
        self._pending_action = None
        self._done = False
        # The executor supplies its last transmitted target as context, while
        # the confirmed servo value is retained separately for Actor history.
        self._sent_target = None
        self._held_target = None
        self._last_move_target = None
        self._opening = None
        self._previous_mode = 4  # START is input-only.
        self._first_instruction_row = True
        # This diagnostic is not given to the original Actor.
        self.velocity_validity = []

    @staticmethod
    def _joints(values):
        if not isinstance(values, (tuple, list)) or len(values) != 4:
            raise ValueError("confirmed control requires four joint angles")
        if any(type(q) not in (int, float) or not math.isfinite(q) or not lo <= q <= hi
               for q, (lo, hi) in zip(values, JOINT_LIMITS)):
            raise ValueError("confirmed control must be finite and within joint limits")
        return [float(q) for q in values]

    def on_scene_registered(self, scene, nominal_slot_xy_mm):
        before = task_catalog.scene(scene)
        slots = registered_slot_xy_mm(nominal_slot_xy_mm)
        self._reset_runtime()
        self._scene, self._slots = before, slots

    def on_transfer_started(self, before, task, applied_targets):
        if self._scene is None or self._task is not None or self._pending_action is not None or self._done:
            raise ValueError("a registered, committed transfer boundary is required")
        before, task = scene_features.transfer_task(before, task)
        if before != self._scene or not isinstance(applied_targets, dict):
            raise ValueError("transfer differs from the expected registered scene")
        held = self._joints(applied_targets.get("joints_deg"))
        opening = applied_targets.get("gripper_opening_m")
        if (type(opening) not in (int, float) or not math.isfinite(opening)
                or not .025 <= opening <= .05):
            raise ValueError("a transfer must start with confirmed open control")
        if self._sent_target is not None and (
            any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0) for a, b in zip(held, self._sent_target))
            or not math.isclose(opening, self._opening, abs_tol=1e-9, rel_tol=0)
        ):
            raise ValueError("control changed across transfers without a confirmed return")
        if self._held_target is None:
            self._held_target = list(held)
        self._sent_target, self._opening = held, float(opening)
        self._task = task
        self._first_instruction_row = True

    def _current_row(self, windows, masks, previous):
        if (not isinstance(windows, torch.Tensor)
                or tuple(windows.shape) != (1, self.history_length, self.feature_count)
                or windows.dtype != torch.float32 or not bool(torch.isfinite(windows).all())
                or not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
                or masks.device != windows.device or tuple(masks.shape) != (1, self.history_length)
                or not bool(masks[0, -1])):
            raise ValueError("one finite current scene feature row is required")
        # A left-padded window may not hide gaps in the observed history.
        if bool((masks[0, :-1] & ~masks[0, 1:]).any()):
            raise ValueError("history mask must be left padded")
        if (not isinstance(previous, torch.Tensor) or tuple(previous.shape) != (1, 4)
                or previous.dtype != torch.float32 or previous.device != windows.device
                or not torch.equal(previous[0], previous.new_tensor(self._sent_target))):
            raise ValueError("previous target differs from confirmed transmitted control")
        row = windows[0, -1]
        dummy = {"schema_version": "PolicyObservation.phase1", "joints_deg": [0, -10, -110, 30],
                 "sensor_valid": True, "moving": False, "gripper_command": "open",
                 "gripper_reported_state": "open", "gripper_reported_valid": True,
                 "gripper_feedback_fresh": True}
        registered = scene_features.encode_scene_features(dummy, self._scene, self._task, self._slots)[14:]
        if not torch.equal(row[14:], row.new_tensor(registered)):
            raise ValueError("student inputs differ from the active registered task")
        commanded = [0., 1., 0.] if self._opening >= .025 else [0., 0., 1.]
        if row[4].item() != 1. or not torch.equal(row[6:9], row.new_tensor(commanded)):
            raise ValueError("public robot state differs from confirmed control")
        return row.detach().cpu().numpy()

    @staticmethod
    def _legacy_target(q_norm):
        """Decode Actor [-1,1] through radian joint limits to degrees/180.

        The shared guard multiplies the returned float32 values by 180.  Round
        inward only when float32 representation would put an endpoint outside
        its actual bound, so a legal Actor endpoint remains a legal command.
        """
        degrees = greenfield.decode_target_deg(q_norm[0].detach().cpu().numpy())
        legacy = torch.as_tensor(degrees / motion_policy.JOINT_SCALE_DEG,
                                 dtype=torch.float32, device=q_norm.device).unsqueeze(0)
        for index, (lo, hi) in enumerate(JOINT_LIMITS):
            value = float(legacy[0, index]) * motion_policy.JOINT_SCALE_DEG
            if value < lo:
                legacy[0, index] = torch.nextafter(legacy[0, index], legacy.new_tensor(math.inf))
            elif value > hi:
                legacy[0, index] = torch.nextafter(legacy[0, index], legacy.new_tensor(-math.inf))
        if any(not lo <= float(legacy[0, i]) * motion_policy.JOINT_SCALE_DEG <= hi
               for i, (lo, hi) in enumerate(JOINT_LIMITS)):
            raise ValueError("decoded target cannot be represented inside joint limits")
        return legacy

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous_target_degrees):
        if self._task is None or self._held_target is None or self._sent_target is None:
            raise ValueError("confirmed transfer-start control is required")
        if self._pending_action is not None or self._done:
            raise ValueError("the previous decision still requires command return or expected commit")
        row = self._current_row(windows, masks, previous_target_degrees)
        # The policy interface exposes no observed timestamp and the per-axis
        # speed limits are not verified.  Never use the simulator's qvel.
        tokens, mask, src, dst, robot, velocity_valid = greenfield.encode_step(
            row, previous_mode=self._previous_mode, last_move_target_deg=self._last_move_target,
            new_instruction=self._first_instruction_row, velocity_deg_s=None)
        mask_array = np.asarray(mask)
        if (mask_array.dtype != np.bool_ or isinstance(src, (bool, np.bool_))
                or isinstance(dst, (bool, np.bool_)) or not isinstance(src, (int, np.integer))
                or not isinstance(dst, (int, np.integer))
                or not isinstance(velocity_valid, (bool, np.bool_)) or bool(velocity_valid)):
            raise ValueError("common encoder returned invalid Actor mask or validity metadata")
        tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.float32, device=windows.device).unsqueeze(0)
        token_mask = torch.as_tensor(mask_array, dtype=torch.bool, device=windows.device).unsqueeze(0)
        src_idx = torch.as_tensor([src], dtype=torch.long, device=windows.device)
        dst_idx = torch.as_tensor([dst], dtype=torch.long, device=windows.device)
        robot = torch.as_tensor(np.asarray(robot), dtype=torch.float32, device=windows.device).unsqueeze(0)
        if (tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[2] != 11
                or tuple(token_mask.shape) != tuple(tokens.shape[:2]) or not bool(token_mask.any())
                or src_idx.item() < 0 or dst_idx.item() < 0
                or src_idx.item() >= tokens.shape[1] or dst_idx.item() >= tokens.shape[1]
                or not bool(token_mask[0, src_idx.item()] and token_mask[0, dst_idx.item()])
                or tuple(robot.shape) != (1, 30)
                or bool(torch.count_nonzero(tokens.masked_select(~token_mask.unsqueeze(-1))))
                or not bool(torch.isfinite(tokens).all() and torch.isfinite(robot).all())):
            raise ValueError("common encoder returned invalid Actor input")
        hidden = self._hidden if self._hidden is not None else self.actor.initial_hidden(1)
        if (not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, 64)
                or hidden.dtype != torch.float32 or hidden.device != windows.device
                or not bool(torch.isfinite(hidden).all())
                or (self._hidden is None and bool(torch.count_nonzero(hidden)))):
            raise ValueError("Actor hidden state must start at zero on the input device")
        output = self.actor(tokens, token_mask, src_idx, dst_idx, robot, hidden)
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise ValueError("Actor must return logits, normalized joints and hidden state")
        logits, q_norm, next_hidden = output
        if (not all(isinstance(x, torch.Tensor) for x in output)
                or tuple(logits.shape) != (1, 4) or tuple(q_norm.shape) != (1, 4)
                or tuple(next_hidden.shape) != (1, 64)
                or any(x.dtype != torch.float32 or x.device != windows.device for x in output)
                or not all(bool(torch.isfinite(x).all()) for x in output)
                or bool(((q_norm < -1.) | (q_norm > 1.)).any())):
            raise ValueError("Actor output shape, range or values are invalid")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        angles = self._legacy_target(q_norm)
        self._hidden = next_hidden.detach()
        self._first_instruction_row = False
        self.velocity_validity.append(bool(velocity_valid))
        if action == "done":
            self._done = True
        else:
            self._pending_action = action
        return logits, angles

    def forward(self, windows, masks):
        raise ValueError("greenfield execution requires confirmed previous control")

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, gripper_opening_m):
        if self._pending_action is None or action != self._pending_action:
            raise ValueError("command return does not match the pending model action")
        sent, actual = self._joints(transmitted_target_deg), self._joints(applied_target_deg)
        if any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0) for a, b in zip(sent, actual)):
            raise ValueError("applied control differs from the transmitted command")
        expected_opening = .04 if action == "open" else 0. if action == "close" else self._opening
        if (type(gripper_opening_m) not in (int, float) or not math.isfinite(gripper_opening_m)
                or not math.isclose(gripper_opening_m, expected_opening, abs_tol=1e-9, rel_tol=0)
                or (action != "move" and (
                    any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0)
                        for a, b in zip(sent, self._sent_target))
                    or any(not math.isclose(a, b, abs_tol=1e-7, rel_tol=0)
                           for a, b in zip(actual, self._held_target))))):
            raise ValueError("command return differs from held joints or expected gripper control")
        if action == "move":
            self._last_move_target = list(actual)
        self._sent_target, self._held_target, self._opening = sent, actual, float(gripper_opening_m)
        self._previous_mode = motion_policy.ACTIONS.index(action)
        self._pending_action = None

    def on_expected_transfer_committed(self, before, task, after):
        before, task = scene_features.transfer_task(before, task)
        after = task_catalog.scene(after)
        if (not self._done or self._pending_action is not None or before != self._scene or task != self._task
                or after != task_catalog.transfer(before, task["source_slot"], task["destination_slot"])):
            raise ValueError("expected commit does not match the completed student transfer")
        self._scene, self._task, self._done = after, None, False
        self._previous_mode = motion_policy.DONE
        self._first_instruction_row = True
