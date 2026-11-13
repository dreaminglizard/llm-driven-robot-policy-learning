"""Recover commanded servo targets from saved simulation evidence only.

No observed joint position is substituted for a missing servo target. Snapshot
decoding calls MuJoCo's state deserializer, never a simulation step.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import mujoco

from raccoon_control.kinematics import JOINT_LIMITS
from raccoon_memory import motion_policy
from raccoon_memory.scene_simultaneous_policy import OPEN, CLOSED


@dataclass(frozen=True)
class ServoTargets:
    joints_deg: tuple[float, float, float, float]
    gripper_opening_m: float
    gripper_class: int


def servo_targets(joints, opening):
    if (not isinstance(joints, (list, tuple)) or len(joints) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in joints)
            or type(opening) not in (int, float) or not math.isfinite(opening)):
        raise ValueError("servo targets must contain four finite joint angles and one finite opening")
    q = tuple(float(x) for x in joints)
    if any(not lo <= value <= hi for value, (lo, hi) in zip(q, JOINT_LIMITS)):
        raise ValueError("commanded joint target is outside the controller's limits")
    opening = float(opening)
    if math.isclose(opening, 0.04, rel_tol=0.0, abs_tol=1e-9):
        label = OPEN
    elif math.isclose(opening, 0.0, rel_tol=0.0, abs_tol=1e-9):
        label = CLOSED
    else:
        raise ValueError("gripper opening is not representable by the open/closed contract")
    return ServoTargets(q, opening, label)


class SnapshotControlReader:
    """Decode saved actuator control values without advancing the physics state."""

    def __init__(self):
        self._models = {}

    def read(self, snapshot):
        if (not isinstance(snapshot, dict) or snapshot.get("schema") != "raccoon-scene-motion-sim-state.v2"
                or snapshot.get("mujoco_version") != mujoco.__version__):
            raise ValueError("unsupported saved simulator state or MuJoCo version")
        xml, digest = snapshot.get("generated_model_xml"), snapshot.get("generated_model_xml_sha256")
        if not isinstance(xml, str) or hashlib.sha256(xml.encode("utf-8")).hexdigest() != digest:
            raise ValueError("saved simulator model XML fingerprint differs")
        if digest not in self._models:
            model = mujoco.MjModel.from_xml_string(xml)
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in
                   ("joint1_servo", "joint2_servo", "joint3_servo", "joint4_servo",
                    "finger_left_servo", "finger_right_servo")]
            if model.nu != 6 or len(set(ids)) != 6 or min(ids) < 0:
                raise ValueError("saved model lacks exactly the expected six actuators")
            self._models[digest] = model, mujoco.MjData(model), ids
        model, data, ids = self._models[digest]
        sig, state = snapshot.get("state_signature"), snapshot.get("state")
        if (type(sig) is not int or not (sig & int(mujoco.mjtState.mjSTATE_CTRL))
                or not isinstance(state, list) or len(state) != mujoco.mj_stateSize(model, sig)):
            raise ValueError("saved state lacks complete actuator control")
        values = np.asarray(state, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("saved simulator state contains nonfinite values")
        mujoco.mj_setState(model, data, values, sig)
        values = [float(data.ctrl[i]) for i in ids]
        if not math.isclose(values[4], values[5], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("saved finger actuator targets are not symmetric")
        return servo_targets([math.degrees(v) for v in values[:4]], values[4] + values[5])


def teacher_command(transition, decision):
    """Validate and return the exact target sent in a successful teacher decision."""
    if (not isinstance(transition, dict) or not isinstance(decision, dict)
            or transition.get("observation") != decision.get("input_observation")
            or transition.get("applied") is not True):
        raise ValueError("teacher decision does not match an applied recorded transition")
    command = decision.get("simulated_servo_target")
    if not isinstance(command, dict) or command.get("seconds") != 1.0:
        raise ValueError("full teacher servo command or duration is missing")
    target = servo_targets(command.get("joints_deg"), command.get("gripper_opening_m"))
    action = transition.get("action")
    if not isinstance(action, dict):
        raise ValueError("recorded teacher action is missing")
    if action.get("kind") == "move_joints":
        if tuple(action.get("joints_deg", ())) != target.joints_deg:
            raise ValueError("recorded movement and full joint command disagree")
        label = motion_policy.MOVE
    elif action.get("kind") == "set_gripper":
        expected = "open" if target.gripper_class == OPEN else "closed"
        if action.get("state") != expected:
            raise ValueError("recorded gripper action and full command disagree")
        label = motion_policy.OPEN if target.gripper_class == OPEN else motion_policy.CLOSE
    else:
        raise ValueError("unsupported teacher action")
    return label, target


def direct_command(sample, pair, event, state_reader):
    """Resolve a direct teacher command or explicit terminal and its prior control."""
    provenance = sample.get("provenance", {})
    if (sample.get("simulated") is not True or sample.get("actual_hardware") is not False
            or event.get("action_source") != "assistant_tool_action"
            or pair.get("episode_id") != sample.get("episode_id")
            or pair.get("decision_index") != sample.get("decision_index")
            or pair.get("teacher_event_sequence") != event.get("sequence")
            or provenance.get("teacher_event_sha256") != event.get("event_sha256")
            or pair.get("teacher_event_sha256") != event.get("event_sha256")
            or pair.get("state_before_sha256") != event.get("state_before_sha256")
            or pair.get("state_after_sha256") != event.get("state_after_sha256")):
        raise ValueError("direct teacher event, state and sample lineage differs")
    before = pair.get("servo_targets_before", {})
    prior = servo_targets(before.get("arm_servo_target_deg"), before.get("gripper_servo_opening_m"))
    decoded = state_reader.read(pair.get("simulation_state_before"))
    if (any(abs(a - b) > 1e-9 for a, b in zip(decoded.joints_deg, prior.joints_deg))
            or abs(decoded.gripper_opening_m - prior.gripper_opening_m) > 1e-9):
        raise ValueError("saved direct servo targets differ from saved simulator state")
    if event.get("event") == "finish":
        if sample.get("action") != motion_policy.DONE or event.get("assistant_done_proposal") is not True or "action" in event:
            raise ValueError("direct terminal must be an explicit no-command proposal")
        return motion_policy.DONE, None, prior
    if event.get("event") != "action" or event.get("policy_action") not in ("move", "open", "close"):
        raise ValueError("direct teacher action is unsupported or missing")
    action = event.get("action", {})
    if action.get("seconds") != 1.0:
        raise ValueError("direct teacher command duration differs")
    target = servo_targets(action.get("joints_deg"), action.get("gripper_opening_m"))
    label = motion_policy.ACTIONS.index(event["policy_action"])
    if sample.get("action") != label:
        raise ValueError("direct teacher event and exclusive label differ")
    if label == motion_policy.MOVE:
        cached_target = sample.get("absolute_target")
        if (not isinstance(cached_target, list) or len(cached_target) != 4
                or any(abs(x - y / 180.0) > 1e-12 for x, y in zip(cached_target, target.joints_deg))):
            raise ValueError("cached move label and original direct target differ")
    elif target.gripper_class != (OPEN if label == motion_policy.OPEN else CLOSED):
        raise ValueError("direct teacher gripper command and label disagree")
    return label, target, prior
