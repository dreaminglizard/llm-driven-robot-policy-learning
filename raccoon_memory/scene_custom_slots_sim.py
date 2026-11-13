"""Simulation-only scenes with explicitly registered nominal slot coordinates.

The four coordinates are calibration inputs, not observations of the objects.
This module does not authorize or control a physical robot.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry
from .motion_sim import OBJECT_RADIUS_M, SLOT_XY_M
from .scene_motion_sim import STATE_SCHEMA, _digest
from .scene_ramped_motion_sim import FINGER_OPEN_INSTANT, RampedSceneMotionSim


def registered_slot_xy_mm(value):
    """Validate a nominal four-slot calibration for this simulation pilot.

    The endpoint IK and spacing tests are necessary checks for this pilot, not
    a collision check or a guarantee that a complete movement is possible.
    """
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("four registered slot coordinates are required")
    result = []
    for point in value:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("each registered slot needs an X and Y coordinate")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError("slot coordinates must be finite numbers in millimetres")
        result.append(tuple(float(v) for v in point))
    # Include a margin for independent slot reset jitter in this simulator.
    minimum_spacing_mm = 2 * OBJECT_RADIUS_M * 1000 + 5.0
    for i in range(4):
        for j in range(i):
            if math.dist(result[i], result[j]) < minimum_spacing_mm:
                raise ValueError("registered slots are too close for the simulation pilot")
    geometry = Geometry()
    try:
        for x, y in result:
            for z in (10.0, 30.0, 80.0):
                geometry.inverse((x, y, z))
    except ControlError as exc:
        raise ValueError("a registered slot fails nominal endpoint IK for this pilot") from exc
    return tuple(result)


CUSTOM_STATE_SCHEMA = "raccoon-custom-slot-scene-state.v1"
# This predecessor uses the same registered coordinates and MuJoCo state representation.
_COMPATIBLE_PHYSICAL_STATE_SOURCES = frozenset((
    "5e5a1166bbad73f2f61a0d35b827db597044fbc0023bf4627ae96f52d48d81c0",
    "736c33072ac33e041f1d0782a47f4ec55797180b9482db0eb76c9814aeb8ae0f",
))


class CustomSlotsSceneMotionSim(RampedSceneMotionSim):
    """Ramped simulator with custom fixed slots, reset once per new scene."""

    def __init__(self, layout, seed=0, randomize=False, *, slot_xy_mm,
                 ramp_seconds=0.5, threshold_degrees=0.0, ramp_shape="minimum_jerk",
                 finger_open_profile=FINGER_OPEN_INSTANT, closed_move_profile=None):
        self._slot_xy_mm = registered_slot_xy_mm(slot_xy_mm)
        super().__init__(layout, seed=seed, randomize=randomize, ramp_seconds=ramp_seconds,
                         threshold_degrees=threshold_degrees, ramp_shape=ramp_shape,
                         finger_open_profile=finger_open_profile, closed_move_profile=closed_move_profile)
        # Complete the initial reset before any observation or motion; retain
        # the independent slot jitter sampled by the base simulator.
        for slot, stack in enumerate(self._reset_info["initial_layout"]):
            delta = np.asarray(self.slot_xy_mm[slot]) / 1000 - np.asarray(SLOT_XY_M[slot])
            for obj in stack:
                address = self.model.jnt_qposadr[self._object_joints_by_id[obj]]
                self.data.qpos[address:address + 2] += delta
        mujoco.mj_forward(self.model, self.data)
        self._anchor_positions = {item["id"]: list(item["position_m"]) for item in self._objects()}
        self.metadata["nominal_slot_xy_m"] = [[x / 1000, y / 1000] for x, y in self.slot_xy_mm]
        self.metadata["slot_coordinate_source"] = "explicit_simulation_registration"
        self._reset_info["nominal_slot_xy_mm"] = [list(point) for point in self.slot_xy_mm]

    @property
    def slot_xy_mm(self):
        return self._slot_xy_mm

    def reset(self, layout=None, seed=None, randomize=None):
        layout = self._reset_info["initial_layout"] if layout is None else layout
        seed = self._reset_info["seed"] if seed is None else seed
        randomize = self._reset_info["randomize"] if randomize is None else randomize
        self.__init__(layout, seed=seed, randomize=randomize, slot_xy_mm=self.slot_xy_mm,
                      ramp_seconds=self.ramp_seconds, threshold_degrees=self.threshold_degrees,
                      ramp_shape=self.ramp_shape, finger_open_profile=self.finger_open_profile,
                      closed_move_profile=self.closed_move_profile)
        return self.observation()

    def _assessment_sample(self, after_layout, moved):
        checks, errors, evidence = super()._assessment_sample(after_layout, moved)
        positions = {obj: slot for slot, stack in enumerate(after_layout) for obj in stack}
        for item in evidence["objects"]:
            obj = item["id"]
            slot = positions[obj]
            expected = np.asarray(self.slot_xy_mm[slot]) / 1000
            error = float(np.linalg.norm(np.asarray(item["position_m"][:2]) - expected))
            if obj == moved:
                errors[str(obj)]["xy_m"] = error
                checks[f"object_{obj}.position_xy"] = error <= 0.012
            else:
                checks[f"object_{obj}.registered_slot_xy"] = error <= 0.012
        return checks, errors, evidence

    def save_state(self, path=None):
        state = super().save_state()
        # A base simulator must not load this snapshot and assess against its
        # fixed slots. Only this subclass restores it after configuration checks.
        state["schema"] = CUSTOM_STATE_SCHEMA
        custom = {"slot_xy_mm": [list(point) for point in self.slot_xy_mm],
                  "custom_simulator_code_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                  "ramp_seconds": self.ramp_seconds, "threshold_degrees": self.threshold_degrees,
                  "ramp_shape": self.ramp_shape}
        state["custom_slot_configuration"] = custom
        state["custom_slot_configuration_sha256"] = _digest(custom)
        if path is not None:
            with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
        return state

    def load_state(self, state_or_path):
        state = (json.loads(Path(state_or_path).read_text(encoding="utf-8"))
                 if isinstance(state_or_path, (str, Path)) else copy.deepcopy(state_or_path))
        expected = {"slot_xy_mm": [list(point) for point in self.slot_xy_mm],
                    "custom_simulator_code_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                    "ramp_seconds": self.ramp_seconds, "threshold_degrees": self.threshold_degrees,
                    "ramp_shape": self.ramp_shape}
        configuration = state.get("custom_slot_configuration") if isinstance(state, dict) else None
        producer = configuration.get("custom_simulator_code_sha256") if isinstance(configuration, dict) else None
        if isinstance(producer, str) and producer in _COMPATIBLE_PHYSICAL_STATE_SOURCES:
            expected["custom_simulator_code_sha256"] = producer
        if (not isinstance(state, dict) or state.get("schema") != CUSTOM_STATE_SCHEMA
                or state.get("custom_slot_configuration") != expected
                or state.get("custom_slot_configuration_sha256") != _digest(expected)
                or not isinstance(state.get("reset_and_randomization"), dict)
                or state["reset_and_randomization"].get("nominal_slot_xy_mm") != expected["slot_xy_mm"]):
            raise ValueError("state does not match the custom slot configuration")
        state["schema"] = STATE_SCHEMA
        return super().load_state(state)
