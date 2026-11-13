"""Offline MuJoCo manipulation environment using nominal Raccoon kinematics.

This is an assumed-parameter physics model, not a validated hardware replica.
The only object forces come from gravity and MuJoCo contacts; objects are
positioned by reset but never attached or repositioned during a command.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import mujoco
import numpy as np

from raccoon_control.kinematics import Geometry, JOINT_LIMITS, check_joints


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "configs" / "sim" / "raccoon_pick_place.xml"
SLOT_XY_M = ((0.15, 0.15), (0.15, 0.0), (-0.15, 0.15), (-0.15, 0.0))
HOME_JOINTS_DEG = (0.0, -10.0, -110.0, 30.0)
OBJECT_RADIUS_M = 0.01
OBJECT_HEIGHT_M = 0.02
MAX_OPENING_M = 0.05
STATE_SCHEMA = "raccoon-motion-sim-state.v1"


def _finite_number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its finite permitted range")
    return value


def _named(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing MuJoCo model element: {name}")
    return index


def _png_rgb(path, rgb):
    """Write rendered RGB pixels without a dependency on a second image library."""
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("rendered image must be uint8 RGB")

    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    raw = b"".join(b"\0" + row.tobytes() for row in rgb)
    image = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
             + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    Path(path).write_bytes(image)


class MotionSim:
    """Physical simulation in SI units except explicitly named ``*_deg/mm``.

    ``command`` sets arm and symmetric finger position servos and advances the
    engine. ``evaluate_success`` also advances the simulation while testing a
    continuous stability interval; see its docstring before calling it.
    """

    def __init__(self, xml_path: str | Path = DEFAULT_XML):
        self.xml_path = Path(xml_path).resolve()
        xml_bytes = self.xml_path.read_bytes()
        self.xml_sha256 = hashlib.sha256(xml_bytes).hexdigest()
        # Parse the same byte snapshot that was hashed, not a later file read.
        self.model = mujoco.MjModel.from_xml_string(xml_bytes.decode("utf-8"))
        self.data = mujoco.MjData(self.model)
        self.geometry = Geometry()
        self._joint_ids = [_named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 5)]
        self._finger_ids = [_named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"finger_{side}_slide") for side in ("left", "right")]
        self._arm_actuators = [_named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}_servo") for i in range(1, 5)]
        self._finger_actuators = [_named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger_{side}_servo") for side in ("left", "right")]
        self._object_bodies = [_named(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{i}") for i in range(1, 5)]
        self._object_joints = [_named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{i}_free") for i in range(1, 5)]
        self._tcp_site = _named(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._validate_model_geometry()
        self.metadata = {
            "data_source": "mujoco_simulation", "simulated": True,
            "model_xml_path": str(self.xml_path), "model_xml_sha256": self.xml_sha256,
            "mujoco_version": mujoco.__version__, "hardware_validated": False,
            "interpretation": "Nominal kinematics with estimated physical parameters; not a validated real-robot model.",
            "nominal_geometry": {
                "base_height_m": self.geometry.base_mm / 1000,
                "upper_length_m": self.geometry.upper_mm / 1000,
                "forearm_length_m": self.geometry.forearm_mm / 1000,
                "wrist_to_tcp_local_m": [0.0, -self.geometry.tool_vertical_mm / 1000, self.geometry.tool_forward_mm / 1000],
                "joint_limits_deg": [list(limits) for limits in JOINT_LIMITS],
                "slot_xy_m": [list(xy) for xy in SLOT_XY_M], "table_z_m": 0.0,
            },
            "assumed_parameters": {
                "object_radius_m": OBJECT_RADIUS_M, "object_height_m": OBJECT_HEIGHT_M,
                "object_mass_kg": 0.015, "finger_pad_half_size_m": [0.003, 0.009, 0.009],
                "object_friction": [1.0, 0.01, 0.001], "finger_friction": [1.5, 0.02, 0.002],
                "link_masses_kg": {"base": 0.2, "upper": 0.08, "forearm": 0.06, "wrist": 0.04, "each_finger": 0.012},
                "arm_servo_kp": [15, 18, 15, 8], "arm_servo_kv": [1.0, 1.2, 1.0, 0.5],
                "finger_servo_kp": 300, "finger_servo_kv": 4,
                "contact_model": "MuJoCo frictional contacts; no object attachment",
            },
        }
        self._reset_info = {}
        self.reset()

    def _validate_model_geometry(self):
        for i, (joint_id, (lo, hi)) in enumerate(zip(self._joint_ids, JOINT_LIMITS), 1):
            actual = np.degrees(self.model.jnt_range[joint_id])
            if not np.allclose(actual, [lo, hi], rtol=0, atol=1e-8):
                raise ValueError(f"XML joint{i} limits differ from Geometry")
        expected_tcp = [0, -self.geometry.tool_vertical_mm / 1000, self.geometry.tool_forward_mm / 1000]
        if not np.allclose(self.model.site_pos[self._tcp_site], expected_tcp, rtol=0, atol=1e-12):
            raise ValueError("XML TCP offset differs from Geometry")

    def reset(self, seed=0, counts=(1, 1, 1, 1), xy_jitter_mm=0):
        """Initialize all four free cylinders at registered nominal slots.

        Counts must add to four; layers in the same slot share their sampled XY
        offset. Reset is the only operation that places object free joints.
        """
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(counts, (list, tuple)) or len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 4:
            raise ValueError("counts must be four nonnegative integers totaling four cylinders")
        jitter = _finite_number(xy_jitter_mm, "xy_jitter_mm", minimum=0)
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        for joint, actuator, angle in zip(self._joint_ids, self._arm_actuators, HOME_JOINTS_DEG):
            angle_rad = math.radians(angle)
            self.data.qpos[self.model.jnt_qposadr[joint]] = angle_rad
            self.data.ctrl[actuator] = angle_rad
        for joint, actuator in zip(self._finger_ids, self._finger_actuators):
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.02
            self.data.ctrl[actuator] = 0.02
        object_index = 0
        offsets = rng.uniform(-jitter / 1000, jitter / 1000, size=(4, 2))
        for slot, count in enumerate(counts):
            for layer in range(count):
                address = self.model.jnt_qposadr[self._object_joints[object_index]]
                x, y = np.asarray(SLOT_XY_M[slot]) + offsets[slot]
                self.data.qpos[address:address + 7] = [x, y, OBJECT_HEIGHT_M * (layer + 0.5) + 0.0005, 1, 0, 0, 0]
                object_index += 1
        self._reset_info = {"seed": seed, "counts": list(counts), "xy_jitter_mm": jitter,
                            "slot_offsets_m": offsets.tolist()}
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def _contacts(self):
        records = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            records.append({"geom1": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                            "geom2": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                            "position_m": contact.pos.tolist(), "distance_m": float(contact.dist),
                            "normal_force_n": float(force[0])})
        return records

    def _objects(self):
        objects = []
        for i, body in enumerate(self._object_bodies, 1):
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            objects.append({"id": i, "position_m": self.data.xpos[body].tolist(),
                            "quaternion_wxyz": self.data.xquat[body].tolist(),
                            "linear_velocity_m_s": velocity[3:].tolist(),
                            "angular_velocity_rad_s": velocity[:3].tolist()})
        return objects

    def _opening(self):
        return float(sum(self.data.qpos[self.model.jnt_qposadr[joint]] for joint in self._finger_ids))

    def observation(self):
        """Report simulated encoder values, nominal TCP, object states and contacts."""
        return {"source": "mujoco_simulation", "simulated": True, "time_seconds": float(self.data.time),
                "joints_deg": [math.degrees(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self._joint_ids],
                "tcp_mm": (self.data.site_xpos[self._tcp_site] * 1000).tolist(),
                "gripper_opening_m": self._opening(), "objects": self._objects(), "contacts": self._contacts(),
                "metadata": copy.deepcopy(self.metadata), "reset": copy.deepcopy(self._reset_info)}

    def command(self, joints_deg, gripper_opening_m, seconds=0.5):
        """Apply finite, in-range servo targets and advance for at least seconds."""
        joints = check_joints(joints_deg)
        opening = _finite_number(gripper_opening_m, "gripper_opening_m", minimum=0, maximum=MAX_OPENING_M)
        duration = _finite_number(seconds, "seconds", minimum=0)
        if duration <= 0:
            raise ValueError("seconds must be positive")
        for actuator, angle in zip(self._arm_actuators, joints):
            self.data.ctrl[actuator] = math.radians(angle)
        for actuator in self._finger_actuators:
            self.data.ctrl[actuator] = opening / 2
        steps = max(1, math.ceil(duration / self.model.opt.timestep - 1e-12))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def evaluate_success(self, object_id, destination_slot, destination_level=1, *, stable_seconds=0.5,
                         xy_tolerance_m=0.012, height_tolerance_m=0.005, max_linear_speed_m_s=0.01,
                         max_angular_speed_rad_s=0.3, min_upright_cos=0.9659, min_opening_m=0.025):
        """Advance and verify position, height, upright pose, velocity and open gripper.

        Every sampled state must satisfy the checks throughout the requested
        interval. This method advances MuJoCo while holding the current controls;
        it does not infer object success from the robot's position.
        """
        if type(object_id) is not int or object_id not in range(1, 5):
            raise ValueError("object_id must be 1 through 4")
        if type(destination_slot) is not int or destination_slot not in range(1, 5):
            raise ValueError("destination_slot must be 1 through 4")
        if type(destination_level) is not int or destination_level not in range(1, 5):
            raise ValueError("destination_level must be 1 through 4")
        duration = _finite_number(stable_seconds, "stable_seconds", minimum=0)
        if duration <= 0:
            raise ValueError("stable_seconds must be positive")
        xy_tol = _finite_number(xy_tolerance_m, "xy_tolerance_m", minimum=0)
        z_tol = _finite_number(height_tolerance_m, "height_tolerance_m", minimum=0)
        linear_max = _finite_number(max_linear_speed_m_s, "max_linear_speed_m_s", minimum=0)
        angular_max = _finite_number(max_angular_speed_rad_s, "max_angular_speed_rad_s", minimum=0)
        upright_min = _finite_number(min_upright_cos, "min_upright_cos", minimum=0, maximum=1)
        open_min = _finite_number(min_opening_m, "min_opening_m", minimum=0, maximum=MAX_OPENING_M)
        target_xy = np.asarray(SLOT_XY_M[destination_slot - 1])
        target_z = (destination_level - 0.5) * OBJECT_HEIGHT_M
        body = self._object_bodies[object_id - 1]
        checks = {key: True for key in ("position_xy", "height", "upright", "linear_speed", "angular_speed", "gripper_open")}
        steps = max(1, math.ceil(duration / self.model.opt.timestep - 1e-12))
        start = float(self.data.time)
        for step in range(steps + 1):
            mujoco.mj_forward(self.model, self.data)
            position = self.data.xpos[body]
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            checks["position_xy"] &= bool(np.linalg.norm(position[:2] - target_xy) <= xy_tol)
            checks["height"] &= bool(abs(position[2] - target_z) <= z_tol)
            checks["upright"] &= bool(self.data.xmat[body].reshape(3, 3)[2, 2] >= upright_min)
            checks["linear_speed"] &= bool(np.linalg.norm(velocity[3:]) <= linear_max)
            checks["angular_speed"] &= bool(np.linalg.norm(velocity[:3]) <= angular_max)
            checks["gripper_open"] &= self._opening() >= open_min
            if step < steps:
                mujoco.mj_step(self.model, self.data)
        return {"success": all(checks.values()), "checks": checks, "object_id": object_id,
                "destination_slot": destination_slot, "destination_level": destination_level,
                "stable_interval_seconds": float(self.data.time) - start,
                "target_position_m": [float(target_xy[0]), float(target_xy[1]), target_z],
                "source": "mujoco_simulation"}

    def save_state(self, path=None):
        """Return the complete MuJoCo integration state, optionally saving new JSON."""
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        values = np.empty(mujoco.mj_stateSize(self.model, signature), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, values, signature)
        state = {"schema": STATE_SCHEMA, "model_xml_sha256": self.xml_sha256, "mujoco_version": mujoco.__version__,
                 "state_signature": signature, "state": values.tolist(), "reset": copy.deepcopy(self._reset_info)}
        if path is not None:
            with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
        return state

    def load_state(self, state_or_path):
        """Restore a state made by this exact XML and MuJoCo version."""
        state = json.loads(Path(state_or_path).read_text(encoding="utf-8")) if isinstance(state_or_path, (str, Path)) else state_or_path
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if (not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA or state.get("model_xml_sha256") != self.xml_sha256
                or state.get("mujoco_version") != mujoco.__version__ or state.get("state_signature") != signature):
            raise ValueError("state is not compatible with this model and MuJoCo version")
        try:
            values = np.asarray(state["state"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid integration state") from exc
        if values.shape != (mujoco.mj_stateSize(self.model, signature),) or not np.all(np.isfinite(values)):
            raise ValueError("integration state has the wrong shape or non-finite values")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, values, signature)
        self._reset_info = copy.deepcopy(state.get("reset", {}))
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def render(self, path=None, *, width=640, height=480, camera="overview"):
        """Render an RGB array, or save it as PNG and return the path."""
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive integers")
        with mujoco.Renderer(self.model, width=width, height=height) as renderer:
            renderer.update_scene(self.data, camera=camera)
            image = renderer.render().copy()
        if path is None:
            return image
        _png_rgb(path, image)
        return str(path)
