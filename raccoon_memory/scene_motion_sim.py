"""Variable-cylinder MuJoCo scene and continuous transfer assessment.

Only the requested cylinder bodies and free joints are included in the model.
Symbolic layouts use registered IDs in bottom-to-top order. Object coordinates
and contacts are privileged simulator diagnostics, not automatic policy inputs.
No object position is written after reset or state restoration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from raccoon_control.kinematics import Geometry, JOINT_LIMITS
from raccoon_memory import task_catalog
from raccoon_memory.motion_sim import (
    DEFAULT_XML, HOME_JOINTS_DEG, MAX_OPENING_M, MotionSim, OBJECT_HEIGHT_M,
    OBJECT_RADIUS_M, SLOT_XY_M, _finite_number, _named,
)


STATE_SCHEMA = "raccoon-scene-motion-sim-state.v2"
ROBOT_GEOMS = frozenset(("base_link", "upper_link", "forearm_link", "tool_link",
                         "finger_left_pad", "finger_right_pad"))
MIN_STABLE_SECONDS = 1.0


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _list_layout(layout):
    return [list(stack) for stack in layout]


def _transfer_contract(before, source_slot, destination_slot):
    before = task_catalog.scene(before)
    after = task_catalog.transfer(before, source_slot, destination_slot)
    return {"before_layout": _list_layout(before), "source_slot": source_slot,
            "destination_slot": destination_slot, "after_layout": _list_layout(after),
            "moved_object_id": before[source_slot - 1][-1]}


def _generated_xml(base: bytes, active_ids: tuple[int, ...]) -> bytes:
    root = ET.fromstring(base)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("base XML has no worldbody")
    found = set()
    for body in list(world):
        if body.tag != "body":
            continue
        for object_id in task_catalog.OBJECT_IDS:
            if body.get("name") == f"object_{object_id}":
                if object_id in found:
                    raise ValueError("duplicate cylinder body in base XML")
                found.add(object_id)
                if object_id not in active_ids:
                    world.remove(body)
                break
    if found != set(task_catalog.OBJECT_IDS):
        raise ValueError("base XML does not contain the four registered cylinder bodies")
    return ET.tostring(root, encoding="utf-8")


class SceneMotionSim(MotionSim):
    """Simulate a registered four-slot layout with one to four active IDs.

    ``evaluate_transfer`` compares all non-target positions to the last
    committed physical layout. The initial reset establishes the first anchor;
    a successful assessment establishes the next anchor. Failed assessments do
    not advance the registered layout. Commands never reset or teleport objects.
    """

    def __init__(self, layout, seed: int = 0, randomize: bool = False):
        layout = task_catalog.scene(layout)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(randomize) is not bool:
            raise ValueError("randomize must be a boolean")
        self.active_ids = tuple(sorted(obj for stack in layout for obj in stack))
        self.xml_path = Path(DEFAULT_XML).resolve()
        base = self.xml_path.read_bytes()
        self.base_xml_sha256 = hashlib.sha256(base).hexdigest()
        self.generated_xml = _generated_xml(base, self.active_ids).decode("utf-8")
        self.xml_sha256 = hashlib.sha256(self.generated_xml.encode("utf-8")).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(self.generated_xml)
        self.data = mujoco.MjData(self.model)
        self.geometry = Geometry()
        self._joint_ids = [_named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 5)]
        self._finger_ids = [_named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"finger_{side}_slide")
                            for side in ("left", "right")]
        self._arm_actuators = [_named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}_servo")
                               for i in range(1, 5)]
        self._finger_actuators = [_named(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger_{side}_servo")
                                  for side in ("left", "right")]
        self._object_bodies_by_id = {obj: _named(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{obj}")
                                     for obj in self.active_ids}
        self._object_joints_by_id = {obj: _named(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{obj}_free")
                                     for obj in self.active_ids}
        self._object_bodies = [self._object_bodies_by_id[obj] for obj in self.active_ids]
        self._object_joints = [self._object_joints_by_id[obj] for obj in self.active_ids]
        self._tcp_site = _named(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._validate_model_geometry()
        for obj in task_catalog.OBJECT_IDS:
            if obj not in self.active_ids:
                for kind, name in ((mujoco.mjtObj.mjOBJ_BODY, f"object_{obj}"),
                                   (mujoco.mjtObj.mjOBJ_JOINT, f"object_{obj}_free"),
                                   (mujoco.mjtObj.mjOBJ_GEOM, f"object_{obj}_geom")):
                    if mujoco.mj_name2id(self.model, kind, name) != -1:
                        raise ValueError("inactive object remains in generated model")

        streams = np.random.SeedSequence(seed).spawn(5)
        if randomize:
            mass_rng, friction_rng, servo_rng, joint_rng = (np.random.default_rng(s) for s in streams[:4])
            mass_factors = mass_rng.uniform(0.8, 1.2, size=4)
            friction_factors = friction_rng.uniform(0.8, 1.2, size=6)
            servo_factors = servo_rng.uniform(0.9, 1.1, size=6)
            joint_offsets = joint_rng.uniform(-3.0, 3.0, size=4)
            reset_seed = int(streams[4].generate_state(1, dtype=np.uint64)[0])
            slot_offsets = np.random.default_rng(reset_seed).uniform(-0.0015, 0.0015, size=(4, 2))
        else:
            mass_factors, friction_factors, servo_factors = np.ones(4), np.ones(6), np.ones(6)
            joint_offsets, slot_offsets = np.zeros(4), np.zeros((4, 2))
            reset_seed = seed
        for obj, body in self._object_bodies_by_id.items():
            self.model.body_mass[body] *= mass_factors[obj - 1]
            self.model.body_inertia[body] *= mass_factors[obj - 1]
        for obj in self.active_ids:
            geom = _named(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"object_{obj}_geom")
            self.model.geom_friction[geom] *= friction_factors[obj - 1]
        for index, side in enumerate(("left", "right"), 4):
            geom = _named(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_pad")
            self.model.geom_friction[geom] *= friction_factors[index]
        for actuator, factor in zip(self._arm_actuators + self._finger_actuators, servo_factors):
            self.model.actuator_gainprm[actuator, 0] *= factor
            self.model.actuator_biasprm[actuator, 1] *= factor
        mujoco.mj_setConst(self.model, self.data)
        self.model_parameters = self._model_parameter_snapshot()
        self.model_parameters_sha256 = _digest(self.model_parameters)
        initial_joints = (np.asarray(HOME_JOINTS_DEG) + joint_offsets).tolist()
        self._reset_info = {
            "initial_layout": _list_layout(layout), "active_object_ids": list(self.active_ids),
            "seed": seed, "randomize": randomize, "reset_seed": reset_seed,
            "slot_offsets_m": slot_offsets.tolist(), "initial_joint_offsets_deg": joint_offsets.tolist(),
            "initial_joints_deg": initial_joints,
            "mass_inertia_factors_by_id": {str(obj): float(mass_factors[obj - 1]) for obj in self.active_ids},
            "object_friction_factors_by_id": {str(obj): float(friction_factors[obj - 1]) for obj in self.active_ids},
            "finger_friction_factors": friction_factors[4:].tolist(), "servo_stiffness_factors": servo_factors.tolist(),
            "randomization_ranges": {"slot_xy_jitter_mm": [-1.5, 1.5], "initial_joint_offset_deg": [-3, 3],
                                     "object_mass_and_inertia_factor": [0.8, 1.2],
                                     "object_and_finger_friction_factor": [0.8, 1.2],
                                     "servo_stiffness_factor": [0.9, 1.1]},
            "assumptions": "Simulation parameters are estimates, not measured hardware properties.",
        }
        self.metadata = {
            "source": "mujoco_simulation", "simulated": True, "hardware_validated": False,
            "mujoco_version": mujoco.__version__, "base_model_xml_path": str(self.xml_path),
            "base_model_xml_sha256": self.base_xml_sha256, "generated_model_xml_sha256": self.xml_sha256,
            "model_parameters_sha256": self.model_parameters_sha256, "active_object_ids": list(self.active_ids),
            "object_radius_m": OBJECT_RADIUS_M, "object_height_m": OBJECT_HEIGHT_M,
            "nominal_slot_xy_m": [list(xy) for xy in SLOT_XY_M],
            "interpretation": "Estimated MuJoCo parameters; not a validated real-robot model.",
        }
        mujoco.mj_resetData(self.model, self.data)
        for joint, actuator, angle in zip(self._joint_ids, self._arm_actuators, initial_joints):
            radians = math.radians(angle)
            self.data.qpos[self.model.jnt_qposadr[joint]] = radians
            self.data.ctrl[actuator] = radians
        for joint, actuator in zip(self._finger_ids, self._finger_actuators):
            self.data.qpos[self.model.jnt_qposadr[joint]] = 0.02
            self.data.ctrl[actuator] = 0.02
        for slot, stack in enumerate(layout):
            for layer, obj in enumerate(stack):
                address = self.model.jnt_qposadr[self._object_joints_by_id[obj]]
                x, y = np.asarray(SLOT_XY_M[slot]) + slot_offsets[slot]
                self.data.qpos[address:address + 7] = [x, y, OBJECT_HEIGHT_M * (layer + 0.5) + 0.0005, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self._committed_layout = layout
        self._anchor_positions = {item["id"]: list(item["position_m"]) for item in self._objects()}
        self.layout_status = "registered"
        self.layout_uncertainty_reason = None
        self._pending_transfer = None

    def _model_parameter_snapshot(self):
        objects = {str(obj): {"mass_kg": float(self.model.body_mass[body]),
                              "inertia_kg_m2": self.model.body_inertia[body].tolist()}
                   for obj, body in self._object_bodies_by_id.items()}
        geoms = [f"object_{obj}_geom" for obj in self.active_ids] + ["finger_left_pad", "finger_right_pad"]
        friction = {name: self.model.geom_friction[_named(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)].tolist()
                    for name in geoms}
        actuators = {}
        for actuator in self._arm_actuators + self._finger_actuators:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            actuators[name] = {"gain_parameters": self.model.actuator_gainprm[actuator].tolist(),
                               "bias_parameters": self.model.actuator_biasprm[actuator].tolist()}
        return {"objects": objects, "geom_friction": friction, "actuators": actuators}

    def reset(self, layout=None, seed=None, randomize=None):
        """Explicitly construct a fresh scene; continuous operations do not call it."""
        layout = self._reset_info["initial_layout"] if layout is None else layout
        seed = self._reset_info["seed"] if seed is None else seed
        randomize = self._reset_info["randomize"] if randomize is None else randomize
        self.__init__(layout, seed=seed, randomize=randomize)
        return self.observation()

    def _objects(self):
        objects = []
        for obj, body in self._object_bodies_by_id.items():
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            objects.append({"id": obj, "position_m": self.data.xpos[body].tolist(),
                            "quaternion_wxyz": self.data.xquat[body].tolist(),
                            "linear_velocity_m_s": velocity[3:].tolist(),
                            "angular_velocity_rad_s": velocity[:3].tolist()})
        return objects

    def observation(self):
        """Return arm values plus explicitly privileged object/contact diagnostics."""
        return {"source": "mujoco_simulation", "simulated": True, "time_seconds": float(self.data.time),
                "joints_deg": [math.degrees(self.data.qpos[self.model.jnt_qposadr[j]]) for j in self._joint_ids],
                "tcp_mm": (self.data.site_xpos[self._tcp_site] * 1000).tolist(),
                "gripper_opening_m": self._opening(), "active_object_ids": list(self.active_ids),
                "registered_layout": _list_layout(self._committed_layout),
                "layout_status": self.layout_status, "layout_uncertainty_reason": self.layout_uncertainty_reason,
                "pending_transfer": copy.deepcopy(self._pending_transfer),
                "metadata": copy.deepcopy(self.metadata),
                "privileged_simulation_only": {"objects": self._objects(), "contacts": self._contacts(),
                                               "reset_and_randomization": copy.deepcopy(self._reset_info)}}

    @property
    def pending_transfer(self):
        """Return the current task contract without exposing mutable internal state."""
        return copy.deepcopy(self._pending_transfer)

    def begin_transfer(self, before_layout, source_slot, destination_slot):
        """Bind one task to a trusted registered layout before any commands."""
        if self.layout_status != "registered":
            raise RuntimeError(f"cannot begin a transfer while layout is {self.layout_status}")
        contract = _transfer_contract(before_layout, source_slot, destination_slot)
        if task_catalog.scene(contract["before_layout"]) != self._committed_layout:
            raise ValueError("before_layout differs from the last committed physical layout")
        self._pending_transfer = contract
        self.layout_status = "in_progress"
        self.layout_uncertainty_reason = None
        return copy.deepcopy(contract)

    def mark_layout_uncertain(self, reason):
        """Block further autonomous commands after a failure or cancellation."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("uncertainty reason must be a nonempty string")
        if self.layout_status != "uncertain":
            self.layout_uncertainty_reason = reason.strip()
        self.layout_status = "uncertain"
        return {"layout_status": self.layout_status, "reason": self.layout_uncertainty_reason}

    def command(self, joints_deg, gripper_opening_m, seconds=0.5):
        """Advance only one active task, or enter an unbound legacy operation."""
        if self.layout_status == "uncertain":
            raise RuntimeError("layout is uncertain; explicit reset is required")
        if self.layout_status == "registered":
            # Legacy teachers issue several commands before providing a task to
            # evaluate_transfer. The first such command makes the scene busy.
            self.layout_status = "in_progress"
            self._pending_transfer = None
        try:
            return super().command(joints_deg, gripper_opening_m, seconds)
        except Exception as exc:
            self.mark_layout_uncertain(f"simulation_command_error: {type(exc).__name__}: {exc}")
            raise

    def evaluate_success(self, *args, **kwargs):
        """Prevent use of the legacy fixed-ID, target-only success criterion."""
        raise NotImplementedError("use evaluate_transfer to assess the whole registered scene")

    def _assessment_sample(self, after_layout, moved):
        objects = {item["id"]: item for item in self._objects()}
        contacts = self._contacts()
        expected_support, positions = {}, {}
        allowed_pairs = set()
        for slot, stack in enumerate(after_layout):
            for layer, obj in enumerate(stack):
                positions[obj] = (slot, layer)
                expected_support[obj] = "table" if layer == 0 else f"object_{stack[layer - 1]}_geom"
                if layer:
                    allowed_pairs.add(frozenset((f"object_{obj}_geom", f"object_{stack[layer - 1]}_geom")))
        checks = {"gripper_open": bool(self._opening() >= 0.025)}
        errors = {}
        object_geom_names = {f"object_{obj}_geom" for obj in self.active_ids}
        unexpected = [c for c in contacts if {c["geom1"], c["geom2"]}.issubset(object_geom_names)
                      and frozenset((c["geom1"], c["geom2"])) not in allowed_pairs]
        checks["no_unexpected_cylinder_contact"] = not unexpected
        for obj in self.active_ids:
            item = objects[obj]
            position = np.asarray(item["position_m"], dtype=float)
            quat = np.asarray(item["quaternion_wxyz"], dtype=float)
            linear = np.asarray(item["linear_velocity_m_s"], dtype=float)
            angular = np.asarray(item["angular_velocity_rad_s"], dtype=float)
            if not all(np.isfinite(v).all() for v in (position, quat, linear, angular)):
                raise ValueError("simulation object state contains non-finite values")
            slot, layer = positions[obj]
            if obj == moved:
                reference = np.asarray([*SLOT_XY_M[slot], OBJECT_HEIGHT_M * (layer + 0.5)])
                xy_tolerance, z_tolerance = 0.012, 0.005
            else:
                reference = np.asarray(self._anchor_positions[obj], dtype=float)
                xy_tolerance, z_tolerance = 0.008, 0.003
            delta = position - reference
            xy_error, z_error = float(np.linalg.norm(delta[:2])), float(abs(delta[2]))
            errors[str(obj)] = {"xy_m": xy_error, "height_m": z_error}
            geom, support = f"object_{obj}_geom", expected_support[obj]
            support_contacts = [c for c in contacts if {c["geom1"], c["geom2"]} == {geom, support}
                                and c["normal_force_n"] > 0]
            robot_contacts = [c for c in contacts if geom in (c["geom1"], c["geom2"])
                              and bool({c["geom1"], c["geom2"]} & ROBOT_GEOMS)]
            prefix = f"object_{obj}"
            checks[f"{prefix}.position_xy"] = xy_error <= xy_tolerance
            checks[f"{prefix}.height"] = z_error <= z_tolerance
            if obj != moved:
                registered_xy = np.asarray(SLOT_XY_M[slot], dtype=float)
                registered_z = OBJECT_HEIGHT_M * (layer + 0.5)
                checks[f"{prefix}.registered_slot_xy"] = bool(np.linalg.norm(position[:2] - registered_xy) <= 0.012)
                checks[f"{prefix}.registered_level_height"] = bool(abs(position[2] - registered_z) <= 0.005)
            checks[f"{prefix}.upright"] = bool(1 - 2 * (quat[1] ** 2 + quat[2] ** 2) >= 0.9659)
            checks[f"{prefix}.linear_speed"] = bool(np.linalg.norm(linear) <= 0.01)
            checks[f"{prefix}.angular_speed"] = bool(np.linalg.norm(angular) <= 0.3)
            checks[f"{prefix}.required_support_contact"] = bool(support_contacts)
            checks[f"{prefix}.free_of_robot_contact"] = not robot_contacts
            if obj == moved and layer > 0:
                lower = objects[after_layout[slot][layer - 1]]
                relative = position - np.asarray(lower["position_m"], dtype=float)
                checks[f"{prefix}.aligned_with_support_xy"] = bool(np.linalg.norm(relative[:2]) <= 0.008)
                checks[f"{prefix}.aligned_with_support_height"] = bool(abs(relative[2] - OBJECT_HEIGHT_M) <= 0.003)
        return checks, errors, {"objects": list(objects.values()), "contacts": contacts,
                                "unexpected_cylinder_contacts": unexpected}

    def evaluate_transfer(self, before_layout, source_slot, destination_slot, stable_seconds=1.0):
        """Assess a completed transfer over a continuous interval of at least 1 s.

        Success commits the after-layout and actual positions; failure marks
        the layout uncertain. A legacy command sequence without begin_transfer
        binds its task at this first valid assessment.
        """
        if self.layout_status == "uncertain":
            raise RuntimeError("layout is uncertain; explicit reset is required")
        contract = _transfer_contract(before_layout, source_slot, destination_slot)
        before = task_catalog.scene(contract["before_layout"])
        after = task_catalog.scene(contract["after_layout"])
        if before != self._committed_layout:
            raise ValueError("before_layout differs from the last committed physical layout")
        duration = _finite_number(stable_seconds, "stable_seconds", minimum=MIN_STABLE_SECONDS)
        if self._pending_transfer is not None and self._pending_transfer != contract:
            raise ValueError("assessment differs from the active transfer")
        if self.layout_status == "registered" or self._pending_transfer is None:
            self._pending_transfer = contract
            self.layout_status = "in_progress"
        try:
            return self._evaluate_transfer_interval(before, after, contract, duration)
        except Exception as exc:
            self.mark_layout_uncertain(f"physical_assessment_error: {type(exc).__name__}: {exc}")
            raise

    def _evaluate_transfer_interval(self, before, after, contract, duration):
        source_slot, destination_slot = contract["source_slot"], contract["destination_slot"]
        dt = float(self.model.opt.timestep)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("invalid simulation time step")
        steps = max(1, math.ceil(duration / dt - 1e-12))
        moved = before[source_slot - 1][-1]
        combined, first_failed, maximum = None, {}, {}
        start = float(self.data.time)
        for step in range(steps + 1):
            mujoco.mj_forward(self.model, self.data)
            sample, errors, evidence = self._assessment_sample(after, moved)
            if combined is None:
                combined = dict(sample)
                maximum = copy.deepcopy(errors)
            else:
                if sample.keys() != combined.keys():
                    raise ValueError("assessment checks changed during stability interval")
                combined = {key: combined[key] and value for key, value in sample.items()}
                for obj in errors:
                    for key in errors[obj]:
                        maximum[obj][key] = max(maximum[obj][key], errors[obj][key])
            for key, value in sample.items():
                if not value and key not in first_failed:
                    first_failed[key] = float(self.data.time) - start
            if step < steps:
                mujoco.mj_step(self.model, self.data)
        elapsed = float(self.data.time) - start
        if not math.isfinite(elapsed) or elapsed < duration - 1e-9:
            raise ValueError("stability interval was shorter than requested")
        success = bool(all(combined.values()))
        result = {"source": "mujoco_simulation", "simulated": True, "actual_hardware": False,
                  "success": success, "before_layout": _list_layout(before), "after_layout": _list_layout(after),
                  "source_slot": source_slot, "destination_slot": destination_slot, "moved_object_id": moved,
                  "stable_interval_seconds": elapsed, "sample_count": steps + 1, "checks": combined,
                  "maximum_errors_m": maximum, "first_failed_check_seconds": first_failed,
                  "checks_scope": "Every physics-step boundary throughout the stability interval.",
                  "privileged_simulation_only": {"initial_anchor_positions_m": copy.deepcopy(self._anchor_positions),
                                                  "last_sample": evidence}}
        if success:
            self._committed_layout = after
            self._anchor_positions = {item["id"]: list(item["position_m"]) for item in evidence["objects"]}
            self.layout_status = "registered"
            self.layout_uncertainty_reason = None
            self._pending_transfer = None
        else:
            self.mark_layout_uncertain("physical_assessment_failed")
        result["layout_status_after"] = self.layout_status
        return result

    def save_state(self, path=None):
        """Save integration state, generated XML, model parameters and scene anchors."""
        if _digest(self._model_parameter_snapshot()) != self.model_parameters_sha256:
            raise ValueError("scene model parameters changed after initialization")
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        values = np.empty(mujoco.mj_stateSize(self.model, signature), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, values, signature)
        state = {"schema": STATE_SCHEMA, "mujoco_version": mujoco.__version__,
                 "base_model_xml_sha256": self.base_xml_sha256, "generated_model_xml_sha256": self.xml_sha256,
                 "generated_model_xml": self.generated_xml, "model_parameters_sha256": self.model_parameters_sha256,
                 "model_parameters": copy.deepcopy(self.model_parameters),
                 "scene_simulator_code_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                 "active_object_ids": list(self.active_ids), "state_signature": signature, "state": values.tolist(),
                 "reset_and_randomization": copy.deepcopy(self._reset_info),
                 "committed_layout": _list_layout(self._committed_layout),
                 "anchor_positions_m": {str(obj): list(pos) for obj, pos in self._anchor_positions.items()},
                 "layout_status": self.layout_status, "layout_uncertainty_reason": self.layout_uncertainty_reason,
                 "pending_transfer": copy.deepcopy(self._pending_transfer)}
        state["layout_safety_sha256"] = _digest({key: state[key] for key in (
            "committed_layout", "anchor_positions_m", "layout_status", "layout_uncertainty_reason", "pending_transfer")})
        if path is not None:
            with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(state, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write("\n")
        return state

    def load_state(self, state_or_path):
        """Restore only a matching generated XML, physical model and MuJoCo version."""
        state = (json.loads(Path(state_or_path).read_text(encoding="utf-8"))
                 if isinstance(state_or_path, (str, Path)) else state_or_path)
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if (not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA
                or state.get("mujoco_version") != mujoco.__version__
                or state.get("base_model_xml_sha256") != self.base_xml_sha256
                or state.get("generated_model_xml_sha256") != self.xml_sha256
                or state.get("generated_model_xml") != self.generated_xml
                or state.get("model_parameters_sha256") != self.model_parameters_sha256
                or _digest(state.get("model_parameters")) != self.model_parameters_sha256
                or _digest(self._model_parameter_snapshot()) != self.model_parameters_sha256
                or state.get("scene_simulator_code_sha256") != hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
                or state.get("active_object_ids") != list(self.active_ids)
                or state.get("state_signature") != signature):
            raise ValueError("state does not match this scene model and physical parameters")
        try:
            layout = task_catalog.scene(state["committed_layout"])
            values = np.asarray(state["state"], dtype=np.float64)
            positions = {int(obj): np.asarray(pos, dtype=np.float64) for obj, pos in state["anchor_positions_m"].items()}
            status = state["layout_status"]
            reason = state["layout_uncertainty_reason"]
            pending = state["pending_transfer"]
            safety = {key: state[key] for key in ("committed_layout", "anchor_positions_m", "layout_status",
                                                   "layout_uncertainty_reason", "pending_transfer")}
            if state["layout_safety_sha256"] != _digest(safety):
                raise ValueError("layout safety fields failed integrity check")
            if status not in ("registered", "in_progress", "uncertain"):
                raise ValueError("invalid layout status")
            if status == "registered" and (pending is not None or reason is not None):
                raise ValueError("registered scene cannot have a pending task or uncertainty")
            if status == "in_progress" and reason is not None:
                raise ValueError("in-progress scene cannot have an uncertainty reason")
            if status == "uncertain" and (not isinstance(reason, str) or not reason.strip()):
                raise ValueError("uncertain scene requires a reason")
            if pending is not None:
                if not isinstance(pending, dict):
                    raise ValueError("invalid pending transfer")
                canonical = _transfer_contract(pending["before_layout"], pending["source_slot"], pending["destination_slot"])
                if pending != canonical or task_catalog.scene(pending["before_layout"]) != layout:
                    raise ValueError("pending transfer does not match the registered layout")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid scene state") from exc
        if (tuple(sorted(obj for stack in layout for obj in stack)) != self.active_ids
                or set(positions) != set(self.active_ids)
                or any(pos.shape != (3,) or not np.all(np.isfinite(pos)) for pos in positions.values())
                or values.shape != (mujoco.mj_stateSize(self.model, signature),) or not np.all(np.isfinite(values))
                or not isinstance(state.get("reset_and_randomization"), dict)):
            raise ValueError("invalid layout, anchor positions or integration state")
        if self.layout_status == "uncertain" and status != "uncertain":
            raise ValueError("cannot clear an uncertain layout by loading a saved state; reset is required")
        if self.layout_status == "in_progress" and status == "registered":
            raise ValueError("cannot abandon an active transfer by loading a registered state")
        if (self.layout_status == "in_progress" and status == "in_progress"
                and (layout != self._committed_layout or pending != self._pending_transfer)):
            raise ValueError("cannot replace an active transfer by loading a different task")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, values, signature)
        self._reset_info = copy.deepcopy(state["reset_and_randomization"])
        self._committed_layout = layout
        self._anchor_positions = {obj: pos.tolist() for obj, pos in positions.items()}
        self.layout_status = status
        self.layout_uncertainty_reason = reason
        self._pending_transfer = copy.deepcopy(pending)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()
