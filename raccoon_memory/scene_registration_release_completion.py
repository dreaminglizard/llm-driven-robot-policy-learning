"""Simulation-only release completion for the frozen registration student.

An OPEN is a pending student decision until the separately guarded child
commands have returned.  All geometry is nominal and conditional on the
registered scene and the earlier closed-gripper observation.  Neither the
student nor this module observes physical object state.  A simulated command
advances one second; its synchronous API does not provide a wall-clock timeout
or an interruptible command.  An exception or missing response is unconfirmed
and never causes a retry.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math

import torch
from torch import nn

from raccoon_control.kinematics import Geometry
from raccoon_control.near_joint_limit_policy import NearJointLimitPolicy
from . import motion_policy, scene_features, scene_policy, task_catalog
from . import scene_oracle_free_session as base
from . import scene_registration_nominal_correction as nominal
from . import scene_registration_distilled_policy as student_policy
from .nominal_bounded_ik import inverse_nearest_feasible_pitch
from .scene_placement_belief_policy import object_geometry
from .scene_model_session import UnsupportedStep


XY_TOLERANCE_MM = 2.5
GAP_MAX_MM = 2.5
APPROACH_GAP_MM = 1.25
PATH_CLEARANCE_MM = 10.0
MAX_CHILD_MOVES = 3
# The simulated finger boxes are 3 x 9 x 9 mm half-extents and their
# centres can be at most 28 mm sideways from the nominal TCP.  This sphere
# bounds their horizontal footprint for every permitted finger opening.
FINGER_XY_RADIUS_MM = math.sqrt(31.0**2 + 9.0**2 + 9.0**2)
FINGER_VERTICAL_RADIUS_MM = math.sqrt(9.0**2 + 9.0**2)
OBJECT_RADIUS_MM = 10.0
PATH_SAMPLE_DEG = 0.1
# This is the existing IK forward-verification precision, not a clearance.
IK_NUMERICAL_MM = 1e-5


class ReleaseStopped(RuntimeError):
    """The parent OPEN cannot complete; the session must remain uncertain."""


def _joints(values):
    q = list(student_policy._vector(values, "public joints"))
    if not base._in_range(q):
        raise ReleaseStopped("observed or proposed joints are out of range")
    return q


def _observed_joints(values):
    # The original student accepts any finite public encoder sample.  Its
    # joint bounds constrain commanded targets, not measured positions.
    return list(student_policy._vector(values, "public joints"))


def _closed_observation(observation):
    if (not isinstance(observation, dict) or observation.get("sensor_valid") is not True
            or observation.get("gripper_reported_valid") is not True
            or observation.get("gripper_feedback_fresh") is not True
            or observation.get("gripper_reported_state") != "closed"):
        raise ReleaseStopped("a fresh closed-gripper public observation is required")
    return _observed_joints(observation.get("joints_deg"))


def _open_observation(observation):
    if (not isinstance(observation, dict) or observation.get("sensor_valid") is not True
            or observation.get("gripper_reported_valid") is not True
            or observation.get("gripper_feedback_fresh") is not True
            or observation.get("gripper_reported_state") != "open"):
        raise ReleaseStopped("a fresh open-gripper public observation is required")
    return _observed_joints(observation.get("joints_deg"))


def _public_return_observation(observation):
    """Require an actual fresh report, not agreement with the open command.

    The existing execution contract records commanded and reported gripper
    states separately.  A fresh closed report after a confirmed OPEN is still
    the reported state; it is not fabricated into an open report or treated
    as physical proof that the released object settled.
    """
    if (not isinstance(observation, dict) or observation.get("sensor_valid") is not True
            or observation.get("gripper_reported_valid") is not True
            or observation.get("gripper_feedback_fresh") is not True
            or observation.get("gripper_reported_state") not in ("open", "closed")):
        raise ReleaseStopped("a fresh public gripper return report is required")
    return _observed_joints(observation.get("joints_deg"))


def _near(a, b, tolerance=3e-5):
    return len(a) == len(b) and all(abs(x-y) <= tolerance for x, y in zip(a, b))


class _FixedMove(nn.Module):
    """Supply one child proposal to the existing guard, never to the student."""

    history_length, feature_count = 8, 82

    def __init__(self, joints):
        super().__init__()
        self.joints = joints

    def forward(self, _windows, _masks):
        return torch.tensor([[1., -1., -1., -1.]]), torch.tensor([self.joints], dtype=torch.float32) / 180.


def _guard_child(proposal):
    # Use the exact existing guard, including its inward floating-point round.
    guard = NearJointLimitPolicy(_FixedMove(list(proposal)))
    logits, angles = guard(None, None)
    if int(logits.argmax()) != motion_policy.MOVE:
        raise ReleaseStopped("child guard changed the command mode")
    target = [float(x)*180. for x in angles[0].tolist()]
    if not base._in_range(target):
        raise ReleaseStopped("child guard rejected the proposal")
    return target, copy.deepcopy(guard.events[-1])


def _geometry(correction, task, joints):
    belief = correction.belief
    if (correction._task != task or belief.active is None or belief.held != task["object_id"]
            or belief.active.get("object") != task["object_id"] or belief.offset is None
            or correction._tracking_error is not None):
        raise ReleaseStopped("the target, destination or closed grasp estimate is ambiguous")
    slots = correction._nominal_slots
    if (not isinstance(slots, list) or len(slots) != 4 or task["destination_slot"] not in range(1, 5)
            or task["destination_level"] not in (1, 2)):
        raise ReleaseStopped("the registered destination is ambiguous")
    height = belief.object_height_mm
    if not math.isfinite(height) or height <= 0:
        raise ReleaseStopped("the nominal object height is invalid")
    try:
        carried = object_geometry(belief.offset)
        center = list(carried.forward(joints))
        target_xy = list(slots[task["destination_slot"]-1])
        support = (task["destination_level"]-1)*height
        obstacles = []
        for obj, point in belief.points.items():
            if obj != belief.held:
                if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
                    raise ValueError("invalid obstacle")
                obstacles.append((float(point[0]), float(point[1]), float(point[2])+height/2.))
        if not all(math.isfinite(float(v)) for v in [*center, *target_xy, support]):
            raise ValueError("invalid geometry")
    except (ValueError, TypeError, IndexError) as exc:
        raise ReleaseStopped("the nominal scene geometry is ambiguous") from exc
    return carried, center, target_xy, support, obstacles


def _path_clear(start, end, carried, height, obstacles):
    """Conservatively certify the complete nominal joint-interpolated path.

    Each midpoint has an analytic Lipschitz bound to every point in its
    interval, so a gap between samples cannot hide an obstacle encounter.
    The checks use nominal TCP/finger boxes and a nominal upright cylinder;
    they are not an assertion about actual servo tracking or object motion.
    """
    start, end = _joints(start), _joints(end)
    count = max(1, math.ceil(max(abs(a-b) for a, b in zip(start, end))/PATH_SAMPLE_DEG))
    delta_rad = [abs(math.radians(b-a))/count/2. for a, b in zip(start, end)]
    nominal_tcp = Geometry()
    def bound(geom):
        tool = abs(geom.tool_forward_mm)+abs(geom.tool_vertical_mm)
        # Maximum displacement under independent joint rotations.
        return ((geom.upper_mm+geom.forearm_mm+tool)*(delta_rad[0]+delta_rad[1])
                +(geom.forearm_mm+tool)*delta_rad[2] + tool*delta_rad[3])
    tcp_bound, object_bound = bound(nominal_tcp), bound(carried)
    pitch_bound = sum(delta_rad[1:])
    finger_z_bound = tcp_bound + FINGER_VERTICAL_RADIUS_MM*pitch_bound
    for index in range(count):
        fraction = (index+.5)/count
        q = [a+(b-a)*fraction for a, b in zip(start, end)]
        tcp, obj = nominal_tcp.forward(q), carried.forward(q)
        pitch = math.radians(sum(q[1:])+90.)
        finger_low = tcp[2] - 9.*(abs(math.sin(pitch))+abs(math.cos(pitch))) - finger_z_bound
        object_low = obj[2] - height/2. - object_bound
        if min(finger_low, object_low) < PATH_CLEARANCE_MM:
            return False
        for ox, oy, top in obstacles:
            near_finger = math.dist(tcp[:2], (ox, oy)) <= FINGER_XY_RADIUS_MM + OBJECT_RADIUS_MM + tcp_bound
            near_object = math.dist(obj[:2], (ox, oy)) <= 2*OBJECT_RADIUS_MM + object_bound
            if (near_finger and finger_low < top+PATH_CLEARANCE_MM) or (near_object and object_low < top+PATH_CLEARANCE_MM):
                return False
    return True


def _vertical_path_diagnostic(start, end, carried, height, obstacles, target_xy, support):
    """Detect explicit sampled nominal penetration, allowing support contact.

    Only a strict cylinder/plane penetration at a sampled commanded posture is
    rejected.  Finger/obstacle bounding-sphere overlap is reported as
    *possible* and is not treated as proven penetration.  This is not a
    continuous collision certificate or a model of physical servo tracking.
    """
    start, end = _joints(start), _joints(end)
    steps = max(1, math.ceil(max(abs(a-b) for a, b in zip(start, end))/PATH_SAMPLE_DEG))
    tool = abs(carried.tool_forward_mm)+abs(carried.tool_vertical_mm)
    # The CLOSE offset was established from float32 normalized features.
    # Bound their angle rounding with the same FK derivative envelope used by
    # the path certificate, in addition to the existing IK tolerance.
    float32_angle_error = math.radians(180.*2.**-24)
    numeric = IK_NUMERICAL_MM + float32_angle_error * (
        2.*(carried.upper_mm+carried.forearm_mm+tool)+(carried.forearm_mm+tool)+tool)
    first, last = carried.forward(start), carried.forward(end)
    max_xy_deviation = 0.
    min_floor = math.inf
    possible_finger = set()
    for index in range(steps+1):
        ratio = index/steps
        q = [a+(b-a)*ratio for a, b in zip(start, end)]
        point, tcp = carried.forward(q), Geometry().forward(q)
        expected_xy = [first[i]+(last[i]-first[i])*ratio for i in range(2)]
        max_xy_deviation = max(max_xy_deviation, math.dist(point[:2], expected_xy))
        pitch = math.radians(sum(q[1:])+90.)
        radius_z = 9.*(abs(math.sin(pitch))+abs(math.cos(pitch)))
        finger_low, finger_high = tcp[2]-radius_z, tcp[2]+radius_z
        obj_low, obj_high = point[2]-height/2., point[2]+height/2.
        min_floor = min(min_floor, obj_low, finger_low)
        if obj_low < -numeric or finger_low < -numeric:
            raise ReleaseStopped("a sampled nominal vertical path penetrates the table")
        for ox, oy, top in obstacles:
            dx = math.dist(point[:2], (ox, oy))
            if (dx < 2.*OBJECT_RADIUS_MM-numeric and obj_low < top-numeric
                    and obj_high > top-height+numeric):
                role = ("destination support" if math.dist((ox, oy), target_xy) <= IK_NUMERICAL_MM
                        and math.isclose(top, support, abs_tol=IK_NUMERICAL_MM, rel_tol=0) else "other obstacle")
                raise ReleaseStopped("a sampled nominal vertical path penetrates " + role)
            if (math.dist(tcp[:2], (ox, oy)) <= FINGER_XY_RADIUS_MM+OBJECT_RADIUS_MM
                    and finger_low < top and finger_high > top-height):
                role = ("destination_support" if math.dist((ox, oy), target_xy) <= IK_NUMERICAL_MM
                        and math.isclose(top, support, abs_tol=IK_NUMERICAL_MM, rel_tol=0) else "other_obstacle")
                possible_finger.add(role)
    return {"method": "sampled_nominal_joint_target_interpolation",
            "samples": steps+1, "max_xy_deviation_from_endpoint_interpolation_mm": max_xy_deviation,
            "minimum_sampled_floor_height_mm": min_floor,
            "numeric_uncertainty_mm": numeric,
            "possible_finger_obstacle_roles": sorted(possible_finger),
            "continuous_collision_certificate": False, "physical_tracking_checked": False}


def _next_child(correction, task, observed_joints, applied_target, *, rise_attempted=False):
    carried, center, target_xy, support, obstacles = _geometry(correction, task, observed_joints)
    height = correction.belief.object_height_mm
    error = math.dist(center[:2], target_xy)
    gap = center[2]-height/2.-support
    diagnostic = {"center_mm": center, "destination_xy_mm": target_xy, "support_mm": support,
                  "xy_error_mm": error, "bottom_gap_mm": gap,
                  "observed_start_joints_deg": list(observed_joints),
                  "applied_start_target_deg": list(applied_target),
                  "start_target_observation_delta_deg": [float(a)-float(b) for a, b in
                                                         zip(applied_target, observed_joints)]}
    if error <= XY_TOLERANCE_MM and 0. <= gap <= GAP_MAX_MM:
        return None, diagnostic
    def guarded_ik(desired):
        raw, ik = inverse_nearest_feasible_pitch([*desired, sum(observed_joints[1:])+90.], geometry=carried)
        guarded, guard_event = _guard_child(raw)
        return raw, guarded, guard_event, ik

    if error <= XY_TOLERANCE_MM:
        phase = "approach"
        desired = [center[0], center[1], support+height/2.+APPROACH_GAP_MM]
    else:
        # Test the actual returned posture first.  A small shortfall from the
        # requested rise target must not consume another MOVE if the entire
        # horizontal joint interpolation is already certified clear.
        align = [*target_xy, center[2]]
        try:
            raw, guarded, guard_event, ik = guarded_ik(align)
            # The ramped simulator interpolates *servo targets* from the
            # previously applied target, not from the observed encoder pose.
            if _path_clear(applied_target, guarded, carried, height, obstacles):
                diagnostic.update(phase="align", desired_center_mm=align, raw_child_target_deg=raw,
                                  guarded_child_target_deg=guarded, child_guard=guard_event, ik=ik)
                return guarded, diagnostic
        except (ValueError, TypeError, OverflowError, ReleaseStopped):
            pass
        if rise_attempted:
            raise ReleaseStopped("the returned posture cannot establish a clear horizontal path")
        # A global obstacle envelope is conservative: it covers every slot,
        # not just those that happen to be crossed by the eventual path.
        highest = max([0.] + [top for _, _, top in obstacles])
        offset_norm = math.sqrt(sum(float(x)**2 for x in correction.belief.offset))
        safe_center = highest + PATH_CLEARANCE_MM + max(height/2., offset_norm+FINGER_VERTICAL_RADIUS_MM)
        if center[2] >= safe_center:
            raise ReleaseStopped("the nominal horizontal joint path lacks 10 mm clearance")
        phase, desired = "rise", [center[0], center[1], safe_center]
    try:
        raw, guarded, guard_event, ik = guarded_ik(desired)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ReleaseStopped("the nominal child path has no allowed IK solution") from exc
    vertical_path = _vertical_path_diagnostic(applied_target, guarded, carried, height, obstacles, target_xy, support)
    diagnostic.update(phase=phase, desired_center_mm=desired, raw_child_target_deg=raw,
                      guarded_child_target_deg=guarded, child_guard=guard_event, ik=ik, vertical_path=vertical_path)
    return guarded, diagnostic


class _PendingOpen:
    """Bridge confirmed child motion to an unchanged, strictly checked OPEN.

    The student adapter has no child-command API.  This bridge updates only
    its expected last target and the pending OPEN's actual pre-command sample;
    it leaves the requested mode, hidden state and normal OPEN callback check
    intact.  A new student decision is impossible while OPEN is pending.
    """

    def __init__(self, guard, task):
        if not isinstance(guard, nominal.RegistrationNominalGuard):
            raise ReleaseStopped("the frozen student's nominal guard is required")
        self.guard, self.correction = guard, guard.policy
        self.adapter = self.correction.registration_adapter
        self.task = task
        self._consumed_children = set()
        self._require_pending()

    def _require_pending(self):
        pending = self.adapter._pending
        if (not isinstance(pending, dict) or pending.get("action") != motion_policy.OPEN
                or not self.guard.events or self.guard.events[-1].get("model_action") != "open"
                or not isinstance(self.correction.last_motion_trace, dict)
                or self.correction.last_motion_trace.get("student_action") != "open"):
            raise ReleaseStopped("the parent OPEN is not pending")

    def confirmed_move(self, sent, applied, opening, observation, record):
        self._require_pending()
        sent, applied = _joints(sent), _joints(applied)
        child_id = record.get("command_id") if isinstance(record, dict) else None
        if (type(child_id) is not int or child_id in self._consumed_children
                or record.get("status") != "confirmed_applied" or record.get("mode") != "move"
                or record.get("parent_requested_mode") != "open"
                or not _near(sent, record.get("target_deg", ()))
                or not _near(applied, record.get("actual_applied_joints_deg", ()))
                or not _near(sent, applied) or not math.isclose(opening, 0., abs_tol=1e-9, rel_tol=0)
                or not math.isclose(self.adapter._expected_opening, 0., abs_tol=1e-9, rel_tol=0)):
            raise ReleaseStopped("confirmed child MOVE does not match its closed command")
        self._consumed_children.add(child_id)
        # A confirmed applied target must survive even if the public return
        # observation is missing and the transfer subsequently stops.
        self.adapter._expected_target = tuple(applied)
        self.adapter._expected_opening = float(opening)
        joints = _closed_observation(observation)
        if self.correction.belief.held != self.task["object_id"]:
            raise ReleaseStopped("closed nominal grasp changed during a child MOVE")
        self.correction.belief.observe(joints, "closed")
        self.adapter._pending["pre_joints"] = tuple(joints)

    def prepare_open(self, before_observation, applied_target, *, duplicate):
        self._require_pending()
        target = _joints(applied_target)
        if not _near(target, self.adapter._expected_target):
            raise ReleaseStopped("OPEN does not use the last confirmed target")
        joints = _open_observation(before_observation) if duplicate else _closed_observation(before_observation)
        if not duplicate:
            self.correction.belief.observe(joints, "closed")
        self.adapter._pending["pre_joints"] = tuple(joints)
        return joints


def _issue(sim, cancel, applied_targets, mode, target, opening, record):
    """Send once; retain confirmed effects before reporting later failures."""
    if "status" in record:
        raise ReleaseStopped("a child command record cannot be sent twice")
    if cancel.is_set():
        record["status"] = "not_sent_cancelled"
        raise InterruptedError("request cancelled before command")
    base._require_servo_targets(sim, applied_targets)
    target = _joints(target)
    record.update(mode=mode, target_deg=list(target), gripper_opening_m=float(opening),
                  seconds=1.0)
    if cancel.is_set():
        record["status"] = "not_sent_cancelled"
        raise InterruptedError("request cancelled immediately before command")
    record["status"] = "sent_unconfirmed"
    started = float(sim.data.time)
    try:
        returned = sim.command(target, opening, seconds=1.0)
        if (not isinstance(returned, dict) or returned.get("source") != "mujoco_simulation"
                or returned.get("simulated") is not True
                or not math.isclose(float(returned["time_seconds"])-started, 1.0, abs_tol=1e-6, rel_tol=0)):
            raise ValueError("simulated command did not return normally")
        actual = base._servo_targets(sim)
        if (not _near(actual["joints_deg"], target, tolerance=1e-7)
                or not math.isclose(actual["gripper_opening_m"], opening, abs_tol=1e-9, rel_tol=0)):
            raise ValueError("returned servo targets differ from the command")
    except Exception as exc:
        record["error_type"] = type(exc).__name__
        raise ReleaseStopped("command return or applied mode is unconfirmed") from exc
    # Do not inspect or persist the simulator's returned object/contact fields.
    record.update(status="confirmed_applied", actual_applied_joints_deg=list(actual["joints_deg"]))
    applied_targets.update(joints_deg=list(actual["joints_deg"]), gripper_opening_m=float(opening))
    try:
        observed = base._robot_observation(sim, None)
        _observed_joints(observed.get("joints_deg"))
        record["return_observation"] = copy.deepcopy(observed)
    except Exception as exc:
        observed = None
        record["return_observation_error"] = type(exc).__name__
    return list(actual["joints_deg"]), observed


def _confirmed_parent(model, action, target, actual, opening, observation):
    adapter = model.policy.registration_adapter
    observer = adapter.return_joint_observer
    adapter.return_joint_observer = (None if observation is None else lambda: list(observation["joints_deg"]))
    try:
        model.on_control_applied(action, list(target), list(actual), opening)
    finally:
        adapter.return_joint_observer = observer


def _complete_open(model, sim, cancel, task, applied_targets, row, auxiliary):
    bridge = _PendingOpen(model, task)
    row["release"] = {"requested_mode": "open", "status": "pending", "children": []}
    release = row["release"]
    rise_attempted = False
    try:
        while True:
            if cancel.is_set():
                raise InterruptedError("request cancelled before release step")
            base._require_servo_targets(sim, applied_targets)
            observed = base._robot_observation(sim, None)
            expected_open = math.isclose(applied_targets["gripper_opening_m"], .04, abs_tol=1e-9, rel_tol=0)
            if expected_open:
                if bridge.correction.belief.held is not None:
                    raise ReleaseStopped("open gripper conflicts with a held-object estimate")
                bridge.prepare_open(observed, applied_targets["joints_deg"], duplicate=True)
                release["duplicate_open"] = True
                break
            if not math.isclose(applied_targets["gripper_opening_m"], 0., abs_tol=1e-9, rel_tol=0):
                raise ReleaseStopped("the gripper command is ambiguous")
            joints = _closed_observation(observed)
            # This is still a closed observation, not the post-release belief.
            bridge.correction.belief.observe(joints, "closed")
            child, diagnostic = _next_child(bridge.correction, task, joints, applied_targets["joints_deg"],
                                             rise_attempted=rise_attempted)
            release["last_nominal_check"] = diagnostic
            if child is None:
                bridge.prepare_open(observed, applied_targets["joints_deg"], duplicate=False)
                break
            if auxiliary["moves"] >= MAX_CHILD_MOVES:
                raise ReleaseStopped("the per-transfer child MOVE budget is exhausted")
            child_row = {"parent_requested_mode": "open", "command_id": auxiliary["moves"], "phase": diagnostic["phase"],
                         "geometry": diagnostic}
            release["children"].append(child_row)
            # A sent-but-unconfirmed attempt also consumes the transfer's
            # budget; the transfer stops and never retries such a command.
            auxiliary["moves"] += 1
            if diagnostic["phase"] == "rise":
                rise_attempted = True
            actual, post = _issue(sim, cancel, applied_targets, "move", child, 0., child_row)
            bridge.confirmed_move(child, actual, 0., post, child_row)
            if cancel.is_set():
                raise InterruptedError("request cancelled after confirmed child MOVE")
        if cancel.is_set():
            raise InterruptedError("request cancelled before actual OPEN")
        target = list(applied_targets["joints_deg"])
        child_row = {"parent_requested_mode": "open", "phase": "release"}
        release["children"].append(child_row)
        actual, post = _issue(sim, cancel, applied_targets, "open", target, .04, child_row)
        # An applied OPEN is recorded even when a later observation or a
        # concurrent cancellation makes the transfer uncertain.
        _confirmed_parent(model, "open", target, actual, .04, post)
        if post is None:
            raise ReleaseStopped("OPEN applied but its public return observation is unavailable")
        _public_return_observation(post)
        release["status"] = "completed"
        row["command"] = {"joints_deg": target, "gripper_opening_m": .04, "seconds": 1.0,
                          "source": "release_completion_actual_open"}
        row["result"] = "command_applied"
        if cancel.is_set():
            raise InterruptedError("request cancelled after confirmed OPEN")
    except BaseException as exc:
        release["status"] = "interrupted" if isinstance(exc, InterruptedError) else "unconfirmed_or_stopped"
        release["error_type"] = type(exc).__name__
        release["reason"] = str(exc)
        raise


def _metrics(decisions, start_time, sim):
    commands = []
    for row in decisions:
        if isinstance(row.get("execution"), dict):
            commands.append(row["execution"])
        commands.extend(row.get("release", {}).get("children", []))
    return {"student_decision_count": len(decisions),
            "additional_move_count": sum(item.get("mode") == "move" and item.get("parent_requested_mode") == "open"
                                         and item.get("status") == "confirmed_applied" for item in commands),
            "actual_command_count": sum(item.get("status") == "confirmed_applied" for item in commands),
            "unconfirmed_command_count": sum(item.get("status") == "sent_unconfirmed" for item in commands),
            "simulated_command_time_seconds": float(sim.data.time)-start_time}


def run_release_transfer(model, sim, before, task, *, max_decisions, device, cancel,
                         applied_targets, decisions=None, slot_xy_mm=None):
    """One unscored virtual transfer, with a single pending-OPEN child loop."""
    if not isinstance(model, nominal.RegistrationNominalGuard):
        raise ValueError("the frozen registration student and existing nominal guard are required")
    if decisions is None:
        decisions = []
    before, task = scene_features.transfer_task(before, task)
    slots = base._registered_slots_for_sim(sim, slot_xy_mm)
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("positive student decision limit required")
    if task_catalog.scene(sim._committed_layout) != before or sim.layout_status != "registered":
        raise ValueError("simulator expected scene bookkeeping differs")
    model.eval()
    base._require_servo_targets(sim, applied_targets)
    opening = applied_targets["gripper_opening_m"]
    if base._robot_observation(sim, "open")["gripper_opening_m"] < .025 or opening < .025:
        raise ValueError("a new transfer must start with an open gripper")
    if cancel.is_set():
        raise InterruptedError("request cancelled before transfer")
    model.on_transfer_started(before, task, copy.deepcopy(applied_targets))
    sim.begin_transfer(before, task["source_slot"], task["destination_slot"])
    history, gripper, auxiliary = [], "open", {"moves": 0}
    start_time = float(sim.data.time)
    for index in range(max_decisions):
        if cancel.is_set():
            raise InterruptedError("request cancelled before student decision")
        base._require_servo_targets(sim, applied_targets)
        last_target = list(applied_targets["joints_deg"])
        features = scene_features.encode_scene_features(base._robot_observation(sim, gripper), before, task, slots)
        history.append(features)
        window, mask = scene_features.history_window(history, model.history_length)
        windows = torch.tensor([window], dtype=torch.float32, device=device)
        masks = torch.tensor([mask], dtype=torch.bool, device=device)
        guard_count = len(model.events)
        with torch.inference_mode():
            logits, angles = model.forward_with_control_state(windows, masks,
                torch.tensor([last_target], dtype=torch.float32, device=device))
        if (not isinstance(logits, torch.Tensor) or not isinstance(angles, torch.Tensor)
                or tuple(logits.shape) != (1, 4) or tuple(angles.shape) != (1, 4)
                or not bool(torch.isfinite(logits).all() and torch.isfinite(angles).all())
                or len(model.events) != guard_count+1):
            raise ValueError("student or guard returned invalid outputs")
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        guard_event = model.events[-1]
        row = {"index": index, "policy_features": features, "valid_history": mask,
               "action_logits": logits.detach().cpu().tolist(), "absolute_joint_output": angles.detach().cpu().tolist(),
               "previous_applied_target_deg": last_target, "action": action, "guard": copy.deepcopy(guard_event)}
        decisions.append(row)
        if cancel.is_set():
            row["result"] = "no_command"
            raise InterruptedError("request cancelled before command")
        base._require_servo_targets(sim, applied_targets)
        if action == "done":
            row["result"] = "done_no_command"
            return {"termination": "model_done", "decisions": decisions, "physical_success": None,
                    "teacher_assistance_count": 0, "actual_hardware": False,
                    **_metrics(decisions, start_time, sim)}
        if action == "open":
            _complete_open(model, sim, cancel, task, applied_targets, row, auxiliary)
            gripper = "open"
            row["guard"] = copy.deepcopy(guard_event)
            continue
        if action == "move":
            target = [float(value)*180. for value in angles[0].tolist()]
            opening = applied_targets["gripper_opening_m"]
        else:
            target, opening = list(last_target), 0.
        if not base._in_range(target):
            row["result"] = "no_command"
            return {"termination": "out_of_range_target", "decisions": decisions, "physical_success": None,
                    "teacher_assistance_count": 0, "actual_hardware": False,
                    **_metrics(decisions, start_time, sim)}
        execution = row["execution"] = {"parent_requested_mode": action}
        row["result"] = "command_unconfirmed"
        actual, observed = _issue(sim, cancel, applied_targets, action, target, opening, execution)
        _confirmed_parent(model, action, target, actual, opening, observed)
        row["command"] = {"joints_deg": list(target), "gripper_opening_m": opening, "seconds": 1.0,
                          "source": "shared_student_policy"}
        row["result"] = "command_applied"
        row["guard"] = copy.deepcopy(guard_event)
        if action == "close":
            gripper = "closed"
        if observed is None:
            raise ReleaseStopped("command applied but the public return observation is unavailable")
        if cancel.is_set():
            raise InterruptedError("request cancelled after confirmed command")
    return {"termination": "decision_limit", "decisions": decisions, "physical_success": None,
            "teacher_assistance_count": 0, "actual_hardware": False,
            **_metrics(decisions, start_time, sim)}


class RegistrationReleaseSceneSession(nominal.RegistrationNominalSceneSession):
    """Separate simulator session; original execution and scoring stay frozen."""

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        # This mirrors the existing expected-only commit protocol, but calls
        # the separate release-aware runner rather than its module-global one.
        self._ready()
        before, task = scene_features.transfer_task(self.current, action)
        after = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
        group = scene_policy.split_group_for(before, task)
        if group not in self.train_groups:
            raise UnsupportedStep(f"transfer is outside allowed groups: {group}")
        event = {"event": "transfer", "before": before, "planned_after": after, "task": task,
                 "split_group": group, "result": None, "committed": False,
                 "commit_kind": "expected_only", "physical_success": None}
        self.events.append(event)
        before_state = self.sim.save_state()
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError("request invalidated before transfer")
            with self._scene_view_lock:
                self._current_confirmed_scene = None
                self._unresolved_transfer = {"expected_object_id": task["object_id"],
                                             "object_id_source": "expected_scene_source_top",
                                             "physical_object_id": None,
                                             "source_slot": task["source_slot"],
                                             "destination_slot": task["destination_slot"]}
        decisions = []
        start_time = float(self.sim.data.time)
        try:
            result = run_release_transfer(self.student, self.sim, before, task, max_decisions=self.max_decisions,
                                          device=self.device, cancel=self.cancel, applied_targets=self._applied_targets,
                                          decisions=decisions, slot_xy_mm=self._registered_slot_xy_mm)
        except BaseException as exc:
            event["result"] = {"termination": "interrupted" if isinstance(exc, InterruptedError) else "error",
                               "decisions": decisions, "physical_success": None, "teacher_assistance_count": 0,
                               "actual_hardware": False, "error_type": type(exc).__name__,
                               **_metrics(decisions, start_time, self.sim)}
            raise
        event["result"] = result
        if result["termination"] != "model_done":
            raise RuntimeError("student transfer did not return DONE: " + result["termination"])
        done_state = self.sim.save_state()
        hold_started = float(self.sim.data.time)
        base._hold_one_second(self.sim, self.cancel)
        result["simulated_hold_time_seconds"] = float(self.sim.data.time)-hold_started
        result["simulated_total_time_seconds"] = float(self.sim.data.time)-start_time
        base._require_servo_targets(self.sim, self._applied_targets)
        if self.cancel.is_set():
            raise InterruptedError("request cancelled after hold")
        hold_state = self.sim.save_state()
        self.memory.observe(after, write=False)
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError("request invalidated before expected-scene update")
            base._require_servo_targets(self.sim, self._applied_targets)
            with self._scene_view_lock:
                self.sim._committed_layout = task_catalog.scene(after)
                self.sim._pending_transfer = None
                self.sim.layout_status = "registered"
                self.sim.layout_uncertainty_reason = None
                self.current = after
                event["committed"] = True
            checkpoint = {"before": before, "task": task, "before_state": before_state,
                          "done_state": done_state, "after_hold_state": hold_state}
            if self._offline_anchor_rebase_required:
                checkpoint["requires_offline_anchor_rebase"] = True
            self._physical_checkpoints.append(checkpoint)
            callback = getattr(self.student, "on_expected_transfer_committed", None)
            if callable(callback):
                callback(before, task, after)
            if on_transfer_commit is not None:
                on_transfer_commit()
            with self._scene_view_lock:
                self._unresolved_transfer = None


# The frozen loader and its checkpoint fingerprint are unchanged.
load_guarded_checkpoint = nominal.load_guarded_checkpoint
