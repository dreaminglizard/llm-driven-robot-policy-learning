"""Pure, observation-only quality evidence for one directly taught release.

This does not command a simulator, infer an action, or assess the subsequent
one-second transfer.  Callers must bind these snapshots to the actual OPEN
before and after states, check all releases, and assess retreat and DONE.
"""
from __future__ import annotations

import math
from numbers import Real

from raccoon_memory import task_catalog
from raccoon_memory.motion_sim import OBJECT_HEIGHT_M, OBJECT_RADIUS_M, SLOT_XY_M


MAX_XY_MM = 4.0
MAX_NONCONTACT_FACE_GAP_MM = 2.0
# This is the current SceneMotionSim whole-scene upright criterion.
MIN_UPRIGHT_COS = 0.9659
_UNIT_TOL = 1e-12
_ROOT_ROUNDOFF = 1e-12
_HALF_HEIGHT_MM = OBJECT_HEIGHT_M * 500.0
_RADIUS_MM = OBJECT_RADIUS_M * 1000.0

_CHECK_NAMES = (
    "task_context_valid", "before_target_pose_valid", "before_support_pose_valid",
    "after_target_pose_valid", "after_support_pose_valid", "before_contacts_valid", "after_contacts_valid",
    "target_upright", "support_upright", "target_registered_xy_within_4mm", "target_support_xy_within_4mm",
    "after_target_upright", "after_support_upright", "after_target_registered_xy_within_4mm",
    "after_target_support_xy_within_4mm",
    "before_support_positive", "before_left_pad_positive", "before_right_pad_positive",
    "projected_faces_overlap_proven", "noncontact_face_gap_0_to_2mm", "after_support_positive",
)


class _MissingEvidence(ValueError):
    pass


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _MissingEvidence("non_numeric_value")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise _MissingEvidence("nonfinite_value") from None
    if not math.isfinite(result):
        raise _MissingEvidence("nonfinite_value")
    return result


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise _MissingEvidence("invalid_vector")
    return [_finite(x) for x in value]


def _pose(observation, object_id):
    try:
        objects = observation["privileged_simulation_only"]["objects"]
    except (KeyError, TypeError):
        raise _MissingEvidence("missing_objects") from None
    if not isinstance(objects, (list, tuple)):
        raise _MissingEvidence("invalid_objects")
    matches = [item for item in objects if isinstance(item, dict) and type(item.get("id")) is int
               and item["id"] == object_id]
    if len(matches) != 1:
        raise _MissingEvidence("missing_or_duplicate_object")
    obj = matches[0]
    try:
        position = _vector(obj["position_m"], 3)
        w, x, y, z = _vector(obj["quaternion_wxyz"], 4)
    except KeyError:
        raise _MissingEvidence("missing_pose") from None
    qnorm = math.hypot(w, x, y, z)
    if not math.isfinite(qnorm) or abs(qnorm - 1.0) > _UNIT_TOL:
        raise _MissingEvidence("quaternion_not_unit")
    # The quaternion is checked, never silently normalized or replaced.
    axis = [2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)]
    if not all(math.isfinite(v) for v in axis) or abs(math.hypot(*axis) - 1.0) > _UNIT_TOL:
        raise _MissingEvidence("axis_not_unit")
    uz = axis[2]
    sin_squared = 1.0 - uz * uz
    if sin_squared < -_ROOT_ROUNDOFF:
        raise _MissingEvidence("axis_vertical_component_invalid")
    # Correct only microscopic negative square-root roundoff.
    sin_tilt = math.sqrt(max(0.0, sin_squared))
    # Cylinder geometry is unchanged if its undirected axis is reversed.
    up_axis = axis if uz >= 0 else [-v for v in axis]
    pos_mm = [v * 1000.0 for v in position]
    if not all(math.isfinite(v) for v in pos_mm):
        raise _MissingEvidence("position_overflow")
    vertical_radius = _HALF_HEIGHT_MM * abs(uz) + _RADIUS_MM * sin_tilt
    face_offset = _HALF_HEIGHT_MM * abs(uz)
    face_rise = _RADIUS_MM * sin_tilt
    result = {"position_mm": pos_mm, "quaternion_norm": qnorm, "axis": axis,
        "axis_norm": math.hypot(*axis), "axis_z": uz,
        "tilt_deg": math.degrees(math.acos(min(1.0, abs(uz)))),
        "upright": uz >= MIN_UPRIGHT_COS,
        "global_lowest_mm": pos_mm[2] - vertical_radius,
        "global_highest_mm": pos_mm[2] + vertical_radius,
        "bottom_face_highest_mm": pos_mm[2] - face_offset + face_rise,
        "top_face_lowest_mm": pos_mm[2] + face_offset - face_rise,
        "bottom_face_center_xy_mm": [pos_mm[i] - _HALF_HEIGHT_MM * up_axis[i] for i in (0, 1)],
        "top_face_center_xy_mm": [pos_mm[i] + _HALF_HEIGHT_MM * up_axis[i] for i in (0, 1)],
        "face_inscribed_xy_radius_mm": _RADIUS_MM * abs(uz)}
    if not all(math.isfinite(v) for v in (result["global_lowest_mm"], result["global_highest_mm"],
            result["bottom_face_highest_mm"], result["top_face_lowest_mm"],
            *result["bottom_face_center_xy_mm"], *result["top_face_center_xy_mm"])):
        raise _MissingEvidence("geometry_overflow")
    return result


def _contacts(observation, pairs):
    try:
        raw = observation["privileged_simulation_only"]["contacts"]
    except (KeyError, TypeError):
        raise _MissingEvidence("missing_contacts") from None
    if not isinstance(raw, (list, tuple)):
        raise _MissingEvidence("invalid_contacts")
    result = {key: {"count": 0, "positive_count": 0, "normal_forces_n": []} for key in pairs}
    for contact in raw:
        if not isinstance(contact, dict) or not isinstance(contact.get("geom1"), str) or \
           not isinstance(contact.get("geom2"), str) or not contact["geom1"] or not contact["geom2"]:
            raise _MissingEvidence("invalid_contact_pair")
        try:
            force = _finite(contact["normal_force_n"])
        except KeyError:
            raise _MissingEvidence("missing_contact_force") from None
        if force < 0:
            # A negative recorded normal force is invalid evidence, not an
            # implicit zero-force contact or a proven absence of support.
            raise _MissingEvidence("negative_normal_force")
        pair = frozenset((contact["geom1"], contact["geom2"]))
        for key, wanted in pairs.items():
            if pair == wanted:
                value = result[key]
                value["count"] += 1
                value["positive_count"] += int(force > 0)
                value["normal_forces_n"].append(force)
    return result


def _surface_overlap(target, support):
    """Prove overlap with inscribed circles; never equate an inconclusive proof with disjointness."""
    if support is None:  # The horizontal table is infinite in this task's workspace.
        return True, {"surface": "table", "projection_overlap_proof": "horizontal_table"}
    a, b = target["bottom_face_center_xy_mm"], support["top_face_center_xy_mm"]
    distance = math.dist(a, b)
    inner_sum = target["face_inscribed_xy_radius_mm"] + support["face_inscribed_xy_radius_mm"]
    if not all(math.isfinite(x) for x in (distance, inner_sum)):
        raise _MissingEvidence("projection_overflow")
    if distance <= inner_sum:
        proved, proof = True, "projected_inscribed_discs_overlap"
    elif distance > 2 * _RADIUS_MM:
        proved, proof = False, "projected_circumscribed_discs_disjoint"
    else:
        proved, proof = None, "projection_overlap_not_proven"
    return proved, {"bottom_face_center_xy_mm": a, "support_top_face_center_xy_mm": b,
        "face_center_distance_xy_mm": distance, "inscribed_radius_sum_mm": inner_sum,
        "circumscribed_radius_sum_mm": 2 * _RADIUS_MM, "projection_overlap_proof": proof}


def evaluate_release(before_observation, after_observation, *, before_layout, source_slot, destination_slot):
    """Classify the recorded OPEN boundary as eligible, ineligible or unconfirmed.

    Only object positions/quaternions and contacts in ``privileged_simulation_only``
    are read.  The caller supplies the original registered scene and is responsible
    for verifying that these are the actual before/after observations of an OPEN.
    A successful result does not replace the physical assessment or retreat check.
    """
    checks = {name: None for name in _CHECK_NAMES}
    measurements = {"target_id": None, "support": None, "before": {}, "after": {}}
    reasons, failures, unknowns = [], [], []
    result = {"schema": "teacher-release-quality.v1", "status": "unconfirmed", "release_mode": None,
              "checks": checks, "measurements": measurements, "reasons": reasons,
              "physical_success_assessed": False, "retreat_assessed": False}
    try:
        before = task_catalog.scene(before_layout)
        task_catalog.transfer(before, source_slot, destination_slot)
        target_id = before[source_slot - 1][-1]
        support_id = before[destination_slot - 1][-1] if before[destination_slot - 1] else None
    except (TypeError, ValueError, IndexError):
        reasons.append("invalid_task_context")
        return result
    checks["task_context_valid"] = True
    target_geom = f"object_{target_id}_geom"
    support_geom = "table" if support_id is None else f"object_{support_id}_geom"
    measurements["target_id"] = target_id
    measurements["support"] = {"kind": "table" if support_id is None else "object", "object_id": support_id,
                               "geometry": support_geom}
    pairs = {"support": frozenset((target_geom, support_geom)),
             "left_pad": frozenset((target_geom, "finger_left_pad")),
             "right_pad": frozenset((target_geom, "finger_right_pad"))}

    poses, contacts = {}, {}
    for phase, obs in (("before", before_observation), ("after", after_observation)):
        for role, obj_id in (("target", target_id), ("support", support_id)):
            key = f"{phase}_{role}_pose_valid"
            if obj_id is None:
                checks[key] = True  # The table has no object pose to recover.
                poses[(phase, role)] = None
                continue
            try:
                value = _pose(obs, obj_id)
                poses[(phase, role)] = value
                checks[key] = True
                measurements[phase][role] = value
            except _MissingEvidence as exc:
                poses[(phase, role)] = None
                unknowns.append(f"{phase}_{role}_pose_{exc}")
        try:
            contacts[phase] = _contacts(obs, pairs)
            checks[f"{phase}_contacts_valid"] = True
            measurements[phase]["contacts"] = contacts[phase]
        except _MissingEvidence as exc:
            contacts[phase] = None
            unknowns.append(f"{phase}_contacts_{exc}")

    target, support = poses[("before", "target")], poses[("before", "support")]
    if target is not None:
        checks["target_upright"] = target["upright"]
        if not target["upright"]:
            failures.append("target_not_upright")
        dest = SLOT_XY_M[destination_slot - 1]
        try:
            distance = math.dist(target["position_mm"][:2], [float(v) * 1000.0 for v in dest])
            if not math.isfinite(distance):
                raise _MissingEvidence("registered_xy_overflow")
            measurements["target_registered_xy_error_mm"] = distance
            checks["target_registered_xy_within_4mm"] = distance <= MAX_XY_MM
            if distance > MAX_XY_MM:
                failures.append("target_registered_xy_exceeds_4mm")
        except _MissingEvidence as exc:
            unknowns.append(str(exc))
    if support_id is None:
        checks["support_upright"] = True
    elif support is not None:
        checks["support_upright"] = support["upright"]
        if not support["upright"]:
            failures.append("support_not_upright")
        if target is not None:
            distance = math.dist(target["position_mm"][:2], support["position_mm"][:2])
            if math.isfinite(distance):
                measurements["target_support_xy_error_mm"] = distance
                checks["target_support_xy_within_4mm"] = distance <= MAX_XY_MM
                if distance > MAX_XY_MM:
                    failures.append("target_support_xy_exceeds_4mm")
            else:
                unknowns.append("target_support_xy_overflow")

    after_target, after_support = poses[("after", "target")], poses[("after", "support")]
    if after_target is not None:
        checks["after_target_upright"] = after_target["upright"]
        if not after_target["upright"]:
            failures.append("after_target_not_upright")
        distance = math.dist(after_target["position_mm"][:2],
                             [float(v) * 1000.0 for v in SLOT_XY_M[destination_slot - 1]])
        if math.isfinite(distance):
            measurements["after_target_registered_xy_error_mm"] = distance
            checks["after_target_registered_xy_within_4mm"] = distance <= MAX_XY_MM
            if distance > MAX_XY_MM:
                failures.append("after_target_registered_xy_exceeds_4mm")
        else:
            unknowns.append("after_target_registered_xy_overflow")
    if support_id is None:
        checks["after_support_upright"] = True
    elif after_support is not None:
        checks["after_support_upright"] = after_support["upright"]
        if not after_support["upright"]:
            failures.append("after_support_not_upright")
        if after_target is not None:
            distance = math.dist(after_target["position_mm"][:2], after_support["position_mm"][:2])
            if math.isfinite(distance):
                measurements["after_target_support_xy_error_mm"] = distance
                checks["after_target_support_xy_within_4mm"] = distance <= MAX_XY_MM
                if distance > MAX_XY_MM:
                    failures.append("after_target_support_xy_exceeds_4mm")
            else:
                unknowns.append("after_target_support_xy_overflow")

    before_contacts, after_contacts = contacts["before"], contacts["after"]
    # A contact list is not sufficient evidence when the same snapshot lacks
    # the target or the required support object/pose.
    if before_contacts is not None and checks["before_target_pose_valid"] is True and \
       checks["before_support_pose_valid"] is True:
        for label in ("support", "left_pad", "right_pad"):
            checks[f"before_{label}_positive"] = before_contacts[label]["positive_count"] > 0
    # Missing after-object poses make the after snapshot incomplete even if a
    # list of contacts happens to be present.
    if after_contacts is not None and checks["after_target_pose_valid"] is True and \
       checks["after_support_pose_valid"] is True:
        checks["after_support_positive"] = after_contacts["support"]["positive_count"] > 0
        if not checks["after_support_positive"]:
            failures.append("after_expected_support_contact_absent")

    geometry_available = target is not None and (support_id is None or support is not None)
    overlap, face_gap = None, None
    if geometry_available:
        try:
            support_high = 0.0 if support_id is None else support["global_highest_mm"]
            support_face_low = 0.0 if support_id is None else support["top_face_lowest_mm"]
            global_gap = target["global_lowest_mm"] - support_high
            face_gap = target["bottom_face_highest_mm"] - support_face_low
            if not all(math.isfinite(x) for x in (support_high, support_face_low, global_gap, face_gap)):
                raise _MissingEvidence("height_geometry_overflow")
            overlap, proof = _surface_overlap(target, support)
            measurements["geometry"] = {"target_global_lowest_mm": target["global_lowest_mm"],
                "support_global_highest_mm": support_high, "global_lowest_minus_highest_mm": global_gap,
                "target_bottom_face_highest_mm": target["bottom_face_highest_mm"],
                "support_top_face_lowest_mm": support_face_low, "conservative_face_gap_mm": face_gap,
                **proof}
            checks["projected_faces_overlap_proven"] = overlap
        except _MissingEvidence as exc:
            unknowns.append(str(exc))

    supported = checks["before_support_positive"]
    if supported is True:
        result["release_mode"] = "supported"
    elif supported is False:
        result["release_mode"] = "small_gap"
        for side in ("left", "right"):
            if checks[f"before_{side}_pad_positive"] is False:
                failures.append(f"before_{side}_pad_positive_absent")
        if overlap is False:
            failures.append("projected_faces_proven_disjoint")
        elif overlap is None:
            unknowns.append("projected_faces_overlap_unconfirmed")
        if face_gap is None:
            unknowns.append("noncontact_face_gap_unavailable")
        elif face_gap < 0:
            unknowns.append("negative_noncontact_face_gap")
        else:
            checks["noncontact_face_gap_0_to_2mm"] = face_gap <= MAX_NONCONTACT_FACE_GAP_MM
            if face_gap > MAX_NONCONTACT_FACE_GAP_MM:
                failures.append("noncontact_face_gap_exceeds_2mm")
    else:
        unknowns.append("before_support_contact_unconfirmed")

    reasons.extend(dict.fromkeys(failures + unknowns))
    result["status"] = "ineligible" if failures else "unconfirmed" if unknowns else "eligible"
    return result
