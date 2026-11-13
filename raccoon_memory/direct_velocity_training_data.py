"""Read completed explicit-velocity journals without running a simulator.

The 50 Hz observed positions and submitted velocities have different origins.
This module never treats simulator qvel as a command, invents demonstrations,
resamples commands, concatenates journals, or authenticates an actor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import scene_features

SCHEMA = "raccoon-submitted-velocity-training.v1"
SAMPLE_SECONDS = 0.02


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _need(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    return float(value)


def _vector(value, name):
    _need(isinstance(value, list) and len(value) == 4, f"invalid {name}")
    return [_number(x, name) for x in value]


def _close(left, right, tolerance=1e-7):
    return len(left) == len(right) and all(math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
                                          for a, b in zip(left, right, strict=True))


def _identity(observation):
    """Only permitted observed state; simulator qvel is deliberately ignored."""
    _need(isinstance(observation, dict) and observation.get("simulated") is True, "invalid public observation")
    return {key: observation.get(key) for key in ("observation_id", "time_seconds", "joints_deg",
        "applied_target_deg", "gripper_command", "gripper_opening_m", "registered_scene", "task",
        "registered_slot_xy_mm", "physical_confirmation")}


def _student_features(before, previous_joints, initial_rest_verified=False):
    joints = _vector(before.get("joints_deg"), "observed joints")
    opening = _number(before.get("gripper_opening_m"), "gripper opening")
    grip = before.get("gripper_command")
    _need(grip in ("open", "closed") and before.get("physical_confirmation") == "not_inspected",
          "unsupported public robot state")
    moving_valid = previous_joints is not None or initial_rest_verified
    moving = (max(abs(a - b) / SAMPLE_SECONDS for a, b in zip(joints, previous_joints, strict=True)) > .1
              if previous_joints is not None else False)
    public = {"schema_version": "PolicyObservation.phase1", "joints_deg": joints, "sensor_valid": True,
              "moving": moving, "gripper_command": grip,
              "gripper_reported_state": "open" if opening >= .03 else "closed",
              "gripper_reported_valid": True, "gripper_feedback_fresh": True}
    return scene_features.encode_scene_features(public, before["registered_scene"], before["task"],
                                                before["registered_slot_xy_mm"]), moving_valid


def from_events(events, *, source_group_id, source_reference):
    """Validate exactly one completed journal and preserve its submitted actions.

    The group must come from caller-supplied lineage evidence. Its validity is
    not inferred from partial public observations. The first moving feature is
    valid only when the journal records and bounds the simulated rest check;
    the qvel used by that check is not a student input.
    """
    _need(isinstance(source_group_id, str) and source_group_id.strip()
          and isinstance(source_reference, str) and source_reference.strip(), "source lineage is required")
    _need(isinstance(events, list) and len(events) >= 4 and events[0].get("event") == "ready"
          and events[-1].get("event") == "finished", "a completed ordered journal is required")
    ready, finished = events[0], events[-1]
    _need(math.isclose(_number(ready.get("sample_seconds"), "sample period"), SAMPLE_SECONDS,
                       rel_tol=0, abs_tol=1e-12)
          and ready.get("feature_names") == list(scene_features.FEATURE_NAMES), "50 Hz feature contract differs")
    _need(finished.get("actor_identity") == "unverified", "unknown actor identity declaration")
    assessment = finished.get("assessment")
    _need(isinstance(assessment, dict) and assessment.get("simulated") is True
          and assessment.get("actual_hardware") is False and type(assessment.get("success")) is bool,
          "invalid completed simulated assessment")
    last_observation = ready.get("observation")
    _identity(last_observation)
    sim_info = ready.get("simulation")
    initial_rest_verified = False
    if isinstance(sim_info, dict) and "initial_moving_source" in sim_info:
        _need(sim_info.get("initial_moving_source") == "verified_initial_simulated_joint_velocity_is_zero"
              and sim_info.get("initial_moving_flag") is False and last_observation.get("moving") is False,
              "inconsistent initial moving evidence")
        tolerance = _number(sim_info.get("initial_rest_tolerance_deg_s"), "initial rest tolerance")
        declared = _vector(sim_info.get("initial_observed_joint_velocity_deg_s"), "initial rest evidence")
        observed = _vector(last_observation.get("joint_velocity_deg_s"), "initial simulator velocity evidence")
        _need(0 <= tolerance <= 1e-9 and all(abs(x) <= tolerance for x in declared)
              and _close(declared, observed, tolerance=1e-12)
              and last_observation.get("joint_velocity_source") == "simulator_observation_not_submitted_command",
              "initial simulated rest evidence differs")
        initial_rest_verified = True
    previous_joints, output, seen_ids, seen_decisions = None, [], set(), set()
    progress_samples, accepted, terminal = [], None, None
    origin = _number(last_observation.get("time_seconds"), "origin")
    for event in events[1:-1]:
        kind = event.get("event")
        if kind == "rejected":
            _need(accepted is None and terminal is None
                  and _identity(event.get("observation")) == _identity(last_observation),
                  "rejected request changed public state")
            continue
        if kind == "batch_accepted":
            _need(accepted is None and terminal is None and isinstance(event.get("data"), dict), "unexpected accepted batch")
            accepted = event["data"]
            _need(accepted.get("observation_id") == last_observation["observation_id"], "stale accepted batch")
            continue
        if kind == "batch_sample":
            _need(accepted is not None and terminal is None and isinstance(event.get("data"), dict), "orphan progress sample")
            progress_samples.append(event["data"])
            continue
        if kind == "finish_requested":
            _need(accepted is None and not progress_samples and terminal is None and isinstance(event.get("terminal"), dict),
                  "invalid explicit finish")
            terminal = event["terminal"]
            continue
        _need(kind == "chunk" and accepted is not None and terminal is None, "unexpected journal event")
        request, result = event.get("request"), event.get("result")
        _need(isinstance(request, dict) and isinstance(result, dict)
              and result.get("schema") == "raccoon-direct-velocity-chunk.v2"
              and result.get("simulated") is True and result.get("source") == "submitted_commands_actor_unverified"
              and result.get("status") == "completed" and result.get("physical_success") is None,
              "invalid direct-command chunk")
        commands, steps, samples = request.get("commands"), result.get("steps"), result.get("samples")
        _need(request.get("observation_id") == last_observation["observation_id"]
              and isinstance(commands, list) and isinstance(steps, list) and isinstance(samples, list)
              and len(commands) == len(steps) > 0 and samples
              and result.get("decisions_submitted") == result.get("decisions_completed") == len(steps)
              and result.get("samples_completed") == len(samples)
              and accepted.get("expected_samples") == len(samples), "chunk counts or request differ")
        # Progress and final chunk contain the SAME samples, not new episodes.
        _need(progress_samples == samples, "progress samples differ from final chunk")
        prepared = accepted.get("decisions")
        _need(isinstance(prepared, list) and len(prepared) == len(steps), "accepted decisions differ")
        sample_index = 0
        for command, step, recorded in zip(commands, steps, prepared, strict=True):
            _need(isinstance(command, dict) and set(command) in
                  ({"joint_velocity_deg_s", "gripper"}, {"joint_velocity_deg_s", "gripper", "duration_seconds"})
                  and command == step.get("submitted") == recorded.get("submitted")
                  and command["gripper"] in ("hold", "open", "closed")
                  and step.get("status") == "completed", "submitted decision differs")
            velocity = _vector(command["joint_velocity_deg_s"], "submitted velocity")
            count = step.get("sample_count")
            duration = _number(step.get("duration_seconds"), "submitted duration")
            decision_id = step.get("decision_id")
            _need(isinstance(decision_id, str) and decision_id and decision_id not in seen_decisions,
                  "duplicate submitted decision ID")
            seen_decisions.add(decision_id)
            _need(type(count) is int and count > 0 and step.get("samples_completed") == count
                  and count == recorded.get("sample_count") and step.get("decision_id") == recorded.get("decision_id")
                  and math.isclose(duration, count * SAMPLE_SECONDS, abs_tol=1e-9)
                  and ("duration_seconds" not in command or math.isclose(command["duration_seconds"], duration, abs_tol=1e-9))
                  and _identity(step.get("before")) == _identity(last_observation), "decision duration or start differs")
            for offset in range(count):
                _need(sample_index < len(samples), "missing decision sample")
                sample = samples[sample_index]
                sample_index += 1
                sid = sample.get("sample_id")
                _need(isinstance(sid, str) and sid and sid not in seen_ids, "duplicate sample ID")
                seen_ids.add(sid)
                _need(sample.get("decision_id") == step["decision_id"] and sample.get("submitted") == command
                      and sample.get("decision_sample_index") == offset and sample.get("decision_sample_count") == count
                      and math.isclose(_number(sample.get("submitted_duration_seconds"), "sample duration"), duration, abs_tol=1e-9)
                      and math.isclose(_number(sample.get("dt_s"), "sample dt"), SAMPLE_SECONDS, abs_tol=1e-12)
                      and math.isclose(_number(sample.get("t_s"), "sample t"), len(output) * SAMPLE_SECONDS, abs_tol=1e-7)
                      and sample.get("terminal") is False, "sample identity, timing, or terminal differs")
                before = last_observation
                joints = _vector(before.get("joints_deg"), "observed joints")
                _need(_close(_vector(sample.get("observed_joint_deg"), "recorded observed joints"), joints)
                      and sample.get("observed_gripper_command") == before.get("gripper_command"),
                      "sample observation differs")
                features, moving_valid = _student_features(before, previous_joints, initial_rest_verified)
                _need(_close(features, sample.get("features", [])), "sample student features differ")
                after = sample.get("next_observation")
                _identity(after)
                _need(math.isclose(_number(after.get("time_seconds"), "next time")
                                   - _number(before.get("time_seconds"), "time"), SAMPLE_SECONDS, abs_tol=1e-6)
                      and after.get("observation_id") != before.get("observation_id")
                      and after.get("registered_scene") == before.get("registered_scene")
                      and after.get("task") == before.get("task")
                      and after.get("registered_slot_xy_mm") == before.get("registered_slot_xy_mm")
                      and after.get("physical_confirmation") == "not_inspected", "sample successor differs")
                _need(_close(_vector(sample.get("target_joint_deg"), "next joint observation"),
                             _vector(after.get("joints_deg"), "successor joints"))
                      and _close(_vector(sample.get("target_velocity_deg_s"), "target velocity"), velocity),
                      "target must be next observed joint and directly submitted velocity")
                effective_grip = before["gripper_command"] if command["gripper"] == "hold" else command["gripper"]
                _need(after.get("gripper_command") == sample.get("target_gripper") == effective_grip,
                      "gripper target differs")
                target = _vector(sample.get("applied_position_servo_target_deg"), "position servo target")
                previous_target = _vector(before.get("applied_target_deg"), "previous servo target")
                _need(_close(target, [q + SAMPLE_SECONDS * v for q, v in zip(previous_target, velocity, strict=True)])
                      and _close(target, _vector(after.get("applied_target_deg"), "returned servo target")),
                      "position target is not the integrated submitted speed")
                output.append({"sample_id": sid, "decision_id": step["decision_id"],
                    "t_s": _number(sample["t_s"], "sample time"), "dt_s": SAMPLE_SECONDS,
                    "features": features, "input_moving_valid": moving_valid,
                    "current_applied_position_servo_target_deg": previous_target,
                    "target_joint_deg": _vector(sample["target_joint_deg"], "next joints"),
                    "target_velocity_deg_s": velocity, "target_gripper": effective_grip,
                    "submitted_gripper": command["gripper"], "submitted_duration_seconds": duration,
                    "applied_position_servo_target_deg": target, "terminal": False})
                previous_joints, last_observation = joints, after
            _need(_identity(step.get("after")) == _identity(last_observation), "decision return differs")
        _need(sample_index == len(samples) and _identity(result.get("observation")) == _identity(last_observation),
              "chunk result differs")
        accepted, progress_samples = None, []
    _need(output and accepted is None and not progress_samples and isinstance(terminal, dict)
          and terminal == finished.get("terminal") and terminal.get("source") == "explicit_finish"
          and terminal.get("observation_id") == last_observation["observation_id"]
          and math.isclose(_number(terminal.get("time_seconds"), "finish time"),
                           _number(last_observation.get("time_seconds"), "last observation time"), abs_tol=1e-6)
          and terminal.get("last_sample_id") == output[-1]["sample_id"]
          and terminal.get("last_sample_terminal") is True and terminal.get("samples") == len(output)
          and terminal.get("decisions") == len(seen_decisions),
          "explicit finish does not bind the final sample")
    output[-1]["terminal"] = True
    sample_count, checks = assessment.get("sample_count"), assessment.get("checks")
    complete = (sample_count == 501 and type(assessment.get("stable_interval_seconds")) in (int, float)
        and math.isclose(assessment["stable_interval_seconds"], 1., abs_tol=1e-6)
        and isinstance(checks, dict) and bool(checks) and all(type(v) is bool for v in checks.values())
        and assessment["success"] == all(checks.values()))
    status = ("success" if assessment["success"] else "failure") if complete else "unconfirmed"
    return {"schema": SCHEMA, "feature_names": list(scene_features.FEATURE_NAMES),
        "metadata": {"source_kind": "explicitly_submitted_velocity_commands_actor_unverified",
            "source_reference": source_reference, "source_group_id": source_group_id,
            "source_group_independence": "provided_by_caller_not_verified_here", "simulated": True,
            "actual_hardware": False, "sample_dt_s": SAMPLE_SECONDS, "resampled": False,
            "position_target": "next simulated observed joint position in degrees; not an independently submitted position command",
            "current_applied_position_servo_target_deg": "position servo target at interval start; from ready for first sample, previous sample thereafter; not a separately submitted angle command",
            "velocity_target": "explicit submitted signed degrees/second held across its stated duration; repeated 50 Hz samples are not independent submissions and are not replay qvel",
            "applied_position_servo_target_deg": "position servo target integrated from submitted speed; not an independently submitted position command",
            "gripper_target": "applied open/closed target during the sample; submitted hold/open/closed also retained",
            "terminal_target": "explicit finish occurs at t+dt after the last sample interval and its submitted velocity/gripper have been applied; the last sample is not skipped",
            "moving_input": "finite difference of past/current observed joints; first sample valid only with recorded simulated rest evidence",
            "initial_rest_evidence": ("recorded_simulated_qvel_check_not_a_student_input" if initial_rest_verified
                                      else "unavailable_first_moving_must_be_masked"),
            "history": "one journal per episode; construct past-only histories within this episode with a validity mask",
            "simulator_qvel_used_as_input": False, "object_positions_or_contacts_as_input": False,
            "posthoc_status": status, "posthoc_sample_count": sample_count, "identity_augmentation": False,
            "submitted_decisions": len(seen_decisions), "observed_samples": len(output),
            "training_eligibility": "not_established_by_parser"},
        "episodes": [{"episode_id": source_reference, "source_group_id": source_group_id, "samples": output}]}


def read_event_journal(path, *, source_group_id):
    """Read a finished UTF-8 JSON-lines journal without editing it."""
    source = Path(path).resolve()
    raw = source.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    _need(lines and all(line.strip() for line in lines), "empty or blank journal line")
    events = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
              for line in lines]
    data = from_events(events, source_group_id=source_group_id, source_reference=str(source))
    data["metadata"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def validate_partition_integrity(entries):
    """Reject duplicate journals, cross-split source groups and failed labels.

    An optional initial_state_sha256 detects equal states under different group
    names. All group and state identities still require independent evidence.
    """
    _need(isinstance(entries, list) and entries, "partition entries are required")
    groups, states, journals, sample_ids, counts = {}, {}, set(), set(), {"success": 0, "failure": 0, "unconfirmed": 0}
    for entry in entries:
        _need(isinstance(entry, dict) and entry.get("split") in ("train", "validation", "excluded"),
              "invalid partition entry")
        data, split = entry.get("data"), entry["split"]
        _need(isinstance(data, dict) and data.get("schema") == SCHEMA, "only direct-command data is supported")
        meta = data.get("metadata", {})
        _need(meta.get("source_kind") == "explicitly_submitted_velocity_commands_actor_unverified"
              and meta.get("simulator_qvel_used_as_input") is False and meta.get("sample_dt_s") == SAMPLE_SECONDS,
              "direct source contract differs")
        group, journal = meta.get("source_group_id"), meta.get("source_sha256")
        _need(isinstance(group, str) and group and isinstance(journal, str) and len(journal) == 64
              and all(c in "0123456789abcdef" for c in journal), "source group and journal digest are required")
        _need(journal not in journals, "the same source journal is listed more than once")
        journals.add(journal)
        episodes = data.get("episodes")
        _need(isinstance(episodes, list) and len(episodes) == 1 and episodes[0].get("source_group_id") == group
              and isinstance(episodes[0].get("samples"), list) and episodes[0]["samples"], "invalid source episode")
        for sample in episodes[0]["samples"]:
            sid = sample.get("sample_id")
            _need(isinstance(sid, str) and sid and sid not in sample_ids, "sample ID occurs in multiple episodes")
            sample_ids.add(sid)
        status = meta.get("posthoc_status")
        _need(status in counts, "unknown posthoc status")
        counts[status] += 1
        _need(status == "success" or split == "excluded", "failed or unconfirmed episode cannot enter a success fit")
        if split == "excluded":
            continue
        _need(group not in groups or groups[group] == split, "one source group crosses partitions")
        groups[group] = split
        state = entry.get("initial_state_sha256")
        if state is not None:
            _need(isinstance(state, str) and len(state) == 64 and all(c in "0123456789abcdef" for c in state),
                  "invalid initial-state digest")
            _need(state not in states or states[state] == split, "one initial state crosses partitions")
            states[state] = split
    return {"journals": len(journals), "status_counts": counts, "admitted_source_groups": len(groups),
            "groups_and_state_identities": "caller_supplied_not_independently_verified"}
