"""Read audited 1103 student-prefix/direct-takeover episodes without executing them.

The caller supplies independently selected source fingerprints and a transcript
audit with references to the actual official JSONL records. Each complete public
observation, literal input, observation ID and returned decision is checked
against those records as well as the collector's ordered event log. Supported
inputs are literal ``write_stdin`` calls understood by the existing public
takeover parser; unsupported or incomplete transcripts fail closed.

The caller supplies physical-evidence references bound to the saved assessment
and checkpoint fingerprints. This module checks the
saved whole-scene assessment and the collector's hold-match record; it does not
read the private checkpoint, rerun the simulator, authenticate the reviewer or
prove that the runtime model had a particular identity. The trust in that review
and the limits of the collector's self-reported hold are explicit in evidence.
Student prefixes remain unlabelled, and unverified cases produce no episode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from raccoon_control.kinematics import Geometry, check_joints
from . import motion_policy, scene_applied_action_student as student, scene_features, scene_policy, task_catalog
from .scene_applied_action_dataset import TrainingEpisode
from .scene_public_takeover_dataset import _literal_stdin_call


_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC = frozenset(("schema_version", "joints_deg", "sensor_valid", "moving", "gripper_command",
                     "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh"))


@dataclass(frozen=True)
class SourceFingerprints:
    """Caller-selected hashes; a hash copied from an untrusted run is not an audit."""

    plan: str
    case: str
    events: str
    trace: str
    execution: str
    result: str
    assessment: str


@dataclass(frozen=True, order=True)
class TranscriptRef:
    """Zero-based JSONL record and event index within that record's output."""

    record: int
    event: int = 0


@dataclass(frozen=True)
class SubmissionRefs:
    input: TranscriptRef
    response: TranscriptRef


@dataclass(frozen=True)
class DecisionRefs:
    observation: TranscriptRef
    submissions: tuple[SubmissionRefs, ...]


@dataclass(frozen=True)
class TranscriptAudit:
    case_id: str
    path: str | Path
    sha256: str
    session_token: str
    reviewer: str
    review_reference: str
    # Omit to discover references by unique observation ID and literal input.
    decisions: tuple[DecisionRefs, ...] | None = None
    # For an append-only transcript, pin an existing complete JSONL prefix.
    # The digest then covers exactly these bytes, not the later appended tail.
    byte_length: int | None = None


@dataclass(frozen=True)
class PhysicalAudit:
    case_id: str
    assessment_sha256: str
    checkpoint_sha256: str
    reviewer: str
    assessment_reference: str
    hold_reference: str


@dataclass(frozen=True)
class LoadedTakeover:
    episode: TrainingEpisode | None
    exclusion: str | None
    case_id: str


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _read(path, sha):
    _need(isinstance(sha, str) and _HEX.fullmatch(sha) is not None, "explicit SHA-256 required")
    raw = Path(path).read_bytes()
    _need(hashlib.sha256(raw).hexdigest() == sha, f"fingerprint mismatch: {Path(path).name}")
    return raw


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate JSON key")
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc


def _jsonl(raw):
    _need(raw.endswith(b"\n"), "incomplete JSONL tail")
    records = [_json(line) for line in raw.splitlines()]
    _need(all(isinstance(row, dict) for row in records), "invalid JSONL record")
    return records


def _near(a, b, tolerance=1e-7):
    try:
        a, b = student._applied_target(a), student._applied_target(b)
    except (TypeError, ValueError):
        return False
    return all(math.isclose(x, y, abs_tol=tolerance, rel_tol=0) for x, y in zip(a, b))


def _tensor_equal(a, b):
    try:
        left, right = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
        return left.shape == right.shape and bool(torch.isfinite(left).all()) and torch.equal(left, right)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return False


def _public_observation(obs, expected_gripper):
    _need(isinstance(obs, dict) and set(obs) == _PUBLIC and obs.get("sensor_valid") is True
          and obs.get("gripper_reported_valid") is True and obs.get("gripper_feedback_fresh") is True
          and obs.get("gripper_command") == expected_gripper and type(obs.get("moving")) is bool,
          "invalid or inconsistent public observation")


def _public_event(value):
    """Expand supported lossless public-window displays before exact comparison."""
    if value.get("event") != "observation":
        return value
    if "window_encoding" in value:
        _need(value["window_encoding"] == "lossless: leading4, each digit of binary70, trailing8"
              and "window_lossless" not in value and isinstance(value.get("window"), list)
              and len(value["window"]) == scene_features.HISTORY_LENGTH,
              "invalid lossless public window")
        window = []
        for row in value["window"]:
            _need(isinstance(row, dict) and set(row) == {"leading4", "binary70", "trailing8"}
                  and isinstance(row["leading4"], list) and len(row["leading4"]) == 4
                  and isinstance(row["binary70"], str) and len(row["binary70"]) == 70
                  and set(row["binary70"]) <= {"0", "1"}
                  and isinstance(row["trailing8"], list) and len(row["trailing8"]) == 8
                  and all(type(v) in (int, float) and math.isfinite(v)
                          for v in row["leading4"] + row["trailing8"]),
                  "invalid lossless public window")
            window.append(row["leading4"] + [int(bit) for bit in row["binary70"]] + row["trailing8"])
        _need(all(len(row) == len(scene_features.FEATURE_NAMES) for row in window),
              "invalid lossless public window")
        return {**{key: item for key, item in value.items() if key not in ("window", "window_encoding")},
                "window": window}
    if "window_lossless" not in value:
        return value
    encoded = value["window_lossless"]
    _need("window" not in value and isinstance(encoded, dict)
          and set(encoded) == {"base", "changes_from_base"}, "invalid lossless public window")
    base, changes = encoded["base"], encoded["changes_from_base"]
    _need(isinstance(base, list) and len(base) == len(scene_features.FEATURE_NAMES)
          and all(type(v) in (int, float) and math.isfinite(v) for v in base)
          and isinstance(changes, list) and len(changes) == scene_features.HISTORY_LENGTH,
          "invalid lossless public window")
    window = []
    for difference in changes:
        _need(isinstance(difference, list), "invalid lossless public window")
        row, previous_index = base.copy(), -1
        for entry in difference:
            _need(isinstance(entry, list) and len(entry) == 2 and type(entry[0]) is int
                  and previous_index < entry[0] < len(base) and type(entry[1]) in (int, float)
                  and math.isfinite(entry[1]), "invalid lossless public window")
            previous_index = entry[0]
            row[entry[0]] = entry[1]
        window.append(row)
    return {**{key: item for key, item in value.items() if key != "window_lossless"}, "window": window}


def _events_in_output(record):
    payload = record.get("payload", {})
    _need(payload.get("type") == "custom_tool_call_output", "transcript reference is not a tool output")
    found = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("event"), str):
                found.append(_public_event(value))
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            # The Windows PTY can wrap a physical line with a cursor-position
            # escape and redraw its last character. Undo that display-only
            # artifact before parsing; exact comparison with collector events
            # below still rejects any ambiguous or incorrect reconstruction.
            value = re.sub(r"(.)\r\n\x1b\[\d+;\d+H\1", lambda match: match.group(1), value)
            value = re.sub(r"\r\n\x1b\[\d+;\d+H", "", value)
            value = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", value)
            value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
            # One retained display stripped cursor escapes before printing a
            # wrapped DONE return.  Repair only its exact, fixed response
            # grammar: the quote at the wrap was redrawn once.  The resulting
            # event must still match the collector's observation ID and DONE
            # result below; no window or command value is inferred here.
            value = re.sub(
                r'(\{"event":"decision_return","observation_id":"[0-9a-f]{32}",")\r?\n"(?=result":"done_submitted"\})',
                r'\1', value)
            # Tool outputs can wrap JSON events in a longer console-text item.
            try:
                parsed = _json(value)
            except ValueError:
                for line in value.splitlines():
                    try:
                        parsed_line = _json(line)
                    except ValueError:
                        continue
                    if isinstance(parsed_line, (dict, list)):
                        visit(parsed_line)
            else:
                if isinstance(parsed, (dict, list)):
                    visit(parsed)

    visit(payload.get("output"))
    return found


def _reference(records, ref, *, output):
    _need(isinstance(ref, TranscriptRef) and type(ref.record) is int and type(ref.event) is int
          and 0 <= ref.record < len(records) and ref.event >= 0, "invalid transcript reference")
    if output:
        events = _events_in_output(records[ref.record])
        _need(ref.event < len(events), "transcript output event is missing or truncated")
        return events[ref.event]
    _need(ref.event == 0, "invalid transcript input reference")
    payload = records[ref.record].get("payload", {})
    _need(payload.get("type") == "custom_tool_call" and payload.get("name") == "exec"
          and isinstance(payload.get("input"), str), "transcript reference is not an exec input")
    call = _literal_stdin_call(payload["input"])
    _need(call is not None, "official input is not a supported literal write_stdin call")
    return call


def _audit_transcript(audit, case_id, observation_ids):
    _need(isinstance(audit, TranscriptAudit) and audit.case_id == case_id
          and _nonempty(audit.session_token) and _nonempty(audit.reviewer)
          and _nonempty(audit.review_reference)
          and (audit.decisions is None or isinstance(audit.decisions, tuple)),
          "independent transcript audit is required")
    if audit.byte_length is None:
        raw = _read(audit.path, audit.sha256)
    else:
        _need(type(audit.byte_length) is int and audit.byte_length > 0
              and isinstance(audit.sha256, str) and _HEX.fullmatch(audit.sha256),
              "invalid transcript prefix fingerprint")
        with Path(audit.path).open("rb") as stream:
            raw = stream.read(audit.byte_length)
        _need(len(raw) == audit.byte_length and hashlib.sha256(raw).hexdigest() == audit.sha256,
              "transcript prefix fingerprint mismatch")
    _need(raw.endswith(b"\n"), "incomplete JSONL tail")
    _need(observation_ids and all(isinstance(v, str) and v for v in observation_ids),
          "teacher observation identities are missing")
    ids = tuple(value.encode("utf-8") for value in observation_ids)
    token_forms = (audit.session_token.encode("utf-8"),
                   json.dumps(audit.session_token, ensure_ascii=False)[1:-1].encode("utf-8"))
    records = []
    for line in raw.splitlines():
        # Hash the complete pinned prefix but decode only records mentioning
        # this case's IDs or its write_stdin session, not unrelated content.
        selected = any(value in line for value in ids) or (
            b"tools.write_stdin" in line and any(value in line for value in token_forms))
        value = _json(line) if selected else {}
        _need(isinstance(value, dict), "invalid selected transcript record")
        records.append(value)
    return records


def _discover_decisions(records, events, session_token):
    """Locate matching transcript records; this does not authenticate their reviewer.

    A recovered copy of the same complete observation is acceptable. Every
    returned reference is still checked by the same strict loader used for
    explicitly supplied references. Unsupported input forms are not guessed.
    """
    outputs, inputs = [], []
    for index, record in enumerate(records):
        payload = record.get("payload", {})
        if payload.get("type") == "custom_tool_call_output":
            outputs.extend((TranscriptRef(index, event_index), value) for event_index, value in
                           enumerate(_events_in_output(record)))
        elif payload.get("type") == "custom_tool_call" and payload.get("name") == "exec":
            try:
                call = _literal_stdin_call(payload.get("input", ""))
            except (TypeError, ValueError):
                continue
            if call is not None and call[0] == session_token:
                inputs.append((TranscriptRef(index), call[1]))
    observed = [(i, e) for i, e in enumerate(events) if e.get("event") == "teacher_observation"]
    discovered, cursor = [], None
    for offset, (position, event) in enumerate(observed):
        end = observed[offset+1][0] if offset+1 < len(observed) else next(
            (i for i in range(position+1, len(events)) if events[i].get("event") == "v2_execution_ended"), len(events))
        attempts = [e for e in events[position+1:end] if e.get("event") == "teacher_submission"]
        visible = {"event": "observation", **{k: v for k, v in event.items() if k != "event"}}
        matches = [ref for ref, item in outputs if (cursor is None or cursor < ref) and item == visible]
        _need(attempts and matches, "complete official observation cannot be located")
        observation_ref, previous = matches[0], matches[0]
        submissions = []
        for index, attempt in enumerate(attempts):
            candidates = [ref for ref, chars in inputs if previous < ref and isinstance(chars, str)
                          and chars.endswith("\n") and len(chars.splitlines()) == 1
                          and chars.rstrip("\r\n") == attempt.get("raw")]
            _need(candidates, "official literal input cannot be located")
            input_ref = candidates[0]
            expected_event = "submission_rejected" if index < len(attempts)-1 else "decision_return"
            returns = [ref for ref, item in outputs if input_ref < ref and item.get("event") == expected_event
                       and item.get("observation_id") == event.get("observation_id")]
            _need(returns, "official decision return cannot be located")
            response_ref = returns[0]
            submissions.append(SubmissionRefs(input_ref, response_ref))
            previous = response_ref
        discovered.append(DecisionRefs(observation_ref, tuple(submissions)))
        cursor = previous
    return tuple(discovered)


def _required_checks(case):
    task, before = case["move"]["task"], case["move"]["before"]
    required = {"gripper_open", "no_unexpected_cylinder_contact"}
    for obj in (obj for stack in before for obj in stack):
        required.update(f"object_{obj}.{suffix}" for suffix in
                        ("position_xy", "height", "upright", "linear_speed", "angular_speed",
                         "required_support_contact", "free_of_robot_contact"))
        if obj != task["object_id"]:
            required.update((f"object_{obj}.registered_slot_xy", f"object_{obj}.registered_level_height"))
    if task["destination_level"] > 1:
        required.update((f"object_{task['object_id']}.aligned_with_support_xy",
                         f"object_{task['object_id']}.aligned_with_support_height"))
    return required


def _physical(case, result, events, assessment, audit, fingerprints):
    score = result.get("posthoc")
    _need(isinstance(audit, PhysicalAudit) and audit.case_id == case["case_id"]
          and audit.assessment_sha256 == fingerprints.assessment
          and _nonempty(audit.reviewer) and _nonempty(audit.assessment_reference)
          and _nonempty(audit.hold_reference), "independent physical evidence is required")
    _need(isinstance(score, dict) and score.get("assessment_sha256") == fingerprints.assessment
          and score.get("checkpoint_sha256") == audit.checkpoint_sha256
          and isinstance(audit.checkpoint_sha256, str) and _HEX.fullmatch(audit.checkpoint_sha256)
          and score.get("status") == "success" and score.get("sample_count") == 501
          and score.get("whole_scene_scope_valid") is True and score.get("physical_hold_matches") is True
          and score.get("failed_checks") == [] and type(score.get("stable_seconds")) in (int, float)
          and math.isclose(score["stable_seconds"], 1., abs_tol=1e-6, rel_tol=0),
          "physical success or hold is unconfirmed")
    checks = assessment.get("checks")
    task, move = case["move"]["task"], case["move"]
    _need(assessment.get("source") == "mujoco_simulation" and assessment.get("simulated") is True
          and assessment.get("actual_hardware") is False and assessment.get("success") is True
          and assessment.get("sample_count") == 501
          and type(assessment.get("stable_interval_seconds")) in (int, float)
          and math.isclose(assessment["stable_interval_seconds"], 1., abs_tol=1e-6, rel_tol=0)
          and assessment.get("before_layout") == move["before"]
          and assessment.get("after_layout") == move["expected_after"]
          and assessment.get("source_slot") == task["source_slot"]
          and assessment.get("destination_slot") == task["destination_slot"]
          and assessment.get("moved_object_id") == task["object_id"]
          and isinstance(checks, dict) and _required_checks(case) <= checks.keys()
          and all(v is True for v in checks.values()), "saved whole-scene assessment is invalid")
    names = [e.get("event") for e in events]
    _need(names.count("v2_execution_ended") == names.count("posthoc_checkpoint_saved")
          == names.count("posthoc_assessment") == 1, "posthoc evidence is incomplete")
    execution, checkpoint, post = (names.index(name) for name in
                                   ("v2_execution_ended", "posthoc_checkpoint_saved", "posthoc_assessment"))
    _need(execution < checkpoint < post < len(events)-1 and events[execution].get("status") == "completed"
          and events[execution].get("teacher_done") is True
          and events[checkpoint].get("saved_after_live") is True
          and events[checkpoint].get("includes") == ["before", "done", "after_hold"]
          and events[checkpoint].get("checkpoint_sha256") == audit.checkpoint_sha256
          and {k: v for k, v in events[post].items() if k != "event"} == score,
          "posthoc evidence order or binding differs")


def _resolve(request, previous):
    _need(isinstance(request, dict) and request.get("action") in ("MOVE", "OPEN", "CLOSE", "DONE"),
          "invalid teacher action")
    action = request["action"]
    if action == "MOVE" and set(request) == {"action", "target_joints_deg"}:
        target = list(check_joints(request["target_joints_deg"]))
    elif action == "MOVE" and set(request) == {"action", "target_xyz_mm"}:
        target = list(Geometry().inverse(request["target_xyz_mm"]))
    elif action != "MOVE" and set(request) == {"action"}:
        target = list(previous)
    else:
        raise ValueError("teacher action target is malformed")
    return action, student._applied_target(target)


def load_takeover(directory, *, plan_path, fingerprints, transcript_audit=None, physical_audit=None):
    """Load one audited success, or return an explicit exclusion with no labels.

    Hashes must be selected outside the records being loaded. References in the
    externally supplied audit identify complete official transcript events and
    literal inputs. The caller is responsible for trusting the named reviewers;
    the audit is not created by this function. No source files are modified.
    """
    _need(isinstance(fingerprints, SourceFingerprints), "source fingerprints are required")
    directory = Path(directory)
    case = _json(_read(directory / "fixed_case.json", fingerprints.case))
    result = _json(_read(directory / "result.json", fingerprints.result))
    _need(isinstance(case, dict) and _nonempty(case.get("case_id")) and isinstance(result, dict)
          and result.get("case_id") == case["case_id"], "case or result identity differs")
    case_id = case["case_id"]
    score = result.get("posthoc")
    if not (isinstance(score, dict) and score.get("status") == "success"):
        return LoadedTakeover(None, "physical_success_failed_or_unconfirmed", case_id)
    if transcript_audit is None or physical_audit is None:
        return LoadedTakeover(None, "independent_evidence_missing", case_id)
    plan = _json(_read(plan_path, fingerprints.plan))
    events = _jsonl(_read(directory / "events.jsonl", fingerprints.events))
    trace = _json(_read(directory / "trace.json", fingerprints.trace))
    execution = _json(_read(directory / "v2_execution.json", fingerprints.execution))
    assessment = _json(_read(directory / "posthoc_assessment.json", fingerprints.assessment))
    _need(isinstance(plan, dict) and plan.get("schema") == "scene-1103-public-takeover48.v1"
          and isinstance(plan.get("cases"), list) and type(case.get("order")) is int
          and 1 <= case["order"] <= len(plan["cases"]) and plan["cases"][case["order"]-1] == case,
          "case does not match the selected plan")
    _need(events and events[0].get("event") == "start" and events[-1].get("event") == "end"
          and events[0].get("case_id") == case_id and events[0].get("plan_sha256") == fingerprints.plan
          and events[0].get("actual_hardware") is False and events[0].get("closed_final_used") is False
          and events[-1].get("summary") == result and result.get("allocation") == case.get("allocation")
          and result.get("teacher_done") is True and result.get("v2_status") == "completed"
          and result.get("error") is None and result.get("student_termination") is None
          and result.get("physical_assessment_after_live_ended") is True
          and result.get("student_prefix_unlabelled") is True and result.get("actual_hardware") is False
          and result.get("final_test_used") is False, "collection is incomplete or out of scope")
    move, slots = case["move"], case["registered_slot_xy_mm"]
    before, task = scene_features.transfer_task(move["before"], move["task"])
    group = scene_policy.split_group_for(before, task)
    before = [list(stack) for stack in before]
    expected_after = [list(stack) for stack in task_catalog.transfer(before, task["source_slot"], task["destination_slot"])]
    _need(before == case["initial_scene"] and move.get("index") == 1 and task == move["task"]
          and move.get("expected_after") == expected_after
          and move.get("split_group") == group
          and ((group == "n3_s2_d2_other00" and case.get("object_count") == 3)
               or (group == "n4_s2_d2_other01" and case.get("object_count") == 4)),
          "registered scene, task or collection group differs")
    _need(isinstance(execution, dict) and execution.get("status") == "completed"
          and execution.get("request_id") == f"public-takeover48-{case['order']:02d}"
          and execution.get("physical_success") is None and isinstance(execution.get("events"), list),
          "execution record is incomplete")
    registrations = [e for e in execution["events"] if e.get("event") == "registered"]
    transfers = [e for e in execution["events"] if e.get("event") == "transfer"]
    _need(len(registrations) == len(transfers) == 1, "registration or transfer boundary differs")
    registration, transfer = registrations[0], transfers[0]
    _need(execution["events"].index(registration) < execution["events"].index(transfer)
          and registration.get("scene") == before and registration.get("nominal_slot_xy_mm") == slots
          and registration.get("memory_cleared") is True and registration.get("scene_source") == "explicit_registration"
          and registration.get("control_target_source") == "validated_explicit_registration"
          and registration.get("physical_assessment_performed") is False
          and transfer.get("before") == before and transfer.get("task") == task
          and transfer.get("planned_after") == move["expected_after"] and transfer.get("split_group") == group
          and transfer.get("committed") is True and transfer.get("commit_kind") == "expected_only"
          and transfer.get("physical_success") is None, "registration or expected transfer differs")
    _physical(case, result, events, assessment, physical_audit, fingerprints)
    observed_ids = [e.get("observation_id") for e in events if e.get("event") == "teacher_observation"]
    records = _audit_transcript(transcript_audit, case_id, observed_ids)
    if transcript_audit.decisions is None:
        transcript_audit = replace(transcript_audit, decisions=_discover_decisions(records, events,
                                                                                  transcript_audit.session_token))
    return _episode(case, result, events, trace, transfer, records, transcript_audit, physical_audit,
                    fingerprints, directory, plan_path)


def _episode(case, result, events, trace, transfer, records, audit, physical_audit, fingerprints,
             directory, plan_path):
    _need(isinstance(trace, dict), "trace is invalid")
    rows, decisions, teachers = trace.get("student_rows"), trace.get("student_decisions"), trace.get("teacher_rows")
    _need(isinstance(rows, list) and rows and isinstance(decisions, list) and len(rows) == len(decisions)
          and isinstance(teachers, list) and teachers and len(teachers) == len(audit.decisions)
          and trace.get("teacher_done") is True and trace.get("student_termination") is None
          and trace.get("handoff") == result.get("handoff"), "trace boundaries differ")
    transfer_result = transfer.get("result")
    _need(isinstance(transfer_result, dict) and transfer_result.get("termination") == "model_done"
          and transfer_result.get("decisions") == decisions and transfer_result.get("physical_success") is None
          and transfer_result.get("actual_hardware") is False
          and transfer_result.get("teacher_assistance_count") == len(teachers), "transfer result differs")
    inputs = [(i, e) for i, e in enumerate(events) if e.get("event") in ("student_input", "teacher_observation")]
    _need(len(inputs) == len(rows) + len(teachers)
          and [e.get("event") for _, e in inputs] == ["student_input"]*len(rows) + ["teacher_observation"]*len(teachers),
          "input event sequence differs")
    handoffs = [(i, e) for i, e in enumerate(events) if e.get("event") == "handoff"]
    _need(len(handoffs) == 1 and inputs[len(rows)-1][0] < handoffs[0][0] < inputs[len(rows)][0]
          and handoffs[0][1].get("handoff") == trace["handoff"]
          and handoffs[0][1].get("student_decisions") == decisions
          and trace["handoff"].get("confirmed_student_commands") == len(decisions), "handoff differs")
    before, task, slots = case["move"]["before"], case["move"]["task"], case["registered_slot_xy_mm"]
    history, features, actions, targets, labels, moves = [], [], [], [], [], []
    previous = student._applied_target(decisions[0].get("previous_applied_target_deg"))
    previous_action, gripper, opening = None, "open", .04
    seen_ids, last_ref = set(), None

    for i, (position, event) in enumerate(inputs):
        next_position = inputs[i+1][0] if i+1 < len(inputs) else next(
            j for j in range(position+1, len(events)) if events[j].get("event") == "v2_execution_ended")
        group_events = events[position+1:next_position]
        is_teacher = i >= len(rows)
        row = teachers[i-len(rows)] if is_teacher else rows[i]
        obs = row.get("visible", {}).get("observation") if is_teacher else row.get("observation")
        _public_observation(obs, gripper)
        encoded = scene_features.encode_scene_features(obs, before, task, slots)
        if is_teacher:
            visible = row.get("visible")
            _need(isinstance(visible, dict) and visible == {k: v for k, v in event.items() if k != "event"}
                  and visible.get("expected_scene") == before and visible.get("task") == task
                  and visible.get("registered_slot_xy_mm") == slots
                  and visible.get("physical_confirmation") == "not_inspected"
                  and _near(visible.get("previous_applied_target_deg"), previous)
                  and visible.get("previous_action") == (previous_action.upper() if previous_action else None),
                  "teacher observation or previous applied control differs")
            history.append(encoded)
        else:
            _need(row == {k: v for k, v in event.items() if k != "event"} and row.get("labelled") is False
                  and _near(row.get("student_input_previous_target_tensor_deg"), previous, 2e-5),
                  "student input or previous applied control differs")
            history.append(row["window"][-1])
            _need(_tensor_equal(history[-1], encoded), "student input does not encode its observation")
        window, mask = scene_features.history_window(history)
        saved = row["visible"] if is_teacher else row
        _need(_tensor_equal(saved.get("window"), window) and saved.get("mask") == mask,
              "recorded observation history is noncausal or differs")
        features.append(student.encode_step(obs, before, task, slot_xy_mm=slots,
            previous_applied_action=previous_action, previous_applied_target_deg=previous,
            transfer_start=i == 0, registration_start=i == 0))

        if not is_teacher:
            decision = decisions[i]
            _need(isinstance(decision, dict) and decision.get("index") == i
                  and decision.get("action") in ("move", "open", "close")
                  and decision.get("result") == "command_applied"
                  and _near(decision.get("previous_applied_target_deg"), previous)
                  and _tensor_equal(decision.get("policy_features"), encoded)
                  and decision.get("valid_history") == mask, "student decision is unconfirmed")
            command, guard = decision.get("command"), decision.get("guard")
            _need(isinstance(command, dict) and isinstance(guard, dict)
                  and command.get("source") == "shared_student_policy" and command.get("seconds") == 1.
                  and guard.get("model_action") == decision["action"]
                  and _near(command.get("joints_deg"), guard.get("actual_applied_joints_deg")),
                  "student applied target is unconfirmed")
            action, applied = decision["action"], student._applied_target(guard["actual_applied_joints_deg"])
            expected_opening = .04 if action == "open" else 0. if action == "close" else opening
            _need(command.get("gripper_opening_m") == expected_opening and (action == "move" or _near(previous, applied)),
                  "student gripper command differs")
            returns = [e for e in group_events if e.get("event") == "command_return_observation"]
            expected_events = ["command_return_observation"] + (["handoff"] if i == len(rows)-1 else [])
            _need([e.get("event") for e in group_events] == expected_events
                  and len(returns) == 1 and _near(returns[0].get("target_joints_deg"), applied)
                  and returns[0].get("opening_m") == expected_opening
                  and returns[0].get("result") == "pending_executor_confirmation", "student command return differs")
            _public_observation(returns[0].get("observation"), "open" if expected_opening >= .025 else "closed")
            labels.append(False); actions.append(-1); targets.append([float("nan")]*4); moves.append(False)
            previous, previous_action, opening = applied, action, expected_opening
            gripper = "open" if opening >= .025 else "closed"
            continue

        index = i-len(rows)
        ref = audit.decisions[index]
        observation_id = visible.get("observation_id")
        _need(isinstance(ref, DecisionRefs) and isinstance(ref.submissions, tuple) and ref.submissions
              and isinstance(observation_id, str) and observation_id and observation_id not in seen_ids,
              "teacher audit or observation identity differs")
        seen_ids.add(observation_id)
        _need(last_ref is None or last_ref < ref.observation, "official observation order differs")
        _need(_reference(records, ref.observation, output=True) == {"event": "observation", **visible},
              "official public observation differs")
        actual_submissions = [e for e in group_events if e.get("event") == "teacher_submission"]
        rejections = [e for e in group_events if e.get("event") == "submission_rejected"]
        _need(len(actual_submissions) == len(ref.submissions) == len(rejections)+1,
              "official submission count differs")
        prior_ref = ref.observation
        for j, (attempt, refs) in enumerate(zip(actual_submissions, ref.submissions, strict=True)):
            _need(isinstance(refs, SubmissionRefs) and prior_ref < refs.input < refs.response,
                  "official input or return order differs")
            token, chars = _reference(records, refs.input, output=False)
            _need(token == audit.session_token and isinstance(chars, str) and chars.endswith("\n")
                  and len(chars.splitlines()) == 1 and chars.rstrip("\r\n") == attempt.get("raw")
                  and attempt.get("observation_id") == observation_id, "official input differs")
            request = _json(attempt["raw"])
            _need(isinstance(request, dict), "teacher input is not an object")
            returned = _reference(records, refs.response, output=True)
            if j < len(rejections):
                rejected = rejections[j]
                _need(request.get("observation_id") != observation_id
                      and rejected.get("reason") == "observation_id_mismatch"
                      and rejected.get("command_dispatched") is False
                      and returned == {"event": "submission_rejected", "observation_id": observation_id,
                                       "reason": "observation_id_mismatch"}, "rejected input differs")
            else:
                _need(request.get("observation_id") == observation_id and row.get("submission") == request,
                      "accepted teacher input identity differs")
            prior_ref = refs.response
        last_ref = prior_ref
        action, resolved = _resolve({k: v for k, v in request.items() if k != "observation_id"}, previous)
        _need(row.get("action") == action and row.get("labelled") is True
              and _near(row.get("resolved_target_deg"), resolved), "teacher action resolution differs")
        names = [e.get("event") for e in group_events]
        expected_names = [name for _ in rejections for name in ("teacher_submission", "submission_rejected")]
        expected_names += ["teacher_submission", "observation_response_claimed"]
        expected_names += (["teacher_done"] if action == "DONE" else
                           ["teacher_command_started", "command_return_observation",
                            "teacher_command_returned", "teacher_command_confirmed"])
        _need(names == expected_names, "teacher submission or command order differs")
        claimed = [e for e in group_events if e.get("event") == "observation_response_claimed"]
        _need(len(claimed) == 1 and claimed[0].get("observation_id") == observation_id,
              "teacher response was not claimed")
        if action == "DONE":
            done = [e for e in group_events if e.get("event") == "teacher_done"]
            _need(index == len(teachers)-1 and row.get("result") == "done_submitted"
                  and len(done) == 1 and done[0].get("observation_id") == observation_id
                  and not any(name in names for name in ("teacher_command_started", "teacher_command_returned",
                                                          "teacher_command_confirmed", "command_return_observation"))
                  and returned == {"event": "decision_return", "observation_id": observation_id,
                                   "result": "done_submitted"}, "teacher DONE is unconfirmed")
        else:
            started = [e for e in group_events if e.get("event") == "teacher_command_started"]
            returned_events = [e for e in group_events if e.get("event") == "teacher_command_returned"]
            confirmed = [e for e in group_events if e.get("event") == "teacher_command_confirmed"]
            observations = [e for e in group_events if e.get("event") == "command_return_observation"]
            _need(len(started) == len(returned_events) == len(confirmed) == len(observations) == 1
                  and names.index("observation_response_claimed") < names.index("teacher_command_started")
                  < names.index("command_return_observation") < names.index("teacher_command_returned")
                  < names.index("teacher_command_confirmed"), "teacher command return order differs")
            sent, ret, confirm, return_obs = started[0], returned_events[0], confirmed[0], observations[0]
            expected_opening = 0. if action == "CLOSE" else .04 if action == "OPEN" else opening
            applied = student._applied_target(confirm.get("actual_applied_target_deg"))
            command_return = ret.get("returned_command")
            _need(row.get("result") == "command_applied" and _near(row.get("actual_applied_target_deg"), applied)
                  and _near(applied, resolved) and _near(sent.get("requested_target_joints_deg"), resolved)
                  and _near(return_obs.get("target_joints_deg"), resolved)
                  and sent.get("action") == confirm.get("action") == action
                  and all(e.get("observation_id") == observation_id for e in (sent, ret, confirm))
                  and sent.get("gripper_opening_m") == return_obs.get("opening_m") == expected_opening
                  and return_obs.get("result") == "pending_executor_confirmation"
                  and ret.get("confirmation") == "pending_executor_verification"
                  and isinstance(command_return, dict) and command_return == confirm.get("returned_command")
                  and command_return.get("source") == "mujoco_simulation" and command_return.get("simulated") is True
                  and type(command_return.get("time_seconds")) in (int, float)
                  and math.isfinite(command_return["time_seconds"])
                  and (action == "MOVE" or _near(previous, applied)), "teacher actual command is unconfirmed")
            _public_observation(return_obs.get("observation"), "open" if expected_opening >= .025 else "closed")
            _need(returned == {"event": "decision_return", "observation_id": observation_id,
                               "result": "command_applied", "actual_applied_target_deg": confirm["actual_applied_target_deg"],
                               "observation": return_obs["observation"]}, "official command return differs")
            previous, previous_action, opening = applied, action.lower(), expected_opening
            gripper = "open" if opening >= .025 else "closed"
        labels.append(True); actions.append(motion_policy.ACTIONS.index(action.lower()))
        moves.append(action == "MOVE")
        targets.append(applied if action == "MOVE" else [float("nan")]*4)

    group = case["move"]["split_group"]
    evidence = {"case_id": case["case_id"], "source_kind": "audited_public_direct_takeover",
        "student_prefix_length": len(rows), "teacher_decisions": len(teachers), "transfer_count": 1,
        "group": group, "group_scope": ("new_candidate_exposure" if group == "n3_s2_d2_other00"
                                         else "original_1103_training_group"),
        "source_directory": str(Path(directory).resolve()), "plan_path": str(Path(plan_path).resolve()),
        "source_fingerprints": asdict(fingerprints), "transcript_path": str(Path(audit.path).resolve()),
        "transcript_sha256": audit.sha256, "transcript_byte_length": audit.byte_length,
        "transcript_review_reference": audit.review_reference,
        "transcript_reviewer": audit.reviewer, "transcript_session_token": audit.session_token,
        "transcript_receipts": [asdict(receipt) for receipt in audit.decisions],
        "transcript_window_reconstruction": "lossless row differences, when present, compared exactly",
        "physical_review_references": (physical_audit.assessment_reference, physical_audit.hold_reference),
        "physical_reviewer": physical_audit.reviewer, "physical_checkpoint_sha256": physical_audit.checkpoint_sha256,
        "hold_verification_scope": "collector-reported match and externally supplied review; checkpoint not read",
        "runtime_model_identity_independently_verified": False, "file_access_isolation_verified": False,
        "transcript_session_token_is_runtime_identity_proof": False,
        "previous_proposals_used_as_labels": False, "physical_outcomes_used_as_model_inputs": False,
        "actual_hardware": False, "final_test_used": False}
    episode = TrainingEpisode(torch.tensor(features, dtype=torch.float32), torch.tensor(actions, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32), torch.ones(len(actions), dtype=torch.bool),
        torch.tensor(labels, dtype=torch.bool), torch.tensor(moves, dtype=torch.bool), evidence)
    return LoadedTakeover(episode, None, case["case_id"])
