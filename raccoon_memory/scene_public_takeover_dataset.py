"""Read an explicitly selected public takeover pair without running a simulator.

One registration remains one recurrent episode, including its unlabelled
student prefix. Only returned direct commands and completed direct DONEs
receive labels. An arm with a failed/unconfirmed physical transfer is retained
in the collection report and excluded as a whole from success supervision.
The normal and takeover arms are paired, not independent starting samples.
Caller-supplied fingerprints bind the selected records; they do not attest
the operator's runtime model or independent decision making.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch

from . import motion_policy, scene_applied_action_student as student, scene_features
from .scene_applied_action_dataset import TrainingEpisode


@dataclass(frozen=True)
class PairFingerprints:
    result_sha256: str
    events_sha256: str
    collector_sha256: str
    transcript_sha256: str


@dataclass(frozen=True)
class LoadedPair:
    episodes: tuple[TrainingEpisode, ...]
    excluded_arms: dict[str, str]
    collection: dict


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, digest):
    _need(isinstance(digest, str) and len(digest) == 64
          and all(c in "0123456789abcdef" for c in digest), "invalid supplied fingerprint")
    raw = Path(path).read_bytes()
    _need(_sha(raw) == digest, f"selected record fingerprint differs: {Path(path).name}")
    return raw


def _near(a, b, tolerance=1e-7):
    return len(a) == len(b) == 4 and all(
        math.isclose(float(x), float(y), abs_tol=tolerance, rel_tol=0) for x, y in zip(a, b))


def _one(events, name):
    values = [event for event in events if event.get("event") == name]
    _need(len(values) == 1, f"expected one {name} record")
    return values[0]


def _tensor(value, expected, message):
    _need(isinstance(value, torch.Tensor) and value.device.type == "cpu"
          and value.dtype == expected.dtype and value.shape == expected.shape
          and torch.equal(value, expected), message)


def _quoted_text(code, start):
    """Decode a literal JS quote without executing the surrounding code."""
    _need(start < len(code) and code[start] in ("'", '"'), "nonliteral official input")
    quote, index, output = code[start], start + 1, []
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "/": "/",
               "\\": "\\", "'": "'", '"': '"'}
    while index < len(code):
        char = code[index]
        index += 1
        if char == quote:
            return "".join(output), index
        if char == "\\":
            _need(index < len(code), "incomplete official literal")
            char = code[index]
            index += 1
            if char == "u":
                digits = code[index:index + 4]
                _need(len(digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in digits),
                      "unsupported official Unicode escape")
                output.append(chr(int(digits, 16)))
                index += 4
            else:
                _need(char in escapes, "unsupported official input escape")
                output.append(escapes[char])
        else:
            _need(char not in "\r\n", "unsupported multiline official literal")
            output.append(char)
    raise ValueError("unterminated official literal")


def _literal_stdin_call(code):
    """Recognize the four retained operator call forms, without evaluating JS.

    A stored session key is only a grouping token. The loader claims literal
    submission-text agreement, not independent runtime variable attestation.
    """
    declaration = r"(?:const|let)\s+\w+\s*=\s*await\s+tools\.write_stdin"
    session = r'(\d+|load\(\s*"[A-Za-z0-9_]+"\s*\))'
    direct = re.match(r"^" + declaration + r"\(\s*\{\s*session_id\s*:\s*" + session
                      + r"\s*,\s*chars\s*:\s*", code)
    if direct:
        chars, end = _quoted_text(code, direct.end())
        _need(re.match(r"\s*[,}]", code[end:]) is not None, "nonliteral official input expression")
        return direct[1], chars
    variable = re.match(r"^(?:const|let)\s+chars\s*=\s*", code)
    if variable:
        chars, end = _quoted_text(code, variable.end())
        call = re.match(r"\s*;\s*" + declaration + r"\(\s*\{\s*session_id\s*:\s*" + session
                        + r"\s*,\s*chars\s*[,}]", code[end:])
        return (call[1], chars) if call else None
    request = re.match(r"^(?:const|let)\s+([A-Za-z_]\w*)\s*=\s*\{\s*session_id\s*:\s*" + session
                       + r"\s*,\s*chars\s*:\s*", code)
    if request:
        chars, end = _quoted_text(code, request.end())
        tail = re.match(r"\s*,\s*yield_time_ms\s*:\s*\d+\s*,\s*max_output_tokens\s*:\s*\d+\s*}\s*;\s*"
                        + declaration + r"\(\s*" + request[1] + r"\s*\)", code[end:])
        return (request[2], chars) if tail else None
    return None


def _official_lines(raw, observation_session_id):
    """Read the existing normalized format or the actual Codex JSONL snapshot.

    For raw exec records, recognize the literal write_stdin call form used by
    this collector's operator. Decode its JSON string; never evaluate code.
    Restrict the transcript to the process that received this session's IDs,
    retaining its rejected responses too. Unrelated tool calls are ignored.
    """
    try:
        transcript = json.loads(raw)
    except json.JSONDecodeError:
        _need(raw.endswith(b"\n"), "incomplete official transcript")
        records = [json.loads(line) for line in raw.splitlines()]
        calls = []
        for record in records:
            payload = record.get("payload", {})
            if (record.get("type") == "response_item" and payload.get("type") == "custom_tool_call"
                    and payload.get("name") == "exec"):
                code = payload.get("input", "")
                call = _literal_stdin_call(code)
                if call:
                    session, chars = call
                    if chars.strip():
                        calls.append((session, chars))
        sessions = set()
        for session, chars in calls:
            try:
                request = json.loads(chars)
            except ValueError:
                continue
            if isinstance(request, dict) and isinstance(request.get("observation_id"), str) \
                    and request["observation_id"].startswith(observation_session_id + ":"):
                sessions.add(session)
        _need(len(sessions) == 1, "official collector process is unconfirmed")
        return [chars for session, chars in calls if session in sessions]
    if isinstance(transcript, dict):
        if transcript.get("schema") == "official-tool-transcript-capture.v1":
            entries = transcript.get("entries")
            _need(isinstance(entries, list), "invalid captured transcript")
            return [entry.get("input", {}).get("chars", "") for entry in entries
                    if entry.get("tool") == "write_stdin" and entry.get("input", {}).get("chars", "").strip()]
        _need(transcript.get("schema") == "manual-official-tool-transcript.v1", "unsupported transcript")
        transcript = transcript.get("entries")
    _need(isinstance(transcript, list), "invalid official transcript")
    return [entry.get("arguments", entry.get("args", {})).get("chars", "")
            for entry in transcript if entry.get("tool") == "write_stdin"
            and entry.get("arguments", entry.get("args", {})).get("chars", "").strip()]


def load_public_takeover_pair(directory, *, fingerprints: PairFingerprints, transcript_path):
    """Load both arms, keeping exclusions and collection denominators explicit.

    ``transcript_path`` is a separately retained official tool-call transcript
    in the existing normalized format or raw Codex JSONL snapshot. Matching it
    establishes text agreement, not operator identity.
    No private object snapshot or student hidden state is read.
    """
    from raccoon_control.kinematics import Geometry
    from scripts import collect_public_takeover as collector

    directory = Path(directory).resolve()
    _need(isinstance(fingerprints, PairFingerprints), "explicit pair fingerprints required")
    result = json.loads(_read(directory / "result.json", fingerprints.result_sha256))
    raw_events = _read(directory / "interactive_events.jsonl", fingerprints.events_sha256)
    _need(raw_events.endswith(b"\n"), "incomplete event tail")
    events = [collector.direct._parse_json(line.decode("utf-8")) for line in raw_events.splitlines()]
    _need(events and all(isinstance(event, dict) and isinstance(event.get("event"), str) for event in events)
          and events[0].get("event") == "start", "missing or malformed pair start")
    start = events[0]
    _need(_sha(Path(collector.__file__).read_bytes()) == fingerprints.collector_sha256
          and result.get("source_sha256", {}).get(str(Path(collector.__file__).resolve()))
          == fingerprints.collector_sha256, "selected collector differs")
    plan = json.loads(json.dumps(collector.load_plan(directory / "precommitted_plan.json")))
    _need(start.get("plan") == plan and result.get("plan_sha256") == plan["plan_sha256"]
          and result.get("split_sha256") == plan["split_sha256"], "pair train plan differs")
    assessment_complete = result.get("physical_assessment_complete")
    _need(result.get("behavior_collection_complete") is True
          and type(assessment_complete) is bool
          and ((assessment_complete and result.get("status") == "collection_complete" and result.get("error") is None)
               or (not assessment_complete and result.get("status") == "incomplete"
                   and result.get("error") == "posthoc_assessment_incomplete"))
          and result.get("source_integrity_verified") is True
          and result.get("student_weights_unchanged") is True
          and result.get("posthoc_matched_initial_state") is True
          and result.get("posthoc_physical_handoffs_match") is True
          and result.get("scoring_after_both_live_arms") is True
          and result.get("actual_hardware") is False and result.get("final_test_used") is False,
          "pair collection is incomplete or out of scope")
    _need(_one(events, "terminal_prepared").get("summary") == result
          and start.get("source_sha256") == result["source_sha256"]
          and start.get("physical_assessment_during_live") is False
          and start.get("student_checkpoint_sha256") == collector.CHECKPOINT_SHA
          and start.get("initial_state_sha256") == result.get("initial_state_sha256")
          and len(set(result["initial_state_sha256"].values())) == 1,
          "pair terminal or initial-state evidence differs")
    arm_names = ("takeover", "normal")
    ended = [event for event in events if event.get("event") == "arm_ended"]
    _need([event.get("arm") for event in ended] == list(arm_names)
          and all(event.get("result") == result.get("arm_results", {}).get(event["arm"])
                  and event["result"].get("status") == "all_direct_done"
                  and event["result"].get("completed_moves") == len(plan["moves"]) for event in ended),
          "pair arms did not complete")
    scoring = _one(events, "posthoc_assessment_started")
    scoring_index = events.index(scoring)
    _need(scoring.get("completed_arms") == list(arm_names)
          and all(events.index(event) < scoring_index for event in ended), "pair scoring began during execution")
    session_id = start.get("observation_session_id")
    _need(isinstance(session_id, str) and session_id, "missing observation session identity")
    submitted_lines = [event["raw_action_line"] for event in events if event.get("event") == "direct_submission"]
    official_lines = _official_lines(_read(transcript_path, fingerprints.transcript_sha256), session_id)
    _need(submitted_lines and official_lines == submitted_lines, "official submissions differ")
    scores = result.get("posthoc_scores")
    _need(isinstance(scores, dict) and set(scores) == set(arm_names)
          and all(len(scores[arm]) == len(plan["moves"]) for arm in arm_names), "pair assessments missing")
    assessment_stages = {arm: [{"move": move} for move in plan["moves"]] for arm in arm_names}
    _need(collector._posthoc_complete(scores, assessment_stages) is assessment_complete,
          "behavior and physical assessment completion were conflated")
    score_events = [{key: value for key, value in event.items() if key != "event"}
                    for event in events if event.get("event") == "posthoc_assessment"]
    _need(score_events == scores["takeover"] + scores["normal"], "pair assessment records differ")
    seen_observation_ids = set()

    payloads = {}
    for arm in arm_names:
        payload = torch.load(io.BytesIO(_read(directory / f"{arm}_trajectory.pt",
                                  result.get("trajectory_sha256", {}).get(arm))),
                             map_location="cpu", weights_only=True)
        _need(payload.get("schema") == collector.TRAJECTORY_SCHEMA and payload.get("arm") == arm
              and payload.get("plan_sha256") == plan["plan_sha256"]
              and payload.get("split_sha256") == plan["split_sha256"]
              and payload.get("source_sha256") == result["source_sha256"]
              and payload.get("student_action_labels") is False
              and payload.get("physical_outcomes_used_for_labels") is False
              and payload.get("actual_hardware") is False and payload.get("final_test_used") is False,
              "pair trajectory metadata differs")
        payloads[arm] = payload
    runner = _one(events, "student_runner_return")
    decisions = runner.get("decisions")
    _need(runner.get("error") is None and isinstance(decisions, list)
          and len(decisions) == plan["student_decisions"]
          and runner.get("result", {}).get("termination") == "decision_limit"
          and all(d.get("result") == "command_applied" for d in decisions), "student prefix incomplete")
    inputs = [(index, event) for index, event in enumerate(events)
              if event.get("event") in ("student_input", "direct_observation")]
    _need(inputs and inputs[-1][0] < scoring_index, "missing live observations")
    converted = {}
    geometry = Geometry()
    for arm in arm_names:
        rows = payloads[arm].get("rows")
        _need(isinstance(rows, list) and rows, "empty arm history")
        current = [(i, event) for i, event in inputs if
                   ("takeover" if event["event"] == "student_input" else event.get("arm")) == arm]
        _need(len(current) == len(rows), "arm input count differs")
        history, features, actions, targets, labelled, move_masks = [], [], [], [], [], []
        prior_action, prior_target, prior_move, gripper = None, None, 1, "open"
        direct_count = 0
        for offset, ((position, event), row) in enumerate(zip(current, rows, strict=True)):
            move_index = row.get("move_index")
            _need(type(move_index) is int and prior_move <= move_index <= prior_move + 1
                  and move_index <= len(plan["moves"]) and event.get("row_index") == offset,
                  "arm transfer or row sequence differs")
            transfer_start = offset == 0 or move_index != prior_move
            if move_index != prior_move:
                _need(actions[-1] == motion_policy.DONE, "transfer advanced before DONE")
                history = []
            move = plan["moves"][move_index - 1]
            obs = row.get("observation")
            _need(isinstance(obs, dict) and set(obs) == set(collector.PUBLIC_FIELDS)
                  and obs.get("gripper_command") == gripper, "arm public observation differs")
            prior = student._applied_target(row.get("previous"))
            if prior_target is None:
                # Captured student ``previous`` tensors are float32; the common
                # runner also retains the original confirmed degree values.
                prior_target = student._applied_target(decisions[0].get("previous_applied_target_deg")) \
                    if arm == "takeover" else list(prior)
            _need(_near(prior, prior_target, 2e-5), "arm prior applied control differs")
            encoded = scene_features.encode_scene_features(obs, move["before"], move["task"],
                                                           plan["registered_slot_xy_mm"])
            history.append(encoded)
            window, mask = scene_features.history_window(history)
            _need(torch.equal(torch.tensor(row.get("window")), torch.tensor(window))
                  and row.get("mask") == mask, "arm public history differs")
            feature = student.encode_step(obs, move["before"], move["task"],
                slot_xy_mm=plan["registered_slot_xy_mm"], previous_applied_action=prior_action,
                previous_applied_target_deg=prior_target, transfer_start=transfer_start,
                registration_start=offset == 0)
            next_position = next((i for i, _event in inputs if i > position), scoring_index)
            group = events[position + 1:next_position]
            is_teacher = row.get("phase") == "direct_operator"
            if not is_teacher:
                _need(arm == "takeover" and offset < len(decisions)
                      and row.get("phase") == "student_context" and row.get("labelled") is False
                      and row.get("student_decision") == decisions[offset]
                      and {k: v for k, v in row.items() if k != "student_decision"}
                      == {k: v for k, v in event.items() if k not in ("event", "row_index")},
                      "student prefix was relabelled or changed")
                decision = decisions[offset]
                action = decision.get("action")
                _need(action in ("move", "open", "close") and decision.get("index") == offset
                      and _near(decision.get("previous_applied_target_deg", []), prior_target, 2e-5)
                      and torch.equal(torch.tensor(decision.get("policy_features")), torch.tensor(encoded)),
                      "student returned decision differs")
                sent = _one(group, "student_command_started")
                returned = _one(group, "student_command_returned")["returned_command"]
                _need([e["event"] for e in group if e["event"] in
                      ("student_command_started", "student_command_returned")]
                      == ["student_command_started", "student_command_returned"], "student return order differs")
                applied = student._applied_target(decision.get("command", {}).get("joints_deg"))
                _need(_near(sent.get("target_joints_deg", []), applied)
                      and sent.get("seconds") == 1., "student applied command differs")
                label_action, target = -1, [float("nan")] * 4
            else:
                _need(offset >= (len(decisions) if arm == "takeover" else 0)
                      and row.get("labelled") is True and event.get("move_index") == move_index,
                      "direct row position or label differs")
                visible = event.get("observation", {})
                _need(all(visible.get(key) == obs[key] for key in collector.PUBLIC_FIELDS)
                      and visible.get("expected_scene") == move["before"]
                      and visible.get("task") == move["task"]
                      and visible.get("registered_slot_xy_mm") == plan["registered_slot_xy_mm"]
                      and visible.get("physical_confirmation") == "not_inspected"
                      and _near(visible.get("previous_applied_target_deg", []), prior_target),
                      "direct visible observation differs")
                direct_count += 1
                observation_id = f"{session_id}:{arm}:{direct_count}"
                _need(event.get("observation_id") == row.get("observation_id") == observation_id
                      and observation_id not in seen_observation_ids, "direct observation identity differs")
                seen_observation_ids.add(observation_id)
                submissions = [e for e in group if e.get("event") == "direct_submission"]
                rejections = [e for e in group if e.get("event") == "response_id_rejected"]
                _need(submissions and len(rejections) == len(submissions) - 1
                      and len(row.get("rejected_submissions", [])) == len(rejections),
                      "rejected response count differs")
                for rejected, submission_attempt, recorded in zip(
                        rejections, submissions[:-1], row["rejected_submissions"], strict=True):
                    attempt = collector.direct._parse_json(submission_attempt["raw_action_line"])
                    _need(rejected.get("observation_id") == observation_id
                          and rejected.get("supplied_observation_id") == attempt.get("observation_id")
                          and rejected.get("command_dispatched") is False
                          and rejected.get("reason") in ("missing_observation_id", "duplicate_response",
                                                        "stale_response", "unknown_observation_id")
                          and recorded == {"raw": submission_attempt["raw_action_line"].rstrip("\r\n"),
                                           "reason": rejected["reason"],
                                           "supplied_observation_id": attempt.get("observation_id")},
                          "rejected response was applied or changed")
                submission = submissions[-1]
                request = collector.direct._parse_json(submission["raw_action_line"])
                claimed = _one(group, "observation_response_claimed")
                _need(request.get("observation_id") == submission.get("observation_id")
                      == claimed.get("observation_id") == observation_id,
                      "direct response is not bound to its current observation")
                explicit = {key: value for key, value in request.items() if key != "observation_id"}
                action_upper, resolved = collector.direct._resolve(explicit, prior_target, geometry)
                action = action_upper.lower()
                label_action = motion_policy.ACTIONS.index(action)
                decision = _one(group, "direct_decision")
                _need(row.get("submitted_raw") == submission["raw_action_line"].rstrip("\r\n")
                      and row.get("submitted") == request and row.get("label_action") == label_action
                      and decision.get("action") == action_upper and decision.get("observation_id") == observation_id
                      and _near(row.get("resolved_target_joints_deg", []), resolved)
                      and _near(decision.get("target_joints_deg", []), resolved)
                      and _near(decision.get("previous_applied_target_deg", []), prior_target),
                      "direct submission or resolved target differs")
                if action == "done":
                    _need(row.get("teacher_result") == "done_hold_completed"
                          and _one(group, "direct_done_hold").get("move_index") == move_index
                          and _one(group, "direct_done_hold").get("observation_id") == observation_id
                          and not any(e.get("event") == "direct_command_started" for e in group),
                          "direct DONE hold is unconfirmed")
                    applied, returned = list(prior_target), None
                    expected_order = ["observation_response_claimed", "direct_decision", "direct_done_hold"]
                else:
                    sent = _one(group, "direct_command_started")
                    returned = _one(group, "direct_command_returned")["returned_command"]
                    confirmed = _one(group, "direct_command_confirmed")
                    applied = student._applied_target(row.get("actual_applied_target_deg"))
                    _need(row.get("teacher_result") == "command_applied"
                          and _near(applied, resolved) and _near(sent.get("target_joints_deg", []), applied)
                          and confirmed.get("action") == action_upper
                          and sent.get("observation_id") == confirmed.get("observation_id") == observation_id
                          and _one(group, "direct_command_returned").get("observation_id") == observation_id
                          and _near(confirmed.get("actual_applied_targets", {}).get("joints_deg", []), applied),
                          "direct command is not normally confirmed")
                    expected_order = ["observation_response_claimed", "direct_decision", "direct_command_started",
                                      "direct_command_returned", "direct_command_confirmed"]
                _need([e["event"] for e in group if e["event"] in
                      ("observation_response_claimed", "direct_decision", "direct_done_hold",
                       "direct_command_started", "direct_command_returned", "direct_command_confirmed")]
                      == expected_order, "direct command return order differs")
                target = applied if action == "move" else [float("nan")] * 4
                _need(_near(row.get("label_target", []), [x / 180. for x in resolved], 1e-10),
                      "direct stored target differs")
            if returned is not None:
                _need(returned.get("source") == "mujoco_simulation" and returned.get("simulated") is True
                      and type(returned.get("time_seconds")) in (int, float)
                      and math.isfinite(returned["time_seconds"]), "command return is unconfirmed")
                _need(action == "move" or _near(applied, prior_target), "gripper command changed held joints")
                prior_action, prior_target = action, list(applied)
                if action in ("open", "close"):
                    gripper = "open" if action == "open" else "closed"
            features.append(feature)
            actions.append(label_action)
            targets.append(target)
            labelled.append(is_teacher)
            move_masks.append(is_teacher and action == "move")
            prior_move = move_index
        _need(actions[-1] == motion_policy.DONE and actions.count(motion_policy.DONE) == len(plan["moves"])
              and prior_move == len(plan["moves"]), "arm final DONE or transfer count differs")
        p = payloads[arm]
        _tensor(p.get("windows"), torch.tensor([r["window"] for r in rows], dtype=torch.float32), "stored windows differ")
        _tensor(p.get("masks"), torch.tensor([r["mask"] for r in rows], dtype=torch.bool), "stored history masks differ")
        _tensor(p.get("previous"), torch.tensor([r["previous"] for r in rows], dtype=torch.float32), "stored prior controls differ")
        _tensor(p.get("label_mask"), torch.tensor(labelled, dtype=torch.bool), "stored supervision mask differs")
        _tensor(p.get("actions"), torch.tensor(actions, dtype=torch.long), "stored action labels differ")
        _tensor(p.get("targets"), torch.tensor([r.get("label_target", [0.] * 4) for r in rows], dtype=torch.float32),
                "stored target tensor differs")
        converted[arm] = (features, actions, targets, labelled, move_masks)
    _need(_near(payloads["takeover"]["rows"][0]["previous"], payloads["normal"]["rows"][0]["previous"], 2e-5),
          "paired initial control differs")
    episodes, excluded = [], {}
    for arm in arm_names:
        physical = scores[arm]
        eligible = all(score.get("move_index") == index and score.get("status") == "success"
            and score.get("sample_count") == 501 and score.get("whole_scene_check_scope_valid") is True
            and score.get("physical_hold_matches") is True and score.get("failed_checks") == []
            and math.isclose(score.get("stable_interval_seconds", 0), 1., abs_tol=1e-6, rel_tol=0)
            for index, score in enumerate(physical, 1))
        if not eligible:
            excluded[arm] = "physical_transfer_failed_or_unconfirmed"
            continue
        features, actions, targets, labelled, moves = converted[arm]
        evidence = {"case_id": f"{directory.name}:{arm}", "source_kind":
            "direct_operator_after_student_prefix" if arm == "takeover" else "direct_operator",
            "source_trajectories": (str(directory / f"{arm}_trajectory.pt"),),
            "trajectory_sha256": result["trajectory_sha256"][arm],
            "student_prefix_length": len(decisions) if arm == "takeover" else 0,
            "train_groups": tuple(move["split_group"] for move in plan["moves"]),
            "paired_start_id": str(directory), "pair_arm": arm, "transfer_count": len(plan["moves"]),
            "physics_seed": plan["physics_seed"], "plan_sha256": plan["plan_sha256"],
            "split_sha256": plan["split_sha256"], "result_sha256": fingerprints.result_sha256,
            "events_sha256": fingerprints.events_sha256, "transcript_sha256": fingerprints.transcript_sha256,
            "collector_sha256": fingerprints.collector_sha256,
            "source_files_sha256": {**result["source_sha256"],
                str(directory / "result.json"): fingerprints.result_sha256,
                str(directory / "interactive_events.jsonl"): fingerprints.events_sha256,
                str(directory / "precommitted_plan.json"): plan["plan_sha256"],
                str(directory / f"{arm}_trajectory.pt"): result["trajectory_sha256"][arm],
                str(Path(transcript_path).resolve()): fingerprints.transcript_sha256},
            "post_transfer_scene": "expected_only",
            "submission_lines_match_supplied_official_transcript": True,
            "transcript_proof_scope": "static_submission_text_and_observation_identity_agreement",
            "numeric_session_binding_independently_verified": False,
            "public_response_and_timing_independently_reconciled": False,
            "command_return": "collector_verified_simulated_servo_target_self_recorded",
            "runtime_model_identity_independently_verified": False,
            "decision_maker_independence_verified": False, "actual_hardware": False, "final_test_used": False}
        episodes.append(TrainingEpisode(torch.tensor(features, dtype=torch.float32), torch.tensor(actions),
            torch.tensor(targets, dtype=torch.float32), torch.ones(len(actions), dtype=torch.bool),
            torch.tensor(labelled, dtype=torch.bool), torch.tensor(moves, dtype=torch.bool), evidence))
    return LoadedPair(tuple(episodes), excluded, {"attempted_arms": 2, "accepted_arms": len(episodes),
        "behavior_collection_complete": True, "physical_assessment_complete": assessment_complete,
        "independent_initial_samples": 1, "posthoc_scores": scores,
        "direct_decisions_by_arm": {arm: sum(v[3]) for arm, v in converted.items()},
        "unlabelled_student_context_rows": len(decisions), "actual_hardware": False, "final_test_used": False})
