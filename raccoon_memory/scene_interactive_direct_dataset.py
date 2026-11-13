"""Load explicitly selected, complete interactive training demonstrations.

Each returned episode retains the whole registration across two transfers.
Only public observations and confirmed applied control enter its tensors.
Submission text, post-DONE assessment and source information are evidence,
not model features. This module does not train or run a robot or simulator.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import torch

from . import motion_policy, scene_applied_action_student as student, scene_features
from .scene_applied_action_dataset import TrainingEpisode
from .scene_direct_applied_dataset import SPLIT_SHA256


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/architecture_research"
COLLECTOR_SHA256 = "b1ee5d8fafe62ab9fcf77a0eb8f2882ceea32932bf61b8ed060e679040fca636"
REVIEWED_RUNS = (
    ("20261103_terra_multitask_manual_trial_v2",
     "04563ab98081c58e70e0bdd3f061fd48940dd0c23945256dd8e2e215071368c6",
     "official_tool_transcript_combined.json", "terra_multitask_manual_trial_v1.json"),
    ("20261103_terra_multitask_stack_manual_trial_v1",
     "6d15c2993d29c28bd2c2b9f4108ad5e1376ab8fa91a2bf276e6804c5e05e187f",
     "combined_official_tool_transcript.json", "terra_multitask_stack_manual_trial_v1.json"),
)
ADDITIONAL_REVIEWED_RUNS = {
    "20261103_terra_multitask_varied_trial_v1": {
        "data_subdir": "collection", "plan": "terra_multitask_varied_trial_v1.json",
        "files": {
            "result.json": ("collection/result.json", "7a6494178554d4e669f9de24f92469c3fba9ef6f5b46940a1871c7868421de6b"),
            "interactive_events.jsonl": ("collection/interactive_events.jsonl", "857c0456afc03392cd766afa37893d99e711a356a9990c2b4ca2fff0f52af632"),
            "direct_trajectory_move1.pt": ("collection/direct_trajectory_move1.pt", "8510a96c9dbcf47e0d63a9887deababf3ab01d13de17d01ca6484b84e66b06e6"),
            "direct_trajectory_move2.pt": ("collection/direct_trajectory_move2.pt", "9a88194a459a8f347ae52a61bced9c4e8fc99f1601f4593768e8c32a93a9d325"),
            "tool_transcript.json": ("combined_official_tool_transcript.json", "a954498836886c86f1ddaac345821fe7e7391d548e3e5ad49cc4a41cfe982d31"),
        },
    },
    "20261103_terra_multitask_unstack_trial_v1": {
        "data_subdir": "", "plan": "terra_multitask_unstack_trial_v1.json",
        "files": {
            "result.json": ("result.json", "0d735479f21b73b6a3c5e508de98ce01925f5656b77145e89d80be1dd81a18b8"),
            "interactive_events.jsonl": ("interactive_events.jsonl", "e41a972d78c5139e30d11c2cbba9b5c2634b775a04ab89aab6493c9036e62ee1"),
            "direct_trajectory_move1.pt": ("direct_trajectory_move1.pt", "04ba07acf8ec7208e488f6403dda48db34872108b482761172d7e02de39b304a"),
            "direct_trajectory_move2.pt": ("direct_trajectory_move2.pt", "f75217652175431f036266e925ada2c54017b27a86549d667c88057278bc7ea7"),
            "tool_transcript.json": ("recovered_official_tool_transcript.json", "71749cac66396229b3d919ebabccfda736ef7d8c067ba453c3dabb99ab537220"),
        },
    },
    "20261103_terra_multitask_four_trial_v1": {
        "data_subdir": "", "plan": "terra_multitask_four_trial_v1.json",
        "files": {
            "result.json": ("result.json", "033fc48e3469762b0f02065d56cf28eeada367b3c740549dce2dae1832881021"),
            "interactive_events.jsonl": ("interactive_events.jsonl", "72a0a647c2aea87cafc7b98b749a4f7bdc965823a25555c000c989e6af0ff9cb"),
            "direct_trajectory_move1.pt": ("direct_trajectory_move1.pt", "10d8c5baa36f1270fedde39b689be0da3584cd048e962a6dd0e83551529e1e2e"),
            "direct_trajectory_move2.pt": ("direct_trajectory_move2.pt", "be3a13fd06fe2932b27783c244aed2362d674bde0c8ef494c720fefec1640658"),
            "tool_transcript.json": ("recovered_official_tool_transcript.json", "8c824ddff8c39b5ac69b5c2a6cba6758228990ec7d422c2c7127d39899ef6575"),
        },
    },
}
REVIEWED_RUN_IDS = tuple(spec[0] for spec in REVIEWED_RUNS) + tuple(ADDITIONAL_REVIEWED_RUNS)
PUBLIC_OBSERVATION_KEYS = ("joints_deg", "sensor_valid", "moving", "gripper_command",
                           "gripper_reported_state", "gripper_reported_valid", "gripper_feedback_fresh")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _near(left, right, tolerance=1e-7):
    return (len(left) == len(right) == 4 and
            all(math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0)
                for a, b in zip(left, right)))


def _load_run(spec):
    """Validate one reviewed full recording, returning one causal episode."""
    name, manifest_sha, transcript_name, plan_name = spec
    directory = (ART / name).resolve()
    manifest_raw = (directory / "fingerprint_manifest.json").read_bytes()
    _need(_sha(manifest_raw) == manifest_sha, "interactive manifest fingerprint differs")
    manifest = json.loads(manifest_raw)
    _need(manifest.get("algorithm") == "SHA-256" and isinstance(manifest.get("files"), list),
          "interactive manifest is invalid")
    hashes = {}
    for entry in manifest["files"]:
        path = Path(entry["path"]).resolve()
        if path.parent == directory:
            _need(path.name not in hashes, "duplicate interactive source file")
            hashes[path.name] = (entry["sha256"], entry["bytes"])

    def read(name):
        _need(name in hashes, "interactive source fingerprint missing")
        raw = (directory / name).read_bytes()
        _need((_sha(raw), len(raw)) == hashes[name], "interactive source fingerprint differs")
        return raw

    return _convert_recording(name, directory, read, transcript_name, plan_name,
        {"manifest_sha256": manifest_sha,
         "source_files_sha256": {str(directory / filename): digest for filename, (digest, _) in hashes.items()}})


def _load_additional_run(name):
    spec = ADDITIONAL_REVIEWED_RUNS[name]
    root = (ART / name).resolve()

    def read(filename):
        _need(filename in spec["files"], "interactive source fingerprint missing")
        relative, digest = spec["files"][filename]
        raw = (root / relative).read_bytes()
        _need(_sha(raw) == digest, "interactive source fingerprint differs")
        return raw

    return _convert_recording(name, root / spec["data_subdir"], read, "tool_transcript.json", spec["plan"],
        {"source_files_sha256": {str(root / relative): digest for relative, digest in spec["files"].values()}})


def _convert_recording(name, directory, read, transcript_name, plan_name, binding):
    # Reuse the collector's train-split/seed/group validation. No simulator is
    # created and no held-out trajectory is opened.
    from scripts import terra_teacher_multitask_train as collector

    result = json.loads(read("result.json"))
    log_raw = read("interactive_events.jsonl")
    _need(log_raw.endswith(b"\n"), "interactive recording is incomplete")
    events = [json.loads(line) for line in log_raw.splitlines()]
    transcript = json.loads(read(transcript_name))
    if isinstance(transcript, dict) and transcript.get("schema") == "manual-official-tool-transcript.v1":
        transcript = transcript.get("entries")
    trajectories = [torch.load(io.BytesIO(read(f"direct_trajectory_move{i}.pt")),
                               map_location="cpu", weights_only=True) for i in (1, 2)]
    _need(isinstance(transcript, list) and all(isinstance(e, dict) for e in events),
          "interactive evidence has invalid shape")

    plan_path = ROOT / "configs" / plan_name
    # The collector returns tuple coordinates, while the durable JSON event
    # stores the same coordinates as arrays.
    plan = json.loads(json.dumps(collector.load_plan(plan_path)))
    _need(_sha(plan_path.read_bytes()) == result.get("plan_sha256") == plan.get("plan_sha256")
          and _sha((ROOT / "configs/scene_split_plan.v1.json").read_bytes()) == SPLIT_SHA256
          and plan.get("split_sha256") == result.get("split_sha256") == SPLIT_SHA256,
          "interactive train plan or split differs")
    _need(_sha(Path(collector.__file__).read_bytes()) == COLLECTOR_SHA256,
          "reviewed interactive collector changed")
    _need(events and events[0].get("event") == "start" and events[0].get("plan") == plan
          and events[0].get("source_sha256") == result.get("source_sha256")
          and result["source_sha256"].get(str(Path(collector.__file__).resolve())) == COLLECTOR_SHA256,
          "interactive collection provenance differs")
    _need(result.get("status") == "collection_complete" and result.get("scoring_after_both_done") is True
          and result.get("actual_hardware") is False and result.get("final_test_used") is False
          and result.get("expected_layout_physically_confirmed") is False,
          "interactive collection did not finish with expected-only layouts")
    scores = result.get("posthoc_scores")
    _need(isinstance(scores, list) and len(scores) == 2 and
          all(score.get("move_index") == index and score.get("status") == "success"
              and score.get("sample_count") == 501
              and math.isclose(score.get("stable_interval_seconds", 0), 1., abs_tol=1e-6, rel_tol=0)
              and score.get("whole_scene_check_scope_valid") is True
              and score.get("physical_hold_matches") is True and score.get("failed_checks") == []
              for index, score in enumerate(scores, 1)), "interactive physical assessment is incomplete")
    terminal = [e for e in events if e.get("event") == "terminal_prepared"]
    _need(len(terminal) == 1 and terminal[0].get("summary") == result,
          "interactive terminal record differs from result")
    scored_events = [{key: value for key, value in event.items() if key != "event"}
                     for event in events if event.get("event") == "posthoc_assessment"]
    _need(scored_events == scores, "interactive assessment records differ")

    observations = [(i, e) for i, e in enumerate(events) if e.get("event") == "observation"]
    submissions = [e["raw_action_line"] for e in events if e.get("event") == "submission"]
    transcript_submissions = [e.get("arguments", e.get("args", {})).get("chars", "") for e in transcript
                              if e.get("tool") == "write_stdin"
                              and e.get("arguments", e.get("args", {})).get("chars", "").strip()]
    _need(observations and submissions == transcript_submissions
          and len(observations) == len(submissions), "official and interactive submissions differ")
    scoring = [i for i, e in enumerate(events) if e.get("event") == "posthoc_assessment_started"]
    _need(len(scoring) == 1 and scoring[0] > observations[-1][0], "assessment timing differs")

    features, actions, targets, moves = [], [], [], []
    previous_action, previous_target, history = None, None, []
    offsets = [0, 0]
    previous_move = 1
    for row_index, (position, obs) in enumerate(observations):
        next_position = observations[row_index + 1][0] if row_index + 1 < len(observations) else scoring[0]
        group = events[position + 1:next_position]
        move_index = obs.get("move_index")
        _need(move_index in (1, 2) and previous_move <= move_index <= previous_move + 1,
              "interactive move sequence differs")
        if move_index != previous_move:
            _need(actions and actions[-1] == motion_policy.DONE,
                  "interactive transfer advanced before DONE")
            transitions = [e for e in events[observations[row_index - 1][0]:position]
                           if e.get("event") == "expected_layout_advanced"]
            _need(len(transitions) == 1 and transitions[0].get("physical_confirmation") == "not_inspected"
                  and transitions[0].get("expected_scene") == plan["moves"][1]["before"]
                  and transitions[0].get("task") == plan["moves"][1]["task"],
                  "interactive expected layout transition differs")
            history = []
        move = plan["moves"][move_index - 1]
        payload = trajectories[move_index - 1]
        offset = offsets[move_index - 1]
        public, raw = obs.get("student_observation"), obs.get("raw_observation")
        _need(isinstance(public, dict) and isinstance(raw, dict)
              and all(key in public and public[key] == raw.get(key) for key in PUBLIC_OBSERVATION_KEYS)
              and public.get("expected_scene") == move["before"] and public.get("task") == move["task"]
              and public.get("registered_slot_xy_mm") == plan["registered_slot_xy_mm"]
              and public.get("physical_confirmation") == "not_inspected", "interactive public input differs")
        prior = student._applied_target(public.get("previous_applied_target_deg"))
        if previous_target is None:
            _need(public.get("previous_target_origin") == "initial_simulated_servo_target"
                  and public.get("last_normal_applied_command") is None,
                  "interactive initial control source differs")
        else:
            last = public.get("last_normal_applied_command")
            _need(_near(prior, previous_target) and isinstance(last, dict)
                  and last.get("action", "").lower() == previous_action
                  and _near(last.get("target_joints_deg", []), previous_target)
                  and last.get("gripper_command") == public.get("gripper_command"),
                  "interactive previous applied control differs")
        public_input = {"schema_version": "PolicyObservation.phase1",
                        **{key: public[key] for key in PUBLIC_OBSERVATION_KEYS}}
        row = student.encode_step(public_input, move["before"], move["task"],
            slot_xy_mm=plan["registered_slot_xy_mm"], previous_applied_action=previous_action,
            previous_applied_target_deg=prior, transfer_start=(offset == 0),
            registration_start=(row_index == 0))
        history.append(row[:len(scene_features.FEATURE_NAMES)])
        window, mask = scene_features.history_window(history)
        _need(torch.equal(torch.tensor(window, dtype=torch.float32), payload["windows"][offset])
              and torch.equal(torch.tensor(mask, dtype=torch.bool), payload["masks"][offset])
              and torch.equal(payload["previous"][offset], torch.tensor(prior, dtype=torch.float32))
              and torch.equal(torch.tensor(obs["window"], dtype=torch.float32), payload["windows"][offset])
              and torch.equal(torch.tensor(obs["mask"], dtype=torch.bool), payload["masks"][offset]),
              "interactive feature history differs")
        selected = {kind: [e for e in group if e.get("event") == kind] for kind in
                    ("submission", "decision_resolved", "command_started", "command_returned", "decision_recorded")}
        _need(all(len(selected[kind]) == 1 for kind in ("submission", "decision_resolved", "decision_recorded")),
              "interactive decision record is incomplete")
        order = [e.get("event") for e in group if e.get("event") in selected]
        submitted = json.loads(selected["submission"][0]["raw_action_line"])
        resolved, recorded = selected["decision_resolved"][0], selected["decision_recorded"][0]
        action = submitted.get("action", "").lower()
        _need(action in motion_policy.ACTIONS and submitted == resolved.get("submitted") == recorded.get("submitted")
              and all(e.get("move_index") == move_index for values in selected.values() for e in values)
              and _near(prior, resolved.get("previous_applied_target_deg", []))
              and _near(prior, recorded.get("previous_applied_target_deg", []))
              and int(payload["actions"][offset]) == motion_policy.ACTIONS.index(action),
              "interactive submitted action or prior control differs")
        proposed = student._applied_target(resolved.get("resolved_target_joints_deg"))
        applied = student._applied_target(recorded.get("applied_target_deg"))
        _need(torch.equal(payload["targets"][offset], torch.tensor([x / 180 for x in proposed], dtype=torch.float32)),
              "interactive saved target differs")
        if action == "done":
            _need(recorded.get("result") == "done_and_hold_completed" and _near(applied, prior)
                  and order == ["submission", "decision_resolved", "decision_recorded"],
                  "interactive DONE did not complete without a command")
        else:
            _need(order == ["submission", "decision_resolved", "command_started",
                            "command_returned", "decision_recorded"],
                  "interactive command return is missing")
            started, returned = selected["command_started"][0], selected["command_returned"][0]["returned_command"]
            _need(recorded.get("result") == "command_applied" and started.get("action", "").lower() == action
                  and _near(started.get("target_joints_deg", []), proposed) and _near(applied, proposed)
                  and returned.get("source") == "mujoco_simulation" and returned.get("simulated") is True
                  and math.isclose(returned["time_seconds"] - raw["observed_monotonic"], 1., abs_tol=1e-6, rel_tol=0)
                  and (action == "move" or _near(applied, prior)), "interactive command was not normally applied")
            previous_action, previous_target = action, applied
        features.append(torch.tensor(row, dtype=torch.float32))
        actions.append(motion_policy.ACTIONS.index(action))
        moves.append(action == "move")
        targets.append(applied if action == "move" else [float("nan")] * 4)
        offsets[move_index - 1] += 1
        previous_move = move_index

    for index, payload in enumerate(trajectories, 1):
        move = plan["moves"][index - 1]
        _need(offsets[index - 1] == len(payload["actions"]) > 0
              and int(payload["actions"][-1]) == motion_policy.DONE
              and payload.get("schema") == "terra-interactive-observation-direct-decision.v1"
              and payload.get("sequence_move_index") == index and payload.get("expected_before") == move["before"]
              and payload.get("expected_after") == move["expected_after"]
              and [list(x) for x in payload.get("registered_slot_xy_mm", ())] == plan["registered_slot_xy_mm"]
              and payload.get("physics_seed") == plan["physics_seed"]
              and payload.get("split_sha256") == SPLIT_SHA256
              and payload.get("plan_sha256") == result["plan_sha256"]
              and payload.get("source_sha256") == result["source_sha256"]
              and payload.get("decision_origin") == "explicit_interactive_submissions"
              and payload.get("physical_outcomes_used_for_labels") is False
              and payload.get("final_test_used") is False, "interactive saved episode metadata differs")
    _need(actions.count(motion_policy.DONE) == 2 and actions[-1] == motion_policy.DONE
          and len([e for e in events if e.get("event") == "expected_layout_advanced"]) == 1,
          "interactive expected layout count differs")
    n = len(actions)
    evidence = {"case_id": name, "source_trajectories": tuple(str(directory / f"direct_trajectory_move{i}.pt")
                 for i in (1, 2)), "transfer_count": 2, "source_kind": "direct_operator",
                **binding, "physics_seed": plan["physics_seed"],
                "train_groups": tuple(move["split_group"] for move in plan["moves"]),
                "submission_lines_match_official_transcript": True,
                "command_return": "collector_verified_simulated_servo_target_self_recorded",
                "initial_target_source": "initial_simulated_servo_target_self_recorded",
                "runtime_model_identity_independently_verified": False,
                "public_response_and_timing_independently_reconciled": False,
                "post_transfer_scene": "expected_only", "physical_assessment": "post_done_summary_501_samples"}
    return TrainingEpisode(torch.stack(features), torch.tensor(actions, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32), torch.ones(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool), torch.tensor(moves, dtype=torch.bool), evidence)


def load_reviewed_interactive_episodes():
    """Load exactly the two pinned demonstrations, without including new runs."""
    return tuple(_load_run(spec) for spec in REVIEWED_RUNS)


def load_selected_interactive_episodes(run_ids):
    """Load only named, reviewed histories; reject duplicates and unknown IDs."""
    _need(isinstance(run_ids, (tuple, list)) and run_ids
          and all(isinstance(name, str) and name in REVIEWED_RUN_IDS for name in run_ids)
          and len(set(run_ids)) == len(run_ids), "select distinct reviewed interactive run IDs")
    original = {spec[0]: spec for spec in REVIEWED_RUNS}
    return tuple(_load_run(original[name]) if name in original else _load_additional_run(name)
                 for name in run_ids)
