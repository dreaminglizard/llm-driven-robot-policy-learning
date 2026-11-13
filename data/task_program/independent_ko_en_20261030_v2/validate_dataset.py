"""Validate frozen development examples and gold compatibility without model calls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from raccoon_memory import task_program as v1  # noqa: E402
from raccoon_memory import task_program_v2 as v2  # noqa: E402
from scripts import evaluate_task_program_v2 as existing_evaluator  # noqa: E402
from judge import classify, projection  # noqa: E402

DATA = HERE / "development80.jsonl"
BASELINE = ROOT / "data/task_program/v2_1/combined_development112.jsonl"
V1_DATA = HERE.parent / "independent_ko_en_20261030_v1" / "development80.jsonl"
V1_VALIDATION = HERE.parent / "independent_ko_en_20261030_v1" / "validation.json"
REVIEW = ROOT / "docs/2026-10-30_독립_한영명령_개발자료_정답검토.md"
REPORT = HERE / "validation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def main():
    rows = read_jsonl(DATA)
    previous_rows = read_jsonl(V1_DATA)
    if sha(V1_DATA) != "3b39eca8cc66cf29eae6e2959fcfc3b54b8ac57dde71ac1b5ed8777a22676cac":
        raise ValueError("reviewed v1 dataset changed")
    if sha(V1_VALIDATION) != "7c46b006e696c2ceb141564c3e0a6051529ae259d3cede7002b3176561674eab":
        raise ValueError("reviewed v1 validation changed")
    baseline = read_jsonl(BASELINE)  # Only the existing 112-case development file.
    if len(rows) != 80 or Counter(row["language"] for row in rows) != {"ko": 40, "en": 40}:
        raise ValueError("the fixed development set must contain 40 examples per language")
    if len({row["id"] for row in rows}) != 80:
        raise ValueError("case IDs are not unique")
    if [row["id"] for row in rows] != [row["id"] for row in previous_rows]:
        raise ValueError("v2 row order or case IDs differ from v1")
    changed_texts, changed_rationales = [], []
    for old, new in zip(previous_rows, rows):
        if old["transcript"] != new["transcript"]:
            changed_texts.append(new["id"])
        if old["grading_reason"] != new["grading_reason"]:
            changed_rationales.append(new["id"])
        if old["reference_context"] != new["reference_context"]:
            raise ValueError(f"context changed: {new['id']}")
        if [projection(alt) for alt in old["expected_any_of"]] != [projection(alt) for alt in new["expected_any_of"]]:
            raise ValueError(f"gold meaning changed: {new['id']}")
        if old["family"] == "finish_current_no_new_round":
            excluded = {"transcript", "grading_reason", "expected_any_of"}
            if {k: v for k, v in old.items() if k not in excluded} != {k: v for k, v in new.items() if k not in excluded}:
                raise ValueError(f"unrelated field changed: {new['id']}")
            for old_alt, new_alt in zip(old["expected_any_of"], new["expected_any_of"]):
                for old_step, new_step in zip(old_alt["steps"], new_alt["steps"]):
                    old_step = json.loads(json.dumps(old_step))
                    new_step = json.loads(json.dumps(new_step))
                    old_step.pop("evidence")
                    new_step.pop("evidence")
                    if old_step["op"] == "continue":
                        old_step["target"].pop("reference")
                        new_step["target"].pop("reference")
                    if old_step != new_step:
                        raise ValueError(f"other gold field changed: {new['id']}")
        elif new["id"] == "independent.negated_memory_positive_move.en":
            if {k: v for k, v in old.items() if k != "grading_reason"} != {k: v for k, v in new.items() if k != "grading_reason"}:
                raise ValueError("singular case changed beyond its rationale")
        elif old != new:
            raise ValueError(f"unrelated input or metadata changed: {new['id']}")
    if set(changed_texts) != {"independent.finish_current_no_new_round.ko", "independent.finish_current_no_new_round.en"}:
        raise ValueError("the changed input set is not the two requested utterances")
    if set(changed_rationales) != set(changed_texts) | {"independent.negated_memory_positive_move.en"}:
        raise ValueError("the changed rationale set differs from the requested scope")
    loaded_rows, loaded_digest = existing_evaluator.load_records(DATA)
    if loaded_rows != rows or loaded_digest != sha(DATA):
        raise ValueError("existing v2 evaluator did not load the same frozen dataset")
    if any(projection(alt) != existing_evaluator.semantic_projection(alt)
           for row in rows for alt in row["expected_any_of"]):
        raise ValueError("semantic projection differs from the existing v2 evaluator")
    family_counts = Counter(row["family"] for row in rows)
    if len(family_counts) != 40 or any(count != 2 for count in family_counts.values()):
        raise ValueError("expected 40 paired families")
    if any(not row["transcript"].strip() or not row["grading_reason"].strip() for row in rows):
        raise ValueError("every row needs text and a rationale; silence is a separate input check")
    baseline_texts = {(row.get("language"), normalize(row.get("transcript") or "")) for row in baseline}
    duplicates = [row["id"] for row in rows if (row["language"], normalize(row["transcript"])) in baseline_texts]
    if duplicates:
        raise ValueError(f"normalized exact text duplicates in prior development data: {duplicates}")

    rejections, schema_failures, input_pairs = [], [], set()
    compatible_rows = 0
    for row in rows:
        v1.normalize_context(row["reference_context"])
        pair = (row["language"], row["transcript"], json.dumps(row["reference_context"], sort_keys=True))
        if pair in input_pairs:
            raise ValueError(f"duplicate text and context: {row['id']}")
        input_pairs.add(pair)
        if not row["expected_any_of"]:
            raise ValueError(f"missing gold alternatives: {row['id']}")
        expected_contains_clarify = any(s["op"] == "clarify" for alt in row["expected_any_of"] for s in alt["steps"])
        if expected_contains_clarify and (row["question_review"] != "manual_for_every_clarify_step"
                                          or row["clarify_requires_return_target_question"] is not True):
            raise ValueError(f"free-form clarification lacks a manual-review marker: {row['id']}")
        accepted = False
        for index, alt in enumerate(row["expected_any_of"]):
            program = {**alt, "request_id": row["id"]}
            if projection(program) is None:
                raise ValueError(f"gold is not projectable: {row['id']}")
            try:
                v2.validate_schema(program)
            except Exception as exc:
                schema_failures.append({"id": row["id"], "alternative": index,
                                        "code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)})
                continue
            for step in program["steps"]:
                if step["evidence"] not in row["transcript"]:
                    raise ValueError(f"ungrounded evidence: {row['id']}")
                if step["op"] in ("return", "continue", "repeat") and step["target"]["reference"] not in row["transcript"]:
                    raise ValueError(f"ungrounded reference: {row['id']}")
            try:
                v2.validate_program(program, row["transcript"], row["reference_context"])
                accepted = True
            except Exception as exc:
                rejections.append({"id": row["id"], "alternative": index,
                                   "code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)})
        compatible_rows += int(accepted)
    if schema_failures:
        raise ValueError(f"gold schema failures: {schema_failures}")

    # Probe the actual scoring boundary without querying a model. A misleading
    # free-form clarification is never accepted by string or keyword matching.
    clarification_case = next(row for row in rows if row["family"] == "unfocused_return" and row["language"] == "en")
    clarification = next(alt for alt in clarification_case["expected_any_of"] if alt["steps"][0]["op"] == "clarify")
    altered = json.loads(json.dumps(clarification))
    altered["steps"][0]["question"] = "Do you want to stop instead?"
    assert classify(clarification_case, altered)["result"] == "manual_review"
    assert classify(clarification_case, clarification)["result"] == "manual_review"
    unresolved = clarification_case["expected_any_of"][0]
    assert classify(clarification_case, unresolved)["result"] == "semantic_match"

    class FixedResult:
        def __init__(self, value):
            self.value = value

        def interpret(self, text, *, request_id, reference_context, timeout_seconds):
            program = {**self.value, "request_id": request_id}
            v2.validate_program(program, text, reference_context)
            return v2.TaskProgramResult(
                status="ok", request_id=request_id, program=program, raw_text=None,
                raw_json=self.value, json_valid=True, schema_valid=True, unresolved=True,
                validated_stop_prefix=[], detail=None, elapsed_ms=0.0, proposal_only=True,
                execution_authorized=False, required_resume_checks=[], required_repeat_checks=[],
                repeat_reference_bindings=[])

    # Feed fixed, already validated results to the evaluator. This avoids both
    # a model call and loading any system or example prompt.
    pending = existing_evaluator.evaluate_rows(
        [clarification_case], FixedResult(altered))[0]
    assert pending["validated_return_question_manual_review_pending"] is True
    assert pending["validated_semantic_exact"] is False
    absent = existing_evaluator.evaluate_rows(
        [clarification_case], FixedResult(clarification))[0]
    assert absent["validated_semantic_exact"] is False

    # Recheck the independent review's unsafe proposal-acceptance examples.
    # These are fixed programs evaluated in process, not model responses.
    by_id = {row["id"]: row for row in rows}
    known_false_acceptances = []

    def check_false_acceptance(identifier, step):
        row = by_id[identifier]
        proposal = {"schema": "task_program.v2", "steps": [step]}
        program = {**proposal, "request_id": identifier}
        v2.validate_program(program, row["transcript"], row["reference_context"])
        classification = classify(row, proposal)
        if classification["result"] != "semantic_mismatch":
            raise ValueError(f"reviewed false acceptance did not remain a gold mismatch: {identifier}")
        known_false_acceptances.append({"id": identifier, "checker_accepted": True,
                                        "dataset_result": classification["result"],
                                        "evidence_scope": "whole_input" if step["evidence"] == row["transcript"] else "partial_input"})

    stop_case = by_id["independent.repeat_stop_not_motion.ko"]
    check_false_acceptance(stop_case["id"], {"op": "stop_motion", "scope": "active_task",
                                           "evidence": stop_case["transcript"]})
    for language, short_evidence in (("ko", "2번에서 1번으로 원통을 옮겨"),
                                     ("en", "Move a cylinder from slot 2 to slot 1")):
        identifier = f"independent.unsupported_height.{language}"
        move = {"op": "task", "kind": "move", "source_slot": 2, "destination_slot": 1,
                "object_id": None, "count": None, "height": None, "layout": None,
                "evidence": short_evidence}
        check_false_acceptance(identifier, move)
        if language == "en":
            check_false_acceptance(identifier, {**move, "evidence": by_id[identifier]["transcript"]})

    report = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent text-only development set v2, not final test",
        "rows": len(rows), "by_language": dict(Counter(r["language"] for r in rows)),
        "families": len(family_counts), "alternatives": sum(len(r["expected_any_of"]) for r in rows),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "unique_text_context_inputs": len(input_pairs),
        "normalized_exact_duplicates_vs_prior_development112": len(duplicates),
        "existing_v2_evaluator_input_load": "passed",
        "existing_v2_semantic_projection_mismatches": 0,
        "v1_unchanged_dataset_sha256": sha(V1_DATA), "v1_unchanged_validation_sha256": sha(V1_VALIDATION),
        "review_sha256": sha(REVIEW), "changed_text_ids": changed_texts,
        "changed_reason_ids": changed_rationales, "unchanged_gold_semantic_projections": True,
        "other_rows_and_all_contexts_unchanged": True,
        "question_review_cases": sum(r["question_review"] != "not_applicable" for r in rows),
        "schema_failures": schema_failures,
        "rows_with_at_least_one_validator_accepted_gold": compatible_rows,
        "gold_alternative_validator_rejections": rejections,
        "reviewed_incorrect_proposals_still_accepted_by_checker": known_false_acceptances,
        "note": "A rejected correct gold is retained as a validator compatibility finding; it is not rewritten to match the checker.",
        "question_review_probe": "local judge marks misleading and null questions manual; existing v2 evaluator never auto-credits them; unresolved target can match structurally",
        "old_development112_sha256": sha(BASELINE), "dataset_sha256": sha(DATA),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (HERE / "build_dataset.py", HERE / "judge.py",
                           HERE / "validate_dataset.py", Path(v1.__file__), Path(v2.__file__), v2.SCHEMA_PATH)},
        "model_calls": 0, "audio_or_microphone_used": False, "device_calls": 0,
        "reserved_final_data_opened": False, "example_prompt_opened": False,
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
