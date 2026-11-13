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
REPORT = HERE / "validation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def main():
    rows = read_jsonl(DATA)
    baseline = read_jsonl(BASELINE)  # Only the existing 112-case development file.
    if len(rows) != 80 or Counter(row["language"] for row in rows) != {"ko": 40, "en": 40}:
        raise ValueError("the fixed development set must contain 40 examples per language")
    if len({row["id"] for row in rows}) != 80:
        raise ValueError("case IDs are not unique")
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

    class FixedProposal:
        def __init__(self, value):
            self.value = value

        def complete(self, messages, *, timeout_seconds):
            return json.dumps(self.value, ensure_ascii=False)

    # This fixed in-process stub is not a language-model call. The existing
    # evaluator must also withhold automatic credit for a misleading question.
    pending = existing_evaluator.evaluate_rows(
        [clarification_case], v2.TaskProgramInterpreter(FixedProposal(altered)))[0]
    assert pending["validated_return_question_manual_review_pending"] is True
    assert pending["validated_semantic_exact"] is False
    absent = existing_evaluator.evaluate_rows(
        [clarification_case], v2.TaskProgramInterpreter(FixedProposal(clarification)))[0]
    assert absent["validated_semantic_exact"] is False

    report = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new text-only development set, not final test",
        "rows": len(rows), "by_language": dict(Counter(r["language"] for r in rows)),
        "families": len(family_counts), "alternatives": sum(len(r["expected_any_of"]) for r in rows),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "unique_text_context_inputs": len(input_pairs),
        "normalized_exact_duplicates_vs_prior_development112": len(duplicates),
        "existing_v2_evaluator_input_load": "passed",
        "existing_v2_semantic_projection_mismatches": 0,
        "question_review_cases": sum(r["question_review"] != "not_applicable" for r in rows),
        "schema_failures": schema_failures,
        "rows_with_at_least_one_validator_accepted_gold": compatible_rows,
        "gold_alternative_validator_rejections": rejections,
        "note": "A rejected correct gold is retained as a validator compatibility finding; it is not rewritten to match the checker.",
        "question_review_probe": "local judge marks misleading and null questions manual; existing v2 evaluator never auto-credits them; unresolved target can match structurally",
        "old_development112_sha256": sha(BASELINE), "dataset_sha256": sha(DATA),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (HERE / "build_dataset.py", HERE / "judge.py",
                           HERE / "validate_dataset.py", Path(v1.__file__), Path(v2.__file__), v2.SCHEMA_PATH, v2.PROMPT_PATH)},
        "model_calls": 0, "audio_or_microphone_used": False, "device_calls": 0,
        "reserved_final_data_opened": False,
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
