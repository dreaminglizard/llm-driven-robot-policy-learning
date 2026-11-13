"""Classify saved proposals only; never call a model or assign question quality by keywords."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from raccoon_memory import task_program_v2 as v2  # noqa: E402


def projection(program):
    """Same ordered, evidence-free meaning compared by the existing v2 evaluator."""
    if not isinstance(program, dict) or program.get("schema") != "task_program.v2" or not isinstance(program.get("steps"), list):
        return None
    out = []
    try:
        for step in program["steps"]:
            op = step["op"]
            if op == "task":
                value = {key: step[key] for key in ("op", "kind", "source_slot", "destination_slot", "object_id", "count", "height")}
                layout = step["layout"]
                value["layout"] = None if layout is None else sorted(layout, key=lambda x: x["slot"])
            elif op == "repeat":
                value = {"op": op, "target_kind": step["target"]["kind"],
                         "step_indices": step["target"]["step_indices"], "count": step["count"]}
            elif op == "stop_motion":
                value = {"op": op, "scope": step["scope"]}
            elif op == "stop_repeat":
                value = {"op": op, "boundary": step["boundary"]}
            elif op in ("return", "continue"):
                value = {"op": op, "target_kind": step["target"]["kind"]}
            elif op == "clarify":
                value = {"op": op, "reason": step["reason"]}
            elif op == "status":
                value = {"op": op, "subject": step["subject"]}
            else:
                return None
            out.append(value)
    except (KeyError, TypeError):
        return None
    return out


def classify(row, proposal):
    """Return a structural result, or a mandatory manual-review state for clarification.

    This does not execute a task and does not judge whether a free-form response
    answers the user's question. Caller must preserve raw and accepted results.
    """
    if not isinstance(proposal, dict):
        return {"result": "invalid_proposal"}
    raw_match = projection(proposal) in [projection(item) for item in row["expected_any_of"]]
    if proposal.get("request_id", row["id"]) != row["id"]:
        return {"result": "invalid_request_id", "raw_semantic_match": raw_match}
    candidate = {**proposal, "request_id": row["id"]}
    try:
        v2.validate_program(candidate, row["transcript"], row["reference_context"])
    except Exception as exc:
        return {"result": "validator_rejected", "raw_semantic_match": raw_match,
                "code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)}
    if not raw_match:
        return {"result": "semantic_mismatch", "raw_semantic_match": False}
    if any(step.get("op") == "clarify" for step in candidate["steps"]):
        return {"result": "manual_review", "raw_semantic_match": True,
                "reason": "A person must assess the clarification or the absence of a question."}
    return {"result": "semantic_match", "raw_semantic_match": True}


def main():
    raise SystemExit("Import classify(row, saved_proposal); this module does not run models.")


if __name__ == "__main__":
    main()
