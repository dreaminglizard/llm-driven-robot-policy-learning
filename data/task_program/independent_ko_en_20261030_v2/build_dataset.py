"""Make v2 from the frozen v1 rows without opening a prompt or calling a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "independent_ko_en_20261030_v1" / "development80.jsonl"
OUT = HERE / "development80.jsonl"
SOURCE_SHA256 = "3b39eca8cc66cf29eae6e2959fcfc3b54b8ac57dde71ac1b5ed8777a22676cac"
TEXT = {
    "ko": "현재 작업 하나만 완료하고 이번 회차의 남은 작업과 이후 반복은 하지 마.",
    "en": "Finish only the current task; skip the remaining tasks in this round, and do not repeat.",
}
BOUNDARY_REASON = ("진행 중인 개별 작업 하나까지만 완료하며, 같은 회차의 남은 작업과 이후 반복을 "
                   "진행하지 않습니다. 따라서 경계는 회차 완료 뒤가 아니라 현재 작업 완료 뒤입니다.")
SINGULAR_REASON = ("기억만 부정하고 이동은 긍정했습니다. 영어 'a cylinder'는 단수 대상을 가리키므로 "
                   "count 1을 허용하고, 명시적 숫자가 없는 현재 제안 형식의 count null도 허용합니다. "
                   "이는 숫자를 생략한 일반 복수 표현과 다릅니다. 두 값의 실행 단계 동등성은 별도 계약에서 확인합니다.")


def main():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("v1 dataset differs from the reviewed frozen source")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    touched = set()
    for row in rows:
        if row["family"] == "finish_current_no_new_round":
            lang = row["language"]
            row["transcript"] = TEXT[lang]
            row["grading_reason"] = BOUNDARY_REASON
            for alternative in row["expected_any_of"]:
                for step in alternative["steps"]:
                    step["evidence"] = TEXT[lang]
                    if step["op"] == "continue" and lang == "ko":
                        step["target"]["reference"] = "현재 작업"
            touched.add(row["id"])
        elif row["id"] == "independent.negated_memory_positive_move.en":
            row["grading_reason"] = SINGULAR_REASON
            touched.add(row["id"])
    expected = {"independent.finish_current_no_new_round.ko",
                "independent.finish_current_no_new_round.en",
                "independent.negated_memory_positive_move.en"}
    if len(rows) != 80 or touched != expected:
        raise ValueError("v1 inputs or required targeted changes differ")
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"created {OUT} ({len(rows)} records)")


if __name__ == "__main__":
    main()
