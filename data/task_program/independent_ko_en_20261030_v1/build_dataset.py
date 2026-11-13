"""Create a frozen text-only development set; no model, audio or device use.

The examples and gold programs are authored here before any model is queried.
The output file is created exclusively and is never overwritten by this script.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "development80.jsonl"
SCHEMA = "task_program.v2"


def task(kind, source=None, destination=None, *, count=None, object_id=None, height=None):
    return {"op": "task", "kind": kind, "source_slot": source,
            "destination_slot": destination, "object_id": object_id,
            "count": count, "height": height, "layout": None}


def stop():
    return {"op": "stop_motion", "scope": "active_task"}


def stop_repeat(boundary):
    return {"op": "stop_repeat", "boundary": boundary}


def ret(kind, reference):
    return {"op": "return", "target": {"kind": kind, "reference": reference}}


def repeat(kind, reference, mode="continuous", value=None, indices=None):
    return {"op": "repeat", "target": {"kind": kind, "reference": reference,
                                        "step_indices": indices},
            "count": {"mode": mode, "value": value}}


def cont(kind, reference):
    return {"op": "continue", "target": {"kind": kind, "reference": reference}}


def clarify(reason):
    # A free-form question, when supplied by a model, always needs human review.
    return {"op": "clarify", "reason": reason, "question": None}


def status(subject):
    return {"op": "status", "subject": subject}


IDLE = {"execution_state": "idle", "repeat_state": "inactive", "state_revision": "synthetic-idle-r1"}
RUN = {"execution_state": "running", "repeat_state": "inactive", "active_request_id": "synthetic-active-1", "state_revision": "synthetic-run-r1"}
LOOP = {**RUN, "repeat_state": "active"}
PREV_REF = {"status": "unique", "reference_id": "synthetic-prev-task-1", "revision": "synthetic-prev-r1"}
PREV_PROGRAM_REF = {"status": "unique", "reference_id": "synthetic-prev-program-1", "revision": "synthetic-program-r1"}
PREVIOUS = {"execution_state": "stopped_confirmed", "repeat_state": "inactive", "state_revision": "synthetic-stop-r1", "previous_task": PREV_REF}
BOTH = {**RUN, "previous_task": PREV_REF}
PREVIOUS_PROGRAM = {"execution_state": "idle", "repeat_state": "inactive", "state_revision": "synthetic-idle-r1", "previous_program": PREV_PROGRAM_REF}


def focused(kind):
    return {**IDLE, "return_focus": {"kind": kind, "source": "explicit_recent", "revision": "synthetic-focus-r1"}}


rows = []


def add(family, ko, en, context, ko_options, en_options, reason, *, category, contrast=None, note=None):
    for lang, text, options in (("ko", ko, ko_options), ("en", en, en_options)):
        alternatives = []
        for option in options:
            steps = copy.deepcopy(option)
            for step in steps:
                step["evidence"] = text
            alternatives.append({"schema": SCHEMA, "steps": steps})
        has_clarify = any(step["op"] == "clarify" for option in options for step in option)
        rows.append({"id": f"independent.{family}.{lang}", "language": lang,
                     "family": family, "category": category,
                     "contrast_group": contrast, "input_kind": "new_development_text",
                     "transcript": text, "reference_context": copy.deepcopy(context),
                     "context_note": note or "합성 실행 문맥이며 실제 장치 실행 이력이 아닙니다.",
                     "expected_any_of": alternatives, "grading_reason": reason,
                     "question_review": "manual_for_every_clarify_step" if has_clarify else "not_applicable",
                     # The existing evaluator's flag is historically named for return targets;
                     # it forces free-form clarifications to manual review in any case.
                     "clarify_requires_return_target_question": has_clarify})


def options(*steps):
    return [list(steps)]


# Ordered programs: the words' order is not always the execution order.
add("order_move_before_memory", "현재 배치를 기억하기 전에 2번에서 4번으로 원통 한 개를 옮겨 주세요.",
    "Before you remember the current layout, move one cylinder from slot 2 to slot 4.", IDLE,
    options(task("move", 2, 4, count=1), task("remember")),
    options(task("move", 2, 4, count=1), task("remember")),
    "기억은 이동 후의 배치를 대상으로 하므로 이동이 먼저입니다.", category="ordered")
add("order_remember_before_swap", "2번과 4번을 바꾸기 전에 지금 배치부터 기억해 주세요.",
    "Remember the current layout before you swap slot 2 with slot 4.", IDLE,
    [[task("remember"), task("swap", 2, 4)], [task("remember"), task("swap", 4, 2)]],
    [[task("remember"), task("swap", 2, 4)], [task("remember"), task("swap", 4, 2)]],
    "교환 전 배치에 쓰기 신호를 주고 이후 두 자리를 교환합니다.", category="ordered")
add("order_middle_memory", "3번에서 4번으로 옮기고 그 배치를 기억한 다음 4번에서 1번으로 옮겨.",
    "Move from slot 3 to slot 4, remember that layout, then move from slot 4 to slot 1.", IDLE,
    options(task("move", 3, 4), task("remember"), task("move", 4, 1)),
    options(task("move", 3, 4), task("remember"), task("move", 4, 1)),
    "두 이동 사이에 기억하며 처음이나 마지막 배치로 저장 시점을 바꾸지 않습니다.", category="ordered")
add("order_swap_then_memory_return", "1번과 3번을 교환한 다음 기억한 배치로 복원해 줘.",
    "Swap slot 1 with slot 3, then restore the remembered layout.", IDLE,
    [[task("swap", 1, 3), ret("learned_memory", "기억한 배치")],
     [task("swap", 3, 1), ret("learned_memory", "기억한 배치")]],
    [[task("swap", 1, 3), ret("learned_memory", "remembered layout")],
     [task("swap", 3, 1), ret("learned_memory", "remembered layout")]],
    "교환 뒤 학습 기억에서 복원하며 기억 목표를 해석기가 생성하지 않습니다.", category="ordered")
add("order_home_then_status", "팔을 대기 자세로 돌려놓은 다음 현재 배치를 확인해 줘.",
    "Return the arm to its home pose, then report the current layout.", IDLE,
    options(ret("arm_home", "팔을 대기 자세"), status("current_state")),
    options(ret("arm_home", "arm to its home pose"), status("current_state")),
    "팔 대기 자세와 원통 배치 조회는 별도 단계이고 그 순서를 보존합니다.", category="ordered")
add("ordered_round_repeat", "2번에서 3번으로 옮기고 3번에서 2번으로 옮겨. 이 두 동작을 계속 반복해.",
    "Move from slot 2 to slot 3 and then from slot 3 to slot 2. Keep repeating those two moves.", IDLE,
    options(task("move", 2, 3), task("move", 3, 2), repeat("prior_steps", "두 동작", indices=[0, 1])),
    options(task("move", 2, 3), task("move", 3, 2), repeat("prior_steps", "those two moves", indices=[0, 1])),
    "명시된 두 이동을 순서대로 반복 대상으로 묶고 다른 이동은 추가하지 않습니다.", category="ordered")

# Four explicit, mutually distinct return targets.
add("explicit_registered", "원통 배치를 등록한 초기 상태로 복원해 줘.",
    "Restore the cylinders to the registered initial layout.", IDLE,
    options(ret("registered_initial", "등록한 초기 상태")), options(ret("registered_initial", "registered initial layout")),
    "명시된 기준은 등록 당시의 초기 배치입니다.", category="return")
add("explicit_task_start", "이번 작업을 시작하기 전 상태로 배치를 되돌려.",
    "Return the layout to the state before this task started.", IDLE,
    options(ret("task_start", "작업을 시작하기 전 상태")), options(ret("task_start", "task started")),
    "등록 시점이나 저장한 기억이 아니라 이번 작업의 시작 전 상태를 가리킵니다.", category="return")
add("explicit_learned", "전에 기억한 배치로 되돌려 줘.", "Return to the layout I memorized earlier.", IDLE,
    options(ret("learned_memory", "기억한 배치")), options(ret("learned_memory", "layout I memorized earlier")),
    "기억한 배치를 요구하므로 목표는 학습 기억으로 회상해야 합니다.", category="return")

# The same utterance has five different inputs, distinguished by trusted context.
for kind in ("arm_home", "registered_initial", "task_start", "learned_memory"):
    add("focused_" + kind, "그 위치로 다시 돌아가.", "Return to that position.", focused(kind),
        options(ret(kind, "그 위치")), options(ret(kind, "that position")),
        "발화만으로는 기준이 모호하지만 최근에 명시된 유일한 복귀 초점이 이 문맥에 있습니다.",
        category="return_context", contrast="focused_return_same_utterance",
        note="합성 문맥의 return_focus만 최근 명시된 유일한 기준으로 사용합니다.")
ambiguous_ko = [ret("unresolved", "그 위치")]
ambiguous_en = [ret("unresolved", "that position")]
add("unfocused_return", "그 위치로 다시 돌아가.", "Return to that position.", IDLE,
    [ambiguous_ko, [clarify("ambiguous_reference")], [clarify("missing_target")]],
    [ambiguous_en, [clarify("ambiguous_reference")], [clarify("missing_target")]],
    "동일한 표현이지만 기준 문맥이 없으므로 미해결로 남기거나 기준을 질문합니다.",
    category="return_context", contrast="focused_return_same_utterance")
add("stop_then_unknown_return", "움직임을 정지한 다음 그 위치로 돌아가.",
    "Stop the motion, then return to that position.", LOOP,
    [[stop(), ret("unresolved", "그 위치")], [stop(), clarify("ambiguous_reference")], [stop(), clarify("missing_target")]],
    [[stop(), ret("unresolved", "that position")], [stop(), clarify("ambiguous_reference")], [stop(), clarify("missing_target")]],
    "명확한 즉시 정지는 보존하고 불명확한 복귀만 미해결로 둡니다.", category="stop_return")

# Independent stop and repeat boundaries, with optional continuation when it is explicit.
add("finish_current_no_new_round", "지금 작업은 마치되 다음 반복은 하지 마.",
    "Finish the current task, but do not start another repetition.", LOOP,
    [[stop_repeat("after_task")], [stop_repeat("after_task"), cont("active_task", "지금 작업")]],
    [[stop_repeat("after_task")], [stop_repeat("after_task"), cont("active_task", "current task")]],
    "현재 작업 완료 뒤 반복을 그치며 별도 계속 단계는 명시해도 됩니다.", category="repeat_boundary")
add("finish_round_no_more", "이번 회차까지만 진행하고 반복을 중단해.",
    "Finish this round and stop repeating after the round.", LOOP,
    options(stop_repeat("after_round")), options(stop_repeat("after_round")),
    "현재 회차의 끝이 경계이며 즉시 동작 정지는 요구하지 않았습니다.", category="repeat_boundary")
add("repeat_then_motion_stop", "반복을 중단하고 로봇 동작도 지금 멈춰.",
    "Stop repeating, then stop the robot motion now.", LOOP,
    options(stop_repeat("after_round"), stop()), options(stop_repeat("after_round"), stop()),
    "반복 종료와 움직임 정지는 서로 다른 두 요청이며 둘 다 보존합니다.", category="repeat_boundary")
add("repeat_stop_not_motion", "현재 작업은 정지하지 말고 반복만 중단해.",
    "Do not stop the current task; stop repeating only.", LOOP,
    [[stop_repeat("after_round")], [stop_repeat("after_round"), cont("active_task", "현재 작업")]],
    [[stop_repeat("after_round")], [stop_repeat("after_round"), cont("active_task", "current task")]],
    "운동 정지는 부정됐고 반복만 종료하며 진행 중인 작업을 새로 시작하지 않습니다.", category="repeat_boundary")

# Previous/current targets and finite, additional, or continuous repetitions.
add("previous_program_twice", "이전 프로그램을 두 번 반복해.",
    "Repeat the previous program two times.", PREVIOUS_PROGRAM,
    options(repeat("previous_program", "이전 프로그램", "total", 2)),
    options(repeat("previous_program", "previous program", "total", 2)),
    "유일한 이전 프로그램을 새 반복 요청으로 두 차례 실행하는 제안입니다.", category="repeat")
add("active_continuous_repeat", "현재 작업을 무한히 되풀이해.",
    "Repeat the current task continuously.", RUN,
    options(repeat("active_task", "현재 작업")), options(repeat("active_task", "current task")),
    "실행 중인 작업을 반복 대상으로 하고 종료 횟수를 임의로 만들지 않습니다.", category="repeat")
add("active_repeat_add_two", "현재 작업을 두 번 더 반복해.", "Repeat the current task two more times.", LOOP,
    options(repeat("active_task", "현재 작업", "additional", 2)),
    options(repeat("active_task", "current task", "additional", 2)),
    "기존 반복 계획에 두 번을 추가하는 제안이며 기존 계획 확인이 필요합니다.", category="repeat")
add("previous_task_once", "끝낸 이전 작업을 한 번 반복해.", "Please repeat the previous task once.", PREVIOUS,
    options(repeat("previous_task", "이전 작업", "total", 1)),
    options(repeat("previous_task", "previous task", "total", 1)),
    "이전 작업을 한 번 새로 반복하는 뜻이지 중단 지점에서 이어 하라는 뜻은 아닙니다.", category="repeat",
    note="합성 직전 작업은 유일하게 식별됩니다. 중단 여부는 참조 계약에 포함되지 않습니다.")
add("repeat_previous_not_current", "현재 작업은 반복하지 말고 이전 작업을 네 번 반복해.",
    "Do not repeat the current task; repeat the previous task four times.", BOTH,
    options(repeat("previous_task", "이전 작업", "total", 4)),
    options(repeat("previous_task", "previous task", "total", 4)),
    "현재 작업의 반복은 부정됐고 유일한 이전 작업의 반복만 요청했습니다.", category="repeat_reference", contrast="repeat_reference_polarity")
add("repeat_current_not_previous", "이전 작업은 반복하지 말고 현재 작업을 한 번 반복해.",
    "Do not repeat the previous task; repeat the current task once.", BOTH,
    options(repeat("active_task", "현재 작업", "total", 1)),
    options(repeat("active_task", "current task", "total", 1)),
    "이전 작업은 제외하고 실행 중인 현재 작업을 반복합니다.", category="repeat_reference", contrast="repeat_reference_polarity")
add("repeat_ambiguous_task", "그 작업을 계속 되풀이해.", "Keep repeating that task.", BOTH,
    [[repeat("unresolved", "그 작업")], [clarify("ambiguous_reference")]],
    [[repeat("unresolved", "that task")], [clarify("ambiguous_reference")]],
    "현재 작업과 직전 작업이 모두 있으나 지시어만으로는 어느 쪽인지 알 수 없습니다.", category="repeat_reference")

# Continue and interruption are not restart or repetition.
add("resume_interrupted_previous", "중단한 이전 작업부터 다시 이어서 해.",
    "Resume the previously interrupted task from where it stopped.", PREVIOUS,
    options(cont("previous_task", "이전 작업")), options(cont("previous_task", "previously interrupted task")),
    "유일한 직전 작업의 이어 하기는 정지 확인, 중단 이력 확인, 현재 상태 확인 및 재계획이 선행돼야 합니다.", category="continuation")
add("continue_active", "현재 작업은 멈추지 않고 계속 진행해.",
    "Keep the current task going without stopping it.", RUN,
    options(cont("active_task", "현재 작업")), options(cont("active_task", "current task")),
    "이미 실행 중인 작업을 이어 가며 새 작업이나 정지 요청을 만들지 않습니다.", category="continuation")
add("continue_ambiguous_task", "그 작업을 계속 이어서 해.", "Continue that task for now.", BOTH,
    [[cont("unresolved", "그 작업")], [clarify("ambiguous_reference")]],
    [[cont("unresolved", "that task")], [clarify("ambiguous_reference")]],
    "현재와 직전 작업이 동시에 문맥에 있으므로 지시어를 임의로 해석하지 않습니다.", category="continuation")

# Negation, mentions, questions and explicit corrections.
add("negated_transfer_only", "원통을 1번에서 4번으로 옮기지 마.",
    "Do not move a cylinder from slot 1 to slot 4.", IDLE,
    options(clarify("negated_only")), options(clarify("negated_only")),
    "이동을 금지했을 뿐 실행할 새 이동은 요청하지 않았습니다.", category="negation")
add("quoted_stop_question", "메뉴에 적힌 ‘멈춰’라는 말은 무슨 뜻이야?",
    "What does the word 'stop' in the menu mean?", RUN,
    options(clarify("question_only")), options(clarify("question_only")),
    "인용된 정지 표현의 의미를 묻는 질문이며 정지 명령이 아닙니다.", category="question")
add("stop_advice_question", "팔이 움직이는 동안 정지해야 할까?",
    "Should the arm be stopped while it is moving?", RUN,
    options(clarify("question_only")), options(clarify("question_only")),
    "정지에 관한 판단을 묻고 있어 정지 명령으로 바꾸지 않습니다.", category="question")
add("correct_destination", "1번에서 2번이 아니라 4번으로 옮겨.",
    "Move from slot 1 to slot 4, not slot 2.", IDLE,
    options(task("move", 1, 4)), options(task("move", 1, 4)),
    "도착 자리는 정정된 4번이며 부정된 2번이 아닙니다.", category="correction")
add("correct_source", "3번이 아니라 4번에서 1번으로 옮겨.",
    "From slot 4, not slot 3, move to slot 1.", IDLE,
    options(task("move", 4, 1)), options(task("move", 4, 1)),
    "출발 자리는 정정된 4번이고 도착 자리는 1번입니다.", category="correction")
add("correct_return_target", "기억한 배치는 말고 등록한 초기 배치로 복원해.",
    "Do not use the remembered layout; restore the registered initial layout.", IDLE,
    options(ret("registered_initial", "등록한 초기 배치")),
    options(ret("registered_initial", "registered initial layout")),
    "기억 배치는 배제되고 등록한 초기 배치가 명시적으로 선택됐습니다.", category="correction")
add("negated_memory_positive_move", "기억은 하지 말고 3번에서 2번으로 원통을 옮겨.",
    "Do not remember the layout; move a cylinder from slot 3 to slot 2.", IDLE,
    options(task("move", 3, 2)), [[task("move", 3, 2)], [task("move", 3, 2, count=1)]],
    "기억만 부정했으므로 명확하게 긍정된 이동은 보존합니다.", category="negation")
add("missing_stack_source", "3번에 원통 쌓아 줘.", "Stack a cylinder at slot 3.", IDLE,
    [[clarify("missing_target")], [clarify("ambiguous_reference")]],
    [[clarify("missing_target")], [clarify("ambiguous_reference")]],
    "목적지는 있으나 출발 자리나 원통 ID가 없어 이동을 만들어서는 안 됩니다.", category="ambiguity")
add("unsupported_slot", "5번 자리에서 1번 자리로 원통을 옮겨 줘.",
    "Move a cylinder from slot 5 to slot 1.", IDLE,
    options(clarify("unsupported")), options(clarify("unsupported")),
    "5번 자리는 명시된 네 자리 범위 밖이므로 다른 번호로 고쳐 실행하지 않습니다.", category="out_of_scope")
add("unsupported_height", "2번에서 1번으로 원통을 옮겨 세 층으로 쌓아 줘.",
    "Move a cylinder from slot 2 to slot 1 to make a three-level stack.", IDLE,
    options(clarify("unsupported")), options(clarify("unsupported")),
    "요청한 세 층은 현재 명령 계약의 최대 두 층을 넘으며 임의로 낮추지 않습니다.", category="out_of_scope")
add("unsupported_object_id", "5번 원통을 2번 자리로 옮겨 줘.",
    "Move cylinder 5 to slot 2.", IDLE,
    options(clarify("unsupported")), options(clarify("unsupported")),
    "원통 ID 5는 계약의 1~4 범위 밖이므로 다른 원통으로 바꾸지 않습니다.", category="out_of_scope")


def main():
    if len(rows) != 80 or sum(row["language"] == "ko" for row in rows) != 40:
        raise ValueError("expected exactly 40 records per language")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate IDs")
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"created {OUT} ({len(rows)} records)")


if __name__ == "__main__":
    main()
