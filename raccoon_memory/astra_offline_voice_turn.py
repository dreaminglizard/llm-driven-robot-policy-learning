"""Offline research router. It proposes data; it owns no robot or audio device."""
from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import unicodedata

from .task_program_v2 import (explicit_object_stack_step, explicit_slot_stack_roles, explicit_top_unstack_roles,
                              is_whisper_four_slot_memory_request, validate_program)
from .task_program import ProgramError, is_korean_request_question_punctuation


NUMBER = {"1":1,"2":2,"3":3,"4":4,"one":1,"two":2,"three":3,"four":4,
          "일":1,"이":2,"삼":3,"사":4,"한":1,"두":2,"세":3,"네":4}
EN_MOVE = re.compile(r"move (?:one |a )cylinder from slot ([1-4]|one|two|three|four) to slot ([1-4]|one|two|three|four)")
KO_ROUTE = r"([1-4일이삼사한두세네])\s*번\s*(?:자리)?에서\s*([1-4일이삼사한두세네])\s*번\s*(?:자리)?(?:으)?로\s*"
KO_MOVE = re.compile(KO_ROUTE+r"원통\s*하나를\s*(?:옮겨|옮겨\s*줘|이동해\s*줘)")
KO_FORBID_MOVE = re.compile(KO_ROUTE+r"원통(?:을|\s*하나를)?\s*옮기지\s*(?:마|마세요|말아\s*줘)")
EN_FORBID_MOVE = re.compile(r"(?:do not|don't) move (?:one |a |the )cylinder from slot (?:[1-4]|one|two|three|four) to slot (?:[1-4]|one|two|three|four)")
MOTION_STOP = {"stop the motion now","stop the robot","stop the robot now","로봇 멈춰","로봇을 멈춰","지금 바로 동작을 멈춰","동작 정지"}
AMBIGUOUS_STOP = {"stop","stop now","please stop","멈춰","멈춰요","그만","정지"}
SPEECH_STOP = {"stop talking","stop speaking","please stop talking","말 그만해","말을 멈춰","말만 멈춰"}
MEMORY = {"remember the current arrangement","remember this arrangement","현재 배치를 기억해","지금 배치를 기억해","지금 배치 기억해"}
ROBOT_TERM = re.compile(r"\b(?:robot|motion|cylinder|slot|stack|unstack|move|stop(?:ping)?|repeat(?:ing)?|remember|restore|return)\b|로봇|원통|배치|옮|이동|멈|정지|중단|쌓|반복|복원|자리|기억",re.I)


def normalized(text: str) -> str:
    clean = " ".join(unicodedata.normalize("NFKC",text).casefold().split()).strip()
    if is_korean_request_question_punctuation(text):
        clean = clean[:-1].rstrip()
    return clean.rstrip(".!。！").strip()


def bounded_proposal(text: str, request_id: str):
    """Propose a program from one complete command; admission owns semantic validation."""
    canonical=normalized(text)
    if canonical in MEMORY or is_whisper_four_slot_memory_request(text):
        step={"op":"task","kind":"remember","source_slot":None,"destination_slot":None,"object_id":None,"count":None,"height":None,"layout":None,"evidence":text}
    elif (named := explicit_object_stack_step(text)) is not None:
        step = named
    else:
        stack = explicit_slot_stack_roles(text)
        unstack = explicit_top_unstack_roles(text)
        if stack is not None:
            source, destination, count = stack
            kind = 'stack'
        elif unstack is not None and unstack[2] is None:
            source, destination, count = unstack
            kind = 'unstack'
        else:
            match=EN_MOVE.fullmatch(canonical) or KO_MOVE.fullmatch(canonical)
            if not match or NUMBER[match[1]]==NUMBER[match[2]]:
                return None
            source, destination = NUMBER[match[1]], NUMBER[match[2]]
            kind, count = 'move', 1
        step={"op":"task","kind":kind,"source_slot":source,"destination_slot":destination,
              "object_id":None,"count":count,"height":None,"layout":None,"evidence":text}
    return {"schema":"task_program.v2","request_id":request_id,"steps":[step]}


def bounded_program(text: str, request_id: str, context: dict):
    """Return a validated rule proposal for independent callers."""
    proposal = bounded_proposal(text, request_id)
    if proposal is not None:
        validate_program(proposal, text, reference_context=context)
    return proposal


@dataclass(frozen=True)
class Turn:
    serial: int
    request_id: str
    active_request_at_onset: str | None
    playback_at_onset: bool


@dataclass(frozen=True)
class Decision:
    kind: str
    text: str=""
    active_request: str | None=None
    program: dict | None=None
    message: str | None=None


class OfflineTurnRouter:
    """One active utterance. A superseded or duplicate ASR final cannot commit."""
    def __init__(self):
        self._lock=threading.Lock()
        self._serial=0
        self._current=None

    def begin(self, *, active_request: str | None, playback_active: bool, near_end_voice: bool):
        if not near_end_voice:
            return None,Decision("no_turn")
        with self._lock:
            self._serial+=1
            turn=Turn(self._serial,f"offline-voice-{self._serial}",active_request,playback_active)
            self._current=turn
        return turn,Decision("duck_voice_output" if playback_active else "listen")

    def partial(self,turn,text):
        with self._lock:
            if turn is None or turn!=self._current:
                return Decision("discard_stale")
            return Decision("display_partial",text)

    def cancel(self,turn):
        with self._lock:
            if turn is not None and turn==self._current:
                self._current=None

    def final(self,turn,text,*,current_active: str | None,context: dict, proposal_only=False):
        with self._lock:
            if turn is None or turn!=self._current:
                return Decision("discard_stale")
            self._current=None
        clean=normalized(text)
        if not clean:
            return Decision("clarify",text)
        if clean in SPEECH_STOP:
            return Decision("stop_voice_output",text)
        if clean in MOTION_STOP or clean in AMBIGUOUS_STOP:
            if clean in AMBIGUOUS_STOP and turn.playback_at_onset:
                return Decision("clarify_stop_target",text)
            if turn.active_request_at_onset is None or turn.active_request_at_onset!=current_active:
                return Decision("no_matching_active_request",text)
            # Caller must revalidate the same ID atomically in the established coordinator.
            return Decision("bound_stop_proposal",text,turn.active_request_at_onset)
        if (("?" in text or "？" in text) and not is_korean_request_question_punctuation(text)
                or any(q in text for q in ('"',"“","”","‘","’","'멈춰'"))):
            return Decision("read_only_conversation",text)
        if turn.active_request_at_onset!=current_active:
            return Decision("discard_context_change",text)
        if KO_FORBID_MOVE.fullmatch(clean) or EN_FORBID_MOVE.fullmatch(clean):
            return Decision("acknowledge_prohibition",text)
        # Clarifications only. An unsupported verb or personal memory scope
        # cannot be corrected into some other action from an ASR hypothesis.
        compound_or_negated=re.search(r"[.!?;:,]|다음|먼저|후에|(?:고|면|거나)\s|멈|정지|중단|반복|취소|아니|지\s*(?:마|말|않)|(?:안|못)\s",clean)
        if not compound_or_negated and re.search(r"원통|빈자리|탑",clean) and re.search(r"내려치|내려쳐|내리쳐|후려치",clean):
            return Decision("clarify_action_verb",text,message="내려치라는 말로 들었습니다. 원통을 내려놓는 뜻이라면 출발 자리와 목적 자리를 다시 말해 주세요.")
        if (not compound_or_negated and "기억" in clean and re.search(r"(?:내|제|저의|나의)\s*자리",clean)
                and not is_whisper_four_slot_memory_request(text)):
            return Decision("clarify_memory_scope",text,message="개인 자리로 들었습니다. 네 자리 전체의 로봇 배치를 기억하는 뜻인지 다시 말해 주세요.")
        if proposal_only:
            program = bounded_proposal(text, turn.request_id)
        else:
            try:
                program = bounded_program(text, turn.request_id, context)
            except ProgramError:
                return Decision("clarify",text)
        if program is not None:
            return Decision("local_program_proposal" if proposal_only else "validated_program_proposal",
                            text, program=program)
        if ROBOT_TERM.search(clean):
            return Decision("full_command_interpretation",text)
        return Decision("read_only_conversation",text)
