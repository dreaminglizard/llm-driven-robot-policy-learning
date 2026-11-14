"""Local demonstration service with explicitly selected controller modes.

The default is virtual; selecting a mode does not connect a controller.
Microphone capture starts only on an explicit recording request.  A student's
DONE updates an expected scene, not an observed physical outcome.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit
import uuid

from raccoon_control.transport import LOOPBACK_HOST

from .concurrency import CombinedEvent as _EitherCancellation
from .voice_request_clarification import recovery_question
from .voice_settings import voice_model_path

ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK_HOST = LOOPBACK_HOST
_LOCAL_MODEL_ALIAS = 'demo-local-e4b'
_LOCAL_LLM_DEVICE = 'CUDA0'
_LOCAL_LLM_CTX_SIZE = 4096
_LOCAL_LLM_THREADS = 4
_LOCAL_LLM_CACHE_RAM_MB = 256
_LOCAL_LLM_FIT_TARGET_MB = 1536
_MAX_LOCAL_LLM_RESPONSE_BYTES = 1024 * 1024
_MAX_STUDENT_CHECKPOINT_MB = 32
_MAX_STUDENT_CHECKPOINT_BYTES = _MAX_STUDENT_CHECKPOINT_MB * _MAX_LOCAL_LLM_RESPONSE_BYTES
DEFAULT_STUDENT = ROOT / 'artifacts/architecture_research/astra_native_geometry_source_approach_v1/mixed/candidate.pt'
RL_STUDENT = ROOT / 'artifacts/architecture_research/native_approach_adaptive_rl_v1/train/geometry_mixed/checkpoint-1000/bc_plus_rl.pt'
RL_SHA256 = 'e5b82442e4218d55ca6ad4368157deff5bcda3aa19732df7be849679adff3589'
CLOSE_STUDENT = ROOT / 'artifacts/architecture_research/astra_native_close_evidence_20261106_v1/distance_and_action/candidate.pt'
CLOSE_SHA256 = '071c6d1bbdf6cd0128ad7586c82f91e9bbaf8163c18ce3a2829d91db05f3ba7c'
BEST_CLOSE_STUDENT = ROOT / 'artifacts/architecture_research/astra_close_reproduction_three_seeds_20261106_v1/seed_20261030/candidate.pt'
BEST_CLOSE_SHA256 = '48170fc8aa8e05b231115367fb2960f9a8f69920c359cf3019472d5c911d3977'
# These catalogue test groups were already used by this selected student's
# training and public A/B development. This is an app-only allowance, not a
# change to the common loader, evaluation split, or a claim of physical success.
_BEST_CLOSE_DEMO_GROUPS = frozenset({'n4_s1_d2_other11', 'n4_s2_d1_other11'})
LLAMA_SERVER = ROOT / '.tools/llama.cpp-b11960-cuda-13.4-x64/llama-server.exe'
LLM = ROOT / '.models/gemma-4-e4b-qat-q4_0/gemma-4-E4B_q4_0-it.gguf'
LLM_PROJECTOR = LLM.with_name('gemma-4-E4B-it-mmproj.gguf')
VOICE = voice_model_path()
_MODEL_SLOT_NAME = 'RaccoonDemoGemma4E4B_v1'
SLOTS = ((152.149, 150.349), (149.651, 2.149), (-152.149, 149.651), (-149.651, -2.149))
# roboid 1.7.11 Raccoon.init_pose uses these arm angles. Its implicit
# gripper opening remains a separate, explicitly prepared operation here.
SDK_INITIAL_JOINTS_DEG = (0., -10., -140., 60.)


def _initial_pose_targets(observation, config):
    """Lift a low gripper before turning toward the SDK home pose.

    These are nominal arm targets, not a measurement of objects or contact.
    Other poses keep the ordinary direct SDK home command.
    """
    from raccoon_control.errors import ControlError
    from raccoon_control.kinematics import Geometry

    home = list(SDK_INITIAL_JOINTS_DEG)
    joints, tcp = observation.get('joints_deg'), observation.get('tcp_mm')
    if (observation.get('gripper_reported_state') not in ('open', 'closed')
            or observation.get('gripper_reported_valid') is not True
            or not isinstance(joints, (list, tuple)) or len(joints) != 4
            or not isinstance(tcp, (list, tuple)) or len(tcp) != 3):
        raise RuntimeError('초기 자세 이동에 필요한 팔 또는 집게 상태를 확인하지 못했습니다.')
    try:
        if not all(math.isfinite(float(value)) for value in (*joints, *tcp)):
            raise ValueError('non-finite arm pose')
        if (tcp[2] >= 103.
                or (tcp[2] >= 65. and max(abs(a - b) for a, b in zip(joints, home)) < 15.)):
            return [home]
        geometry = Geometry(tool_forward_mm=config.tool_forward_mm,
                            tool_vertical_mm=config.tool_vertical_mm)
        pitch = sum(joints[1:]) + 90.
        targets = [list(geometry.inverse((tcp[0], tcp[1], height), pitch_deg=pitch))
                   for height in (16., 23., 38., 63., 103.) if height > tcp[2] + .5]
        elevated = list(targets[-1] if targets else joints)
        elevated[0] = 0.
        targets.append(elevated)
        targets.extend([[a + (b - a) * fraction for a, b in zip(elevated, home)]
                        for fraction in (1. / 3., 2. / 3.)])
        targets.append(home)
        return targets
    except (ControlError, TypeError, ValueError, OverflowError) as exc:
        # Do not turn a failed low-pose lift plan into a direct home move.
        raise RuntimeError('초기 자세 상승 경로를 계산하지 못했습니다.') from exc


FINGER_OPEN_PROFILES = {
    'instant': '즉시 열기',
    'minimum_jerk_0p5s': '부드럽게 열기 · 0.5초',
}
CLOSED_MOVE_PROFILES = {
    'minimum_jerk': '기존 곡선',
    'smoothstep': '부드러운 3차 곡선',
}
CONTROL_TIMING_PROFILES = {
    'demo_1s': '시연 시간 · 1초',
    'app_2s': '기존 앱 시간 · 2초',
}


def _controller_runtime_capabilities(controller):
    """Detect the current resident's API without treating it as physical validation."""
    probe = getattr(controller, 'motion_interruption_capability', None)
    if (not callable(probe) or not callable(getattr(controller, 'interrupt_motion', None))
            or not callable(getattr(controller, 'resume_interrupted_motion', None))):
        return {}
    try:
        result = probe()
    except Exception:
        return {}
    instance_id = getattr(controller, 'instance_id', None)
    valid = (isinstance(result, dict) and result.get('supported') is True
            and result.get('nominal_correction_supported') is True
            and isinstance(instance_id, str) and bool(instance_id)
            and result.get('instance_id') == instance_id
            and result.get('backend') == getattr(getattr(controller, 'config', None), 'backend', None))
    return result if valid else {}


def _controller_supports_retarget(controller):
    return bool(_controller_runtime_capabilities(controller))


_UNCONFIRMED_DIALOGUE = re.compile(
    r'로봇|원통|물체|배치|집게|가상\s*작업|움직|옮|이동|완료|성공|끝났|끝냈|마쳤|'
    r'다\s*(?:했|됐|되었습니다)|처리했|수행했|집었|잡았|놓았|내려놨|'
    r'(?:기억|복원|실행|검사|확인)(?:했|하였|했습니다|됐)|'
    r'\b(?:robot|cylinder|object|gripper|manipulator|moved|transferred|executed|completed|restored|'
    r'verified|picked|placed|grasped|success|successful|finished|done)\b', re.I)
_HIDDEN_DIALOGUE = '동작이나 상태에 관한 답변은 확인할 수 없어 표시하지 않았습니다.'
_MEANING_QUESTION = re.compile(
    r'(?:(?:(?:친구가|누군가가?|누가)\s*)?QUOTE\s*(?:이?라고\s*(?:했는데|말했는데|들었는데)\s*)?'
    r'(?:그\s*말(?:은|이)?\s*)?무슨\s*뜻(?:인가요|일까요|이에요|이야)|'
    r'someone\s+said\s+QUOTE\s*\.?\s*what\s+(?:did\s+that\s+phrase\s+mean|does\s+that\s+mean)|'
    r'what\s+does\s+(?:the\s+(?:word|phrase)\s+)?QUOTE\s+mean)\s*[?？]', re.I)
_DIALOGUE_SLOT_NUMBER = r'(?:[1-4]|one|two|three|four)'
_QUOTED_ENGLISH_MOVE = re.compile(
    rf'move (?:the|a|one) cylinder from slot ({_DIALOGUE_SLOT_NUMBER}) to slot ({_DIALOGUE_SLOT_NUMBER})')
_REPORTS_A_REQUEST = re.compile(
    r'(?:[^.!?;。？！\n]{0,260}(?:멈추|중단하|옮기|이동하)라는\s*(?:뜻|지시|명령)(?:입니다|이에요|예요|일\s*거예요|였을\s*거예요)|'
    r'(?:it|that) means (?:someone|the speaker|they) (?:asked|wants?|is asking) '
    r'(?:(?:for )?the (?:robot|motion|movement) to (?:stop|pause)|(?:you )?to (?:stop|pause)(?: the (?:robot|motion))?|'
    rf'(?:you )?to move (?:the|a|one) cylinder from slot {_DIALOGUE_SLOT_NUMBER} to slot {_DIALOGUE_SLOT_NUMBER})|'
    r'it sounds like a command(?: in a (?:specific )?(?:system or machine|system|machine))?, (?:likely )?referring to '
    r'(?:physically(?: or digitally)? )?relocating (?:a cylinder|a component labeled "cylinder") '
    rf'from a designated position \("slot {_DIALOGUE_SLOT_NUMBER}"\) to another \("slot {_DIALOGUE_SLOT_NUMBER}"\))[.]?', re.I)
_ADDITIONAL_DIALOGUE_CLAIM = re.compile(
    r'이미|완료|했습|했어|됐어|되었|멈췄|옮겼|중입|중이|움직이고\s*있|멈춰|움직여|옮겨\s*(?:줘|라)|'
    r'(?:실행|이동|정지)해\s*(?:줘|주세요)|그리고|그\s*다음|'
    r'\b(?:already|currently|completed|executed|stopped|moving|stationary|I|we|and\s+now)\b', re.I)

_ROBOT_ACTION_REFERENCE = re.compile(
    r'로봇|로보트|그리퍼|집게|원통|동작|작업|(?<![가-힣])팔|'
    r'\b(?:robot|robotic|gripper|jaws?|arm|manipulator|cylinder|operations?|motion)\b', re.I)
_KO_ROBOT_REQUEST = re.compile(
    r'(?:열어|닫아|들어|올려|내려|옮겨|움직여|멈춰|정지해|중단해|돌려|회전해|잡아|집어|놓아|놔|벌려)'
    r'\s*(?:줘|주세요|주시겠|주실|줄래|줄\s*수|주시|주셔)(?!라고|라는|라며|라더)')
_EN_ROBOT_REQUEST = re.compile(
    r'\b(?:can|could|would|will)\s+you\s+(?:please\s+)?'
    r'(?:(?:mind\s+)?(?:opening|closing|lifting|raising|lowering|moving|rotating|releasing|stopping|halting)|'
    r'(?:be\s+able\s+to\s+)?(?:open|close|lift|raise|lower|move|rotate|turn|release|grasp|pick|place|halt|stop|start|continue|resume))\b|'
    r'(?:^|[.!?;。？！]\s*)(?:please\s+)?(?:open|close|lift|raise|lower|move|rotate|release|halt|stop|start)\b', re.I)
_REPORTED_EN_PREFIX = re.compile(r'\b(?:said|asked|shouted|wrote|reported|heard|told|means)\b[^.!?;。？！]*$', re.I)
_FRESH_REQUEST_EN_PREFIX = re.compile(r'\b(?:but|however|instead|now|and)\s*$', re.I)
_REPORTED_KO_PREFIX = re.compile(r'(?:누가|누군가|친구가|그\s*사람이|사람이)[^.!?;。？！]*$')
_REPORTED_KO_AFTER = re.compile(r'\s*(?:래요|래|겠어요|겠습니다|세요|수\s*있나요)?\s*[?？]?\s*(?:라고|라는|라며|라더)')
_KO_CONVERSATION_ACTIVITY = re.compile(r'(?:말|이야기|대화|설명|질문|답변)(?:하|하던|하는|한|하시는)?\s*(?:걸|것을|것|거를)\s*$')
_EN_CONVERSATION_ACTIVITY = re.compile(
    r'\s+(?:talking|speaking|discussing|explaining|answering|asking|writing|'
    r'(?:(?:a|an|the|this|that)\s+)?(?:story|conversation|discussion|explanation))\b', re.I)
_EN_NON_ACTION_SUFFIX = re.compile(r'\s+(?:on(?:\s+(?:to|with))?\b|talking\b|speaking\b|discussing\b|explaining\b|answering\b|asking\b|writing\b)', re.I)
_EN_INDIRECT_ROBOT_REQUEST = re.compile(
    r'\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:make|have|get)\s+'
    r'(?:the\s+)?(?:robot(?:ic)?(?:\s+arm)?|arm|gripper|jaws?)\s+'
    r'(?:rise|move|open|close|lift|raise|lower|rotate|stop|halt)\b', re.I)
_KO_ABILITY_ROBOT_REQUEST = re.compile(r'(?:열|닫|들|올릴|내릴|옮길|움직일|멈출|정지할)\s*수\s*있')
_ROBOT_PROHIBITION = re.compile(
    r'지\s*(?:마|말)|\b(?:refrain\s+from|avoid|do\s+not|don[\'’]t|never)\b|'
    r'\b(?:can|could|would|will)\s+you\s+(?:please\s+)?not\b', re.I)
_KO_ACTION_PROMISE = re.compile(
    r'(?:열(?=어|게|겠|을)|닫(?=아|게|겠|을)|들어(?=볼|\s|게|겠)|올(?=리|려|릴)|'
    r'내(?=리|려|릴)|옮|움직|멈|정지|중단)[^.!?。？！\n]{0,28}'
    r'(?:게요|겠습니다)(?!라고|라는|라며)')
_KO_GENERIC_PROMISE = re.compile(
    r'(?:그렇게|그걸|그것을|말씀하신\s*대로|요청하신\s*대로)\s*(?:해|하|할)[^.!?。？！\n]{0,12}'
    r'(?:게요|겠습니다)(?!라고|라는|라며)|'
    r'(?:^|[.!?。？！]\s*)(?:네[,!]?\s*|예[,!]?\s*)?(?:제가\s*)?(?:(?:지금|바로|곧)\s*)?'
    r'(?:할게요|하겠습니다|해\s*드릴게요|해\s*드리겠습니다|'
    r'(?:진행|실행|수행|처리)(?:할게요|하겠습니다|합니다|중입니다))(?!라고|라는|라며)')
_EN_ACTION_PROMISE = re.compile(
    r'\b(?:i|we)\s*(?:[\'’]ll|will|am\s+going\s+to|are\s+going\s+to|can)\s+'
    r'(?:open|close|lift|raise|lower|move|rotate|release|grasp|pick|place|halt|stop|start|continue|resume)\b', re.I)
_EN_GENERIC_PROMISE = re.compile(
    r'\b(?:i|we)\s*(?:[\'’]ll|will|am\s+going\s+to|are\s+going\s+to|can)\s+'
    r'(?:do\s+(?:that|it)|handle|take\s+care\s+of)\b', re.I)
_KO_ACTION_PROGRESS = re.compile(
    r'(?:열|열리|닫|닫히|들|올리|내리|옮기|움직이|멈추|정지하|중단하)'
    r'(?:고\s*있(?:습니다|어요)|는\s*중입니다)')
_EN_ACTION_PROGRESS = re.compile(
    r'\b(?:i\s*(?:[\'’]m|am)|we\s*(?:[\'’]re|are))\s+'
    r'(?:opening|closing|lifting|raising|lowering|moving|rotating|releasing|grasping|picking|placing|halting|stopping|starting)\b|'
    r'(?:^|[.!?]\s*)(?:sure[,!]?\s*)?(?:opening|closing|lifting|raising|lowering|moving|rotating|releasing|grasping|picking|placing|halting|stopping|starting)'
    r'\s+[^.!?\n]{0,40}\b(?:now|for\s+you)\b', re.I)
_POLITE_IMMEDIATE_HALT = re.compile(r'\s*could\s+you\s+halt\s+operations\s+now\s*[?？.!。！]?\s*', re.I)
_DIRECT_SPOKEN_STOP = re.compile(
    r'^(?:멈춰|멈춰요|멈춰\s*주세요|정지|정지해|정지해\s*주세요|그만|그만해|'
    r'(?:로봇(?:을)?|robot)\s*멈춰|지금\s*바로\s*동작을\s*멈춰|동작\s*정지|'
    r'stop|stop\s+now|please\s+stop|stop\s+the\s+motion\s+now|stop\s+the\s+robot(?:\s+now)?|'
    r'not\s+tomorrow(?:\s*,\s*|\s+)stop\s+now)'
    r'[.!。！\s]*$', re.I)
_NO_STOP_TARGET = object()


def _is_polite_immediate_halt(text):
    """Recognize exactly the tested request, including ASR punctuation variants."""
    return _POLITE_IMMEDIATE_HALT.fullmatch(text) is not None


def _is_direct_stop_text(text):
    return bool(_DIRECT_SPOKEN_STOP.fullmatch(text.strip()) or _is_polite_immediate_halt(text))


def _is_text_stop_candidate(text):
    if _is_direct_stop_text(text):
        return True
    from .scene_program_coordinator import _immediate_stop_span
    from .task_program_v2 import _current_stop_spans
    return any(_immediate_stop_span(text, start, end) for start, end in _current_stop_spans(text))


def _validated_pending_stop_prefix(text, context):
    """Independently validate a stop mention before cancelling pending work."""
    from .scene_program_coordinator import _immediate_stop_span
    from .task_program_v2 import _current_stop_spans, _validated_initial_stop
    for start, end in _current_stop_spans(text):
        if not _immediate_stop_span(text, start, end):
            continue
        candidate = {'schema': 'task_program.v2', 'steps': [
            {'op': 'stop_motion', 'scope': 'active_task', 'evidence': text[start:end]}]}
        if _validated_initial_stop(candidate, text, context, _request_id()):
            return True
    return False


class _RetargetServiceCancellation:
    """Keep a follow-up tied to its session without holding service locks."""
    def __init__(self, service, coordinator, request_cancel):
        self.service, self.coordinator, self.request_cancel = service, coordinator, request_cancel

    def is_set(self):
        # Never acquire the service lock here: the coordinator checks this
        # while holding its own lock, and stop takes the locks in the other order.
        return (self.service._shutting_down or self.service.coordinator is not self.coordinator
                or (self.request_cancel is not None and self.request_cancel.is_set()))


class _AuxiliaryOperation:
    """Bind one screen operation's cancellation to its own commit point."""
    def __init__(self):
        self.request_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._committed = False

    def is_set(self):
        return self._cancelled.is_set()

    def cancel(self):
        with self._lock:
            if self._committed:
                return False
            self._cancelled.set()
            return True

    def commit(self, apply, shutdown):
        with self._lock:
            if self._cancelled.is_set() or shutdown.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            apply()
            self._committed = True


def _mask_dialogue_quotes(text):
    from . import task_program
    chars = list(text)
    for left, right in task_program._quote_spans(text):
        chars[left:right] = ' ' * (right - left)
    return ''.join(chars)


def _clause_mentions_robot(text, position):
    boundaries = '.!?;。？！\n'
    left = max(text.rfind(mark, 0, position) for mark in boundaries) + 1
    ends = [text.find(mark, position) for mark in boundaries]
    right = min((end for end in ends if end >= 0), default=len(text))
    return bool(_ROBOT_ACTION_REFERENCE.search(text[left:right]))


def _direct_robot_action_question(text):
    """Catch a direct robot request that the question router sent to dialogue."""
    if not _ROBOT_ACTION_REFERENCE.search(text):
        return False
    unquoted = _mask_dialogue_quotes(text)
    for match in _KO_ROBOT_REQUEST.finditer(unquoted):
        if (not _clause_mentions_robot(text, match.end() - 1)
                or _KO_CONVERSATION_ACTIVITY.search(unquoted[:match.start()])):
            continue
        if (_REPORTED_KO_PREFIX.search(unquoted[:match.start()])
                and _REPORTED_KO_AFTER.match(unquoted[match.end():])):
            continue
        return True
    for match in _EN_ROBOT_REQUEST.finditer(unquoted):
        if (not _clause_mentions_robot(text, match.end() - 1)
                or _EN_CONVERSATION_ACTIVITY.match(unquoted[match.end():])):
            continue
        # A reported request is not a new instruction. A new sentence after it
        # still has to be checked independently.
        prefix = unquoted[:match.start()]
        if not _REPORTED_EN_PREFIX.search(prefix) or _FRESH_REQUEST_EN_PREFIX.search(prefix):
            return True
    return False


def _robot_request_restatement(text):
    if re.search(r'[가-힣]', _mask_dialogue_quotes(text)):
        return ('이 말로 로봇을 움직이거나 멈추지 않았습니다. '
                '원하시는 요청을 분명히 다시 말씀해 주세요.')
    return ('I did not move or stop the robot in response to that. '
            'Please restate your request clearly.')


def _robot_prohibition_acknowledgement(text):
    if re.search(r'[가-힣]', _mask_dialogue_quotes(text)):
        return '이 요청으로 로봇을 움직이거나 멈추지 않았습니다.'
    return 'I did not move or stop the robot in response to that.'


def _read_only_action_promise(text, answer):
    """Do not surface a first-person action promise as a dialogue result."""
    if not _ROBOT_ACTION_REFERENCE.search(text):
        return False
    unquoted = _mask_dialogue_quotes(answer)
    if _KO_ACTION_PROMISE.search(unquoted):
        return True
    if _KO_ACTION_PROGRESS.search(unquoted):
        return True
    if any(not _REPORTED_EN_PREFIX.search(unquoted[:match.start()])
           and not _EN_NON_ACTION_SUFFIX.match(unquoted[match.end():])
           and not _EN_CONVERSATION_ACTIVITY.match(unquoted[match.end():])
           and not (re.search(r'\bstart(?:ing)?$', match.group(), re.I)
                    and re.match(r'\s+with\b', unquoted[match.end():], re.I))
           for pattern in (_EN_ACTION_PROMISE, _EN_ACTION_PROGRESS)
           for match in pattern.finditer(unquoted)):
        return True
    action_request = (_direct_robot_action_question(text) or _KO_ABILITY_ROBOT_REQUEST.search(text)
                      or _EN_INDIRECT_ROBOT_REQUEST.search(text)
                      or _ROBOT_PROHIBITION.search(_mask_dialogue_quotes(text)))
    return bool(action_request and (_KO_GENERIC_PROMISE.search(unquoted) or _EN_GENERIC_PROMISE.search(unquoted)))


def _read_only_dialogue_reply(text, answer):
    from . import astra_offline_voice_turn as routes, task_program
    spans = task_program._quote_spans(text)
    quoted = [routes.normalized(text[left + 1:right - 1]) for left, right in spans]
    if _read_only_action_promise(text, answer):
        if any(_ROBOT_ACTION_REFERENCE.search(phrase) for phrase in quoted):
            return _HIDDEN_DIALOGUE, True
        if _ROBOT_PROHIBITION.search(_mask_dialogue_quotes(text)):
            return _robot_prohibition_acknowledgement(text), True
        return _robot_request_restatement(text), True
    if not any(routes.ROBOT_TERM.search(phrase) for phrase in quoted):
        return (_HIDDEN_DIALOGUE, True) if _UNCONFIRMED_DIALOGUE.search(answer) else (answer, False)
    if len(spans) == 1:
        left, right = spans[0]
        frame = (text[:left] + 'QUOTE' + text[right:]).strip()
        if (_MEANING_QUESTION.fullmatch(frame) and _REPORTS_A_REQUEST.fullmatch(answer.strip())
                and not _ADDITIONAL_DIALOGUE_CLAIM.search(answer)):
            # Display only a meaning derived from the quoted source. No part of
            # the untrusted answer or user-supplied free text reaches this reply.
            phrase = quoted[0]
            korean = bool(re.search(r'[가-힣]', frame))
            if (phrase in routes.MOTION_STOP or re.fullmatch(r'(?:지금\s*)?로봇(?:의)?\s*동작을\s*멈춰', phrase)):
                return ('그 말은 로봇의 동작을 멈춰 달라는 뜻입니다.' if korean
                        else "The speaker is asking for the robot's motion to stop."), False
            movement = _QUOTED_ENGLISH_MOVE.fullmatch(phrase)
            if movement:
                source, destination = (routes.NUMBER[word] for word in movement.groups())
                if source != destination:
                    return (f'그 말은 원통을 {source}번 자리에서 {destination}번 자리로 옮겨 달라는 뜻입니다.' if korean
                            else f'The speaker is asking for the cylinder to be moved from slot {source} to slot {destination}.'), False
    return _HIDDEN_DIALOGUE, True


_SLOT = r'[1-4일이삼사한두세네]\s*번\s*(?:자리)?'
_OBJECT = r'(?:원통|물체)(?:\s*(?:하나(?:를)?|한\s*개(?:를)?|[을를]))?'
_MOVE = r'(?:옮겨(?:\s*(?:줘|주세요))?|이동해(?:\s*(?:줘|주세요))?)'
_SOURCE_ONLY = re.compile(rf'^\s*(?:{_SLOT}에서\s*{_OBJECT}|{_OBJECT}\s*{_SLOT}에서)\s*{_MOVE}[.!。！]*\s*$')
_DEST_ONLY = re.compile(rf'^\s*(?:{_SLOT}(?:으로|로|에)\s*{_OBJECT}|{_OBJECT}\s*{_SLOT}(?:으로|로|에))\s*{_MOVE}[.!。！]*\s*$')
_MOVE_WITHOUT_SLOTS = re.compile(rf'^\s*{_OBJECT}\s*{_MOVE}[.!。！]*\s*$')
_AMBIGUOUS_RETURN = re.compile(
    r'^\s*(?:(?:원위치(?:로)?|제자리(?:로)?|다시)\s*(?:돌아가|돌려(?:\s*(?:줘|주세요))?)|'
    r'put\s+it\s+back)[.!。！]*\s*$', re.I)

# These bounded forms only select wording for a rejected turn. They do not
# produce a program, save a pending command, or supply missing slot values.
_QUESTION_KO_SLOT = r'([1-4일이삼사한두세네])\s*번\s*(?:자리|위치)?'
_QUESTION_KO_OBJECT = r'원통\s*(?:한\s*개|하나)(?:를|을)?'
_QUESTION_KO_SCENE = (r'(?:(?:현재|지금(?:\s*놓인)?)(?:의)?\s*)?'
                      r'(?:(?:네|4)\s*자리(?:의)?\s*)?(?:배치|상태|장면)(?:를|을|도)?')
_QUESTION_KO_ROUTES = (
    rf'{_QUESTION_KO_SLOT}에서\s*{_QUESTION_KO_SLOT}(?:으로|로)\s*{_QUESTION_KO_OBJECT}',
    rf'{_QUESTION_KO_OBJECT}\s*{_QUESTION_KO_SLOT}에서\s*{_QUESTION_KO_SLOT}(?:으로|로)',
    rf'{_QUESTION_KO_SLOT}에\s*있는\s*{_QUESTION_KO_OBJECT}\s*{_QUESTION_KO_SLOT}(?:으로|로)',
)
_QUESTION_KO_REMEMBER = rf'{_QUESTION_KO_SCENE}\s*(?:기억|저장)해(?:\s*(?:줘|주세요))?'
_QUESTION_EN_SLOT = r'(?:slot|position)\s*([1-4]|one|two|three|four)'
_QUESTION_EN_ROUTE = rf'one\s+cylinder\s+from\s+{_QUESTION_EN_SLOT}\s+to\s+{_QUESTION_EN_SLOT}'
_QUESTION_EN_SCENE = (r'(?:the\s+)?(?:(?:current|present)\s+)?(?:layout|arrangement|scene)'
                      r'(?:\s+of\s+the\s+four\s+(?:slots|positions))?')
_QUESTION_END = r'\s*[.!。！]?\s*'
_BEFORE_QUESTIONS = []
for _route in _QUESTION_KO_ROUTES:
    _BEFORE_QUESTIONS.append(('action', 'ko', re.compile(
        rf'\s*{_route}\s*(?:옮기기|이동하기)\s*전에\s*{_QUESTION_KO_REMEMBER}{_QUESTION_END}')))
    _BEFORE_QUESTIONS.append(('memory', 'ko', re.compile(
        rf'\s*{_QUESTION_KO_SCENE}\s*(?:기억|저장)하기\s*전에\s*{_route}\s*'
        rf'(?:옮겨|이동해)(?:\s*(?:줘|주세요))?{_QUESTION_END}')))
_BEFORE_QUESTIONS.extend((
    ('action', 'en', re.compile(
        rf'\s*before\s+(?:(?:you\s+)?(?:move|transfer)|moving|transferring)\s+{_QUESTION_EN_ROUTE}\s*,\s*'
        rf'(?:remember|save|memorize)\s+{_QUESTION_EN_SCENE}{_QUESTION_END}', re.I | re.ASCII)),
    ('action', 'en', re.compile(
        rf'\s*(?:remember|save|memorize)\s+{_QUESTION_EN_SCENE}\s+before\s+'
        rf'(?:you\s+)?(?:move|transfer)\s+{_QUESTION_EN_ROUTE}{_QUESTION_END}', re.I | re.ASCII)),
    ('memory', 'en', re.compile(
        rf'\s*before\s+(?:you\s+)?(?:remember|save|memorize)\s+{_QUESTION_EN_SCENE}\s*,\s*'
        rf'(?:move|transfer)\s+{_QUESTION_EN_ROUTE}{_QUESTION_END}', re.I | re.ASCII)),
    ('memory', 'en', re.compile(
        rf'\s*(?:move|transfer)\s+{_QUESTION_EN_ROUTE}\s+before\s+(?:you\s+)?'
        rf'(?:remember|save|memorize)\s+{_QUESTION_EN_SCENE}{_QUESTION_END}', re.I | re.ASCII)),
))
_QUESTION_SLOT_VALUES = {'일': 1, '한': 1, '이': 2, '두': 2, '삼': 3, '세': 3, '사': 4, '네': 4,
                         'one': 1, 'two': 2, 'three': 3, 'four': 4}


def _request_id():
    return 'demo-' + uuid.uuid4().hex


def _read_bounded_response(response, *, too_large_message, limit=_MAX_LOCAL_LLM_RESPONSE_BYTES):
    data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(too_large_message)
    return data


def _controller_status_idle(status, expected_instance_id):
    return (isinstance(status, dict)
            and status.get('instance_id') == expected_instance_id
            and status.get('fault') is None
            and status.get('worker_alive') is True
            and status.get('fresh_feedback') is True
            and status.get('active_command_id') is None
            and status.get('lease_owner') is None)


def _controller_observation_stationary(observation, expected_instance_id):
    return (isinstance(observation, dict)
            and observation.get('instance_id') == expected_instance_id
            and observation.get('sensor_valid') is True
            and observation.get('moving') is False)


def _fresh_stationary_observation_valid(observation, instance_id, limits, *, now=None):
    if now is None:
        now = time.monotonic()
    if not _controller_observation_stationary(observation, instance_id):
        return False
    observed = observation.get('observed_monotonic')
    age = observation.get('sensor_age_seconds')
    return (type(observation.get('revision')) is int and observation['revision'] >= 0
            and type(observation.get('sensor_sequence')) is int and observation['sensor_sequence'] >= 1
            and type(observed) in (int, float) and math.isfinite(observed)
            and type(age) in (int, float) and math.isfinite(age)
            and 0 <= now - observed <= limits.observation_max_age
            and age >= 0 and age + now - observed <= limits.sensor_max_age
            and observation.get('gripper_reported_valid') is True
            and observation.get('gripper_feedback_fresh') is True)


def _loopback_json(port, path):
    """Read this app's local server directly, with no proxy or redirects."""
    connection = http.client.HTTPConnection(_LOOPBACK_HOST, port, timeout=2)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        if response.status != 200:
            raise OSError('로컬 언어 모델 서버가 준비되지 않았습니다.')
        data = _read_bounded_response(response, too_large_message='로컬 언어 모델 응답이 너무 큽니다.')
        return json.loads(data)
    finally:
        connection.close()


def _local_dialogue(url, text):
    """Ask the owned local server for text only; there is no action channel."""
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != _LOOPBACK_HOST or not parsed.port:
        raise ValueError('일반 대화는 이 컴퓨터의 로컬 모델만 사용할 수 있습니다.')
    body = json.dumps({'model': _LOCAL_MODEL_ALIAS, 'messages': [
        {'role': 'system', 'content': '일반 대화에 짧고 자연스럽게 답하세요. 도구나 실행 권한이 없습니다. '
         '로봇을 움직이거나 배치를 기억하거나 실제 상태를 확인했다고 말하지 마세요.'},
        {'role': 'user', 'content': text}], 'temperature': .3, 'max_tokens': 128, 'stream': False},
        ensure_ascii=False, allow_nan=False).encode('utf-8')
    connection = http.client.HTTPConnection(_LOOPBACK_HOST, parsed.port, timeout=180)
    try:
        connection.request('POST', '/v1/chat/completions', body=body,
                           headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError('로컬 대화 모델에서 응답을 받지 못했습니다.')
        data = _read_bounded_response(response, too_large_message='로컬 대화 응답이 너무 큽니다.')
        envelope = json.loads(data)
        choice = envelope['choices'][0]
        if choice.get('finish_reason') not in (None, 'stop'):
            raise ValueError('로컬 대화 응답이 끝까지 생성되지 않았습니다.')
        reply = choice['message']['content']
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError('로컬 대화 응답이 비어 있습니다.')
        return reply.strip()[:500]
    finally:
        connection.close()


def _task(kind, evidence):
    return {'op': 'task', 'kind': kind, 'source_slot': None, 'destination_slot': None,
            'object_id': None, 'count': None, 'height': None, 'layout': None, 'evidence': evidence}


def _require_owned_connection(child, connection, port):
    """Confirm the live socket belongs to this window's server process."""
    import psutil
    if child is None or child.poll() is not None:
        raise RuntimeError('로컬 언어 모델 프로세스가 종료됐습니다.')
    peer = connection.getsockname()
    connections = psutil.Process(child.pid).net_connections(kind='tcp')
    owned = any(c.status == 'ESTABLISHED' and c.laddr and c.raddr
                and c.laddr.ip == _LOOPBACK_HOST and c.laddr.port == port
                and c.raddr.ip == peer[0] and c.raddr.port == peer[1] for c in connections)
    if not owned:
        raise RuntimeError('로컬 모델 연결의 소유자를 확인하지 못했습니다.')


class _ModelSlot:
    """Prevent multiple demo windows from owning duplicate E4B servers."""
    def __init__(self):
        self.handle = None
        self.file = None
        self.job = None
        self.offloaded_layers = None
        self.device_messages = []
        self._offload_report = threading.Event()
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
            kernel.CreateMutexW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle.restype = wintypes.BOOL
            ctypes.set_last_error(0)
            handle = kernel.CreateMutexW(None, False, 'Local\\' + _MODEL_SLOT_NAME)
            error = ctypes.get_last_error()
            if not handle:
                raise ctypes.WinError(error)
            if error == 183:  # ERROR_ALREADY_EXISTS; never attach to the other server.
                kernel.CloseHandle(handle)
                raise RuntimeError('다른 시연 프로그램이 언어 모델을 사용 중입니다. 그 프로그램을 종료하고 다시 열어 주세요.')
            self.handle, self.kernel = handle, kernel
            try:
                from .model_lifecycle import WindowsModelJob
                self.job = WindowsModelJob(LLM)
            except BaseException:
                self.close()
                raise
        else:
            import fcntl
            import tempfile
            path = Path(tempfile.gettempdir()) / f'{_MODEL_SLOT_NAME}-{os.getuid()}.lock'
            file = path.open('a+b')
            try:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                file.close()
                raise RuntimeError('다른 시연 프로그램이 언어 모델을 사용 중입니다. 그 프로그램을 종료하고 다시 열어 주세요.')
            self.file = file

    def launch(self, args):
        spawn = self.job.popen if self.job is not None else subprocess.Popen
        child = spawn(args, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

        def read_log():
            with child.stdout as stream:
                for line in stream:
                    if len(self.device_messages) < 48 and any(
                        word in line.lower() for word in (b'offload', b'model buffer', b'fit:', b'cuda0', b'cuda device')
                    ):
                        self.device_messages.append(line.decode('utf-8', errors='replace').strip())
                    layers = re.search(rb'offloaded (\d+)/(\d+) layers to GPU', line)
                    if layers:
                        self.offloaded_layers = tuple(map(int, layers.groups()))
                        self._offload_report.set()
            self._offload_report.set()

        threading.Thread(target=read_log, name='gemma-device-report', daemon=True).start()
        return child

    def confirm_gpu(self):
        self._offload_report.wait(2)
        if not self.offloaded_layers or self.offloaded_layers[0] == 0:
            raise RuntimeError('Gemma 레이어가 GPU에 적재된 것을 확인하지 못했습니다.')

    def close(self):
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        if self.file is not None:
            self.file.close()
            self.file = None


def _warm_local_model(url, child, cancelled):
    """Prime the task prompt/schema on our local model without a ticket or action."""
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != _LOOPBACK_HOST or not parsed.port:
        raise ValueError('언어 모델 예열은 이 컴퓨터에서만 할 수 있습니다.')
    from .task_program_v2 import TaskProgramChatClient, build_messages

    def owned_transport(request_url, body, timeout_seconds):
        destination = urlsplit(request_url)
        if (destination.scheme != 'http' or destination.hostname != _LOOPBACK_HOST
                or destination.port != parsed.port or destination.path != '/v1/chat/completions'):
            raise ValueError('언어 모델 예열 주소가 올바르지 않습니다.')
        if cancelled.is_set():
            raise InterruptedError('로컬 모델 준비를 종료했습니다.')
        connection = http.client.HTTPConnection(_LOOPBACK_HOST, parsed.port, timeout=timeout_seconds)
        try:
            connection.connect()
            _require_owned_connection(child, connection.sock, parsed.port)
            connection.request('POST', destination.path, body=body,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError('로컬 언어 모델 예열 응답이 올바르지 않습니다.')
            return _read_bounded_response(response, too_large_message='로컬 언어 모델 예열 응답이 너무 큽니다.')
        finally:
            connection.close()

    client = TaskProgramChatClient(f'http://{_LOOPBACK_HOST}:{parsed.port}/v1', _LOCAL_MODEL_ALIAS,
                                   transport=owned_transport)
    text = client.complete(build_messages('현재 배치를 기억해 줘.', None), timeout_seconds=180)
    if not text.strip():
        raise ValueError('로컬 언어 모델 예열 결과가 비었습니다.')
    if cancelled.is_set():
        raise InterruptedError('로컬 모델 준비를 종료했습니다.')


class LocalInterpreter:
    """Own one offline loopback model process and its optional background warm-up."""
    def __init__(self):
        self.process = None
        self.interpreter = None
        self._slot = None
        self._cv = threading.Condition()
        self._starting = False
        self._closed = threading.Event()
        self._warm_state = 'idle'
        self._warm_error = None
        self._warm_thread = None
        self._warm_emit = lambda *_: None

    def _set_warm_state(self, state, message):
        with self._cv:
            if self._closed.is_set():
                return
            self._warm_state = state
            if state == 'failed':
                self._warm_error = message
            self._cv.notify_all()
        self._warm_emit({'state': state, 'message': message})

    def prepare_async(self, emit):
        """Start exactly one warm-up, independently of any command cancellation."""
        with self._cv:
            if self._closed.is_set():
                raise RuntimeError('종료한 로컬 모델은 다시 시작할 수 없습니다.')
            if self._warm_state != 'idle':
                return self._warm_thread
            self._warm_emit = emit
            self._warm_state = 'loading'
            worker = threading.Thread(target=self._prepare, daemon=True, name='demo-e4b-warmup')
            self._warm_thread = worker
            # Start under the same lock used by close(): it can always join
            # the published worker, even when shutdown races this call.
            worker.start()
        return worker

    def _prepare(self):
        try:
            self._warm_emit({'state': 'loading', 'message': 'Gemma 4 E4B를 로컬에서 불러오는 중입니다.'})
            interpreter = self._ensure_server(self._closed)
            self._set_warm_state('warming', 'Gemma 4 E4B 적재 완료 · 작업 형식 예열 중입니다. 시간이 걸릴 수 있습니다.')
            _warm_local_model(interpreter.client.url, self.process, self._closed)
            self._set_warm_state('ready', 'Gemma 4 E4B 작업 형식 예열 완료 · 실제 요청은 별도 처리합니다.')
        except Exception as exc:
            if not self._closed.is_set():
                detail = str(exc) or type(exc).__name__
                try:
                    self._discard_process()
                except Exception as cleanup_error:
                    detail += ' · 프로세스 정리 실패: ' + (str(cleanup_error) or type(cleanup_error).__name__)
                self._set_warm_state('failed', 'Gemma 4 E4B 준비 실패: ' + detail)

    def start(self, cancelled):
        self.check_liveness()
        with self._cv:
            while self._warm_state in ('loading', 'warming'):
                if cancelled.is_set() or self._closed.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                self._cv.wait(.1)
            if cancelled.is_set() or self._closed.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            if self._warm_state == 'failed':
                raise RuntimeError(self._warm_error or '로컬 언어 모델 준비에 실패했습니다.')
        return self._ensure_server(cancelled)

    def _ensure_server(self, cancelled):
        failed = None
        with self._cv:
            while True:
                if cancelled.is_set() or self._closed.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                if self.interpreter is not None and self.process is not None and self.process.poll() is None:
                    return self.interpreter
                if self._slot is not None and not self._starting:
                    failed = 'Gemma 4 E4B 프로세스가 종료됐습니다. 창을 다시 열어 주세요.'
                    self._warm_state, self._warm_error = 'failed', failed
                    self._cv.notify_all()
                    break
                if not self._starting:
                    self._starting = True
                    break
                self._cv.wait(.1)
        if failed is not None:
            self._warm_emit({'state': 'failed', 'message': failed})
            raise RuntimeError(failed)
        child = None
        try:
            if not all(path.is_file() for path in (LLAMA_SERVER, LLM, LLM_PROJECTOR)):
                raise FileNotFoundError('Gemma 모델, 음성 인코더 또는 CUDA 실행 파일을 찾지 못했습니다.')
            with socket.socket() as probe:
                probe.bind((_LOOPBACK_HOST, 0))
                port = probe.getsockname()[1]
            args = [str(LLAMA_SERVER), '-m', str(LLM), '--host', _LOOPBACK_HOST, '--port', str(port),
                    '--alias', _LOCAL_MODEL_ALIAS, '--ctx-size', str(_LOCAL_LLM_CTX_SIZE), '--parallel', '1',
                    '--threads', str(_LOCAL_LLM_THREADS), '--threads-batch', str(_LOCAL_LLM_THREADS),
                    '--reasoning', 'off', '--verbosity', '4',
                    '--no-webui', '--offline', '--cache-ram', str(_LOCAL_LLM_CACHE_RAM_MB),
                    '--device', _LOCAL_LLM_DEVICE,
                    '--gpu-layers', 'auto', '--fit', 'on', '--fit-target', str(_LOCAL_LLM_FIT_TARGET_MB),
                    '--mmproj', str(LLM_PROJECTOR), '--no-mmproj-offload']
            # Register the child atomically with respect to close(), so closing
            # during Popen cannot leave an unowned process running.
            with self._cv:
                if cancelled.is_set() or self._closed.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                self._slot = _ModelSlot()
                child = self._slot.launch(args)
                self.process = child
                self.interpreter = None
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if cancelled.is_set() or self._closed.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                if child.poll() is not None:
                    raise RuntimeError('로컬 언어 모델을 시작하지 못했습니다.')
                try:
                    ready = _loopback_json(port, '/health').get('status') == 'ok'
                    if ready:
                        with socket.create_connection((_LOOPBACK_HOST, port), timeout=2) as conn:
                            _require_owned_connection(child, conn, port)
                        props = _loopback_json(port, '/props')
                        if Path(props['model_path']).resolve() != LLM.resolve():
                            raise RuntimeError('선택한 로컬 언어 모델과 실행 중인 모델이 다릅니다.')
                        if props.get('modalities', {}).get('audio') is not True:
                            raise RuntimeError('Gemma 음성 인코더가 준비되지 않았습니다.')
                        self._slot.confirm_gpu()
                        from .task_program_v2 import TaskProgramChatClient, TaskProgramInterpreter
                        interpreter = TaskProgramInterpreter(TaskProgramChatClient(
                            f'http://{_LOOPBACK_HOST}:{port}/v1', _LOCAL_MODEL_ALIAS))
                        with self._cv:
                            if cancelled.is_set() or self._closed.is_set() or self.process is not child:
                                raise InterruptedError('요청을 취소했습니다.')
                            self.interpreter = interpreter
                        return interpreter
                except (OSError, KeyError, ValueError):
                    self._closed.wait(.25)
            raise TimeoutError('로컬 언어 모델 준비 시간이 초과됐습니다.')
        except BaseException:
            self._discard_process(child)
            raise
        finally:
            with self._cv:
                self._starting = False
                self._cv.notify_all()

    @staticmethod
    def _stop_child(child):
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except OSError:
                if child.poll() is None:
                    raise
                return
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)

    def _discard_process(self, expected=None):
        with self._cv:
            child = self.process
            if expected is not None and child is not expected:
                return
            self.process = self.interpreter = None
            slot, self._slot = self._slot, None
        try:
            self._stop_child(child)
            if slot is not None:
                slot.close()
        except BaseException:
            with self._cv:
                if self.process is None and self._slot is None:
                    self.process, self._slot = child, slot
            raise

    def check_liveness(self):
        with self._cv:
            failed = (not self._closed.is_set() and self._warm_state == 'ready'
                      and (self.process is None or self.process.poll() is not None))
        if failed:
            self._set_warm_state('failed', 'Gemma 4 E4B 프로세스가 종료됐습니다. 창을 다시 열어 주세요.')

    def request_close(self):
        """Cancel loading and interrupt an owned server request without waiting."""
        with self._cv:
            self._closed.set()
            child = self.process
            self._cv.notify_all()
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except OSError:
                pass

    def close(self):
        self.request_close()
        worker = self._warm_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        self._discard_process()

    def dialogue(self, text, cancelled):
        interpreter = self.start(cancelled)
        if cancelled.is_set():
            raise InterruptedError('대화를 취소했습니다.')
        reply = _local_dialogue(interpreter.client.url, text)
        if cancelled.is_set():
            raise InterruptedError('대화를 취소했습니다.')
        return reply


class _BoundedProposal:
    """A rule-generated proposal, never an execution ticket."""
    def __init__(self, program):
        self.program = program

    def complete(self, _messages, *, timeout_seconds):
        return json.dumps({'schema': self.program['schema'], 'steps': self.program['steps']}, ensure_ascii=False)


def _before_clause_question(text, interpreted, admission):
    """Ask about one clear, unexecuted before-clause without creating a task."""
    if (admission.get('kind') != 'rejected' or admission.get('program_executed') is not False
            or admission.get('status') != 'unresolved_reference' or interpreted.status != 'unresolved_reference'
            or interpreted.program is not None or interpreted.execution_authorized
            or interpreted.validated_stop_prefix or not interpreted.schema_valid):
        return None
    from . import task_program_v2 as v2
    try:
        v2._reject_ambiguous_before_clause(text)
    except v2.ProgramError as exc:
        if exc.code != 'unresolved_reference':
            return None
        diagnostic = str(exc)
    else:
        return None
    for subject, language, pattern in _BEFORE_QUESTIONS:
        match = pattern.fullmatch(text)
        if not match or diagnostic != f'a future before-clause does not clearly request the later {subject}':
            continue
        slots = [int(value) if value.isdecimal() else _QUESTION_SLOT_VALUES[value.lower()]
                 for value in match.groups()]
        if len(slots) != 2 or slots[0] == slots[1]:
            return None
        if language == 'ko':
            return ('그 뒤에 원통도 이동할까요? 이동 여부를 포함해 전체 요청을 다시 말씀해 주세요.'
                    if subject == 'action' else
                    '그 뒤에 배치도 기억할까요? 기억 여부를 포함해 전체 요청을 다시 말씀해 주세요.')
        return ('Should I also move the cylinder afterward? Please restate the full request, including your answer.'
                if subject == 'action' else
                'Should I also remember the layout afterward? Please restate the full request, including your answer.')
    return None


def _voice_feedback(text, interpreted, admission, *, virtual=True):
    """Ground a short question independently; never speak a model's raw question."""
    if admission.get('kind') == 'stop_prefix':
        task_name = '가상 작업' if virtual else '작업'
        lead = {'stop_requested': task_name + ' 중단을 요청했습니다.',
                'already_stopping': task_name + '은 이미 중단 중입니다.',
                'no_active_task': '중단할 ' + task_name + '이 없습니다.'}.get(
                    admission.get('status'), '중단 상태를 확인해 주세요.')
        if admission.get('follow_up_status') != 'none':
            return lead + ' 이후 작업은 실행하지 않았습니다. 전체 요청을 다시 말씀해 주세요.'
        return lead
    if admission.get('kind') != 'rejected':
        return None
    if admission.get('status') == 'slot_unavailable':
        return f"{admission['slot']}번 자리는 없습니다. 이 로봇의 자리는 1번부터 4번까지입니다."
    if admission.get('status') == 'unsupported_TransferScope':
        slots = admission.get('supported_slots')
        if isinstance(slots, (list, tuple)) and slots:
            return '현재 연결 설정에서는 ' + '번과 '.join(map(str, slots)) + '번 자리로 이동할 수 있습니다.'
        return '요청한 이동 경로는 현재 연결 설정에서 지원하지 않습니다.'
    if admission.get('status') in ('infeasible_scene', 'target_unavailable'):
        return '현재 배치에서는 그 이동을 할 수 없습니다. 출발 자리의 원통과 목적지 높이를 확인해 주세요.'
    if admission.get('status') == 'unsupported_step':
        return '요청하신 작업은 현재 지원 범위에서 실행할 수 없습니다.'
    if admission.get('status') == 'preflight_rejected':
        return '요청은 이해했지만 현재 배치나 지원 범위에서는 실행할 수 없습니다.'
    if admission.get('status') == 'recovery_required':
        return '작업이 중단되어 물체 위치를 확인하지 못했습니다. 현재 배치를 확인해 주세요.'
    generic = '요청을 정확히 확인하지 못했습니다. 전체 요청을 다시 말씀해 주세요.'
    if not admission.get('unresolved') or admission.get('status') != 'ok':
        return generic
    reasons = [item.get('reason') for item in admission.get('clarifications', []) if isinstance(item, dict)]
    if reasons == ['missing_target']:
        if _SOURCE_ONLY.fullmatch(text):
            return '어느 자리로 옮길지 포함해 전체 요청을 다시 말씀해 주세요.'
        if _DEST_ONLY.fullmatch(text):
            return '어느 자리에서 가져올지 포함해 전체 요청을 다시 말씀해 주세요.'
        if _MOVE_WITHOUT_SLOTS.fullmatch(text):
            return '어느 자리에서 어느 자리로 옮길지 포함해 전체 요청을 다시 말씀해 주세요.'
    program = interpreted.program if isinstance(interpreted.program, dict) else {}
    steps = program.get('steps') or []
    if (len(steps) == 1 and steps[0].get('op') == 'return'
            and (steps[0].get('target') or {}).get('kind') == 'unresolved'
            and _AMBIGUOUS_RETURN.fullmatch(text)):
        return '무엇을 어디로 되돌릴지 포함해 전체 요청을 다시 말씀해 주세요.'
    return generic


def _execution_failure_message(execution):
    """Describe a known failure without treating command delivery as success."""
    if not isinstance(execution, dict):
        return '작업을 마치지 못했습니다. 물체 상태를 확인해 주세요.'
    if execution.get('status') == 'motion_not_observed':
        return '팔의 움직임이 감지되지 않아 작업을 중단했습니다. 연결과 장치 상태를 확인해 주세요.'
    if (execution.get('status') == 'step_unavailable'
            and (execution.get('not_started_reason') == 'controller_not_ready'
                 or any(isinstance(event, dict) and event.get('event') == 'transfer'
                        and event.get('not_started_reason') == 'controller_not_ready'
                        for event in execution.get('events') or []))):
        return '제어기 수신이 준비되지 않아 움직이지 않았습니다. 연결 상태를 확인해 주세요.'
    termination = execution.get('failure_termination')
    error = execution.get('failure_error')
    released = execution.get('release_expected') is True
    for event in reversed(execution.get('events') or []):
        if isinstance(event, dict) and event.get('event') == 'transfer':
            result = event.get('result')
            if isinstance(result, dict):
                termination = result.get('termination') or termination
                error = result.get('error') or error
                released = event.get('release_expected') is True
            break
    reasons = {
        'motion_not_observed': '팔의 움직임이 감지되지 않아 작업을 중단했습니다.',
        'device_warning': '장치 경고로 작업을 중단했습니다.',
        'warning_unavailable': '장치 경고 상태를 확인하지 못해 작업을 중단했습니다.',
        'invalid_observation': '장치의 최신 상태를 확인하지 못해 작업을 중단했습니다.',
        'controller_changed': '제어기 연결 상태가 바뀌어 작업을 중단했습니다.',
        'controller_error': '제어기 오류로 작업을 중단했습니다.',
        'command_unconfirmed': '제어기 명령 결과를 확인하지 못해 작업을 중단했습니다.',
        'target_unconfirmed': '팔이 목표 위치에 도착했는지 확인하지 못했습니다.',
        'gripper_unconfirmed': '집게 상태를 확인하지 못해 작업을 중단했습니다.',
        'end_unconfirmed': '작업 뒤 정지를 확인하지 못했습니다.',
        'stop_unconfirmed': '정지를 확인하지 못했습니다.',
        'next_reference_unconfirmed': '작업 뒤 팔의 상태를 확인하지 못했습니다.',
        'decision_limit': '판단 횟수 한도에 도달해 작업을 중단했습니다.',
        'out_of_range_target': '관절 목표가 허용 범위를 벗어나 작업을 중단했습니다.',
        'cancelled': '요청에 따라 작업이 중단됐습니다.',
    }
    if isinstance(error, dict) and error.get('code') in ('DISCONNECTED', 'SENSOR_STALE'):
        reason = '연결이 끊겨 팔 이동을 마치지 못했습니다.'
    else:
        reason = reasons.get(termination, '작업을 마치지 못했습니다.')
    if released:
        return '놓기 명령을 확인해 예상 배치를 갱신했습니다. ' + reason + ' 실제 물체 위치는 확인하지 못했습니다.'
    return reason + ' 물체 상태는 미확인입니다.'


class DemoService:
    def __init__(self, emit=lambda *_: None, *, visual_pace_seconds=0., controller_runtime=None,
                 allow_runtime_retarget=True):
        if type(allow_runtime_retarget) is not bool:
            raise ValueError('allow_runtime_retarget must be boolean')
        self.emit = emit
        # A capable connected controller handles bounded runtime redirection.
        # This option neither opens a device nor certifies physical execution.
        self._allow_runtime_retarget = allow_runtime_retarget
        self.visual_pace_seconds = visual_pace_seconds
        self.coordinator = None
        self.sim = None
        self.forecast = None
        self.model_hash = None
        self.cancelled = threading.Event()
        self._text_request_lock = threading.RLock()
        self._text_request = None
        self._pending_text_plans = {}
        self._record_request = None
        self._registration_requests = {}
        self._registration_id = None
        self._transcription_requests = set()
        self._conversation_requests = set()
        self._request_context = threading.local()
        self._shutting_down = False
        self.interpreter = LocalInterpreter()
        self._voice_lock = threading.Lock()
        self._voice_closed = False
        self.voice = None
        self.mode = 'virtual'
        self._controller = None
        self._controller_initial_target = None
        self._controller_calibration = None
        self._registered_calibration = None
        self._controller_profile = None
        self._controller_profile_path = None
        self._controller_connecting = None
        self._physical_connection_uncertain = False
        self._pending_physical_start = None
        self._physical_initial_pose_required = False
        self._physical_connection_prepared = False
        self._controller_preparation = None
        self._controller_recovery = None
        self._automatic_recovery = None
        self._automatic_recovered = None
        self._automatic_recovery_last_started = 0.
        self._idle_connection_recovery = None
        self._replacement_started = None
        self._replacement_last_attempt = 0.
        self._unregistered_replacement = None
        self._controller_reconnect_revision = 0
        self._unregistered_idle_controller = None
        self._unregistered_interrupted_controller = None
        self._physical_recovery_required = False
        self._physical_recovery_epoch = 0
        # A failed read or stale feedback while known idle is not evidence
        # that a completed command or registered scene became uncertain.
        # Only the same resident can clear this temporary interruption.
        self._status_read_recovery = None
        self._last_healthy_status_owner = None
        self._last_healthy_gripper = None
        self._status_read_observation_refresh = None
        self._status_read_waiter = None
        self._physical_stop_cleared_session = None
        self._controller_runtime = Path(controller_runtime) if controller_runtime is not None else ROOT / '.runtime/demo-web-mock'
        self._last_disconnect = None

    def _set_physical_recovery_locked(self, required, *, status_read_owner=None):
        if not required:
            if self._status_read_recovery is not None:
                self._status_read_observation_refresh = self._status_read_recovery
            self._status_read_recovery = None
        elif status_read_owner is None:
            # A real stop, command uncertainty, or other recovery reason must
            # never be cleared by a later successful display read.
            self._status_read_recovery = None
            self._status_read_observation_refresh = None
        elif not self._physical_recovery_required:
            self._status_read_recovery = status_read_owner
        if self._physical_recovery_required != required:
            self._physical_recovery_required = required
            self._physical_recovery_epoch += 1

    def _require_idle_locked(self, *, registration=False):
        if (self._shutting_down or self._text_request is not None or self._record_request is not None
                or self._conversation_requests or self._transcription_requests or self._controller_preparation is not None
                or self._controller_connecting is not None
                or self._controller_recovery is not None
                or (self._registration_requests and not registration)):
            raise RuntimeError('진행 중인 작업을 먼저 종료해 주세요.')
        current = self.coordinator
        if current is not None:
            with current._lock:
                if (current._active is not None or current._interpretations or current._pending
                        or current._arm_home_reservation is not None):
                    raise RuntimeError('진행 중인 작업을 먼저 종료해 주세요.')

    def _recovery_blocks_new_request_locked(self):
        # During automatic idle reconnection new language requests can be
        # interpreted, then wait before acquiring the controller.
        return self._controller_recovery is not None and self._idle_connection_recovery is None

    def set_mode(self, mode):
        """Switch only while idle; invalidate the old scene and close its owner."""
        if mode not in ('virtual', 'mock', 'physical'):
            raise ValueError('지원되는 실행 방식을 선택해 주세요.')
        old_forecast = None
        with self._text_request_lock:
            self._require_idle_locked()
            if mode != self.mode:
                self._pending_physical_start = None
                self._disconnect_controller_locked()
                old_forecast = self.forecast
                self.forecast = self.sim = self.coordinator = self.model_hash = None
                self._registration_id = None
                self.mode = mode
        if old_forecast is not None:
            old_forecast.close()
        return self.connection_snapshot()

    def connect_controller(self, *, physical_confirmed=False, profile_path=None, operation=None):
        """Explicitly connect one selected controller; never run on mode selection."""
        operation = operation or _AuxiliaryOperation()
        with self._text_request_lock:
            existing_session = (self.mode == 'physical' and self._controller is not None
                                and self.coordinator is not None and self._physical_recovery_required)
            if existing_session:
                self._require_idle_locked()
                if physical_confirmed is not True:
                    raise ValueError('실물 로봇 연결을 명시적으로 선택해 주세요.')
                old_controller, coordinator = self._controller, self.coordinator
        if existing_session:
            replacement = self._start_controller_replacement(old_controller, coordinator, None, operation=operation)
            if replacement is None:
                raise RuntimeError('현재 연결을 다시 준비하고 있습니다. 잠시 뒤 다시 연결해 주세요.')
            replacement.join()
            operation.commit(lambda: None, self.cancelled)
            return self.connection_snapshot()
        reconnect_attempt = False
        with self._text_request_lock:
            self._require_idle_locked()
            if self.mode == 'virtual':
                raise ValueError('가상 실행에는 제어기 연결이 필요하지 않습니다.')
            mode = self.mode
            if mode == 'mock' and (physical_confirmed is not False or profile_path is not None):
                raise ValueError('모의 연결에는 실물 설정을 사용할 수 없습니다.')
            if mode == 'physical' and physical_confirmed is not True:
                raise ValueError('실물 로봇 연결을 명시적으로 선택해 주세요.')
            pending = self._pending_physical_start if mode == 'physical' else None
            requested_profile = Path(profile_path or ROOT / 'configs/app.hardware.json').resolve()
            if mode == 'physical' and self._physical_connection_uncertain and self._controller is None:
                if pending is None:
                    # No outstanding child handle remains. Resolve any old
                    # resident by its recorded identity instead of permanently
                    # blocking the new connection on this local flag.
                    reconnect_attempt = True
                elif (pending[1].is_set() or self.cancelled.is_set() or self._shutting_down
                        or pending[2] != self._physical_recovery_epoch or pending[3] != requested_profile
                        or not self._physical_recovery_required or not self._physical_initial_pose_required):
                    child = pending[0][0].process
                    if child.poll() is None:
                        raise RuntimeError('이전 연결 준비가 아직 진행 중입니다.')
                    self._pending_physical_start = None
                    pending = None
                    reconnect_attempt = True
            if self._controller is not None:
                if mode == 'physical' and self._physical_recovery_required:
                    if self.coordinator is not None:
                        raise RuntimeError('작업 결과가 남아 있어 재연결하기 전에 현재 상태를 확인해야 합니다.')
                    self._disconnect_controller_locked()
                    reconnect_attempt = True
                elif mode == 'physical' and self._physical_initial_pose_required:
                    raise RuntimeError('초기 자세가 확인되지 않았습니다. 연결을 해제하고 다시 연결해 주세요.')
                else:
                    return self.connection_snapshot()
            recovery_required = self._physical_recovery_required
            initial_pose_required = self._physical_initial_pose_required
            recovery_epoch = self._physical_recovery_epoch
            self._controller_connecting = (operation, None)
        preflight_failed = False
        resumed_start = None
        try:
            from .demo_controller import (mock_controller, physical_controller, PhysicalProfileValidationError,
                                          resume_pending_physical_controller, verify_pending_physical_profile)
            if mode == 'mock':
                controller, initial = mock_controller(self._controller_runtime)
                calibration = profile = None
            else:
                try:
                    if pending is None:
                        if reconnect_attempt:
                            controller, calibration, profile = physical_controller(requested_profile, force_reconnect=True)
                        else:
                            controller, calibration, profile = physical_controller(requested_profile)
                    else:
                        controller, calibration, profile = resume_pending_physical_controller(pending[0], requested_profile)
                        resumed_start = pending[0][0].request_id
                except PhysicalProfileValidationError:
                    preflight_failed = True
                    raise
                initial = None
            with self._text_request_lock:
                if pending is not None:
                    if (self._pending_physical_start is not pending or pending[1].is_set()
                            or operation.is_set() or self.cancelled.is_set() or self._shutting_down
                            or self._physical_recovery_epoch != pending[2]):
                        self._pending_physical_start = None
                        raise InterruptedError('연결 준비가 취소되었거나 이전 상태가 바뀌었습니다.')
                    # Once attached, even an uncertain home result is never replayed.
                    self._pending_physical_start = None
                self._controller, self._controller_initial_target = controller, initial
                self._controller_calibration, self._controller_profile = calibration, profile
                self._controller_profile_path = requested_profile if mode == 'physical' else None
                self._controller_connecting = (operation, controller)
            if operation.is_set() or self.cancelled.is_set() or self._shutting_down:
                raise InterruptedError('연결 준비를 취소했습니다. 제어기 상태를 확인해 주세요.')
            if mode == 'physical':
                reconnected = (reconnect_attempt or
                    bool((getattr(controller, 'attachment', None) or {}).get('reconnected')))
                if recovery_required and pending is None and not reconnected:
                    raise RuntimeError('이전 결과가 미확인입니다. 초기 자세 이동 전 제어기 정지와 오류를 확인해 주세요.')
                if pending is not None:
                    verify_pending_physical_profile(pending[0], requested_profile)
                if reconnected:
                    # A replacement never repeats a previous home or move.
                    status, observation = controller.status(), controller.observe()
                    if not (_controller_status_idle(status, controller.instance_id)
                            and _controller_observation_stationary(observation, controller.instance_id)):
                        raise RuntimeError('새 제어기의 정지된 수신을 확인하지 못했습니다.')
                else:
                    # The first connection still performs the requested initial pose.
                    move_cancel = _EitherCancellation(operation, pending[1]) if pending is not None else operation
                    if pending is not None:
                        self._move_controller_initial_pose(controller, move_cancel, owner='demo-connect',
                                                           bind_final_revision=True)
                    else:
                        self._move_controller_initial_pose(controller, move_cancel, owner='demo-connect')
            if pending is not None:
                pending[1].commit(lambda: None, self.cancelled)
            operation.commit(lambda: None, self.cancelled)
            if mode == 'physical':
                with self._text_request_lock:
                    if pending is not None:
                        if self._physical_recovery_epoch != pending[2] or not self._physical_recovery_required:
                            raise RuntimeError('연결 중 복구 상태가 바뀌었습니다. 제어기를 확인해 주세요.')
                        self._set_physical_recovery_locked(False)
                        self._physical_connection_uncertain = False
                    elif reconnected:
                        if self._physical_recovery_epoch != recovery_epoch:
                            raise RuntimeError('재연결 중 복구 상태가 바뀌었습니다.')
                        self._set_physical_recovery_locked(False)
                        self._physical_connection_uncertain = False
                    # A reconnect uses the newly observed stopped pose; it
                    # does not claim that an earlier home motion completed.
                    self._physical_connection_prepared = True
                    self._physical_initial_pose_required = False
        except BaseException as exc:
            if mode == 'physical' and not preflight_failed:
                with self._text_request_lock:
                    self._set_physical_recovery_locked(True)
                    self._physical_initial_pose_required = True
                    if getattr(exc, 'code', None) in {'APP_CONTROLLER_START_PENDING',
                                                      'APP_CONTROLLER_READY_CONNECTION_UNKNOWN'}:
                        self._physical_connection_uncertain = True
                        context = getattr(exc, 'physical_context', None)
                        from .app_runtime import PendingControllerStart
                        if (pending is None and not recovery_required and not initial_pose_required
                                and recovery_epoch + 1 == self._physical_recovery_epoch
                                and self.coordinator is None and self._registration_id is None
                                and isinstance(context, tuple) and len(context) == 3
                                and isinstance(context[0], PendingControllerStart)
                                and not operation.is_set() and not self.cancelled.is_set() and not self._shutting_down):
                            self._pending_physical_start = (context, operation, self._physical_recovery_epoch,
                                                            requested_profile)
                    elif pending is not None and (self._controller is not None or operation.is_set()
                            or pending[1].is_set() or self.cancelled.is_set() or self._shutting_down
                            or getattr(exc, 'code', None) not in {'NOT_READY', 'CONTROLLER_NOT_FOUND', 'CONTROLLER_UNREACHABLE'}):
                        self._pending_physical_start = None
                    if pending is not None and (operation.is_set() or pending[1].is_set()
                            or self.cancelled.is_set() or self._shutting_down):
                        self._pending_physical_start = None
                    if pending is not None and self._pending_physical_start is pending:
                        try:
                            # Order cancellation against returning a retryable
                            # status; a cancelled attempt cannot leave a later
                            # request eligible to continue its initial pose.
                            operation.commit(lambda: None, self.cancelled)
                        except InterruptedError:
                            self._pending_physical_start = None
                    context = getattr(exc, 'physical_context', None)
                    if (pending is None and (getattr(exc, 'code', None) in
                            {'APP_CONTROLLER_START_FAULT', 'APP_CONTROLLER_WORKER_LOST',
                             'APP_CONTROLLER_FEEDBACK_LOST', 'APP_CONTROLLER_STATUS_INVALID',
                             'APP_CONTROLLER_START_FAILED'} or
                            (getattr(exc, 'code', None) not in {'APP_CONTROLLER_START_PENDING',
                                                                 'APP_CONTROLLER_READY_CONNECTION_UNKNOWN'}
                             and isinstance(context, tuple) and len(context) == 3))):
                        # Ready was reported, but the spawned controller is no
                        # longer confirmed healthy. Do not start another one.
                        self._physical_connection_uncertain = True
            raise
        finally:
            with self._text_request_lock:
                if self._controller_connecting and self._controller_connecting[0] is operation:
                    self._controller_connecting = None
        result = self.connection_snapshot()
        if resumed_start is not None:
            result['resumed_pending_start'] = resumed_start
        return result

    def _move_controller_initial_pose(self, controller, cancellation, *, owner, bind_final_revision=False):
        """Use one stopped, exclusive episode for the SDK home targets."""
        from raccoon_control.errors import ControlError

        def fail_for_connection_fault(fault):
            if isinstance(fault, dict) and fault.get('code') in ('DISCONNECTED', 'SENSOR_STALE'):
                details = fault.get('details') if isinstance(fault.get('details'), dict) else {}
                raise ControlError(fault['code'], fault.get('message', '제어기 수신이 끊겼습니다.'), **details)

        lease, ended = None, False
        def check_cancel():
            if cancellation.is_set() or self.cancelled.is_set() or self._shutting_down:
                raise InterruptedError('초기 자세 준비를 취소했습니다.')
        try:
            check_cancel()
            lease = controller.acquire(owner + '-stop-check')['lease_id']
            ended = True  # A lost response must never cause a resend.
            prior = controller.end_episode(lease)
            if prior.get('stop', {}).get('confirmed') is not True or prior.get('fault') is not None:
                fail_for_connection_fault(prior.get('fault'))
                raise RuntimeError('초기 자세 이동 전 현재 팔 정지를 확인하지 못했습니다.')
            lease, ended = None, False
            check_cancel()
            lease = controller.acquire(owner + '-initial-pose')['lease_id']
            check_cancel()
            observation = controller.observe(lease)
            stop = prior['stop']
            if (observation.get('instance_id') != controller.instance_id
                    or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                    or observation.get('sensor_sequence', -1) < stop.get('confirmed_sensor_sequence', float('inf'))
                    or observation.get('observed_monotonic', -1) < stop.get('confirmed_sensor_monotonic', float('inf'))):
                raise RuntimeError('초기 자세 이동 전 최신 정지 관측을 확인하지 못했습니다.')
            initial_observation = observation
            targets = _initial_pose_targets(observation, controller.config)
            for index, target in enumerate(targets):
                check_cancel()
                response = controller.apply_action({'lease_id': lease, 'command_id': uuid.uuid4().hex,
                    'observation_id': observation['observation_id'],
                    'action': {'kind': 'move_joints', 'joints_deg': target,
                               'speed_pct': min(8., controller.config.limits.max_speed_pct)}})
                if (response.get('status') not in ('completed', 'already_at_target')
                        or (response.get('status') == 'completed' and response.get('applied') is not True)
                        or (response.get('status') == 'completed'
                            and response.get('stop', {}).get('confirmed') is not True)):
                    detail = response.get('error') or {}
                    fail_for_connection_fault(detail)
                    raise RuntimeError('초기 자세 도착 또는 정지 확인이 완료되지 않았습니다.'
                                       + (' ' + str(detail.get('message')) if detail.get('message') else ''))
                action_observation = response.get('observation') if bind_final_revision else None
                if (bind_final_revision and
                        (type(observation.get('revision')) is not int
                         or not isinstance(action_observation, dict)
                         or action_observation.get('instance_id') != observation.get('instance_id')
                         or type(action_observation.get('revision')) is not int
                         or action_observation['revision'] != observation['revision'] + 2
                         or response.get('replayed') is not False)):
                    raise RuntimeError('초기 자세 요청 뒤 최신 정지 상태를 확인하지 못했습니다.')
                if index + 1 < len(targets):
                    check_cancel()
                    observation = controller.observe(lease)
                    if (observation.get('instance_id') != initial_observation.get('instance_id')
                            or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                            or (bind_final_revision and observation.get('revision') != action_observation['revision'])):
                        raise RuntimeError('초기 자세 이동 중 최신 정지 상태를 확인하지 못했습니다.')
            check_cancel()
            ended = True
            end = controller.end_episode(lease)
            if end.get('stop', {}).get('confirmed') is not True or end.get('fault') is not None:
                fail_for_connection_fault(end.get('fault'))
                raise RuntimeError('초기 자세 도착 또는 정지 확인이 완료되지 않았습니다.')
            final, stop = controller.observe(), end['stop']
            # The resident's accepted action increments revision once on
            # admission and once on completion. end_episode increments it once
            # for stop and once for lease release. Bind the action's own
            # returned observation and the final one to those transitions.
            revision_changed = (bind_final_revision and
                (type(final.get('revision')) is not int
                 or final['revision'] != action_observation['revision'] + 2
                 or final.get('state') != 'stopped'))
            if (final.get('instance_id') != initial_observation.get('instance_id')
                    or final.get('sensor_valid') is not True or final.get('moving') is not False
                    or revision_changed
                    or final.get('sensor_sequence', -1) < stop.get('confirmed_sensor_sequence', float('inf'))
                    or final.get('observed_monotonic', -1) < stop.get('confirmed_sensor_monotonic', float('inf'))):
                raise RuntimeError('초기 자세 요청 뒤 최신 정지 상태를 확인하지 못했습니다.')
            return {'response': response, 'prior_stop': prior['stop'], 'arm_stop': end['stop'],
                    'final_observation': final}
        finally:
            if lease is not None and not ended:
                try:
                    controller.end_episode(lease)
                except Exception:
                    pass

    def _execute_arm_home(self, coordinator, admission, cancellation, *, recovery_before_admission):
        def move(cancel_home, run):
            controller = self._controller
            with self._text_request_lock, coordinator._lock:
                if (self._shutting_down or self.coordinator is not coordinator or controller is None
                        or coordinator.session.controller is not controller or self.mode not in ('mock', 'physical')
                        or (self.mode == 'physical' and (self._physical_initial_pose_required
                            or (self._physical_recovery_required and
                                (recovery_before_admission or not admission.get('target_request_id')))))):
                    raise RuntimeError('연결된 제어기와 복구 상태를 확인해 주세요.')
                scene_before = coordinator.snapshot()
                scene_state = scene_before.get('scene_state') or {}
                owner = self._last_healthy_status_owner
                gripper = self._last_healthy_gripper
                arm_only_scene = (self.mode == 'physical' and admission.get('target_request_id') is None
                    and scene_before.get('session_status') == 'registered'
                    and scene_before.get('stop_confirmation') != 'unconfirmed'
                    and scene_state.get('expected_scene') is not None
                    and scene_state.get('unresolved_transfer') is None
                    and owner is not None and owner[:3] == (controller, coordinator, self._registration_id)
                    and owner[3] == getattr(controller, 'instance_id', None)
                    and owner[4] == scene_before.get('registration_revision')
                    and gripper is not None and gripper[0] == owner and gripper[1:] == ('open', 'open'))
            def stop_home():
                try:
                    controller.stop('ARM_HOME_CANCELLED')
                except Exception:
                    pass
            with coordinator._lock:
                if cancel_home.is_set():
                    raise InterruptedError('원위치 요청을 취소했습니다.')
                run['cancel_home_action'] = stop_home
            try:
                outcome = self._move_controller_initial_pose(controller, cancel_home, owner='demo-arm-home')
                if cancel_home.is_set():
                    raise InterruptedError('원위치 동작이 중단되었습니다.')
                coordinator.session.note_arm_home(outcome['final_observation'])
                return {'request_id': admission['request_id'], 'status': 'arm_home_completed',
                        'stop_request': admission.get('stop_request'), 'arm_stop': outcome['arm_stop'],
                        'interrupted_task_state': ('unconfirmed' if admission.get('target_request_id') else None),
                        'physical_success': None}
            except BaseException as exc:
                if (arm_only_scene and isinstance(exc, Exception) and not isinstance(exc, InterruptedError)
                        and not cancel_home.is_set() and not cancellation.is_set()
                        and not coordinator.session.cancel.is_set()):
                    with self._text_request_lock, coordinator._lock:
                        current = coordinator.snapshot()
                        if (self._controller is controller and self.coordinator is coordinator
                                and self._registration_id == owner[2]
                                and current.get('registration_revision') == owner[4]
                                and coordinator._active is run and not run.get('stop_requested')
                                and not cancel_home.is_set() and not cancellation.is_set()
                                and not coordinator.session.cancel.is_set()
                                and not self.cancelled.is_set() and not self._shutting_down
                                and coordinator.session.current == scene_state['expected_scene']
                                and (current.get('scene_state') or {}).get('unresolved_transfer') is None):
                            with coordinator.session._scene_view_lock:
                                reason = getattr(exc, 'code', None) or type(exc).__name__
                                coordinator.session._pending_release_recovery = {
                                    'expected_scene': coordinator.session.current,
                                    'instance_id': owner[3], 'basis': 'arm_only'}
                                coordinator.session.events.append({'event': 'arm_home_interrupted',
                                    'request_id': admission['request_id'], 'reason': reason,
                                    'physical_success': None, 'replayed_commands': 0})
                            coordinator.session._halt('arm home did not complete')
                            self._set_physical_recovery_locked(True)
                            return {'request_id': admission['request_id'], 'status': 'arm_home_interrupted',
                                    'reason': reason, 'physical_success': None,
                                    'replayed_commands': 0}
                with coordinator.session._scene_view_lock:
                    coordinator.session._confirmed_current_scene = None
                coordinator.session._halt('arm home did not finish with confirmed stop')
                with coordinator._lock:
                    coordinator._stop_unconfirmed = True
                if self.mode == 'physical':
                    with self._text_request_lock:
                        self._set_physical_recovery_locked(True)
                raise
        return coordinator.execute_arm_home(admission['reservation'], move, cancel_event=cancellation)

    def _disconnect_controller_locked(self, *, forecast_closer=None):
        controller = self._controller
        if controller is None:
            return
        if self._status_read_recovery is not None:
            # A detached owner cannot later prove that the same resident
            # returned; retain recovery as an ordinary unresolved state.
            self._set_physical_recovery_locked(True)
        self._last_healthy_status_owner = None
        self._last_healthy_gripper = None
        self._status_read_observation_refresh = None
        if self.mode == 'physical' and self.coordinator is not None:
            state = self.coordinator.snapshot()
            if ((state.get('session_status') == 'uncertain' or state.get('stop_confirmation') == 'unconfirmed')
                    and self._physical_stop_cleared_session is not self.coordinator):
                self._set_physical_recovery_locked(True)
        if self.forecast is not None:
            if forecast_closer is None:
                self.forecast.close()
            else:
                forecast_closer(self.forecast)
            self.forecast = None
        controller._forecast = None
        result = controller.close()
        self._last_disconnect = result
        retained = result.get('controller_connection_retained') is True
        exited = (result.get('worker', {}).get('worker_exited') is True
                  or result.get('worker_alive') is False)
        if not retained and (result.get('closed') is not True or not exited):
            raise RuntimeError('제어기 종료가 확인되지 않았습니다. 연결 상태를 확인해 주세요.')
        self._controller = self._controller_initial_target = None
        self._controller_calibration = self._controller_profile = None
        self._controller_profile_path = None
        self._registered_calibration = None
        self._physical_stop_cleared_session = None

    def disconnect_controller(self):
        """End an idle owned connection and discard its registration and memory."""
        with self._text_request_lock:
            self._require_idle_locked()
            self._pending_physical_start = None
            self._disconnect_controller_locked()
            self.coordinator = self.sim = self.model_hash = None
            self._registration_id = None
        return self.connection_snapshot()

    def prepare_gripper(self, *, operation=None, physical_confirmed=False):
        """Explicitly command and confirm an open gripper on the selected controller."""
        operation = operation or _AuxiliaryOperation()
        with self._text_request_lock:
            self._require_idle_locked()
            controller = self._controller
            if self.mode not in ('mock', 'physical') or controller is None:
                raise RuntimeError('먼저 제어기를 연결해 주세요.')
            if self.mode == 'physical' and self._physical_recovery_required:
                raise RuntimeError('먼저 현장 상태를 확인하고 제어기 정지를 확인해 주세요.')
            if self.mode == 'physical' and self._physical_initial_pose_required:
                raise RuntimeError('초기 자세가 확인되지 않았습니다. 연결을 해제하고 다시 연결해 주세요.')
            if self.mode == 'physical' and physical_confirmed is not True:
                raise ValueError('실물 집게 열기 동작을 직접 확인해 주세요.')
            if operation.is_set():
                raise InterruptedError('집게 준비를 취소했습니다.')
            self._controller_preparation = (operation, controller)
        lease = None
        ended = False
        try:
            lease = controller.acquire('demo-gripper-prepare')['lease_id']
            if operation.is_set() or self.cancelled.is_set():
                raise InterruptedError('집게 준비를 취소했습니다.')
            observation = controller.observe(lease)
            if operation.is_set() or self.cancelled.is_set():
                raise InterruptedError('집게 준비를 취소했습니다.')
            response = controller.apply_action({'lease_id': lease, 'command_id': uuid.uuid4().hex,
                'observation_id': observation['observation_id'],
                'action': {'kind': 'set_gripper', 'state': 'open'}})
            end = controller.end_episode(lease)
            ended = True
            if (response.get('status') != 'gripper_reported' or response.get('applied') is not True
                    or response.get('gripper_reported_confirmed') is not True
                    or end.get('stop', {}).get('confirmed') is not True or end.get('fault') is not None):
                raise RuntimeError('집게 열기 또는 팔 정지 확인이 완료되지 않았습니다.')
            operation.commit(lambda: None, self.cancelled)
            return {'request_id': operation.request_id, 'status': 'prepared',
                    'gripper_reported_confirmed': True, 'arm_stop': end.get('stop'),
                    'physical_success': None}
        except BaseException:
            if self.mode == 'physical':
                with self._text_request_lock:
                    self._set_physical_recovery_locked(True)
            raise
        finally:
            if lease is not None and not ended:
                try:
                    controller.end_episode(lease)
                except Exception:
                    pass
            with self._text_request_lock:
                if self._controller_preparation and self._controller_preparation[0] is operation:
                    self._controller_preparation = None

    def stop_controller_preparation(self, *, target_request_id):
        """Cancel just the identified preparation; arm stop is not gripper proof."""
        with self._text_request_lock:
            entry = self._controller_preparation or self._controller_recovery or self._controller_connecting
            if entry is None or entry[0].request_id != target_request_id:
                return {'status': 'no_matching_active_request'}
            operation, controller = entry
            if not operation.cancel():
                return {'status': 'already_completed', 'target_request_id': target_request_id}
            if self.mode == 'physical':
                self._set_physical_recovery_locked(True)
        if controller is None:
            return {'status': 'stop_requested', 'target_request_id': target_request_id,
                    'arm_stop': {'requested': False, 'confirmed': False}, 'gripper_stop_confirmed': None}
        try:
            stop = controller.stop('PREPARATION_CANCELLED')
        except Exception as exc:
            stop = {'requested': True, 'confirmed': False, 'error': type(exc).__name__}
        return {'status': 'stop_requested', 'target_request_id': target_request_id,
                'arm_stop': stop, 'gripper_stop_confirmed': None}

    def recover_controller(self, *, layout_confirmed=False, confirm_stop=False, operation=None):
        """Explicitly retry an idle fault's stop and existing controller recovery checks."""
        if layout_confirmed is not True or type(confirm_stop) is not bool:
            raise ValueError('현재 배치를 직접 확인한 뒤 복구를 요청해 주세요.')
        operation = operation or _AuxiliaryOperation()
        with self._text_request_lock:
            self._require_idle_locked()
            controller = self._controller
            if controller is None or self.mode not in ('physical', 'mock'):
                raise RuntimeError('먼저 제어기를 연결해 주세요.')
            if operation.is_set():
                raise InterruptedError('제어기 복구를 취소했습니다.')
            self._controller_recovery = (operation, controller)
        try:
            status = controller.status()
            if status.get('active_command_id') is not None or status.get('lease_owner') is not None:
                raise RuntimeError('다른 제어 작업이 종료된 뒤 복구해 주세요.')
            if status.get('fault') is None:
                if confirm_stop:
                    if operation.is_set() or self.cancelled.is_set():
                        raise InterruptedError('제어기 복구를 취소했습니다.')
                    stopped = controller.stop('RECOVERY_STOP_REQUESTED')
                    if stopped.get('confirmed') is not True:
                        return {'request_id': operation.request_id, 'status': 'stop_unconfirmed',
                                'arm_stop': stopped, 'physical_success': None}
                    with self._text_request_lock:
                        if self.coordinator is not None:
                            with self.coordinator._lock:
                                self.coordinator._refresh_external_state()
                                self.coordinator.session._halt('controller_stop_confirmed')
                                self.coordinator._known_session_stamp = self.coordinator._session_stamp()
                                self.coordinator._bump()
                            self._registration_id = None
                    operation.commit(lambda: None, self.cancelled)
                    if controller.status().get('fault') is not None:
                        raise RuntimeError('정지 뒤 제어기 오류가 확인되어 별도 복구가 필요합니다.')
                    with self._text_request_lock:
                        if self.mode == 'physical':
                            self._set_physical_recovery_locked(False)
                            self._physical_stop_cleared_session = self.coordinator
                    return {'request_id': operation.request_id, 'status': 'stop_confirmed',
                            'arm_stop': stopped, 'physical_success': None}
                return {'request_id': operation.request_id, 'status': 'no_fault',
                        'message': '제어기 오류는 없지만 배치와 집게 준비는 따로 확인해야 합니다.'}
            # A cleared controller fault is not evidence that the scene or
            # prior learned memory remained valid through the fault.
            with self._text_request_lock:
                if self.coordinator is not None:
                    with self.coordinator._lock:
                        self.coordinator._refresh_external_state()
                        self.coordinator.session._halt('controller_fault_recovery')
                        self.coordinator._known_session_stamp = self.coordinator._session_stamp()
                        self.coordinator._bump()
                    self._registration_id = None
            stopped = None
            if self.mode == 'physical' and self._physical_recovery_required and not confirm_stop:
                raise ValueError('먼저 명시적으로 제어기 정지를 확인해 주세요.')
            if confirm_stop:
                observation = controller.observe()
                if operation.is_set() or self.cancelled.is_set():
                    raise InterruptedError('제어기 복구를 취소했습니다.')
                stopped = controller.confirm_idle_stop(observation['observation_id'])
                if stopped.get('confirmed') is not True:
                    return {'request_id': operation.request_id, 'status': 'stop_unconfirmed',
                            'arm_stop': stopped, 'physical_success': None}
            observation = controller.observe()
            if operation.is_set() or self.cancelled.is_set():
                raise InterruptedError('제어기 복구를 취소했습니다.')
            recovered = controller.recover(observation['observation_id'], True)
            operation.commit(lambda: None, self.cancelled)
            if stopped is not None and stopped.get('confirmed') is True:
                with self._text_request_lock:
                    if self.mode == 'physical':
                        self._set_physical_recovery_locked(False)
                        self._physical_stop_cleared_session = self.coordinator
            return {'request_id': operation.request_id, 'status': 'recovered',
                    'controller_recovery': recovered, 'arm_stop': stopped, 'physical_success': None}
        finally:
            with self._text_request_lock:
                if self._controller_recovery and self._controller_recovery[0] is operation:
                    self._controller_recovery = None

    def connection_snapshot(self):
        """Return a cached display view; never create a controller observation."""
        with self._text_request_lock:
            mode, controller, coordinator = self.mode, self._controller, self.coordinator
            registration_id = self._registration_id
            calibration = (self._registered_calibration if self._registration_id is not None
                           and self._registered_calibration is not None else self._controller_calibration)
            profile = self._controller_profile
            recovery_epoch = self._physical_recovery_epoch
            preparation = self._controller_preparation
            recovery = self._controller_recovery
            connecting = self._controller_connecting
            result = {'mode': mode, 'connection': 'not_applicable' if mode == 'virtual' else 'disconnected',
                      'connected': False, 'simulated': None if mode == 'virtual' else mode == 'mock',
                      'observation': None, 'controller': None, 'current_command': None,
                      'remaining_target_path_mm': [], 'path_source': None, 'scene_state': None,
                      'forecast': None,
                      'confirmed_applied_target': None,
                      'physical_recovery_required': self._physical_recovery_required,
                      'controller_reconnect_revision': self._controller_reconnect_revision,
                      'initial_pose_required': self._physical_initial_pose_required if mode == 'physical' else False,
                      'calibration': None, 'requirements': [], 'last_disconnect': self._last_disconnect,
                      'preparation': ({'request_id': preparation[0].request_id,
                                       'status': 'in_progress'} if preparation else None),
                      'recovery': ({'request_id': recovery[0].request_id,
                                    'status': 'in_progress'} if recovery else None),
                      'connecting': ({'request_id': connecting[0].request_id,
                                      'status': 'in_progress'} if connecting else None)}
            if mode == 'physical':
                result['requirements'] = ['명시적 실물 연결', '현재 배치와 제어기 준비 상태 확인',
                                          '마지막 적용 목표각과 열린 집게의 제어기 확인']
            elif mode == 'mock' and controller is None:
                result['requirements'] = ['모의 제어기 연결', '시작 배치 등록']
            if calibration is not None:
                result['calibration'] = {'calibration_id': calibration.calibration_id,
                    'coordinate_source': calibration.coordinate_source, 'verified': calibration.verified,
                    'slot_xy_mm': getattr(calibration, 'slot_xy_mm', None),
                    'table_z_mm': getattr(calibration, 'table_z_mm', None),
                    'workspace_verified': controller.config.workspace_verified,
                    'active_slots': list(profile.enabled_slots), 'max_height': profile.max_height}
            if controller is not None:
                result['ownership'] = ('resident_retained' if getattr(controller, 'attachment', None)
                                       is not None else 'service_owned')
        # Resident status is a read-only RPC. Never hold the service lock while
        # waiting for it, so a slow status response cannot delay a stop request.
        if controller is not None:
            try:
                result.update(controller.display_snapshot())
                status = result['controller']
                candidate = status.get('last_confirmed_applied_target')
                if (isinstance(candidate, dict) and candidate.get('instance_id') == status.get('instance_id')
                        and isinstance(candidate.get('command_id'), str)
                        and isinstance(candidate.get('joints_deg'), (list, tuple))
                        and len(candidate['joints_deg']) == 4 and result['observation'] is not None
                        and status.get('active_command_id') is None and status.get('lease_owner') is None
                        and status.get('fault') is None and status.get('worker_alive') is True
                        and status.get('fresh_feedback') is True):
                    result['confirmed_applied_target'] = candidate
                result['connected'] = bool(status.get('worker_alive') and status.get('fresh_feedback'))
                result['connection'] = ('fault' if status.get('fault') else
                                        'connected' if result['connected'] else 'disconnected')
                if result['observation'] is None:
                    result['requirements'].append('상주 제어기가 화면용 캐시 관측을 제공하지 않습니다. 소유자가 판본을 확인해야 합니다.')
            except Exception:
                result['connection'] = 'unknown'
                result['requirements'].append('제어기 상태를 받지 못했습니다. 연결과 장치 상태는 미확인입니다.')
        session_snapshot = None
        if coordinator is not None and mode != 'virtual':
            session_snapshot = coordinator.snapshot()
            result['scene_state'] = session_snapshot.get('scene_state')
            if (mode == 'physical' and self._physical_stop_cleared_session is not coordinator
                    and (session_snapshot.get('session_status') == 'uncertain'
                         or session_snapshot.get('stop_confirmation') == 'unconfirmed')):
                result['physical_recovery_required'] = True
        with self._text_request_lock:
            changed = self.mode != mode or self._controller is not controller or self.coordinator is not coordinator
            if not changed and mode == 'physical':
                if recovery_epoch == self._physical_recovery_epoch:
                    session_uncertain = (isinstance(session_snapshot, dict)
                        and self._physical_stop_cleared_session is not coordinator
                        and (session_snapshot.get('session_status') == 'uncertain'
                             or session_snapshot.get('stop_confirmation') == 'unconfirmed'))
                    unavailable = (not connecting and controller is not None
                                   and result['connection'] in ('fault', 'unknown', 'disconnected'))
                    if (unavailable and coordinator is None and registration_id is None
                            and self._unregistered_idle_controller is controller
                            and not self._physical_recovery_required and self._controller_preparation is None
                            and self._controller_recovery is None and not self._registration_requests):
                        self._unregistered_interrupted_controller = controller
                    status_owner = self._last_healthy_status_owner
                    same_registration = (registration_id is not None and self._registration_id == registration_id
                        and status_owner is not None and status_owner[:3] == (controller, coordinator, registration_id)
                        and isinstance(session_snapshot, dict)
                        and status_owner[4] == session_snapshot.get('registration_revision'))
                    idle_stale_owner = None
                    stale_status = result.get('controller') or {}
                    waiter = self._status_read_waiter
                    waiting_ticket = (waiter[1] if waiter is not None and waiter[0] is coordinator
                                      and waiter[3] == 'ticket' else None)
                    waiting_home = (waiter[1] if waiter is not None and waiter[0] is coordinator
                                    and waiter[3] == 'home' else None)
                    waiting_request = (waiter[2] if waiting_ticket is not None or waiting_home is not None else None)
                    connection_fault = (isinstance(stale_status.get('fault'), dict)
                        and stale_status['fault'].get('code') in ('DISCONNECTED', 'SENSOR_STALE')
                        and (stale_status.get('capabilities') or {}).get('expected_layout_recovery') is True)
                    auto_idle = self._idle_connection_recovery
                    # A stop confirmation initiated by our own recovery may
                    # temporarily appear as an active controller operation.
                    auto_stopping = (auto_idle is not None and auto_idle[1:] == (controller, coordinator, status_owner)
                                     and stale_status.get('active_command_id') == 'idle-stop-confirmation')
                    if (unavailable and (result['connection'] == 'disconnected' or connection_fault) and same_registration
                            and stale_status.get('worker_alive') is True
                            and (connection_fault or (stale_status.get('fresh_feedback') is False
                                 and 'fault' in stale_status and stale_status['fault'] is None))
                            and 'active_command_id' in stale_status
                            and (stale_status['active_command_id'] is None or auto_stopping)
                            and 'lease_owner' in stale_status and stale_status['lease_owner'] is None
                            and stale_status.get('instance_id') == status_owner[3]
                            and not self._shutting_down and not self.cancelled.is_set()
                            and not self._registration_requests
                            and self._controller_preparation is None
                            and (self._controller_recovery is None or auto_idle is not None)):
                        with coordinator._lock:
                            fresh_scene = coordinator.snapshot()
                            scene = fresh_scene.get('scene_state') or {}
                            home = coordinator._arm_home_reservation
                            # Captured or admitted language requests have not
                            # acquired the arm. Their executor waits below if
                            # feedback is stale; they are not failed motion.
                            unstarted_home = (home is not None and not home['claimed']
                                              and home['target_request_id'] is None)
                            if (coordinator._active is None and (home is None or unstarted_home)
                                    and not coordinator.session._lock.locked()
                                    and not coordinator.session.cancel.is_set()
                                    and fresh_scene.get('session_status') == 'registered'
                                    and fresh_scene.get('registration_revision') == status_owner[4]
                                    and fresh_scene.get('stop_confirmation') != 'unconfirmed'
                                    and scene.get('expected_scene') is not None
                                    and scene.get('unresolved_transfer') is None):
                                idle_stale_owner = status_owner
                    if session_uncertain or (unavailable and result['connection'] not in ('unknown', 'disconnected')
                                             and idle_stale_owner is None):
                        self._set_physical_recovery_locked(True)
                    elif unavailable:
                        # A stale connection is recoverable only when the
                        # complete registered session was idle when lost.
                        self._set_physical_recovery_locked(True,
                            status_read_owner=(status_owner if same_registration and result['connection'] == 'unknown'
                                               else idle_stale_owner))
                    elif self._status_read_recovery is not None:
                        status = result.get('controller') or {}
                        observation = result.get('observation') or {}
                        owner = (controller, coordinator, registration_id, status.get('instance_id'),
                                 (session_snapshot or {}).get('registration_revision'))
                        previous_gripper = (self._last_healthy_gripper[1:]
                            if self._last_healthy_gripper is not None
                            and self._last_healthy_gripper[0] == owner else None)
                        reported_gripper = ((observation.get('gripper_command'), observation.get('gripper_reported_state'))
                            if observation.get('gripper_reported_valid') is True else None)
                        if (result['connection'] == 'connected' and owner != self._status_read_recovery):
                            self._set_physical_recovery_locked(True)
                        elif (coordinator is not None and coordinator.session.cancel.is_set()):
                            self._set_physical_recovery_locked(True)
                        elif (reported_gripper is not None and previous_gripper is not None
                              and reported_gripper != previous_gripper):
                            self._set_physical_recovery_locked(True)
                        elif (owner == self._status_read_recovery and result['connection'] == 'connected'
                                and status.get('fault') is None and status.get('active_command_id') is None
                                and status.get('lease_owner') is None and observation.get('sensor_valid') is True
                                and observation.get('moving') is False and reported_gripper is not None and coordinator is not None
                                and not self._shutting_down and not self.cancelled.is_set()
                                and self._controller_recovery is None and self._controller_preparation is None
                                and self._controller_connecting is None):
                            with coordinator._lock:
                                fresh_scene = coordinator.snapshot()
                                scene = fresh_scene.get('scene_state') or {}
                                waiting_only = (waiting_ticket is not None
                                    and set(coordinator._pending) == {waiting_ticket}
                                    and coordinator._pending[waiting_ticket]['program']['request_id'] == waiting_request)
                                home = coordinator._arm_home_reservation
                                waiting_home_only = (waiting_home is not None and home is not None
                                    and home['id'] == waiting_home and home['request_id'] == waiting_request
                                    and not home['claimed'])
                                if (coordinator._active is None and (home is None or waiting_home_only)
                                        and (not coordinator._pending or waiting_only) and not coordinator._interpretations
                                        and not coordinator.session.cancel.is_set()
                                        and fresh_scene.get('session_status') == 'registered'
                                        and fresh_scene.get('registration_revision') == owner[4]
                                        and fresh_scene.get('stop_confirmation') != 'unconfirmed'
                                        and scene.get('expected_scene') is not None
                                        and scene.get('unresolved_transfer') is None):
                                    self._set_physical_recovery_locked(False)
                    status = result.get('controller') or {}
                    observation = result.get('observation') or {}
                    if (coordinator is None and registration_id is None
                            and self._unregistered_interrupted_controller is controller
                            and self._physical_connection_prepared and result['connection'] == 'connected'
                            and status.get('fault') is None and status.get('active_command_id') is None
                            and status.get('lease_owner') is None and observation.get('sensor_valid') is True
                            and observation.get('moving') is False and self._controller_recovery is None
                            and self._controller_preparation is None and self._controller_connecting is None
                            and not self._registration_requests and not self.cancelled.is_set()):
                        self._set_physical_recovery_locked(False)
                        self._physical_initial_pose_required = False
                        self._unregistered_interrupted_controller = None
                        self._controller_reconnect_revision += 1
                    if (coordinator is None and registration_id is None and not self._physical_recovery_required
                            and self._physical_connection_prepared and result['connection'] == 'connected'
                            and status.get('fault') is None and status.get('active_command_id') is None
                            and status.get('lease_owner') is None and observation.get('sensor_valid') is True
                            and observation.get('moving') is False):
                        self._unregistered_idle_controller = controller
                    if (not self._physical_recovery_required and registration_id is not None
                            and self._registration_id == registration_id and coordinator is not None
                            and result['connection'] == 'connected' and status.get('fault') is None
                            and isinstance(status.get('instance_id'), str)
                            and isinstance(session_snapshot, dict)
                            and observation.get('sensor_valid') is True):
                        self._last_healthy_status_owner = (controller, coordinator, registration_id,
                            status['instance_id'], session_snapshot.get('registration_revision'))
                        if (status.get('active_command_id') is None and status.get('lease_owner') is None
                                and observation.get('moving') is False
                                and observation.get('gripper_reported_valid') is True):
                            self._last_healthy_gripper = (self._last_healthy_status_owner,
                                observation.get('gripper_command'), observation.get('gripper_reported_state'))
                        else:
                            self._last_healthy_gripper = None
                else:
                    # Do not let a slow pre-recovery status overwrite the
                    # newer stop confirmation or present its old pose.
                    result.update(connection='unknown', connected=False, observation=None,
                                  current_command=None, remaining_target_path_mm=[], forecast=None,
                                  confirmed_applied_target=None)
                    result['requirements'].append('복구 상태가 바뀌어 현재 연결 상태를 다시 확인하고 있습니다.')
                result['physical_recovery_required'] = self._physical_recovery_required
                result['status_read_recovery_pending'] = self._status_read_recovery is not None
                if self._physical_recovery_required:
                    scene = result.get('scene_state') or {}
                    pending_release = (bool(scene.get('connection_recovery_pending'))
                        or (scene.get('unresolved_transfer') or {}).get('release_expected') is True)
                    result['requirements'].append('제어기 상태를 다시 확인하고 있습니다.'
                        if result['status_read_recovery_pending'] else
                        '제어기 수신 복구를 기다리고 있습니다.' if pending_release else
                        '현장 상태를 확인하고 정지 확인을 완료한 뒤 집게와 배치를 다시 준비해 주세요.')
        status = result.get('controller') or {}
        # A still-connected transport with a temporarily stale sample may
        # deliver the next sample without replacing its controller. If the
        # transport itself closed or the resident is unreachable, reconnect.
        ready_to_replace = (result.get('connection') in ('fault', 'unknown', 'disconnected')
                            and (result.get('connection') in ('fault', 'unknown')
                                 or status.get('transport_connected') is not True))
        if not changed and mode == 'physical' and controller is not None and coordinator is not None:
            self._start_idle_connection_recovery(controller, coordinator, result.get('controller'))
            self._start_expected_connection_recovery(controller, coordinator, result.get('controller'))
            if ready_to_replace:
                self._start_controller_replacement(controller, coordinator, result.get('controller'))
        elif (not changed and mode == 'physical' and controller is not None and coordinator is None
              and ready_to_replace):
            self._start_unregistered_controller_replacement(controller, result.get('controller'))
        if connecting and not changed:
            result.update(connection='connecting', connected=False, confirmed_applied_target=None)
        return self.connection_snapshot() if changed else result

    def _start_unregistered_controller_replacement(self, controller, status):
        """Restore a connection lost before registration without moving the arm."""
        fault = status.get('fault') if isinstance(status, dict) else None
        if isinstance(fault, dict) and fault.get('code') not in ('DISCONNECTED', 'SENSOR_STALE'):
            return
        with self._text_request_lock:
            if (self.mode != 'physical' or self._controller is not controller or self.coordinator is not None
                    or self._registration_id is not None or self._shutting_down or self.cancelled.is_set()
                    or self._controller_profile_path is None or self._controller_recovery is not None
                    or self._controller_connecting is not None or self._controller_preparation is not None
                    or self._registration_requests or time.monotonic() - self._replacement_last_attempt < 3.):
                return
            operation = _AuxiliaryOperation()
            profile_path = self._controller_profile_path
            self._unregistered_replacement = (operation, controller)
            self._controller_recovery = (operation, controller)
            self._replacement_last_attempt = time.monotonic()

        def reconnect():
            replacement = None
            try:
                from .demo_controller import physical_controller
                replacement, calibration, profile = physical_controller(profile_path, force_reconnect=True,
                                                                          previous_controller=controller)
                status = replacement.status()
                observation = replacement.observe()
                if not (_controller_status_idle(status, replacement.instance_id)
                        and _controller_observation_stationary(observation, replacement.instance_id)):
                    return
                with self._text_request_lock:
                    if (operation.is_set() or self.cancelled.is_set() or self._shutting_down
                            or self._unregistered_replacement != (operation, controller)
                            or self._controller is not controller or self.coordinator is not None
                            or self._registration_id is not None or self._registration_requests
                            or profile != self._controller_profile or calibration != self._controller_calibration):
                        return
                    self._controller = replacement
                    self._controller_calibration, self._controller_profile = calibration, profile
                    self._physical_connection_uncertain = False
                    self._physical_initial_pose_required = not self._physical_connection_prepared
                    if self._physical_connection_prepared:
                        self._set_physical_recovery_locked(False)
                    self._controller_reconnect_revision += 1
                    replacement = None
            except Exception:
                # The next screen refresh can try again; no movement is resent.
                pass
            finally:
                with self._text_request_lock:
                    if self._unregistered_replacement == (operation, controller):
                        self._unregistered_replacement = None
                    if self._controller_recovery == (operation, controller):
                        self._controller_recovery = None
                if replacement is not None:
                    replacement.close()
        threading.Thread(target=reconnect, daemon=True, name='replace-unregistered-controller').start()

    def _replacement_waiter_valid_locked(self, coordinator):
        """Allow only an unstarted request already waiting for this connection."""
        if coordinator._interpretations:
            return False
        waiter = self._status_read_waiter
        home = coordinator._arm_home_reservation
        if waiter is None:
            return not coordinator._pending and home is None
        if waiter[0] is not coordinator:
            return False
        _, ticket, request_id, kind = waiter
        if kind == 'ticket':
            entry = coordinator._pending.get(ticket)
            return (home is None and set(coordinator._pending) == {ticket} and entry is not None
                    and entry['program']['request_id'] == request_id)
        if kind == 'home':
            return (not coordinator._pending and home is not None and home['id'] == ticket
                    and home['request_id'] == request_id and not home['claimed']
                    and home['target_request_id'] is None)
        return False

    def _start_controller_replacement(self, controller, coordinator, status, *, operation=None):
        """Replace a dead connection after its session has stopped executing.

        The model, symbolic memory and expected scene stay in the same session.
        No previous operation is submitted again, and an unresolved transfer is
        not treated as completed merely because a new controller responds.
        """
        fault = status.get('fault') if isinstance(status, dict) else None
        if isinstance(fault, dict) and fault.get('code') not in ('DISCONNECTED', 'SENSOR_STALE'):
            return
        with self._text_request_lock, coordinator._lock:
            session = coordinator.session
            if (self.mode != 'physical' or self._controller is not controller
                    or self.coordinator is not coordinator or self._shutting_down or self.cancelled.is_set()
                    or self._controller_profile_path is None or self._registration_id is None
                    or self._controller_recovery is not None or self._controller_connecting is not None
                    or self._controller_preparation is not None or self._registration_requests
                    or coordinator._active is not None or not self._replacement_waiter_valid_locked(coordinator)
                    or session.cancel.is_set() or time.monotonic() - self._replacement_last_attempt < 3.
                    or not session._lock.acquire(blocking=False)):
                return
            operation = operation or _AuxiliaryOperation()
            generation = coordinator._generation
            registration = self._registration_id
            profile_path = self._controller_profile_path
            self._replacement_started = (operation, controller, coordinator)
            self._controller_recovery = (operation, controller)
            self._replacement_last_attempt = time.monotonic()

        def replace_connection():
            replacement = None
            try:
                from .demo_controller import physical_controller
                replacement, calibration, profile = physical_controller(profile_path, force_reconnect=True,
                                                                          previous_controller=controller)
                current = replacement.status()
                observation = replacement.observe()
                if not (_controller_status_idle(current, replacement.instance_id)
                        and _controller_observation_stationary(observation, replacement.instance_id)):
                    return
                with self._text_request_lock, coordinator._lock:
                    if (operation.is_set() or self.cancelled.is_set() or self._shutting_down
                            or self._replacement_started != (operation, controller, coordinator)
                            or self._controller is not controller or self.coordinator is not coordinator
                            or self._registration_id != registration or coordinator._generation != generation
                            or coordinator._active is not None
                            or not self._replacement_waiter_valid_locked(coordinator)):
                        return
                    # A changed calibration/profile is not silently imposed on
                    # an existing scene or its model inputs.
                    if (profile != self._controller_profile or calibration != self._controller_calibration):
                        return
                    session.rebind_controller(replacement, observation)
                    if session.recover_expected_after_disconnect(observation, replacement=True):
                        # This clears only the old arm-stop latch. The known
                        # expected scene stays expected; no physical object
                        # success is inferred and no command is repeated.
                        coordinator._stop_unconfirmed = False
                    forecast = self.forecast
                    if forecast is not None:
                        controller._forecast = None
                        replacement._forecast = forecast
                        if hasattr(forecast, 'controller'):
                            forecast.controller = replacement
                        try:
                            forecast.invalidate('controller_reconnected')
                        except Exception:
                            pass
                    self._controller = replacement
                    self._controller_calibration, self._controller_profile = calibration, profile
                    self._last_healthy_status_owner = None
                    self._last_healthy_gripper = None
                    self._status_read_observation_refresh = None
                    self._automatic_recovered = None
                    coordinator._known_session_stamp = coordinator._session_stamp()
                    coordinator._expected_memory_stamp = coordinator._memory_stamp()
                    # The admitted request has not run and its symbolic
                    # context did not change. Keep its original ticket valid
                    # so the user need not repeat it after reconnecting.
                    if self._status_read_waiter is None:
                        coordinator._bump()
                    if session.status == 'registered' and not coordinator._stop_unconfirmed:
                        self._set_physical_recovery_locked(False)
                        self._status_read_observation_refresh = None
                    else:
                        self._set_physical_recovery_locked(True)
                    self._physical_initial_pose_required = False
                    self._physical_connection_uncertain = False
                    replacement = None
            except Exception:
                # A live owner or unknown command outcome is not proof that it
                # is safe to start a second controller. A later display refresh
                # can try again after the old owner has exited.
                pass
            finally:
                session._lock.release()
                with self._text_request_lock:
                    if self._replacement_started == (operation, controller, coordinator):
                        self._replacement_started = None
                    if self._controller_recovery == (operation, controller):
                        self._controller_recovery = None
                if replacement is not None:
                    replacement.close()
        worker = threading.Thread(target=replace_connection, daemon=True, name='replace-controller-connection')
        worker.start()
        return worker

    def _start_idle_connection_recovery(self, controller, coordinator, status):
        """Clear a same-resident connection fault without replaying a command."""
        if (not isinstance(status, dict) or not isinstance(status.get('fault'), dict)
                or status['fault'].get('code') not in ('DISCONNECTED', 'SENSOR_STALE')
                or (status.get('capabilities') or {}).get('expected_layout_recovery') is not True
                or status.get('worker_alive') is not True or status.get('fresh_feedback') is not True
                or status.get('active_command_id') is not None or status.get('lease_owner') is not None):
            return
        with self._text_request_lock, coordinator._lock:
            owner = self._status_read_recovery
            scene = coordinator.snapshot()
            home = coordinator._arm_home_reservation
            if (owner is None or owner[:3] != (controller, coordinator, self._registration_id)
                    or status.get('instance_id') != owner[3] or scene.get('registration_revision') != owner[4]
                    or self._controller is not controller or self.coordinator is not coordinator
                    or self._shutting_down or self.cancelled.is_set() or self._controller_recovery is not None
                    or self._controller_connecting is not None or self._controller_preparation is not None
                    or self._registration_requests or coordinator._active is not None
                    or (home is not None and (home['claimed'] or home['target_request_id'] is not None))
                    or coordinator.session._lock.locked() or coordinator.session.cancel.is_set()
                    or scene.get('session_status') != 'registered' or scene.get('stop_confirmation') == 'unconfirmed'
                    or (scene.get('scene_state') or {}).get('expected_scene') is None
                    or (scene.get('scene_state') or {}).get('unresolved_transfer') is not None
                    or time.monotonic() - self._automatic_recovery_last_started < 1.):
                return
            stamp = coordinator._session_stamp()
            operation = _AuxiliaryOperation()
            self._controller_recovery = (operation, controller)
            self._automatic_recovery = (operation, controller, coordinator)
            self._idle_connection_recovery = (operation, controller, coordinator, owner)
            self._automatic_recovery_last_started = time.monotonic()

        def still_current():
            with self._text_request_lock, coordinator._lock:
                home = coordinator._arm_home_reservation
                return (not operation.is_set() and not self.cancelled.is_set() and not self._shutting_down
                    and self._idle_connection_recovery == (operation, controller, coordinator, owner)
                    and self._status_read_recovery == owner and self._controller is controller
                    and self.coordinator is coordinator and self._registration_id == owner[2]
                    and coordinator.snapshot().get('registration_revision') == owner[4]
                    and coordinator._session_stamp() == stamp and coordinator._active is None
                    and (home is None or (not home['claimed'] and home['target_request_id'] is None))
                    and not coordinator.session._lock.locked() and not coordinator.session.cancel.is_set())

        def restore():
            stop_attempted = False
            try:
                def valid_observation(observation):
                    if (not isinstance(observation, dict) or not still_current()
                            or observation.get('instance_id') != owner[3]
                            or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                            or observation.get('gripper_reported_valid') is not True
                            or observation.get('gripper_feedback_fresh') is not True):
                        return False
                    previous = (self._last_healthy_gripper[1:] if self._last_healthy_gripper is not None
                                and self._last_healthy_gripper[0] == owner else None)
                    if (observation.get('gripper_command') != 'open'
                            or observation.get('gripper_reported_state') != 'open'
                            or (previous is not None and previous != ('open', 'open'))
                            or (observation.get('device_diagnostics') or {}).get('firmware_warning_raw') != 0):
                        with self._text_request_lock:
                            if self._status_read_recovery == owner:
                                self._set_physical_recovery_locked(True)
                        return False
                    return True

                observation = controller.observe()
                if not valid_observation(observation):
                    return
                if (status.get('last_stop') or {}).get('confirmed') is not True:
                    stop_attempted = True
                    stopped = controller.confirm_idle_stop(observation['observation_id'])
                    if not isinstance(stopped, dict) or stopped.get('confirmed') is not True:
                        raise RuntimeError('automatic stop confirmation failed')
                    stop_attempted = False
                    observation = controller.observe()
                if not valid_observation(observation):
                    return
                recovered = controller.recover(observation['observation_id'], False, layout_basis='expected')
                if (not isinstance(recovered, dict) or recovered.get('recovered') is not True
                        or recovered.get('layout_confirmed') is not False or recovered.get('layout_basis') != 'expected'
                        or recovered.get('replayed_commands') != 0 or recovered.get('device_commands_sent') != 0):
                    raise RuntimeError('automatic recovery was not confirmed')
                # The next healthy status clears the temporary interruption
                # and refreshes only the stationary arm reference when needed.
            except Exception as exc:
                # Never repeat a stop whose result was unconfirmed, and do not
                # erase a controller's explicit review requirement.
                if stop_attempted or getattr(exc, 'code', None) in ('RECOVERY_REVIEW_REQUIRED', 'DEVICE_WARNING',
                                                                   'CONTROLLER_CHANGED'):
                    with self._text_request_lock:
                        if self._status_read_recovery == owner:
                            self._set_physical_recovery_locked(True)
            finally:
                with self._text_request_lock:
                    if self._idle_connection_recovery == (operation, controller, coordinator, owner):
                        self._idle_connection_recovery = None
                    if self._automatic_recovery == (operation, controller, coordinator):
                        self._automatic_recovery = None
                    if self._controller_recovery == (operation, controller):
                        self._controller_recovery = None
        threading.Thread(target=restore, daemon=True, name='idle-connection-recovery').start()

    def _start_expected_connection_recovery(self, controller, coordinator, status):
        """Reuse the current resident when the expected scene survived the interruption."""
        if not isinstance(status, dict):
            return
        fault = status.get('fault')
        pending_scene = getattr(coordinator.session, '_pending_release_recovery', None)
        replacement_pending = (isinstance(pending_scene, dict) and pending_scene.get('replacement') is True
                               and pending_scene.get('instance_id') == status.get('instance_id'))
        already_recovered = self._automatic_recovered == (controller, coordinator)
        ready_without_fault = (fault is None and ((status.get('last_stop') or {}).get('confirmed') is True
                                                  or replacement_pending))
        if ((not already_recovered and
             not ready_without_fault and
             (not isinstance(fault, dict) or fault.get('code') not in ('DISCONNECTED', 'SENSOR_STALE')))
                or (already_recovered and fault is not None)
                or status.get('worker_alive') is not True or status.get('fresh_feedback') is not True
                or status.get('active_command_id') is not None or status.get('lease_owner') is not None
                or (fault is not None and
                    (status.get('capabilities') or {}).get('expected_layout_recovery') is not True)):
            return
        with self._text_request_lock:
            if (self._controller is not controller or self.coordinator is not coordinator or self._shutting_down
                    or self.cancelled.is_set() or self._controller_recovery is not None
                    or self._controller_connecting is not None or self._controller_preparation is not None
                    or time.monotonic() - self._automatic_recovery_last_started < 1.):
                return
            with coordinator._lock:
                session = coordinator.session
                if (coordinator._active is not None or coordinator._arm_home_reservation is not None
                        or (coordinator._stop_unconfirmed and not replacement_pending) or session.status != 'uncertain'
                        or session.cancel.is_set() or not isinstance(getattr(session, '_pending_release_recovery', None), dict)
                        or session._pending_release_recovery.get('instance_id') != status.get('instance_id')
                        or not session._lock.acquire(blocking=False)):
                    return
                generation = coordinator._generation
                operation = _AuxiliaryOperation()
                self._controller_recovery = (operation, controller)
                self._automatic_recovery = (operation, controller, coordinator)
                self._automatic_recovery_last_started = time.monotonic()
        def restore():
            try:
                observation = controller.observe()
                if operation.is_set() or self.cancelled.is_set():
                    return
                if (not already_recovered and not replacement_pending
                        and (status.get('last_stop') or {}).get('confirmed') is not True):
                    stopped = controller.confirm_idle_stop(observation['observation_id'])
                    if stopped.get('confirmed') is not True or operation.is_set() or self.cancelled.is_set():
                        return
                    observation = controller.observe()
                if not already_recovered:
                    if (observation.get('instance_id') != status.get('instance_id')
                            or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                            or observation.get('gripper_command') not in (('open', None) if replacement_pending else ('open',))
                            or observation.get('gripper_reported_state') != 'open'
                            or observation.get('gripper_reported_valid') is not True
                            or observation.get('gripper_feedback_fresh') is not True
                            or (observation.get('device_diagnostics') or {}).get('firmware_warning_raw') != 0):
                        return
                    if not ready_without_fault:
                        recovered = controller.recover(observation['observation_id'], layout_confirmed=False,
                                                       layout_basis='expected')
                        if (not isinstance(recovered, dict) or recovered.get('recovered') is not True
                                or recovered.get('layout_confirmed') is not False or recovered.get('layout_basis') != 'expected'
                                or recovered.get('replayed_commands') != 0 or recovered.get('device_commands_sent') != 0
                                or operation.is_set() or self.cancelled.is_set()):
                            return
                    with self._text_request_lock:
                        if operation.is_set() or self.cancelled.is_set():
                            return
                        self._automatic_recovered = (controller, coordinator)
                fresh = controller.observe()
                current_status = controller.status()
                if (not _controller_status_idle(current_status, status.get('instance_id'))
                        or ((current_status.get('last_stop') or {}).get('confirmed') is not True
                            and not replacement_pending)):
                    return
                with self._text_request_lock:
                    with coordinator._lock:
                        if (operation.is_set() or self.cancelled.is_set() or self._shutting_down
                                or self._controller is not controller or self.coordinator is not coordinator
                                or self._automatic_recovery != (operation, controller, coordinator)
                                or coordinator._generation != generation or coordinator._active is not None
                                or (coordinator._stop_unconfirmed and not replacement_pending)
                                or coordinator.session.cancel.is_set()):
                            return
                        if coordinator.session.recover_expected_after_disconnect(fresh, replacement=replacement_pending):
                            if replacement_pending:
                                coordinator._stop_unconfirmed = False
                            coordinator._known_session_stamp = coordinator._session_stamp()
                            coordinator._expected_memory_stamp = coordinator._memory_stamp()
                            coordinator._bump()
                            self._set_physical_recovery_locked(False)
                            self._automatic_recovered = None
            except Exception:
                # A failed or unsupported recovery leaves the prior fault and
                # uncertain session visible; it never retries a motion command.
                pass
            finally:
                session._lock.release()
                with self._text_request_lock:
                    if self._automatic_recovery == (operation, controller, coordinator):
                        self._automatic_recovery = None
                    if self._controller_recovery == (operation, controller):
                        self._controller_recovery = None
        threading.Thread(target=restore, daemon=True, name='expected-connection-recovery').start()

    def register(self, layout, model_path=BEST_CLOSE_STUDENT, expected_hash=None, *,
                 finger_open_profile='auto', closed_move_profile='auto', control_timing_profile='auto', operation=None,
                 initial_applied_target_deg=None, controller_ready_confirmed=False,
                 initial_applied_target_evidence=None, slot_xy_mm=None, table_z_mm=None):
        """Register a declared scene without commanding a controller."""
        operation = operation or _AuxiliaryOperation()
        with self._text_request_lock:
            if self._shutting_down:
                raise RuntimeError('서비스가 종료 중입니다.')
            if self._registration_requests:
                raise RuntimeError('다른 배치 등록이 진행 중입니다.')
            self._registration_requests[operation.request_id] = operation
        try:
            if self.mode != 'physical' and (slot_xy_mm is not None or table_z_mm is not None):
                raise ValueError('자리 좌표와 탁상 높이는 실물 등록에서만 입력할 수 있습니다.')
            if self.mode != 'virtual':
                return self._register_controller(layout, model_path, expected_hash, operation,
                    initial_applied_target_deg=initial_applied_target_deg,
                    controller_ready_confirmed=controller_ready_confirmed,
                    initial_applied_target_evidence=initial_applied_target_evidence,
                    slot_xy_mm=slot_xy_mm, table_z_mm=table_z_mm)
            if (initial_applied_target_deg is not None or controller_ready_confirmed is not False
                    or initial_applied_target_evidence is not None):
                raise ValueError('가상 등록에는 실물 제어기 확인을 사용할 수 없습니다.')
            return self._register(layout, model_path, expected_hash, finger_open_profile=finger_open_profile,
                                  closed_move_profile=closed_move_profile, control_timing_profile=control_timing_profile,
                                  operation=operation)
        finally:
            with self._text_request_lock:
                self._registration_requests.pop(operation.request_id, None)

    def _registration_cancelled(self, cancelled):
        with self._text_request_lock:
            return self._shutting_down or cancelled.is_set()

    @staticmethod
    def _stationary_registration_reference(controller, expected=None):
        """Read a stationary initial policy reference without claiming a prior target."""
        from .native_physical_session import _applied_target

        try:
            first = controller.observe()
            view = controller.display_snapshot()
            second = controller.observe()
        except Exception as exc:
            raise RuntimeError('제어기의 정지 관절 관측을 읽지 못했습니다. 연결 상태를 확인해 주세요.') from exc
        status = view.get('controller') if isinstance(view, dict) else None
        display = view.get('observation') if isinstance(view, dict) else None
        if not isinstance(status, dict) or not isinstance(display, dict):
            raise RuntimeError('제어기의 최신 상태를 확인할 수 없습니다.')
        last_stop = status.get('last_stop')
        never_stopped_idle = (status.get('state') == 'idle' and isinstance(last_stop, dict)
                              and last_stop.get('requested') is False)
        if (status.get('worker_alive') is not True or status.get('fresh_feedback') is not True
                or status.get('fault') is not None or status.get('active_command_id') is not None
                or status.get('lease_owner') is not None or status.get('state') not in ('idle', 'stopped')
                or not isinstance(last_stop, dict)
                or (not never_stopped_idle and last_stop.get('confirmed') is not True)
                or display.get('sensor_valid') is not True or display.get('moving') is not False):
            raise RuntimeError('정지된 제어기의 신선한 관측과 오류 없는 상태를 확인해 주세요.')
        instance = status.get('instance_id')
        if (not isinstance(instance, str) or not instance.strip()
                or getattr(controller, 'instance_id', None) != instance):
            raise RuntimeError('제어기 연결이 바뀌어 정지 관측을 사용할 수 없습니다.')
        limits = controller.config.limits
        values = []
        for observation in (first, second):
            if not isinstance(observation, dict):
                raise RuntimeError('관절 관측을 확인할 수 없습니다.')
            diagnostics = observation.get('device_diagnostics')
            warning = diagnostics.get('firmware_warning_raw') if isinstance(diagnostics, dict) else None
            warning_clear = (type(warning) is int and warning == 0) or (
                warning is None and isinstance(diagnostics, dict)
                and controller.config.backend == 'mock' and observation.get('simulated') is True)
            if (not _fresh_stationary_observation_valid(observation, instance, limits)
                    or observation.get('schema_version') != 'PolicyObservation.phase1'
                    or observation.get('state') not in ('idle', 'stopped')
                    or not warning_clear
                    or observation.get('gripper_command') != 'open'
                    or observation.get('gripper_reported_state') != 'open'):
                raise RuntimeError('관절·정지·집게 또는 장치 경고의 최신 확인이 필요합니다.')
            try:
                joints = _applied_target(observation.get('joints_deg'), require_quantized=False)
            except (TypeError, ValueError):
                raise RuntimeError('유효한 범위의 관절 관측을 확인해 주세요.') from None
            values.append(joints)
        if (first['revision'] != second['revision']
                or second['sensor_sequence'] < first['sensor_sequence']
                or any(abs(a-b) > limits.observation_joint_drift_deg for a, b in zip(*values))):
            raise RuntimeError('관측 중 제어 상태 또는 관절 위치가 바뀌었습니다. 다시 확인해 주세요.')
        reference = {'instance_id': instance, 'revision': first['revision'], 'joints_deg': values[0]}
        if expected is not None and (reference['instance_id'] != expected['instance_id']
                or reference['revision'] != expected['revision']
                or any(abs(a-b) > limits.observation_joint_drift_deg
                       for a, b in zip(reference['joints_deg'], expected['joints_deg']))):
            raise RuntimeError('등록 중 제어기나 정지 관측이 바뀌었습니다. 배치를 다시 등록해 주세요.')
        return reference

    @staticmethod
    def _validate_registration_target_evidence(view, target, evidence, limits):
        """Bind an auto-filled target to a current, stopped controller snapshot."""
        from .native_physical_session import _applied_target

        if (not isinstance(evidence, dict) or set(evidence) != {'instance_id', 'command_id'}
                or any(not isinstance(evidence[key], str) or not evidence[key].strip()
                       for key in ('instance_id', 'command_id'))):
            raise ValueError('자동 채운 목표의 제어기와 명령 정보를 다시 확인해 주세요.')
        status = view.get('controller')
        observation = view.get('observation')
        if not isinstance(status, dict) or not isinstance(observation, dict):
            raise RuntimeError('최신 제어기 상태를 확인할 수 없습니다. 목표를 다시 확인해 주세요.')
        candidate = status.get('last_confirmed_applied_target')
        last_stop = status.get('last_stop')
        if (status.get('worker_alive') is not True or status.get('fresh_feedback') is not True
                or status.get('fault') is not None or status.get('active_command_id') is not None
                or status.get('lease_owner') is not None or status.get('state') not in ('idle', 'stopped')
                or not isinstance(last_stop, dict) or last_stop.get('confirmed') is not True
                or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                or observation.get('gripper_command') != 'open'
                or observation.get('gripper_reported_state') != 'open'
                or observation.get('gripper_reported_valid') is not True):
            raise RuntimeError('최신 정지와 관측, 집게 열기 상태를 다시 확인해 주세요.')
        if (not isinstance(candidate, dict) or status.get('instance_id') != evidence['instance_id']
                or candidate.get('instance_id') != evidence['instance_id']
                or candidate.get('command_id') != evidence['command_id']):
            raise RuntimeError('자동 채운 목표의 제어기 또는 명령이 바뀌었습니다. 목표를 다시 확인해 주세요.')
        try:
            current_target = _applied_target(candidate.get('joints_deg'))
        except (TypeError, ValueError):
            raise RuntimeError('확인된 적용 목표각이 유효하지 않습니다. 목표를 다시 확인해 주세요.') from None
        if current_target != target:
            raise RuntimeError('자동 채운 적용 목표각이 현재 확인값과 다릅니다. 목표를 다시 확인해 주세요.')
        joints = observation.get('joints_deg')
        if (not isinstance(joints, (list, tuple)) or len(joints) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in joints)):
            raise RuntimeError('현재 관절 관측을 다시 확인해 주세요.')
        if any(abs(observed - applied) > limits.joint_tolerance_deg + limits.observation_joint_drift_deg
               for observed, applied in zip(joints, target)):
            raise RuntimeError('현재 관절 관측이 확인된 마지막 적용 목표각과 다릅니다.')

    def _register_controller(self, layout, model_path, expected_hash, operation, *,
                             initial_applied_target_deg, controller_ready_confirmed,
                             initial_applied_target_evidence, slot_xy_mm=None, table_z_mm=None):
        """Register against an owned controller without issuing device commands."""
        from . import native_physical_session as physical, scene_recall
        from raccoon_control.kinematics import quantize_joints

        path = Path(model_path).resolve()
        if not path.is_file() or path.stat().st_size > _MAX_STUDENT_CHECKPOINT_BYTES:
            raise ValueError('선정 학생 모델을 확인해 주세요.')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != BEST_CLOSE_SHA256 or (expected_hash is not None and expected_hash != digest):
            raise ValueError('제어기 실행에는 선정된 20261030 학생 모델이 필요합니다.')
        scene = scene_recall.scene(layout)
        with self._text_request_lock:
            self._require_idle_locked(registration=True)
            controller, coordinator, mode = self._controller, self.coordinator, self.mode
            if controller is None:
                raise RuntimeError('먼저 제어기를 연결해 주세요.')
            if mode == 'physical' and self._physical_recovery_required:
                raise RuntimeError('먼저 현장 상태를 확인하고 제어기 정지를 확인해 주세요.')
            if mode == 'physical' and self._physical_initial_pose_required:
                raise RuntimeError('초기 자세가 확인되지 않았습니다. 연결을 해제하고 다시 연결해 주세요.')
            if operation.is_set() or self.cancelled.is_set():
                raise InterruptedError('등록을 취소했습니다.')
            view = controller.display_snapshot()
            observation, status = view['observation'], view['controller']
            if observation is None:
                raise RuntimeError('상주 제어기에 화면용 캐시 관측이 없습니다. 기존 연결을 유지하고 판본을 확인해 주세요.')
            if (not status.get('worker_alive') or status.get('fault') is not None
                    or status.get('active_command_id') is not None or status.get('lease_owner') is not None
                    or status.get('state') not in ('idle', 'stopped')
                    or observation.get('sensor_valid') is not True or observation.get('moving') is not False
                    or observation.get('gripper_command') != 'open'
                    or observation.get('gripper_reported_state') != 'open'
                    or observation.get('gripper_reported_valid') is not True):
                raise RuntimeError('신선한 정지 관측과 확인된 집게 열기 준비가 필요합니다.')
            reference = None
            reference_options = {}
            if mode == 'physical':
                if controller_ready_confirmed is not True:
                    raise ValueError('현재 배치와 제어기 준비 상태를 직접 확인해 주세요.')
                if initial_applied_target_deg is None:
                    if initial_applied_target_evidence is not None:
                        raise ValueError('적용 목표 없이 목표 근거를 사용할 수 없습니다.')
                    reference = self._stationary_registration_reference(controller)
                    target = reference['joints_deg']
                    reference_options = {'initial_target_source': 'stationary_observation',
                        'initial_observation_instance_id': reference['instance_id'],
                        'initial_observation_revision': reference['revision']}
                else:
                    target = physical._applied_target(initial_applied_target_deg)
                    if initial_applied_target_evidence is not None:
                        self._validate_registration_target_evidence(
                            view, target, initial_applied_target_evidence, controller.config.limits)
            else:
                if (initial_applied_target_deg is not None or controller_ready_confirmed is not False
                        or initial_applied_target_evidence is not None):
                    raise ValueError('모의 실행은 모의 제어기가 확인한 목표각을 사용합니다.')
                if coordinator is not None and coordinator.session.status == 'uncertain':
                    raise RuntimeError('모의 제어기를 끊고 다시 연결한 뒤 집게와 배치를 등록해 주세요.')
                target = (coordinator.session.scene_state()['current_applied_target_deg']
                          if coordinator is not None else self._controller_initial_target)
                if target is None:
                    raise RuntimeError('마지막 적용 목표각이 미확인입니다. 모의 제어기를 다시 연결해 주세요.')
                _, quantized = quantize_joints(target)
                if any(abs(a-b) > 1e-7 for a, b in zip(target, quantized)):
                    raise RuntimeError('모의 목표각을 다시 확인해야 합니다. 제어기를 다시 연결해 주세요.')
            if any(abs(a-b) > controller.config.limits.joint_tolerance_deg
                   + controller.config.limits.observation_joint_drift_deg
                   for a, b in zip(observation['joints_deg'], target)):
                raise ValueError('현재 관절 관측이 확인된 마지막 적용 목표각과 다릅니다.')
            calibration, profile = self._controller_calibration, self._controller_profile
            if mode == 'physical' and table_z_mm is not None:
                if (type(table_z_mm) not in (int, float) or not math.isfinite(table_z_mm)
                        or table_z_mm != calibration.table_z_mm):
                    raise ValueError('현재 학생은 탁상 높이 변경을 지원하지 않습니다.')
            if mode == 'physical' and slot_xy_mm is not None:
                candidate = replace(calibration, slot_xy_mm=slot_xy_mm)
                if candidate.slot_xy_mm != calibration.slot_xy_mm:
                    calibration = replace(candidate, coordinate_source='operator_input', verified=False)
                    calibration.require_ready(controller.config)
            # Do not leave an earlier registration usable if preparation or
            # memory reset fails after this point.
            self._registration_id = None

        new_session = coordinator is None
        execution_calibration = calibration
        if not new_session and mode == 'physical' and getattr(coordinator.session, 'supports_runtime_source_return', False):
            execution_calibration = replace(calibration, compact_waypoints=True)
        if new_session:
            import torch
            torch.set_num_threads(1)
            slots = SLOTS if mode == 'mock' else calibration.slot_xy_mm
            scope = physical.SIMULATION_FOUR_SLOT_PROFILE if mode == 'mock' else physical.PHYSICAL_INITIAL_PROFILE
            runtime_capabilities = (_controller_runtime_capabilities(controller)
                                    if mode == 'physical' and self._allow_runtime_retarget else {})
            runtime_retarget = bool(runtime_capabilities)
            blended_path = runtime_retarget and runtime_capabilities.get('blended_joint_path_supported') is True
            execution_calibration = (replace(calibration, compact_waypoints=True)
                                     if runtime_retarget else calibration)
            coordinator = physical.load_session(controller, slot_xy_mm=slots,
                initial_applied_target_deg=target, speed_pct=12 if mode == 'mock' else calibration.speed_pct,
                student_path=path, execution_profile=scope,
                **reference_options,
                active_slots=None if mode == 'mock' else profile.enabled_slots,
                # Use the existing mock source-return profile consistently;
                # it is incompatible with the separate virtual-descent path.
                execute_mock_virtual_descent=False,
                enable_mock_source_return=mode == 'mock',
                enable_mock_interrupt_retarget=mode == 'mock',
                path_calibration=execution_calibration if mode == 'physical' else None,
                # The teacher and the selected student command joint positions.
                # The SDK has no synchronized 0.5-second ramp: use the calibrated
                # position adapter without changing the student's final target.
                enable_experimental_segmented_path=mode == 'physical',
                enable_runtime_source_return=runtime_retarget,
                enable_staged_destination_retarget=runtime_retarget,
                **({'enable_arrival_correction': True} if runtime_retarget else {}),
                **({'enable_release_setdown': True} if runtime_retarget else {}),
                **({'enable_blended_path': True} if blended_path else {}))
        if self._registration_cancelled(operation):
            raise InterruptedError('등록을 취소했습니다.')
        if reference is not None:
            self._stationary_registration_reference(controller, expected=reference)
        if mode == 'physical' and initial_applied_target_evidence is not None:
            self._validate_registration_target_evidence(
                controller.display_snapshot(), target, initial_applied_target_evidence, controller.config.limits)
        if coordinator.session.status == 'uncertain':
            if mode != 'physical':
                raise RuntimeError('모의 제어기를 다시 연결해 주세요.')
            coordinator.recover_registration(_request_id(), scene, initial_applied_target_deg=target,
                                              controller_ready_confirmed=True, scene_confirmed=True,
                                              preserve_memory=True, **reference_options)
        else:
            if reference is not None and not new_session:
                with self._text_request_lock, coordinator._lock:
                    self._require_idle_locked(registration=True)
                    if not coordinator.session._lock.acquire(blocking=False):
                        raise RuntimeError('진행 중인 작업이 있어 배치를 등록하지 않았습니다.')
                    try:
                        with coordinator.session._scene_view_lock:
                            coordinator.session._current_target = list(target)
                            coordinator.session._current_target_source = 'stationary_observation'
                            coordinator.session._initial_observation_instance_id = reference['instance_id']
                            coordinator.session._initial_observation_revision = reference['revision']
                    finally:
                        coordinator.session._lock.release()
            request = _request_id()
            text = 'Register the scene'
            capture = coordinator.begin_interpretation(text, request_id=request, registration_scene=scene)
            ticket = coordinator.admit(capture['interpretation_id'],
                {'schema': 'task_program.v2', 'request_id': request, 'steps': [_task('register', text)]})
            result = coordinator.execute(ticket)
            if result.get('status') != 'completed':
                raise RuntimeError('배치 등록이 완료되지 않았습니다: ' + str(result.get('status')))

        forecast = self.forecast
        if new_session:
            from .demo_controller import UnavailableControllerForecast
            student = coordinator.session.student
            try:
                from .demo_controller_forecast import ControllerForecast
                forecast = ControllerForecast(controller, coordinator, mock=mode == 'mock')
            except Exception:
                # Registration already succeeded; a display-only copy must
                # not prevent it from being installed or replace the student.
                coordinator.session.student = student
                forecast = UnavailableControllerForecast()
                controller._forecast = forecast

        def install():
            if mode == 'physical':
                with coordinator.session._scene_view_lock:
                    coordinator.session.path_calibration = execution_calibration
                    coordinator.session._registered_slot_xy_mm = execution_calibration.slot_xy_mm
                self._registered_calibration = execution_calibration
            else:
                self._registered_calibration = None
            self.coordinator, self.sim, self.forecast, self.model_hash = coordinator, None, forecast, digest
            self._registration_id = operation.request_id
            if mode == 'physical':
                self._physical_stop_cleared_session = None
                # The normal registration path already obtains and rechecks a
                # stopped observation from this resident. Keep that identity
                # so even the first later display read can recover on its own.
                if reference is not None:
                    self._last_healthy_status_owner = (controller, coordinator, operation.request_id,
                        reference['instance_id'], coordinator.snapshot().get('registration_revision'))
                    self._last_healthy_gripper = (self._last_healthy_status_owner, 'open', 'open')
                    self._status_read_observation_refresh = None
        try:
            with self._text_request_lock:
                self._require_idle_locked(registration=True)
                if self._controller is not controller or self.mode != mode:
                    raise RuntimeError('연결이 바뀌어 배치를 등록하지 않았습니다.')
                if reference is not None:
                    self._stationary_registration_reference(controller, expected=reference)
                if mode == 'physical' and initial_applied_target_evidence is not None:
                    self._validate_registration_target_evidence(
                        controller.display_snapshot(), target, initial_applied_target_evidence, controller.config.limits)
                operation.commit(install, self.cancelled)
        except BaseException:
            if new_session:
                forecast.close()
                controller._forecast = None
            raise
        return {'snapshot': coordinator.snapshot(), 'model_path': str(path), 'model_sha256': digest,
                'registration_id': operation.request_id, 'finger_open_profile': 'controller_managed',
                'closed_move_profile': 'controller_managed',
                'straight_descent': bool(getattr(coordinator.session, 'execute_mock_virtual_descent', False)),
                'mode': mode}

    def _register(self, layout, model_path, expected_hash, *, finger_open_profile, closed_move_profile,
                  control_timing_profile, operation):
        """Prepare a tracked registration before atomically installing it."""
        if self._registration_cancelled(operation):
            raise InterruptedError('등록을 취소했습니다.')
        if not isinstance(finger_open_profile, str) or (finger_open_profile != 'auto' and finger_open_profile not in FINGER_OPEN_PROFILES):
            raise ValueError('지원되는 가상 집게 열기 방식을 선택해 주세요.')
        if not isinstance(closed_move_profile, str) or (closed_move_profile != 'auto' and closed_move_profile not in CLOSED_MOVE_PROFILES):
            raise ValueError('지원되는 가상 이동 곡선을 선택해 주세요.')
        if not isinstance(control_timing_profile, str) or (control_timing_profile != 'auto' and control_timing_profile not in CONTROL_TIMING_PROFILES):
            raise ValueError('지원되는 가상 이동 시간을 선택해 주세요.')
        cancelled = self.cancelled if operation is None else _EitherCancellation(self.cancelled, operation)
        from . import native_approach_session as native, scene_recall
        from .terra_native_approach_policy import load_student_checkpoint
        import torch
        scene = scene_recall.scene(layout)
        path = Path(model_path).resolve()
        if not path.is_file() or path.stat().st_size > _MAX_STUDENT_CHECKPOINT_BYTES:
            raise ValueError(f'{_MAX_STUDENT_CHECKPOINT_MB}MB 이하의 호환되는 학생 모델 파일을 선택해 주세요.')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path == CLOSE_STUDENT.resolve() and digest != CLOSE_SHA256:
            raise ValueError('선택한 모델의 지문이 등록된 지문과 다릅니다.')
        if path == BEST_CLOSE_STUDENT.resolve() and digest != BEST_CLOSE_SHA256:
            raise ValueError('선택한 모델의 지문이 등록된 지문과 다릅니다.')
        if path == DEFAULT_STUDENT.resolve() and expected_hash is None:
            expected_hash = native.STUDENT_SHA256
        if expected_hash is not None and digest != expected_hash:
            raise ValueError('선택한 모델의 지문이 등록된 지문과 다릅니다.')
        selected = digest == BEST_CLOSE_SHA256
        control_timing_profile = (control_timing_profile if control_timing_profile != 'auto' else
                                  'demo_1s' if selected else 'app_2s')
        previous_selected_profile = selected and control_timing_profile == 'app_2s'
        finger_open_profile = (finger_open_profile if finger_open_profile != 'auto' else
                               'minimum_jerk_0p5s' if previous_selected_profile else 'instant')
        closed_move_profile = (closed_move_profile if closed_move_profile != 'auto' else
                               'smoothstep' if previous_selected_profile else 'minimum_jerk')
        # The separately registered candidate has its own pinned loader and
        # session contract. Other files still use the original student loader.
        is_close_candidate = digest in (CLOSE_SHA256, BEST_CLOSE_SHA256)
        student = None
        if not is_close_candidate:
            try:
                student = load_student_checkpoint(io.BytesIO(raw), device='cpu').requires_grad_(False)
            except Exception as exc:
                raise ValueError('지원되는 학생 모델이 아닙니다. 선택한 파일을 적용할 수 없습니다.') from exc
        torch.set_num_threads(1)
        if self._registration_cancelled(cancelled):
            raise InterruptedError('등록을 취소했습니다.')
        sim = native.PublicCustomSlotsSceneMotionSim(scene, seed=620028, randomize=True, slot_xy_mm=SLOTS,
            ramp_seconds=.5, threshold_degrees=0., ramp_shape='minimum_jerk',
            finger_open_profile=finger_open_profile, closed_move_profile=closed_move_profile)
        sim._control_timing_profile = control_timing_profile
        # The public straight-descent comparison used this exact student and
        # these two profiles. Explicitly different selections stay unchanged.
        sim._straight_descent_enabled = (previous_selected_profile and finger_open_profile == 'minimum_jerk_0p5s'
                                         and closed_move_profile == 'smoothstep')
        # This nominal height adjustment was evaluated with the same selected
        # virtual student and profiles; it is not a physical-device setting.
        sim._support_height_guard_enabled = sim._straight_descent_enabled
        if is_close_candidate:
            from . import native_close_evidence_session
            # The shared loader reads its path independently. Give it the exact
            # bytes hashed above, so a file replacement during registration
            # cannot attach the selected student's allowance to another model.
            with tempfile.TemporaryDirectory(prefix='raccoon-demo-student-') as directory:
                snapshot_path = Path(directory) / 'student.pt'
                snapshot_path.write_bytes(raw)
                coordinator = native_close_evidence_session.load_session(sim, student_path=snapshot_path)
            if digest == BEST_CLOSE_SHA256:
                coordinator.session.train_groups = coordinator.session.train_groups | _BEST_CLOSE_DEMO_GROUPS
        else:
            coordinator = native.load_session(sim)
            # Only this new virtual session changes; the on-disk model and the
            # pinned GRU, safeguards and simulator remain untouched.
            coordinator.session.student.policy.student = student
        from .demo_motion import attach_motion_monitor
        forecast_holder = {}
        forecast_session = uuid.uuid4().hex
        def command_hook(phase, number, target, opening):
            forecast = forecast_holder.get('forecast')
            if forecast is not None:
                if phase == 'start':
                    forecast.command_started(number, target, opening)
                elif phase == 'progress':
                    forecast.command_progress(number)
                else:
                    forecast.command_finished(number)
        motion_cancel = _EitherCancellation(self.cancelled, coordinator.session.cancel)
        def request_owner():
            with coordinator._lock:
                return coordinator._active['request_id'] if coordinator._active is not None else None
        attach_motion_monitor(sim, self.emit, motion_cancel, pause_per_sample=self.visual_pace_seconds,
                              command_hook=command_hook, session_token=forecast_session, request_owner=request_owner)
        request = _request_id()
        text = 'Register the scene'
        capture = coordinator.begin_interpretation(text, request_id=request, registration_scene=scene)
        ticket = coordinator.admit(capture['interpretation_id'],
            {'schema': 'task_program.v2', 'request_id': request, 'steps': [_task('register', text)]})
        if self._registration_cancelled(cancelled):
            raise InterruptedError('등록을 취소했습니다.')
        result = coordinator.execute(ticket)
        if result.get('status') != 'completed':
            raise RuntimeError('가상 배치 등록이 완료되지 않았습니다: ' + str(result.get('status')))
        forecast = None
        if isinstance(coordinator.session, native.runtime.OracleFreeSceneModelSession):
            from .demo_forecast import DemoForecast
            forecast = DemoForecast(sim, coordinator.session, self.emit, motion_cancel,
                                    coordinator=coordinator, session_token=forecast_session)
            forecast_holder['forecast'] = forecast
        previous_forecast = []
        def install():
            previous_forecast.append(self.forecast)
            self.forecast = forecast
            self.sim, self.coordinator, self.model_hash = sim, coordinator, digest
            self._registration_id = operation.request_id
        try:
            with self._text_request_lock:
                if self._shutting_down or cancelled.is_set():
                    raise InterruptedError('등록을 취소했습니다.')
                current = self.coordinator
                if (self._text_request is not None or self._record_request is not None
                        or self._conversation_requests):
                    raise RuntimeError('다른 요청이 진행 중이어서 배치를 바꾸지 않았습니다.')
                if current is not None:
                    with current._lock:
                        if current._active is not None or current._interpretations or current._pending:
                            raise RuntimeError('다른 요청이 진행 중이어서 배치를 바꾸지 않았습니다.')
                        operation.commit(install, self.cancelled)
                else:
                    operation.commit(install, self.cancelled)
        except BaseException:
            if forecast is not None:
                forecast.close(wait=True)
            raise
        warnings = []
        if previous_forecast[0] is not None:
            try:
                previous_forecast[0].close()
            except Exception as exc:
                warnings.append({'stage': 'previous_forecast', 'detail': f'{type(exc).__name__}: {exc}'})
        try:
            self.emit('observation', sim.observation())
        except Exception as exc:
            warnings.append({'stage': 'observation', 'detail': f'{type(exc).__name__}: {exc}'})
        result = {'snapshot': coordinator.snapshot(), 'model_path': str(path), 'model_sha256': digest,
                  'registration_id': operation.request_id,
                  'finger_open_profile': sim.finger_open_profile, 'closed_move_profile': sim.closed_move_profile,
                  'control_timing_profile': sim._control_timing_profile,
                  'straight_descent': sim._straight_descent_enabled}
        if warnings:
            result['post_commit_warnings'] = warnings
        return result

    def _voice_backend(self):
        """Return an explicitly injected recognizer, if this caller supplied one."""
        with self._voice_lock:
            if self._voice_closed:
                raise InterruptedError('음성 인식을 취소했습니다.')
            return self.voice

    def interpret_voice_audio(self, audio, *, capture=None, coordinator=None, language='auto', cancel_event=None):
        from .task_program_v2 import audio_stop_flags
        cancelled = self.cancelled if cancel_event is None else cancel_event
        if cancelled.is_set() or self._shutting_down:
            raise InterruptedError('음성 해석이 취소되었습니다.')
        interpreter = self.interpreter.start(cancelled)
        if coordinator is None and capture is None:
            text, result = interpreter.interpret_audio(audio, request_id=_request_id(), language=language,
                                                       cancel_event=cancelled)
        elif coordinator is self.coordinator and coordinator is not None and isinstance(capture, dict):
            text, result = coordinator.interpret_voice_audio(capture['interpretation_id'], interpreter, audio,
                                                            language=language, cancel_event=cancelled)
        else:
            raise RuntimeError('발화가 시작된 세션이 바뀌었습니다.')
        if cancelled.is_set() or self._shutting_down:
            raise InterruptedError('음성 해석이 취소되었습니다.')
        return {'text': text, 'direct_audio': True, 'engine': 'gemma-direct-audio', 'device': 'CUDA0',
                'requested_language': language, 'program': copy.deepcopy(result.program),
                'voice_output': getattr(result, 'voice_output', None), **audio_stop_flags(result)}

    def interpret_pending_voice_audio(self, audio, *, capture, coordinator, pending, language, cancel_event,
                                      rebase=False):
        if cancel_event.is_set() or self._shutting_down or coordinator is not self.coordinator:
            raise InterruptedError('음성 해석이 취소되었습니다.')
        interpreter = self.interpreter.start(cancel_event)
        text, result, disposition = coordinator.interpret_pending_voice_audio(
            capture['interpretation_id'], interpreter, audio, pending, language=language,
            cancel_event=cancel_event, rebase=rebase)
        if cancel_event.is_set() or self._shutting_down:
            raise InterruptedError('음성 해석이 취소되었습니다.')
        return {'text': text, 'direct_audio': True, 'engine': 'gemma-direct-audio', 'device': 'CUDA0',
                'pending_disposition': disposition,
                'program': copy.deepcopy(result.program) if result is not None else None}

    def transcribe(self, path, *, operation=None):
        operation = operation or _AuxiliaryOperation()
        with self._text_request_lock:
            if self._shutting_down:
                raise RuntimeError('서비스가 종료 중입니다.')
            if operation.is_set():
                raise InterruptedError('음성 변환을 취소했습니다.')
            if operation.request_id in self._transcription_requests:
                raise RuntimeError('이미 진행 중인 음성 변환 요청입니다.')
            self._transcription_requests.add(operation.request_id)
        try:
            voice = self._voice_backend()
            if operation.is_set():
                raise InterruptedError('음성 변환을 취소했습니다.')
            result = (voice.transcribe(path, language='auto') if voice is not None else
                      self.interpret_voice_audio(path, language='auto', cancel_event=operation))
            with self._voice_lock:
                if self._voice_closed:
                    raise InterruptedError('음성 변환을 취소했습니다.')
                operation.commit(lambda: None, self.cancelled)
            return result
        finally:
            with self._text_request_lock:
                self._transcription_requests.discard(operation.request_id)

    def reserve_text_request(self):
        """Reserve a cancellable text request before handing it to a worker."""
        with self._text_request_lock:
            if self._shutting_down:
                raise RuntimeError('서비스가 종료 중입니다.')
            if (self._text_request is not None or self._controller_preparation is not None or self._recovery_blocks_new_request_locked()
                    or (self.mode != 'virtual' and self._registration_requests)):
                raise RuntimeError('진행 중인 요청이 있습니다.')
            request = _request_id()
            self._text_request = (request, threading.Event(), self.coordinator)
            self._pending_text_plans[request] = {'ready': threading.Event(), 'result': None, 'text': None}
            if self.coordinator is not None:
                self.coordinator.register_pending_voice(request)
            return request

    @staticmethod
    def _hold_pending_followup(coordinator, text, payload):
        with coordinator._lock:
            capture = coordinator.begin_interpretation(text, request_id=_request_id())
            hold = coordinator.hold_pending_voice(capture['interpretation_id'], payload['target'])
            if hold is None:
                coordinator.discard_interpretation(capture['interpretation_id'])
                return None
            return {'coordinator': coordinator, 'capture': capture, 'hold': hold, **payload}

    @staticmethod
    def _resolve_followup_disposition(coordinator, hold, ident, cancelled, disposition, result, on_replace):
        if disposition == 'replace':
            coordinator.replace_pending_text_plan(hold, ident, result)
            on_replace(result)
            return '변경한 요청으로 이어서 실행합니다.'
        if disposition in ('stop', 'uncertain'):
            cancelled.set()
            coordinator.resolve_pending_voice(hold, ident, cancel=True)
            return ('이전 요청을 취소했습니다.' if disposition == 'stop' else
                    '변경할 내용을 확인하지 못해 이전 요청을 실행하지 않았습니다. 다시 입력해 주세요.')
        coordinator.resolve_pending_voice(hold, ident)
        return '이전 요청을 유지합니다.'

    def begin_pending_text_followup(self, target, text):
        """Capture a correction before an interpreting text request can start."""
        with self._text_request_lock:
            entry, coordinator = self._text_request, self.coordinator
            if (entry is None or entry[0] != target or entry[2] is not coordinator
                    or entry[1].is_set() or coordinator is None or target not in self._pending_text_plans):
                return None
            return self._hold_pending_followup(coordinator, text, {
                'cancelled': entry[1], 'plan': self._pending_text_plans[target], 'target': target})

    def begin_pending_voice_text_followup(self, chat, target, text):
        """Hold the same unstarted spoken request while a typed edit is interpreted."""
        with self._text_request_lock, chat._state_lock:
            coordinator = self.coordinator
            if (self._shutting_down or coordinator is None or chat.coordinator is not coordinator
                    or chat.stop_event.is_set() or chat._pending_id != target
                    or chat._pending_phase != 'processing' or chat._pending_cancel is None
                    or chat._pending_cancel.is_set()):
                return None
            return self._hold_pending_followup(coordinator, text, {
                'cancelled': chat._pending_cancel, 'chat': chat, 'target': target})

    def interpret_pending_voice_text_followup(self, followup):
        """Apply a typed edit to a held spoken plan without starting another request."""
        from .task_program_v2 import TaskProgramInterpreter
        coordinator, capture, chat = followup['coordinator'], followup['capture'], followup['chat']
        hold, cancelled, target = followup['hold'], followup['cancelled'], followup['target']
        ident, released, promoted = capture['interpretation_id'], False, False
        try:
            # Earlier voice edits may replace the pending request ID. Wait
            # until they have applied, then bind to that latest held plan.
            coordinator.wait_pending_followup_turn(
                hold, ident, _EitherCancellation(self.cancelled, chat.stop_event))
            with self._text_request_lock, chat._state_lock, coordinator._lock:
                target = hold['current']
                if chat._pending_id == target and chat._pending_cancel is not None:
                    cancelled = chat._pending_cancel
            while True:
                with self._text_request_lock, chat._state_lock, coordinator._lock:
                    if (self._shutting_down or self.coordinator is not coordinator or chat.stop_event.is_set()
                            or cancelled.is_set() or chat._pending_id != target
                            or chat._pending_cancel is not cancelled
                            or chat._pending_phase not in ('processing', 'nonactionable_waiting')
                            or hold.get('cancelled')):
                        raise InterruptedError('이전 음성 요청을 더 이상 변경할 수 없습니다.')
                    plan = copy.deepcopy(chat._pending_plan)
                    original = coordinator._pending_model_results.get(target)
                    program = (plan or {}).get('program') if isinstance(plan, dict) else None
                    if program is None and original is not None:
                        program = original.program
                    steps = program.get('steps') if isinstance(program, dict) else None
                    nonactionable = bool(steps and all(step.get('op') in ('clarify', 'status') for step in steps))
                    if nonactionable or (chat._pending_phase == 'nonactionable_waiting'
                                         and hold.get('finished_nonactionable')):
                        if (self._text_request is not None or self._record_request is not None
                                or coordinator._active is not None or coordinator._arm_home_reservation is not None):
                            raise InterruptedError('새 요청을 시작할 수 없습니다.')
                        new_request = capture['request_id']
                        new_cancel = threading.Event()
                        previous_voice = chat._task_thread
                        coordinator.resolve_pending_voice(hold, ident, replacement=new_request)
                        # The original answer must be free to finish; later
                        # follow-ups remain ordered on the new request.
                        coordinator._voice_holds.pop(target, None)
                        hold.pop('finished_nonactionable', None)
                        hold.pop('nonactionable_followup_used', None)
                        chat._pending_id, chat._pending_cancel = new_request, new_cancel
                        chat._pending_phase, chat._pending_plan = 'processing', None
                        self._text_request = (new_request, new_cancel, coordinator)
                        self._pending_text_plans[new_request] = {'ready': threading.Event(), 'result': None, 'text': None}
                        released = promoted = True
                        return {'promoted_text': {'request': new_request, 'capture': capture,
                                                  'coordinator': coordinator, 'cancelled': new_cancel,
                                                  'chat': chat, 'text': capture['text'],
                                                  'previous_voice': previous_voice}}
                    if isinstance(plan, dict) and isinstance(plan.get('program'), dict):
                        pending = plan
                        break
                    if original is not None:
                        if original.status != 'ok' or original.unresolved or not isinstance(original.program, dict):
                            raise ValueError('이전 음성 요청의 작업 내용을 확인하지 못했습니다.')
                        pending = {'transcript': (plan or {}).get('transcript', ''),
                                   'program': copy.deepcopy(original.program)}
                        break
                cancelled.wait(.05)
            interpreter = TaskProgramInterpreter(self.interpreter.start(cancelled).client)
            _text, result, disposition = interpreter.interpret_pending_text(
                capture['text'], request_id=target, reference_context=capture['reference_context'],
                pending=pending, cancel_event=_EitherCancellation(self.cancelled, cancelled, chat.stop_event))
            with self._text_request_lock, chat._state_lock, coordinator._lock:
                if (self.coordinator is not coordinator or chat.stop_event.is_set() or cancelled.is_set()
                        or chat._pending_id != target or chat._pending_cancel is not cancelled
                        or chat._pending_phase != 'processing'):
                    raise InterruptedError('이전 요청이 취소되었습니다.')

                def apply_voice_replace(replaced):
                    chat._pending_plan = {'transcript': pending['transcript'] + '\n' + capture['text'],
                                          'program': copy.deepcopy(replaced.program)}
                    coordinator._pending_model_results[target] = copy.deepcopy(replaced)

                message = self._resolve_followup_disposition(
                    coordinator, hold, ident, cancelled, disposition, result, apply_voice_replace)
                released = True
            return {'reply': {'kind': 'pending_text_followup', 'text': message,
                              'filtered': False, 'speak': True}, 'snapshot': coordinator.snapshot()}
        finally:
            with self._text_request_lock, chat._state_lock, coordinator._lock:
                if not released and ident in hold.get('tokens', ()):
                    cancelled.set()
                    coordinator.resolve_pending_voice(hold, ident, cancel=True)
                if not promoted:
                    coordinator.discard_interpretation(ident)
                if (chat._pending_id == target and chat._pending_phase == 'nonactionable_waiting'
                        and hold.get('finished_nonactionable') and not hold.get('tokens')):
                    chat._clear_pending_state_locked()
                    coordinator.forget_pending_voice(target)

    def execute_promoted_text(self, promotion):
        """Run the independently captured text after a non-actionable voice turn."""
        from .task_program_v2 import TaskProgramInterpreter
        coordinator, capture, chat = promotion['coordinator'], promotion['capture'], promotion['chat']
        request, text, request_cancel = promotion['request'], promotion['text'], promotion['cancelled']
        cancelled = _EitherCancellation(self.cancelled, request_cancel)
        def update_plan(result):
            with chat._state_lock:
                if chat._pending_id == request and chat._pending_cancel is request_cancel:
                    chat._pending_plan = {'transcript': text, 'program': copy.deepcopy(result.program)}
        try:
            previous_voice = promotion.get('previous_voice')
            while previous_voice is not None and previous_voice.is_alive():
                if cancelled.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                previous_voice.join(.05)
            interpreter = TaskProgramInterpreter(self.interpreter.start(cancelled).client)
            result = interpreter.interpret_standalone_text(text, request_id=request,
                reference_context=capture['reference_context'], cancel_event=cancelled)
            return self._interpret_and_execute(coordinator, capture, text, request,
                cancel_event=cancelled, preinterpreted=result, plan_callback=update_plan)
        finally:
            self.finish_promoted_text(promotion)

    def finish_promoted_text(self, promotion, *, cancel=False):
        coordinator, capture, chat = promotion['coordinator'], promotion['capture'], promotion['chat']
        request, request_cancel = promotion['request'], promotion['cancelled']
        with self._text_request_lock, chat._state_lock, coordinator._lock:
            if cancel:
                request_cancel.set()
                coordinator.cancel_pending_voice(request)
            coordinator.discard_interpretation(capture['interpretation_id'])
            pending = self._pending_text_plans.pop(request, None)
            if pending is not None:
                pending['ready'].set()
            coordinator.forget_pending_voice(request)
            if self._text_request is not None and self._text_request[0] == request:
                self._text_request = None
            if chat._pending_id == request and chat._pending_cancel is request_cancel:
                chat._clear_pending_state_locked()

    def interpret_pending_text_followup(self, followup):
        """Interpret one typed follow-up while its original plan remains unstarted."""
        from .task_program_v2 import TaskProgramInterpreter
        coordinator, capture = followup['coordinator'], followup['capture']
        hold, cancelled, pending = followup['hold'], followup['cancelled'], followup['plan']
        ident, target = capture['interpretation_id'], followup['target']
        released = False
        try:
            coordinator.wait_pending_followup_turn(hold, ident, _EitherCancellation(self.cancelled, cancelled))
            while not pending['ready'].wait(.1):
                if cancelled.is_set() or self.cancelled.is_set() or self._shutting_down:
                    raise InterruptedError('요청을 취소했습니다.')
            original = pending['result']
            if (cancelled.is_set() or self.cancelled.is_set() or original is None
                    or original.status != 'ok' or original.unresolved or original.program is None):
                raise InterruptedError('이전 요청을 더 이상 변경할 수 없습니다.')
            interpreter = TaskProgramInterpreter(self.interpreter.start(cancelled).client)
            _text, result, disposition = interpreter.interpret_pending_text(
                capture['text'], request_id=target, reference_context=capture['reference_context'],
                pending={'transcript': pending['text'], 'program': original.program},
                cancel_event=_EitherCancellation(self.cancelled, cancelled))
            with self._text_request_lock, coordinator._lock:
                if (self.coordinator is not coordinator or self._text_request is None
                        or self._text_request[0] != target or cancelled.is_set()):
                    raise InterruptedError('이전 요청이 취소되었습니다.')

                def apply_text_replace(replaced):
                    pending['result'] = replaced
                    pending['text'] += '\n' + capture['text']

                message = self._resolve_followup_disposition(
                    coordinator, hold, ident, cancelled, disposition, result, apply_text_replace)
                released = True
            return {'reply': {'kind': 'pending_text_followup', 'text': message,
                              'filtered': False, 'speak': True}, 'snapshot': coordinator.snapshot()}
        finally:
            with coordinator._lock:
                if not released and ident in hold.get('tokens', ()):
                    cancelled.set()
                    coordinator.resolve_pending_voice(hold, ident, cancel=True)
                coordinator.discard_interpretation(ident)

    def reserve_record_request(self):
        """Reserve one cancellable recording before handing it to a worker."""
        with self._text_request_lock:
            if self._shutting_down:
                raise RuntimeError('서비스가 종료 중입니다.')
            if self.coordinator is None:
                raise RuntimeError('먼저 ' + ('가상 시작 배치' if self.mode == 'virtual' else '시작 배치') + '를 등록해 주세요.')
            if (self._record_request is not None or self._controller_preparation is not None or self._recovery_blocks_new_request_locked()
                    or (self.mode != 'virtual' and self._registration_requests)):
                raise RuntimeError('진행 중인 녹음이 있습니다.')
            request = _request_id()
            self._record_request = (request, threading.Event(), self.coordinator)
            return request

    def _text_stop_reply(self, text, kind, message, stop=None):
        result = {'reply': {'kind': kind, 'text': message, 'filtered': False, 'speak': True},
                  'snapshot': self.coordinator.snapshot()}
        if stop is not None:
            result['stop_request'] = stop
        return result

    def _submit_direct_text_stop(self, text, target):
        korean = bool(re.search(r'[가-힣]', text))
        if target is _NO_STOP_TARGET:
            # A newly submitted explicit stop can capture the current target;
            # an unbound stop() callback cannot infer which past request it meant.
            with self._text_request_lock:
                candidates = self._stop_candidates()
                if len(candidates) > 1:
                    message = ('중단할 요청을 구분할 수 없습니다. 해당 작업의 중단을 다시 요청해 주세요.' if korean else
                               'The request to stop is ambiguous. Please stop the intended task directly.')
                    return self._text_stop_reply(text, 'clarify_stop_target', message)
                target = next(iter(candidates), None)
        stop = self.stop(target_request_id=target)
        if stop.get('status') == 'stop_requested':
            message = (('가상 작업' if self.mode == 'virtual' else '작업') + ' 중단을 요청했습니다. 결과는 아직 확인되지 않았습니다.' if korean else
                       'Stop requested for the ' + ('virtual ' if self.mode == 'virtual' else '') + 'task. Its final state is not yet confirmed.')
            return self._text_stop_reply(text, 'bound_stop_proposal', message, stop)
        if stop.get('status') == 'cancelled_pending_request':
            message = ('해당 요청의 실행을 취소했습니다. 다른 작업은 중단하지 않았습니다.' if korean else
                       'I cancelled that request. No other task was stopped.')
            return self._text_stop_reply(text, 'cancelled_pending_request', message)
        if stop.get('status') == 'already_completed':
            message = (('해당 가상 작업' if self.mode == 'virtual' else '해당 작업') + '은 이미 완료됐습니다. 중단을 요청하지 않았습니다.' if korean else
                       'The ' + ('virtual ' if self.mode == 'virtual' else '') + 'task has already completed; no stop was requested.')
            return self._text_stop_reply(text, 'already_completed', message)
        if stop.get('status') == 'ambiguous_target':
            message = ('중단할 요청을 구분할 수 없습니다. 해당 작업의 중단을 다시 요청해 주세요.' if korean else
                       'The request to stop is ambiguous. Please stop the intended task directly.')
            return self._text_stop_reply(text, 'clarify_stop_target', message)
        message = (('해당 시점의 중단할 가상 작업이 없습니다.' if self.mode == 'virtual' else '해당 시점의 중단할 작업이 없습니다.') if korean else
                   'There is no matching ' + ('virtual ' if self.mode == 'virtual' else '') + 'task to stop.')
        return self._text_stop_reply(text, 'no_matching_active_request', message)

    def submit(self, text, *, request_id=None, stop_target=_NO_STOP_TARGET):
        if self.coordinator is None:
            raise RuntimeError('먼저 ' + ('가상 시작 배치' if self.mode == 'virtual' else '시작 배치') + '를 등록해 주세요.')
        if not text.strip():
            raise ValueError('요청을 입력해 주세요.')
        with self._text_request_lock:
            if (self._shutting_down and (request_id is None or self._text_request is None
                                        or self._text_request[0] != request_id)):
                raise RuntimeError('서비스가 종료 중입니다.')
        coordinator = self.coordinator
        if _is_direct_stop_text(text):
            return self._submit_direct_text_stop(text, stop_target)
        if stop_target is not _NO_STOP_TARGET:
            if not _is_text_stop_candidate(text):
                raise ValueError('진행 중에는 명확한 즉시 정지 요청만 처리할 수 있습니다.')
            cancelled_pending = False
            pending_completed = False
            with self._text_request_lock:
                pending = next((entry for entry in (self._text_request, self._record_request)
                                if entry is not None and entry[0] == stop_target), None)
                if pending is not None and stop_target is not None:
                    with pending[2]._lock:
                        current = pending[2].snapshot()['reference_context']
                        if (current.get('active_request_id') != stop_target
                                and _validated_pending_stop_prefix(text, current)):
                            stopped = self.stop(target_request_id=stop_target)
                            cancelled_pending = stopped.get('status') == 'cancelled_pending_request'
                            pending_completed = stopped.get('status') == 'already_completed'
            if cancelled_pending:
                korean = bool(re.search(r'[가-힣]', text))
                message = ('해당 요청의 실행을 취소했습니다. 이후 작업은 실행하지 않았습니다.' if korean else
                           'I cancelled that request. The follow-up actions were not executed.')
                return self._text_stop_reply(text, 'cancelled_pending_request', message)
            if pending_completed:
                korean = bool(re.search(r'[가-힣]', text))
                message = (('해당 가상 작업' if self.mode == 'virtual' else '해당 작업') + '은 이미 완료됐습니다. 중단이나 이후 작업은 실행하지 않았습니다.' if korean else
                           'The ' + ('virtual ' if self.mode == 'virtual' else '') + 'task already completed. No stop or follow-up action was executed.')
                return self._text_stop_reply(text, 'already_completed', message)
            request = _request_id()
            capture = coordinator.begin_interpretation(text, request_id=request)
            if capture['reference_context'].get('active_request_id') != stop_target or stop_target is None:
                coordinator.discard_interpretation(capture['interpretation_id'])
                return self._text_stop_reply(text, 'no_matching_active_request',
                                             '해당 시점의 중단할 ' + ('가상 작업' if self.mode == 'virtual' else '작업') + '이 없습니다.')
            return self._interpret_and_execute(coordinator, capture, text, request, stop_only=True)
        with self._text_request_lock:
            if (self._shutting_down and (request_id is None or self._text_request is None
                                        or self._text_request[0] != request_id)):
                raise RuntimeError('서비스가 종료 중입니다.')
            if request_id is None:
                request_id = self.reserve_text_request()
            elif self._text_request is None:
                if self._controller_preparation is not None or self._recovery_blocks_new_request_locked() or (self.mode != 'virtual' and self._registration_requests):
                    raise RuntimeError('진행 중인 작업을 먼저 종료해 주세요.')
                self._text_request = (request_id, threading.Event(), coordinator)
                self._pending_text_plans[request_id] = {'ready': threading.Event(), 'result': None, 'text': None}
                coordinator.register_pending_voice(request_id)
            if self._text_request[0] != request_id:
                raise RuntimeError('진행 중인 요청이 있습니다.')
            if self._text_request[2] is not coordinator:
                self._text_request = None
                raise RuntimeError(('가상 배치' if self.mode == 'virtual' else '배치') + '가 바뀌어 요청을 실행하지 않았습니다.')
            request_cancel = self._text_request[1]
        previous_request = getattr(self._request_context, 'request_id', None)
        self._request_context.request_id = request_id
        try:
            capture = coordinator.begin_interpretation(text, request_id=request_id)
            return self._interpret_and_execute(coordinator, capture, text, request_id,
                                               cancel_event=_EitherCancellation(self.cancelled, request_cancel))
        finally:
            self._request_context.request_id = previous_request
            with self._text_request_lock:
                pending_plan = self._pending_text_plans.pop(request_id, None)
                if pending_plan is not None:
                    pending_plan['ready'].set()
                coordinator.forget_pending_voice(request_id)
                if self._text_request is not None and self._text_request[0] == request_id:
                    self._text_request = None

    def record_and_submit(self, *, request_id=None):
        """Capture explicitly requested speech and bind it before recognition."""
        coordinator = self.coordinator
        if coordinator is None:
            raise RuntimeError('먼저 ' + ('가상 시작 배치' if self.mode == 'virtual' else '시작 배치') + '를 등록해 주세요.')
        with self._text_request_lock:
            if (self._shutting_down and (request_id is None or self._record_request is None
                                        or self._record_request[0] != request_id)):
                raise RuntimeError('서비스가 종료 중입니다.')
            if request_id is None:
                request_id = self.reserve_record_request()
            elif self._record_request is None:
                if self._controller_preparation is not None or self._recovery_blocks_new_request_locked() or (self.mode != 'virtual' and self._registration_requests):
                    raise RuntimeError('진행 중인 작업을 먼저 종료해 주세요.')
                self._record_request = (request_id, threading.Event(), coordinator)
            if self._record_request[0] != request_id:
                raise RuntimeError('진행 중인 녹음이 있습니다.')
            if self._record_request[2] is not coordinator:
                self._record_request = None
                raise RuntimeError(('가상 배치' if self.mode == 'virtual' else '배치') + '가 바뀌어 녹음을 실행하지 않았습니다.')
            cancelled = _EitherCancellation(self.cancelled, self._record_request[1])
        ident = None
        previous_request = getattr(self._request_context, 'request_id', None)
        self._request_context.request_id = request_id
        try:
            if cancelled.is_set():
                raise InterruptedError('녹음을 취소했습니다.')
            capture = coordinator.begin_voice_processing(request_id=request_id)
            ident = capture['interpretation_id']
            voice = self._voice_backend()
            if voice is None:
                from .demo_voice_chat import capture_voice_pcm
                audio = capture_voice_pcm(seconds=7, cancel=cancelled)
                speech = self.interpret_voice_audio(audio, capture=capture, coordinator=coordinator,
                                                    cancel_event=cancelled)
            else:
                speech = voice.record(seconds=7, cancel=cancelled)
            if cancelled.is_set():
                raise InterruptedError('녹음을 취소했습니다.')
            if not isinstance(speech, dict) or not isinstance(speech.get('text'), str) or not speech['text'].strip():
                raise ValueError('음성을 인식하지 못했습니다. 다시 말하거나 텍스트를 입력해 주세요.')
            text = speech['text'].strip()
            capture = coordinator.complete_voice_transcription(ident, text)
            self.emit('transcript', {'text': text, 'device': speech.get('device'),
                                      'cuda_runtime': speech.get('cuda_runtime'),
                                      'fallback_reason': speech.get('fallback_reason')})
            # The shared helper now owns and releases the captured capability.
            ident = None
            return self._voice_route_and_execute(coordinator, capture, text, request_id, cancel_event=cancelled)
        finally:
            self._request_context.request_id = previous_request
            if ident is not None:
                coordinator.discard_interpretation(ident)
            with self._text_request_lock:
                if self._record_request is not None and self._record_request[0] == request_id:
                    self._record_request = None

    def request_voice_retarget(self, capture, text, target_request_id, *, cancel_event=None, coordinator=None):
        """Interpret a bounded redirect without issuing a new ordinary task.

        ``None`` means this is not a redirect and the caller retains the voice
        capture. For a recognized or ambiguous redirect this method consumes
        the original capture; only the coordinator may authorize a request.
        """
        from .voice_retarget import parse_voice_retarget

        kind, destination = parse_voice_retarget(text)
        if kind == 'unrelated':
            return None
        coordinator = coordinator or self.coordinator
        if not isinstance(capture, dict) or not isinstance(capture.get('interpretation_id'), str):
            raise ValueError('음성 요청의 출처를 확인할 수 없습니다.')
        ident = capture['interpretation_id']
        try:
            dispatch = False
            with self._text_request_lock:
                if coordinator is None or coordinator is not self.coordinator or self._shutting_down:
                    result = {'status': 'retarget_rejected', 'reason': 'captured_session_changed'}
                elif cancel_event is not None and cancel_event.is_set():
                    result = {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
                elif kind == 'ambiguous':
                    result = {'status': 'retarget_rejected', 'reason': 'destination_unclear'}
                elif not isinstance(target_request_id, str):
                    result = {'status': 'retarget_rejected', 'reason': 'captured_transfer_not_active'}
                elif capture.get('text') != text:
                    result = {'status': 'retarget_rejected', 'reason': 'captured_transcript_changed'}
                else:
                    dispatch = True
            if dispatch:
                # The executor may wait for a command boundary. Holding either
                # service or coordinator lock here would also hold up stop.
                # The coordinator rechecks the exact transcript and owner.
                cancellation = _RetargetServiceCancellation(self, coordinator, cancel_event)
                result = coordinator.request_transfer_retarget(
                    ident, target_request_id, destination, cancel_event=cancellation,
                    expected_transcript=text)
            return self._retarget_reply(coordinator, text, result)
        finally:
            if coordinator is not None:
                coordinator.discard_interpretation(ident)

    @staticmethod
    def _retarget_reply(coordinator, text, result):
        english = bool(re.search(r'[A-Za-z]', text)) and not bool(re.search(r'[가-힣]', text))
        status, reason = result.get('status'), result.get('reason')
        if status == 'retarget_requested':
            reply = ('The change was requested but is not yet confirmed. I will stop the remaining previous tasks.' if english
                     else '목표 변경을 요청했습니다. 적용은 아직 확인되지 않았으며, 남은 기존 작업은 중단합니다.')
        elif status == 'retarget_unknown':
            reply = ('I could not confirm whether the destination change was accepted. Please check the task status.' if english
                     else '목적지 변경이 접수됐는지 확인하지 못했습니다. 작업 상태를 확인해 주세요.')
            if result.get('remaining_program_stopped'):
                reply = ("I could not confirm the change. The remaining previous tasks will not run." if english
                         else '목적지 변경 결과를 확인하지 못했습니다. 남은 기존 작업은 실행하지 않습니다.')
        elif status == 'retarget_unsupported':
            reply = ('This destination change is not supported during the current transfer; the existing task was not changed.' if english
                     else '현재 이동에서는 해당 목적지 변경을 지원하지 않습니다. 기존 작업을 변경하지 않았습니다.')
        elif reason == 'destination_unchanged':
            reply = ('That is already the current destination.' if english else '이미 그 자리로 이동하고 있습니다.')
        elif reason == 'retarget_already_requested':
            reply = ('That change is already requested or its result is unconfirmed.' if english
                     else '해당 자리로의 변경은 이미 요청됐거나 처리 결과가 확인되지 않았습니다.')
        elif reason == 'original_destination_mismatch':
            reply = ('The previous destination you named does not match the current move.' if english
                     else '말씀하신 기존 목적지가 현재 이동 목적지와 다릅니다.')
        elif reason in ('destination_unclear', 'destination_not_grounded', 'destination_infeasible'):
            reply = ('I could not confirm the new destination. Please state the destination again.' if english
                     else '새 목적지를 확인하지 못했습니다. 바꿀 자리를 다시 말씀해 주세요.')
        else:
            reply = ('The original transfer changed or was stopped, so I did not change its destination.' if english
                     else '처음 말씀하신 이동이 바뀌었거나 중단돼 목적지를 변경하지 않았습니다.')
        if status in ('retarget_rejected', 'retarget_unsupported') and result.get('remaining_program_stopped'):
            reply = ('The change was not accepted, and the remaining previous tasks were stopped.' if english
                     else '목적지 변경은 접수되지 않았고, 남은 기존 작업은 중단했습니다.')
        return {'reply': {'kind': status, 'text': reply, 'filtered': False, 'speak': True},
                'retarget': result,
                'snapshot': coordinator.snapshot() if coordinator is not None else None}

    def interpret_voice_followup(self, capture, text, target_request_id, *, cancel_event=None, coordinator=None):
        coordinator = coordinator or self.coordinator
        if coordinator is None or coordinator is not self.coordinator or self._shutting_down:
            return self._retarget_reply(coordinator, text, {'status': 'retarget_rejected', 'reason': 'captured_session_changed'})
        valid_pending = target_request_id is None and capture.get('reference_context', {}).get('active_request_id') is None
        if (not isinstance(target_request_id, str) and not valid_pending) or capture.get('text') != text:
            coordinator.discard_interpretation(capture['interpretation_id'])
            return self._retarget_reply(coordinator, text, {'status': 'retarget_rejected', 'reason': 'captured_transcript_changed'})
        cancellation = _RetargetServiceCancellation(self, coordinator, cancel_event)
        return self._interpret_and_execute(coordinator, capture, text, capture['request_id'],
                                          cancel_event=cancellation, followup_target=target_request_id)

    def _voice_route_and_execute(self, coordinator, capture, text, request, *, before_execute=None, before_model=None,
                                 cancel_event=None):
        """Route a final transcript, binding the exact tested stop to its original task."""
        from .astra_offline_voice_turn import SPEECH_STOP
        from .scene_program_coordinator import StaleReference
        ident = capture['interpretation_id']
        owns_capture = True
        cancelled = self.cancelled if cancel_event is None else cancel_event
        try:
            if cancelled.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            context = capture['reference_context']
            at_onset = context.get('active_request_id')
            current = coordinator.snapshot()['reference_context'].get('active_request_id')
            if (capture.get('immediate_stop')
                    or (not capture.get('direct_audio') and _is_direct_stop_text(text))):
                stop = None
                if at_onset is not None and current == at_onset:
                    try:
                        # The capture already owns ``request``; a stop control
                        # needs its own ID while retaining the captured target.
                        stop = coordinator.request_motion_stop(_request_id(), target_request_id=at_onset)
                    except StaleReference:
                        pass  # The captured task finished or was replaced before the atomic stop.
                if stop is not None and stop.get('status') == 'stop_requested':
                    kind = 'bound_stop_proposal'
                    reply = 'Stop requested for the ' + ('virtual ' if self.mode == 'virtual' else '') + 'task. Its final state is not yet confirmed.'
                else:
                    kind = 'no_matching_active_request'
                    label = 'virtual task' if self.mode == 'virtual' else 'task'
                    reply = ('There was no active ' + label + ' when you spoke.' if at_onset is None
                             else 'The original ' + label + ' is no longer active; no other task was stopped.')
                self.emit('voice_route', {'kind': kind})
                result = {'reply': {'kind': kind, 'text': reply, 'filtered': False, 'speak': True},
                          'snapshot': coordinator.snapshot()}
                if stop is not None:
                    result['stop_request'] = stop
                return result
            if text.strip().casefold().rstrip('.!。！').strip() in SPEECH_STOP:
                return {'reply': {'kind': 'stop_voice_output', 'text': '음성 출력을 중단했습니다.',
                                  'speak': False, 'filtered': False}, 'snapshot': coordinator.snapshot()}
            if at_onset != current:
                return {'reply': {'kind': 'discard_context_change', 'text': '발화 당시 작업이 바뀌었습니다.',
                                  'speak': True, 'filtered': False}, 'snapshot': coordinator.snapshot()}
            self.emit('voice_route', {'kind': 'full_command_interpretation'})
            # Direct audio already has its model result bound to this capture.
            if before_model is not None and not capture.get('direct_audio'):
                before_model()
            owns_capture = False
            return self._interpret_and_execute(coordinator, capture, text, request,
                before_execute=before_execute, cancel_event=cancelled)
        finally:
            if owns_capture:
                coordinator.discard_interpretation(ident)

    def _refresh_stationary_reference_after_status_loss(self, coordinator, cancelled):
        """Read the resumed arm state without inferring an object or servo target."""
        with self._text_request_lock:
            owner = self._status_read_observation_refresh
            controller = self._controller
            if owner is None:
                return True
            if (owner[:3] != (controller, coordinator, self._registration_id)
                    or self.coordinator is not coordinator):
                return False
        try:
            observation = controller.observe()
        except Exception:
            return False  # no motion was requested; a later request can retry the read
        if cancelled.is_set():
            raise InterruptedError('요청을 취소했습니다.')
        if not _fresh_stationary_observation_valid(observation, owner[3], controller.config.limits):
            return False
        gripper = (observation.get('gripper_command'), observation.get('gripper_reported_state'))
        with self._text_request_lock, coordinator._lock:
            if (cancelled.is_set() or self._shutting_down):
                raise InterruptedError('요청을 취소했습니다.')
            if (self._status_read_observation_refresh != owner or self._controller is not controller
                    or self.coordinator is not coordinator or self._registration_id != owner[2]
                    or coordinator.snapshot().get('registration_revision') != owner[4]
                    or coordinator._active is not None or coordinator.session.cancel.is_set()):
                return False
            prior = (self._last_healthy_gripper[1:] if self._last_healthy_gripper is not None
                     and self._last_healthy_gripper[0] == owner else None)
            if prior is not None and gripper != prior:
                self._set_physical_recovery_locked(True)
                return False
            try:
                updated = coordinator.session.note_stationary_observation(observation)
            except (TypeError, ValueError):
                return False
            if updated is not True:
                return False
            self._status_read_observation_refresh = None
        return True

    def _await_temporary_connection(self, coordinator, ticket, request, cancelled, *, kind='ticket'):
        """Keep an admitted request unexecuted while idle feedback returns."""
        with self._text_request_lock:
            if self.mode != 'physical':
                return True
            if self.coordinator is not coordinator or self._shutting_down:
                return False
            if self._status_read_waiter is not None:
                raise RuntimeError('다른 요청의 연결 복구를 기다리고 있습니다.')
            waiter = (coordinator, ticket, request, kind)
            self._status_read_waiter = waiter
        announced = False
        try:
            while True:
                if cancelled.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                with self._text_request_lock:
                    if (self._shutting_down or self.mode != 'physical' or self.coordinator is not coordinator
                            or self._status_read_waiter is not waiter):
                        return False
                with coordinator._lock:
                    if kind == 'home':
                        pending = coordinator._arm_home_reservation
                        if (pending is None or pending['id'] != ticket or pending['cancelled'].is_set()):
                            raise InterruptedError('요청을 취소했습니다.')
                    elif ticket not in coordinator._pending:
                        raise InterruptedError('요청을 취소했습니다.')
                status = self.connection_snapshot()
                if cancelled.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                if (status.get('connection') == 'connected' and status.get('connected') is True
                        and not status.get('physical_recovery_required')):
                    return self._refresh_stationary_reference_after_status_loss(coordinator, cancelled)
                if not status.get('status_read_recovery_pending'):
                    return False
                if not announced:
                    self.emit('status', '연결이 돌아오면 요청을 이어서 실행합니다. 기다리는 동안 중단할 수 있습니다.')
                    announced = True
                wait = getattr(cancelled, 'wait', None)
                if callable(wait):
                    wait(.1)
                else:
                    time.sleep(.1)
        finally:
            with self._text_request_lock:
                if self._status_read_waiter is waiter:
                    self._status_read_waiter = None

    def _interpret_and_execute(self, coordinator, capture, text, request, *, before_execute=None,
                               cancel_event=None, stop_only=False, followup_target=None,
                               preinterpreted=None, plan_callback=None):
        from .scene_model_session import UnsupportedStep
        from .scene_program_coordinator import PendingTextPlanChanged, SessionNeedsRecovery

        ident = capture['interpretation_id']
        cancelled = self.cancelled if cancel_event is None else cancel_event
        try:
            if cancelled.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            self.emit('status', '로컬 언어 모델이 요청을 해석하고 있습니다.')
            interpret_started = time.monotonic()
            client = (None if capture.get('direct_audio') or preinterpreted is not None
                      else self.interpreter.start(cancelled).client)
            if cancelled.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            interpreted = None
            recovery_before_admission = False
            def report_interpretation(result):
                nonlocal interpreted, recovery_before_admission
                interpreted = result
                with self._text_request_lock:
                    pending_plan = self._pending_text_plans.get(request)
                    if pending_plan is not None:
                        pending_plan.update(result=copy.deepcopy(result), text=text)
                        pending_plan['ready'].set()
                    with coordinator._lock:
                        if coordinator._voice_turn_started.get(request) is False:
                            coordinator._pending_model_results[request] = copy.deepcopy(result)
                if plan_callback is not None:
                    plan_callback(result)
                self.emit('interpretation', {'status': result.status, 'program': result.program,
                                             'detail': result.detail, 'model_elapsed_ms': result.elapsed_ms,
                                             'total_elapsed_ms': (time.monotonic() - interpret_started) * 1000,
                                             'raw_proposal': result.raw_json, 'source': 'local_e4b'})
                if cancelled.is_set():
                    raise InterruptedError('요청을 취소했습니다.')
                # The display can mark recovery required immediately after
                # this very stop. Do not confuse that with an older fault.
                with self._text_request_lock:
                    recovery_before_admission = self._physical_recovery_required
            try:
                if preinterpreted is None:
                    interpreted, admission = coordinator.interpret_local_model_and_admit(
                        ident, client, timeout_seconds=180, before_admission=report_interpretation,
                        followup_target=followup_target, cancel_event=cancelled, stop_only=stop_only)
                else:
                    report_interpretation(preinterpreted)
                    interpreted, admission = coordinator.admit_local_model_result(
                        ident, preinterpreted, followup_target=followup_target,
                        cancel_event=cancelled, stop_only=stop_only)
            except SessionNeedsRecovery as exc:
                admission = {'kind': 'rejected', 'status': 'recovery_required',
                             'detail': str(exc), 'program_executed': False}
                coordinator.discard_interpretation(ident)
            except UnsupportedStep as exc:
                # An unexecutable plan rejected before issuing a ticket is
                # not an unknown physical result. Keep listening without
                # demanding registration or pretending ASR was at fault.
                from .native_physical_session import _ScopedPathUnavailable
                scoped = isinstance(exc, _ScopedPathUnavailable)
                infeasible = str(exc) in {
                    'single-cylinder target is not unique in the known source',
                    'the requested quantity does not fit the source and destination',
                    'destination height differs from the requested height',
                    'stack requires a two-level destination',
                    'unstack requires a two-level source',
                    'specified cylinder is not the source top object',
                    'the requested empty destination is occupied in the expected layout',
                }
                admission = {'kind': 'rejected',
                             'status': ('unsupported_TransferScope' if scoped else
                                        'infeasible_scene' if infeasible else 'preflight_rejected'),
                             'detail': str(exc), 'program_executed': False}
                if scoped:
                    admission['supported_slots'] = list(coordinator.session._active_slots)
                coordinator.discard_interpretation(ident)
            ident = None
            while True:
                try:
                    return self._execute_admitted_plan(
                        coordinator, admission, interpreted, text, request, cancelled,
                        before_execute=before_execute, stop_only=stop_only,
                        recovery_before_admission=recovery_before_admission)
                except PendingTextPlanChanged as changed:
                    # The ticket/home reservation was removed before this
                    # point. Reuse this request, cancellation and its already
                    # interpreted replacement; do not ask the model again.
                    capture = coordinator.recapture_pending_text_plan(
                        request, text, changed, cancel_event=cancelled)
                    ident = capture['interpretation_id']
                    interpreted, admission = coordinator.admit_local_model_result(
                        ident, changed.result, cancel_event=cancelled, stop_only=stop_only)
                    ident = None
                    with self._text_request_lock:
                        recovery_before_admission = self._physical_recovery_required
        finally:
            if ident is not None:
                coordinator.discard_interpretation(ident)

    @staticmethod
    def _release_unclaimed_arm_home(coordinator, reservation_id):
        with coordinator._lock:
            pending = coordinator._arm_home_reservation
            if pending is not None and pending['id'] == reservation_id and not pending['claimed']:
                coordinator._arm_home_reservation = None
                coordinator._bump()

    def _execute_admitted_plan(self, coordinator, admission, interpreted, text, request, cancelled,
                               *, before_execute, stop_only, recovery_before_admission):
        if admission['kind'] == 'arm_home':
            if admission.get('target_request_id') is None:
                try:
                    ready = self._await_temporary_connection(coordinator, admission['reservation'], request,
                                                             cancelled, kind='home')
                except BaseException:
                    self._release_unclaimed_arm_home(coordinator, admission['reservation'])
                    raise
                if not ready:
                    self._release_unclaimed_arm_home(coordinator, admission['reservation'])
                    return {'admission': {'kind': 'rejected', 'status': 'connection_unavailable',
                                          'program_executed': False},
                            'snapshot': coordinator.snapshot(),
                            'voice_feedback': '연결을 확인하지 못해 요청을 실행하지 않았습니다. 연결 후 다시 말씀해 주세요.'}
                recovery_before_admission = False
            try:
                if before_execute is not None and admission.get('target_request_id') is None:
                    before_execute(interpreted.program)
            except BaseException:
                self._release_unclaimed_arm_home(coordinator, admission['reservation'])
                raise
            self.emit('status', '정지를 확인한 뒤 팔을 초기 자세로 이동합니다.')
            result = self._execute_arm_home(coordinator, admission, cancelled,
                                            recovery_before_admission=recovery_before_admission)
            interrupted = result.get('status') == 'arm_home_interrupted'
            feedback = ('팔 초기 자세 이동을 마치지 못했습니다. 기존 예상 배치를 유지하며 상태를 확인하고 있습니다.'
                        if interrupted else
                        '팔이 초기 자세에 도착하고 정지가 확인됐습니다. 중단된 물체와 집게 상태는 확인되지 않았습니다. '
                        '현장 배치를 확인한 뒤 다시 등록해 주세요.' if admission.get('target_request_id') else
                        '팔이 초기 자세에 도착하고 정지가 확인됐습니다.')
            return {'result': result, 'snapshot': coordinator.snapshot(),
                    'reply': {'kind': 'arm_home_interrupted' if interrupted else 'arm_home_completed', 'text': feedback,
                              'filtered': False, 'speak': True}, 'voice_feedback': feedback}
        if admission['kind'] != 'ticket':
            if admission['kind'] == 'retarget':
                return self._retarget_reply(coordinator, text, admission['retarget'])
            snapshot = coordinator.snapshot()
            if admission['kind'] == 'status':
                context = snapshot.get('reference_context') or {}
                scene = (snapshot.get('scene_state') or {}).get('expected_scene')
                feedback = ('현재 작업이 진행 중입니다.' if context.get('active_request_id')
                            else '현재 진행 중인 작업이 없습니다.')
                if 'current_state' in admission['subjects'] and scene is not None:
                    feedback += ' 예상 배치는 ' + str([list(stack) for stack in scene]) + '입니다.'
                if 'repeat' in admission['subjects']:
                    feedback += ' 반복 상태는 ' + str(context.get('repeat_state', 'unknown')) + '입니다.'
            elif admission.get('status') == 'active_followup_only':
                feedback = '진행 중인 운반의 목적지 변경이나 중단을 요청할 수 있습니다.'
            elif admission.get('unresolved') and admission['kind'] != 'stop_prefix':
                feedback = ' '.join(q['question'].strip() for q in admission.get('clarifications', [])
                                    if isinstance(q.get('question'), str) and q['question'].strip())
                feedback = feedback or '요청할 동작이나 대상을 조금 더 구체적으로 말씀해 주세요.'
            elif admission.get('status') == 'recovery_required' and snapshot.get('last_execution'):
                waiting_release = (((snapshot.get('scene_state') or {}).get('unresolved_transfer') or {})
                                   .get('release_expected') is True)
                feedback = _execution_failure_message(snapshot['last_execution']) + (
                    ' 연결 상태를 확인하고 있습니다.' if waiting_release else ' 배치를 다시 등록해 주세요.')
            else:
                feedback = _voice_feedback('', interpreted, admission, virtual=self.mode == 'virtual')
            return {'admission': admission, 'snapshot': snapshot, 'voice_feedback': feedback}
        ticket = admission['ticket']
        if stop_only:
            with coordinator._lock:
                coordinator._pending.pop(ticket, None)
            return {'admission': {'kind': 'rejected', 'status': 'stop_only', 'program_executed': False},
                    'snapshot': coordinator.snapshot(),
                    'voice_feedback': '즉시 정지 외의 작업은 실행하지 않았습니다.'}
        try:
            connected = self._await_temporary_connection(coordinator, ticket, request, cancelled)
        except BaseException:
            with coordinator._lock:
                coordinator._pending.pop(ticket, None)
            raise
        if not connected:
            with coordinator._lock:
                coordinator._pending.pop(ticket, None)
            return {'admission': {'kind': 'rejected', 'status': 'connection_unavailable',
                                  'program_executed': False},
                    'snapshot': coordinator.snapshot(),
                    'voice_feedback': '연결을 확인하지 못해 요청을 실행하지 않았습니다. 연결 후 다시 말씀해 주세요.'}
        if before_execute is not None:
            try:
                before_execute(interpreted.program)
            except BaseException:
                with coordinator._lock:
                    coordinator._pending.pop(ticket, None)
                raise
        # A stop before execute bumps the coordinator revision. Thus the
        # ticket cannot start even if the stop races this check.
        if cancelled.is_set():
            with coordinator._lock:
                coordinator._pending.pop(ticket, None)
            raise InterruptedError('요청을 취소했습니다.')
        self.emit('status', ('가상 로봇' if self.mode == 'virtual' else '제어기 작업') +
                  '이 실행 중입니다.')
        result = coordinator.execute(ticket, cancel_event=cancelled)
        return {'result': result, 'snapshot': coordinator.snapshot()}

    def _stop_candidates(self):
        pending = [entry for entry in (self._text_request, self._record_request) if entry is not None]
        candidates = {entry[0] for entry in pending}
        coordinator = self.coordinator
        if coordinator is not None:
            with coordinator._lock:
                active = coordinator.snapshot()['reference_context']['active_request_id']
                if active is not None:
                    candidates.add(active)
                if coordinator._arm_home_reservation is not None:
                    candidates.add(coordinator._arm_home_reservation['request_id'])
        return candidates

    def arm_home_request_id(self):
        coordinator = self.coordinator
        if coordinator is None:
            return None
        with coordinator._lock:
            reservation = coordinator._arm_home_reservation
            return reservation['request_id'] if reservation is not None else None

    def motion_stop_requested(self, request_id):
        """Check a recorded stop for one request, independent of transient UI flags."""
        coordinator = self.coordinator
        if coordinator is None or not isinstance(request_id, str):
            return False
        with coordinator._lock:
            return any(event.get('event') == 'stop_requested' and event.get('target_request_id') == request_id
                       for event in reversed(coordinator.events))

    def begin_shutdown(self):
        """Prevent new service requests while existing independent work drains."""
        with self._text_request_lock:
            self._shutting_down = True

    def has_other_requests(self, own_request_ids):
        """Check captured, ticketed and active work without changing ownership."""
        own = {item for item in own_request_ids if item is not None}
        with self._text_request_lock:
            requests = {entry[0] for entry in (self._text_request, self._record_request) if entry is not None}
            requests.update(self._registration_requests)
            requests.update(self._transcription_requests)
            requests.update(self._conversation_requests)
            if self._controller_preparation is not None:
                requests.add(self._controller_preparation[0].request_id)
            if self._controller_connecting is not None:
                requests.add(self._controller_connecting[0].request_id)
            if self._controller_recovery is not None:
                requests.add(self._controller_recovery[0].request_id)
            coordinator = self.coordinator
            if coordinator is not None:
                with coordinator._lock:
                    if coordinator._active is not None:
                        requests.add(coordinator._active['request_id'])
                    if coordinator._arm_home_reservation is not None:
                        requests.add(coordinator._arm_home_reservation['request_id'])
                    requests.update(entry['request_id'] for entry in coordinator._interpretations.values())
                    requests.update(entry['program']['request_id'] for entry in coordinator._pending.values())
            return bool(requests - own)

    def stop(self, *, target_request_id=_NO_STOP_TARGET):
        """Cancel exactly one identified request without a global stop flag."""
        with self._text_request_lock:
            if self._automatic_recovery is not None:
                self._automatic_recovery[0].cancel()
            self._automatic_recovered = None
            pending = [entry for entry in (self._text_request, self._record_request) if entry is not None]
            coordinator = self.coordinator
            if target_request_id is _NO_STOP_TARGET:
                target_request_id = getattr(self._request_context, 'request_id', None)
                if target_request_id is None:
                    candidates = self._stop_candidates()
                    if len(candidates) > 1:
                        return {'status': 'ambiguous_target'}
                    return {'status': 'target_required' if candidates else 'no_active_task'}
            matched = next((entry for entry in pending if entry[0] == target_request_id), None)
            if matched is not None:
                coordinator = matched[2]
            if coordinator is None or target_request_id is None:
                return {'status': 'no_matching_active_request'}
            with coordinator._lock:
                active = coordinator.snapshot()['reference_context']['active_request_id']
                reservation = coordinator._arm_home_reservation
                if (reservation is not None and reservation['request_id'] == target_request_id
                        and active != target_request_id):
                    reservation['cancelled'].set()
                    if matched is not None:
                        matched[1].set()
                        coordinator.cancel_pending_voice(target_request_id)
                    return {'status': 'cancelled_pending_request', 'target_request_id': target_request_id,
                            'stop_confirmation': 'not_requested'}
                if (active != target_request_id
                        and any(event.get('event') == 'program_finished'
                                and event.get('request_id') == target_request_id
                                and event.get('status') == 'completed' for event in coordinator.events)):
                    return {'status': 'already_completed', 'target_request_id': target_request_id,
                            'stop_confirmation': 'not_requested'}
                if matched is not None:
                    matched[1].set()
                    for ticket, entry in list(coordinator._pending.items()):
                        if entry['program']['request_id'] == target_request_id:
                            coordinator._pending.pop(ticket, None)
                if active == target_request_id:
                    return coordinator.request_motion_stop(_request_id(), target_request_id=target_request_id)
            if matched is not None:
                return {'status': 'cancelled_pending_request', 'target_request_id': target_request_id,
                        'stop_confirmation': 'not_requested'}
            return {'status': 'no_matching_active_request'}

    def close(self):
        self.begin_shutdown()
        errors = []
        forecast_attempted = None

        def close_controller_forecast(forecast):
            nonlocal forecast_attempted
            forecast_attempted = forecast
            forecast.close()

        with self._text_request_lock:
            if self._controller is not None:
                if self.has_other_requests(set()):
                    raise RuntimeError('진행 중인 작업이 끝난 뒤 제어기를 종료해 주세요.')
                try:
                    self._disconnect_controller_locked(forecast_closer=close_controller_forecast)
                except Exception as exc:
                    errors.append(exc)
                else:
                    self.coordinator = self.sim = self.model_hash = None
                    self._registration_id = None
        # Once other owners have drained, each of these resources can be
        # closed independently.  A failure must not skip the remaining ones.
        if self.forecast is not None and self.forecast is not forecast_attempted:
            try:
                self.forecast.close(wait=True)
            except Exception as exc:
                errors.append(exc)
            else:
                self.forecast = None
        try:
            self.interpreter.close()
        except Exception as exc:
            errors.append(exc)
        try:
            with self._voice_lock:
                voice = None if self._voice_closed else self.voice
                self._voice_closed = True
            if voice is not None and hasattr(voice, 'close'):
                voice.close()
        except Exception as exc:
            errors.append(exc)
        if errors:
            first, *later = errors
            for exc in later:
                first.add_note('추가 자원 종료 오류: ' + repr(exc))
            raise first
