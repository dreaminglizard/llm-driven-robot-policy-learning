"""Loopback-only browser interface for the robot demonstration.

The simulator, interpreter and voice conversation remain owned by DemoService.
Hardware and microphone use require separate explicit actions.
"""
from __future__ import annotations

import argparse
import json
import math
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import product_instance
from .demo_controller import PhysicalProfileValidationError
from .demo_service import (BEST_CLOSE_SHA256, BEST_CLOSE_STUDENT, CLOSE_SHA256, CLOSE_STUDENT,
                           CLOSED_MOVE_PROFILES, CONTROL_TIMING_PROFILES, DEFAULT_STUDENT, FINGER_OPEN_PROFILES,
                           RL_SHA256, RL_STUDENT, DemoService, _AuxiliaryOperation,
                           _execution_failure_message, _is_direct_stop_text, _is_text_stop_candidate, _request_id)

ASSETS = Path(__file__).with_name('demo_web_assets')
MODELS = {
    'default': ('기존 접근거리 학생', DEFAULT_STUDENT, None),
    'close': ('집게·거리 결합 학생 · 기본', BEST_CLOSE_STUDENT, BEST_CLOSE_SHA256),
    'close_previous': ('집게·거리 결합 학생 · 기존 시드', CLOSE_STUDENT, CLOSE_SHA256),
    'rl': ('강화학습 후보 · 실험용', RL_STUDENT, RL_SHA256),
}


class RegistrationStateChanged(ValueError):
    """The coordinates were submitted against an older screen state."""


def parse_layout(values):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError('네 자리의 배치를 확인해 주세요.')
    stacks = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError('원통 번호는 쉼표로 구분해 입력해 주세요.')
        parts = [part.strip() for part in value.split(',')] if value.strip() else []
        if any(part not in ('1', '2', '3', '4') for part in parts):
            raise ValueError('원통 번호는 1~4이며, 쉼표로 아래층부터 입력해 주세요.')
        stacks.append([int(part) for part in parts])
    ids = [item for stack in stacks for item in stack]
    if any(len(stack) > 2 for stack in stacks) or not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise ValueError('원통은 1~4개, 번호는 중복 없이, 각 자리는 최대 2층이어야 합니다.')
    return stacks


def parse_registration_coordinates(data):
    if 'slot_xy_mm' not in data and 'table_z_mm' not in data:
        return {}
    xy, z = data.get('slot_xy_mm'), data.get('table_z_mm', 0)
    def finite(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if (not isinstance(xy, list) or len(xy) != 4
            or any(not isinstance(pair, list) or len(pair) != 2 or not all(finite(value) for value in pair) for pair in xy)
            or not finite(z)):
        raise ValueError('네 자리의 X·Y에 유효한 숫자를 입력해 주세요.')
    if z != 0:
        raise ValueError('현재 테이블 높이는 0 mm 기준만 지원합니다.')
    # Reject the same coordinates as the physical calibration before pausing
    # listening or invalidating the existing registration.
    from .robot_bridge import validate_slot_coordinates
    validate_slot_coordinates(xy)
    return {'slot_xy_mm': [list(pair) for pair in xy], 'table_z_mm': z}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if hasattr(value, 'item'):
        return _plain(value.item())
    return str(value)


def _describe_program(program):
    names = {'move': '이동', 'stack': '쌓기', 'unstack': '풀기', 'swap': '교환',
             'arrange': '배치', 'remember': '현재 배치 기억', 'register': '배치 등록'}
    targets = {'registered_initial': '등록된 처음 배치로 복원', 'learned_memory': '기억한 배치로 복원',
               'task_start': '직전 작업 시작 배치로 복원', 'arm_home': '팔 대기 자세로 복귀'}
    labels = []
    for step in (program or {}).get('steps', []):
        op = step.get('op')
        if op == 'task':
            label = names.get(step.get('kind'), '작업')
            if step.get('source_slot') is not None and step.get('destination_slot') is not None:
                label += f" · {step['source_slot']}번 → {step['destination_slot']}번"
            if step.get('object_id') is not None:
                label += f" · 원통 {step['object_id']}"
        elif op == 'return':
            label = targets.get(step.get('target', {}).get('kind'), '복귀 대상 확인 필요')
        else:
            label = {'clarify': '추가 확인 필요', 'repeat': '반복', 'stop_motion': '중단',
                     'stop_repeat': '반복 종료', 'continue': '계속', 'status': '상태 확인'}.get(op, str(op))
        labels.append(label)
    return ' → '.join(labels) if labels else '해석된 작업이 없습니다.'


class _DeferredSetupConversation:
    """Start setup listening without holding the web state lock during imports."""

    def __init__(self, owner, language):
        self.owner, self.language = owner, language
        self.stop_event, self.closed = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._chat = None

    def current_target(self):
        return None

    def stop_motion(self):
        return False

    def start(self):
        threading.Thread(target=self._start, daemon=True, name='setup-voice-start').start()

    def stop(self):
        self.stop_event.set()
        def stop_started_chat():
            with self._lock:
                chat = self._chat
            if chat is not None:
                try:
                    chat.stop()
                except Exception as exc:
                    if hasattr(chat, 'stop_event'):
                        chat.stop_event.set()
                    with self.owner.lock:
                        self.owner._log('음성 대화 종료 오류: ' + str(exc))
        threading.Thread(target=stop_started_chat, daemon=True, name='setup-voice-stop').start()

    def finish_registration(self):
        self.stop_event.set()
        def finish_started_chat():
            with self._lock:
                chat = self._chat
            if chat is not None:
                try:
                    finish = getattr(chat, 'finish_registration', None)
                    (finish if callable(finish) else chat.stop)()
                except Exception as exc:
                    if hasattr(chat, 'stop_event'):
                        chat.stop_event.set()
                    with self.owner.lock:
                        self.owner._log('음성 대화 종료 오류: ' + str(exc))
        threading.Thread(target=finish_started_chat, daemon=True, name='setup-voice-finish').start()

    def _emit(self, kind, value):
        if kind == 'chat_closed':
            self.closed.set()
        self.owner.emit(kind, value, source_chat=self)

    def _start(self):
        try:
            from .demo_setup_voice import SetupConversation
            if self.stop_event.is_set():
                self._emit('chat_closed', None)
                return
            chat = SetupConversation(self._emit, language=self.language, audio_service=self.owner.service,
                                     stop_target=self.owner._setup_voice_target,
                                     stop_captured=self.owner._stop_setup_voice_target)
            with self._lock:
                cancelled = self.stop_event.is_set()
                if not cancelled:
                    self._chat = chat
                    chat.start()
            if cancelled:
                self._emit('chat_closed', None)
        except Exception as exc:
            self.closed.set()
            if self.stop_event.is_set():
                self._emit('chat_closed', None)
                return
            with self.owner.lock:
                if self.owner.chat is self:
                    self.owner.chat = None
                    self.owner.state['chat'] = False
                    self.owner.state['chat_setup_only'] = False
                    self.owner.state['chat_listening'] = False
                    self.owner._auto_voice_active = False
                    self.owner.chat_error = '마이크를 사용할 수 없습니다. 운영체제의 마이크 권한과 장치를 확인한 뒤 다시 듣기를 눌러 주세요: ' + str(exc)
                    self.owner.state['status'] = self.owner.chat_error
                    self.owner._log(self.owner.chat_error)


class DemoController:
    def __init__(self, *, prepare_model=True, service_factory=DemoService, development=False):
        self.lock = threading.RLock()
        self._chat_changed = threading.Condition(self.lock)
        self.development = development is True
        self.service = service_factory(self.emit, visual_pace_seconds=.02)
        # Selecting the physical mode does not connect a controller or issue a
        # command. Connection remains a separate, explicit user action.
        initial_mode = 'virtual' if self.development else 'physical'
        if initial_mode == 'physical':
            self.service.set_mode(initial_mode)
        self.worker = None
        self.stop_worker = None
        self.text_control_worker = None
        self._text_control_workers = set()
        self._text_target = None
        self._aux_operation = None
        self._registration_preparation = None
        self._registered_service_id = None
        self._registered_coordinator_revision = None
        self.chat_stop_worker = None
        self._chat_stop_returned = True
        self.chat = None
        self.chat_error = None
        self._auto_voice_active = False
        self._auto_voice_after_registration = False
        self._auto_voice_language = 'auto'
        self.closing = False
        self._close_started = False
        self._close_complete = threading.Event()
        self._close_error = None
        self._stopped_motion_requests = set()
        self._latest_motion_numbers = {}
        self._view_generation = 0
        self._snapshot_revision = 0
        self._connection_recovery_revision = 0
        self._controller_reconnect_revision = 0
        self._status_read_interruption_seen = False
        self._pending_connect_screen = None
        self._pending_connect_unknown = False
        self.state = {
            'mode': initial_mode, 'development_mode': self.development,
            'registered': False, 'registration_revision': 0, 'recovery_required': False, 'recovery_guarded': False,
            'release_waiting_connection': False,
            'busy': None, 'chat': False, 'chat_setup_only': False, 'chat_listening': False, 'chat_stopping': False,
            'chat_audio_mode': None,
            'stop_requested': False, 'text_control_pending': False, 'text_target': None, 'closing': False,
            'motion_stop_target': None,
            'status': ('시작 배치와 학생 모델을 선택하고 등록해 주세요.' if self.development
                       else '실물 로봇을 연결하고 준비 상태와 시작 배치를 확인해 주세요.'),
            'connection_error': None,
            'llm': {'state': 'waiting', 'message': '로컬 언어 모델 준비 대기 중'},
            'scene_label': '입력한 설정 · 아직 등록되지 않음',
            'layout': [[1], [2], [3], [4]],
            'registration_coordinates': None,
            'observation': None,
            'path': [], 'forecast': None, 'samples': [], 'motion': None, 'model': None,
            'finger_open_profile': None, 'closed_move_profile': None, 'control_timing_profile': None,
            'straight_descent': False,
            'logs': [], 'transcript': '',
            'models': [{'id': key, 'name': name, 'available': path.is_file()}
                       for key, (name, path, _) in MODELS.items()],
        }
        if prepare_model:
            self.service.interpreter.prepare_async(self._prepare_event)

    def _log(self, message):
        self.state['logs'] = (self.state['logs'] + [str(message)])[-40:]

    def _mode_name(self):
        return {'virtual': '가상', 'mock': '모의', 'physical': '실물'}.get(self.state.get('mode'), '가상')

    def _prepare_event(self, value):
        with self.lock:
            self.state['llm'] = _plain(value)

    def _motion_request_stopped(self, request_id):
        if not isinstance(request_id, str):
            return False
        if request_id in self._stopped_motion_requests:
            return True
        check = getattr(self.service, 'motion_stop_requested', None)
        if callable(check) and check(request_id):
            self._stopped_motion_requests.add(request_id)
            self.state['motion_stop_target'] = request_id
            return True
        return False

    def _require_chat_recovery(self):
        if self.chat is not None:
            restrict = getattr(self.chat, 'require_recovery', None)
            if callable(restrict):
                restrict()

    def _registration_preserves_memory(self):
        coordinator = self.service.coordinator
        controller = getattr(self.service, '_controller', None)
        return bool(getattr(self.service, 'mode', None) == 'physical' and controller is not None
                    and coordinator is not None and coordinator.session.status == 'uncertain'
                    and getattr(coordinator.session, 'controller', None) is controller)

    def _recovery_memory_message(self):
        return ('배치 복구 시 이전 기억은 유지됩니다.' if self._registration_preserves_memory()
                else '재등록하면 이전 기억이 지워집니다.')

    def _invalidate_registration(self):
        self._require_chat_recovery()
        self.state['registered'] = False
        self.state['recovery_required'] = True
        self.state['recovery_guarded'] = False
        self.state['layout'] = None
        self.state['observation'] = None
        self.state['path'] = []
        self.state['forecast'] = None
        self.state['scene_label'] = '배치가 다른 경로에서 변경됨 · 다시 등록 필요'
        self.state['status'] = self._mode_name() + ' 배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.'

    def _snapshot(self, snapshot, *, in_motion=False):
        if not isinstance(snapshot, dict):
            return
        if (self.state['registered']
                and snapshot.get('registration_revision') != self._registered_coordinator_revision):
            self._invalidate_registration()
            return
        scene = snapshot.get('scene_state')
        if not isinstance(scene, dict):
            return
        # An expected layout is symbolic bookkeeping, not an observation of
        # where the cylinders are. A missing layout must not preserve an older
        # drawing as if it were the current one.
        self.state['layout'] = _plain(scene.get('expected_scene'))
        self.state['release_waiting_connection'] = bool(
            (scene.get('unresolved_transfer') or {}).get('release_expected') is True
            or scene.get('connection_recovery_pending'))
        context = snapshot.get('reference_context') or {}
        guarded = (snapshot.get('session_status') == 'uncertain'
                   or context.get('execution_state') == 'uncertain'
                   or snapshot.get('stop_confirmation') == 'unconfirmed')
        self.state['recovery_guarded'] = guarded
        self.state['recovery_required'] = (self.state['recovery_required'] or
            self.state['layout'] is None
            or scene.get('confirmation') not in ('explicit_registration', 'caller_confirmed', 'unconfirmed_after_motion')
            or guarded)
        if self.state['recovery_required']:
            self._require_chat_recovery()
        confirmation = scene.get('confirmation')
        self.state['scene_confirmation'] = confirmation
        if self.state['layout'] is None:
            self.state['scene_label'] = '배치 표시 불가 · 다시 등록 필요'
        elif self.state['recovery_required']:
            self.state['scene_label'] = ('놓기 후 예상 배치 · 팔과 연결 상태 미확인'
                if (scene.get('unresolved_transfer') or {}).get('release_expected') is True
                else '기존 예상 배치 · 팔과 연결 상태 미확인' if scene.get('connection_recovery_pending')
                else '마지막 예상 배치 · 이동 결과 미확인 · 다시 등록 필요')
        elif in_motion:
            self.state['scene_label'] = self._mode_name() + ' 조작 중 · 마지막 예상 배치 · 현재 물체 위치 미확인'
        else:
            self.state['scene_label'] = {
                'explicit_registration': '등록된 시작 배치 · 물체 실측 없음',
                'caller_confirmed': '사용자가 확인한 배치 · 물체 센서 실측 없음',
                'unconfirmed_after_motion': '예상 배치 · 물체 위치 미확인',
            }.get(confirmation, '배치 미확인 · 다시 등록 필요')

    def emit(self, kind, value, *, source_chat=None):
        if source_chat is not None:
            with self.lock:
                if self.chat is not source_chat or (source_chat.stop_event.is_set()
                        and kind not in ('chat_closed', 'chat_input_closed')):
                    return
        motion_coordinator = None
        motion_snapshot = None
        if (kind == 'motion' and value.get('phase') == 'start') or kind == 'chat_error':
            # Pick up newly committed expectations at each command, and
            # refresh the recovery state after an error in voice conversation.
            motion_coordinator = self.service.coordinator
            if motion_coordinator is not None:
                try:
                    motion_snapshot = motion_coordinator.snapshot()
                except Exception:
                    pass
        with self.lock:
            if source_chat is not None and (self.chat is not source_chat
                    or (source_chat.stop_event.is_set() and kind not in ('chat_closed', 'chat_input_closed'))):
                return
            if kind == 'observation':
                self.state['observation'] = _plain(value)
            elif kind == 'motion_stop_requested':
                target = value.get('target_request_id')
                current_target = (self._text_target if self.state['busy'] == 'submit' and self._text_target is not None
                                  else self.chat.current_target() if self.chat is not None else self._text_target)
                current_target = current_target or (self.state['motion'] or {}).get('request_id')
                if not isinstance(target, str) or (current_target is not None and current_target != target):
                    return
                self._stopped_motion_requests.add(target)
                self._require_chat_recovery()
                self.state['motion_stop_target'] = target
                self.state['path'] = []
                self.state['forecast'] = None
                motion = self.state['motion'] or {}
                if motion.get('phase') != 'interrupted' or motion.get('request_id') != target:
                    self.state['status'] = self._mode_name() + ' 작업 중단을 요청했습니다. 결과는 아직 확인되지 않았습니다.'
            elif kind == 'motion':
                value = _plain(value)
                live_session = getattr(getattr(self.service, 'forecast', None), 'session_token', None)
                if (live_session and value.get('forecast_session')
                        and value['forecast_session'] != live_session):
                    return
                current_target = (self._text_target if self.state['busy'] == 'submit' and self._text_target is not None
                                  else self.chat.current_target() if self.chat is not None else self._text_target)
                if (value.get('request_id') is not None and current_target is not None
                        and value['request_id'] != current_target):
                    return
                session, number = value.get('forecast_session'), value.get('number')
                if session and type(number) is int:
                    if number < self._latest_motion_numbers.get(session, 0):
                        return
                    self._latest_motion_numbers[session] = number
                stopped = self._motion_request_stopped(value.get('request_id'))
                if stopped and value.get('phase') not in ('end', 'interrupted'):
                    self.state['path'] = []
                    self.state['forecast'] = None
                    return
                if motion_snapshot is not None and motion_coordinator is self.service.coordinator:
                    self._snapshot(motion_snapshot, in_motion=True)
                self.state['observation'] = value['observation']
                tcp = value['observation'].get('tcp_mm')
                if tcp:
                    self.state['samples'] = (self.state['samples'] + [tcp])[-250:]
                self.state['path'] = value.get('remaining_target_path_mm', [])
                if (value.get('phase') in ('start', 'end', 'interrupted') or value.get('cancel_requested')
                        or self.state['stop_requested']):
                    self.state['forecast'] = None
                if value.get('phase') == 'interrupted':
                    self._require_chat_recovery()
                self.state['motion'] = value
            elif kind == 'forecast':
                motion = self.state['motion']
                sequence = value.get('forecast_sequence')
                current_sequence = (self.state['forecast'] or {}).get('forecast_sequence')
                newer = (type(sequence) is not int or type(current_sequence) is not int
                         or sequence >= current_sequence)
                if (motion and not self._motion_request_stopped(motion.get('request_id'))
                        and motion.get('phase') not in ('end', 'interrupted') and not motion.get('cancel_requested')
                        and newer
                        and not self.state['stop_requested'] and not self.service.cancelled.is_set()
                        and (self.state['forecast'] or {}).get('reason') != 'cancelled'
                        and motion.get('forecast_session')
                        and value.get('forecast_session') == motion.get('forecast_session')
                        and value.get('command_number') == motion.get('number')):
                    self.state['forecast'] = _plain(value)
            elif kind == 'forecast_skipped':
                motion = self.state['motion']
                if (motion and not self._motion_request_stopped(motion.get('request_id'))
                        and motion.get('phase') not in ('end', 'interrupted') and not motion.get('cancel_requested')
                        and not self.state['stop_requested'] and not self.service.cancelled.is_set()
                        and (self.state['forecast'] or {}).get('reason') != 'cancelled'
                        and motion.get('forecast_session')
                        and motion.get('forecast_session') == value.get('forecast_session')
                        and motion.get('number') == value.get('command_number')):
                    self.state['forecast'] = {'reason': 'skipped', 'segments': []}
            elif kind == 'forecast_clear':
                motion = self.state['motion']
                if (motion and motion.get('forecast_session')
                        and motion.get('forecast_session') == value.get('forecast_session')
                        and motion.get('number') == value.get('command_number')):
                    self.state['forecast'] = {'reason': 'cancelled', 'segments': []}
            elif kind == 'motion_error':
                self.state['path'] = []
                self.state['forecast'] = None
                self.state['motion'] = None
                self._log(self._mode_name() + ' 명령 오류로 예상 경로 표시를 중지했습니다.')
            elif kind == 'chat_input_opened':
                self.state['chat_listening'] = True
            elif kind == 'chat_input_closed':
                self.state['chat_listening'] = False
                if (self.state['chat_setup_only'] and self._auto_voice_after_registration
                        and not self.closing and not self.state['chat_stopping']):
                    self.state['status'] = '음성 설정을 마저 처리한 뒤 듣기를 다시 시작합니다.'
            elif kind == 'chat_audio_mode':
                self.state['chat_audio_mode'] = value.get('mode') if isinstance(value, dict) else None
            elif kind == 'chat_utterance_start':
                self.chat_error = None
            elif kind in ('status', 'chat_state'):
                if kind == 'chat_state' and self.chat_error and '듣' in str(value):
                    return
                target = (self._text_target if self.state['busy'] == 'submit' and self._text_target is not None
                          else self.chat.current_target() if self.chat is not None else self._text_target)
                if kind == 'status' and self._motion_request_stopped(target):
                    return
                motion = self.state['motion'] or {}
                stop_target = self.state['motion_stop_target']
                if (kind == 'chat_state' and stop_target is not None
                        and target in (None, stop_target) and motion.get('request_id') == stop_target
                        and ('듣' in str(value) or '음성 응답 중' in str(value))):
                    return
                self.state['status'] = (str(value) + (' 제어기 연결을 확인하고 있습니다.'
                                        if self.state['release_waiting_connection'] else
                                        ' 이동이나 새 기억 저장 전에는 배치를 다시 등록해 주세요. '
                                        + self._recovery_memory_message())
                                        if kind == 'chat_state' and self.state['recovery_required'] else str(value))
                if kind == 'status' and '실행 중' in str(value):
                    self.state['scene_label'] = '요청 실행 중 · 마지막 표시 배치 · 현재 물체 위치 미확인'
            elif kind == 'interpretation':
                source = '로컬 규칙' if value.get('source') == 'local_rule' else '로컬 언어 모델'
                self._log('요청 해석: ' + source + ' · ' + str(value.get('status', '')))
                if value.get('program'):
                    self._log('이해한 작업: ' + _describe_program(value['program']))
                elif value.get('detail'):
                    self._log(value['detail'])
            elif kind in ('transcript', 'chat_transcript'):
                self.state['transcript'] = value.get('text', '')
                self._log('인식한 말: ' + self.state['transcript'])
            elif kind == 'chat_response':
                self._log('음성 안내: ' + str(value))
            elif kind == 'chat_error':
                if motion_snapshot is not None and motion_coordinator is self.service.coordinator:
                    self._snapshot(motion_snapshot)
                elif self.state['registered']:
                    self.state['recovery_required'] = True
                    self._require_chat_recovery()
                    self.state['scene_label'] = '상태 미확인 · 다시 등록 필요'
                self.chat_error = str(value)
                self.state['status'] = self.chat_error + (' 음성 대화를 종료한 뒤 배치를 다시 등록해 주세요. '
                                                         + self._recovery_memory_message()
                                                         if self.state['recovery_required'] else '')
                self._log(self.chat_error)
            elif kind == 'chat_result':
                # A later completed request supersedes an earlier recoverable error.
                # Its error remains visible in the activity log.
                self.chat_error = None
                self._show_result(value)
            elif kind == 'chat_closed':
                was_setup = self.state['chat_setup_only']
                self.chat = None
                self._chat_changed.notify_all()
                self.state['chat'] = False
                self.state['chat_setup_only'] = False
                self.state['chat_listening'] = False
                if was_setup and self.chat_error and not self._auto_voice_after_registration:
                    self._auto_voice_active = False
                if self._chat_stop_returned:
                    self.state['chat_stopping'] = False
                self.state['path'] = []
                self.state['forecast'] = None
                self._clear_motion_unless_interrupted()
                self.state['status'] = self.chat_error or '음성 대화가 종료됐습니다.'
                if self.state['recovery_required']:
                    self.state['status'] += ' 배치를 다시 등록해 주세요. ' + self._recovery_memory_message()
                self._log(self.state['status'])
                self._resume_automatic_voice()

    def _clear_motion_unless_interrupted(self):
        if (self.state['motion'] or {}).get('phase') != 'interrupted':
            self.state['motion'] = None

    def _show_result(self, result):
        execution = result.get('result')
        snapshot = result.get('snapshot')
        # A result produced before an already-applied connection recovery is
        # still an incomplete result, but must not roll the live screen or
        # listener back to its older scene and recovery state.
        if (self.state['registered'] and isinstance(snapshot, dict)
                and snapshot.get('registration_revision') == self._registered_coordinator_revision
                and self._connection_recovery_revision > 0):
            old_revision = (snapshot.get('scene_state') or {}).get('connection_recovery_revision', 0)
            with self.service._text_request_lock:
                coordinator = self.service.coordinator
                if (type(old_revision) is int and old_revision < self._connection_recovery_revision
                        and coordinator is not None and self.service._registration_id == self._registered_service_id
                        and not self.service._physical_recovery_required and not self.service.cancelled.is_set()
                        and not self.state['stop_requested']):
                    with coordinator._lock:
                        latest = coordinator.snapshot()
                        if (latest.get('registration_revision') == self._registered_coordinator_revision
                                and latest.get('session_status') == 'registered'
                                and latest.get('stop_confirmation') != 'unconfirmed'
                                and not coordinator.session.cancel.is_set()
                                and (latest.get('scene_state') or {}).get('connection_recovery_revision')
                                    == self._connection_recovery_revision):
                            message = ((result.get('reply') or {}).get('text') or result.get('voice_feedback')
                                or (_execution_failure_message(execution) if isinstance(execution, dict)
                                    and execution.get('status') != 'completed' else '이전 요청이 끝났습니다.'))
                            self._log(message + ' 이후 상태는 이미 복구됐습니다.')
                            return
        if (isinstance(execution, dict)
                and (execution.get('status') == 'uncertain' or execution.get('stop_confirmation') == 'unconfirmed')):
            # A later external recovery can change the coordinator snapshot;
            # the already-finished request still records an uncertain outcome.
            self.state['recovery_required'] = True
            self._require_chat_recovery()
        if (self.state['registered'] and isinstance(snapshot, dict)
                and snapshot.get('registration_revision') != self._registered_coordinator_revision):
            self._invalidate_registration()
        else:
            self._snapshot(snapshot)
        # A redirect acknowledgement or rejection is not the completion of
        # the original motion; keep its live visualization intact.
        if result.get('retarget') is None and result.get('in_motion_followup') is not True:
            self.state['path'] = []
            self.state['forecast'] = None
            self._clear_motion_unless_interrupted()
        if result.get('reply') is not None:
            message = result['reply'].get('text', '')
        elif execution is None:
            message = result.get('voice_feedback') or '요청을 실행하지 않았습니다. 해석 결과를 확인해 주세요.'
        else:
            status = execution.get('status')
            if self.state['recovery_required']:
                message = _execution_failure_message(execution)
                released = any(isinstance(event, dict) and event.get('release_expected') is True
                               for event in execution.get('events') or [])
                if released:
                    message += ' 연결 상태를 확인하고 있습니다.'
                else:
                    message += (' 음성 대화를 종료한 뒤 배치를 다시 등록해 주세요.' if self.state['chat']
                                else ' 배치를 다시 등록해 주세요.')
            else:
                labels = {'completed': '동작을 마쳤습니다.',
                          'motion_not_observed': _execution_failure_message(execution),
                          'step_unavailable': (_execution_failure_message(execution)
                              if (execution.get('not_started_reason') == 'controller_not_ready'
                                  or any(isinstance(event, dict) and event.get('event') == 'transfer'
                                         and event.get('not_started_reason') == 'controller_not_ready'
                                         for event in execution.get('events') or []))
                              else '현재 배치에서는 다음 작업을 실행할 수 없음'),
                          'retargeted_expected': '변경한 동작을 마쳤습니다.',
                          'retarget_not_applied': '요청한 목표 변경이 적용되지 않음',
                          'retarget_incomplete': '변경한 동작을 마치지 못했습니다.',
                          'retarget_unconfirmed': '목표 변경 결과를 확인하지 못했습니다.',
                          'repeat_stopped': '반복 종료',
                          'repeat_unavailable': '반복할 작업이 없음', 'no_progress': '추가 진행 없음'}
                message = labels.get(status, self._mode_name() + ' 실행 상태: ' + str(status or 'unknown'))
            if execution.get('detail'):
                self._log(str(execution['detail']))
        if self.state['recovery_required'] and execution is None:
            message += (' 연결 상태를 확인하고 있습니다.' if self.state['release_waiting_connection'] else
                        ' 음성 대화를 종료한 뒤 배치를 다시 등록해 주세요.' if self.state['chat'] else
                        ' 배치를 다시 등록해 주세요.')
        if self.state['recovery_required'] and not self.state['release_waiting_connection']:
            message += ' ' + self._recovery_memory_message()
        self.state['status'] = message
        self._log(message)

    def snapshot(self):
        self.service.interpreter.check_liveness()
        # A resident controller may need a slow status RPC. Read it without
        # holding the web or service locks so a stop can proceed independently.
        with self.lock:
            generation = self._view_generation
        with self.service._text_request_lock:
            before = (getattr(self.service, 'mode', 'virtual'), getattr(self.service, '_controller', None),
                      self.service.coordinator, self.service._registration_id)
        control_snapshot = getattr(self.service, 'connection_snapshot', None)
        control = control_snapshot() if callable(control_snapshot) else {
            'mode': 'virtual', 'connection': 'not_applicable', 'connected': False,
            'simulated': True, 'observation': None, 'current_command': None,
            'remaining_target_path_mm': [], 'requirements': []}
        control = _plain(control)
        with self.lock, self.service._text_request_lock:
            after = (getattr(self.service, 'mode', 'virtual'), getattr(self.service, '_controller', None),
                     self.service.coordinator, self.service._registration_id)
            changed_during_read = (before[0] != after[0] or before[1] is not after[1] or before[2] is not after[2]
                                   or before[3] != after[3] or generation != self._view_generation)
            if changed_during_read:
                control = {'mode': after[0], 'connection': 'unknown', 'connected': False,
                           'observation': None, 'current_command': None, 'remaining_target_path_mm': [],
                           'requirements': ['제어 상태가 바뀌어 다시 확인하고 있습니다.']}
            service_mode = control.get('mode', 'virtual')
            if service_mode != self.state['mode']:
                self.state['mode'] = service_mode
                if self.state['registered']:
                    self._invalidate_registration()
            if self.state['registered'] and self._registered_service_id != self.service._registration_id:
                self._invalidate_registration()
            if (not changed_during_read and self.state['registered'] and service_mode != 'virtual'
                    and (not control.get('connected') or control.get('connection') == 'fault')
                    and not control.get('status_read_recovery_pending')):
                self.state['recovery_required'] = True
                self._require_chat_recovery()
                self.state['path'] = []
                self.state['forecast'] = None
                scene = control.get('scene_state') or {}
                released = (scene.get('unresolved_transfer') or {}).get('release_expected') is True
                waiting_release = bool(scene.get('connection_recovery_pending')) or released
                self.state['scene_label'] = ('놓기 후 예상 배치 · 연결 상태 미확인' if released
                                             else '기존 예상 배치 · 연결 상태 미확인' if waiting_release
                                             else '마지막 예상 배치 · 연결 상태 미확인 · 다시 등록 필요')
                self.state['status'] = (('제어기 상태를 확인하고 있습니다.' if control.get('connection') == 'connected'
                                         else '제어기 연결이 끊겼습니다. 수신이 돌아오는지 확인하고 있습니다.') if waiting_release
                                        else '제어기 연결이나 상태를 확인할 수 없습니다. 다시 연결한 뒤 배치를 다시 등록해 주세요.')
            pending = getattr(self.service, '_pending_physical_start', None)
            owner = self.service._controller
            connecting = getattr(self.service, '_controller_connecting', None)
            attachment = getattr(owner, 'attachment', None)
            controller_status = control.get('controller')
            status_fields = controller_status if isinstance(controller_status, dict) else {}
            active_command = status_fields.get('active_command_id')
            display_command = control.get('current_command')
            if not isinstance(display_command, dict):
                display_command = {}
            live_connect = (self.state['busy'] == 'connect' and self._aux_operation is not None
                            and not self._aux_operation.is_set())
            awaiting_child = (pending is not None and pending[0][0].request_id == self._pending_connect_screen
                              and pending[2] == self.service._physical_recovery_epoch and not pending[1].is_set()
                              and self.service._physical_connection_uncertain and owner is None
                              and not controller_status and not control.get('ownership')
                              and control.get('connection') in ('disconnected', 'connecting')
                              and (not self.state['busy'] or live_connect))
            initializing_child = (live_connect and self._pending_connect_screen is not None and owner is not None
                                  and connecting is not None and connecting[0] is self._aux_operation
                                  and connecting[1] is owner and isinstance(attachment, dict)
                                  and attachment.get('resumed_start') == self._pending_connect_screen
                                  and control.get('connection') == 'connecting'
                                  and isinstance(controller_status, dict) and 'fault' in controller_status
                                  and controller_status['fault'] is None
                                  and controller_status.get('worker_alive') is True
                                  and controller_status.get('fresh_feedback') is True
                                  and controller_status.get('state') in ('idle', 'armed', 'executing', 'stopped')
                                  and controller_status.get('lease_owner') in
                                      (None, 'demo-connect-stop-check', 'demo-connect-initial-pose')
                                  and (active_command is None or active_command == display_command.get('command_id')))
            pending_connect_retry = bool(
                not changed_during_read and service_mode == 'physical' and (awaiting_child or initializing_child)
                and not self.service.cancelled.is_set() and not self.closing and not self.state['stop_requested']
                and not self.state['registered'] and control.get('physical_recovery_required'))
            if not changed_during_read and service_mode == 'physical' and control.get('physical_recovery_required'):
                self.state['path'] = []
                self.state['forecast'] = None
                if control.get('status_read_recovery_pending'):
                    self._status_read_interruption_seen = True
                    if not self.state['recovery_required']:
                        self.state['status'] = '제어기 연결 상태를 다시 확인하고 있습니다.'
                else:
                    self._status_read_interruption_seen = False
                    self.state['recovery_required'] = True
                    self.state['recovery_guarded'] = False
                    recovery_scene = control.get('scene_state') or {}
                    expected_connection = bool(recovery_scene.get('connection_recovery_pending'))
                    self.state['scene_label'] = (('제어기 연결 결과 미확인 · 배치 등록 전'
                                                  if self._pending_connect_unknown and not self.state['busy']
                                                  else '연결 준비 중 · 배치 등록 전') if pending_connect_retry
                                                 else '상태 복구 중 · 기존 예상 배치' if expected_connection
                                                 else '실물 정지 확인 및 배치 재등록 필요')
                    stopping_connect = (self.state['busy'] == 'connect' and (self.state['stop_requested']
                                        or (self._aux_operation is not None and self._aux_operation.is_set())))
                    if not pending_connect_retry and not stopping_connect and not self.state.get('connection_error'):
                        scene = control.get('scene_state') or {}
                        waiting_release = (bool(scene.get('connection_recovery_pending'))
                            or (scene.get('unresolved_transfer') or {}).get('release_expected') is True)
                        self.state['status'] = (('제어기 상태를 확인하고 있습니다.' if control.get('connection') == 'connected'
                                                 else '제어기 연결이 끊겼습니다. 수신이 돌아오는지 확인하고 있습니다.') if waiting_release
                                                else '현장 배치를 확인하고 제어기 정지와 오류 복구를 요청해 주세요.')
            elif (not changed_during_read and self._status_read_interruption_seen
                  and not control.get('physical_recovery_required')):
                self._status_read_interruption_seen = False
                if (self.state['registered'] and not self.state['recovery_required']
                        and not self.state['stop_requested']):
                    self.state['status'] = '제어기 연결 상태가 복구됐습니다.'
            reconnect_revision = control.get('controller_reconnect_revision', 0)
            if (not changed_during_read and service_mode == 'physical' and not self.state['registered']
                    and type(reconnect_revision) is int and reconnect_revision > self._controller_reconnect_revision
                    and control.get('connection') == 'connected' and control.get('connected') is True
                    and not control.get('physical_recovery_required') and not control.get('initial_pose_required')
                    and not self.service.cancelled.is_set() and not self.state['stop_requested']):
                self._controller_reconnect_revision = reconnect_revision
                self.state['recovery_required'] = False
                self.state['recovery_guarded'] = False
                self.state['connection_error'] = None
                self.state['status'] = '로봇 연결을 복구했습니다. 시작 배치를 등록해 주세요.'
            coordinator = self.service.coordinator
            if self.state['registered'] and coordinator is not None:
                with coordinator._lock:
                    latest = coordinator.snapshot()
                    if latest.get('registration_revision') != self._registered_coordinator_revision:
                        self._invalidate_registration()
                    else:
                        scene = latest.get('scene_state') or {}
                        revision = scene.get('connection_recovery_revision', 0)
                        if (type(revision) is int and revision > self._connection_recovery_revision
                                and not self.service._physical_recovery_required and not self.service.cancelled.is_set()
                                and control.get('connection') == 'connected' and control.get('connected') is True
                                and latest.get('session_status') == 'registered'
                                and latest.get('stop_confirmation') != 'unconfirmed'
                                and not coordinator.session.cancel.is_set() and not self.state['stop_requested']):
                            self._connection_recovery_revision = revision
                            self.state['recovery_required'] = False
                            self.state['recovery_guarded'] = False
                            if self.chat is not None:
                                allow = getattr(self.chat, 'allow_after_connection_recovery', None)
                                if callable(allow):
                                    allow()
                            self._snapshot(latest)
                            self.state['status'] = '상태가 복구되어 기존 예상 배치를 유지합니다. 물체 위치는 실측하지 않았습니다.'
            state = _plain(self.state)
            state['control'] = control
            state['registration_preserves_memory'] = self._registration_preserves_memory()
            state['coordinate_registration_revision_required'] = True
            state['pending_connect_retry'] = pending_connect_retry
            state['pending_connect_unknown'] = pending_connect_retry and self._pending_connect_unknown
            self._snapshot_revision += 1
            state['view_revision'] = self._snapshot_revision
            if service_mode != 'virtual':
                # Device feedback is read from the controller's latest sample,
                # never from a previous virtual or disconnected session.
                state['observation'] = control.get('observation') if control.get('connected') else None
                if not control.get('connected'):
                    state['samples'] = []
                    state['path'] = []
                    state['forecast'] = None
            home = getattr(self.service, 'arm_home_request_id', lambda: None)()
            state['arm_home_active'] = home is not None
            if home is not None:
                state['text_target'] = home
            elif self.state['busy'] == 'submit' and self._text_target is not None:
                state['text_target'] = self._text_target
            elif self.chat is not None:
                state['text_target'] = self.chat.current_target()
            return state

    def _start_worker(self, action, operation, *, auxiliary=None):
        with self.lock:
            chat_blocks = (self.chat is not None and not self.state['chat_setup_only']
                           and action != 'submit')
            if (self.closing or self.state['busy'] or chat_blocks or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            self._aux_operation = auxiliary
            self.state['busy'] = action
            self.state['stop_requested'] = False
            self.state['path'] = []
            self.state['forecast'] = None
            self.state['samples'] = []
            self.state['motion'] = None

            def run():
                try:
                    result = operation()
                    with self.lock:
                        if action == 'register':
                            with self.service._text_request_lock:
                                if result.get('registration_id') != self.service._registration_id:
                                    raise RuntimeError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
                                self._registered_service_id = result.get('registration_id')
                                self._registered_coordinator_revision = result['snapshot'].get('registration_revision')
                                self._connection_recovery_revision = (result['snapshot'].get('scene_state') or {}).get(
                                    'connection_recovery_revision', 0)
                            self.state['registered'] = True
                            self.state['recovery_required'] = False
                            self.state['recovery_guarded'] = False
                            self._snapshot(result['snapshot'])
                            self.state['model'] = {'name': MODELS[self._selected_model][0],
                                                   'sha256': result['model_sha256']}
                            self.state['finger_open_profile'] = result['finger_open_profile']
                            self.state['closed_move_profile'] = result['closed_move_profile']
                            self.state['control_timing_profile'] = result.get('control_timing_profile')
                            self.state['straight_descent'] = result.get('straight_descent', False)
                            warnings = result.get('post_commit_warnings') or []
                            self.state['status'] = ('가상 시작 배치는 등록됐지만 부가 처리에 오류가 있었습니다. 로그를 확인해 주세요.'
                                                    if warnings else
                                                    '가상 시작 배치를 등록했습니다. 물체의 실제 상태는 확인하지 않았습니다.')
                            if self.state['mode'] != 'virtual':
                                self.state['status'] = (self._mode_name() + ' 시작 배치는 등록됐지만 부가 처리에 오류가 있었습니다. 로그를 확인해 주세요.'
                                                        if warnings else self._mode_name() + ' 시작 배치를 등록했습니다. 입력한 배치이며 물체를 실측하지 않았습니다.')
                            self._log(self.state['status'])
                            if not self.development and self._auto_voice_active:
                                self._auto_voice_after_registration = True
                                if self.chat is not None and self.state['chat_setup_only']:
                                    finish = getattr(self.chat, 'finish_registration', None)
                                    (finish if callable(finish) else self.chat.stop)()
                            for warning in warnings:
                                self._log('등록 후 오류 (' + str(warning.get('stage')) + '): '
                                          + str(warning.get('detail')))
                        elif action == 'connect':
                            if (result.get('connected') is not True or result.get('connection') != 'connected'
                                    or result.get('physical_recovery_required') or result.get('initial_pose_required')):
                                raise RuntimeError('연결 또는 초기 자세의 완료 상태를 확인하지 못했습니다.')
                            if (result.get('resumed_pending_start') is not None
                                    and result['resumed_pending_start'] == self._pending_connect_screen):
                                # Invalidate status reads begun before this recovery cleared.
                                self._view_generation += 1
                                self.state['recovery_required'] = False
                                self.state['recovery_guarded'] = False
                            self._pending_connect_screen = None
                            self._pending_connect_unknown = False
                            self.state['connection_error'] = None
                            self.state['status'] = '실물 로봇 연결과 초기 자세를 확인했습니다. 시작 배치를 등록해 주세요.'
                            self._log(self.state['status'])
                            if not self.development:
                                self._auto_voice_active = True
                                try:
                                    self._start_setup_voice(self._auto_voice_language)
                                except Exception as exc:
                                    self._auto_voice_active = False
                                    self.chat_error = '마이크를 사용할 수 없습니다. 운영체제의 마이크 권한과 장치를 확인한 뒤 다시 듣기를 눌러 주세요: ' + str(exc)
                                    self.state['status'] = self.chat_error
                                    self._log(self.chat_error)
                        elif action == 'prepare_gripper':
                            if result.get('status') != 'prepared' or result.get('gripper_reported_confirmed') is not True:
                                raise RuntimeError('집게 준비 완료와 피드백을 확인하지 못했습니다.')
                            self.state['status'] = (self._mode_name() + ' 집게 준비 요청이 끝났습니다. '
                                                    '집게 피드백을 확인하고 시작 배치를 등록해 주세요.')
                            self._log(self.state['status'])
                        elif action == 'recover_controller':
                            if result.get('status') == 'no_fault':
                                raise RuntimeError('제어기 오류는 없지만 별도의 정지 확인은 완료되지 않았습니다.')
                            if result.get('status') == 'stop_unconfirmed':
                                raise RuntimeError('제어기의 정지를 확인하지 못했습니다.')
                            if (result.get('status') not in ('recovered', 'stop_confirmed')
                                    or (result.get('arm_stop') or {}).get('confirmed') is not True):
                                raise RuntimeError('제어기 복구와 정지 확인을 완료하지 못했습니다.')
                            self.state['recovery_required'] = False
                            self.state['recovery_guarded'] = False
                            self.state['connection_error'] = None
                            recovered = result.get('status') == 'recovered'
                            self.state['scene_label'] = ('제어기 복구됨' if recovered else '팔 정지 확인됨') + ' · 배치 재등록 필요'
                            self.state['status'] = (('제어기 오류 복구와 팔 정지를 확인했습니다.' if recovered else '제어기 오류가 없고 팔 정지를 확인했습니다.')
                                                    + ' 집게를 준비하고 현재 배치를 다시 등록해 주세요.')
                            if self.service._physical_initial_pose_required:
                                self.state['status'] = ('팔 정지를 확인했습니다. 초기 자세가 미확인이므로 연결을 해제하고 '
                                                        '다시 연결한 뒤 집게와 배치를 준비해 주세요.')
                            self._log(self.state['status'])
                        else:
                            self._show_result(result)
                except Exception as exc:
                    failure_snapshot = None
                    if action not in ('register', 'connect', 'prepare_gripper', 'recover_controller') and self.service.coordinator is not None:
                        try:
                            failure_snapshot = self.service.coordinator.snapshot()
                        except Exception:
                            pass
                    with self.lock:
                        retry_profile = False
                        if action == 'connect' and isinstance(exc, PhysicalProfileValidationError):
                            with self.service._text_request_lock:
                                retry_profile = (not self.state['recovery_required']
                                    and self.service.mode == 'physical' and self.service._controller is None
                                    and not self.service._physical_recovery_required
                                    and not self.service._physical_connection_uncertain)
                        if action == 'connect':
                            with self.service._text_request_lock:
                                pending_start = self.service._pending_physical_start
                                if pending_start is not None:
                                    pending_id = pending_start[0][0].request_id
                                    if self._pending_connect_screen != pending_id:
                                        if auxiliary is not None and not auxiliary.preexisting_screen_recovery:
                                            self._pending_connect_screen = pending_id
                                            self._pending_connect_unknown = False
                                        else:
                                            self.service._pending_physical_start = None
                                            pending_start = None
                                    if (pending_start is not None
                                            and getattr(exc, 'code', None) == 'APP_CONTROLLER_READY_CONNECTION_UNKNOWN'):
                                        self._pending_connect_unknown = True
                                else:
                                    self._pending_connect_screen = None
                                    self._pending_connect_unknown = False
                            if retry_profile:
                                self.state['scene_label'] = '연결 설정 확인 · 수정 후 다시 연결'
                            else:
                                self.state['recovery_required'] = True
                                self.state['scene_label'] = '연결 또는 초기 자세 미확인 · 정지와 장치 확인 필요'
                        elif action in ('prepare_gripper', 'recover_controller'):
                            self.state['recovery_required'] = True
                            self.state['scene_label'] = ('집게 준비 결과 미확인 · 연결을 확인해 주세요' if action == 'prepare_gripper'
                                                         else '제어기 복구 결과 미확인 · 배치 재등록 필요')
                        elif action != 'register':
                            if failure_snapshot is not None:
                                self._snapshot(failure_snapshot)
                            else:
                                self.state['recovery_required'] = True
                                self.state['scene_label'] = '실행 결과 미확인 · 다시 등록 필요'
                        if action == 'connect':
                            self.state['status'] = (('로봇 연결 설정을 확인해 주세요: ' + str(exc)
                                                     + ' 설정을 수정한 뒤 다시 연결해 주세요.') if retry_profile else
                                                    (('시작한 제어기의 연결 결과를 확인하지 못했습니다. 연결을 다시 눌러 확인해 주세요.'
                                                      if self._pending_connect_unknown else
                                                      '연결 준비가 계속되고 있습니다. 잠시 뒤 연결을 다시 눌러 주세요.') if pending_start is not None else
                                                    ('로봇 연결 또는 초기 자세를 완료하지 못했습니다: ' + str(exc)
                                                     + ' 장치 상태와 팔 정지를 확인해 주세요.')))
                            self.state['connection_error'] = self.state['status']
                        else:
                            self.state['status'] = ('요청을 취소했습니다.' if isinstance(exc, InterruptedError)
                                                    else '완료하지 못했습니다: ' + str(exc))
                        if action == 'prepare_gripper':
                            self.state['status'] += ' 제어기를 다시 연결하고 준비 상태를 확인해 주세요.'
                        elif action == 'recover_controller':
                            self.state['status'] += ' 정지와 오류 상태를 확인한 뒤 다시 시도해 주세요.'
                        elif action != 'connect' and self.state['recovery_required']:
                            self.state['status'] += ' 배치를 다시 등록해 주세요.'
                        if action == 'register':
                            self.state['scene_label'] = '등록하지 못한 설정 · 다시 등록해 주세요'
                            if self._auto_voice_after_registration and not self.state['registered']:
                                self.state['status'] += ' 음성 듣기는 다시 시작되지 않았습니다.'
                        self._log(self.state['status'])
                finally:
                    with self.lock:
                        self._view_generation += 1
                        if action == 'submit':
                            self._text_target = None
                        if auxiliary is not None and self._aux_operation is auxiliary:
                            self._aux_operation = None
                        if action == 'register':
                            self._registration_preparation = None
                        self.state['busy'] = None
                        self.state['path'] = []
                        self.state['forecast'] = None
                        self._clear_motion_unless_interrupted()
                        self._resume_automatic_voice()
            self.worker = threading.Thread(target=run, daemon=True, name='demo-web-worker')
            self.worker.start()

    def register(self, data):
        layout = parse_layout(data.get('layout'))
        coordinate_options = parse_registration_coordinates(data)
        voice_language = data.get('language')
        if 'language' in data and voice_language not in ('auto', 'ko', 'en'):
            raise ValueError('음성 인식 언어를 다시 선택해 주세요.')
        model = data.get('model')
        if model not in MODELS or not MODELS[model][1].is_file():
            raise ValueError('사용 가능한 학생 모델을 선택해 주세요.')
        profile = data.get('finger_open_profile', 'auto')
        if not isinstance(profile, str) or (profile != 'auto' and profile not in FINGER_OPEN_PROFILES):
            raise ValueError('지원되는 가상 집게 열기 방식을 선택해 주세요.')
        move_profile = data.get('closed_move_profile', 'auto')
        if not isinstance(move_profile, str) or (move_profile != 'auto' and move_profile not in CLOSED_MOVE_PROFILES):
            raise ValueError('지원되는 가상 이동 곡선을 선택해 주세요.')
        timing_profile = data.get('control_timing_profile', 'auto')
        if not isinstance(timing_profile, str) or (timing_profile != 'auto' and timing_profile not in CONTROL_TIMING_PROFILES):
            raise ValueError('지원되는 가상 이동 시간을 선택해 주세요.')
        prepare_if_needed = data.get('prepare_gripper_if_needed', False)
        if type(prepare_if_needed) is not bool:
            raise ValueError('집게 준비 요청을 확인해 주세요.')
        with self.lock:
            if self.state['mode'] == 'physical' and coordinate_options:
                revision = data.get('registration_revision')
                if type(revision) is not int or revision != self.state['registration_revision']:
                    raise RegistrationStateChanged('연결 또는 배치 설정이 바뀌었습니다. 좌표를 다시 확인해 주세요.')
            live_chat = self.chat if self.chat is not None and not self.state['chat_setup_only'] else None
            may_pause_chat = (live_chat is not None and not self.development
                              and self.state['mode'] == 'physical' and self.state['registered'])
            if (self.closing or self.state['busy'] or (live_chat is not None and not may_pause_chat)
                    or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            if 'mode' in data and data['mode'] != self.state['mode']:
                raise ValueError('다른 화면에서 모드가 바뀌었습니다. 현재 모드를 확인해 주세요.')
            if self.state['mode'] != 'virtual' and self.state['recovery_required']:
                raise ValueError('제어기를 다시 연결하고 집게를 준비한 뒤 배치를 등록해 주세요.')
            if prepare_if_needed and (self.development or self.state['mode'] != 'physical'):
                raise ValueError('실물 배치 등록에서만 집게 준비를 함께 요청할 수 있습니다.')
            physical_options = {}
            if self.state['mode'] == 'physical':
                if data.get('controller_ready_confirmed') is not True:
                    raise ValueError('현장 정지, 집게, 배치와 장치 준비를 확인해 주세요.')
                physical_options = {'controller_ready_confirmed': True}
                if self.development:
                    target = data.get('initial_applied_target_deg')
                    if (not isinstance(target, list) or len(target) != 4
                            or any(type(value) not in (int, float) or not math.isfinite(value) for value in target)):
                        raise ValueError('확인한 마지막 적용 목표각 네 개를 입력해 주세요.')
                    physical_options['initial_applied_target_deg'] = target
                    evidence = data.get('initial_applied_target_evidence')
                    if evidence is not None:
                        if (not isinstance(evidence, dict) or set(evidence) != {'instance_id', 'command_id'}
                                or not all(isinstance(evidence[key], str) and evidence[key] for key in evidence)):
                            raise ValueError('확인된 목표각의 출처를 다시 확인해 주세요.')
                        physical_options['initial_applied_target_evidence'] = dict(evidence)
                elif 'initial_applied_target_deg' in data or 'initial_applied_target_evidence' in data:
                    raise ValueError('실물 초기 상태는 연결된 기기의 관측으로 확인합니다.')
            if may_pause_chat:
                pause = getattr(live_chat, 'pause_if_idle_for_registration', None)
                if not callable(pause) or not pause():
                    raise ValueError('음성 입력이나 작업 상태를 확인하고 잠시 후 다시 등록해 주세요.')
                transition = _AuxiliaryOperation()
                self._aux_operation = transition
                previous_auto_active = self._auto_voice_active
                previous_auto_after = self._auto_voice_after_registration
                self._auto_voice_active = True
                self._auto_voice_after_registration = True
                self._auto_voice_language = voice_language or live_chat.language
                self.state['busy'] = 'registration_voice_pause'
                self.state['status'] = '배치를 등록하기 위해 음성 듣기를 잠시 정리하고 있습니다.'
                self._log(self.state['status'])
                deadline = time.monotonic() + 10
                while (self.chat is live_chat and not live_chat.registration_pause_declined()
                       and not transition.is_set() and not self.state['chat_stopping'] and not self.closing
                       and time.monotonic() < deadline):
                    self._chat_changed.wait(min(.1, max(0, deadline - time.monotonic())))
                with self.service._text_request_lock:
                    same_registration = (self.service.coordinator is live_chat.coordinator
                        and self._registered_service_id == self.service._registration_id)
                    if same_registration:
                        with live_chat.coordinator._lock:
                            same_registration = (live_chat.coordinator.snapshot().get('registration_revision')
                                                 == self._registered_coordinator_revision)
                closed = (self.chat is None and not transition.is_set() and not self.closing
                          and self.state['busy'] == 'registration_voice_pause'
                          and not self.state['stop_requested'] and not self.state['chat_stopping']
                          and not self.state['recovery_required']
                          and same_registration)
                if self.state['busy'] == 'registration_voice_pause':
                    self.state['busy'] = None
                if self._aux_operation is transition:
                    self._aux_operation = None
                if not closed:
                    declined = live_chat.registration_pause_declined()
                    listener_stopping = live_chat.cancel_registration_pause()
                    if not listener_stopping and self._auto_voice_after_registration:
                        self._auto_voice_active = previous_auto_active
                        self._auto_voice_after_registration = previous_auto_after
                    message = ('말씀이나 작업을 처리하고 있습니다. 끝난 뒤 다시 등록해 주세요.' if declined
                               else '배치 등록 요청을 취소했습니다.' if transition.is_set() or self.state['stop_requested']
                               else '상태가 바뀌었거나 음성 듣기 종료를 확인하지 못해 배치를 등록하지 않았습니다.')
                    self.state['status'] = message
                    self._log(message)
                    self._resume_automatic_voice()
                    raise ValueError(message)
            self._selected_model = model
            # This applies only to the next conversation. The setup listener
            # keeps the language with which its current capture was started.
            if voice_language is not None:
                self._auto_voice_language = voice_language
            self.state['registration_revision'] += 1
            self.state['registered'] = False
            self._registered_service_id = None
            self._registered_coordinator_revision = None
            self.state['recovery_required'] = False
            self.state['recovery_guarded'] = False
            self.state['model'] = None
            self.state['finger_open_profile'] = None
            self.state['closed_move_profile'] = None
            self.state['control_timing_profile'] = None
            self.state['straight_descent'] = False
            self.state['layout'] = layout
            self.state['registration_coordinates'] = coordinate_options or None
            self.state['observation'] = None
            self.state['scene_label'] = '입력한 설정 · 등록 중'
            self.state['status'] = self._mode_name() + ' 배치와 학생 모델을 준비하고 있습니다.'
            _, path, digest = MODELS[model]
            auxiliary = _AuxiliaryOperation()
            preparation = _AuxiliaryOperation() if prepare_if_needed else None
            self._registration_preparation = preparation

            def register_operation():
                if preparation is not None:
                    control = self.service.connection_snapshot()
                    observation = control.get('observation') or {}
                    status = control.get('controller') or {}
                    if not control.get('connected') or control.get('connection') != 'connected':
                        raise RuntimeError('제어기 연결 상태를 확인한 뒤 다시 등록해 주세요.')
                    ready = (observation.get('sensor_valid') is True and observation.get('moving') is False
                             and observation.get('gripper_command') == 'open'
                             and observation.get('gripper_reported_state') == 'open'
                             and observation.get('gripper_reported_valid') is True
                             and observation.get('gripper_feedback_fresh') is not False
                             and status.get('fault') is None and status.get('state') in ('idle', 'stopped'))
                    if auxiliary.is_set() or preparation.is_set():
                        raise InterruptedError('등록을 취소했습니다.')
                    if not ready:
                        with self.lock:
                            self.state['status'] = '집게를 열고 상태를 확인한 뒤 배치를 등록합니다.'
                            self._log(self.state['status'])
                        try:
                            self.service.prepare_gripper(operation=preparation, physical_confirmed=True)
                        except Exception:
                            with self.lock, self.service._text_request_lock:
                                if self.service._physical_recovery_required:
                                    self.state['recovery_required'] = True
                            raise
                    if auxiliary.is_set():
                        raise InterruptedError('등록을 취소했습니다.')
                return self.service.register(
                    layout, path, digest, finger_open_profile=profile, closed_move_profile=move_profile,
                    control_timing_profile=timing_profile, operation=auxiliary, **physical_options, **coordinate_options)

            try:
                self._start_worker('register', register_operation, auxiliary=auxiliary)
            except Exception:
                self._registration_preparation = None
                raise
            return self.state['registration_revision']

    def set_mode(self, data):
        mode = data.get('mode')
        if mode not in ('virtual', 'mock', 'physical'):
            raise ValueError('지원되는 제어 모드를 선택해 주세요.')
        if not self.development and mode != 'physical':
            raise ValueError('일반 시연에서는 실물 로봇만 사용할 수 있습니다.')
        with self.lock:
            if (self.closing or self.state['busy'] or self.chat is not None or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            if mode == self.state['mode']:
                return self.state['registration_revision']
            self.service.set_mode(mode)
            self._pending_connect_screen = None
            self.state.update(mode=mode, registered=False, recovery_required=False, recovery_guarded=False,
                              layout=None, observation=None, path=[], forecast=None, samples=[], motion=None,
                              model=None, finger_open_profile=None, closed_move_profile=None, control_timing_profile=None,
                              straight_descent=False, scene_confirmation=None,
                              scene_label='모드 변경 · 시작 배치를 다시 등록해 주세요',
                              status={'virtual': '가상', 'mock': '모의', 'physical': '실물'}[mode]
                                     + ' 모드를 선택했습니다. 배치를 다시 등록해 주세요.')
            self._registered_service_id = None
            self._registered_coordinator_revision = None
            self.state['registration_revision'] += 1
            self._log(self.state['status'])
            return self.state['registration_revision']

    def _require_current_mode(self, data):
        if data.get('mode') != self.state['mode']:
            raise ValueError('다른 화면에서 모드가 바뀌었거나 선택한 모드가 없습니다. 현재 모드를 확인해 주세요.')

    def _setup_voice_target(self):
        with self.lock:
            if self.state['busy'] == 'register' and self._registration_preparation is not None:
                return self._registration_preparation.request_id
            if self.state['busy'] in ('prepare_gripper', 'recover_controller') and self._aux_operation is not None:
                return self._aux_operation.request_id
            return None

    def _stop_setup_voice_target(self, target):
        with self.lock:
            operation = (self._registration_preparation if self.state['busy'] == 'register'
                         else self._aux_operation if self.state['busy'] in ('prepare_gripper', 'recover_controller')
                         else None)
            if operation is None or operation.request_id != target:
                return False
            self.stop()
            return True

    def _start_setup_voice(self, language):
        self._auto_voice_language = language
        self.chat_error = None
        self.state['transcript'] = ''
        self.chat = _DeferredSetupConversation(self, language)
        self.state['chat'] = True
        self.state['chat_setup_only'] = True
        self.state['chat_listening'] = False
        self.state['chat_stopping'] = False
        self._chat_stop_returned = True
        self.state['status'] = '음성을 준비하고 있습니다.'
        try:
            self.chat.start()
        except Exception:
            self.chat = None
            self.state['chat'] = False
            self.state['chat_setup_only'] = False
            raise

    def _resume_automatic_voice(self):
        if (not self._auto_voice_after_registration or self.chat is not None or self.state['busy']
                or self.state['chat_stopping'] or self.state['stop_requested']
                or self.closing or not self.state['registered']):
            return
        self._auto_voice_after_registration = False
        try:
            self.start_chat({'language': self._auto_voice_language,
                             'registration_revision': self.state['registration_revision']})
        except Exception as exc:
            self._auto_voice_active = False
            self.chat_error = '음성 대화를 시작하지 못했습니다. 마이크 권한과 장치를 확인하고 다시 듣기를 눌러 주세요: ' + str(exc)
            self.state['status'] = self.chat_error
            self._log(self.chat_error)

    def connect_controller(self, data):
        language = data.get('language', 'auto')
        if language not in ('auto', 'ko', 'en'):
            raise ValueError('음성 인식 언어를 다시 선택해 주세요.')
        with self.lock:
            if (self.closing or self.state['busy'] or self.chat is not None or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            self._require_current_mode(data)
            if self.state['mode'] == 'virtual':
                raise ValueError('가상 모드는 제어기 연결이 필요하지 않습니다.')
            if self.state['mode'] == 'physical':
                auxiliary = _AuxiliaryOperation()
                auxiliary.preexisting_screen_recovery = self.state['recovery_required']
                self._auto_voice_language = language
                self.state['connection_error'] = None
                self.state['status'] = '실물 로봇에 연결하고 초기 자세와 정지를 확인하고 있습니다.'
                self._log(self.state['status'])
                self._start_worker('connect', lambda: self.service.connect_controller(
                    physical_confirmed=True, operation=auxiliary), auxiliary=auxiliary)
                return
            self.service.connect_controller()
            self.state['status'] = self._mode_name() + ' 제어기에 연결했습니다. 시작 배치를 등록해 주세요.'
            self._log(self.state['status'])

    def disconnect_controller(self, data):
        with self.lock:
            if (self.closing or self.state['busy'] or (self.chat is not None and not self.state['chat_setup_only'])
                    or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            self._require_current_mode(data)
            if self.state['mode'] == 'virtual':
                raise ValueError('가상 모드는 제어기 연결이 필요하지 않습니다.')
            self._auto_voice_active = False
            self._auto_voice_after_registration = False
            if self.chat is not None:
                self.chat.stop()
            self.service.disconnect_controller()
            self._pending_connect_screen = None
            self.state.update(registered=False, recovery_required=False, recovery_guarded=False,
                              observation=None, path=[], forecast=None, samples=[], motion=None,
                              model=None, scene_confirmation=None,
                              scene_label='연결 해제 · 배치를 다시 등록해 주세요',
                              status=self._mode_name() + ' 제어기 연결을 해제했습니다. 다시 연결한 뒤 배치를 등록해 주세요.')
            self._registered_service_id = None
            self._registered_coordinator_revision = None
            self.state['registration_revision'] += 1
            self._log(self.state['status'])
            return self.state['registration_revision']

    def prepare_gripper(self, data):
        with self.lock:
            if (self.closing or self.state['busy'] or (self.chat is not None and not self.state['chat_setup_only'])
                    or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            self._require_current_mode(data)
            if self.state['mode'] == 'virtual':
                raise ValueError('가상 모드는 별도의 집게 준비가 필요하지 않습니다.')
            if self.state['recovery_required']:
                raise ValueError('제어기를 다시 연결한 뒤 집게를 준비해 주세요.')
            if self.state['registered']:
                raise ValueError('집게를 다시 준비하려면 제어기를 다시 연결하고 배치를 재등록해 주세요.')
            auxiliary = _AuxiliaryOperation()
            mode = self.state['mode']
            self.state['status'] = self._mode_name() + ' 집게 열기 준비를 요청했습니다.'
            self._log(self.state['status'])
            options = {'operation': auxiliary}
            if mode == 'physical':
                options['physical_confirmed'] = True
            self._start_worker('prepare_gripper', lambda: self.service.prepare_gripper(**options), auxiliary=auxiliary)

    def recover_controller(self, data):
        with self.lock:
            if (self.closing or self.state['busy'] or (self.chat is not None and not self.state['chat_setup_only'])
                    or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            self._require_current_mode(data)
            if self.state['mode'] != 'physical':
                raise ValueError('실물 제어기 모드에서 복구를 요청해 주세요.')
            if data.get('layout_confirmed') is not True or data.get('confirm_stop') is not True:
                raise ValueError('현재 배치와 정지 확인 요청을 직접 확인해 주세요.')
            auxiliary = _AuxiliaryOperation()
            self.state['registered'] = False
            self.state['recovery_required'] = True
            self.state['recovery_guarded'] = False
            self.state['registration_revision'] += 1
            self._registered_service_id = None
            self._registered_coordinator_revision = None
            self.state['scene_label'] = '제어기 복구 중 · 배치 재등록 필요'
            self.state['status'] = '실물 제어기의 정지 확인과 오류 복구를 요청했습니다.'
            self._log(self.state['status'])
            self._start_worker('recover_controller', lambda: self.service.recover_controller(
                layout_confirmed=True, confirm_stop=True, operation=auxiliary), auxiliary=auxiliary)
            return self.state['registration_revision']

    def _require_current_registration(self, data):
        revision = data.get('registration_revision')
        if type(revision) is not int or revision != self.state['registration_revision']:
            raise ValueError('다른 화면에서 배치가 변경됐습니다. 이 화면에서 배치를 다시 등록해 주세요.')

    def _start_text_control(self, target, name):
        def run():
            try:
                target()
            finally:
                with self.lock:
                    self._text_control_workers.discard(threading.current_thread())
                    self.state['text_control_pending'] = bool(self._text_control_workers)
        worker = threading.Thread(target=run, daemon=True, name=name)
        self._text_control_workers.add(worker)
        self.state['text_control_pending'] = True
        self.text_control_worker = worker
        try:
            worker.start()
        except BaseException:
            self._text_control_workers.discard(worker)
            self.state['text_control_pending'] = bool(self._text_control_workers)
            raise

    def _cancel_unstarted_text_request(self, target, chat=None):
        """Make a submitted stop win before any background stop worker runs."""
        with self.service._text_request_lock:
            pending = self.service._text_request
            coordinator = self.service.coordinator
            if coordinator is None:
                return False
            if pending is None or pending[0] != target or pending[2] is not coordinator:
                if chat is None or getattr(chat, 'coordinator', None) is not coordinator:
                    return False
                with chat._state_lock, coordinator._lock:
                    if (chat._pending_id != target or chat._pending_cancel is None
                            or chat._pending_phase == 'finished'
                            or coordinator._voice_turn_started.get(target) is not False
                            or (coordinator._active is not None and coordinator._active['request_id'] == target)):
                        return False
                    chat._pending_cancel.set()
                    coordinator.cancel_pending_voice(target)
                    reservation = coordinator._arm_home_reservation
                    if reservation is not None and reservation['request_id'] == target:
                        reservation['cancelled'].set()
                    return True
            with coordinator._lock:
                # The service request can outlive completed execution briefly.
                # Only report a pre-start cancellation when the coordinator has
                # never handed this exact request to the execution path.
                if (coordinator._voice_turn_started.get(target) is not False
                        or (coordinator._active is not None and coordinator._active['request_id'] == target)):
                    return False
                pending[1].set()
                coordinator.cancel_pending_voice(target)
                reservation = coordinator._arm_home_reservation
                if reservation is not None and reservation['request_id'] == target:
                    reservation['cancelled'].set()
                return True

    def submit(self, data):
        text = data.get('text')
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError('명령을 입력해 주세요. (최대 2,000자)')
        with self.lock:
            home = getattr(self.service, 'arm_home_request_id', lambda: None)()
            if home is not None and _is_direct_stop_text(text):
                if data.get('stop_target') != home:
                    raise ValueError('원위치 요청이 바뀌었습니다. 화면 상태를 확인해 주세요.')
                self.stop(target_request_id=home)
                return
            if not self.state['registered'] or self.state['recovery_required']:
                raise ValueError('먼저 ' + self._mode_name() + ' 시작 배치를 등록해 주세요.')
            self._require_current_registration(data)
            direct_stop = _is_direct_stop_text(text)
            if (self.closing or self.state['stop_requested'] or self.state['chat_stopping']
                    or (self.state['text_control_pending'] and not direct_stop)):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            chat_target = self.chat.current_target() if self.chat is not None else None
            if self.state['busy'] or chat_target is not None:
                if (self.state['busy'] not in (None, 'submit')
                        or (self.state['text_control_pending'] and not direct_stop)):
                    raise ValueError('현재 요청 처리가 끝나면 다시 말씀해 주세요.')
                chat = self.chat if chat_target is not None else None
                if chat is not None and self.state['busy'] != 'submit':
                    target = chat.current_target()
                else:
                    chat = None
                    target = self._text_target
                if target is None or data.get('stop_target') != target:
                    raise ValueError('작업이 바뀌었거나 진행 중인 작업이 없습니다. 화면 상태를 확인해 주세요.')
                # During an active transfer Gemma interprets the entire
                # follow-up, including sentences that mention stopping in a
                # negative or contrastive clause. Only a direct stop takes
                # the immediate text shortcut.
                coordinator = self.service.coordinator
                active = (coordinator.snapshot()['reference_context'].get('active_request_id')
                          if coordinator is not None else None)
                if not _is_direct_stop_text(text) and (active == target or not _is_text_stop_candidate(text)):
                    if coordinator is None:
                        raise ValueError('진행 중인 작업을 확인하지 못했습니다.')
                    pending_text = None
                    if active != target:
                        pending_text = (self.service.begin_pending_voice_text_followup(chat, target, text.strip())
                                        if chat is not None else
                                        self.service.begin_pending_text_followup(target, text.strip()))
                    if pending_text is not None:
                        self._log('요청 변경: ' + text.strip())
                        def pending_followup():
                            try:
                                result = (self.service.interpret_pending_voice_text_followup(pending_text)
                                          if chat is not None else
                                          self.service.interpret_pending_text_followup(pending_text))
                                promotion = result.get('promoted_text')
                                if promotion is not None:
                                    try:
                                        with self.lock:
                                            self._text_control_workers.discard(threading.current_thread())
                                            self.state['text_control_pending'] = bool(self._text_control_workers)
                                            if promotion['cancelled'].is_set() or self.closing:
                                                raise InterruptedError('요청을 취소했습니다.')
                                            self._text_target = promotion['request']
                                            self._start_worker('submit', lambda: self.service.execute_promoted_text(promotion))
                                            with promotion['chat']._state_lock:
                                                if promotion['chat']._pending_id == promotion['request']:
                                                    promotion['chat']._task_thread = self.worker
                                    except BaseException:
                                        self.service.finish_promoted_text(promotion, cancel=True)
                                        raise
                                    return
                                with self.lock:
                                    self._show_result(result)
                            except Exception as exc:
                                with self.lock:
                                    self.state['status'] = '변경 요청을 확인하지 못해 이전 요청을 실행하지 않았습니다: ' + str(exc)
                                    self._log(self.state['status'])
                        self._start_text_control(pending_followup, 'demo-web-pending-text-followup')
                        return
                    capture = coordinator.begin_interpretation(text.strip(), request_id=_request_id())
                    if capture['reference_context'].get('active_request_id') != target:
                        coordinator.discard_interpretation(capture['interpretation_id'])
                        raise ValueError('작업이 바뀌었거나 진행 중인 작업이 없습니다. 화면 상태를 확인해 주세요.')
                    self._log('작업 중 요청: ' + text.strip())
                    def followup():
                        try:
                            result = self.service.interpret_voice_followup(
                                capture, text.strip(), target, coordinator=coordinator)
                            with self.lock:
                                self._show_result(result)
                        except Exception as exc:
                            with self.lock:
                                self.state['status'] = '요청을 처리하지 못했습니다: ' + str(exc)
                                self._log(self.state['status'])
                        finally:
                            coordinator.discard_interpretation(capture['interpretation_id'])
                    self._start_text_control(followup, 'demo-web-text-followup')
                    return
                cancelled_before_dispatch = False
                if _is_direct_stop_text(text):
                    # Close the unstarted-request boundary synchronously:
                    # a delayed stop worker must not allow a simultaneous
                    # correction to release the old plan first.
                    cancelled_before_dispatch = self._cancel_unstarted_text_request(target, chat)
                    self.state['motion_stop_target'] = target
                    self._stopped_motion_requests.add(target)
                self._log('정지 요청: ' + text.strip())
                def control():
                    try:
                        language = 'ko' if any('가' <= c <= '힣' for c in text) else 'en'
                        if cancelled_before_dispatch and chat is not None:
                            with self.lock:
                                self.state['status'] = '해당 요청의 실행을 취소했습니다.'
                                self._log(self.state['status'])
                            return
                        handled = (chat._stop_target(target, language=language) if chat is not None and _is_direct_stop_text(text)
                                   else chat.stop_pending_prefix(target, text, language=language) if chat is not None
                                   else None)
                        if handled is None:
                            result = self.service.submit(text.strip(), stop_target=target)
                            if (cancelled_before_dispatch and
                                    (result.get('reply') or {}).get('kind') == 'no_matching_active_request'):
                                # The original worker has already cleaned itself
                                # up after the synchronous cancellation above.
                                result = {**result, 'reply': {**result['reply'],
                                    'kind': 'cancelled_pending_request',
                                    'text': '해당 요청의 실행을 취소했습니다. 다른 작업은 중단하지 않았습니다.'}}
                            with self.lock:
                                self._show_result(result)
                    except Exception as exc:
                        with self.lock:
                            self.state['status'] = '정지 요청을 처리하지 못했습니다: ' + str(exc)
                            self._log(self.state['status'])
                self._start_text_control(control, 'demo-web-text-stop')
                return
            if data.get('stop_target') is not None:
                raise ValueError('작업이 바뀌었거나 중단할 작업이 없습니다. 화면 상태를 확인해 주세요.')
            self.state['status'] = '요청을 확인하고 있습니다.'
            self._log('요청: ' + text.strip())
            with self.service._text_request_lock:
                coordinator = self.service.coordinator
                if self.chat is not None and coordinator is not None:
                    with coordinator._lock:
                        if coordinator._active is not None or coordinator._interpretations or coordinator._pending:
                            raise ValueError('음성 요청을 처리하고 있습니다. 처리가 끝난 뒤 입력해 주세요.')
                request_id = None if _is_direct_stop_text(text) else self.service.reserve_text_request()
            self._text_target = request_id
            self._start_worker('submit', lambda: self.service.submit(text.strip(), request_id=request_id))

    def start_chat(self, data):
        language = data.get('language')
        if language not in ('auto', 'ko', 'en'):
            raise ValueError('음성 인식 언어를 다시 선택해 주세요.')
        with self.lock, self.service._text_request_lock:
            if not self.state['registered']:
                if (not self.development and self.state['mode'] == 'physical'
                        and self.service._controller is not None):
                    if (self.state['busy'] or self.chat is not None or self.closing
                            or self.state['stop_requested'] or self.state['chat_stopping']):
                        raise ValueError('진행 중인 작업을 먼저 종료해 주세요.')
                    self._auto_voice_active = True
                    self._start_setup_voice(language)
                    return
                raise ValueError('먼저 시작 배치를 등록해 주세요.')
            if self._registered_service_id != self.service._registration_id:
                self._invalidate_registration()
                raise ValueError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
            if self.service._registration_requests:
                raise ValueError('다른 배치 등록이 진행 중입니다. 등록 상태를 확인해 주세요.')
            expected_registration = self._registered_service_id
            coordinator = self.service.coordinator
            planning = None
            if coordinator is not None:
                with coordinator._lock:
                    current_snapshot = coordinator.snapshot()
                    if current_snapshot.get('registration_revision') != self._registered_coordinator_revision:
                        self._invalidate_registration()
                        raise ValueError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
                    self._snapshot(current_snapshot)
                    planning = coordinator._planning_snapshot()
            recovery_read_only = self.state['recovery_required']
            if recovery_read_only:
                if not self.state['recovery_guarded'] or coordinator is None:
                    raise ValueError('현재 보호 상태를 확인할 수 없어 배치를 다시 등록해 주세요.')
                if not (planning['session_status'] == 'uncertain' or planning['stop_unconfirmed']
                        or planning['cancelled']):
                    raise ValueError('보호 상태가 바뀌어 배치를 다시 등록해 주세요.')
            if expected_registration != self.service._registration_id or coordinator is not self.service.coordinator:
                self._invalidate_registration()
                raise ValueError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
            self._require_current_registration(data)
            if (self.state['busy'] or self.chat is not None or self.closing or self.state['text_control_pending']
                    or self.state['stop_requested'] or self.state['chat_stopping']):
                raise ValueError('진행 중인 작업이나 음성 대화를 먼저 종료해 주세요.')
            from .demo_voice_chat import LocalConversation
            self.chat_error = None
            self.state['transcript'] = ''
            self.state['samples'] = []
            self.state['path'] = []
            self.state['forecast'] = None
            self.chat = (LocalConversation(self.service, self.emit, language=language, recovery_read_only=True)
                         if recovery_read_only else LocalConversation(self.service, self.emit, language=language))
            if (expected_registration != self.service._registration_id
                    or coordinator is not self.service.coordinator
                    or getattr(self.chat, 'coordinator', coordinator) is not coordinator):
                self.chat = None
                self._invalidate_registration()
                raise ValueError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
            self.state['chat'] = True
            self.state['chat_setup_only'] = False
            self.state['chat_listening'] = False
            self.state['chat_stopping'] = False
            self.state['stop_requested'] = False
            self._chat_stop_returned = True
            self.state['status'] = (('음성 대화를 시작합니다. 이동이나 새 기억 저장 전에는 배치를 다시 등록해 주세요. '
                                     + self._recovery_memory_message()) if recovery_read_only
                                    else '음성 대화를 시작하고 있습니다.')
            self._log('음성 대화를 시작합니다. 음성 응답 중에는 듣지 않습니다.')
            try:
                with coordinator._lock if coordinator is not None else nullcontext():
                    if coordinator is not None:
                        current_snapshot = coordinator.snapshot()
                        active = coordinator._active or {}
                        if (current_snapshot.get('registration_revision') != self._registered_coordinator_revision
                                or active.get('kind') == 'recovery_registration'):
                            self._invalidate_registration()
                            raise ValueError('배치가 다른 경로에서 변경됐습니다. 화면에서 다시 등록해 주세요.')
                        self._snapshot(current_snapshot)
                        if self.state['recovery_required'] and not self.state['recovery_guarded']:
                            raise ValueError('현재 보호 상태를 확인할 수 없어 배치를 다시 등록해 주세요.')
                        if self.state['recovery_required']:
                            self.state['status'] = ('음성 대화를 시작합니다. 이동이나 새 기억 저장 전에는 배치를 다시 등록해 주세요. '
                                                    + self._recovery_memory_message())
                    self.chat.start()
            except Exception:
                self.chat = None
                self.state['chat'] = False
                raise

    def end_chat(self):
        with self.lock:
            chat = self.chat
            if self.closing:
                return
            self._auto_voice_active = False
            self._auto_voice_after_registration = False
            # A registration may already have closed the old listener. An
            # explicit stop from another page must still cancel its restart.
            if chat is None or self.state['chat_stopping']:
                return
            if self.state['busy'] == 'registration_voice_pause' and chat.stop_event.is_set():
                return
            if isinstance(chat, _DeferredSetupConversation):
                chat.stop_event.set()
            self.state['chat_stopping'] = True
            self._chat_stop_returned = False
            self.state['status'] = '음성 대화를 종료하고 있습니다.'
            def request():
                try:
                    chat.stop()
                except Exception as exc:
                    with self.lock:
                        self.state['chat_stopping'] = False
                        self.state['status'] = '음성 대화 종료 요청 오류: ' + str(exc)
                        self._log(self.state['status'])
                finally:
                    with self.lock:
                        self._chat_stop_returned = True
                        if self.chat is None:
                            self.state['chat_stopping'] = False
            self.chat_stop_worker = threading.Thread(target=request, daemon=True)
            self.chat_stop_worker.start()

    def stop(self, *, target_request_id=None):
        with self.lock:
            chat, action, auxiliary = self.chat, self.state['busy'], self._aux_operation
            preparation = self._registration_preparation if action == 'register' else None
            setup_chat = self.state['chat_setup_only']
            current_home = getattr(self.service, 'arm_home_request_id', lambda: None)()
            target = (target_request_id if target_request_id is not None else current_home or
                      (self._text_target if action == 'submit' else
                       chat.current_target() if chat is not None else self._text_target))
            home_target = target if target is not None and current_home == target else None
            if (chat is None and action is None and home_target is None) or self.state['stop_requested'] or self.closing:
                return
            cancelled_before_dispatch = self._cancel_unstarted_text_request(target, chat)
            self.state['stop_requested'] = True
            if target is not None:
                self.state['motion_stop_target'] = target
                self._stopped_motion_requests.add(target)
            self.state['path'] = []
            self.state['forecast'] = None
            self.state['status'] = ('연결 요청을 중단할 수 있는지 확인하고 있습니다.' if action == 'connect'
                                    else '중단을 요청했습니다. ' + self._mode_name() + ' 명령의 중단 결과를 확인 중입니다.')
            self._log(self.state['status'])
            def request():
                try:
                    if action == 'registration_voice_pause' and auxiliary is not None:
                        auxiliary.cancel()
                        with self.lock:
                            self._chat_changed.notify_all()
                    elif home_target is not None:
                        result = self.service.stop(target_request_id=home_target)
                        if cancelled_before_dispatch and result.get('status') == 'no_matching_active_request':
                            result = {**result, 'status': 'cancelled_pending_request'}
                        with self.lock:
                            if result.get('status') not in ('stop_requested', 'cancelled_pending_request'):
                                if self.state['motion_stop_target'] == target:
                                    self.state['motion_stop_target'] = None
                                self._stopped_motion_requests.discard(target)
                            self.state['status'] = ('원위치 중단을 요청했습니다. 정지 여부를 확인해 주세요.'
                                                    if result.get('status') == 'stop_requested' else
                                                    '대기 중인 원위치 요청을 취소했습니다.'
                                                    if result.get('status') == 'cancelled_pending_request' else
                                                    '해당 원위치 요청이 이미 끝났거나 바뀌었습니다.')
                            self._log(self.state['status'])
                    elif chat is not None and not setup_chat and action != 'submit':
                        if cancelled_before_dispatch:
                            with self.lock:
                                self.state['status'] = '해당 요청의 실행을 취소했습니다.'
                                self._log(self.state['status'])
                            return
                        accepted = chat._stop_captured_target(target, {}) if target_request_id is not None else chat.stop_motion()
                        if not accepted:
                            with self.lock:
                                if self.state['motion_stop_target'] == target:
                                    self.state['motion_stop_target'] = None
                                self._stopped_motion_requests.discard(target)
                    elif action == 'submit':
                        result = self.service.submit('멈춰', stop_target=target)
                        if (cancelled_before_dispatch and
                                (result.get('reply') or {}).get('kind') == 'no_matching_active_request'):
                            result = {**result, 'reply': {**result['reply'],
                                'kind': 'cancelled_pending_request',
                                'text': '해당 요청의 실행을 취소했습니다. 다른 작업은 중단하지 않았습니다.'}}
                        with self.lock:
                            if ((result.get('reply') or {}).get('kind') not in
                                    ('bound_stop_proposal', 'cancelled_pending_request')
                                    and self.state['motion_stop_target'] == target):
                                self.state['motion_stop_target'] = None
                                self._stopped_motion_requests.discard(target)
                            self.state['status'] = result['reply']['text']
                            self._log(self.state['status'])
                    elif action in ('connect', 'prepare_gripper', 'recover_controller') and auxiliary is not None:
                        cancelled = auxiliary.cancel()
                        result = (self.service.stop_controller_preparation(target_request_id=auxiliary.request_id)
                                  if cancelled else {'status': 'already_completed'})
                        with self.lock:
                            if not cancelled or result.get('status') == 'already_completed':
                                message = '해당 요청은 이미 처리됐습니다. 현재 상태를 확인해 주세요.'
                            elif action == 'connect' and (result.get('arm_stop') or {}).get('requested') is not True:
                                message = '연결 요청 취소를 접수했습니다. 제어기 상태를 확인해 주세요.'
                            else:
                                message = ('연결 및 초기 자세 중단을 요청했습니다. 팔 정지와 장치 상태를 확인해 주세요.'
                                           if action == 'connect' else
                                           '집게 준비 중단을 요청했습니다. 팔 정지와 집게 상태를 확인해 주세요.'
                                           if action == 'prepare_gripper' else
                                           '제어기 복구 중단을 요청했습니다. 정지와 오류 상태를 확인해 주세요.')
                            # A completed worker may already have a newer result.
                            if self.state['busy'] == action and self._aux_operation is auxiliary:
                                self.state['status'] = message
                            self._log(message)
                    elif auxiliary is not None:
                        cancelled = auxiliary.cancel()
                        if preparation is not None:
                            preparation.cancel()
                            self.service.stop_controller_preparation(target_request_id=preparation.request_id)
                        with self.lock:
                            self.state['status'] = (self._mode_name() + ' 배치 등록을 취소하고 있습니다.' if cancelled
                                                    else '해당 작업은 이미 완료됐습니다.')
                            self._log(self.state['status'])
                except Exception as exc:
                    with self.lock:
                        if self.state['motion_stop_target'] == target:
                            self.state['motion_stop_target'] = None
                        self._stopped_motion_requests.discard(target)
                        self._log('중단 요청 오류: ' + str(exc))
                finally:
                    with self.lock:
                        self.state['stop_requested'] = False
                        self._resume_automatic_voice()
            self.stop_worker = threading.Thread(target=request, daemon=True)
            self.stop_worker.start()

    def begin_close(self):
        """Stop admitting new work before acknowledging a close request."""
        with self.lock, self.service._text_request_lock:
            if self.closing:
                return False
            self.closing = True
            self.state['closing'] = True
            self.service.begin_shutdown()
            return True

    def close(self):
        begin_error = None
        try:
            self.begin_close()
        except Exception as exc:
            # A failure to announce shutdown must not skip the remaining
            # cleanup.  Keep the failure visible to this caller.
            begin_error = exc
            with self.lock:
                self.closing = True
                self.state['closing'] = True
                self._log('종료 준비 오류: ' + str(exc))
        with self.lock:
            if self._close_started:
                owner = False
            else:
                self._close_started = True
                owner = True
            chat, worker = self.chat, self.worker
            action, target, auxiliary = self.state['busy'], self._text_target, self._aux_operation
            preparation = self._registration_preparation if action == 'register' else None
            own = {target}
            if action in ('register', 'connect', 'prepare_gripper', 'recover_controller') and auxiliary is not None:
                own.add(auxiliary.request_id)
            if chat is not None:
                own.add(chat.current_target())
                own.add(getattr(chat, '_session_id', None))
            stop_worker, chat_stop_worker = self.stop_worker, self.chat_stop_worker
            text_control_workers = tuple(self._text_control_workers)
        if not owner:
            self._close_complete.wait()
            if begin_error is not None:
                raise begin_error
            if self._close_error is not None:
                raise self._close_error
            return
        close_error = begin_error
        try:
            if chat is not None:
                try:
                    chat.stop()
                except Exception as exc:
                    chat.stop_event.set()
                    with self.lock:
                        self._log('음성 대화 종료 오류: ' + str(exc))
            if action == 'submit' and target is not None:
                try:
                    self.service.stop(target_request_id=target)
                except Exception as exc:
                    # A failed stop request does not establish that the arm
                    # stopped.  Continue closing our model and web resources;
                    # do not retry the hardware command or claim success.
                    with self.lock:
                        self.state['status'] = '종료 중 작업 중단 여부를 확인하지 못했습니다.'
                        self._log('종료 중 작업 중단 요청 오류: ' + str(exc))
            elif action == 'register' and auxiliary is not None:
                auxiliary.cancel()
                if preparation is not None:
                    preparation.cancel()
                    try:
                        self.service.stop_controller_preparation(target_request_id=preparation.request_id)
                    except Exception as exc:
                        with self.lock:
                            self.state['status'] = '종료 중 준비 작업의 중단 여부를 확인하지 못했습니다.'
                            self._log('종료 중 준비 작업 중단 요청 오류: ' + str(exc))
            elif action in ('connect', 'prepare_gripper', 'recover_controller') and auxiliary is not None:
                try:
                    auxiliary.cancel()
                    self.service.stop_controller_preparation(target_request_id=auxiliary.request_id)
                except Exception as exc:
                    with self.lock:
                        self._log('집게 준비 중단 오류: ' + str(exc))
            while self.service.has_other_requests(own):
                time.sleep(.05)
            self.service.cancelled.set()
            try:
                self.service.interpreter.request_close()
            except Exception as exc:
                with self.lock:
                    self._log('모델 종료 요청 오류: ' + str(exc))
                if close_error is None:
                    close_error = exc
            if chat is not None:
                chat.closed.wait()
            if worker is not None and worker is not threading.current_thread():
                worker.join()
            for pending_worker in (stop_worker, chat_stop_worker, *text_control_workers):
                if pending_worker is not None and pending_worker is not threading.current_thread():
                    pending_worker.join()
            try:
                self.service.close()
            except Exception as exc:
                with self.lock:
                    self._log('서비스 종료 오류: ' + str(exc))
                    for note in getattr(exc, '__notes__', ()):
                        self._log(note)
                if close_error is None:
                    close_error = exc
        except Exception as exc:
            with self.lock:
                self._log('종료 처리 오류: ' + str(exc))
            if close_error is None:
                close_error = exc
        finally:
            self._close_error = close_error
            self._close_complete.set()
        if close_error is not None:
            raise close_error


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, controller, port=0, *, product_owner=None, open_initial_screen=False):
        self.controller = controller
        self.token = secrets.token_urlsafe(24)
        self._lifecycle_lock = threading.RLock()
        self._shutdown_requested = False
        self._closing = False
        self._serving = False
        self._product_owner = product_owner
        self._open_initial_screen = open_initial_screen
        super().__init__(('127.0.0.1', port), DemoHandler)

    def server_bind(self):
        # On Windows, another process must not be able to bind over this
        # authenticated loopback endpoint while a screen is being opened.
        exclusive = getattr(socket, 'SO_EXCLUSIVEADDRUSE', None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()

    def open_product_screen(self):
        return product_instance.open_owned_screen(
            self, lifecycle_lock=self._lifecycle_lock,
            is_serving=lambda: self._serving,
            is_closing=lambda: self._closing or self._product_owner.closing,
            browser_open=webbrowser.open)

    def begin_close(self):
        with self._lifecycle_lock:
            self._closing = True
            self._serving = False
            if self._product_owner is not None:
                self._product_owner.begin_close()

    def serve_forever(self, poll_interval=.5):
        try:
            with self._lifecycle_lock:
                if self._closing:
                    return
                self._serving = True
                if self._product_owner is not None:
                    self._product_owner.set_reopener(self.open_product_screen)
            if self._open_initial_screen:
                threading.Thread(target=self.open_product_screen, daemon=True,
                                 name='product-initial-screen').start()
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self.begin_close()

    def server_close(self):
        with self._lifecycle_lock:
            self.begin_close()
            super().server_close()

    def request_shutdown(self):
        with self._lifecycle_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self.begin_close()
        begin_error = None
        try:
            self.controller.begin_close()
        except Exception as exc:
            begin_error = exc
            # Keep the reason available even as the server is shutting down.
            # Test doubles need not implement the product controller's log.
            try:
                with self.controller.lock:
                    self.controller._log('종료 준비 오류: ' + str(exc))
            except Exception:
                pass
        with self._lifecycle_lock:
            def shutdown():
                try:
                    self.controller.close()
                finally:
                    # The HTTP server must not remain reachable if controller
                    # cleanup itself raises.  The exception is still reported
                    # by the thread; it is not treated as successful cleanup.
                    self.shutdown()
            threading.Thread(target=shutdown, daemon=True, name='demo-web-shutdown').start()
        if begin_error is not None:
            raise begin_error


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, *_args):
        pass

    def _send(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        origin = self.headers.get('Origin')
        expected_origin = f'http://127.0.0.1:{self.server.server_port}'
        return (self.headers.get('X-Demo-Token') == self.server.token
                and (origin is None or origin == expected_origin))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/state':
            if not self._authorized():
                self._send(403, {'error': '접근 권한이 없습니다.'})
                return
            self._send(200, self.server.controller.snapshot())
            return
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.css': ('app.css', 'text/css; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8')}
        if path not in files:
            self._send(404, {'error': '찾을 수 없습니다.'})
            return
        name, kind = files[path]
        self._send(200, (ASSETS / name).read_bytes(), kind)

    def do_POST(self):
        if not self._authorized():
            self._send(403, {'error': '접근 권한이 없습니다.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 8192:
                raise ValueError('요청이 너무 깁니다.')
            data = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(data, dict):
                raise ValueError('요청 형식을 확인해 주세요.')
            action = urlsplit(self.path).path
            controller = self.server.controller
            result = {'ok': True}
            if action == '/api/register':
                result['registration_revision'] = controller.register(data)
            elif action == '/api/mode':
                result['registration_revision'] = controller.set_mode(data)
            elif action == '/api/connect':
                controller.connect_controller(data)
            elif action == '/api/disconnect':
                result['registration_revision'] = controller.disconnect_controller(data)
            elif action == '/api/prepare-gripper':
                controller.prepare_gripper(data)
            elif action == '/api/recover-controller':
                result['registration_revision'] = controller.recover_controller(data)
            elif action == '/api/submit':
                controller.submit(data)
            elif action == '/api/chat/start':
                controller.start_chat(data)
            elif action == '/api/chat/end':
                controller.end_chat()
            elif action == '/api/stop':
                controller.stop(target_request_id=data.get('stop_target'))
            elif action == '/api/close':
                self.server.request_shutdown()
                self._send(200, {'ok': True})
                return
            else:
                self._send(404, {'error': '찾을 수 없습니다.'})
                return
            self._send(200, result)
        except RegistrationStateChanged as exc:
            self._send(400, {'error': str(exc), 'code': 'registration_state_changed'})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {'error': str(exc)})
        except Exception as exc:
            self._send(500, {'error': '요청을 처리하지 못했습니다: ' + str(exc)})


def main():
    parser = argparse.ArgumentParser(description='로컬 로봇 시연 웹앱')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--no-prepare', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--development', action='store_true', help='개발·테스트용 가상 및 모의 모드를 활성화합니다.')
    args = parser.parse_args()
    owner = None
    if not args.development:
        try:
            claim = product_instance.claim_or_reopen(request_reopen=not args.no_browser)
        except Exception as exc:
            print('기존 라쿤봇 실행을 확인하지 못해 새 앱을 시작하지 않았습니다: ' + str(exc), flush=True)
            return 1
        if claim.owner is None:
            messages = {
                'open_requested': '실행 중인 라쿤봇에 화면 열기를 요청했습니다.',
                'starting': '라쿤봇이 시작 중입니다. 준비가 끝난 뒤 다시 실행해 주세요.',
                'closing': '라쿤봇이 종료 중입니다. 종료가 끝난 뒤 다시 실행해 주세요.',
                'unavailable': '기존 라쿤봇 실행을 확인하지 못해 새 앱을 시작하지 않았습니다. 열려 있는 앱을 확인해 주세요.',
                'unconfirmed': '화면 열기 요청의 결과를 확인하지 못했습니다. 브라우저를 확인해 주세요.',
            }
            print(messages.get(claim.status, messages['unavailable']), flush=True)
            return 0 if claim.status == 'open_requested' else 1
        owner = claim.owner
    controller = None
    server = None
    try:
        if owner is not None:
            legacy = product_instance.legacy_app_status()
            if legacy != 'clear':
                message = ('이전 버전의 라쿤봇이 실행 중입니다. 기존 화면을 이용하거나 기존 앱을 종료한 뒤 다시 실행해 주세요.'
                           if legacy == 'found' else
                           '기존 라쿤봇 실행 여부를 확인하지 못해 새 앱을 시작하지 않았습니다. 실행 중인 앱을 확인해 주세요.')
                print(message, flush=True)
                return 1
        controller = DemoController(prepare_model=not args.no_prepare, development=args.development)
        server = DemoServer(controller, args.port, product_owner=owner,
                            open_initial_screen=owner is not None and not args.no_browser)
        url = f'http://127.0.0.1:{server.server_port}/?token={server.token}'
        print(('개발용 웹앱' if args.development else '실물 로봇 시연')
              + '이 실행 중입니다. 브라우저의 시연 종료 버튼으로 정리할 수 있습니다.', flush=True)
        print(url, flush=True)
        if owner is None and not args.no_browser:
            webbrowser.open(url)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.begin_close()
        elif owner is not None:
            owner.begin_close()
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                if server is not None:
                    server.server_close()
            finally:
                if owner is not None:
                    owner.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
