"""Explicitly started local voice conversation for either operating mode.

Audio input, ASR, command execution and local speech output are separable so
tests can replace all devices. The default SAPI output has no echo reference;
an explicitly selected duplex path can provide reference-fed echo cancellation.
"""
from __future__ import annotations

from collections import deque
import copy
import queue
import re
import threading
import time

import numpy as np

from .concurrency import CombinedEvent as _EitherEvent
from .demo_service import (_DIRECT_SPOKEN_STOP, _is_polite_immediate_halt,
                           _execution_failure_message, _request_id, _validated_pending_stop_prefix)
from .voice import AUDIO_BLOCK_SIZE as BLOCK, AUDIO_SAMPLE_RATE as RATE
_LOCAL_KOREAN_VOICE = (r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens'
                       r'\TTS_MS_KO-KR_HEAMI_11.0')
_STOP = _DIRECT_SPOKEN_STOP


def _pending_stop_output(disposition):
    if disposition == 'stop':
        return True, None
    if disposition in ('stop_voice_output_mute', 'stop_voice_output_unmute'):
        return True, disposition.removeprefix('stop_voice_output_')
    return False, None

# The general v2 stop-prefix check does not, by itself, establish the intent
# of every complete utterance.  This fast path is deliberately limited to
# fully matched, direct stop-and-return requests; other language stays on the
# ordinary interpretation path and is not promoted into an immediate stop.
_KO_COMPOUND_STOP_RETURN = re.compile(
    r'\s*(?:(?:지금|즉시|바로|당장|일단)\s*)?'
    r'(?:(?:가상\s*)?로봇(?:의)?\s*)?'
    r'(?:(?:동작|작업|움직임)(?:을|를)?\s*)?'
    r'(?:(?:지금|즉시|바로|당장)\s*)?'
    r'(?P<stop>멈추고|멈춰서|멈춰\s*주시고|정지하고|중단하고|(?:멈춘|정지한|중단한)\s*뒤(?:에)?)\s*'
    r'(?:(?:로봇\s*)?팔(?:을|를)?\s*)?'
    r'(?:원위치(?:로)?|대기\s*자세(?:로)?|'
    r'(?:(?:등록한|등록된|처음|초기|기억한|작업\s*시작\s*전)\s*)'
    r'(?:초기\s*)?(?:배치|상태)(?:로)?)\s*'
    r'(?:해(?:\s*(?:줘|주세요))?|시켜(?:\s*(?:줘|주세요))?|'
    r'보내(?:\s*(?:줘|주세요))?|돌아가(?:\s*(?:줘|주세요))?|'
    r'돌려(?:\s*(?:줘|주세요))?|복원해(?:\s*(?:줘|주세요))?)\s*[.!。！]*\s*')
_EN_COMPOUND_STOP_RETURN = re.compile(
    r'\s*(?:(?P<polite>can|could|would|will)\s+you\s+)?(?:please\s+)?'
    r'(?P<stop>(?:stop|halt)(?:\s+(?:the\s+)?(?:robot|arm|motion|movement|operations?))?'
    r'(?:\s+(?:now|immediately|right\s+now))?)\s*,?\s*'
    r'(?:and(?:\s+then)?|then)\s+'
    r'(?:return\s+(?:(?:the\s+)?(?:robot(?:ic)?\s+)?arm\s+)?(?:to\s+)?'
    r'(?:its\s+|the\s+)?home(?:\s+(?:pose|position))?|'
    r'send\s+(?:the\s+)?(?:robot(?:ic)?\s+)?arm\s+to\s+(?:its\s+|the\s+)?'
    r'home(?:\s+(?:pose|position))?)\s*[.!?。！？]*\s*', re.I | re.ASCII)


def _validated_voice_compound_stop(text, context):
    """Check a narrow full utterance as well as the existing v2 stop evidence."""
    korean = _KO_COMPOUND_STOP_RETURN.fullmatch(text)
    english = _EN_COMPOUND_STOP_RETURN.fullmatch(text)
    if korean is None and english is None:
        return False
    if _validated_pending_stop_prefix(text, context):
        return True
    if korean is not None:
        # The full request binds "first stop, then return". v2's independent
        # stop check currently rejects the dependent 뒤 clause, so validate
        # the exact stop word from that already-matched original sentence.
        word = re.match(r'(?:멈춘|정지한|중단한)', korean.group('stop'))
        return bool(word and _validated_pending_stop_prefix(word.group(), context))
    if english is None or english.group('polite') is None:
        return False
    # The v2 validator rejects questions, including the bounded polite request
    # above. The full grammar has established who is being asked and its entire
    # requested sequence; independently check the exact original stop clause.
    from .task_program_v2 import _validated_initial_stop
    stop = english.group('stop')
    candidate = {'schema': 'task_program.v2', 'steps': [
        {'op': 'stop_motion', 'scope': 'active_task', 'evidence': stop}]}
    return bool(_validated_initial_stop(candidate, stop, context, _request_id()))


def spoken_plan(program):
    """Summarize only the already validated program; no invented outcome."""
    phrases = []
    for step in program.get('steps', []):
        op = step.get('op')
        if op == 'task' and step.get('kind') == 'remember':
            phrases.append('현재의 기대 배치를 기억하겠습니다')
        elif op == 'task' and step.get('kind') in ('move', 'stack', 'unstack'):
            names = {'move': '옮기겠습니다', 'stack': '쌓겠습니다', 'unstack': '풀겠습니다'}
            source, destination = step.get('source_slot'), step.get('destination_slot')
            if source is not None and destination is not None:
                phrases.append(f'{source}번에서 {destination}번으로 원통을 {names[step["kind"]]}')
            else:
                phrases.append('원통 작업을 시작하겠습니다')
        elif op == 'return':
            kind = (step.get('target') or {}).get('kind')
            phrases.append({'learned_memory': '기억한 배치로 복원하겠습니다',
                            'registered_initial': '등록한 시작 배치로 복원하겠습니다',
                            'task_start': '작업 시작 배치로 복원하겠습니다',
                            'arm_home': '팔을 초기 자세로 이동하겠습니다'}.get(kind, '복원 요청을 처리하겠습니다'))
        else:
            phrases.append('요청을 처리하겠습니다')
    return '. 이어서 '.join(phrases) + '.' if phrases else '요청을 처리하겠습니다.'


def spoken_result(result):
    if result.get('reply') is not None:
        return result['reply']['text']
    if result.get('voice_feedback'):
        return result['voice_feedback']
    execution = result.get('result')
    if not execution:
        return '요청을 실행하지 않았습니다. 화면에서 내용을 확인해 주세요.'
    status = execution.get('status')
    if status == 'motion_not_observed':
        return _execution_failure_message(execution)
    if status == 'step_unavailable':
        if (execution.get('not_started_reason') == 'controller_not_ready'
                or any(isinstance(event, dict) and event.get('event') == 'transfer'
                       and event.get('not_started_reason') == 'controller_not_ready'
                       for event in execution.get('events') or [])):
            return _execution_failure_message(execution)
        return '현재 배치에서는 요청한 다음 작업을 할 수 없습니다.'
    if status == 'retargeted_expected':
        return '변경한 동작을 마쳤습니다.'
    if status == 'retarget_not_applied':
        return '요청한 목표 변경은 적용되지 않았습니다.'
    if status == 'retarget_incomplete':
        return '새 목표는 적용됐지만 변경한 동작을 마치지 못했습니다. 물체 상태를 확인해 주세요.'
    if status == 'retarget_unconfirmed':
        return '요청한 목표 변경의 적용 여부를 확인하지 못했습니다. 물체 상태를 확인해 주세요.'
    if status != 'completed':
        if (execution.get('retarget') or {}).get('applied') is True:
            return '새 목표를 적용한 뒤 작업이 중단되었거나 완료되지 않았습니다. 물체 상태를 확인해 주세요.'
        terminations = {(event.get('result') or {}).get('termination') for event in execution.get('events') or []
                        if event.get('event') == 'transfer'}
        if 'decision_limit' in terminations:
            return '판단 횟수 한도에 도달해 작업을 계속할 수 없습니다. 배치를 확인해 주세요.'
        if 'out_of_range_target' in terminations:
            return '관절 목표가 허용 범위를 벗어나 작업을 계속할 수 없습니다. 화면을 확인해 주세요.'
        if status == 'uncertain':
            return _execution_failure_message(execution)
        return '요청 실행이 중단되었거나 완료되지 않았습니다. 화면을 확인해 주세요.'
    events = execution.get('events') or []
    remembered = any(event.get('event') == 'remember' for event in events)
    moved = any(event.get('event') == 'transfer' for event in events)
    if remembered and not moved:
        return '현재의 기대 배치를 기억했습니다.'
    if remembered:
        return '기대 배치를 기억하고 동작을 마쳤습니다.'
    return '동작을 마쳤습니다.'


def _retarget_observation(snapshot, target, english):
    """Describe only the captured transfer's executor evidence."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    active = (snapshot.get('reference_context') or {}).get('active_request_id')
    if active == target:
        state = ((snapshot.get('scene_state') or {}).get('retarget') or {}).get('status')
        if state == 'applied':
            return ('applied', 'The new target has been applied by the executor.' if english
                    else '실행기에 새 목표가 적용됐습니다.')
        return ('pending', 'The change was requested. The new target has not yet been confirmed.' if english
                else '목표 변경을 요청했습니다. 실제 적용은 아직 확인되지 않았습니다.')
    last = snapshot.get('last_execution') or {}
    if last.get('request_id') == target:
        result = last.get('retarget') or {}
        status = result.get('status', 'unconfirmed')
        messages = {
            'completed_expected': ('The changed motion has finished.', '변경한 동작을 마쳤습니다.'),
            'incomplete': ('The new target was applied, but the motion did not finish.',
                           '새 목표는 적용됐지만 동작을 마치지 못했습니다.'),
            'not_applied': ('The requested target change was not applied.',
                            '요청한 목표 변경은 적용되지 않았습니다.'),
            'unconfirmed': ('I could not confirm whether the new target was applied.',
                            '새 목표의 적용 여부를 확인하지 못했습니다.'),
        }
        message = messages.get(status, messages['unconfirmed'])
        return ('finished', message[0 if english else 1])
    return ('finished', 'I could not confirm the outcome of the captured task.' if english
            else '말씀하신 작업의 변경 결과를 확인하지 못했습니다.')


class SpeechDetector:
    """Bounded RMS speech gate; settings are not validated microphone accuracy."""
    def __init__(self, *, threshold=.012, start_frames=2, end_frames=15, max_frames=120):
        self.threshold, self.start_frames = threshold, start_frames
        self.end_frames, self.max_frames = end_frames, max_frames
        self.reset()

    def reset(self):
        # Keep 0.5 s before the first threshold-crossing block, in addition
        # to the consecutive blocks needed to confirm speech has started.
        self.pre = deque(maxlen=self.start_frames + 5)
        self.parts = []
        self._extra_pre_frames = 0
        self.active = False
        self.discarding = False
        self._discard_quiet_frames = self.end_frames
        self.voiced = self.quiet = 0

    def discard_after_overflow(self, *, known_speech):
        """Wait for a short quiet boundary when lost samples may contain speech."""
        continuing_tail = self.discarding and self._discard_quiet_frames == self.end_frames
        self.reset()
        self.discarding = True
        # After an idle-looking gap, 0.3 s of quiet is enough to accept a new
        # utterance. Once speech is heard, use the normal utterance boundary so
        # a pause within that potentially damaged utterance cannot become a
        # second command.
        self._discard_quiet_frames = (self.end_frames if known_speech or continuing_tail
                                      else min(3, self.end_frames))

    def feed_muted(self, frame):
        """Keep an interrupted utterance discarded while half-duplex is muted."""
        if not self.discarding:
            self.reset()
            return
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if len(frame) == BLOCK and np.isfinite(frame).all():
            self.feed(frame)

    def feed(self, frame):
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if len(frame) != BLOCK or not np.isfinite(frame).all():
            raise ValueError('유효한 0.1초 모노 음성 입력이 필요합니다.')
        frame = frame.copy()
        speech = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= self.threshold
        if self.discarding:
            if speech:
                self._discard_quiet_frames = self.end_frames
            self.quiet = 0 if speech else self.quiet + 1
            if self.quiet >= self._discard_quiet_frames:
                self.reset()
            return []
        if not self.active:
            self.pre.append(frame)
            self.voiced = self.voiced + 1 if speech else 0
            if self.voiced >= self.start_frames:
                self.active = True
                self.parts = list(self.pre)
                # Preserve the existing utterance-length limit while retaining
                # the additional start context outside its original 3 blocks.
                self._extra_pre_frames = max(0, len(self.parts) - 3)
                self.quiet = 0
                return [('start', None)]
            return []
        self.parts.append(frame)
        self.quiet = 0 if speech else self.quiet + 1
        if len(self.parts) > self.max_frames + self._extra_pre_frames:
            # Never execute the tail of one overlong utterance as a new one.
            # Wait for a full silence boundary before accepting speech again.
            self.active, self.discarding = False, True
            self.parts = []
            self.pre.clear()
            self.quiet = self.voiced = 0
            return [('discard', None)]
        if self.quiet >= self.end_frames:
            audio = np.concatenate(self.parts)
            self.reset()
            return [('end', audio)]
        return []


class WindowsSpeaker:
    """Use the installed Windows Korean desktop voice, without cloud fallback."""
    def speak(self, text, stop_event, *, started=None):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        speaker = voice = None
        try:
            speaker = win32com.client.Dispatch('SAPI.SpVoice')
            for voice in speaker.GetVoices():
                if voice.Id.casefold() == _LOCAL_KOREAN_VOICE.casefold():
                    speaker.Voice = voice
                    break
            else:
                raise RuntimeError('오프라인 한국어 음성을 찾지 못했습니다. 화면의 응답을 확인해 주세요.')
            if stop_event.is_set():
                return
            speaker.Speak(text, 1 | 16)  # asynchronous and literal text, not SAPI XML
            if started is not None:
                started.set()
            while not stop_event.is_set() and not speaker.WaitUntilDone(100):
                pass
            if stop_event.is_set():
                speaker.Speak('', 3)  # async + purge queued speech
        finally:
            if started is not None:
                started.set()
            voice = speaker = None
            pythoncom.CoUninitialize()


def _microphone_stream(*, device=None):
    import sounddevice as sd
    return sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=BLOCK, device=device)


def capture_voice_pcm(*, seconds=7, cancel=None, device=None, stream_factory=None):
    """Capture one explicit request as 16 kHz mono samples, without an ASR model."""
    if not 1 <= seconds <= 12:
        raise ValueError('녹음 시간은 1~12초 범위여야 합니다.')
    cancel = cancel or threading.Event()
    if cancel.is_set():
        raise InterruptedError('녹음을 취소했습니다.')
    factory = stream_factory or (lambda: _microphone_stream(device=device))
    samples = []
    with factory() as stream:
        for _ in range(round(seconds * RATE / BLOCK)):
            if cancel.is_set():
                raise InterruptedError('녹음을 취소했습니다.')
            data, overflow = stream.read(BLOCK)
            if overflow:
                raise ValueError('마이크 입력이 밀렸습니다. 다시 녹음해 주세요.')
            data = np.asarray(data, dtype=np.float32)
            if data.shape != (BLOCK, 1) or not np.isfinite(data).all():
                raise ValueError('유효한 모노 음성 데이터가 필요합니다.')
            samples.append(data[:, 0].copy())
    if cancel.is_set():
        raise InterruptedError('녹음을 취소했습니다.')
    audio = np.concatenate(samples)
    if float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < .002:
        raise ValueError('마이크 입력이 너무 작습니다. 장치와 볼륨을 확인해 주세요.')
    return audio


class _SpeechTurns:
    """Invalidate old audio on speech onset without cancelling robot work."""
    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._current = None

    def current(self):
        with self._lock:
            return self._generation

    def interrupt(self):
        with self._lock:
            self._generation += 1
            if self._current is not None:
                self._current.set()
            return self._generation

    def begin(self, generation, muted_state=None):
        with self._lock:
            if generation != self._generation or (muted_state is not None and muted_state.is_set()):
                return None
            event = threading.Event()
            self._current = event
            return event

    def set_output_muted(self, muted_state, muted):
        """Change playback state without invalidating a later captured turn."""
        with self._lock:
            if muted:
                muted_state.set()
                if self._current is not None:
                    self._current.set()
            else:
                muted_state.clear()

    def finish(self, event):
        with self._lock:
            if self._current is event:
                self._current = None


def _shared_voice_output_state(service):
    """Share a mute state across listeners owned by the same demo service."""
    lock = getattr(service, '_voice_lock', None)
    if lock is None:
        return threading.Event()
    with lock:
        state = getattr(service, '_voice_output_muted', None)
        if not isinstance(state, threading.Event):
            state = threading.Event()
            service._voice_output_muted = state
        return state


class LocalConversation:
    def __init__(self, service, emit, *, stream_factory=None, recognizer=None, speaker=None,
                 response_guard_seconds=.5, language='auto', recovery_read_only=False):
        if service.coordinator is None:
            raise ValueError('먼저 배치를 등록해 주세요.')
        if language not in ('auto', 'ko', 'en'):
            raise ValueError('음성 인식 언어는 자동, 한국어, 영어 중에서 선택해 주세요.')
        self.service, self.coordinator, self.emit = service, service.coordinator, emit
        self.stream_factory = stream_factory or _microphone_stream
        # A supplied recognizer is an explicit legacy/test injection. Product
        # conversations send the captured samples to the shared Gemma service.
        self.recognizer = recognizer
        self.speaker = speaker or WindowsSpeaker()
        if stream_factory is None and speaker is None:
            from .demo_voice_duplex import AdaptiveVoiceIO, duplex_requested
            if duplex_requested():
                duplex = AdaptiveVoiceIO(_microphone_stream, self.speaker)
                self.stream_factory, self.speaker = lambda: duplex, duplex
        # A conversation keeps the language selected before its microphone opens.
        self.language = language
        self._recovery_read_only = threading.Event()
        if recovery_read_only:
            self._recovery_read_only.set()
        self.response_guard_seconds = response_guard_seconds
        self.stop_event = threading.Event()
        self.speaking = threading.Event()
        self._voice_output_muted = _shared_voice_output_state(service)
        self.closed = threading.Event()
        self._state_lock, self._speaker_lock = threading.Lock(), threading.Lock()
        # The registration transition shares this gate with the speech detector,
        # including the first voiced frame before a full utterance is detected.
        self._capture_gate = threading.Lock()
        self._capturing_utterance = False
        self._capture_stream = None
        self._registration_pause_requested = False
        self._registration_pause_declined = False
        self._registration_pause_remaining = 0
        self._registration_pause_generation = 0
        self._registration_pause_source = None
        self._speech_turns = _SpeechTurns()
        self._feedback_thread = None
        self._feedback_pending = None
        self._duplex_active = False
        self._duplex_fallback = False
        self.audio_fallback_reason = None
        self.audio_fallback_stage = None
        self._clear_pending_state_locked()
        self._queue = queue.Queue(maxsize=2)
        self._task_thread = None
        self._retarget_thread = None
        self._prepare_thread = None
        self._started = False
        self._session_id = _request_id()
        self._bound_registration_revision = None

    @property
    def recovery_read_only(self):
        return self._recovery_read_only.is_set()

    @property
    def voice_output_enabled(self):
        return not self._voice_output_muted.is_set()

    def _set_voice_output(self, enabled, *, generation=None):
        """Change only speech playback; never cancel or stop robot work."""
        self._speech_turns.set_output_muted(self._voice_output_muted, not enabled)
        self.emit('chat_voice_output', {'enabled': enabled})
        message = ('음성 출력을 다시 켰습니다. 계속 듣고 있습니다.' if enabled else
                   '음성 출력을 껐습니다. 계속 듣고 있습니다.')
        self.emit('chat_state', message)
        if enabled and generation is not None:
            # The acknowledgement belongs to the captured utterance. A later
            # onset invalidates it even if recognition of this turn was slow.
            self._say_feedback('음성 출력을 다시 켰습니다.', generation=generation)

    def require_recovery(self):
        """Keep this conversation read-only after recovery becomes necessary."""
        self._recovery_read_only.set()

    def allow_after_connection_recovery(self):
        """Resume the same listener after the service restored its expected scene."""
        self._recovery_read_only.clear()

    def pause_if_idle_for_registration(self):
        """Stop only an idle listener, atomically against detected speech and work."""
        service, coordinator = self.service, self.coordinator
        with service._text_request_lock, self._capture_gate, self._state_lock, coordinator._lock:
            busy_thread = any(thread is not None and thread.is_alive()
                              for thread in (self._task_thread, self._retarget_thread))
            if (not self._started or self.stop_event.is_set() or service._shutting_down or service.coordinator is not coordinator
                    or self._capturing_utterance or self._registration_pause_requested or self._pending_id is not None
                    or not self._queue.empty() or self.speaking.is_set() or self._feedback_thread is not None
                    or busy_thread
                    or service._text_request is not None or service._record_request is not None
                    or service._transcription_requests or service._registration_requests
                    or service._controller_preparation is not None or service._controller_connecting is not None
                    or service._controller_recovery is not None
                    or service._conversation_requests != {self._session_id}
                    or coordinator._active is not None or coordinator._interpretations or coordinator._pending
                    or coordinator._arm_home_reservation is not None or coordinator.session._lock.locked()):
                return False
            # Snapshot a finite buffer at the request boundary, plus the block
            # the input worker may already have removed from that buffer.
            stream = self._capture_stream
            try:
                snapshot = getattr(stream, 'input_buffer_snapshot', None)
                source, frames = snapshot() if callable(snapshot) else (stream, stream.read_available)
                if type(frames) is not int or frames < 0 or frames > 50 * BLOCK:
                    return False
            except Exception:
                return False
            self._registration_pause_remaining = 1 + (frames + BLOCK - 1) // BLOCK
            self._registration_pause_source = source
            self._registration_pause_generation += 1
            self._registration_pause_requested = True
            self._registration_pause_declined = False
            return True

    def registration_pause_declined(self):
        with self._capture_gate:
            return self._registration_pause_declined

    def cancel_registration_pause(self):
        with self._capture_gate:
            self._registration_pause_requested = False
            return self.stop_event.is_set()

    def _resolve_registration_pause_locked(self, speech, stream):
        if not self._registration_pause_requested:
            return False
        source = getattr(stream, 'input_source', None)
        current_source = source() if callable(source) else stream
        if speech or current_source is not self._registration_pause_source:
            self._registration_pause_requested = False
            self._registration_pause_declined = True
            return False
        self._registration_pause_remaining -= 1
        if self._registration_pause_remaining <= 0:
            self._registration_pause_requested = False
            self.stop_event.set()
            return True
        return False

    def _clear_capture_gate(self):
        with self._capture_gate:
            self._capturing_utterance = False

    def _feed_muted_capture(self, detector, data):
        with self._capture_gate:
            detector.feed_muted(data)
            self._capturing_utterance = detector.discarding

    def _remember_audio_fallback(self, stream):
        reason = getattr(stream, 'fallback_reason', None)
        if reason is not None:
            self.audio_fallback_reason = reason
            self.audio_fallback_stage = getattr(stream, 'fallback_stage', None)
        self._duplex_fallback = bool(self.audio_fallback_reason)

    def start(self):
        with self.service._text_request_lock, self.coordinator._lock:
            if self._started:
                raise RuntimeError('이미 시작한 음성 대화입니다.')
            if self.service._shutting_down:
                raise RuntimeError('서비스가 종료 중입니다.')
            if self.service.coordinator is not self.coordinator:
                raise RuntimeError('등록된 배치가 바뀌어 음성 대화를 시작하지 않았습니다.')
            self._bound_registration_revision = self.coordinator.snapshot().get('registration_revision')
            if (self.service._text_request is None and self.service._record_request is None
                    and not self.coordinator._interpretations and self.coordinator._active is None):
                self.service.cancelled.clear()
            self.service._conversation_requests.add(self._session_id)
            self._started = True
        self._recognition = threading.Thread(target=self._recognize, daemon=True, name='demo-voice-asr')
        self._capture_thread = threading.Thread(target=self._capture, daemon=True, name='demo-voice-input')
        if callable(getattr(self.recognizer, 'prepare', None)):
            self._prepare_thread = threading.Thread(target=self._prepare_recognizer, daemon=True,
                                                    name='demo-voice-prepare')
        try:
            if self._prepare_thread is not None:
                self._prepare_thread.start()
            self._recognition.start()
            self._capture_thread.start()  # first possible microphone access
            threading.Thread(target=self._supervise, daemon=True, name='demo-voice-shutdown').start()
        except BaseException:
            self.stop_event.set()
            try:
                if hasattr(self.recognizer, 'cancel_pending'):
                    self.recognizer.cancel_pending()
            except Exception:
                pass
            for worker in (self._capture_thread, self._recognition, self._prepare_thread):
                if worker is None:
                    continue
                if worker.ident is not None:
                    worker.join()
            if self._prepare_thread is not None and self._prepare_thread.ident is not None:
                try:
                    if hasattr(self.recognizer, 'close'):
                        self.recognizer.close()
                except Exception:
                    pass
            with self.service._text_request_lock:
                self.service._conversation_requests.discard(self._session_id)
            self.closed.set()
            raise

    def _prepare_recognizer(self):
        try:
            if not self.stop_event.is_set():
                self.recognizer.prepare()
        except Exception:
            # The ordinary transcription path reports any persistent error and
            # retains its existing GPU-to-CPU fallback with the original audio.
            pass

    def _listening_message(self):
        if self._voice_output_muted.is_set():
            return '음성 출력 꺼짐 · 계속 듣고 있습니다. 다시 켜려면 “다시 말해”'
        if self._duplex_active:
            return '말씀을 듣는 중 · 답변 중에도 말씀할 수 있습니다. 중단: “멈춰”'
        if self._duplex_fallback:
            return '끼어들기를 사용할 수 없어 답변이 끝난 뒤 듣습니다. 중단: “멈춰”'
        return '말씀을 듣는 중 · 마치면 처리합니다. 중단: “멈춰”'

    def _say(self, text, *, started=None, cancel_on_request=False, cancel_event=None, generation=None):
        token = self.service.cancelled if cancel_event is None else cancel_event
        request_stop = _EitherEvent(self.stop_event, token) if cancel_on_request else self.stop_event
        generation = self._speech_turns.current() if generation is None else generation
        if request_stop.is_set():
            if started is not None:
                started.set()
            return False
        with self._speaker_lock:
            if self._voice_output_muted.is_set():
                if started is not None:
                    started.set()
                if not request_stop.is_set() and generation == self._speech_turns.current():
                    self.emit('chat_response', text)
                return not request_stop.is_set()
            playback_stop = self._speech_turns.begin(generation, self._voice_output_muted)
            if request_stop.is_set() or playback_stop is None:
                if playback_stop is not None:
                    self._speech_turns.finish(playback_stop)
                if started is not None:
                    started.set()
                return not request_stop.is_set()
            stop = _EitherEvent(request_stop, playback_stop)
            self.speaking.set()
            try:
                self.emit('chat_response', text)
                self.emit('chat_state', '답변 중에도 말씀할 수 있습니다.' if self._duplex_active
                          else '음성 응답 중에는 듣기를 잠시 중지합니다.')
                self.speaker.speak(text, stop, started=started)
            except Exception as exc:
                self.emit('chat_error', '음성 응답을 재생하지 못했습니다: ' + str(exc))
            finally:
                if started is not None:
                    started.set()
                self.stop_event.wait(self.response_guard_seconds)
                self.speaking.clear()
                self._speech_turns.finish(playback_stop)
            if not self.stop_event.is_set():
                self.emit('chat_state', self._listening_message())
            return not request_stop.is_set()

    def _say_feedback(self, text, *, generation):
        """Play recognition feedback without delaying the next recorded turn."""
        def play():
            while True:
                with self._state_lock:
                    if self.stop_event.is_set() or self._feedback_pending is None:
                        self._feedback_pending = None
                        self._feedback_thread = None
                        return
                    message, turn = self._feedback_pending
                    self._feedback_pending = None
                try:
                    self._say(message, generation=turn)
                except Exception as exc:
                    try:
                        self.emit('chat_error', '음성 응답을 재생하지 못했습니다: ' + str(exc))
                    except Exception:
                        pass

        with self._state_lock:
            if self.stop_event.is_set():
                return
            # New speech invalidates the preceding generation. Retain only the
            # latest waiting feedback while one playback may still be stopping.
            self._feedback_pending = (text, generation)
            if self._feedback_thread is None:
                worker = threading.Thread(target=play, daemon=True, name='demo-voice-feedback')
                self._feedback_thread = worker
                try:
                    worker.start()
                except BaseException:
                    self._feedback_thread = None
                    self._feedback_pending = None
                    raise

    def _capture(self):
        detector = SpeechDetector()
        active = None
        mute_until = 0.
        overflow_streak = 0
        stream_owner = None
        try:
            stream_owner = self.stream_factory()
            with stream_owner as stream:
                with self._capture_gate:
                    self._capture_stream = stream
                self._remember_audio_fallback(stream)
                if self.stop_event.is_set():
                    return
                self._duplex_active = bool(getattr(stream, 'echo_cancelled', False))
                self.emit('chat_audio_mode', {'mode': 'aec' if self._duplex_active else 'half_duplex',
                                              'fallback': self._duplex_fallback})
                self.emit('chat_voice_output', {'enabled': self.voice_output_enabled})
                self.emit('chat_input_opened', None)
                self.emit('chat_state', self._listening_message())
                while not self.stop_event.is_set():
                    data, overflow = stream.read(BLOCK)
                    inspected_generation = None
                    with self._capture_gate:
                        if self.stop_event.is_set():
                            break
                        if self._registration_pause_requested:
                            inspected_generation = self._registration_pause_generation
                            frame = np.asarray(data, dtype=np.float32).reshape(-1)
                            valid = len(frame) == BLOCK and np.isfinite(frame).all()
                            speech = (overflow or not valid or
                                      float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= detector.threshold)
                            if self._resolve_registration_pause_locked(speech, stream):
                                break
                    if overflow:
                        # An input overflow loses part of the current audio,
                        # not necessarily the microphone itself. Discard only
                        # the affected utterance and keep listening. Wait for
                        # silence before accepting its tail as a new request.
                        overflow_streak += 1
                        frame = np.asarray(data, dtype=np.float32).reshape(-1)
                        valid = len(frame) == BLOCK and np.isfinite(frame).all()
                        current_speech = (valid
                            and float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= detector.threshold)
                        had_utterance = (active is not None or detector.active or bool(detector.voiced)
                                         or current_speech)
                        if active is not None:
                            self._abort_held_followup(active)
                            self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
                            active = None
                        detector.discard_after_overflow(known_speech=bool(had_utterance or not valid))
                        with self._capture_gate:
                            self._capturing_utterance = True
                        if had_utterance and overflow_streak == 1:
                            self.emit('chat_state', '음성이 잠깐 끊겨 해당 발화를 처리하지 못했습니다. 다시 말씀해 주세요.')
                        elif overflow_streak == 1:
                            self.emit('chat_state', '마이크 입력이 잠깐 끊겼습니다. 계속 듣고 있습니다.')
                        elif overflow_streak == 3:
                            self.emit('chat_state', '마이크 입력이 밀리고 있습니다. 입력이 돌아오면 계속 듣겠습니다.')
                        # A permanently overflowing mocked or broken stream may
                        # return immediately, so avoid a busy loop while keeping
                        # it open for recovery.
                        self.stop_event.wait(.01)
                        continue
                    overflow_streak = 0
                    duplex_active = bool(getattr(stream, 'echo_cancelled', False))
                    if duplex_active != self._duplex_active:
                        interrupted = bool(active is not None or detector.active or detector.voiced or detector.discarding)
                        if active is not None:
                            self._abort_held_followup(active)
                            self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
                            active = None
                        if interrupted:
                            detector.discard_after_overflow(known_speech=True)
                        else:
                            detector.reset()
                        with self._capture_gate:
                            self._capturing_utterance = detector.discarding
                        self._speech_turns.interrupt()
                        self._duplex_active = duplex_active
                        self._remember_audio_fallback(stream)
                        # Do not pass the old playback's residual sound to the
                        # fallback's raw microphone after a mid-stream failure.
                        mute_until = time.monotonic() + self.response_guard_seconds
                        self.emit('chat_audio_mode', {'mode': 'aec' if duplex_active else 'half_duplex',
                                                      'fallback': self._duplex_fallback})
                        self.emit('chat_state', self._listening_message())
                        continue
                    if not duplex_active and time.monotonic() < mute_until:
                        self._feed_muted_capture(detector, data)
                        continue
                    if self.speaking.is_set() and not self._duplex_active:
                        if active is not None:
                            self._abort_held_followup(active)
                            self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
                            active = None
                        self._feed_muted_capture(detector, data)
                        continue
                    with self._capture_gate:
                        if self.stop_event.is_set():
                            break
                        events = detector.feed(data)
                        self._capturing_utterance = bool(detector.active or detector.discarding or detector.voiced
                            or any(event in ('end', 'discard') for event, _ in events))
                        if (self._registration_pause_requested
                                and inspected_generation != self._registration_pause_generation
                                and self._resolve_registration_pause_locked(self._capturing_utterance, stream)):
                            break
                    for event, audio in events:
                        if event == 'start':
                            with self.service._text_request_lock, self._state_lock, self.coordinator._lock:
                                capture = self.coordinator.begin_voice_processing(request_id=_request_id())
                                candidates = {entry[0] for entry in
                                              (self.service._text_request, self.service._record_request)
                                              if entry is not None and entry[2] is self.coordinator}
                                # The worker retains its ID while speaking the
                                # final result. That finished task is no longer
                                # a target for a newly started utterance.
                                pending = self._pending_id if self._pending_phase != 'finished' else None
                                for request in (capture['reference_context']['active_request_id'], pending):
                                    if request is not None:
                                        candidates.add(request)
                                home = getattr(self.service, 'arm_home_request_id', lambda: None)()
                                if home is not None:
                                    # A queued or running home operation owns
                                    # subsequent stop requests, even while the
                                    # previous turn is finishing its reply.
                                    candidates = {home}
                                target = next(iter(candidates)) if len(candidates) == 1 else None
                                target_cancel = (self._pending_cancel if target is not None and target == pending
                                                 else next((entry[1] for entry in
                                                            (self.service._text_request, self.service._record_request)
                                                            if entry is not None and entry[0] == target), None))
                                hold = None
                                text_pending = None
                                text_entry = self.service._text_request
                                text_plans = getattr(self.service, '_pending_text_plans', {})
                                if (target is not None and target != pending and text_entry is not None
                                        and text_entry[0] == target and text_entry[2] is self.coordinator
                                        and not text_entry[1].is_set() and target in text_plans):
                                    text_pending = {'target': target, 'cancelled': text_entry[1],
                                                    'plan': text_plans[target]}
                                hold_pending = getattr(self.coordinator, 'hold_pending_voice', None)
                                if (target is not None and len(candidates) == 1 and callable(hold_pending)
                                        and ((self.recognizer is None and target == pending
                                              and callable(getattr(self.service, 'interpret_pending_voice_audio', None)))
                                             or text_pending is not None)):
                                    hold = hold_pending(capture['interpretation_id'], target)
                                if hold is None:
                                    text_pending = None
                                # Releasing playback can let the previous task
                                # finish immediately. Bind the onset first while
                                # its ownership is still protected by the lock.
                                generation = self._speech_turns.interrupt()
                            active = {'capture': capture, 'target': target, 'ambiguous_target': len(candidates) > 1,
                                      'target_cancel': target_cancel, 'hold': hold, 'text_pending': text_pending,
                                      'started': time.monotonic(), 'generation': generation}
                            self.emit('chat_utterance_start', None)
                            self.emit('chat_state', '말씀을 듣고 있습니다.')
                        elif event == 'discard':
                            if active is not None:
                                self._abort_held_followup(active)
                                self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
                                active = None
                            self._clear_capture_gate()
                            self.emit('chat_error', '발화가 너무 길어 처리하지 않았습니다. 짧게 다시 말씀해 주세요.')
                        elif event == 'end' and active is not None:
                            active['audio'] = audio
                            active['ended'] = time.monotonic()
                            with self._state_lock:
                                available = self._pending_id is None or self._pending_phase == 'finished'
                                active['primary'] = (available and active['target'] is None
                                                     and not active['ambiguous_target'])
                                if active['primary']:
                                    self._pending_id = active['capture']['request_id']
                                    self._pending_cancel = threading.Event()
                                    self._pending_phase = 'processing'
                                    self._pending_speech_generation = active['generation']
                                    active['cancel_event'] = self._pending_cancel
                                    register = getattr(self.coordinator, 'register_pending_voice', None)
                                    if callable(register):
                                        register(self._pending_id)
                            try:
                                self._queue.put_nowait(active)
                            except queue.Full:
                                self._abort_held_followup(active)
                                self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
                                with self._state_lock:
                                    if active['primary'] and self._pending_id == active['capture']['request_id']:
                                        self._pending_id = None
                                        self._pending_cancel = None
                                        self._pending_phase = None
                                        self._pending_speech_generation = None
                                self.emit('chat_error', '음성 처리가 밀려 이번 발화를 버렸습니다.')
                            active = None
                            self._clear_capture_gate()
        except Exception as exc:
            if stream_owner is not None:
                self._remember_audio_fallback(stream_owner)
            if not self.stop_event.is_set():
                self.emit('chat_error', '음성 입력을 계속할 수 없습니다. 마이크 연결과 운영체제 권한을 확인한 뒤 다시 듣기를 시작해 주세요: ' + str(exc))
            self.stop()
        finally:
            if active is not None:
                self._abort_held_followup(active)
                self.coordinator.discard_interpretation(active['capture']['interpretation_id'])
            with self._capture_gate:
                self._capture_stream = None
                self._capturing_utterance = False

    def _recognize(self):
        while not self.stop_event.is_set():
            try:
                item = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            capture = item['capture']
            ident = capture['interpretation_id']
            handed_off = False
            try:
                self.emit('chat_state', '음성 요청을 확인하고 있습니다. 로컬 처리에는 시간이 걸릴 수 있습니다.')
                if item.get('hold') is not None:
                    if item.get('text_pending') is not None:
                        handed_off = self._recognize_pending_text_audio(item)
                        continue
                    if not self._release_nonactionable_hold(item):
                        handed_off = self._recognize_pending_followup(item)
                        continue
                started = time.monotonic()
                if self.recognizer is None:
                    cancellation = (_EitherEvent(self.stop_event, item['cancel_event'])
                                    if item['primary'] else self.stop_event)
                    speech = self.service.interpret_voice_audio(
                        item['audio'], capture=capture, coordinator=self.coordinator,
                        language=self.language, cancel_event=cancellation)
                    if not isinstance(speech, dict) or speech.get('direct_audio') is not True:
                        raise ValueError('직접 음성 처리 결과를 확인하지 못했습니다.')
                    immediate_stop = speech.get('immediate_stop') is True
                    stop_prefix = speech.get('stop_prefix') is True
                else:
                    speech = self.recognizer.transcribe(item['audio'], language=self.language)
                    immediate_stop = stop_prefix = False
                asr_seconds = time.monotonic() - started
                if self.stop_event.is_set() or (item['primary'] and item['cancel_event'].is_set()):
                    continue
                text = speech.get('text', '').strip() if isinstance(speech, dict) else ''
                if not text:
                    raise ValueError('인식된 문장이 없습니다.')
                if self.recognizer is not None:
                    immediate_stop = bool(_STOP.fullmatch(text) or _is_polite_immediate_halt(text))
                    stop_prefix = _validated_voice_compound_stop(text, capture['reference_context'])
                self.emit('chat_transcript', {'text': text, 'asr_seconds': asr_seconds,
                                             'capture_seconds': len(item['audio']) / RATE,
                                             'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                             'direct_audio': speech.get('direct_audio', False),
                                             'engine': speech.get('engine'),
                                             'device': speech.get('device'),
                                             'cuda_runtime': speech.get('cuda_runtime'),
                                             'fallback_reason': speech.get('fallback_reason')})
                from .task_program_v2 import direct_voice_output_control
                voice_output = speech.get('voice_output') or direct_voice_output_control(text)
                if voice_output in ('mute', 'unmute') and not immediate_stop and not stop_prefix:
                    self._finish_nonactionable_primary(item)
                    self._set_voice_output(voice_output == 'unmute', generation=item['generation'])
                elif immediate_stop:
                    # Bind a clear stop to the app request captured at speech
                    # onset. The same request may have moved from interpreting
                    # to executing while ASR ran; never retarget a later task.
                    finished = self._finish_nonactionable_primary(item)
                    self._stop_captured_target(item['target'], capture['reference_context'],
                                               ambiguous=item['ambiguous_target'],
                                               language='ko' if re.search(r'[가-힣]', text) else 'en',
                                               generation=item['generation'],
                                               background_feedback=not item['primary'] or finished)
                    if voice_output in ('mute', 'unmute'):
                        self._set_voice_output(voice_output == 'unmute', generation=item['generation'])
                elif (speech.get('direct_audio') and stop_prefix and item['target'] is not None
                      and item['target'] == getattr(self.service, 'arm_home_request_id', lambda: None)()):
                    # Stop the already reserved home; do not queue a second one.
                    self._stop_captured_target(item['target'], capture['reference_context'],
                                               language='ko' if re.search(r'[가-힣]', text) else 'en',
                                               follow_up=True, generation=item['generation'],
                                               background_feedback=True)
                elif ((item['target'] is not None or item['ambiguous_target']) and stop_prefix
                      and not speech.get('direct_audio')):
                    # A separately validated immediate stop must not wait for
                    # the language model or be rejected as a secondary turn.
                    # The remaining words are not an execution capability.
                    self._stop_captured_target(item['target'], capture['reference_context'],
                                               ambiguous=item['ambiguous_target'],
                                               language='ko' if re.search(r'[가-힣]', text) else 'en',
                                               follow_up=True, generation=item['generation'],
                                               background_feedback=True)
                elif (speech.get('direct_audio') and stop_prefix and item['target'] is not None
                      and capture['reference_context'].get('active_request_id') is None):
                    completed, pending_cancelled = self._prepare_pending_compound(capture, text, item['target'])
                    with self._state_lock:
                        if self._retarget_thread is not None:
                            self.coordinator.discard_interpretation(ident)
                            raise RuntimeError('이전 변경 요청을 확인하고 있습니다.')
                        handler = self._pending_compound if pending_cancelled else self._retarget
                        target = item['target'] if not pending_cancelled else None
                        self._retarget_thread = threading.Thread(target=handler,
                            args=(completed, text, target, self.stop_event, item['generation']),
                            kwargs={'pending_target': item['target']} if pending_cancelled else {},
                            daemon=True, name='demo-voice-pending-stop-home')
                        self._retarget_thread.start()
                        handed_off = True
                elif item['target'] is not None and not item['ambiguous_target'] and (
                        callable(getattr(self.service, 'interpret_voice_followup', None))
                        or callable(getattr(self.service, 'request_voice_retarget', None))):
                    with self._state_lock:
                        busy = self._retarget_thread is not None
                    if busy:
                        self.emit('chat_error', '이전 변경 요청을 확인하고 있습니다. 잠시 뒤 다시 말씀해 주세요.')
                        self._say_feedback('이전 변경 요청을 확인하고 있습니다. 잠시 뒤 다시 말씀해 주세요.',
                                           generation=item['generation'])
                        continue
                    completed = self.coordinator.complete_voice_transcription(ident, text)
                    cancellation = (_EitherEvent(self.stop_event, item['target_cancel'])
                                    if item.get('target_cancel') is not None else self.stop_event)
                    with self._state_lock:
                        if cancellation.is_set():
                            continue
                        self._retarget_thread = threading.Thread(target=self._retarget,
                            args=(completed, text, item['target'], cancellation, item['generation']),
                            daemon=True, name='demo-voice-retarget')
                        self._retarget_thread.start()
                        handed_off = True
                elif not item['primary'] or item['target'] is not None or item['ambiguous_target']:
                    self.emit('chat_error', '이전 요청 처리 중에 시작한 발화는 새 작업으로 실행하지 않았습니다. 끝난 뒤 다시 말씀해 주세요.')
                    self._say_feedback('다른 요청을 처리하는 동안 시작한 발화라 새 작업을 실행하지 않았습니다. 다시 말씀해 주세요.',
                                       generation=item['generation'])
                else:
                    completed = self.coordinator.complete_voice_transcription(ident, text)
                    with self._state_lock:
                        if self.stop_event.is_set() or item['cancel_event'].is_set():
                            continue
                        self._pending_plan = {'transcript': text, 'program': copy.deepcopy(speech.get('program'))}
                        self._task_thread = threading.Thread(target=self._execute,
                            args=(completed, text, capture['request_id'], item['cancel_event'], item['generation']),
                            daemon=True, name='demo-voice-command')
                        self._task_thread.start()
                        handed_off = True
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.emit('chat_error', '음성 요청을 처리하지 못했습니다: ' + str(exc))
                    finished = self._finish_nonactionable_primary(item)
                    say = self._say_feedback if not item['primary'] or finished else self._say
                    say('음성을 처리하지 못했습니다. 다시 말씀해 주세요.', generation=item['generation'])
            finally:
                if not handed_off:
                    self._abort_held_followup(item)
                    self.coordinator.discard_interpretation(ident)
                    with self._state_lock:
                        if item['primary'] and not (self._pending_id == capture['request_id']
                                                    and self._pending_phase == 'nonactionable_waiting'):
                            if self._pending_id == capture['request_id']:
                                self._clear_pending_state_locked()
                            forget = getattr(self.coordinator, 'forget_pending_voice', None)
                            if callable(forget):
                                forget(capture['request_id'])
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._abort_held_followup(item)
            self.coordinator.discard_interpretation(item['capture']['interpretation_id'])

    def _finish_nonactionable_primary(self, item):
        """Finish an idle stop or failed turn without transferring its target."""
        if not item['primary']:
            return False
        request = item['capture']['request_id']
        with self._state_lock, self.coordinator._lock:
            if (self._pending_id != request or self._pending_plan is not None
                    or self._pending_cancel is None or self._pending_cancel.is_set()):
                return False
            holds = getattr(self.coordinator, '_voice_holds', {})
            hold = holds.get(request)
            if hold and hold['current'] == request and hold['tokens'] and not hold['cancelled']:
                hold['finished_nonactionable'] = True
                self._pending_phase = 'nonactionable_waiting'
            else:
                self._pending_phase = 'finished'
            return True

    def _release_nonactionable_hold(self, item):
        """Transfer queued follow-ups when the preceding turn had no action."""
        hold = item['hold']
        ident = item['capture']['interpretation_id']
        with self.service._text_request_lock, self._state_lock, self.coordinator._lock:
            if not hold.get('finished_nonactionable'):
                return False
            if ident not in hold['tokens']:
                raise RuntimeError('후속 발화의 보류 상태를 확인하지 못했습니다.')
            old = hold['current']
            first = not hold.get('nonactionable_followup_used', False)
            promote = (first and not self.stop_event.is_set() and not self.service._shutting_down
                       and self.service.coordinator is self.coordinator
                       and self._pending_id == old and self._pending_phase == 'nonactionable_waiting'
                       and self._pending_cancel is not None and not self._pending_cancel.is_set()
                       and self.service._text_request is None and self.service._record_request is None
                       and self.coordinator._active is None and not self.coordinator._pending
                       and self.coordinator._arm_home_reservation is None)
            if promote:
                request = item['capture']['request_id']
                self.coordinator.resolve_pending_voice(hold, ident, replacement=request)
                hold.pop('finished_nonactionable', None)
                hold.pop('nonactionable_followup_used', None)
                self._pending_id = request
                self._pending_cancel = threading.Event()
                self._pending_phase = 'processing'
                self._pending_plan = None
                self._pending_speech_generation = item['generation']
                item.update(primary=True, target=None, ambiguous_target=False,
                            target_cancel=None, cancel_event=self._pending_cancel)
                self.coordinator.forget_pending_voice(old)
            else:
                hold['nonactionable_followup_used'] = True
                self.coordinator.resolve_pending_voice(hold, ident)
            item['hold_resolved'] = True
            item['hold'] = None
            if not hold['tokens']:
                if self._pending_id == old and self._pending_phase == 'nonactionable_waiting':
                    self._clear_pending_state_locked()
                self.coordinator.forget_pending_voice(old)
            return True

    def _abort_held_followup(self, item):
        """An incomplete follow-up cannot release the old plan for execution."""
        hold = item.get('hold')
        if hold is None or item.get('hold_resolved'):
            return
        ident = item['capture']['interpretation_id']
        with self._state_lock, self.coordinator._lock:
            if ident in hold['tokens']:
                if hold.get('finished_nonactionable'):
                    old = hold['current']
                    self.coordinator.resolve_pending_voice(hold, ident)
                    if not hold['tokens']:
                        if self._pending_id == old and self._pending_phase == 'nonactionable_waiting':
                            self._clear_pending_state_locked()
                        self.coordinator.forget_pending_voice(old)
                    item['hold_resolved'] = True
                    return
                if self._pending_id == hold['current'] and self._pending_cancel is not None:
                    self._pending_cancel.set()
                text_pending = item.get('text_pending')
                if text_pending is not None and text_pending['target'] == hold['current']:
                    text_pending['cancelled'].set()
                self.coordinator.resolve_pending_voice(hold, ident, cancel=True)
                if self._pending_id != hold['current'] and text_pending is None:
                    self.coordinator.forget_pending_voice(hold['current'])
            item['hold_resolved'] = True

    def _resolve_voice_output_hold(self, item, output):
        """Release this voice-only follow-up without changing the held plan."""
        self._set_voice_output(output == 'unmute', generation=item['generation'])
        hold, ident = item['hold'], item['capture']['interpretation_id']
        with self._state_lock, self.coordinator._lock:
            if hold['cancelled'] or ident not in hold['tokens']:
                raise InterruptedError('이전 요청이 취소되었습니다.')
            self.coordinator.resolve_pending_voice(hold, ident)
            if self._pending_id == hold['current']:
                self._pending_speech_generation = item['generation']
            item['hold_resolved'] = True

    def _recognize_pending_text_audio(self, item):
        """Apply a spoken follow-up to the same still-unstarted text request."""
        from .task_program_v2 import TaskProgramInterpreter

        capture, hold, owner = item['capture'], item['hold'], item['text_pending']
        ident, target = capture['interpretation_id'], owner['target']
        cancelled, plan = owner['cancelled'], owner['plan']
        cancellation = _EitherEvent(self.stop_event, cancelled)
        started = time.monotonic()
        recognized = None
        if self.recognizer is not None:
            recognized = self.recognizer.transcribe(item['audio'], language=self.language)
            text = recognized.get('text', '').strip() if isinstance(recognized, dict) else ''
            if not text:
                raise ValueError('인식된 문장이 없습니다.')
            from .task_program_v2 import direct_voice_output_control
            output = direct_voice_output_control(text)
            if output is not None:
                self.emit('chat_transcript', {'text': text, 'asr_seconds': time.monotonic() - started,
                                             'capture_seconds': len(item['audio']) / RATE,
                                             'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                             'direct_audio': False})
                self._resolve_voice_output_hold(item, output)
                return False
            if _STOP.fullmatch(text) or _is_polite_immediate_halt(text):
                self.emit('chat_transcript', {'text': text, 'asr_seconds': time.monotonic() - started,
                                             'capture_seconds': len(item['audio']) / RATE,
                                             'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                             'direct_audio': False})
                self._stop_captured_target(target, capture['reference_context'],
                                           language='ko' if re.search(r'[가-힣]', text) else 'en',
                                           generation=item['generation'], background_feedback=True)
                self._abort_held_followup(item)
                return False

        def latest_plan():
            with self.service._text_request_lock, self._state_lock, self.coordinator._lock:
                entry = self.service._text_request
                if (cancellation.is_set() or self.service._shutting_down
                        or self.service.coordinator is not self.coordinator or entry is None
                        or entry[0] != target or entry[1] is not cancelled or entry[2] is not self.coordinator
                        or hold['current'] != target or hold['cancelled'] or ident not in hold['tokens']):
                    raise InterruptedError('이전 요청을 더 이상 확인할 수 없습니다.')
                result = plan['result']
                if result is None:
                    if plan['ready'].is_set():
                        raise InterruptedError('이전 요청을 더 이상 확인할 수 없습니다.')
                    return None
                if result.status != 'ok' or result.unresolved or not isinstance(result.program, dict):
                    raise ValueError('이전 요청의 작업 내용을 확인하지 못했습니다.')
                return {'transcript': plan['text'], 'program': copy.deepcopy(result.program)}

        while True:
            pending = latest_plan()
            if pending is not None:
                break
            if cancellation.wait(.05):
                raise InterruptedError('음성 요청을 취소했습니다.')

        interpreter = TaskProgramInterpreter(self.service.interpreter.start(cancellation).client)

        class TextTargetInterpreter:
            def interpret_pending_audio(self, audio, *, request_id, **kwargs):
                return interpreter.interpret_pending_audio(audio, request_id=target, **kwargs)

        def interpret(current, *, rebase=False):
            if recognized is not None:
                return interpreter.interpret_pending_text(
                    recognized['text'], request_id=target, reference_context=capture['reference_context'],
                    pending=current, cancel_event=cancellation)
            return self.coordinator.interpret_pending_voice_audio(
                ident, TextTargetInterpreter(), item['audio'], current, language=self.language,
                cancel_event=cancellation, rebase=rebase)

        text, result, disposition = interpret(pending)
        is_stop, stop_output = _pending_stop_output(disposition)
        if not is_stop:
            # A spoken stop cancels as soon as recognized. Other edits use the
            # results of earlier text/voice follow-ups in their arrival order.
            self.coordinator.wait_pending_followup_turn(hold, ident, cancellation)
            latest = latest_plan()
            if latest != pending:
                pending = latest
                text, result, disposition = interpret(pending, rebase=True)

        is_stop, stop_output = _pending_stop_output(disposition)
        if text:
            self.emit('chat_transcript', {'text': text, 'asr_seconds': time.monotonic() - started,
                                         'capture_seconds': len(item['audio']) / RATE,
                                         'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                         'direct_audio': recognized is None})
        if disposition in ('voice_output_mute', 'voice_output_unmute'):
            self._resolve_voice_output_hold(item, disposition.removeprefix('voice_output_'))
            return False
        if is_stop:
            steps = (result.program if result is not None else {}).get('steps', [])
            if len(steps) > 1 and recognized is None:
                completed, pending_cancelled = self._prepare_pending_compound(capture, text, target)
                self._abort_held_followup(item)
                with self._state_lock:
                    if self._retarget_thread is not None:
                        raise RuntimeError('이전 변경 요청을 확인하고 있습니다.')
                    handler = self._pending_compound if pending_cancelled else self._retarget
                    self._retarget_thread = threading.Thread(target=handler,
                        args=(completed, text, None if pending_cancelled else target, self.stop_event,
                              item['generation']),
                        kwargs={'pending_target': target} if pending_cancelled else {},
                        daemon=True, name='demo-voice-pending-stop-home')
                    self._retarget_thread.start()
                if stop_output is not None:
                    self._set_voice_output(stop_output == 'unmute', generation=item['generation'])
                return True
            self._stop_captured_target(target, capture['reference_context'],
                                       language='ko' if re.search(r'[가-힣]', text) else 'en',
                                       generation=item['generation'], background_feedback=True)
            self._abort_held_followup(item)
            if stop_output is not None:
                self._set_voice_output(stop_output == 'unmute', generation=item['generation'])
            return False

        if disposition == 'uncertain':
            self._abort_held_followup(item)
            message = '변경할 내용을 확인하지 못해 이전 요청을 실행하지 않았습니다. 다시 말씀해 주세요.'
            self.emit('chat_state', message)
            self._say_feedback(message, generation=item['generation'])
            return False

        with self.service._text_request_lock, self._state_lock, self.coordinator._lock:
            entry = self.service._text_request
            if (cancellation.is_set() or self.service.coordinator is not self.coordinator or entry is None
                    or entry[0] != target or entry[1] is not cancelled or hold['cancelled']):
                raise InterruptedError('이전 요청이 취소되었습니다.')
            if disposition == 'replace' and text and result is not None:
                self.coordinator.replace_pending_text_plan(hold, ident, result)
                plan['result'] = copy.deepcopy(result)
                plan['text'] = pending['transcript'] + '\n' + text
                self.coordinator._pending_model_results[target] = copy.deepcopy(result)
                message = '변경한 요청으로 이어서 실행합니다.'
            elif disposition == 'unchanged':
                self.coordinator.resolve_pending_voice(hold, ident)
                message = '이전 요청을 유지합니다.'
            else:
                raise ValueError('후속 발화의 해석 결과를 확인하지 못했습니다.')
            item['hold_resolved'] = True
        self.emit('chat_state', message)
        self._say_feedback(message, generation=item['generation'])
        return False

    def _recognize_pending_followup(self, item):
        """Interpret the new recording in the context of the latest unstarted plan."""
        capture, hold = item['capture'], item['hold']
        ident = capture['interpretation_id']
        while True:
            with self._state_lock, self.coordinator._lock:
                if (hold['cancelled'] or self._pending_id != hold['current']
                        or self._pending_cancel is None or self._pending_cancel.is_set()):
                    raise InterruptedError('이전 요청을 더 이상 확인할 수 없습니다.')
                if self._pending_plan is not None:
                    pending = copy.deepcopy(self._pending_plan)
                    break
            if self.stop_event.wait(.05):
                raise InterruptedError('음성 요청을 취소했습니다.')
        started = time.monotonic()
        speech = self.service.interpret_pending_voice_audio(
            item['audio'], capture=capture, coordinator=self.coordinator, pending=pending,
            language=self.language, cancel_event=self.stop_event)
        is_stop, _ = _pending_stop_output(speech.get('pending_disposition'))
        if not is_stop:
            # A spoken stop can be acted on as soon as the model recognizes
            # it. Other results apply in input order and must use the latest
            # plan if an earlier text/voice edit completed in the meantime.
            self.coordinator.wait_pending_followup_turn(hold, ident, self.stop_event)
            with self._state_lock, self.coordinator._lock:
                if (hold['cancelled'] or self._pending_id != hold['current'] or self._pending_plan is None
                        or self._pending_cancel is None or self._pending_cancel.is_set()):
                    raise InterruptedError('이전 요청을 더 이상 확인할 수 없습니다.')
                latest = copy.deepcopy(self._pending_plan)
            if latest != pending:
                pending = latest
                speech = self.service.interpret_pending_voice_audio(
                    item['audio'], capture=capture, coordinator=self.coordinator, pending=pending,
                    language=self.language, cancel_event=self.stop_event, rebase=True)
        text = speech.get('text', '').strip()
        disposition = speech.get('pending_disposition')
        is_stop, stop_output = _pending_stop_output(disposition)
        if text:
            self.emit('chat_transcript', {'text': text, 'asr_seconds': time.monotonic() - started,
                                         'capture_seconds': len(item['audio']) / RATE,
                                         'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                         'direct_audio': True, 'engine': speech.get('engine'),
                                         'device': speech.get('device')})
        if disposition in ('voice_output_mute', 'voice_output_unmute'):
            self._resolve_voice_output_hold(item, disposition.removeprefix('voice_output_'))
            return False
        if disposition == 'unchanged':
            with self._state_lock, self.coordinator._lock:
                if (hold['cancelled'] or self._pending_id != hold['current'] or self._pending_cancel is None
                        or self._pending_cancel.is_set()):
                    raise InterruptedError('이전 요청이 취소되었습니다.')
                self.coordinator.resolve_pending_voice(hold, ident)
                # Only this interpretation can carry the original request's
                # announcements across its own interruption. A later onset
                # still invalidates this generation until it is classified.
                self._pending_speech_generation = item['generation']
                item['hold_resolved'] = True
            return False
        steps = (speech.get('program') or {}).get('steps') or []
        if is_stop and len(steps) > 1:
            with self._state_lock, self.coordinator._lock:
                target = hold['current']
                if (self.stop_event.is_set() or self._retarget_thread is not None
                        or self._pending_id != target or hold['cancelled']
                        or self._pending_cancel is None or self._pending_cancel.is_set()):
                    raise RuntimeError('정지 이후 요청을 확인하지 못했습니다.')
                self._promote_pending_followup_locked(
                    ident, text, capture, item, hold, pending, speech,
                    self._run_pending_compound, 'demo-voice-pending-stop-home',
                    use_stop_event=True, set_retarget=True)
            if stop_output is not None:
                self._set_voice_output(stop_output == 'unmute', generation=item['generation'])
            return True
        if is_stop or disposition == 'uncertain':
            self._abort_held_followup(item)
            if stop_output is not None:
                self._set_voice_output(stop_output == 'unmute', generation=item['generation'])
            message = ('이전 요청을 취소했습니다.' if is_stop else
                       '변경할 내용을 확인하지 못해 이전 요청을 실행하지 않았습니다. 다시 말씀해 주세요.')
            self.emit('chat_state', message)
            self._say_feedback(message, generation=item['generation'])
            return False
        if disposition != 'replace' or not text or not isinstance(speech.get('program'), dict):
            raise ValueError('후속 발화의 해석 결과를 확인하지 못했습니다.')
        with self._state_lock, self.coordinator._lock:
            if (self.stop_event.is_set() or hold['cancelled'] or self._pending_id != hold['current']
                    or self._pending_cancel is None or self._pending_cancel.is_set()):
                raise InterruptedError('이전 요청이 취소되었습니다.')
            self._promote_pending_followup_locked(
                ident, text, capture, item, hold, pending, speech,
                self._run_pending_replacement, 'demo-voice-correction')
        return True

    def _run_pending_replacement(self, previous, capture, text, request, cancellation, generation, hold):
        if previous is not None:
            previous.join()
        self._execute(capture, text, request, cancellation, generation, pending_hold=hold)

    def _run_pending_compound(self, previous, capture, text, request, cancellation, generation, hold):
        try:
            if previous is not None:
                previous.join()
            if cancellation.is_set():
                return
            self.coordinator.rebase_voice_after_cancelled_home(capture, hold)
            self._retarget(capture, text, None, cancellation, generation)
        except Exception as exc:
            if not cancellation.is_set():
                self.emit('chat_error', '정지 이후 요청을 처리하지 못했습니다: ' + str(exc))
        finally:
            self.coordinator.discard_interpretation(capture['interpretation_id'])
            with self._state_lock:
                if self._pending_id == request:
                    self._pending_id = self._pending_cancel = self._pending_phase = self._pending_plan = None
                    self._pending_speech_generation = None
                if self._retarget_thread is threading.current_thread():
                    self._retarget_thread = None
                self.coordinator.forget_pending_voice(request)

    def _clear_pending_state_locked(self):
        self._pending_id = None
        self._pending_cancel = None
        self._pending_phase = None
        self._pending_plan = None
        self._pending_speech_generation = None

    def _promote_pending_followup_locked(self, ident, text, capture, item, hold, pending, speech,
                                         runner, thread_name, *, use_stop_event=False, set_retarget=False):
        completed = self.coordinator.complete_voice_transcription(ident, text)
        previous = self._task_thread
        self._pending_cancel.set()
        self.coordinator.resolve_pending_voice(hold, ident, replacement=capture['request_id'])
        item['hold_resolved'] = True
        self._pending_id = capture['request_id']
        self._pending_cancel = threading.Event()
        self._pending_phase = 'processing'
        self._pending_speech_generation = item['generation']
        self._pending_plan = {'transcript': pending['transcript'] + '\n' + text,
                              'program': copy.deepcopy(speech['program'])}
        cancellation = (_EitherEvent(self.stop_event, self._pending_cancel)
                        if use_stop_event else self._pending_cancel)
        worker = threading.Thread(
            target=runner,
            args=(previous, completed, text, capture['request_id'], cancellation, item['generation'], hold),
            daemon=True, name=thread_name)
        if set_retarget:
            self._retarget_thread = self._task_thread = worker
        else:
            self._task_thread = worker
        worker.start()

    def _prepare_pending_compound(self, capture, text, target):
        """Cancel exactly the pending turn, or bind it if it became active."""
        coordinator = self.coordinator
        with self.service._text_request_lock, self._state_lock, coordinator._lock:
            current = coordinator.snapshot()['reference_context']['active_request_id']
            if current == target:
                coordinator.bind_pending_voice_target(capture['interpretation_id'], target)
                return coordinator.complete_voice_transcription(capture['interpretation_id'], text), False
            if current is not None:
                raise RuntimeError('발화 당시 요청이 바뀌어 다른 작업은 중단하지 않았습니다.')
            own = self._pending_id == target and self._pending_cancel is not None
            matched = next((entry for entry in (self.service._text_request, self.service._record_request)
                            if entry is not None and entry[0] == target and entry[2] is coordinator), None)
            if not own and matched is None:
                raise RuntimeError('발화 당시 요청이 이미 끝났습니다.')
            completed = coordinator.complete_voice_transcription(capture['interpretation_id'], text)
            if own:
                self._pending_cancel.set()
                for ticket, entry in list(coordinator._pending.items()):
                    if entry['program']['request_id'] == target:
                        coordinator._pending.pop(ticket, None)
            else:
                result = self.service.stop(target_request_id=target)
                if result.get('status') != 'cancelled_pending_request':
                    raise RuntimeError('기존 요청의 취소를 확인하지 못했습니다.')
        self.emit('chat_state', '기존 요청의 실행을 취소했습니다. 처리가 끝난 뒤 후속 요청을 확인합니다.')
        return completed, True

    def _pending_compound(self, capture, text, target, cancellation, generation=None, *, pending_target):
        while not cancellation.is_set():
            with self.service._text_request_lock, self._state_lock:
                pending = (self._pending_id == pending_target or any(
                    entry is not None and entry[0] == pending_target
                    for entry in (self.service._text_request, self.service._record_request)))
            if not pending:
                return self._retarget(capture, text, target, cancellation, generation)
            cancellation.wait(.05)
        self.coordinator.discard_interpretation(capture['interpretation_id'])
        with self._state_lock:
            self._retarget_thread = None

    def _retarget(self, capture, text, target, cancellation, generation=None):
        """Interpret a follow-up without blocking recognition of a stop."""
        try:
            if cancellation.is_set():
                return
            followup = getattr(self.service, 'interpret_voice_followup', None) or self.service.request_voice_retarget
            result = followup(capture, text, target, cancel_event=cancellation, coordinator=self.coordinator)
            if cancellation.is_set():
                return
            if result is None:
                self._read_only_followup(capture, text, target, cancellation, generation)
                return
            if 'admission' in result or 'result' in result:
                self.emit('chat_result', {**result, 'in_motion_followup': True})
                self._say(spoken_result(result), generation=generation)
                return
            reply = result.get('reply') if isinstance(result, dict) else None
            if (not isinstance(reply, dict) or reply.get('kind') not in
                    ('retarget_requested', 'retarget_rejected', 'retarget_unsupported', 'retarget_unknown')
                    or not isinstance(reply.get('text'), str) or not reply['text'].strip()):
                raise ValueError('변경 요청의 결과를 확인하지 못했습니다.')
            result = {**result, 'in_motion_followup': True}
            if reply['kind'] in ('retarget_requested', 'retarget_unknown'):
                english = bool(re.search(r'[A-Za-z]', text)) and not bool(re.search(r'[가-힣]', text))
                observed, message = _retarget_observation(result.get('snapshot'), target, english)
                if (observed == 'pending' and (reply['kind'] == 'retarget_unknown'
                        or (result.get('retarget') or {}).get('remaining_program_stopped') is True)):
                    message = reply['text']
                elif (result.get('retarget') or {}).get('remaining_program_stopped') is True:
                    # A fast executor may already have applied or finished the
                    # change before the first voice response. Still explain
                    # that the old compound/repeated remainder will not run.
                    message += (' The remaining previous tasks will not run.' if english
                                else ' 남은 기존 작업은 실행하지 않습니다.')
                result = {**result, 'reply': {**reply, 'text': message}}
            else:
                observed = 'finished'
            self.emit('chat_result', result)
            announcement = threading.Event()
            if observed != 'finished':
                with self._state_lock:
                    own_target = self._pending_id == target
                threading.Thread(target=self._watch_retarget,
                    args=(target, english, observed == 'applied', own_target, cancellation, announcement, generation),
                    daemon=True, name='demo-voice-retarget-status').start()
            try:
                self._say(result['reply']['text'], generation=generation)
            finally:
                announcement.set()
        except Exception as exc:
            if not cancellation.is_set():
                self.emit('chat_error', '변경 요청을 처리하지 못했습니다: ' + str(exc))
                self._say('변경 요청을 처리하지 못했습니다. 화면을 확인해 주세요.', generation=generation)
        finally:
            try:
                self.coordinator.discard_interpretation(capture['interpretation_id'])
            finally:
                with self._state_lock:
                    self._retarget_thread = None

    def _watch_retarget(self, target, english, announced_applied, own_target, cancellation, announcement,
                        generation=None):
        """Report later executor evidence while the original task is active."""
        while not self.stop_event.wait(.1):
            snapshot = self.coordinator.snapshot()
            observed, message = _retarget_observation(snapshot, target, english)
            if observed == 'pending' or (observed == 'applied' and announced_applied):
                continue
            if observed == 'applied':
                announced_applied = True
            elif own_target and not cancellation.is_set():
                # The original voice task emits and speaks its own final result.
                return
            self.emit('chat_result', {'reply': {'kind': 'retarget_' + observed, 'text': message, 'speak': True},
                                      'snapshot': snapshot, 'in_motion_followup': True})
            while not announcement.wait(.1):
                if self.stop_event.is_set():
                    return
            if not self.stop_event.is_set():
                self._say(message, generation=generation)
            if observed == 'finished':
                return

    def _read_only_followup(self, capture, text, target, cancellation, generation=None):
        """Answer ordinary conversation without admitting another work program."""
        from .astra_offline_voice_turn import OfflineTurnRouter
        from .demo_service import (_direct_robot_action_question, _read_only_dialogue_reply,
                                   _robot_request_restatement)

        router = OfflineTurnRouter()
        turn, _ = router.begin(active_request=target, playback_active=False, near_end_voice=True)
        current = self.coordinator.snapshot()['reference_context'].get('active_request_id')
        decision = router.final(turn, text, current_active=current, context=capture['reference_context'])
        if decision.kind == 'read_only_conversation' and not _direct_robot_action_question(text):
            answer = self.service.interpreter.dialogue(text, cancellation)
            if cancellation.is_set():
                return
            message, filtered = _read_only_dialogue_reply(text, answer)
            reply = {'kind': 'read_only_conversation', 'text': message, 'filtered': filtered, 'speak': True}
        elif decision.kind == 'read_only_conversation':
            reply = {'kind': 'clarify', 'text': _robot_request_restatement(text), 'speak': True}
        else:
            message = decision.message or '변경 요청을 확인하지 못해 작업을 바꾸지 않았습니다. 필요한 내용을 다시 말씀해 주세요.'
            reply = {'kind': 'clarify', 'text': message, 'speak': True}
        self.emit('chat_result', {'reply': reply, 'snapshot': self.coordinator.snapshot(),
                                  'in_motion_followup': True})
        self._say(reply['text'], generation=generation)

    def _execute(self, capture, text, request, cancel_event, generation=None, *, pending_hold=None):
        announcement = None

        def request_generation():
            with self._state_lock:
                if self._pending_id == request and self._pending_speech_generation is not None:
                    return self._pending_speech_generation
                return generation

        def announce_wait():
            nonlocal announcement
            started = threading.Event()
            announcement = threading.Thread(target=self._say,
                args=('잠시만 기다려 주세요. 요청을 확인하고 있습니다.',),
                kwargs={'started': started, 'cancel_on_request': True, 'cancel_event': cancel_event,
                        'generation': request_generation()},
                daemon=True, name='demo-voice-wait')
            announcement.start()
            # Proceed when the local speaker accepted the async utterance, not
            # when it finishes. _speaker_lock serializes subsequent speech.
            started.wait()
        def announce_plan(program):
            def check_recovery():
                work = any(step.get('op') in ('task', 'return', 'repeat', 'continue')
                           for step in program.get('steps', []))
                if work:
                    with self.coordinator._lock:
                        revision = self.coordinator.snapshot().get('registration_revision')
                        if revision != self._bound_registration_revision:
                            self.require_recovery()
                if work and self.recovery_read_only:
                    raise RuntimeError('배치를 다시 등록하기 전에는 이동이나 새 기억 저장을 실행할 수 없습니다.')
            check_recovery()
            if not self._say(spoken_plan(program), cancel_on_request=True, cancel_event=cancel_event,
                             generation=request_generation()):
                raise InterruptedError('요청을 취소했습니다.')
            check_recovery()
        try:
            if pending_hold is not None:
                self.coordinator.rebase_voice_after_cancelled_home(capture, pending_hold)
            wait_pending = getattr(self.coordinator, 'wait_pending_voice', None)
            # Let the original utterance finish interpretation while a typed
            # correction waits for its plan. Admission itself remains held by
            # the coordinator until that correction is resolved.
            result = self.service._voice_route_and_execute(self.coordinator, capture, text, request,
                before_execute=announce_plan,
                before_model=announce_wait, cancel_event=cancel_event)
            has_execution_result = result.get('result') is not None
            while True:
                try:
                    if (callable(wait_pending) and not (has_execution_result
                            and (self.stop_event.is_set() or cancel_event.is_set()))):
                        wait_pending(request, _EitherEvent(self.stop_event, cancel_event))
                except InterruptedError:
                    # An already returned execution result remains evidence
                    # even when cancellation interrupts a later voice hold.
                    if not (has_execution_result and (self.stop_event.is_set() or cancel_event.is_set())):
                        raise
                with self._state_lock, self.coordinator._lock:
                    cancelled = self.stop_event.is_set() or cancel_event.is_set()
                    if cancelled and not has_execution_result:
                        raise InterruptedError('요청을 취소했습니다.')
                    held = getattr(self.coordinator, 'pending_voice_held', None)
                    if not cancelled and callable(held) and held(request):
                        continue
                    if self._pending_id == request:
                        self._pending_phase = 'finished'
                    break
            self.emit('chat_result', result)
            if not (self.stop_event.is_set() or cancel_event.is_set()) and (
                    not result.get('reply') or result['reply'].get('speak', True)):
                self._say(spoken_result(result), cancel_on_request=True, cancel_event=cancel_event,
                          generation=request_generation())
        except InterruptedError:
            self.emit('chat_state', '음성 요청이 취소됐습니다.')
        except Exception as exc:
            self.emit('chat_error', '요청을 실행하지 못했습니다: ' + str(exc))
            self._say('요청을 처리하지 못했습니다. 화면을 확인한 뒤 전체 요청을 다시 말씀해 주세요.',
                      generation=request_generation())
        finally:
            if announcement is not None:
                announcement.join()
            self.coordinator.discard_interpretation(capture['interpretation_id'])
            with self._state_lock:
                if self._pending_id == request:
                    self._clear_pending_state_locked()
                forget = getattr(self.coordinator, 'forget_pending_voice', None)
                if callable(forget):
                    forget(request)

    def _stop_target(self, target, *, language='ko', follow_up=False, generation=None,
                     background_feedback=False):
        say = self._say_feedback if background_feedback else self._say
        no_target = None
        stopped_motion = False
        completed = False
        finished = False
        with self._state_lock:
            if target is None or target != self._pending_id or self._pending_cancel is None:
                no_target = '해당 발화가 시작될 때의 작업이 없거나 이미 끝났습니다.'
            else:
                # Serialize against admission/execution. Cancel only this
                # request's token and tickets, never a newly active request.
                with self.coordinator._lock:
                    snapshot = self.coordinator.snapshot()
                    current = snapshot['reference_context']['active_request_id']
                    last = snapshot.get('last_execution') or {}
                    completed = ((last.get('request_id') == target and last.get('status') == 'completed')
                                 or any(event.get('event') == 'program_finished'
                                        and event.get('request_id') == target and event.get('status') == 'completed'
                                        for event in self.coordinator.events))
                    finished = completed or self._pending_phase == 'finished'
                    self._pending_cancel.set()
                    cancel_voice = getattr(self.coordinator, 'cancel_pending_voice', None)
                    if callable(cancel_voice):
                        cancel_voice(target)
                    if not finished:
                        for ticket, entry in list(self.coordinator._pending.items()):
                            if entry['program']['request_id'] == target:
                                self.coordinator._pending.pop(ticket, None)
                        if current == target:
                            self.coordinator.request_motion_stop(_request_id(), target_request_id=target)
                            stopped_motion = True
        if no_target is not None:
            if follow_up:
                no_target += ' 이후 작업은 실행하지 않았습니다.'
            self.emit('chat_state', no_target)
            spoken = ('There is no matching task to stop.' if language == 'en'
                      else '중단할 작업이 없습니다.')
            if follow_up:
                spoken += (' The follow-up actions were not executed.' if language == 'en'
                           else ' 이후 작업은 실행하지 않았습니다.')
            say(spoken, generation=generation)
            return False
        if finished:
            message = ('해당 요청의 실행은 이미 끝났습니다. 중단을 요청하지 않았습니다.' if completed
                       else '해당 요청은 이미 끝났습니다. 중단을 요청하지 않았습니다.')
            if follow_up:
                message += ' 이후 작업은 실행하지 않았습니다.'
            self.emit('chat_state', message)
            if language == 'en':
                spoken = ('Execution of the request has already completed. No stop was requested.' if completed
                          else 'That request has already finished; no stop was requested.')
                say(spoken + (' The follow-up actions were not executed.' if follow_up else ''),
                    generation=generation)
            else:
                say(message, generation=generation)
            return False
        message = ('작업 중단을 요청했습니다. 정지 여부는 아직 확인되지 않았습니다.' if stopped_motion
                   else '해당 요청의 실행을 취소했습니다. 다른 작업은 중단하지 않았습니다.')
        if stopped_motion:
            self._emit_motion_stop(target)
        if follow_up:
            message += ' 이후 작업은 실행하지 않았습니다.'
        self.emit('chat_state', message)
        if language == 'en':
            spoken = ('A stop was requested for the task. Whether it has stopped is not yet confirmed.' if stopped_motion
                      else 'I cancelled that request. No other task was stopped.')
            say(spoken + (' The follow-up actions were not executed.' if follow_up else ''),
                generation=generation)
        else:
            say(message, generation=generation)
        return True

    def _stop_captured_target(self, target, context, *, ambiguous=False, language='ko', follow_up=False,
                              generation=None, background_feedback=False):
        """Stop only the task or pending turn recorded at speech onset."""
        with self._state_lock:
            own_target = (target is not None and target == self._pending_id
                          and self._pending_cancel is not None)
        if own_target:
            return self._stop_target(target, language=language, follow_up=follow_up, generation=generation,
                                     background_feedback=background_feedback)

        # A conversation may be listening while another service caller owns
        # the active task. The captured ID still has to name that same task at
        # the atomic stop boundary; a later task is never substituted.
        status = None
        if target is not None:
            with self.service._text_request_lock:
                if (not self.stop_event.is_set() and not self.service._shutting_down
                        and self.service.coordinator is self.coordinator):
                    # This ID was the sole active or pending candidate captured
                    # at speech onset. stop() checks it again under the same
                    # service/coordinator locks, including pending -> active.
                    status = self.service.stop(target_request_id=target).get('status')
        if status == 'stop_requested':
            self._emit_motion_stop(target)
            message = '작업 중단을 요청했습니다. 정지 여부는 아직 확인되지 않았습니다.'
            spoken = 'A stop was requested for the task. Whether it has stopped is not yet confirmed.'
        elif status == 'already_stopping':
            message = '작업 중단이 이미 요청됐으며 정지 여부는 아직 확인되지 않았습니다.'
            spoken = 'A stop has already been requested. Whether the task has stopped is not yet confirmed.'
        elif status == 'cancelled_pending_request':
            message = '해당 요청의 실행을 취소했습니다. 다른 작업은 중단하지 않았습니다.'
            spoken = 'I cancelled that request. No other task was stopped.'
        elif ambiguous:
            message = '중단할 요청을 구분할 수 없습니다.'
            spoken = 'The request to stop is ambiguous.'
        else:
            message = '발화가 시작될 때의 작업이 없거나 이미 끝났습니다.'
            spoken = 'The task captured when you spoke is no longer active.'
        if follow_up:
            message += ' 이후 작업은 실행하지 않았습니다.'
            spoken += ' The follow-up actions were not executed.'
        self.emit('chat_state', message)
        say = self._say_feedback if background_feedback else self._say
        say(spoken if language == 'en' else message, generation=generation)
        return status is not None

    def _emit_motion_stop(self, target):
        # The stop has already been dispatched; a UI notification failure
        # must not undo it or misreport the original stop as unsuccessful.
        try:
            self.emit('motion_stop_requested', {'target_request_id': target})
        except Exception:
            pass

    def stop_pending_prefix(self, target, text, *, language='ko'):
        """Cancel a validated stop prefix during interpretation; None means active."""
        rejected = False
        with self._state_lock:
            if target is None or target != self._pending_id or self._pending_cancel is None:
                rejected = True
            else:
                with self.coordinator._lock:
                    context = self.coordinator.snapshot()['reference_context']
                    if context.get('active_request_id') == target:
                        return None
                    rejected = not _validated_pending_stop_prefix(text, context)
        if rejected:
            self.emit('chat_state', '정지 대상이나 요청을 확인하지 못했습니다. 다시 확인해 주세요.')
            return False
        return self._stop_target(target, language=language, follow_up=True)

    def stop_motion(self):
        target = self.current_target()
        home = getattr(self.service, 'arm_home_request_id', lambda: None)()
        if home is not None and target == home:
            return self._stop_captured_target(target, {}, language=self.language)
        return self._stop_target(target)

    def current_target(self):
        with self._state_lock:
            # The worker keeps its ID until TTS finishes, but its completed
            # task must not capture a new text request during playback.
            pending = self._pending_id if self._pending_phase != 'finished' else None
        home = getattr(self.service, 'arm_home_request_id', lambda: None)()
        return home if home is not None else pending

    def stop(self):
        self.stop_event.set()
        cancel_error = None
        try:
            if hasattr(self.recognizer, 'cancel_pending'):
                self.recognizer.cancel_pending()
        except Exception as exc:
            cancel_error = exc
        with self._state_lock:
            pending = self._pending_id
        home = getattr(self.service, 'arm_home_request_id', lambda: None)()
        if home is not None and home != pending:
            try:
                self.service.stop(target_request_id=home)
            except Exception as exc:
                self.emit('chat_error', '원위치 중단 요청 오류: ' + str(exc))
        if pending is not None:
            try:
                self._stop_target(pending)
            except Exception as exc:
                self.emit('chat_error', '중단 요청 오류: ' + str(exc))
        if cancel_error is not None:
            self.emit('chat_error', '음성 인식 취소 오류: ' + str(cancel_error))

    def _supervise(self):
        close_error = None
        try:
            self._capture_thread.join()
            self.stop_event.set()
            self._recognition.join()
            with self._state_lock:
                feedback_thread = self._feedback_thread
            if feedback_thread is not None:
                feedback_thread.join()
            if self._prepare_thread is not None:
                self._prepare_thread.join()
            try:
                if hasattr(self.recognizer, 'close'):
                    self.recognizer.close()
            except Exception as exc:
                close_error = exc
            if self._task_thread is not None:
                self._task_thread.join()
            if self._retarget_thread is not None:
                self._retarget_thread.join()
        finally:
            with self.service._text_request_lock:
                self.service._conversation_requests.discard(self._session_id)
            try:
                if close_error is not None:
                    self.emit('chat_error', '음성 인식기 종료 오류: ' + str(close_error))
            finally:
                self.closed.set()
                self.emit('chat_closed', None)
