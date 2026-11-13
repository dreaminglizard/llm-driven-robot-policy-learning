"""Registration guidance while a connected robot has no registered scene.

This listener never creates an execution ticket. A recognized utterance can
only request cancellation of the preparation captured at speech onset, or
receive a prompt to register the scene.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .demo_service import _DIRECT_SPOKEN_STOP, _is_polite_immediate_halt
from .demo_voice_chat import (BLOCK, RATE, SpeechDetector, WindowsSpeaker, _EitherEvent,
                              _SpeechTurns, _microphone_stream, _shared_voice_output_state)


class SetupConversation:
    def __init__(self, emit, *, language='auto', stop_target=lambda: None,
                 stop_captured=lambda _target: False, stream_factory=None, recognizer=None,
                 speaker=None, response_guard_seconds=.5, audio_service=None):
        self.emit, self.language = emit, language
        self.stop_target, self.stop_captured = stop_target, stop_captured
        self.stream_factory = stream_factory or _microphone_stream
        self.recognizer, self.audio_service = recognizer, audio_service
        self.speaker = speaker or WindowsSpeaker()
        if stream_factory is None and speaker is None:
            from .demo_voice_duplex import AdaptiveVoiceIO, duplex_requested
            if duplex_requested():
                duplex = AdaptiveVoiceIO(_microphone_stream, self.speaker)
                self.stream_factory, self.speaker = lambda: duplex, duplex
        self.response_guard_seconds = response_guard_seconds
        self.stop_event, self.closed, self.speaking = threading.Event(), threading.Event(), threading.Event()
        self._handoff, self._capture_done = threading.Event(), threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._speech_turns = _SpeechTurns()
        self._voice_output_muted = _shared_voice_output_state(audio_service)
        self._duplex_active = self._duplex_fallback = False
        self._queue = queue.Queue(maxsize=2)
        self._started = False
        self._prepare_thread = None

    @property
    def voice_output_enabled(self):
        return not self._voice_output_muted.is_set()

    def current_target(self):
        return None

    def stop_motion(self):
        return False

    def start(self):
        if self._started:
            raise RuntimeError('이미 시작한 음성 대화입니다.')
        self._started = True
        self._capture_thread = threading.Thread(target=self._capture, daemon=True, name='setup-voice-input')
        self._recognition = threading.Thread(target=self._recognize, daemon=True, name='setup-voice-asr')
        if callable(getattr(self.recognizer, 'prepare', None)):
            self._prepare_thread = threading.Thread(target=self._prepare, daemon=True, name='setup-voice-prepare')
        try:
            if self._prepare_thread is not None:
                self._prepare_thread.start()
            self._recognition.start()
            self._capture_thread.start()
            threading.Thread(target=self._supervise, daemon=True, name='setup-voice-shutdown').start()
        except BaseException:
            self.stop_event.set()
            if callable(getattr(self.recognizer, 'cancel_pending', None)):
                self.recognizer.cancel_pending()
            for worker in (self._capture_thread, self._recognition, self._prepare_thread):
                if worker is not None and worker.ident is not None:
                    worker.join()
            if callable(getattr(self.recognizer, 'close', None)):
                self.recognizer.close()
            self.closed.set()
            raise

    def _prepare(self):
        try:
            if not self.stop_event.is_set():
                self.recognizer.prepare()
        except Exception:
            pass  # Transcription reports a persistent model error.

    def _capture(self):
        detector, active = SpeechDetector(), None
        mute_until = 0.
        overflow_streak = 0
        try:
            with self.stream_factory() as stream:
                self._duplex_active = bool(getattr(stream, 'echo_cancelled', False))
                self._duplex_fallback = bool(getattr(stream, 'fallback_reason', None))
                self.emit('chat_audio_mode', {'mode': 'aec' if self._duplex_active else 'half_duplex',
                                              'fallback': self._duplex_fallback})
                self.emit('chat_voice_output', {'enabled': self.voice_output_enabled})
                self.emit('chat_input_opened', None)
                self.emit('chat_state', self._listening_message())
                while not self.stop_event.is_set():
                    # Finish an utterance already in progress, including its
                    # silence boundary, before handing off the listener.
                    if (self._handoff.is_set() and active is None
                            and not detector.active and not detector.voiced):
                        break
                    data, overflow = stream.read(BLOCK)
                    if self.stop_event.is_set():
                        break
                    if overflow:
                        # A lost input block only invalidates the current
                        # utterance; retain the listener and ignore its tail
                        # until the detector sees a full silence boundary.
                        overflow_streak += 1
                        frame = np.asarray(data, dtype=np.float32).reshape(-1)
                        valid = len(frame) == BLOCK and np.isfinite(frame).all()
                        current_speech = (valid
                            and float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= detector.threshold)
                        had_utterance = (active is not None or detector.active or bool(detector.voiced)
                                         or current_speech)
                        active = None
                        detector.discard_after_overflow(known_speech=bool(had_utterance or not valid))
                        if had_utterance and overflow_streak == 1:
                            self.emit('chat_state', '음성이 잠깐 끊겨 해당 발화를 처리하지 못했습니다. 다시 말씀해 주세요.')
                        elif overflow_streak == 1:
                            self.emit('chat_state', '마이크 입력이 잠깐 끊겼습니다. 계속 듣고 있습니다.')
                        elif overflow_streak == 3:
                            self.emit('chat_state', '마이크 입력이 밀리고 있습니다. 입력이 돌아오면 계속 듣겠습니다.')
                        # Broken streams can return immediately; keep the
                        # listener cancellable without spinning the CPU.
                        self.stop_event.wait(.01)
                        continue
                    overflow_streak = 0
                    duplex_active = bool(getattr(stream, 'echo_cancelled', False))
                    if duplex_active != self._duplex_active:
                        interrupted = bool(active is not None or detector.active or detector.voiced or detector.discarding)
                        if interrupted:
                            detector.discard_after_overflow(known_speech=True)
                        else:
                            detector.reset()
                        active = None
                        self._speech_turns.interrupt()
                        self._duplex_active = duplex_active
                        self._duplex_fallback = bool(getattr(stream, 'fallback_reason', None))
                        mute_until = time.monotonic() + self.response_guard_seconds
                        self.emit('chat_audio_mode', {'mode': 'aec' if duplex_active else 'half_duplex',
                                                      'fallback': self._duplex_fallback})
                        self.emit('chat_state', self._listening_message())
                        continue
                    if not duplex_active and time.monotonic() < mute_until:
                        detector.feed_muted(data)
                        continue
                    if self.speaking.is_set() and not self._duplex_active:
                        detector.feed_muted(data)
                        active = None
                        continue
                    for event, audio in detector.feed(data):
                        if event == 'start':
                            active = {'stop_target': self.stop_target(),
                                      'generation': self._speech_turns.interrupt()}
                            self.emit('chat_utterance_start', None)
                            self.emit('chat_state', '말씀을 듣고 있습니다.')
                        elif event == 'discard':
                            active = None
                            self.emit('chat_error', '발화가 너무 길어 처리하지 않았습니다. 짧게 다시 말씀해 주세요.')
                        elif event == 'end' and active is not None:
                            active.update(audio=audio, ended=time.monotonic())
                            try:
                                self._queue.put_nowait(active)
                            except queue.Full:
                                self.emit('chat_error', '음성 처리가 밀려 이번 발화를 버렸습니다.')
                            active = None
        except Exception as exc:
            if not self.stop_event.is_set():
                self.emit('chat_error', '마이크를 사용할 수 없습니다. 운영체제의 마이크 권한과 장치를 확인한 뒤 다시 듣기를 눌러 주세요: ' + str(exc))
            self.stop()
        finally:
            try:
                if self._handoff.is_set():
                    self.emit('chat_input_closed', None)
            finally:
                self._capture_done.set()

    def _recognize(self):
        while not self.stop_event.is_set():
            try:
                item = self._queue.get(timeout=.1)
            except queue.Empty:
                if self._handoff.is_set() and self._capture_done.is_set():
                    break
                continue
            try:
                self.emit('chat_state', '음성 요청을 확인하고 있습니다.')
                started = time.monotonic()
                if self.recognizer is None:
                    if self.audio_service is None:
                        raise RuntimeError('직접 음성 처리 서비스를 확인해 주세요.')
                    speech = self.audio_service.interpret_voice_audio(
                        item['audio'], language=self.language, cancel_event=self.stop_event)
                    if not isinstance(speech, dict) or speech.get('direct_audio') is not True:
                        raise ValueError('직접 음성 처리 결과를 확인하지 못했습니다.')
                    is_stop = speech.get('immediate_stop') is True or speech.get('stop_prefix') is True
                else:
                    speech = self.recognizer.transcribe(item['audio'], language=self.language)
                    is_stop = None
                if self.stop_event.is_set():
                    break
                text = speech.get('text', '').strip() if isinstance(speech, dict) else ''
                if not text:
                    raise ValueError('인식된 문장이 없습니다.')
                if is_stop is None:
                    is_stop = bool(_DIRECT_SPOKEN_STOP.fullmatch(text) or _is_polite_immediate_halt(text))
                self.emit('chat_transcript', {'text': text, 'asr_seconds': time.monotonic() - started,
                                             'capture_seconds': len(item['audio']) / RATE,
                                             'end_to_transcript_seconds': time.monotonic() - item['ended'],
                                             'direct_audio': speech.get('direct_audio', False),
                                             'engine': speech.get('engine')})
                from .task_program_v2 import direct_voice_output_control
                output = speech.get('voice_output') or direct_voice_output_control(text)
                if self._handoff.is_set():
                    # Registration has completed. Preserve only output control
                    # from already captured turns; do not dispatch old setup work.
                    if output in ('mute', 'unmute') and self._apply_voice_output(output):
                        self.emit('chat_voice_output', {'enabled': self.voice_output_enabled})
                    continue
                if output in ('mute', 'unmute') and not is_stop:
                    if not self._apply_voice_output(output):
                        break
                    self.emit('chat_voice_output', {'enabled': self.voice_output_enabled})
                    self.emit('chat_state', '음성 출력을 다시 켰습니다. 계속 듣고 있습니다.' if self.voice_output_enabled
                              else '음성 출력을 껐습니다. 계속 듣고 있습니다.')
                    if not self.voice_output_enabled:
                        continue
                    reply = '음성 출력을 다시 켰습니다.'
                    generation = item['generation']
                else:
                    stopped = bool(item['stop_target'] and is_stop and self.stop_captured(item['stop_target']))
                    if is_stop and output in ('mute', 'unmute') and self._apply_voice_output(output):
                        self.emit('chat_voice_output', {'enabled': self.voice_output_enabled})
                    reply = ('준비 중단을 요청했습니다. 정지와 집게 상태를 확인해 주세요.' if stopped else
                             '먼저 시작 배치를 등록해 주세요. 등록 전에는 이동을 실행하지 않습니다.')
                    generation = item['generation']
                if self._voice_output_muted.is_set():
                    if generation == self._speech_turns.current():
                        self.emit('chat_response', reply)
                    self.emit('chat_state', self._listening_message())
                    continue
                playback_stop = self._speech_turns.begin(generation, self._voice_output_muted)
                if playback_stop is not None and not self.stop_event.is_set():
                    self.emit('chat_response', reply)
                    self.speaking.set()
                    try:
                        self.speaker.speak(reply, _EitherEvent(self.stop_event, playback_stop))
                    finally:
                        self.stop_event.wait(self.response_guard_seconds)
                        self.speaking.clear()
                        self._speech_turns.finish(playback_stop)
                if not self.stop_event.is_set():
                    self.emit('chat_state', self._listening_message())
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.emit('chat_error', '음성 요청을 처리하지 못했습니다. 다시 말씀하거나 화면을 사용해 주세요: ' + str(exc))

    def _listening_message(self):
        if self._voice_output_muted.is_set():
            return '음성 출력 꺼짐 · 계속 듣고 있습니다. 다시 켜려면 “다시 말해”'
        if self._duplex_active:
            return '음성을 듣고 있습니다. 답변 중에도 말씀할 수 있습니다. 작업 전 시작 배치를 등록해 주세요.'
        if self._duplex_fallback:
            return '끼어들기를 사용할 수 없어 답변이 끝난 뒤 듣습니다. 작업 전 시작 배치를 등록해 주세요.'
        return '음성을 듣고 있습니다. 작업 전 시작 배치를 등록해 주세요.'

    def _apply_voice_output(self, output):
        with self._lifecycle_lock:
            if self.stop_event.is_set():
                return False
            self._speech_turns.set_output_muted(self._voice_output_muted, output == 'mute')
            return True

    def finish_registration(self):
        """Drain captured output controls before handing over to the new listener."""
        with self._lifecycle_lock:
            if not self.stop_event.is_set():
                self._handoff.set()

    def stop(self):
        with self._lifecycle_lock:
            self.stop_event.set()
        if callable(getattr(self.recognizer, 'cancel_pending', None)):
            try:
                self.recognizer.cancel_pending()
            except Exception as exc:
                self.emit('chat_error', '음성 인식 취소 오류: ' + str(exc))

    def _supervise(self):
        try:
            self._capture_thread.join()
            if not self._handoff.is_set():
                self.stop_event.set()
            self._recognition.join()
            self.stop_event.set()
            if self._prepare_thread is not None:
                self._prepare_thread.join()
            if callable(getattr(self.recognizer, 'close', None)):
                self.recognizer.close()
        except Exception as exc:
            self.emit('chat_error', '음성 인식기 종료 오류: ' + str(exc))
        finally:
            self.closed.set()
            self.emit('chat_closed', None)
