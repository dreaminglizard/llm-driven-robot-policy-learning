"""Local, reference-fed duplex audio for barge-in.

The default SAPI speaker renders outside this process, so its output cannot be
used as an echo reference. This path synthesizes SAPI to memory and renders it
through the same PortAudio duplex stream that captures the microphone. Device
and room echo performance still requires real-world verification.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from copy import copy

import numpy as np

RATE = 16000
BLOCK = 1600
_LOG = logging.getLogger(__name__)


def _only_output_underflow(status):
    """Accept only that flag; unknown or combined flags remain failures."""
    try:
        if not status.output_underflow:
            return False
        remaining = copy(status)
        if remaining is status:
            return False
        remaining.output_underflow = False
        return not remaining
    except Exception:
        return False


def duplex_requested():
    requested = os.environ.get('RACCOON_VOICE_BARGE_IN', 'auto').strip().lower() != 'off'
    if not requested:
        _LOG.warning('Duplex audio is disabled by RACCOON_VOICE_BARGE_IN=off.')
    return requested


def _synthesize_windows(text):
    """Produce 16 kHz mono PCM in memory; never play through SAPI itself."""
    import pythoncom
    import win32com.client
    from .demo_voice_chat import _LOCAL_KOREAN_VOICE

    pythoncom.CoInitialize()
    speaker = memory = fmt = voice = None
    try:
        speaker = win32com.client.Dispatch('SAPI.SpVoice')
        for voice in speaker.GetVoices():
            if voice.Id.casefold() == _LOCAL_KOREAN_VOICE.casefold():
                speaker.Voice = voice
                break
        else:
            raise RuntimeError('오프라인 한국어 음성을 찾지 못했습니다.')
        fmt = win32com.client.Dispatch('SAPI.SpAudioFormat')
        fmt.Type = 18  # SAPI SAFT16kHz16BitMono
        memory = win32com.client.Dispatch('SAPI.SpMemoryStream')
        memory.Format = fmt
        speaker.AllowAudioOutputFormatChangesOnNextSet = False
        speaker.AudioOutputStream = memory
        speaker.Speak(text, 16)  # literal text, synchronous into memory
        raw = bytes(memory.GetData())
        if memory.Format.Type != 18 or not raw or len(raw) % 2:
            raise RuntimeError('음성 출력 형식을 확인하지 못했습니다.')
        return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    finally:
        voice = speaker = memory = fmt = None
        pythoncom.CoUninitialize()


class DuplexAudio:
    """One callback shares an output reference and 100 ms microphone frames."""

    echo_cancelled = True
    mode = 'aec'

    def __init__(self, *, stream_class=None, processor_factory=None, synthesize=None):
        self._stream_class = stream_class
        self._processor_factory = processor_factory
        self._synthesize = synthesize or _synthesize_windows
        self._frames = queue.Queue(maxsize=10)
        self._lock = threading.Lock()
        self._playback = None
        self._failure = None
        self._stream = None
        self._output_audio_submitted = False
        self._initial_output_underflows = 0
        self._reported_initial_underflow = False

    def __enter__(self):
        if self._stream_class is None:
            import sounddevice
            self._stream_class = sounddevice.Stream
        if self._processor_factory is None:
            from pywebrtc_audio import AudioProcessor
            self._processor_factory = AudioProcessor
        self._processor = self._processor_factory(sample_rate=RATE, echo_cancellation=True,
                                                   noise_suppression=True)
        self._stream = self._stream_class(samplerate=RATE, blocksize=BLOCK, channels=1,
                                          dtype='float32', callback=self._callback)
        try:
            self._stream.__enter__()
        except BaseException:
            self._stream.close()
            self._stream = None
            raise
        return self

    def __exit__(self, *args):
        with self._lock:
            stream, self._stream = self._stream, None
            pending, self._playback = self._playback, None
            if pending is not None:
                pending['done'].set()
        if stream is not None:
            return stream.__exit__(*args)

    def _callback(self, indata, outdata, frames, times, status):
        outdata.fill(0)
        # Before any speech is submitted there is no non-silent application
        # output for the gap to misalign. Backend-inserted samples themselves
        # are unobservable. After speech, retain the existing fallback because
        # the inserted data or gap can invalidate the echo reference.
        initial_silent_underflow = (frames == BLOCK and status and not self._output_audio_submitted
                                    and _only_output_underflow(status))
        if (status and not initial_silent_underflow) or frames != BLOCK:
            self._failure = ('전이중 음성 입출력의 버퍼를 유지하지 못했습니다. '
                             f'(status={status!s}, frames={frames}, expected={BLOCK})')
            return
        if initial_silent_underflow:
            self._initial_output_underflows += 1
        with self._lock:
            playback = self._playback
            if playback is not None:
                if not playback['stop'].is_set():
                    start = playback['offset']
                    end = min(start + frames, len(playback['samples']))
                    outdata[:end - start, 0] = playback['samples'][start:end]
                    if end > start:
                        self._output_audio_submitted = True
                    playback['offset'] = end
                if playback['stop'].is_set() or playback['offset'] >= len(playback['samples']):
                    self._playback = None
                    playback['done'].set()
        try:
            # DAC time is in the future relative to ADC time: this is the
            # input+output buffering delay WebRTC needs for its reference.
            delay = max(0, min(500, round((times.outputBufferDacTime - times.inputBufferAdcTime) * 1000)))
            self._frames.put_nowait((indata[:, 0].copy(), outdata[:, 0].copy(), delay))
        except (queue.Full, AttributeError, TypeError, ValueError) as exc:
            self._failure = ('전이중 음성 기준 신호를 유지하지 못했습니다. '
                             f'({type(exc).__name__}: {exc})')

    def read(self, size):
        if size != BLOCK:
            raise ValueError('음성 입력 프레임 크기가 맞지 않습니다.')
        if self._failure:
            raise RuntimeError(self._failure)
        try:
            near, far, delay = self._frames.get(timeout=1)
        except queue.Empty as exc:
            raise RuntimeError('전이중 음성 입력이 중단됐습니다.') from exc
        if self._failure:
            raise RuntimeError(self._failure)
        if self._initial_output_underflows and not self._reported_initial_underflow:
            self._reported_initial_underflow = True
            _LOG.warning('Duplex output underflow before speech; continuing with silent reference.')
        self._processor.stream_delay_ms = delay
        clean = np.asarray(self._processor.process(near, far), dtype=np.float32)
        if clean.shape != (BLOCK,) or not np.isfinite(clean).all():
            raise RuntimeError('에코 제거 결과를 확인하지 못했습니다.')
        return clean.reshape(-1, 1), False

    def input_buffer_snapshot(self):
        """Return the active input and the amount already queued at this instant."""
        return self, self._frames.qsize() * BLOCK

    def speak(self, text, stop_event, *, started=None):
        try:
            if stop_event.is_set():
                return
            samples = np.asarray(self._synthesize(text), dtype=np.float32).reshape(-1)
            if not len(samples) or not np.isfinite(samples).all():
                raise RuntimeError('생성된 음성 출력이 올바르지 않습니다.')
            if stop_event.is_set():
                return
            done = threading.Event()
            with self._lock:
                if self._stream is None or self._failure:
                    raise RuntimeError('전이중 음성 출력을 사용할 수 없습니다.')
                if stop_event.is_set():
                    return
                self._playback = {'samples': samples, 'offset': 0, 'done': done, 'stop': stop_event}
            if started is not None:
                started.set()
            while not done.wait(.05):
                if self._failure:
                    raise RuntimeError(self._failure)
            # Last callback buffers have not necessarily reached the device.
            # Keep output marked active for the known device output latency.
            latency = getattr(self._stream, 'latency', (0, 0))
            output_latency = latency[1] if isinstance(latency, (tuple, list)) else latency
            stop_event.wait(max(0, min(.5, float(output_latency))))
        finally:
            if started is not None:
                started.set()


class AdaptiveVoiceIO:
    """Fall back to SAPI + muted input when duplex setup cannot open."""

    def __init__(self, fallback_stream, fallback_speaker, *, duplex_factory=DuplexAudio):
        self._fallback_stream = fallback_stream
        self._fallback_speaker = fallback_speaker
        self._duplex_factory = duplex_factory
        self._active = None
        self.fallback_reason = None
        self.fallback_stage = None

    @property
    def echo_cancelled(self):
        return bool(self._active and getattr(self._active, 'echo_cancelled', False))

    @property
    def mode(self):
        return 'aec' if self.echo_cancelled else 'half_duplex'

    def __enter__(self):
        try:
            self._owner = self._duplex_factory()
            self._active = self._owner.__enter__()
        except Exception as exc:
            self.fallback_reason = str(exc) or type(exc).__name__
            self.fallback_stage = 'setup'
            _LOG.warning('Duplex audio setup failed; falling back to half-duplex.', exc_info=True)
            self._owner = self._fallback_stream()
            self._active = self._owner.__enter__()
        return self

    def __exit__(self, *args):
        if self._active is not None:
            return self._owner.__exit__(*args)

    def read(self, size):
        try:
            return self._active.read(size)
        except Exception as exc:
            if not self.echo_cancelled:
                raise
            # A broken reference or processing path must never continue as if
            # its input were echo-cancelled. Stop this stream before falling
            # back; the capture loop resets any partial utterance on the change.
            self.fallback_reason = str(exc) or type(exc).__name__
            self.fallback_stage = 'read'
            _LOG.warning('Duplex audio input failed; falling back to half-duplex.', exc_info=True)
            self._owner.__exit__(None, None, None)
            self._owner = self._fallback_stream()
            self._active = self._owner.__enter__()
            return self._active.read(size)

    def input_source(self):
        return self._active

    def input_buffer_snapshot(self):
        source = self._active
        if source is None:
            raise RuntimeError('음성 입력 상태를 확인할 수 없습니다.')
        snapshot = getattr(source, 'input_buffer_snapshot', None)
        return snapshot() if callable(snapshot) else (source, source.read_available)

    def speak(self, text, stop_event, *, started=None):
        speaker = self._active if self.echo_cancelled else self._fallback_speaker
        return speaker.speak(text, stop_event, started=started)
