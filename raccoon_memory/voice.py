"""Local Korean/English ASR. Recording is explicit, never a listener."""
from pathlib import Path
import threading
from raccoon_control.timing import measured, span

AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_SIZE = 1600  # 100 ms per block
AUDIO_MAX_DURATION_SECONDS = 60
AUDIO_MIN_RMS = 0.002
SUPPORTED_VOICE_LANGUAGES = ("auto", "ko", "en")
RECORD_MIN_SECONDS = 1
RECORD_MAX_SECONDS = 12


def normalize_mono_audio(audio, *, language="auto", read_audio=None):
    """Validate and resample a mono audio array or file path to target ASR format."""
    if language not in SUPPORTED_VOICE_LANGUAGES:
        raise ValueError("음성 언어는 auto, ko, en 중 하나여야 합니다.")
    import math
    import numpy as np
    if isinstance(audio, (str, Path)):
        if read_audio is None:
            import soundfile as sf
            audio, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        else:
            audio, rate = read_audio(audio)
        audio = audio.mean(axis=1)
        if rate != AUDIO_SAMPLE_RATE:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, AUDIO_SAMPLE_RATE)
            audio = resample_poly(audio, AUDIO_SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("유효한 모노 음성 데이터가 필요합니다.")
    if len(audio) > AUDIO_SAMPLE_RATE * AUDIO_MAX_DURATION_SECONDS:
        raise ValueError(f"명령 음성은 {AUDIO_MAX_DURATION_SECONDS}초 이하로 입력해 주세요.")
    if float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) < AUDIO_MIN_RMS:
        raise ValueError("음성이 없거나 너무 작습니다. 텍스트 입력도 사용할 수 있습니다.")
    return audio


class NoSpeechDetected(ValueError):
    """A completed decode returned no transcript, not an ASR backend failure."""


class LocalVoice:
    def __init__(self, model_dir="artifacts/whisper-medium"):
        self.model_dir = Path(model_dir).resolve()
        self._model = None
        self._lock = threading.Lock()

    @measured('voice_transcribe')
    def transcribe(self, audio, *, language="auto"):
        def _read_audio(path):
            import soundfile as sf
            with span('audio_file_read'):
                return sf.read(str(path), dtype="float32", always_2d=True)

        audio = normalize_mono_audio(audio, language=language, read_audio=_read_audio)
        from faster_whisper import WhisperModel
        with self._lock:
            if self._model is None:
                if not (self.model_dir/"model.bin").is_file():
                    raise ValueError("로컬 음성 모델이 없습니다. 음성 모델 폴더를 확인해 주세요.")
                with span('model_load'):
                    self._model = WhisperModel(str(self.model_dir), device="cpu", compute_type="int8",
                                              cpu_threads=4, num_workers=1, local_files_only=True)
            with span('decode_and_collect_segments'):
                # Whisper treats this context as preceding speech, not as
                # instructions. A procedural prompt was itself transcribed
                # on noisy input; decode only the supplied audio instead.
                segments, info = self._model.transcribe(audio, language=None if language == "auto" else language, beam_size=5,
                    condition_on_previous_text=False, vad_filter=True)
                text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            raise NoSpeechDetected("음성을 인식하지 못했습니다. 텍스트 입력도 사용할 수 있습니다.")
        return {"text":text, "language":info.language, "requested_language": language,
                "language_probability": getattr(info, "language_probability", None), "offline":True,
                "engine":"faster-whisper-cpu-int8", "model":self.model_dir.name}

    @measured('voice_record')
    def record(self, *, seconds=7, cancel=None, device=None):
        import numpy as np
        import sounddevice as sd
        if not RECORD_MIN_SECONDS <= seconds <= RECORD_MAX_SECONDS:
            raise ValueError(f"녹음 시간은 {RECORD_MIN_SECONDS}~{RECORD_MAX_SECONDS}초 범위여야 합니다.")
        cancel = cancel or threading.Event()
        samples = []
        blocks_per_second = AUDIO_SAMPLE_RATE // AUDIO_BLOCK_SIZE
        # No playback, operating-system speech or global microphone listener.
        with span('microphone_capture'), sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32",
                            device=device, blocksize=AUDIO_BLOCK_SIZE) as stream:
            for _ in range(round(seconds * blocks_per_second)):
                if cancel.is_set():
                    raise InterruptedError("녹음을 취소했습니다.")
                data, overflow = stream.read(AUDIO_BLOCK_SIZE)
                if overflow:
                    raise ValueError("마이크 입력이 밀렸습니다. 다시 녹음해 주세요.")
                samples.append(data[:,0].copy())
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        audio = np.concatenate(samples)
        if float(np.sqrt(np.mean(audio*audio))) < AUDIO_MIN_RMS:
            raise ValueError("마이크 입력이 너무 작습니다. 장치와 볼륨을 확인해 주세요.")
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        result = self.transcribe(audio)
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        return result
