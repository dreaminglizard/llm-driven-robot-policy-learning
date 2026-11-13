"""Local Korean/English ASR. Recording is explicit, never a listener."""
from pathlib import Path
import threading
from raccoon_control.timing import measured, span


class NoSpeechDetected(ValueError):
    """A completed decode returned no transcript, not an ASR backend failure."""


class LocalVoice:
    def __init__(self, model_dir="artifacts/whisper-medium"):
        self.model_dir = Path(model_dir).resolve()
        self._model = None
        self._lock = threading.Lock()

    @measured('voice_transcribe')
    def transcribe(self, audio, *, language="auto"):
        if language not in ("auto", "ko", "en"):
            raise ValueError("음성 언어는 auto, ko, en 중 하나여야 합니다.")
        import math
        import numpy as np
        if isinstance(audio, (str, Path)):
            import soundfile as sf
            from scipy.signal import resample_poly
            with span('audio_file_read'):
                audio, rate = sf.read(str(audio),dtype="float32",always_2d=True)
            audio = audio.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(rate,16000)
                audio = resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
        if not isinstance(audio,np.ndarray) or audio.ndim != 1 or len(audio)==0 or not np.isfinite(audio).all():
            raise ValueError("유효한 모노 음성 데이터가 필요합니다.")
        if len(audio)>16000*60:
            raise ValueError("명령 음성은 60초 이하로 입력해 주세요.")
        if float(np.sqrt(np.mean(audio.astype(np.float64)**2))) < 0.002:
            raise ValueError("음성이 없거나 너무 작습니다. 텍스트 입력도 사용할 수 있습니다.")
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
        if not 1 <= seconds <= 12:
            raise ValueError("녹음 시간은 1~12초 범위여야 합니다.")
        cancel = cancel or threading.Event()
        samples = []
        # No playback, operating-system speech or global microphone listener.
        with span('microphone_capture'), sd.InputStream(samplerate=16000, channels=1, dtype="float32", device=device,
                            blocksize=1600) as stream:
            for _ in range(round(seconds*10)):
                if cancel.is_set():
                    raise InterruptedError("녹음을 취소했습니다.")
                data, overflow = stream.read(1600)
                if overflow:
                    raise ValueError("마이크 입력이 밀렸습니다. 다시 녹음해 주세요.")
                samples.append(data[:,0].copy())
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        audio = np.concatenate(samples)
        if float(np.sqrt(np.mean(audio*audio))) < 0.002:
            raise ValueError("마이크 입력이 너무 작습니다. 장치와 볼륨을 확인해 주세요.")
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        result = self.transcribe(audio)
        if cancel.is_set():
            raise InterruptedError("녹음을 취소했습니다.")
        return result
