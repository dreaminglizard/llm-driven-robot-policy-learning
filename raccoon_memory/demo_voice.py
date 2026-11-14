"""Offline-only ASR adapter for the simulation window."""
from contextlib import contextmanager
from importlib import import_module
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from raccoon_control.timing import span

from .voice import LocalVoice, NoSpeechDetected, normalize_mono_audio

_WINDOWS_DRIVE_REMOTE = 4

_MODEL_LOAD_LOCK = threading.Lock()


class DemoLocalVoice(LocalVoice):
    """Refuse missing local files and block Whisper's tokenizer fallback.

    The installed faster-whisper can fetch a tokenizer even with
    local_files_only=True when tokenizer.json is absent. Its tokenizer module
    is temporarily replaced only while this app loads/transcribes, and is
    restored afterward. Both GUI audio paths use this adapter.
    """

    @contextmanager
    def _local_tokenizer(self):
        if not all((self.model_dir / name).is_file() for name in ('model.bin', 'tokenizer.json')):
            raise ValueError('로컬 음성 모델과 토크나이저가 필요합니다. 다운로드하지 않았습니다.')
        module = import_module('faster_whisper.transcribe')
        original = module.tokenizers

        class LocalTokenizer:
            from_file = staticmethod(original.Tokenizer.from_file)
            from_buffer = staticmethod(original.Tokenizer.from_buffer)

            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise RuntimeError('음성 토크나이저의 온라인 다운로드는 허용되지 않습니다.')

        module.tokenizers = SimpleNamespace(Tokenizer=LocalTokenizer)
        try:
            yield
        finally:
            module.tokenizers = original

    def prepare(self):
        """Load the same model used by transcribe, without making audio."""
        if self._model is not None:
            return
        with _MODEL_LOAD_LOCK:
            if self._model is not None:
                return
            with self._local_tokenizer():
                from faster_whisper import WhisperModel
                with self._lock, span('model_load'):
                    self._model = WhisperModel(str(self.model_dir), device='cpu', compute_type='int8',
                                              cpu_threads=4, num_workers=1, local_files_only=True)

    def transcribe(self, audio, *, language='auto'):
        if self._model is not None:
            return super().transcribe(audio, language=language)
        with _MODEL_LOAD_LOCK:
            if self._model is not None:
                return super().transcribe(audio, language=language)
            with self._local_tokenizer():
                return super().transcribe(audio, language=language)


def find_cuda_runtime():
    """Find an existing CUDA 12 DLL directory; never install or download."""
    candidates = []
    override = os.environ.get('RACCOON_CUDA_DLL_DIR')
    if override:
        candidates.append(Path(override))
    cuda_home = os.environ.get('CUDA_PATH')
    if cuda_home:
        candidates.append(Path(cuda_home) / 'bin')
    candidates.append(Path(sys.prefix) / 'Lib/site-packages/torch/lib')
    appdata = os.environ.get('LOCALAPPDATA')
    if appdata:
        projects = Path(appdata) / 'pnpm/store/v11/projects'
        if projects.is_dir():
            candidates.extend(sorted(projects.glob('*/.runtime/python*/Lib/site-packages/torch/lib')))
    for path in candidates:
        if (path.is_absolute() and (path / 'cublas64_12.dll').is_file()
                and (path / 'cublasLt64_12.dll').is_file()):
            return path.resolve()
    return None


def _cuda_worker(channel, model_dir, runtime_dir):
    """Keep a warm GPU model in a child so native failures cannot kill the UI."""
    dll_handle = None
    try:
        os.environ['PATH'] = str(runtime_dir) + os.pathsep + os.environ.get('PATH', '')
        dll_handle = os.add_dll_directory(str(runtime_dir))
        import faster_whisper
        native_whisper = faster_whisper.WhisperModel

        def cuda_whisper(model_path, **kwargs):
            expected = {'device': 'cpu', 'compute_type': 'int8', 'cpu_threads': 4,
                        'num_workers': 1, 'local_files_only': True}
            if kwargs != expected:
                raise RuntimeError('음성 모델의 로컬 설정이 예상과 다릅니다.')
            return native_whisper(model_path, device='cuda', compute_type='float16',
                                  num_workers=1, local_files_only=True)

        faster_whisper.WhisperModel = cuda_whisper
        voice = DemoLocalVoice(model_dir)
        while True:
            message = channel.recv()
            if message is None:
                break
            try:
                if message == 'prepare':
                    voice.prepare()
                    result = None
                else:
                    audio, language = message
                    # LocalVoice collects the entire segment generator before it
                    # returns. No partial transcript is ever sent to the app.
                    result = voice.transcribe(audio, language=language)
                channel.send(('ok', result))
            except NoSpeechDetected as exc:
                # A completed empty decode is a per-utterance result. Keep the
                # loaded GPU worker available for the next utterance.
                channel.send(('no_speech', str(exc)))
            except Exception as exc:
                channel.send(('error', f'{type(exc).__name__}: {exc}'))
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            channel.send(('error', f'{type(exc).__name__}: {exc}'))
        except (EOFError, OSError):
            pass
    finally:
        if dll_handle is not None:
            dll_handle.close()
        channel.close()


class _GPUTranscriber:
    def __init__(self, model_dir, runtime_dir):
        ctx = multiprocessing.get_context('spawn')
        self.channel, worker_channel = ctx.Pipe()
        self.process = ctx.Process(target=_cuda_worker, args=(worker_channel, str(model_dir), str(runtime_dir)),
                                   daemon=True, name='demo-offline-gpu-asr')
        try:
            self.process.start()
        except BaseException:
            self.channel.close()
            worker_channel.close()
            raise
        worker_channel.close()

    def _exchange(self, message, cancel):
        if cancel.is_set():
            raise InterruptedError('음성 인식을 취소했습니다.')
        self.channel.send(message)
        deadline = time.monotonic() + 120
        while True:
            if cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            if self.channel.poll(.1):
                status, result = self.channel.recv()
                if status == 'no_speech':
                    raise NoSpeechDetected(str(result))
                if status != 'ok':
                    raise RuntimeError(str(result))
                return result
            if not self.process.is_alive():
                raise RuntimeError('GPU 음성 인식 프로세스가 종료됐습니다.')
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU 음성 인식 대기 시간이 초과됐습니다.')

    def prepare(self, cancel):
        self._exchange('prepare', cancel)

    def transcribe(self, audio, language, cancel):
        return self._exchange((audio, language), cancel)

    def close(self):
        try:
            if self.process.is_alive():
                try:
                    self.channel.send(None)
                except (OSError, BrokenPipeError, EOFError):
                    pass
                self.process.join(1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(5)
        finally:
            self.channel.close()


class DemoAutoVoice(DemoLocalVoice):
    """Honor CPU-only selection, or prefer GPU and retry on CPU when automatic."""
    def __init__(self, model_dir, *, gpu_factory=_GPUTranscriber, runtime_finder=find_cuda_runtime, device='auto'):
        super().__init__(model_dir)
        if device not in ('auto', 'cpu'):
            raise ValueError('Whisper 실행 장치는 auto 또는 cpu여야 합니다.')
        self._device = device
        self._gpu_factory, self._runtime_finder = gpu_factory, runtime_finder
        self._gpu, self._runtime = None, None
        self._fallback_reason = None
        self._cancel = threading.Event()
        self._serial = threading.Lock()
        self._prepared = False

    @staticmethod
    def _prepare(audio, language):
        def _require_local_drive(candidate):
            if os.name == 'nt':
                import ctypes
                if (str(candidate).startswith('\\\\')
                        or ctypes.windll.kernel32.GetDriveTypeW(candidate.anchor) == _WINDOWS_DRIVE_REMOTE):
                    raise ValueError('로컬 드라이브에 있는 음성 파일을 선택해 주세요.')

        def _read_local_only(path):
            local_path = Path(path).absolute()
            _require_local_drive(local_path)
            resolved = local_path.resolve()
            _require_local_drive(resolved)
            import soundfile as sf
            return sf.read(str(resolved), dtype='float32', always_2d=True)

        return normalize_mono_audio(audio, language=language, read_audio=_read_local_only)

    def _gpu_backend(self):
        if self._gpu is None:
            self._runtime = self._runtime_finder()
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            if self._runtime is None:
                raise RuntimeError('기존 CUDA 12 라이브러리를 찾지 못했습니다.')
            self._gpu = self._gpu_factory(self.model_dir, self._runtime)
        if self._cancel.is_set():
            raise InterruptedError('음성 인식을 취소했습니다.')
        return self._gpu

    def prepare(self):
        """Prepare the conversation's own backend, serialized with real audio."""
        with self._serial:
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            if self._prepared:
                return
            if not all((self.model_dir / name).is_file() for name in ('model.bin', 'tokenizer.json')):
                raise ValueError('로컬 음성 모델과 토크나이저가 필요합니다. 다운로드하지 않았습니다.')
            if self._device != 'cpu' and self._fallback_reason is None:
                try:
                    self._gpu_backend().prepare(self._cancel)
                    if self._cancel.is_set():
                        raise InterruptedError('음성 인식을 취소했습니다.')
                    self._prepared = True
                    return
                except InterruptedError:
                    self._close_gpu()
                    raise
                except Exception as exc:
                    if self._cancel.is_set():
                        self._close_gpu()
                        raise InterruptedError('음성 인식을 취소했습니다.') from exc
                    self._fallback_reason = str(exc) or type(exc).__name__
                    self._close_gpu()
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            super().prepare()
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            self._prepared = True

    def transcribe(self, audio, *, language='auto'):
        audio = self._prepare(audio, language)
        with self._serial:
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            if not all((self.model_dir / name).is_file() for name in ('model.bin', 'tokenizer.json')):
                raise ValueError('로컬 음성 모델과 토크나이저가 필요합니다. 다운로드하지 않았습니다.')
            if self._device != 'cpu' and self._fallback_reason is None:
                try:
                    result = self._gpu_backend().transcribe(audio, language, self._cancel)
                    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
                        raise RuntimeError('GPU 음성 인식 결과가 올바르지 않습니다.')
                    self._prepared = True
                    result.update(device='GPU', engine='faster-whisper-cuda-float16',
                                  cuda_runtime=str(self._runtime), fallback_reason=None)
                    return result
                except NoSpeechDetected as exc:
                    if self._cancel.is_set():
                        self._close_gpu()
                        raise InterruptedError('음성 인식을 취소했습니다.') from exc
                    self._prepared = True
                    raise
                except InterruptedError:
                    self._close_gpu()
                    raise
                except Exception as exc:
                    if self._cancel.is_set():
                        self._close_gpu()
                        raise InterruptedError('음성 인식을 취소했습니다.') from exc
                    self._fallback_reason = str(exc) or type(exc).__name__
                    self._close_gpu()
            result = super().transcribe(audio, language=language)
            if self._cancel.is_set():
                raise InterruptedError('음성 인식을 취소했습니다.')
            self._prepared = True
            result.update(device='CPU', cuda_runtime=None, fallback_reason=self._fallback_reason)
            return result

    def cancel_pending(self):
        self._cancel.set()

    def _close_gpu(self):
        worker, self._gpu = self._gpu, None
        if worker is not None:
            worker.close()

    def close(self):
        self._cancel.set()
        with self._serial:
            self._close_gpu()
            self._model = None
