"""Wrap a local JSONL file-ASR worker in the app's voice interface.

The worker receives only temporary 16 kHz mono PCM WAV files.  It is loaded
once and all requests are serialized; neither a microphone nor a model is
opened by constructing this object.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import wave

from .demo_voice import DemoAutoVoice
from .model_lifecycle import WindowsModelJob
from .voice import LocalVoice, NoSpeechDetected


ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT = 120


def _local_path(path):
    """Refuse Windows network paths both before and after resolving links."""
    path = Path(path).absolute()
    if os.name == 'nt':
        import ctypes
        for candidate in (path, path.resolve()):
            if str(candidate).startswith('\\\\') or ctypes.windll.kernel32.GetDriveTypeW(candidate.anchor) == 4:
                raise ValueError('로컬 드라이브의 파일을 사용해 주세요.')
    return path.resolve()


class _Worker:
    def __init__(self, command, model_path):
        env = dict(os.environ)
        env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        self.job = WindowsModelJob(model_path) if os.name == 'nt' else None
        self.process = None
        try:
            spawn = self.job.popen if self.job is not None else subprocess.Popen
            self.process = spawn(
                command, cwd=str(ROOT), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                start_new_session=(os.name != 'nt'))
        except BaseException:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            if self.job:
                self.job.close()
            raise
        self.lines, self.errors = queue.Queue(), deque(maxlen=40)

        def stdout_reader():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        def stderr_reader():
            for line in self.process.stderr:
                self.errors.append(line[-2048:])

        self.readers = [threading.Thread(target=stdout_reader, daemon=True),
                        threading.Thread(target=stderr_reader, daemon=True)]
        for reader in self.readers:
            reader.start()

    def exchange(self, request, *, cancelled, timeout=_TIMEOUT):
        if cancelled():
            raise InterruptedError('음성 인식을 취소했습니다.')
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError('음성 인식 프로세스에 요청하지 못했습니다.') from exc
        deadline = time.monotonic() + timeout
        check_members = 0.0
        while True:
            if cancelled():
                raise InterruptedError('음성 인식을 취소했습니다.')
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError('음성 인식 대기 시간이 초과됐습니다.')
            if self.job is not None and request['type'] == 'load' and time.monotonic() >= check_members:
                self.job.capture()
                check_members = time.monotonic() + .5
            try:
                line = self.lines.get(timeout=min(left, .1))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError('음성 인식 프로세스가 종료됐습니다.')
                continue
            if line is None:
                raise RuntimeError('음성 인식 프로세스가 응답 없이 종료됐습니다.')
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError('음성 인식 프로세스의 응답 형식이 올바르지 않습니다.') from exc
            if not isinstance(message, dict) or message.get('id') != request['id']:
                raise RuntimeError('음성 인식 응답이 요청과 일치하지 않습니다.')
            if message.get('type') == 'partial':
                continue  # Only the completed transcript is returned to the app.
            if message.get('type') == 'error':
                raise RuntimeError(f"음성 인식 오류: {message.get('error')}")
            if message.get('type') in ('ready', 'final', 'closed'):
                if message['type'] == 'ready' and self.job is not None:
                    self.job.capture()
                return message
            raise RuntimeError('음성 인식 응답 종류가 올바르지 않습니다.')

    def stop(self):
        # The job/session contains only processes launched by this worker.
        if self.job is not None:
            self.job.close()
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if self.job is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()
            self.process.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


class WorkerVoice(LocalVoice):
    """Local persistent JSONL ASR with the same interface as DemoAutoVoice."""
    def __init__(self, model_dir, worker_command, *, device='cuda', precision='bfloat16'):
        super().__init__(model_dir)
        if isinstance(worker_command, (str, os.PathLike)):
            worker_command = [worker_command]
        self.worker_command = [str(item) for item in worker_command]
        if not self.worker_command or not all(self.worker_command):
            raise ValueError('로컬 음성 인식 실행기가 필요합니다.')
        self.device, self.precision = str(device), str(precision)
        self._serial, self._cancel = threading.Lock(), threading.Event()
        self._record_context = threading.local()
        self._worker, self._load_metadata = None, None

    def _cancelled(self):
        recording_cancel = getattr(self._record_context, 'cancel', None)
        return self._cancel.is_set() or (recording_cancel is not None and recording_cancel.is_set())

    def _check_cancel(self):
        if self._cancelled():
            raise InterruptedError('음성 인식을 취소했습니다.')

    def _stop_worker(self, *, graceful):
        worker = self._worker
        self._load_metadata = None
        if worker is None:
            return
        try:
            if graceful and worker.process.poll() is None:
                if worker.job is not None:
                    worker.job.capture()
                try:
                    worker.exchange({'type': 'close', 'id': uuid.uuid4().hex},
                                    cancelled=lambda: False, timeout=10)
                except Exception:
                    pass
        finally:
            worker.stop()
            self._worker = None

    def _ensure_ready(self):
        self._check_cancel()
        if self._load_metadata is not None:
            return
        if self._worker is not None:
            self._stop_worker(graceful=False)
        model_path = _local_path(self.model_dir)
        if not model_path.exists():
            raise ValueError('로컬 음성 모델이 없습니다. 음성 모델 경로를 확인해 주세요.')
        self._worker = _Worker(self.worker_command, model_path)
        try:
            request = {'type': 'load', 'id': uuid.uuid4().hex, 'model_path': str(model_path),
                       'device': self.device, 'precision': self.precision, 'offline': True}
            response = self._worker.exchange(request, cancelled=self._cancelled)
            if response.get('type') != 'ready':
                raise RuntimeError('음성 인식 모델을 준비하지 못했습니다.')
            self._check_cancel()
            metadata = response.get('metadata') or {}
            if not isinstance(metadata, dict):
                raise RuntimeError('음성 모델 준비 응답이 올바르지 않습니다.')
            self._load_metadata = metadata
        except BaseException:
            self._stop_worker(graceful=False)
            raise

    def prepare(self):
        with self._serial:
            self._ensure_ready()

    def transcribe(self, audio, *, language='auto'):
        self._check_cancel()
        audio = DemoAutoVoice._prepare(audio, language)
        import numpy as np

        with self._serial:
            self._ensure_ready()
            self._check_cancel()
            temp_root = _local_path(tempfile.gettempdir())
            with tempfile.TemporaryDirectory(prefix='raccoon-asr-', dir=str(temp_root)) as directory:
                path = Path(directory) / 'input.wav'
                pcm = np.clip(np.rint(np.clip(audio, -1.0, 1.0) * 32768.0),
                              -32768, 32767).astype('<i2')
                with wave.open(str(path), 'wb') as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(pcm.tobytes())
                try:
                    request = {'type': 'transcribe', 'id': uuid.uuid4().hex, 'audio_path': str(path),
                               'language': language, 'feed_mode': 'file_eager'}
                    response = self._worker.exchange(request, cancelled=self._cancelled)
                    self._check_cancel()
                    if response.get('type') != 'final':
                        raise RuntimeError('음성 인식 결과를 받지 못했습니다.')
                    status, text = response.get('status'), response.get('text')
                    if status == 'no_speech' or (status == 'ok' and isinstance(text, str) and not text.strip()):
                        raise NoSpeechDetected('음성을 인식하지 못했습니다. 텍스트 입력도 사용할 수 있습니다.')
                    if status != 'ok' or not isinstance(text, str):
                        raise RuntimeError(f"음성 인식 오류: {response.get('error') or status}")
                    metadata = response.get('metadata') or {}
                    if not isinstance(metadata, dict):
                        raise RuntimeError('음성 인식 결과의 형식이 올바르지 않습니다.')
                    return {'text': text.strip(), 'language': response.get('detected_language'),
                            'requested_language': language, 'language_probability': metadata.get('language_probability'),
                            'offline': True, 'engine': 'local-jsonl-worker', 'model': self.model_dir.name,
                            'device': self._load_metadata.get('device_actual', self._load_metadata.get(
                                'actual_device', self._load_metadata.get('device', self.device))),
                            'worker_metadata': metadata, 'worker_load_metadata': dict(self._load_metadata)}
                except NoSpeechDetected:
                    raise
                except BaseException:
                    self._stop_worker(graceful=False)
                    raise

    def record(self, *, seconds=7, cancel=None, device=None):
        self._check_cancel()
        if cancel is not None and cancel.is_set():
            raise InterruptedError('녹음을 취소했습니다.')
        self._record_context.cancel = cancel

        class CombinedCancel:
            def is_set(inner_self):
                return self._cancelled()

        try:
            return super().record(seconds=seconds, cancel=CombinedCancel(), device=device)
        finally:
            del self._record_context.cancel

    def cancel_pending(self):
        self._cancel.set()
        if self._serial.acquire(blocking=False):
            try:
                self._stop_worker(graceful=False)
            finally:
                self._serial.release()

    def close(self):
        self._cancel.set()
        with self._serial:
            self._stop_worker(graceful=True)
