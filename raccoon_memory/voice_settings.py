"""One local ASR selection shared by the product's audio input paths."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / 'configs/asr.runtime.json'


def local_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def voice_profile():
    if not PROFILE_PATH.is_file():
        return {'name': 'Whisper large-v3-turbo', 'backend': 'whisper',
                'model_path': 'artifacts/whisper-large-v3-turbo', 'device': 'cpu', 'precision': 'int8'}
    profile = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    if profile.get('backend') not in ('whisper', 'jsonl'):
        raise ValueError('선택한 로컬 음성 실행기를 확인해 주세요.')
    if not isinstance(profile.get('model_path'), str) or not profile['model_path']:
        raise ValueError('로컬 음성 모델 경로가 필요합니다.')
    return profile


def voice_model_path():
    return local_path(voice_profile()['model_path'])


def create_voice_recognizer():
    profile = voice_profile()
    model = local_path(profile['model_path'])
    if profile['backend'] == 'whisper':
        from .demo_voice import DemoAutoVoice
        device = profile.get('device', 'auto')
        precision = profile.get('precision')
        if device not in ('auto', 'cpu'):
            raise ValueError('Whisper 실행 장치는 auto 또는 cpu여야 합니다.')
        if precision is not None and (device != 'cpu' or precision != 'int8'):
            raise ValueError('CPU 전용 Whisper는 int8 정밀도만 지원합니다.')
        if device == 'cpu':
            return DemoAutoVoice(model, device='cpu')
        return DemoAutoVoice(model)
    from .worker_voice import WorkerVoice
    command = profile.get('worker_command')
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError('로컬 음성 실행 경로가 필요합니다.')
    # Current profiles contain only a local interpreter, flags, and a script.
    resolved = [x if x.startswith('-') else str(local_path(x)) for x in command]
    return WorkerVoice(model, resolved, device=profile.get('device', 'cuda'),
                       precision=profile.get('precision', 'bfloat16'))
