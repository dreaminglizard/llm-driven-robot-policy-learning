"""Local learning artifact paths; existing runs and evidence are never replaced."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import uuid


def unique_name(label):
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+label+'-'+uuid.uuid4().hex[:8]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_report_path(root,label,output=None):
    path=Path(output) if output else Path(root)/'evaluations'/(unique_name(label)+'.json')
    if path.exists():
        raise FileExistsError(f'기존 결과를 보존합니다. 새 출력 경로를 사용해 주세요: {path}')
    return path


def write_new_json(path,value):
    path=Path(path)
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(encoded)


def reserve_training_directory(output=None):
    path=Path(output) if output else Path('artifacts/training_runs')/unique_name('policy')
    path.mkdir(parents=True,exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f'기존 학습 결과를 보존합니다. 빈 출력 폴더를 지정해 주세요: {path}')
    # Exclusive creation also rejects two simultaneous attempts to use one run.
    write_new_json(path/'run.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'purpose':'isolated_learning_run','deployed_automatically':False})
    return path
