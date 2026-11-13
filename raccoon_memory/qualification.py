"""Bind expanded physical app scope to recorded route results, never mock results."""
import hashlib
import json
from pathlib import Path

from raccoon_control.errors import ControlError


def calibration_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_cases(slots,max_height):
    return [(source,destination,pick,place) for source in slots for destination in slots
            if source!=destination for pick in range(1,max_height+1) for place in range(1,max_height+1)]


def check_expansion(profile,config):
    # Calibration-mode trials remain unverified; their existing workspace,
    # path and controller checks still apply independently of route evidence.
    if config.backend!='sdk' or config.mode=='calibration' or set(profile.enabled_slots)<={1,3}:
        return
    path=profile.route_evidence
    try:
        data=json.loads(path.read_text(encoding='utf-8')) if path else None
    except (OSError,ValueError) as exc:
        raise ControlError('ROUTE_EVIDENCE_REQUIRED','확장할 실물 경로의 검증 기록을 읽을 수 없습니다.') from exc
    if (not isinstance(data,dict) or data.get('backend')!='sdk'
            or data.get('configuration_sha256')!=config.fingerprint()
            or data.get('calibration_sha256')!=calibration_hash(profile.calibration)):
        raise ControlError('ROUTE_EVIDENCE_REQUIRED','같은 제어기·교정 설정으로 확인한 실물 경로 기록이 필요합니다.')
    rows=data.get('cases')
    if not isinstance(rows,list):
        raise ControlError('ROUTE_EVIDENCE_REQUIRED','실물 경로별 확인 결과가 없습니다.')
    latest={}
    for row in rows:
        if not isinstance(row,dict):continue
        key=tuple(row.get(k) for k in ('source','destination','pick_level','place_level'))
        if not all(type(x) is int for x in key):continue
        latest[key]=(row.get('physical_success') is True and row.get('verified_by') in ('human','video')
                     and isinstance(row.get('evidence'),str) and bool(row['evidence'].strip()))
    missing=[key for key in required_cases(profile.enabled_slots,profile.max_height) if not latest.get(key,False)]
    if missing:
        raise ControlError('ROUTE_UNVERIFIED','확장 범위의 집기·놓기 또는 연결 경로가 아직 검증되지 않았습니다.',missing_cases=missing)
