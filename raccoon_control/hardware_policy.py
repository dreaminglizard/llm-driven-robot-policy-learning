"""Project hardware time restriction, checked before importing the robot SDK."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .errors import ControlError

POLICY_FILE = Path(__file__).resolve().parents[1] / "configs" / "hardware_policy.json"


def check_hardware_time(*, now=None, policy_path=None):
    path = Path(policy_path or POLICY_FILE)
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
        not_before = datetime.fromisoformat(policy["not_before"])
        if not_before.tzinfo is None or type(policy["enabled"]) is not bool:
            raise ValueError("Invalid time restriction")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ControlError("HARDWARE_POLICY_INVALID", "A valid project hardware policy is required") from exc
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ControlError("HARDWARE_POLICY_INVALID", "The current time must include a timezone")
    if not policy["enabled"] or current < not_before:
        raise ControlError("HARDWARE_TIME_LOCKED", "사용자 지시로 실제 로봇 연결을 제한합니다.",
                           not_before=not_before.isoformat(), enabled=policy["enabled"])
    return {"allowed_by_time": True, "not_before": not_before.isoformat(), "automatic_start": False}
