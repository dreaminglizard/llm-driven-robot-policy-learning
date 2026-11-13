import json
import math

from .errors import ControlError


def number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError("INVALID_NUMBER", f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ControlError("INVALID_NUMBER", f"{name} is too large") from exc
    if not math.isfinite(value):
        raise ControlError("INVALID_NUMBER", f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ControlError("OUT_OF_RANGE", f"{name} is outside the permitted range")
    return value


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ControlError("INVALID_VECTOR", f"{name} must contain {length} numbers")
    return tuple(number(v, f"{name}[{i}]") for i, v in enumerate(value))


def identifier(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ControlError("INVALID_ID", f"{name} must be a nonempty string of at most 128 characters")
    return value


def keys(data, *, allowed, required=()):
    if not isinstance(data, dict):
        raise ControlError("INVALID_REQUEST", "Expected a JSON object")
    extra, missing = set(data) - set(allowed), set(required) - set(data)
    if extra or missing:
        raise ControlError("INVALID_FIELDS", "Unexpected or missing fields", extra=sorted(extra), missing=sorted(missing))


def json_loads(text):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    try:
        return json.loads(text, parse_constant=reject)
    except (ValueError, TypeError) as exc:
        raise ControlError("INVALID_JSON", "Invalid JSON") from exc
