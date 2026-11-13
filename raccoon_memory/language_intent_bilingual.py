"""Bilingual intent proposals using the unchanged Korean schema and transport.

This is a separate experimental path: the previously frozen Korean interpreter,
prompt and schema are not modified. Pure Korean text delegates to its exact
existing context validator; English and mixed text use the checks below. These
checks only reject unsupported proposals; they never manufacture task fields or
replace a model response with a rule-generated action.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import socket
import time
import unicodedata

from . import language_intent as common
from .task_catalog import scene as validate_scene

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "data/language_intent/bilingual_system_prompt.v1.txt"
PROMPT_SHA256 = "c01edec2cbfbc0fa5065402290c5505f081511d45d2ea83b6bc9d19ded796f48"


def load_prompt():
    raw = PROMPT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPT_SHA256:
        raise ValueError("bilingual prompt changed; create a new explicitly versioned comparison")
    return raw.decode("utf-8")


def build_messages(text):
    return [{"role": "system", "content": load_prompt()}, {"role": "user", "content": text}]


_EN_VALUES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "first": 1, "second": 2,
              "third": 3, "fourth": 4}
_FLAGS = re.IGNORECASE | re.ASCII
_NUM = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|fourth)"
_SLOT = r"(?:slot|spot|position|place|자리)\s*(?:number\s*|no\.?\s*|#\s*)?" + _NUM + r"\s*(?:번)?"
_EN_SLOT = re.compile(r"\b" + _SLOT, _FLAGS)
_EN_SOURCE_FROM = re.compile(r"\bfrom\s+(?:the\s+)?" + _SLOT, _FLAGS)
_EN_SOURCE_TO = re.compile(r"\b" + _SLOT + r"\s+(?:to|onto)\b", _FLAGS)
_EN_SOURCE_SUFFIX = re.compile(r"\b" + _SLOT + r"\s*(?:에서|의|에\s*있는)", _FLAGS)
_EN_DEST_PREFIX = re.compile(r"\b(?:to|onto|into|in|on)\s+(?:the\s+)?" + _SLOT, _FLAGS)
_EN_DEST_SUFFIX = re.compile(r"\b" + _SLOT + r"\s*(?:으로|로|에|위)", _FLAGS)
_REVERSE_SOURCE = re.compile(_NUM + r"\s*번\s*(?:slot|spot|position)\s*(?:에서|의)", _FLAGS)
_REVERSE_DEST = re.compile(_NUM + r"\s*번\s*(?:slot|spot|position)\s*(?:으로|로|에|위)", _FLAGS)
_OBJECT_BEFORE = re.compile(r"\b(?:cylinder|object)\s*(?:number\s*|no\.?\s*|#\s*)?" + _NUM + r"\s*(?:번)?", _FLAGS)
_OBJECT_AFTER = re.compile(_NUM + r"\s*번\s*(?:cylinder|object)\b", _FLAGS)
_KOREAN_OBJECT_EN_NUM = re.compile(r"(?:원통|물체)\s*" + _NUM + r"\s*번", _FLAGS)
_COUNT = re.compile(r"\b" + _NUM + r"\s+(?:cylinders?|objects?|pieces?)\b", _FLAGS)
_COUNT_ONE = re.compile(r"\b(?:move|stack|transfer)\s+(one|two|three|four|\d+)\s+(?:from|to)\b", _FLAGS)
_HEIGHT_BEFORE = re.compile(r"\b(?:height|level|floor|layer)\s*" + _NUM, _FLAGS)
_HEIGHT_AFTER = re.compile(r"\b" + _NUM + r"\s*(?:st|nd|rd|th)?\s+(?:level|floor|layer)\b", _FLAGS)
_EN_RANGE = re.compile(r"\b(?:slot|spot|position|place|cylinder|object|height|level|floor|layer)\s*(?:number\s*|no\.?\s*|#\s*)?" + _NUM, _FLAGS)
_EN_ACTION = re.compile(r"\b(?:move|transfer|put|place|bring|carry|stack|swap|exchange|switch|remember|memorize|restore|register|stop|halt)\b", _FLAGS)
_EN_UNSAFE = (
    ("negation", re.compile(r"\b(?:do\s+not|don['’]?t|never|must\s+not|mustn['’]?t|should\s+not|shouldn['’]?t|cannot|can['’]?t|without)\b", _FLAGS)),
    ("compound", re.compile(r"\b(?:and\s+then|then|afterwards|after\s+that)\b|\band\s+(?:move|stack|swap|remember|restore|register|stop)\b", _FLAGS)),
    ("ambiguous_reference", re.compile(r"\b(?:red|blue|green|yellow|white|black|purple|orange|pink|brown|gray|grey|left|right|nearest|closest|farthest|behind|there|it|them|those)\b|\b(?:this|that)\s+(?:cylinder|object|slot|one)\b", _FLAGS)),
    ("prompt_injection", re.compile(r"\bignore\b.{0,60}\b(?:rules|instructions|previous|system)\b|\b(?:system\s+prompt|developer\s+instruction)\b", _FLAGS)),
    ("conditional", re.compile(r"\b(?:if|unless|whenever|after\s+the\s+task)\b", _FLAGS)),
    ("direct_control", re.compile(r"\b(?:joint|joints|servo|motor|trajectory|path|angle|degrees?|torque|velocity|speed|quickly|slowly)\b|\b\d+\s*(?:mm|cm|seconds?\s+later)\b", _FLAGS)),
)
_EN_INTENTS = {
    "move": re.compile(r"\b(?:move|transfer|put|place|bring|carry|take|send|shift)\b", _FLAGS),
    "stack": re.compile(r"\b(?:stack|pile)\b", _FLAGS),
    "swap": re.compile(r"\b(?:swap|exchange|switch)\b", _FLAGS),
    "remember": re.compile(r"\b(?:remember|memorize|save|store)\b", _FLAGS),
    "restore": re.compile(r"\b(?:restore|recall|recover|reconstruct)\b", _FLAGS),
    "initial": re.compile(r"\b(?:initial|starting|original|default)\b", _FLAGS),
    "status": re.compile(r"\b(?:status|layout|state|where|scene)\b", _FLAGS),
    "stop": re.compile(r"\b(?:stop|halt|cancel)\b", _FLAGS),
    "register": re.compile(r"\b(?:register|registration)\b", _FLAGS),
}


def _value(token):
    token = token.lower()
    return int(token) if token.isdecimal() else _EN_VALUES[token]


def _matches(patterns, text):
    return {_value(match.group(1)) for pattern in patterns for match in pattern.finditer(text)}


def _korean_matches(pattern, text):
    result = set()
    for match in pattern.finditer(text):
        token = match.group(1)
        result.add(int(token) if token.isdecimal() else common._NUMBERS[token])
    return result


def _registration_counts(text, registrations, available_counts):
    """Conservatively bind counts to slots; never assign a missing slot or ID."""
    if not registrations:
        return
    slot_mentions = _matches((_EN_SLOT,), text) | _korean_matches(common._ANY_SLOT, text)
    if len(registrations) == 1 and len(slot_mentions) == 1:
        item = registrations[0]
        if item["slot"] not in slot_mentions or item["count"] not in available_counts:
            raise common.IntentValidationError("ungrounded_value", "registration slot or count was not stated")
        return
    # For multiple English/mixed slots, demand an explicit slot-then-count
    # segment. Shared counts or preposed counts require clarification downstream.
    anchors = list(_EN_SLOT.finditer(text))
    associations = {}
    for index, match in enumerate(anchors):
        end = anchors[index + 1].start() if index + 1 < len(anchors) else len(text)
        fragment = text[match.end():end]
        counts = _matches((_COUNT,), fragment) | {n for n, unit in common._mentioned_numbers(fragment) if unit.startswith("개")}
        associations.setdefault(_value(match.group(1)), set()).update(counts)
    for item in registrations:
        if item["count"] not in associations.get(item["slot"], set()):
            raise common.IntentValidationError("ungrounded_value", "multiple registration slots require explicit per-slot counts")


def _english_or_mixed_context(candidate, text, registered_scene):
    intent = candidate["intent"]
    if intent in common.NON_ACTIONABLE:
        return candidate
    issue = common._input_issue(text)
    if issue:
        raise common.IntentValidationError("blocked_input", issue)
    for code, pattern in _EN_UNSAFE:
        if pattern.search(text):
            raise common.IntentValidationError("blocked_input", code)
    action_text = re.sub(r"기억해\s*(?:둔|두었던|놓은)", "기억한", text)
    if len(_EN_ACTION.findall(action_text)) + len(common._ACTION_VERB.findall(action_text)) > 1:
        raise common.IntentValidationError("blocked_input", "compound")
    if intent in ("move", "stack", "swap", "register") and re.search(r"\b(?:this|that)\b", text, _FLAGS):
        raise common.IntentValidationError("blocked_input", "ambiguous_reference")
    if re.search(r'["“”‘’]', text) or re.search(r"(?<!\w)'[^']+'", text):
        raise common.IntentValidationError("blocked_input", "quoted_utterance")
    is_question = ("?" in text or "？" in text or re.search(r"(?:수\s*있|가능|되나요|되니|할까|어떨까)", text)
                   or re.search(r"\b(?:is\s+it\s+possible|can\s+(?:it|the\s+robot)|could\s+(?:it|the\s+robot))\b", text, _FLAGS))
    korean_action = re.search(r"옮|이동|쌓|바꾸|바꿔|교환|복원|복구|정지|멈추|기억|등록", text)
    if is_question and (intent != "status" or korean_action or _EN_ACTION.search(text)):
        raise common.IntentValidationError("blocked_input", "question_is_not_a_status_request")
    if not (common._INTENT_WORDS[intent].search(text) or _EN_INTENTS[intent].search(text)):
        raise common.IntentValidationError("ungrounded_intent", "the requested action is not stated")
    if intent == "restore" and not re.search(r"기억|저장|아까|이전|\b(?:remember(?:ed)?|saved|previous|earlier|prior|memory)\b", text, _FLAGS):
        raise common.IntentValidationError("ungrounded_intent", "restoration requires an explicit remembered reference")
    for match in _EN_RANGE.finditer(text):
        number = _value(match.group(1))
        kind = match.group(0).lower()
        upper = 2 if kind.startswith(("height", "level", "floor", "layer")) else 4
        if not 1 <= number <= upper:
            raise common.IntentValidationError("blocked_input", "out_of_range")

    slots = _matches((_EN_SLOT,), text) | _korean_matches(common._ANY_SLOT, text)
    sources = (_matches((_EN_SOURCE_FROM, _EN_SOURCE_TO, _EN_SOURCE_SUFFIX, _REVERSE_SOURCE), text)
               | _korean_matches(common._SOURCE_SLOT, text))
    destinations = (_matches((_EN_DEST_PREFIX, _EN_DEST_SUFFIX, _REVERSE_DEST), text)
                    | _korean_matches(common._DESTINATION_SLOT, text))
    objects = (_matches((_OBJECT_BEFORE, _OBJECT_AFTER, _KOREAN_OBJECT_EN_NUM), text)
               | _korean_matches(common._OBJECT_BEFORE, text) | _korean_matches(common._OBJECT_AFTER, text))
    counts = (_matches((_COUNT, _COUNT_ONE), text)
              | {n for n, unit in common._mentioned_numbers(text) if unit.startswith("개")})
    heights = (_matches((_HEIGHT_BEFORE, _HEIGHT_AFTER), text)
               | {n for n, unit in common._mentioned_numbers(text) if unit == "층"})
    if intent in ("swap", "status"):
        sources = slots
    if intent == "swap":
        destinations = slots
    supports = {"source_slot": sources, "destination_slot": destinations, "object_id": objects,
                "count": counts, "height": heights}
    for name, values in supports.items():
        if candidate[name] is not None and candidate[name] not in values:
            raise common.IntentValidationError("ungrounded_value", f"{name} and its role were not stated in the utterance")
    if intent in ("move", "stack"):
        if any(n != 1 for n in counts) or re.search(r"\b(?:all|both|multiple|several)\b", text, _FLAGS):
            raise common.IntentValidationError("blocked_input", "only one explicitly requested cylinder can be moved")
        if heights and candidate["height"] not in heights:
            raise common.IntentValidationError("ungrounded_value", "stated destination height was omitted or altered")
    if intent == "swap" and (counts or objects or re.search(r"맨\s*위|\btop\s+(?:cylinder|object)\b", text, _FLAGS)):
        raise common.IntentValidationError("blocked_input", "swap is limited to whole slots")
    if intent in ("remember", "restore", "initial") and (slots or objects or counts or heights):
        raise common.IntentValidationError("blocked_input", "partial or numbered memory goals are unsupported")

    registrations = candidate["registrations"]
    if registrations is not None:
        _registration_counts(text, registrations, counts | ({0} if re.search(r"없|\b(?:empty|none|zero)\b", text, _FLAGS) else set()))
        ids = [obj for item in registrations for obj in (item["object_ids"] or [])]
        slot_ids = [item["slot"] for item in registrations]
        if (len(slot_ids) != len(set(slot_ids)) or len(ids) != len(set(ids)) or sum(item["count"] for item in registrations) > 4
                or any(item["object_ids"] is not None and len(item["object_ids"]) != item["count"] for item in registrations)):
            raise common.IntentValidationError("invalid_registration", "inconsistent or duplicated registration")
        if any(obj not in objects for obj in ids):
            raise common.IntentValidationError("ungrounded_value", "registration object ID was not stated")

    scene = None
    if registered_scene is not None:
        try:
            scene = validate_scene(registered_scene)
        except (TypeError, ValueError) as exc:
            raise common.IntentValidationError("invalid_registered_scene", str(exc)) from exc
    object_id, source, destination = candidate["object_id"], candidate["source_slot"], candidate["destination_slot"]
    if intent in ("move", "stack", "swap") and scene is None:
        raise common.IntentValidationError("missing_registered_scene", "registered scene is required for target validation")
    if object_id is not None and (scene is None or object_id not in {obj for slot in scene for obj in slot}):
        raise common.IntentValidationError("inactive_object", "object is not in the registered scene")
    if intent in ("move", "stack"):
        object_slot = next((i + 1 for i, slot in enumerate(scene) if object_id in slot), None) if object_id else None
        if source is not None and object_slot is not None and source != object_slot:
            raise common.IntentValidationError("target_mismatch", "source slot and object ID disagree")
        actual_source = source if source is not None else object_slot
        if not scene[actual_source - 1] or (object_id is not None and scene[actual_source - 1][-1] != object_id):
            raise common.IntentValidationError("invalid_source", "source is empty or requested object is not on top")
        if actual_source == destination or len(scene[destination - 1]) >= 2:
            raise common.IntentValidationError("invalid_destination", "source equals destination or destination is full")
        if intent == "stack" and not scene[destination - 1]:
            raise common.IntentValidationError("invalid_destination", "stack requires an existing destination support")
        if candidate["height"] is not None and candidate["height"] != len(scene[destination - 1]) + 1:
            raise common.IntentValidationError("invalid_height", "requested destination height disagrees with registered scene")
    elif intent == "swap":
        if source == destination or not scene[source - 1] or not scene[destination - 1]:
            raise common.IntentValidationError("invalid_swap", "swap needs two different occupied registered slots")
    return candidate


def validate_context(candidate, text, registered_scene):
    if not re.search(r"[A-Za-z]", text):
        return common.validate_context(candidate, text, registered_scene)
    return _english_or_mixed_context(candidate, text, registered_scene)


class BilingualIntentInterpreter:
    """Same result and shared schema as the Korean interpreter, new fixed prompt."""
    def __init__(self, client):
        self.client = client
        self.prompt = load_prompt()

    def interpret(self, text, *, registered_scene=None, timeout_seconds=30.0):
        start = time.perf_counter()
        def result(status, proposal=None, raw_text=None, raw_json=None, json_valid=None, schema_valid=None, detail=None):
            return common.IntentResult(status, proposal, raw_text, raw_json, json_valid, schema_valid, detail,
                                       (time.perf_counter() - start) * 1000)
        if not isinstance(text, str) or not text.strip() or len(text) > common.MAX_TEXT_CHARS:
            return result("invalid_input", detail="empty, non-string, or overlong utterance")
        text = unicodedata.normalize("NFKC", text)
        try:
            raw = self.client.complete([{"role": "system", "content": self.prompt}, {"role": "user", "content": text}],
                                       timeout_seconds=timeout_seconds)
        except (TimeoutError, socket.timeout):
            return result("timeout", detail="local inference timed out")
        except Exception as exc:
            return result("transport_error", detail=f"{type(exc).__name__}: {exc}")
        try:
            candidate = common.parse_json(raw)
        except common.IntentValidationError as exc:
            return result(exc.code, raw_text=raw if isinstance(raw, str) else None, json_valid=False, schema_valid=False, detail=str(exc))
        try:
            common.validate_schema(candidate)
        except common.IntentValidationError as exc:
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=False, detail=str(exc))
        try:
            validate_context(candidate, text, registered_scene)
        except common.IntentValidationError as exc:
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True, detail=str(exc))
        return result("ok", proposal=candidate, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True)
