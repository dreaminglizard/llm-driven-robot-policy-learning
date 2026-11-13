"""Bounded, ordered task-program proposals; never dispatches robot commands.

The caller owns the request ID and a small, versioned reference context.  The
model never receives a remembered goal layout or joint/path data. Validation
does not repair model actions or turn a negation into an affirmative command.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import time

from . import language_intent as common
from . import language_intent_bilingual as bilingual

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs/task_program.schema.json"
PROMPT_PATH = ROOT / "configs/task_program.prompt.v1.txt"
MAX_STEPS = 8
MAX_TEXT_CHARS = 500
VALIDATOR_VERSION = "task-program-validator.v2"
GENERATION = {"temperature": 0, "top_p": 1.0, "max_tokens": 512, "seed": 20261030, "stream": False}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")


class ProgramError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fail(code, message):
    raise ProgramError(code, message)


def _valid_id(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def normalize_context(value):
    """Only status and opaque references; arbitrary goals and paths are refused."""
    defaults = {"execution_state": "unknown", "repeat_state": "unknown", "active_request_id": None,
                "state_revision": None, "previous_task": None, "previous_program": None, "return_focus": None}
    if value is None:
        return defaults
    if not isinstance(value, dict) or set(value) - set(defaults):
        _fail("invalid_context", "context contains unsupported fields")
    result = {**defaults, **copy.deepcopy(value)}
    if result["execution_state"] not in ("unknown", "idle", "planning", "running", "stopping", "stopped_confirmed", "uncertain"):
        _fail("invalid_context", "unknown execution state")
    if result["repeat_state"] not in ("unknown", "inactive", "active", "ending"):
        _fail("invalid_context", "unknown repeat state")
    for key in ("active_request_id", "state_revision"):
        if result[key] is not None and not _valid_id(result[key]):
            _fail("invalid_context", f"{key} must be an opaque ID")
    if result["execution_state"] in ("planning", "running", "stopping") and (result["active_request_id"] is None or result["state_revision"] is None):
        _fail("invalid_context", "active state requires a request ID and state revision")
    for key in ("previous_task", "previous_program"):
        reference = result[key]
        if reference is None:
            continue
        if not isinstance(reference, dict) or set(reference) != {"status", "reference_id", "revision"}:
            _fail("invalid_context", f"{key} must contain status, reference_id and revision only")
        if reference["status"] not in ("unique", "ambiguous", "none", "unknown"):
            _fail("invalid_context", "invalid previous-reference status")
        if reference["status"] == "unique":
            if not _valid_id(reference["reference_id"]) or not _valid_id(reference["revision"]):
                _fail("invalid_context", "unique previous reference requires ID and revision")
        elif reference["reference_id"] is not None or reference["revision"] is not None:
            _fail("invalid_context", "ambiguous or absent reference cannot carry an ID")
    focus = result["return_focus"]
    if focus is not None:
        if not isinstance(focus, dict) or set(focus) != {"kind", "source", "revision"}:
            _fail("invalid_context", "return focus must contain kind, source and revision only")
        if focus["kind"] not in ("arm_home", "registered_initial", "task_start", "learned_memory") or focus["source"] != "explicit_recent" or not _valid_id(focus["revision"]):
            _fail("invalid_context", "return focus must be an explicit, current, versioned reference")
    return result


def canonical_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def proposal_schema():
    """Same step schema; the trusted caller, not the model, supplies request_id."""
    schema = canonical_schema()
    del schema["properties"]["request_id"]
    schema["required"].remove("request_id")
    schema["description"] = "Model proposal only; the caller attaches the request ID. " + schema["description"]
    return schema


def load_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_messages(text, reference_context=None):
    context = normalize_context(reference_context)
    payload = {"transcript": text, "reference_context": context}
    return [{"role": "system", "content": load_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def _type_matches(value, expected):
    return {"null": value is None, "string": isinstance(value, str), "integer": type(value) is int,
            "array": isinstance(value, list), "object": isinstance(value, dict),
            "boolean": type(value) is bool}.get(expected, False)


def _schema_check(value, schema, path="$", *, code="invalid_schema"):
    """Check the small subset actually used by the shipped JSON schema."""
    if "oneOf" in schema:
        matches = 0
        for choice in schema["oneOf"]:
            try:
                _schema_check(value, choice, path, code=code)
                matches += 1
            except ProgramError:
                pass
        if matches != 1:
            _fail(code, f"{path}: expected exactly one matching variant")
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_matches(value, kind) for kind in kinds):
            _fail(code, f"{path}: incorrect type")
    if "const" in schema and value != schema["const"]:
        _fail(code, f"{path}: incorrect constant")
    if "enum" in schema and value not in schema["enum"]:
        _fail(code, f"{path}: unsupported value")
    if type(value) is int:
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            _fail(code, f"{path}: integer outside range")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            _fail(code, f"{path}: string length outside range")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            _fail(code, f"{path}: string does not match pattern")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            _fail(code, f"{path}: list length outside range")
        if schema.get("uniqueItems") and len({json.dumps(x, ensure_ascii=False, sort_keys=True) for x in value}) != len(value):
            _fail(code, f"{path}: duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                _schema_check(item, schema["items"], f"{path}[{index}]", code=code)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            _fail(code, f"{path}: missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            _fail(code, f"{path}: unsupported fields")
        for key in set(value) & set(props):
            _schema_check(value[key], props[key], f"{path}.{key}", code=code)


def validate_schema(program):
    _schema_check(program, canonical_schema())
    return program


def _contains_any(text, pattern):
    return re.search(pattern, text, re.IGNORECASE | re.ASCII) is not None


def _quote_spans(text):
    """Find paired quotation marks without treating contractions as endings."""
    pairs = {'"': '"', '“': '”', '‘': '’', "'": "'"}
    spans, index = [], 0
    while index < len(text):
        opener = text[index]
        if opener not in pairs or (opener == "'" and index and text[index - 1].isalnum()):
            index += 1
            continue
        closer = pairs[opener]
        end = index + 1
        while end < len(text):
            if text[end] == closer:
                # The apostrophe in don't / don’t is part of the quoted word.
                if (closer in ("'", '’') and end + 1 < len(text)
                        and text[end - 1].isalnum() and text[end + 1].isalnum()):
                    end += 1
                    continue
                spans.append((index, end + 1))
                index = end
                break
            end += 1
        index += 1
    return spans


def _quoted(text, start, end):
    return any(left <= start and end <= right for left, right in _quote_spans(text))


def _span_for(text, evidence, after):
    start = text.find(evidence, after)
    if start < 0:
        _fail("ungrounded_evidence", "evidence is missing or not in step order")
    return start, start + len(evidence)


def _negative_stop(text):
    return _contains_any(text, r"(?:멈추|정지하|중단하|멈춰)\s*지\s*(?:마|말|않)|(?:멈추|정지하|중단하)면\s*안|\b(?:do\s+not|don['’]?t|never)\s+(?:stop|halt)\b")


def _negative_repeat_stop(text):
    return _contains_any(text, r"반복.{0,12}(?:중단|정지|멈추)\s*지\s*(?:마|말|않)|\b(?:do\s+not|don['’]?t|never)\s+stop\s+repeating\b")


def _negated_reference(evidence, reference):
    start = evidence.find(reference)
    if start < 0:
        return True
    return _negated_span(evidence, start, start + len(reference))


def _negated_span(text, start, end):
    """Check a mention's adjacent negation, not an unrelated later clause."""
    prefix = text[max(0, start - 30):start]
    suffix = text[end:min(len(text), end + 30)]
    return (re.match(r"\s*(?:이|가|은|는|을|를|으로|로)?\s*(?:아니|말고|빼고|제외)", suffix) is not None
            or re.search(r"\b(?:not|except|instead\s+of)\s+(?:the\s+)?$", prefix, re.IGNORECASE | re.ASCII) is not None)


_EN_NEGATION_CUE = re.compile(
    # Treat these markers conservatively throughout the clause, including when
    # the model supplies only the verb as evidence. This intentionally rejects
    # ambiguous uses of "no" or "not" rather than guessing their scope.
    r"\b(?:no|not|never|cannot|avoid(?:s|ed|ing)?|(?:don|doesn|didn|can|mustn|shouldn|won|wouldn)['’]?t|"
    r"[a-z]+n['’]t)\b", re.IGNORECASE | re.ASCII)
_EN_NEGATION_BOUNDARY = re.compile(
    r"[;.!?。？！]|\b(?:then|but|instead|however)\b", re.IGNORECASE | re.ASCII)
_EN_REPEAT_BOUNDARY_CORRECTION = re.compile(
    r"(?P<negation>\b(?:do\s+not|don['’]?t|never))\s+stop\s+repeating\s+"
    r"after\s+(?:the\s+)?(?:current|this)\s+(?P<rejected>transfer|task)\b"
    r"(?:\s*,\s*|\s+)(?P<affirmed_stop>stop)\s+repeating\s+"
    r"after\s+(?:the\s+)?(?:current|this)\s+(?P<affirmed>transfer|task)\b",
    re.IGNORECASE | re.ASCII)


def _corrected_repeat_stop_ends_negation(text, cue, start):
    """Recognize only a complete, explicit correction of a repeat boundary."""
    for correction in _EN_REPEAT_BOUNDARY_CORRECTION.finditer(text):
        if (not (correction.start('negation') <= cue.start() and cue.end() <= correction.end('negation'))
                or correction.group('rejected').lower() == correction.group('affirmed').lower()
                or any(left < correction.end() and correction.start() < right for left, right in _quote_spans(text))):
            continue
        if start == correction.start('affirmed_stop'):
            return True
        # A separate, explicitly targeted motion stop immediately following a
        # correction must not disappear from the full-request stop check.
        if (start >= correction.end()
                and re.fullmatch(r'\s*,\s*', text[correction.end():start])
                and re.match(r'\bstop(?=\s*(?:[,.;!?。？！]|$)|'
                             r'\s+(?:now|immediately|moving|(?:the\s+)?(?:robot|arm|motion))\b)',
                             text[start:], re.IGNORECASE | re.ASCII)):
            left = max((text.rfind(mark, 0, correction.start()) for mark in '.;!?。？！'), default=-1) + 1
            right = min((pos for mark in '.;!?。？！' if (pos := text.find(mark, start)) >= 0),
                        default=len(text))
            context = re.compile(
                r'\b(?:if|unless|when|provided|assuming|said|says|asked|asks|told|wrote|'
                r'requested|suggested|mentioned|quoted|say|quote|write|explain|describe)\b',
                re.IGNORECASE | re.ASCII)
            if not any(not _quoted(text, *item.span()) for item in context.finditer(text, left, right)):
                return True
    return False


def _english_negated_action(text, start):
    """Conservatively keep a lexical negation across coordinated verbs."""
    for cue in reversed(list(_EN_NEGATION_CUE.finditer(text, 0, start))):
        if _quoted(text, *cue.span()):
            continue
        if any(not _quoted(text, *boundary.span())
               for boundary in _EN_NEGATION_BOUNDARY.finditer(text, cue.end(), start)):
            continue
        if _corrected_repeat_stop_ends_negation(text, cue, start):
            continue
        # Commas also enumerate verbs ("do not move, stack, or stop"), so a
        # comma alone is not enough to turn a later verb into a new command.
        return True
    return False


def _negated_action(text, start, end):
    """Bind a negation to this verb instead of rejecting the whole utterance."""
    prefix = text[:start]
    suffix = text[end:min(len(text), end + 32)]
    return (re.search(r"(?:안|못)\s*$", prefix) is not None
            or _english_negated_action(text, start)
            or (text[start:end] == '이어' and re.match(r'서\s*하?지\s*(?:마|말|않)', suffix) is not None)
            or re.match(r"[가-힣]{0,6}(?:지\s*(?:마|말|않)|(?:으)?면\s*안\s*(?:돼|되))", suffix) is not None)


_KOREAN_REQUEST_QUESTION_END = re.compile(
    r'(?:쌓아|올려|옮겨|이동해|내려(?:놔|놓아)?|돌려(?:놔|놓아)?|되돌려|'
    r'바꿔|변경해|기억해|저장해|복원해|복귀해|돌아가|보내|'
    r'놔|놓아|둬|가져(?:다(?:가)?)?|멈춰|정지해|중단해|반복해|계속해)\s*'
    r'(?:줘(?:요)?|주세요|주십시오)\s*[?？]\s*$')
_KOREAN_REQUEST_QUESTION_EXCLUDE = re.compile(
    r'왜|어떻게|언제|어디|누가|누구|무엇|뭐|몇|어느|무슨|어떤|얼마|'
    r'질문|물어|설명|달라고|라고\s*(?:했|말|들)|'
    r'할까|할까요|하나요|있나요|없나요|인가요|인지|해야\s*하나|'
    r'(?:해도|어도|아도)\s*(?:돼|되|될)|'
    r'지\s*(?:마|말|않)|(?:으)?면\s*안\s*(?:돼|되)|'
    r'(?<![가-힣])(?:안|못)\s*(?=[가-힣])|만약|(?:으)?면\s+', re.I)


def is_korean_request_question_punctuation(text):
    """Recognize a single direct Korean request with a final question mark.

    This identifies punctuation only; the caller must still validate the
    action, destination and original evidence before doing anything.
    """
    if not isinstance(text, str):
        return False
    value = text.strip()
    if (len(value) > MAX_TEXT_CHARS or len(re.findall(r'[?？]', value)) != 1
            or not _KOREAN_REQUEST_QUESTION_END.search(value)
            or re.search(r'[.!。！;；\n\r"“”‘’\']', value)
            or _KOREAN_REQUEST_QUESTION_EXCLUDE.search(value)):
        return False
    # A correction opener is common in retargeting, but another comma may
    # introduce a separate question or condition into the same utterance.
    remainder = re.sub(r'^\s*(?:아니|아니요|아니야|아니에요|아냐)\s*[,，]\s*', '', value)
    return re.search(r'[,，]', remainder) is None


def _question_only(text, evidence_start, evidence_end):
    beginning = 0
    for marker in (".", "!", "。", "?", "？", ";"):
        beginning = max(beginning, text.rfind(marker, 0, evidence_start) + 1)
    end = len(text)
    for marker in (".", "!", "。"):
        position = text.find(marker, evidence_end)
        if position >= 0:
            end = min(end, position)
    clause = text[beginning:end]
    return ((("?" in clause or "？" in clause)
             and not is_korean_request_question_punctuation(text))
            or _contains_any(clause, r"\b(?:should|could|can|would|may)\s+(?:i|we|you|the\s+robot|it)\b")
            or re.search(r"(?:아야|어야|춰야|해야)\s*(?:하나|하나요|할까|할까요)|(?:해도|어도|아도)\s*(?:돼|되나요|될까)", clause) is not None)


_REPORTING_PREFIX = re.compile(
    r"(?:\b(?:say|quote|write|mention|explain|describe|read|spell|print|display|tell)"
    r"(?:\s+(?:[a-z]+ly|aloud|me|us|them|the|a|these|those|following|word|words|phrase|"
    r"command|instruction|to|that))*\s*:?\s+|"
    r"\b(?:what|how|why|when)\s+(?:does|did|is|would|could|can|should)\s+)$",
    re.IGNORECASE | re.ASCII)
_CONTROL_CLAUSE_BOUNDARY = re.compile(
    r"[;.!?。？！]|\b(?:then|but|instead)\b|"
    r"\band\s+(?=(?:please\s+)?(?:continue|resume|finish|complete|carry\s+on|"
    r"return|restore|go\s+back)\b)", re.IGNORECASE | re.ASCII)
_REFERENCE_NEGATION = re.compile(r"아니|말고|빼고|제외|지\s*(?:마|말|않)|(?:으)?면\s*안\s*(?:돼|되)")


def _reported_action(text, start):
    return _REPORTING_PREFIX.search(text[:start]) is not None


def _affirmed_control_clauses(text, start, end, pattern, reference, *, target_kind=None):
    """Find affirmative clauses containing both a control and its reference."""
    boundaries = [m for m in _CONTROL_CLAUSE_BOUNDARY.finditer(text)
                  if not _quoted(text, *m.span())]
    clauses = []
    for verb in re.finditer(pattern, text[start:end], re.IGNORECASE | re.ASCII):
        left, right = start + verb.start(), start + verb.end()
        if (_quoted(text, left, right) or _negated_action(text, left, right)
                or _question_only(text, left, right) or _reported_action(text, left)):
            continue
        clause_left = max((m.end() for m in boundaries if m.end() <= left), default=0)
        clause_right = min((m.start() for m in boundaries if m.start() >= right), default=len(text))
        if target_kind in ('active_task', 'previous_task'):
            roles, _ = _reference_roles(text[clause_left:clause_right])
            if roles and target_kind not in roles:
                continue
        for ref in re.finditer(re.escape(reference), text[start:end]):
            ref_left, ref_right = start + ref.start(), start + ref.end()
            # A full reference retains its original affirmative sentence ending.
            closing = text[clause_right:ref_right]
            includes_ending = (ref_right == clause_right + 1
                               and (closing == '.' or (closing in ('?', '？')
                                    and is_korean_request_question_punctuation(text)))
                               and not _REFERENCE_NEGATION.search(reference)
                               and not _EN_NEGATION_CUE.search(reference)
                               and not any(ref_left < qright and qleft < ref_right
                                           for qleft, qright in _quote_spans(text)))
            within_clause = ref_right <= clause_right or includes_ending
            if (clause_left <= ref_left and within_clause
                    and not _quoted(text, ref_left, ref_right)
                    and not _negated_span(text, ref_left, ref_right)):
                clause = (clause_left, clause_right)
                if clause not in clauses:
                    clauses.append(clause)
                break
    return clauses


def _affirmed_control(text, start, end, pattern, reference, *, target_kind=None):
    return bool(_affirmed_control_clauses(text, start, end, pattern, reference,
                                          target_kind=target_kind))


_TARGETED_REPEAT_NEGATION = tuple(re.compile(pattern, re.IGNORECASE | re.ASCII) for pattern in (
    r"(?:현재|지금|이번|이전|앞선|지난|직전)(?:의)?\s*(?:작업|동작|프로그램)"
    r"(?:은|는|을|를|이|가|만)?\s*(?:(?:다시|다시는|더|더는)\s*)?"
    r"(?P<neg>(?:반복|되풀이)(?:을|은|는|도)?\s*(?:(?:더|더는|이제|다시)\s*)?하?지\s*(?:마|말|않))",
    r"(?:작업|동작|프로그램|이동|명령)(?:은|는|을|를|이|가|만)\s*"
    r"(?:(?:다시|다시는|더|더는)\s*)?"
    r"(?P<neg>(?:반복|되풀이)(?:을|은|는|도)?\s*(?:(?:더|더는|이제|다시)\s*)?하?지\s*(?:마|말|않))",
    r"(?:옮기|옮겨|이동|쌓|풀|바꾸|교환)[^,;.!?。？！]{0,32}?(?:것|행위)(?:은|는|을|를|이|가)\s*"
    r"(?P<neg>(?:반복|되풀이)(?:을|은|는|도)?\s*(?:(?:더|더는|이제|다시)\s*)?하?지\s*(?:마|말|않))",
    r"(?P<neg>\b(?:do\s+not|don['’]?t|never)\s+(?:ever\s+)?repeat)\s+"
    r"(?:(?:the|a|an|this|that)\s+)?"
    r"(?:(?:current|active|previous|prior|last|first|second|other)\s+)?"
    r"(?:task|action|step|program|move|movement|transfer|moving|transferring|swapping)\b",
))
_REPEAT_NEGATION = re.compile(
    r"(?:반복|되풀이)(?:을|은|는|도)?\s*(?:(?:더|더는|이제|다시)\s*)?하?지\s*(?:마|말|않)"
    r"|\b(?:do\s+not|don['’]?t|never)\s+(?:ever\s+)?repeat\b",
    re.IGNORECASE | re.ASCII,
)
_EXPLICIT_REPEAT_STOP = re.compile(
    r"(?:반복|되풀이)(?:을|만|은|는|도|이)?\s*(?:(?:지금|즉시|바로|이후|더는)\s*)?"
    r"(?P<ko>중단|종료|정지|멈추|멈춰|그만|끝내)"
    r"|(?P<ko_first>그만|중단|정지|멈춰)\s*(?:반복|되풀이)"
    r"|\b(?P<en>stop|end|cancel)\s+(?:the\s+)?(?:repeat(?:ing)?|repetition)\b",
    re.IGNORECASE | re.ASCII,
)


def _has_global_repeat_stop(text, evidence_start, evidence_end):
    """Require an independent stop of repetition, not a rejected task choice."""
    targeted = {
        match.span("neg") for pattern in _TARGETED_REPEAT_NEGATION for match in pattern.finditer(text)
        if evidence_start <= match.start("neg") and match.end("neg") <= evidence_end
    }
    for match in _EXPLICIT_REPEAT_STOP.finditer(text, evidence_start, evidence_end):
        group = next(name for name in ("ko", "ko_first", "en") if match.group(name) is not None)
        start, end = match.span(group)
        if not (_quoted(text, match.start(), match.end()) or _negated_action(text, start, end)
                or _question_only(text, start, end)):
            return True
    for match in _REPEAT_NEGATION.finditer(text, evidence_start, evidence_end):
        if (match.span() not in targeted and not _quoted(text, match.start(), match.end())
                and not _question_only(text, match.start(), match.end())):
            return True
    return False


def _affirmed_values(patterns, text, *, whole=None, offset=0, korean=False):
    whole = text if whole is None else whole
    values = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            if _negated_span(whole, offset + match.start(), offset + match.end()):
                continue
            token = match.group(1)
            values.add((int(token) if token.isdecimal() else common._NUMBERS[token]) if korean else bilingual._value(token))
    return values


_KO_TOWER = r"(?:더미|덤이|탑)"
# The app names each tower by its slot. 덤이 is accepted only as this
# numbered manipulation argument, never by rewriting the transcript.
_KO_NUMBERED_TOWER = (common._NUMBER + r"\s*번(?:째)?\s*"
                      r"(?:(?:자리|위치)\s*(?:의\s*|에\s*(?:있는|놓인|놓여\s*있는)\s*)?)?" + _KO_TOWER)
_KO_TOWER_SOURCE = re.compile(_KO_NUMBERED_TOWER + r"\s*(?:에서|의\s*(?=원통|물체)|(?:의\s*)?(?=(?:맨|가장)\s*위))")
_KO_SLOT_STACK_DESTINATION = re.compile(
    _KO_NUMBERED_TOWER + r"\s*(?:위(?:에다가|에다|에|로)|(?:으)?로|에(?!서))"
    r"(?!\s*(?:있는|있던|놓인|놓여\s*있는|올려진))")
_KO_PLACE = r"(?:자리|위치)"
_KO_SLOT_OCCUPANT = r"(?:(?:의|에\s*(?:있는|놓인|놓여\s*있는))\s*)?(?:원통|물체)"
_KO_PLACE_ABOVE = (_KO_PLACE + r"\s*(?:" + _KO_SLOT_OCCUPANT + r"\s*(?:의\s*)?)?"
                   r"위(?:에다가|에다|에|로|(?![가-힣]))"
                   r"(?!\s*(?:있는|있던|놓인|놓여\s*있는|올려진))")
_KO_PLACE_STACK_DESTINATION = re.compile(common._NUMBER + r"\s*번(?:째)?\s*" + _KO_PLACE_ABOVE)
_EN_STACK_SLOT_PREFIX = r"(?:on\s+top\s+of|onto|on)\s+(?:the\s+cylinder\s+(?:in|at)\s+)?slot\s+"
_EN_STACK_SLOT_DESTINATION = re.compile(r"\b" + _EN_STACK_SLOT_PREFIX + bilingual._NUM + r"\b",
                                        re.IGNORECASE | re.ASCII)


def _korean_transfer_slots(text, *, whole=None, offset=0):
    """Bind explicit Korean slot roles, keeping object IDs separate."""
    whole = text if whole is None else whole
    stack_patterns = (_KO_SLOT_STACK_DESTINATION, _KO_PLACE_STACK_DESTINATION)
    stack_destinations = [match for pattern in stack_patterns for match in pattern.finditer(text)]
    sources, destinations = set(), set()

    for match in common._SOURCE_SLOT.finditer(text):
        if _negated_span(whole, offset + match.start(), offset + match.end()):
            continue
        # A bare "3번 원통" names object 3, not the slot containing it.
        if re.search(r"번(?:째)?\s*(?:원통|물체)$", match.group()):
            continue
        # The cylinder or pile on a named slot can be the support of a stack.
        if any(target.start() <= match.start() and match.end() <= target.end()
               for target in stack_destinations):
            continue
        token = match.group(1)
        sources.add(int(token) if token.isdecimal() else common._NUMBERS[token])

    for match in common._DESTINATION_SLOT.finditer(text):
        suffix = text[match.end():]
        # The old "에" alternative also matched the start of "에서" and
        # "자리에 있는 원통", both of which describe a source/location.
        if match.group().endswith('에') and re.match(
                r"(?:서|게|\s*(?:있는|있던|놓인|놓여\s*있는))", suffix):
            continue
        if _negated_span(whole, offset + match.start(), offset + match.end()):
            continue
        token = match.group(1)
        destinations.add(int(token) if token.isdecimal() else common._NUMBERS[token])

    destinations.update(_affirmed_values(stack_patterns, text, whole=whole, offset=offset, korean=True))
    sources.update(_affirmed_values((_KO_TOWER_SOURCE,), text, whole=whole, offset=offset, korean=True))
    return sources, destinations


def _task_supports(text, *, whole=None, offset=0):
    korean_sources, korean_destinations = _korean_transfer_slots(text, whole=whole, offset=offset)
    slots = (_affirmed_values((bilingual._EN_SLOT,), text, whole=whole, offset=offset)
             | _affirmed_values((common._ANY_SLOT,), text, whole=whole, offset=offset, korean=True))
    sources = (_affirmed_values((bilingual._EN_SOURCE_FROM, bilingual._EN_SOURCE_TO, bilingual._EN_SOURCE_SUFFIX,
                                  bilingual._REVERSE_SOURCE), text, whole=whole, offset=offset)
               | korean_sources)
    destinations = (_affirmed_values((bilingual._EN_DEST_PREFIX, bilingual._EN_DEST_SUFFIX,
                                      bilingual._REVERSE_DEST, _EN_STACK_SLOT_DESTINATION),
                                     text, whole=whole, offset=offset)
                    | korean_destinations)
    objects = (bilingual._matches((bilingual._OBJECT_BEFORE, bilingual._OBJECT_AFTER, bilingual._KOREAN_OBJECT_EN_NUM), text)
               | bilingual._korean_matches(common._OBJECT_BEFORE, text) | bilingual._korean_matches(common._OBJECT_AFTER, text))
    counts = (bilingual._matches((bilingual._COUNT, bilingual._COUNT_ONE), text)
              | {n for n, unit in common._mentioned_numbers(text) if unit.startswith("개")})
    heights = (bilingual._matches((bilingual._HEIGHT_BEFORE, bilingual._HEIGHT_AFTER), text)
               | {n for n, unit in common._mentioned_numbers(text) if unit == "층"})
    return slots, sources, destinations, objects, counts, heights


def _validate_multi_slot_layout(entries, evidence):
    """Bind each count/ID to its own stated slot; never shuffle a set of values."""
    if len(entries) <= 1:
        return
    anchors = []
    for match in common._REGISTRATION_SLOT.finditer(evidence):
        token = match.group(1) or match.group(2)
        number = int(token) if token.isdecimal() else common._NUMBERS[token]
        anchors.append((match.start(), match.end(), number))
    for match in bilingual._EN_SLOT.finditer(evidence):
        anchors.append((match.start(), match.end(), bilingual._value(match.group(1))))
    anchors.sort()
    associations = {}
    for i, (start, end, number) in enumerate(anchors):
        segment_end = anchors[i + 1][0] if i + 1 < len(anchors) else len(evidence)
        if segment_end < end:
            _fail("ungrounded_value", "overlapping layout slot references are ambiguous")
        fragment = evidence[end:segment_end]
        _, _, _, ids, counts, _ = _task_supports(fragment)
        if _contains_any(fragment, r"없|\b(?:empty|zero|none)\b"):
            counts.add(0)
        associations[number] = (counts, ids)
    for entry in entries:
        counts, ids = associations.get(entry["slot"], (set(), set()))
        if entry["count"] is not None and entry["count"] not in counts:
            _fail("ungrounded_value", "layout count is not bound to its stated slot")
        if entry["object_ids"] is not None and any(object_id not in ids for object_id in entry["object_ids"]):
            _fail("ungrounded_value", "layout object ID is not bound to its stated slot")


_TASK_WORDS = {
    "move": r"옮|이동|가져|보내|놓|\b(?:move|transfer|put|place|bring|carry|send)\b",
    "stack": r"쌓|올려|\b(?:stack|pile)\b",
    "unstack": r"풀|내려|분리|\b(?:unstack|unpile)\b",
    "swap": r"바꾸|바꿔|교환|맞바꿔|\b(?:swap|exchange|switch)\b",
    "remember": r"기억|저장|메모|\b(?:remember|memorize|save|store)\b",
    "register": r"등록|\b(?:register|registration)\b",
    "arrange": r"배치|놓|\b(?:arrange|layout|place)\b",
}
_NEGATED_TASK = r"(?:옮기|이동하|쌓|풀|바꾸|기억하|등록하)\s*지\s*(?:마|말|않)|\b(?:do\s+not|don['’]?t|never)\s+(?:move|stack|unstack|swap|remember|register)\b"
_TASK_UNRESOLVED = (r"빨간|빨강|파란|파랑|초록|녹색|노란|노랑|왼쪽|오른쪽|가까운|이것|저것|그것|이거|저거|그거|"
                    r"여기|저기|거기|\b(?:red|blue|green|yellow|left|right|nearest|closest|it|them|there)\b")
_INJECTED_INSTRUCTION = r"(?:지시|규칙|시스템|프롬프트).{0,25}무시|무시.{0,25}(?:지시|규칙|시스템|프롬프트)|\bignore\b.{0,60}\b(?:rules|instructions|previous|system)\b"


def _validate_task(step, text, evidence_start):
    kind, evidence = step["kind"], step["evidence"]
    verbs = list(re.finditer(_TASK_WORDS[kind], evidence, re.IGNORECASE | re.ASCII))
    if (not verbs or not any(not _negated_action(text, evidence_start + verb.start(), evidence_start + verb.end())
                             and not _quoted(text, evidence_start + verb.start(), evidence_start + verb.end())
                             and not _reported_action(text, evidence_start + verb.start())
                             and not _question_only(text, evidence_start + verb.start(), evidence_start + verb.end())
                             for verb in verbs)):
        _fail("unsupported_action", "task evidence does not affirm this action")
    if _contains_any(evidence, _TASK_UNRESOLVED) or _contains_any(text, _INJECTED_INSTRUCTION):
        _fail("unresolved_reference", "task includes an unknown description or an instruction override")
    src, dst, obj = step["source_slot"], step["destination_slot"], step["object_id"]
    slots, sources, destinations, objects, counts, heights = _task_supports(evidence, whole=text, offset=evidence_start)
    if kind == "swap":
        sources = destinations = slots
    for field, value, supported in (("source_slot", src, sources), ("destination_slot", dst, destinations),
                                    ("object_id", obj, objects), ("count", step["count"], counts), ("height", step["height"], heights)):
        if value is not None and value not in supported:
            _fail("ungrounded_value", f"{field} is not supported by its evidence")
    if kind in ("move", "stack", "unstack"):
        if dst is None or (src is None and obj is None) or (src is not None and src == dst) or step["layout"] is not None:
            _fail("invalid_task", "transfer needs a distinct destination and a source slot or object")
        if counts and step["count"] not in counts:
            _fail("ungrounded_value", "stated object count was omitted or changed")
        if heights and step["height"] not in heights:
            _fail("ungrounded_value", "stated target height was omitted or changed")
    elif kind == "swap":
        if src is None or dst is None or src == dst or any(step[key] is not None for key in ("object_id", "count", "height", "layout")):
            _fail("invalid_task", "slot swap needs exactly two distinct slots")
    elif kind == "remember":
        if any(step[key] is not None for key in ("source_slot", "destination_slot", "object_id", "count", "height", "layout")):
            _fail("invalid_task", "remember does not contain a generated layout")
    elif kind in ("register", "arrange"):
        if any(step[key] is not None for key in ("source_slot", "destination_slot", "object_id", "count", "height")):
            _fail("invalid_task", "layout requests use only explicit layout entries")
        if kind == "arrange" and step["layout"] is None:
            _fail("invalid_task", "arrange requires explicitly stated layout entries")
        if step["layout"] is not None:
            entries = step["layout"]
            all_ids = [obj_id for entry in entries for obj_id in (entry["object_ids"] or [])]
            if len({entry["slot"] for entry in entries}) != len(entries) or len(all_ids) != len(set(all_ids)):
                _fail("invalid_task", "duplicate slot or object ID")
            total = sum(entry["count"] if entry["count"] is not None else len(entry["object_ids"] or []) for entry in entries)
            if total > 4:
                _fail("invalid_task", "more than four objects")
            for entry in entries:
                if entry["slot"] not in slots or (entry["count"] is None and entry["object_ids"] is None):
                    _fail("ungrounded_value", "layout entry lacks an explicit slot or contents")
                if entry["count"] is not None and entry["count"] not in counts and not (entry["count"] == 0 and _contains_any(evidence, r"없|\b(?:empty|zero|none)\b")):
                    _fail("ungrounded_value", "layout count is not stated")
                if entry["object_ids"] is not None and (any(obj_id not in objects for obj_id in entry["object_ids"])
                                                       or (entry["count"] is not None and len(entry["object_ids"]) != entry["count"])):
                    _fail("ungrounded_value", "layout IDs are missing or inconsistent")
            _validate_multi_slot_layout(entries, evidence)


def _return_explicit(kind, evidence):
    patterns = {"arm_home": r"(?:팔|arm).{0,30}(?:원위치|대기|home)|(?:원위치|home).{0,30}(?:팔|arm)",
                "registered_initial": r"(?:등록|registered).{0,30}(?:초기|initial|starting)",
                "task_start": r"(?:작업|task).{0,30}(?:시작|start|before)",
                "learned_memory": r"기억|\b(?:remembered|memory|memorized)\b"}
    return _contains_any(evidence, patterns[kind])


_TASK_REFERENCE = re.compile(
    r"(?P<current>지금|현재|이번|진행\s*중(?:인)?|하고\s*있는)\s*(?:하는\s*)?(?:작업|동작|프로그램)"
    r"|(?P<previous>이전|앞선|지난|직전)\s*(?:작업|동작|프로그램)"
    r"|\b(?P<en_current>current|active|ongoing|this)\s+(?:task|action|program)\b"
    r"|\b(?P<en_previous>previous|prior|last)\s+(?:task|action|program)\b", re.IGNORECASE | re.ASCII)

_RETURN_REFERENCE = {
    "arm_home": r"팔(?:을|이|은|는|의)?\s*(?:대기|원위치)(?:\s*자세)?|\barm\b.{0,25}\bhome(?:\s+pose)?\b",
    "registered_initial": (r"등록[가-힣]*\s*초기(?:\s*(?:배치|상태|위치))?|"
                           r"(?:맨\s*)?처음(?:에|으로)?\s*등록(?:한|했던|된)\s*(?:배치|상태|위치)|"
                           r"\bregistered\s+(?:initial|starting)(?:\s+(?:arrangement|layout|position|state))?\b"),
    "task_start": r"작업(?:을|이)?\s*시작(?:하기|한)?\s*(?:전|때|이전)?\s*(?:배치|상태|시점)?|\b(?:this\s+)?task\s+start(?:ed|ing)?\b",
    "learned_memory": r"기억(?:한|했던)?\s*(?:배치|상태|위치)|\b(?:remembered|memorized)\s+(?:arrangement|layout|position|state)\b",
}


def _reference_roles(text):
    """Collect affirmative and negated active/previous reference mentions."""
    positive, negative = set(), set()
    for match in _TASK_REFERENCE.finditer(text):
        word = match.group(0).lower()
        if match.group('current') or match.group('en_current'):
            role = 'active_task'
        else:
            role = 'previous_program' if ('프로그램' in word or 'program' in word) else 'previous_task'
        (negative if _negated_span(text, match.start(), match.end()) else positive).add(role)
    return positive, negative


def _validate_task_reference(kind, reference, text):
    if _negated_reference(text, reference):
        _fail("unsupported_action", "the selected task reference is negated")
    positive, negative = _reference_roles(text)
    selected, _ = _reference_roles(reference)
    if selected and kind not in selected:
        _fail("unresolved_reference", "reference points to a different task or program")
    if kind in negative and kind not in positive:
        _fail("unsupported_action", "the requested task reference is negated")
    if positive and kind not in positive:
        _fail("unresolved_reference", "reference disagrees with the affirmative task mention")


def _repeat_stop_verb(text, start, end):
    prefix = text[max(0, start - 35):start]
    suffix = text[end:min(len(text), end + 35)]
    return (re.search(r"반복(?:을|이|의)?\s*(?:(?:지금|즉시|바로)\s*)?$", prefix) is not None
            or re.match(r"\s+(?:repeating|the\s+repeat|the\s+repetition)\b", suffix, re.IGNORECASE | re.ASCII) is not None)


def _affirmed_motion_stops(text):
    result = []
    for match in re.finditer(r"멈|정지|중단|\b(?:stop|halt|cancel)\b", text, re.IGNORECASE | re.ASCII):
        start, end = match.span()
        if (_negated_action(text, start, end) or _repeat_stop_verb(text, start, end)
                or _question_only(text, start, end) or _quoted(text, start, end)):
            continue
        if _reported_action(text, start):
            continue
        result.append((start, end))
    return result


def validate_program(program, text, reference_context=None, *, on_valid_step=None):
    validate_schema(program)
    context = normalize_context(reference_context)
    if not isinstance(text, str) or not text.strip():
        _fail("invalid_input", "transcript is empty")
    after = 0
    stop_spans = []
    for index, step in enumerate(program["steps"]):
        start, end = _span_for(text, step["evidence"], after)
        after = end
        op, evidence = step["op"], step["evidence"]
        if op not in ("clarify", "status") and _quoted(text, start, end):
            _fail("quoted_action", "quoted words do not authorize a task or control step")
        if op == "task":
            _validate_task(step, text, start)
        elif op == "stop_motion":
            matched = [(a, b) for a, b in _affirmed_motion_stops(text) if start <= a and b <= end]
            if not matched:
                _fail("unsupported_action", "stop is negated, questioned, or not stated")
            if all(_contains_any(text[a:min(len(text), b + 24)], r"정지(?:라는|란)\s*(?:말|단어|버튼)")
                   for a, b in matched):
                _fail("unsupported_action", "mentioning a stop is not a stop request")
            stop_spans.extend(matched)
        elif op == "stop_repeat":
            if (not _contains_any(evidence, r"반복|\brepeat(?:ing)?\b")
                    or not _contains_any(evidence, r"멈|정지|중단|그만|하지\s*(?:마|말)|\b(?:stop|end|do\s+not|don['’]?t|never)\b")
                    or _negative_repeat_stop(text) or _question_only(text, start, end)
                    or not _has_global_repeat_stop(text, start, end)):
                _fail("unsupported_action", "repeat stop is negated, questioned, or not stated")
            if step["boundary"] == "after_task" and not _contains_any(text, r"(?:이번|현재|이)\s*(?:작업|동작)|\b(?:current|this)\s+(?:task|action|step)\b"):
                _fail("ungrounded_value", "after-task boundary was not stated")
        elif op == "return":
            target = step["target"]
            if target["reference"] not in evidence or not _contains_any(evidence, r"원위치|복귀|돌아|돌려|돌리|되돌|복원|\b(?:return|home|restore|back)\b"):
                _fail("ungrounded_value", "return reference is not supported by evidence")
            return_verbs = r"원위치|복귀|돌아|돌려|돌리|되돌|복원|\b(?:return|restore|go\s+back)\b"
            if not _contains_any(evidence, return_verbs):
                return_verbs += r"|\b(?:home|back)\b"
            control_clauses = _affirmed_control_clauses(text, start, end, return_verbs, target["reference"])
            if not control_clauses:
                _fail("unsupported_action", "return or its reference is negated")
            kind = target["kind"]
            focus = context["return_focus"]
            if kind != "unresolved":
                role = re.compile(_RETURN_REFERENCE[kind], re.IGNORECASE | re.ASCII)
                mentions = [m for left, right in control_clauses for m in role.finditer(text, left, right)
                            if not _quoted(text, *m.span())]
                affirmative = any(not _negated_span(text, *m.span()) for m in mentions)
                if mentions and not affirmative:
                    _fail("unsupported_action", "return target kind is negated")
                if not affirmative and not _return_explicit(kind, evidence) and not (focus is not None and focus["kind"] == kind):
                    _fail("unresolved_reference", "return target is not explicitly stated or uniquely focused")
        elif op == "continue":
            target = step["target"]
            if target["reference"] not in evidence or not _contains_any(evidence, r"계속|마쳐|완료|\b(?:continue|finish|complete|carry\s+on)\b"):
                _fail("ungrounded_value", "continue reference is not supported by evidence")
            if not _affirmed_control(text, start, end,
                                     r"계속|마쳐|완료|\b(?:continue|finish|complete|carry\s+on)\b",
                                     target["reference"], target_kind=target["kind"]):
                _fail("unsupported_action", "continuation is negated")
            if target["kind"] == "active_task" and (context["execution_state"] != "running"
                                                     or any(previous["op"] == "stop_motion" for previous in program["steps"][:index])):
                _fail("unresolved_reference", "continuation needs the current running task")
            if target["kind"] == "active_task":
                _validate_task_reference("active_task", target["reference"], text)
        elif op == "repeat":
            target, count = step["target"], step["count"]
            if target["reference"] not in text or not _contains_any(evidence, r"반복|되풀이|\b(?:repeat|repeating|again)\b"):
                _fail("ungrounded_value", "repeat reference is not supported by evidence")
            if _contains_any(text, r"반복\s*하?지\s*(?:마|말|않)|\b(?:do\s+not|don['’]?t|never)\s+repeat\b"):
                _fail("unsupported_action", "repeat is negated")
            if target["kind"] in ("active_task", "previous_task", "previous_program"):
                _validate_task_reference(target["kind"], target["reference"], text)
            indices = target["step_indices"]
            if target["kind"] == "prior_steps":
                if indices is None or indices != sorted(indices) or any(i >= index or program["steps"][i]["op"] not in ("task", "return") for i in indices):
                    _fail("invalid_repeat_reference", "repeat body must reference earlier task or return steps in order")
            elif indices is not None:
                _fail("invalid_repeat_reference", "step indices are allowed only for prior_steps")
            if target["kind"] in ("previous_task", "previous_program"):
                ref = context[target["kind"]]
                if ref is None or ref["status"] != "unique":
                    _fail("unresolved_reference", "previous reference is not unique and current")
            if target["kind"] == "active_task" and context["execution_state"] != "running":
                _fail("unresolved_reference", "repeat needs the currently running task")
            if count["mode"] in ("continuous", "unresolved") and count["value"] is not None:
                _fail("invalid_repeat_count", "continuous or unresolved count cannot have a value")
            if count["mode"] == "continuous" and not _contains_any(evidence, r"계속|끊임|무한|\b(?:keep|continuous(?:ly)?|indefinitely)\b"):
                _fail("ungrounded_value", "continuous repetition was not stated")
            if count["mode"] in ("total", "additional"):
                if count["value"] is None:
                    _fail("invalid_repeat_count", "finite repeat count needs a value")
                mentioned = {n for n, unit in common._mentioned_numbers(evidence) if unit.startswith("번")}
                mentioned |= bilingual._matches((re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:more\s+)?times\b", re.IGNORECASE | re.ASCII),), evidence)
                if _contains_any(evidence, r"\bonce\b"): mentioned.add(1)
                if _contains_any(evidence, r"\btwice\b"): mentioned.add(2)
                if count["value"] not in mentioned:
                    _fail("ungrounded_value", "repeat count was not stated")
                has_more = _contains_any(evidence, r"더|추가|\b(?:more|additional)\b")
                if (count["mode"] == "additional") != has_more:
                    _fail("invalid_repeat_count", "total and additional repeat counts disagree with the transcript")
        elif op == "status":
            if not _contains_any(evidence, r"상태|배치|어디|확인|\b(?:status|state|where|layout)\b"):
                _fail("ungrounded_value", "status was not stated")
        if on_valid_step is not None:
            on_valid_step(index, step)
    if any(span not in stop_spans for span in _affirmed_motion_stops(text)):
        _fail("missing_explicit_stop", "an affirmed motion-stop clause was omitted from the proposal")
    return program


class IncompleteResponse(RuntimeError):
    def __init__(self, reason, raw_text):
        super().__init__(f"generation did not finish normally: {reason}")
        self.raw_text = raw_text


class TaskProgramChatClient:
    """OpenAI-compatible, loopback-only transport with fixed generation options."""
    def __init__(self, base_url, model, *, transport=None):
        self.url = common._loopback_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("a local model name is required")
        self.model = model
        self.transport = common._http_transport if transport is None else transport

    def complete(self, messages, *, timeout_seconds):
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be positive and finite")
        request = {"model": self.model, "messages": messages, **GENERATION,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "task_program", "strict": True, "schema": proposal_schema()}}}
        raw = self.transport(self.url, json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"), timeout_seconds)
        if not isinstance(raw, bytes) or len(raw) > common.MAX_REPLY_BYTES:
            raise RuntimeError("invalid server response size or type")
        try:
            envelope = json.loads(raw, object_pairs_hook=common._no_duplicate_keys, parse_constant=common._bad_constant)
            choice = envelope["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("content must be a string")
            if choice.get("finish_reason") not in (None, "stop"):
                raise IncompleteResponse(choice.get("finish_reason"), content)
            return content
        except IncompleteResponse:
            raise
        except (ValueError, KeyError, IndexError, TypeError, RecursionError) as exc:
            raise RuntimeError("malformed chat-completions response") from exc


@dataclass(frozen=True)
class TaskProgramResult:
    status: str
    request_id: str | None
    program: dict | None
    raw_text: str | None
    raw_json: object | None
    json_valid: bool | None
    schema_valid: bool | None
    unresolved: bool | None
    validated_stop_prefix: list
    detail: str | None
    elapsed_ms: float


class TaskProgramInterpreter:
    def __init__(self, client):
        self.client = client

    def interpret(self, text, *, request_id, reference_context=None, timeout_seconds=30.0):
        start = time.perf_counter()
        def result(status, program=None, raw_text=None, raw_json=None, json_valid=None, schema_valid=None, unresolved=None,
                   validated_stop_prefix=None, detail=None):
            return TaskProgramResult(status, request_id if _valid_id(request_id) else None, program, raw_text, raw_json,
                                     json_valid, schema_valid, unresolved, validated_stop_prefix or [], detail, (time.perf_counter() - start) * 1000)
        if not _valid_id(request_id) or not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            return result("invalid_input", detail="invalid request ID or transcript")
        try:
            context = normalize_context(reference_context)
        except ProgramError as exc:
            return result(exc.code, detail=str(exc))
        try:
            raw = self.client.complete(build_messages(text, context), timeout_seconds=timeout_seconds)
        except IncompleteResponse as exc:
            try:
                parsed = common.parse_json(exc.raw_text)
                json_valid = True
            except common.IntentValidationError:
                parsed, json_valid = None, False
            return result("incomplete_response", raw_text=exc.raw_text, raw_json=parsed, json_valid=json_valid,
                          schema_valid=None, detail=str(exc))
        except (TimeoutError, socket.timeout):
            return result("timeout", detail="local inference timed out")
        except Exception as exc:
            return result("transport_error", detail=f"{type(exc).__name__}: {exc}")
        try:
            candidate = common.parse_json(raw)
        except common.IntentValidationError as exc:
            return result("invalid_json", raw_text=raw if isinstance(raw, str) else None, json_valid=False, schema_valid=False, detail=str(exc))
        try:
            _schema_check(candidate, proposal_schema())
            program = {"schema": candidate["schema"], "request_id": request_id, "steps": candidate["steps"]}
            validate_schema(program)
        except ProgramError as exc:
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=False, detail=str(exc))
        validated_stops = []
        def keep_stop(index, step):
            if step["op"] == "stop_motion":
                validated_stops.append({"step_index": index, "step": copy.deepcopy(step)})
        try:
            validate_program(program, text, context, on_valid_step=keep_stop)
        except ProgramError as exc:
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True,
                          validated_stop_prefix=validated_stops, detail=str(exc))
        unresolved = any((step["op"] == "clarify" or (step["op"] in ("return", "repeat", "continue") and step["target"]["kind"] == "unresolved")
                          or (step["op"] == "repeat" and step["count"]["mode"] == "unresolved")
                          or (step["op"] == "stop_repeat" and step["boundary"] == "unresolved")) for step in program["steps"])
        return result("ok", program=program, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True,
                      unresolved=unresolved, validated_stop_prefix=validated_stops)
