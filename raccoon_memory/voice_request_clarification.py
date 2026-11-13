"""Research-only feedback for an already nonexecuting turn. Returns text only."""
import re

from . import task_program as v1
from . import task_program_v2 as v2


_MISSING_OPERATIONS = {
    "an explicit operation in the composite request was omitted",
    "a composite operation has no corresponding supported step",
    "composite operations are missing, ambiguous or in the wrong order",
    "composite operations are in the wrong stated order",
}
_TOWER_QUESTION = (
    "탑의 맨 위 원통을 내리는 뜻인가요? 어느 자리의 탑에서 어느 빈자리로 내릴지 "
    "‘몇 번 자리에서 몇 번 자리로’라고 넣어 전체 요청을 다시 말씀해 주세요."
)
_TRANSFER_DIAGNOSTICS = {
    ("unresolved_reference", "move values need the same unquoted affirmative action clause"),
    ("ungrounded_value", "source_slot is not supported by its evidence"),
    ("ungrounded_value", "destination_slot is not supported by its evidence"),
    ("ungrounded_value", "object_id is not supported by its evidence"),
}
_TRANSFER_VERBS = re.compile(
    "|".join("(?:" + v1._TASK_WORDS[kind] + ")" for kind in ("move", "stack", "unstack")),
    re.IGNORECASE | re.ASCII,
)
_KO_DIRECT_END = re.compile(
    r"(?:옮겨|옮기세요|이동해|이동하세요|쌓아|쌓으세요|올려|올리세요|"
    r"내려(?:놓아|놔)?|내리세요|가져와|놓아|놓으세요|놔)"
    r"\s*(?:줘|주세요|주십시오)?\s*[.!。！]?\s*$"
)
_EN_DIRECT_START = re.compile(
    r"^\s*(?:please\s+)?(?:move|transfer|put|place|bring|carry|stack|pile|unstack)\b",
    re.IGNORECASE | re.ASCII,
)
# These count action mentions, not inferred steps or values. Ambiguous cases
# fall back to the existing full-request prompt. Familiar two-verb Korean
# expressions such as 내려놓아 and 쌓아 올려 remain one mention.
_KO_ACTION_MENTIONS = re.compile(
    r"(?:쌓아\s*(?:올려|올리)|내려\s*(?:놓|놔)|올려\s*(?:놓|놔)|옮겨\s*(?:놓|놔)|"
    r"옮|이동|쌓|올려|올리|내려|내리|풀|놓|놔|보내|바꾸|바꿔|교환)"
)
_EN_ACTION_MENTIONS = re.compile(
    r"\b(?:move|moving|transfer|transferring|put|putting|place|placing|bring|bringing|"
    r"carry|carrying|stack|stacking|pile|piling|unstack|unstacking|swap|swapping|"
    r"exchange|exchanging|switch|switching)\b", re.IGNORECASE | re.ASCII,
)
_EN_LOCATION_NOUN_PREFIX = re.compile(
    r"\b(?:a|an|the|this|that|one|another|first|second)\s+$",
    re.IGNORECASE | re.ASCII,
)
_EN_UNCLEAR_INFINITIVE = re.compile(
    r"\bto\s+(?!(?:a|an|the|my|our|your|this|that|slot|position|tower|pile|stack|"
    r"place|cylinder|one|two|three|four|[1-4])\b)[a-z]+\b", re.IGNORECASE | re.ASCII,
)
_OTHER_SCOPE = re.compile(
    r"[\"'‘’“”?,;:？！；，：\r\n]|[.!。！].+|"
    r"멈|정지|중단|그만|취소|반복|되풀이|기억|저장|복원|등록|"
    r"먼저|다음|이어서|그리고|그\s*(?:뒤|후)|나서|또는|혹은|"
    r"(?:은|긴|린|한|운|푼)\s*(?:뒤|후)|"
    r"(?:집|잡|가져오|가져와)(?:고|아서|어서|면서|자마자)|"
    r"(?:옮겨|이동해|쌓아|올려|내려|놓아|놔)\s*(?:주고|놓고)|"
    r"(?:지\s*(?:마|말|않)|(?:안|못)\s+|말고|아니|달라고|달래|라고|사람|말했|들었|부탁했|요청했)|"
    r"\b(?:stop|halt|cancel|repeat|remember|save|restore|register|if|unless|when|"
    r"not|never|without|then|before|after|afterwards|next|also|plus|or|and|"
    r"followed|said|says|told|asked|wanted|someone|somebody)\b",
    re.IGNORECASE | re.ASCII,
)


def _single_action_mention(text, *, korean):
    if korean:
        return len(list(_KO_ACTION_MENTIONS.finditer(text))) == 1
    matches = []
    for match in _EN_ACTION_MENTIONS.finditer(text):
        if (match.group().lower() in ("pile", "place", "stack")
                and _EN_LOCATION_NOUN_PREFIX.search(text[max(0, match.start() - 30):match.start()])):
            continue
        matches.append(match)
    return len(matches) == 1


def _linked_korean_clause(text):
    # A preceding connective can introduce another operation even when its
    # verb is absent from our transfer vocabulary. Locative particles such as
    # '자리에서' do not themselves introduce another clause.
    words = re.findall(r"[가-힣]+", text)
    return any(len(word) > 1 and word.endswith(("고", "며", "면서", "거나", "자마자", "서"))
               and not word.endswith(("에서", "께서", "로서", "로써")) for word in words[:-1])


def _direct_transfer_request(text):
    """Conservatively identify a single direct request without resolving its values."""
    if _OTHER_SCOPE.search(text) or v2._ARM_HOME_CONDITION.search(text):
        return False
    korean = bool(re.search(r"[가-힣]", text))
    if ((korean and _linked_korean_clause(text))
            or (not korean and _EN_UNCLEAR_INFINITIVE.search(text))):
        return False
    if not _single_action_mention(text, korean=korean):
        return False
    if korean:
        direct = re.search(r"원통|원기둥", text) and _KO_DIRECT_END.search(text)
    else:
        direct = re.search(r"\bcylinders?\b", text, re.IGNORECASE | re.ASCII) and _EN_DIRECT_START.search(text)
    if not direct:
        return False
    return any(v2._explicit_span(text, *match.span(), match.group()) for match in _TRANSFER_VERBS.finditer(text))


def _transfer_question(text, interpretation, admission):
    """Explain a rejected value binding without selecting values or retaining a turn."""
    if (admission.get("kind") != "rejected" or admission.get("program_executed") is not False
            or admission.get("status") != interpretation.status or interpretation.program is not None
            or interpretation.execution_authorized or interpretation.validated_stop_prefix
            or interpretation.schema_valid is not True
            or (interpretation.status, interpretation.detail) not in _TRANSFER_DIAGNOSTICS
            or not _direct_transfer_request(text)):
        return None
    if re.search(r"[가-힣]", text):
        return ("요청을 실행하지 않았습니다. 원통 번호와 자리 번호, 동작 횟수를 구분해 "
                "어떤 원통을 어디에 놓을지 전체 요청을 다시 말씀해 주세요.")
    return ("The request was not executed. Please restate the full request, saying which cylinder to move, "
            "where to put it, and distinguishing cylinder numbers, slot numbers and the number of moves.")


def recovery_question(text, *, route, interpretation=None, admission=None):
    """Supplement a nonexecuting routing/validation result; never reuse a previous command."""
    if not isinstance(text, str) or not text.strip():
        return None
    if route == "clarify" and interpretation is None and admission is None:
        top = v2.explicit_top_unstack_roles(text)
        return _TOWER_QUESTION if top is not None and top[2] is None else None
    if route == "full_command_interpretation" and interpretation is not None and isinstance(admission, dict):
        question = _transfer_question(text, interpretation, admission)
        if question:
            return question
    if (route != "full_command_interpretation" or interpretation is None or not isinstance(admission, dict)
            or admission.get("kind") != "rejected" or admission.get("program_executed") is not False
            or interpretation.status != "unresolved_reference" or admission.get("status") != interpretation.status
            or interpretation.program is not None or interpretation.execution_authorized
            or interpretation.validated_stop_prefix or interpretation.detail not in _MISSING_OPERATIONS):
        return None
    try:
        pair = v2._required_pair(text)
    except v2.ProgramError:
        return None
    if pair is None or pair[0] != "move":
        return None
    if re.search(r"[가-힣]", text):
        return ("배치 기억과 원통 이동 중 어느 쪽을 먼저 할까요? ‘먼저 …, 그다음 …’처럼 "
                "두 단계와 출발·도착 자리를 넣어 전체 요청을 다시 말씀해 주세요.")
    return ("Which comes first, remembering the layout or moving the cylinder? "
            "Please restate both steps in order, including the source and destination slots.")
