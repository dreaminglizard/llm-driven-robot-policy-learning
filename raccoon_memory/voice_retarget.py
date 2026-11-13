"""Bounded recognition of an explicitly changed destination in one utterance.

The result proposes a destination only; it is not execution authority. The
coordinator binds the original transcript to the active program and transfer.
"""
from __future__ import annotations

import re

from .task_program import is_korean_request_question_punctuation

_KO_NUMBER = {'1': 1, '2': 2, '3': 3, '4': 4, '일': 1, '이': 2, '삼': 3, '사': 4,
              '한': 1, '두': 2, '세': 3, '네': 4}
_EN_NUMBER = {'1': 1, '2': 2, '3': 3, '4': 4, 'one': 1, 'two': 2, 'three': 3, 'four': 4}
_KO_ACTION = r'(?:바꿔|변경해|옮겨|내려놔|올려|돌려놔|돌려놓아|돌려|되돌려|놔|놓아|놓|둬|보내|가|해)(?:\s*(?:줘요|줘|주세요))?'
_EN_SLOT = r'(?:slot|position|place)\s*(?:number\s*)?'
_EN_ACTION = r'(?:put|place|move|send|take|bring)\s+(?:it|that|the\s+(?:cylinder|object))'

_KOREAN_REPLACE = re.compile(
    r'\s*(?:(?:아니|아니요|아니야|아니에요|아냐)\s*[,，]?\s*)?'
    r'(?P<previous>[1-4일이삼사한두세네])\s*번(?P<previous_place>\s*자리)?\s*'
    r'(?:말고|대신|(?:이|가)?\s*아니라|(?:이|가)?\s*아니고)\s*'
    r'(?:(?:지금|바로|그냥|대신|다시)\s*)?'
    r'(?P<slot>[1-4일이삼사한두세네])\s*번(?P<slot_place>\s*자리)?\s*(?:으로|로|에|위에)'
    r'(?:\s*' + _KO_ACTION + r')?\s*[.!。！]*\s*')
_KOREAN = re.compile(
    r'\s*(?:(?:아니|아니요|아니야|아니에요|아냐)\s*[,，]?\s*)?'
    r'(?:(?:지금|바로|그냥|대신|다시)\s*)?'
    r'(?:(?:목적지(?:를|는)?|그\s*원통(?:을|은)?|원통(?:을|은)?|그거(?:를|는)?|이거(?:를|는)?)\s*)?'
    r'(?P<slot>[1-4일이삼사한두세네])\s*번(?P<slot_place>\s*자리)?\s*(?:으로|로|에|위에)'
    r'(?:\s*' + _KO_ACTION + r')?\s*[.!。！]*\s*')
_ENGLISH_REPLACE_BEFORE = re.compile(
    r'\s*(?:no\s*[,，]?\s*)?(?:not|instead\s+of)\s+(?:the\s+)?' + _EN_SLOT +
    r'(?P<previous>[1-4]|one|two|three|four)\s*[,，]?\s*'
    r'(?:please\s+)?' + _EN_ACTION + r'\s+(?:to|into|in|on|onto)\s+(?:the\s+)?' + _EN_SLOT +
    r'(?P<slot>[1-4]|one|two|three|four)\s*(?:instead)?\s*[.!]*\s*', re.I | re.ASCII)
_ENGLISH_REPLACE_AFTER = re.compile(
    r'\s*(?:please\s+)?' + _EN_ACTION + r'\s+(?:to|into|in|on|onto)\s+(?:the\s+)?' + _EN_SLOT +
    r'(?P<slot>[1-4]|one|two|three|four)\s+(?:instead\s+of|not)\s+(?:the\s+)?' + _EN_SLOT +
    r'(?P<previous>[1-4]|one|two|three|four)\s*[.!]*\s*', re.I | re.ASCII)
_ENGLISH = re.compile(
    r'\s*(?:(?:no|actually|instead)\s*[,，]?\s*)?(?:please\s+)?'
    r'(?:(?:put|place|move|send|take|bring)\s+(?:it|that|the\s+(?:cylinder|object))\s+'
    r'|(?:change|switch)\s+(?:(?:it|the\s+destination)\s+)?|go\s+)'
    r'(?:to|into|in|on|onto)\s+(?:the\s+)?(?:slot|position|place)\s*'
    r'(?P<slot>[1-4]|one|two|three|four)\s*(?:instead)?\s*[.!]*\s*', re.I | re.ASCII)
_KOREAN_ORIGIN = re.compile(
    r'\s*(?:(?:아니|아니요|아니야|아니에요|아냐)\s*[,，]?\s*)?'
    r'(?:(?:지금|바로|그냥|대신|다시)\s*)?'
    r'(?:(?:그\s*원통|원통|그거|이거)(?:을|를|은|는)?\s*)?'
    r'(?:(?:원래|처음|출발(?:한)?)\s*(?:자리|위치)|제자리|원위치)(?:로|에)\s*'
    r'(?:돌아가|돌려놔|돌려놓아|돌려|되돌려|옮겨|놔|놓아|보내|해)'
    r'(?:\s*(?:줘요|줘|주세요))?\s*[.!。！]*\s*')
_ENGLISH_ORIGIN = re.compile(
    r'\s*(?:(?:no|actually|instead)\s*[,，]?\s*)?'
    r'(?:(?:put|place|bring|take|move|send)\s+(?:it|that|the\s+(?:cylinder|object))\s+back|go\s+back)\s+'
    r'(?:to\s+)?(?:the\s+)?(?:original|starting|source)\s+(?:slot|position|place)\s*[.!]*\s*',
    re.I | re.ASCII)
_LOOKS_LIKE_REDIRECT = re.compile(
    r'(?:[0-9일이삼사한두세네]\s*번.{0,35}(?:으로|로|에|위).{0,30}'
    r'(?:바꿔|바꾸|바꿀|변경|옮|놓|놔|둬|보내|돌려|가\s*(?:줘|주세요))|'
    r'(?:목적지|자리).{0,30}(?:바꿔|변경)|'
    r'(?:(?:원래|처음|출발).{0,20}(?:자리|위치)|제자리|원위치).{0,25}(?:돌아|돌려|옮|놓|놔|보내)|'
    r'(?:\b(?:change|switch)\b|\b(?:put|place|move|send|take|bring)\s+'
    r'(?:it|that|the\s+(?:cylinder|object))\b).{0,60}'
    r'\b(?:slot|position|place|destination)\b|'
    r'\b(?:slot|position|place|destination)\b.{0,40}\b(?:instead|change|switch)\b)',
    re.I | re.ASCII)


def parse_voice_retarget_details(text):
    """Return (kind, new slot, mentioned old slot), without execution authority."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        return 'unrelated', None, None
    # Only the local grammar sees the optional request punctuation removed;
    # the coordinator and all evidence retain the caller's original text.
    parsed_text = re.sub(r'[?？]\s*$', '', text) if is_korean_request_question_punctuation(text) else text
    match = _KOREAN_REPLACE.fullmatch(parsed_text)
    if match:
        # Native-number homophones such as "세 번" can mean repetitions.
        # Require the speaker to say "자리" when one occurs in a redirect.
        if ((match.group('previous') in '한두세네' and match.group('previous_place') is None)
                or (match.group('slot') in '한두세네' and match.group('slot_place') is None)):
            return 'ambiguous', None, None
        previous = _KO_NUMBER[match.group('previous')]
        destination = _KO_NUMBER[match.group('slot')]
        if previous == destination:
            return 'ambiguous', None, None
        return 'valid', destination, previous
    for pattern in (_ENGLISH_REPLACE_BEFORE, _ENGLISH_REPLACE_AFTER):
        match = pattern.fullmatch(parsed_text)
        if match:
            previous = _EN_NUMBER[match.group('previous').lower()]
            destination = _EN_NUMBER[match.group('slot').lower()]
            if previous == destination:
                return 'ambiguous', None, None
            return 'valid', destination, previous
    match = _KOREAN.fullmatch(parsed_text)
    if match:
        if match.group('slot') in '한두세네' and match.group('slot_place') is None:
            return 'ambiguous', None, None
        return 'valid', _KO_NUMBER[match.group('slot')], None
    match = _ENGLISH.fullmatch(parsed_text)
    if match:
        return 'valid', _EN_NUMBER[match.group('slot').lower()], None
    if _KOREAN_ORIGIN.fullmatch(parsed_text) or _ENGLISH_ORIGIN.fullmatch(parsed_text):
        return 'origin', None, None
    return ('ambiguous', None, None) if _LOOKS_LIKE_REDIRECT.search(text) else ('unrelated', None, None)


def parse_voice_retarget(text):
    """Return (kind, slot): valid, origin, ambiguous, or unrelated."""
    kind, destination, _ = parse_voice_retarget_details(text)
    return kind, destination
