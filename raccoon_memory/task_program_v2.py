"""Versioned task-program proposals with execution-order-independent evidence.

Nothing here dispatches an action. Previous-task continuation is a proposal
whose stop confirmation, current state and replanning remain to be checked.
"""
from __future__ import annotations

from collections import Counter
import base64
import copy
from dataclasses import dataclass, field, replace
import json
import io
import math
from pathlib import Path
import re
import socket
import time

from . import task_program as v1

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / 'configs/task_program.v2_1.schema.json'
PROMPT_PATH = ROOT / 'configs/task_program.v2.prompt_detailed.v1.txt'
LOCAL_MODEL_PROMPT_PATH = ROOT / 'configs/task_program_local_model.prompt.txt'
GENERATION = dict(v1.GENERATION)
ProgramError = v1.ProgramError
_KOREAN_TOP_TO_EMPTY_NOTE = (
    "Korean slot and tower wording: read each number by the noun and particle attached to it. "
    "N번 자리에서 gives source_slot N; N번 빈 자리로 or 빈자리에 gives destination_slot N. "
    "Spoken 일/이/삼/사 and 한/두/세/네 can number those slots. "
    "원통 한 개 or 하나 gives count 1, never an object_id or a height. "
    "Taking the top cylinder down from a tower to an empty slot is unstack; "
    "setting a cylinder on another is stack. An empty destination is not a requested final height. "
    "For a positive task copy the contiguous original clause covering the verb and both named slots. "
    "Keep unspecified fields null and keep questions, quotations, prohibitions and stop requests in their own roles."
)


def canonical_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))


def proposal_schema():
    schema = canonical_schema()
    del schema['properties']['request_id']
    schema['required'].remove('request_id')
    return schema


def local_model_schema(*, with_request_id=False):
    schema = canonical_schema() if with_request_id else proposal_schema()
    items = schema['properties']['steps']['items']
    branches = []
    for branch in items['oneOf']:
        fields = branch['properties']
        source = fields.get('source_slot', {})
        if isinstance(source.get('const'), int) and source['const'] != 1:
            # The strict grammar enumerates each valid source/destination pair.
            # A single integer-source variant represents all of them on the product path.
            continue
        for name in ('source_slot', 'destination_slot', 'object_id', 'destination_object_id'):
            field = fields.get(name)
            if field is not None and 'integer' in field.get('type', ()):
                fields[name] = {'type': field['type']}
        branches.append(branch)
    branches.sort(key=lambda b: b['properties']['op']['const'] != 'clarify')
    slot = {'type': ['integer', 'null']}
    branches.append({
        'type': 'object', 'additionalProperties': False,
        'required': ['op', 'destination_slot', 'previous_slot', 'evidence'],
        'properties': {'op': {'const': 'retarget'}, 'destination_slot': slot, 'previous_slot': slot,
                       'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 300}}})
    items['oneOf'] = branches
    return schema


def load_prompt():
    return PROMPT_PATH.read_text(encoding='utf-8')


def build_messages(text, reference_context=None):
    context = v1.normalize_context(reference_context)
    prompt = load_prompt()
    if (context['execution_state'] == 'idle' and context['active_request_id'] is None
            and context['repeat_state'] == 'inactive' and explicit_top_unstack_roles(text) is not None):
        prompt += '\n' + _KOREAN_TOP_TO_EMPTY_NOTE
    payload = {'transcript': text, 'reference_context': context}
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]


def build_local_model_messages(text, reference_context=None, *, active_transfer_at_onset=False):
    """Product prompt and the captured user transcript, independent of its wording."""
    payload = {'transcript': text, 'reference_context': v1.normalize_context(reference_context),
               'active_transfer_at_onset': active_transfer_at_onset}
    return [{'role': 'system', 'content': LOCAL_MODEL_PROMPT_PATH.read_text(encoding='utf-8')},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]


def local_audio_model_schema(*, voice_output=True):
    schema = local_model_schema()
    schema['properties']['transcript'] = {'type': 'string', 'maxLength': v1.MAX_TEXT_CHARS}
    schema['required'].append('transcript')
    # Conversation output is independent of the executable robot program.
    if voice_output:
        schema['properties']['voice_output'] = {'type': 'string', 'enum': ['none', 'mute', 'unmute']}
        schema['required'].append('voice_output')
    return schema


_VOICE_OUTPUT_INSTRUCTIONS = (
    '\nAlso set voice_output to mute when the speaker asks only for your spoken '
    'answers to stop, for example "조용히 해", "말하지 마", or "stop talking". '
    'Set it to unmute when they ask to hear spoken answers again, for example '
    '"다시 말해" or "you can speak again". The microphone must remain listening; '
    'these are not robot stop, pause, movement, memory, or target-change commands. '
    'For either voice-only request return one clarify step with reason question_only, '
    'question null and evidence copied from this utterance. In a pending request, '
    'also set pending_disposition to unchanged and leave the old plan untouched. '
    'For an utterance requesting both a robot stop and a voice-output change, '
    'retain the stop_motion step and set voice_output to the requested change; '
    'for a pending request use pending_disposition stop. Set voice_output to none '
    'for robot-only stops, questions about quiet, quoted or negated requests, '
    'and all other requests. Do not classify a separate robot command as '
    'voice-only. Return the transcript exactly as heard.'
)


def direct_voice_output_control(text):
    """Fallback for a few complete, unambiguous voice-only requests."""
    if not isinstance(text, str):
        return None
    clean = text.strip().casefold().strip(' .!。！?？')
    if re.fullmatch(r'(?:조용히\s*해(?:\s*(?:줘|주세요))?|말\s*그만해(?:\s*(?:줘|주세요))?|'
                    r'be\s+quiet|stop\s+talking|stop\s+speaking)', clean):
        return 'mute'
    if re.fullmatch(r'(?:다시\s*말해(?:\s*(?:줘|주세요))?|speak\s+again|talk\s+again)', clean):
        return 'unmute'
    return None


def _direct_robot_stop_for_audio(text):
    from .demo_service import _is_direct_stop_text
    if _is_direct_stop_text(text):
        return True
    # Only a complete request that names the robot/motion is promoted here;
    # speech-related stop wording continues to use the voice-output intent.
    return bool(re.fullmatch(
        r'\s*(?:(?:지금|바로|즉시)\s*)?(?:로봇|로보트|팔|동작|작업|움직임)'
        r'(?:을|를)?\s*(?:(?:지금|바로|즉시)\s*)?'
        r'(?:멈춰|멈추어|정지해|중단해)(?:\s*(?:줘|주세요))?\s*[.!。！]*\s*', text))


def _clear_motion_stop(text, reference_context):
    """Validate a clear robot stop without treating 'stop talking' as motion."""
    from .demo_service import _validated_pending_stop_prefix
    prefix = re.match(
        r'^\s*(?:(?:지금|바로|즉시|일단)\s*)?'
        r'(?:(?:로봇|로보트|팔|동작|작업|움직임)(?:을|를)?\s*)?'
        r'(?:(?:지금|바로|즉시)\s*)?'
        r'(?:멈춰|멈추어|멈추고|멈춰서|정지해|정지하고|중단해|중단하고)'
        r'(?:\s*(?:줘|주세요))?(?=\s*(?:[,;.!。！]|그리고|그다음|이어서|$))', text)
    named_target = re.search(
        r'(?:로봇|로보트|팔|동작|작업|움직임)(?:을|를|은|는)?\s*'
        r'(?:(?:지금|바로|즉시)\s*)?(?:멈춰|멈추어|멈추고|멈춰서|정지해|정지하고|중단해|중단하고)|'
        r'\b(?:stop|halt)\s+(?:(?:the|your)\s+)?(?:robot|arm|motion|movement|task)\b', text, re.I)
    bare_control = re.search(
        r'(?:^|[,;.!。！]|그리고|이어서|그\s*다음|하고|한\s*뒤(?:에)?|\b(?:and|then)\b)\s*'
        r'(?:(?:지금|바로|즉시)\s*)?(?:멈춰|멈추어|멈추고|멈춰서|정지해|정지하고|중단해|중단하고|'
        r'(?:stop|halt)(?:\s+now)?)'
        r'(?:\s*(?:줘|주세요))?(?=\s*(?:[,;.!。！]|그리고|이어서|$))', text, re.I)
    return bool((prefix or named_target or bare_control)
                and _validated_pending_stop_prefix(text, v1.normalize_context(reference_context)))


def _mentions_pending_action(text):
    """Do not release a held plan when a voice-control turn may edit it."""
    from .voice_retarget import parse_voice_retarget
    clauses = re.split(r'[,;.!。！]|그리고|이어서|\b(?:and|then)\b|고\s+', text, flags=re.I)
    if any(parse_voice_retarget(part.strip())[0] != 'unrelated' for part in [text, *clauses] if part.strip()):
        return True
    return bool(re.search(r'원통|배치|작업|동작|목적지|\b(?:cylinder|arrangement|task|destination)\b', text, re.I)
                and re.search(r'옮|이동|쌓|풀|기억|복원|돌려|바꾸|바꿔|변경|'
                              r'\b(?:move|stack|unstack|remember|restore|change|switch)\b', text, re.I))


def _voice_output_is_question_or_quote(text):
    if re.search(r'["“”‘’]', text):
        return True
    if re.search(r'[?？]', text):
        # A question about stopping or silence is not itself an instruction.
        # Retain direct polite requests that happen to use question marks.
        polite = (re.search(r'(?:줘|줘요|주세요|주실래요|줄래요|주겠어요)\s*[?？]\s*$', text)
                  or re.match(r'\s*(?:can|could|would|will)\s+you\b', text, re.I))
        if (not polite or re.search(r'왜|어떻게|무슨|무엇|뭐|설명|뜻|'
                                    r'\b(?:why|what|how|explain|mean)\b', text, re.I)):
            return True
    return False


def _audio_voice_output(text, response, reference_context=None):
    """Use model intent only for a voice-only proposal; retain robot controls."""
    if _voice_output_is_question_or_quote(text):
        return None
    explicit = direct_voice_output_control(text)
    if explicit is not None:
        return explicit
    if _direct_robot_stop_for_audio(text):
        return None
    if (_clear_motion_stop(text, reference_context) or _mentions_pending_action(text)):
        return None
    output = response.get('voice_output', 'none')
    steps = response.get('steps')
    if (output in ('mute', 'unmute') and isinstance(steps, list) and steps
            and all(isinstance(step, dict) and step.get('op') in ('clarify', 'status') for step in steps)):
        return output
    return None


def _voice_only_proposal(text):
    return {'schema': 'task_program.v2', 'steps': [
        {'op': 'clarify', 'reason': 'question_only', 'question': None, 'evidence': text[:300]}]}


def _preserve_direct_motion_stop(text, response):
    """Keep an explicit robot stop when a voice-output label contradicts it."""
    if response.get('voice_output') in ('mute', 'unmute') and _direct_robot_stop_for_audio(text):
        return {'schema': 'task_program.v2', 'steps': [
            {'op': 'stop_motion', 'scope': 'active_task', 'evidence': text[:300]}]}
    return None


_PENDING_EDIT_FIELDS = ('source_slot', 'destination_slot', 'object_id',
                        'destination_object_id', 'count', 'height')


def pending_audio_model_schema(*, voice_output=True):
    schema = local_audio_model_schema(voice_output=voice_output)
    schema['properties']['pending_disposition'] = {
        'type': 'string', 'enum': ['edit', 'replace', 'unchanged', 'stop', 'uncertain']}
    schema['required'].append('pending_disposition')
    # Match the top-level response order shown in the pending-audio prompt.
    schema['properties']['steps'] = schema['properties'].pop('steps')
    # This is a pending-audio proposal, not an executable task-program step.
    # The ordinary model schema and every downstream program remain unchanged.
    schema['properties']['steps']['items']['oneOf'].append({
        'type': 'object', 'additionalProperties': False,
        'required': ['op', 'step_index', 'changes', 'evidence'],
        'properties': {
            'op': {'const': 'edit_step'},
            'step_index': {'type': 'integer', 'minimum': 0, 'maximum': 7},
            'changes': {
                'type': 'array', 'minItems': 1, 'maxItems': 6,
                'items': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['field', 'value'],
                    'properties': {
                        'field': {'enum': list(_PENDING_EDIT_FIELDS)},
                        'value': {'type': ['integer', 'null']},
                    },
                },
            },
            'evidence': {'type': 'string', 'minLength': 1, 'maxLength': 300},
        },
    })
    return schema


def audio_stop_flags(result):
    steps = result.program['steps']
    stop = steps[0]['op'] == 'stop_motion'
    return {'immediate_stop': stop and len(steps) == 1, 'stop_prefix': stop and len(steps) > 1}


def build_local_audio_messages(audio, reference_context=None, *, active_transfer_at_onset=False, language='auto'):
    import numpy as np
    from scipy.io import wavfile
    from .demo_voice import DemoAutoVoice

    samples = DemoAutoVoice._prepare(audio, language)
    wav = io.BytesIO()
    wavfile.write(wav, 16000, samples.astype(np.float32, copy=False))
    messages = build_local_model_messages('', reference_context, active_transfer_at_onset=active_transfer_at_onset)
    messages[0]['content'] += (
        '\nThis request contains the actual audio. Interpret the command directly from the recording. '
        'Also return transcript with the words you heard. Keep uncertain or missing words missing. '
        'If no speech is intelligible, return a clarification and an empty transcript.'
        + _VOICE_OUTPUT_INSTRUCTIONS)
    context = {'reference_context': v1.normalize_context(reference_context),
               'active_transfer_at_onset': active_transfer_at_onset, 'language': language}
    messages[1]['content'] = [
        {'type': 'text', 'text': json.dumps(context, ensure_ascii=False, separators=(',', ':'))},
        {'type': 'input_audio', 'input_audio': {'format': 'wav',
         'data': base64.b64encode(wav.getvalue()).decode('ascii')}}]
    return messages


def _pending_change_instructions():
    return (
        '\nThe previous request has NOT started. Its transcript and full proposed plan are context, '
        'not the new request. Interpret the new request in that context. Set pending_disposition '
        'to edit when it clearly changes one or more numeric fields of ONE existing move, stack, '
        'unstack or swap step, without adding, deleting, replacing or reordering steps. Return '
        'exactly one edit_step in steps: step_index is the zero-based index in the existing plan, '
        'changes is a list of {"field": "destination_slot", "value": 2}-style field/value '
        'pairs containing only the requested fields, and evidence quotes the new request. '
        'Editable fields are source_slot, destination_slot, object_id, destination_object_id, '
        'count and height. To switch a destination between a slot and a named object, include '
        'both fields, using null for the no-longer-applicable field. Do not copy unrelated steps '
        'into an edit_step response; the application preserves them. For example, when the prior '
        'steps are [remember, move 3 to 4], a correction to destination 2 is '
        '{"schema":"task_program.v2","transcript":"2번으로 바꿔 줘",'
        '"pending_disposition":"edit","steps":[{"op":"edit_step","step_index":1,'
        '"changes":[{"field":"destination_slot","value":2}],"evidence":"2번으로 바꿔 줘"}]}. '
        'If the speaker explicitly replaces the whole request, adds or deletes steps, changes a '
        'repeat, or the prior plan includes a repeat or control, do not use edit. '
        'Set it to replace for a clear replacement and return the entire corrected plan, preserving '
        'all unchanged steps when they are still requested. Set it to stop for an immediate cancellation or stop and return a '
        'stop_motion step. Set it to unchanged for a question or confidently identified background '
        'noise that does not change the request. Set it to uncertain if a correction or its meaning '
        'is unclear. Do not invent missing words. The transcript is only the new request.')


def build_pending_audio_messages(audio, reference_context, pending, *, language='auto'):
    """Keep the new recording separate and show the still unexecuted plan as context."""
    messages = build_local_audio_messages(audio, reference_context, language=language)
    messages[0]['content'] += _pending_change_instructions()
    context = json.loads(messages[1]['content'][0]['text'])
    context['unexecuted_request'] = copy.deepcopy(pending)
    messages[1]['content'][0]['text'] = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    return messages


def build_pending_text_messages(text, reference_context, pending):
    """Interpret typed follow-up text against the same unstarted plan as speech."""
    messages = build_local_model_messages(text, reference_context)
    messages[0]['content'] += _pending_change_instructions()
    messages[0]['content'] += ('\nThis request is typed text, not a recording. Return the supplied '
                               'transcript unchanged; only the supplied text is the new request.')
    context = json.loads(messages[1]['content'])
    context['unexecuted_request'] = copy.deepcopy(pending)
    messages[1]['content'] = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    return messages


def _apply_pending_step_edit(pending, edit, transcript):
    """Copy one unstarted plan and apply only the model's named field changes."""
    original = pending.get('program') if isinstance(pending, dict) else None
    steps = original.get('steps') if isinstance(original, dict) else None
    if (not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps)
            or any(step.get('op') in ('repeat', 'stop_repeat', 'stop_motion', 'retarget', 'continue')
                   for step in steps)):
        raise ValueError('변경할 이전 작업을 확인하지 못했습니다.')
    index = edit['step_index']
    if index >= len(steps):
        raise ValueError('변경할 작업 단계를 확인하지 못했습니다.')
    before = steps[index]
    if before.get('op') != 'task' or before.get('kind') not in ('move', 'stack', 'unstack', 'swap'):
        raise ValueError('변경할 이동 단계를 확인하지 못했습니다.')
    names = [item['field'] for item in edit['changes']]
    if len(names) != len(set(names)):
        raise ValueError('변경할 항목이 중복되었습니다.')

    updated = copy.deepcopy(steps)
    step = updated[index]
    for item in edit['changes']:
        name, value = item['field'], item['value']
        if name == 'destination_object_id':
            # Only the named-object variant carries this optional property.
            if value is None:
                step.pop(name, None)
            else:
                step[name] = value
        elif name in step:
            step[name] = value
        else:
            raise ValueError('이동 단계에 없는 항목은 변경할 수 없습니다.')

    # Preserve both sources when space permits. Long citations stay available
    # in the normalization record and the accumulated pending transcript.
    evidence = edit['evidence'] if edit['evidence'] in transcript else transcript
    evidence = evidence[:300]
    previous_evidence = before.get('evidence', '')
    available = 299 - len(evidence)
    step['evidence'] = (previous_evidence[:available] + '\n' + evidence
                        if available > 0 and previous_evidence else evidence)
    proposal = {'schema': original.get('schema'), 'steps': updated}
    record = {'kind': 'pending_step_edit', 'step_index': index,
              'before': copy.deepcopy(before), 'changes': copy.deepcopy(edit['changes']),
              'correction_transcript': transcript, 'after': copy.deepcopy(step)}
    return proposal, record


def validate_schema(program):
    v1._schema_check(program, canonical_schema())
    return program


_TRANSFER_BOUNDARY = re.compile(
    r'(?:이번|현재|지금)\s*(?:(?:원통|물체)\s*)?(?:한\s*개(?:의)?\s*)?이동'
    r'(?:이|가)?\s*(?:끝난\s*(?:뒤|후)|끝나면|완료(?:된)?\s*(?:뒤|후)|마친\s*(?:뒤|후))'
    r'|\bafter\s+(?:the\s+)?(?:current|this|ongoing)\s+'
    r'(?:(?:single|individual)\s+)?(?:(?:cylinder|object)\s+)?(?:transfer|move|movement)\b'
    r'|\b(?:when|once)\s+(?:the\s+)?(?:current|this|ongoing)\s+'
    r'(?:(?:single|individual)\s+)?(?:(?:cylinder|object)\s+)?(?:transfer|move|movement)\s+'
    r'(?:finishes|ends|completes|is\s+(?:done|finished|complete)|has\s+(?:finished|ended|completed))\b',
    re.IGNORECASE | re.ASCII)
_TASK_BOUNDARY = re.compile(
    r'(?:이번|현재|지금|이)\s*(?:작업|동작)(?:이|가)?\s*'
    r'(?:끝난\s*(?:뒤|후)|끝나면|완료(?:된)?\s*(?:뒤|후)|마친\s*(?:뒤|후))'
    r'|\bafter\s+(?:the\s+)?(?:current|this)\s+(?:task|action|step)\b'
    r'|\b(?:when|once)\s+(?:the\s+)?(?:current|this)\s+(?:task|action|step)\s+'
    r'(?:finishes|ends|completes|is\s+(?:done|finished|complete)|has\s+(?:finished|ended|completed))\b',
    re.IGNORECASE | re.ASCII)
_ROUND_BOUNDARY = re.compile(
    r'(?:이번|현재|지금)\s*(?:한\s*)?(?:바퀴|회차)(?:가|이)?\s*'
    r'(?:끝난\s*(?:뒤|후)|끝나면|완료(?:된)?\s*(?:뒤|후)|마친\s*(?:뒤|후))')
_REPEAT_WORD = re.compile(r'반복|되풀이|\b(?:repeat(?:ing)?|repetition)\b', re.IGNORECASE | re.ASCII)
_TRANSFER_STOP_VERB = re.compile(r'중단|종료|정지|멈추|멈춰|그만|끝내|\b(?:stop|end|cancel)\b',
                                  re.IGNORECASE | re.ASCII)


def _affirmed_boundaries(text, start, end, pattern):
    return [match for match in pattern.finditer(text[start:end])
                  if not v1._quoted(text, start + match.start(), start + match.end())
                  and not v1._negated_span(text, start + match.start(), start + match.end())
                  and not re.search(r'\b(?:not|never|except|instead\s+of|rather\s+than)\s+'
                                    r'(?:(?:only|just)\s+)?$', text[max(0, start + match.start() - 40):start + match.start()],
                                    re.IGNORECASE | re.ASCII)
                  and not re.match(r"\s+(?:(?:is|has|was|will|does|did)\s+(?:not|never)|"
                                   r"(?:isn|hasn|wasn|won|doesn|didn)['’]t)\s+"
                                   r'(?:finished|finish|ended|end|completed|complete|done)\b',
                                   text[start + match.end():], re.IGNORECASE | re.ASCII)]


def _unrepresented_repeat_condition(text, clause_left, clause_right, allowed_boundaries):
    # A completion-time clause belongs to the requested boundary; another
    # condition requires sensing or logic absent from this control schema.
    pattern = r'\b(?:if|unless|provided|assuming|when)\b|만약|만일|[가-힣]{2,}(?:면|으면)|[가-힣]+\s*때(?:에|만)?|경우(?:에)?'
    for match in re.finditer(pattern, text[clause_left:clause_right], re.IGNORECASE | re.ASCII):
        left, right = clause_left + match.start(), clause_left + match.end()
        if (v1._quoted(text, left, right) or
                any(a <= left and right <= b for a, b in allowed_boundaries)):
            continue
        if re.match(r'\s*(?:이라는|이란|라는|이라고|라고)\s*(?:말|단어|문구|표현|문장|설명)', text[right:]):
            continue
        return True
    return False


def _corrected_korean_repeat_stops(text, start, end, boundaries):
    """Resolve an elided repeat object after '중단하지 말고 … 중단'."""
    evidence = text[start:end]
    correction = re.compile(
        r'(?:반복|되풀이)(?:을|를|은|는|이|도|만)?[^;,.!?。？！]{0,50}?'
        r'(?P<negative>중단|종료|정지|멈추|멈춰|그만|끝내)'
        r'[가-힣]{0,5}지\s*말고')
    accepted = []
    for item in correction.finditer(evidence):
        negative_start, negative_end = start + item.start('negative'), start + item.end('negative')
        if (v1._quoted(text, start + item.start(), start + item.end()) or
                not v1._negated_action(text, negative_start, negative_end)):
            continue
        for boundary in boundaries:
            if boundary.start() < item.end():
                continue
            gap = evidence[item.end():boundary.start()]
            if re.search(r'[;,.!?。？！]|\b(?:but|then)\b', gap, re.IGNORECASE | re.ASCII):
                continue
            for verb in _TRANSFER_STOP_VERB.finditer(evidence, boundary.end()):
                gap = evidence[boundary.end():verb.start()]
                if re.search(r'[;,.!?。？！]|말고|동작|로봇|팔(?:을|이|은|도)|\b(?:but|then)\b',
                             gap, re.IGNORECASE | re.ASCII):
                    break
                left, right = start + verb.start(), start + verb.end()
                if _explicit_span(text, left, right, verb.group()):
                    accepted.append((left, right))
                break
    return accepted


def _boundary_preceded_by_repeat_control(text, start, boundary, action_start):
    """Do not attach a boundary following one stop to a later stop clause."""
    left, right = start + boundary.start(), start + boundary.end()
    if action_start < right:
        return False
    # Korean often places a fresh repeat object between the new completion
    # boundary and its verb without punctuation: "… 이동이 끝난 뒤에 반복을
    # 중단해". That explicitly starts the later clause even when a previous
    # stop was negated. Do not attach its boundary to the previous clause.
    if re.search(r'(?:반복|되풀이)(?:을|를|은|는|이|도|만)?\s*'
                 r'(?:(?:바로|즉시|이제|그만)\s*)?$', text[right:action_start]):
        return False
    clause_left = max((text.rfind(mark, 0, left) for mark in '.,;!?。？！'), default=-1) + 1
    prefix = text[clause_left:left]
    pattern = (r'\b(?:stop|end|cancel)\s+(?:the\s+)?(?:repeat(?:ing)?|repetition)\b|'
               r'(?:반복|되풀이)(?:을|를|은|는|이|도|만)?[^;,.!?。？！]{0,45}?'
               r'(?:중단|종료|정지|멈추|멈춰|그만|끝내)')
    for prior in re.finditer(pattern, prefix, re.IGNORECASE | re.ASCII):
        if not v1._quoted(text, clause_left + prior.start(), clause_left + prior.end()):
            return True
    for prior in v1._REPEAT_NEGATION.finditer(text, clause_left, left):
        if v1._has_global_repeat_stop(text, prior.start(), prior.end()):
            return True
    return False


def _negative_repeat_commands(text, start, end, boundaries):
    """'Do not repeat' stops repetition; 'do not stop repeating' does not."""
    accepted = []
    for match in v1._REPEAT_NEGATION.finditer(text, start, end):
        left, right = match.span()
        if (not v1._has_global_repeat_stop(text, left, right) or
                re.search(r"\b(?:do\s+not|don['’]?t|never)\s+$", text[max(0, left - 25):left],
                          re.IGNORECASE | re.ASCII) or
                re.match(r'\s*지\s*(?:마|말|않)', text[right:])):
            continue
        for boundary in boundaries:
            bleft, bright = start + boundary.start(), start + boundary.end()
            if _boundary_preceded_by_repeat_control(text, start, boundary, left):
                continue
            gap = text[min(right, bright):max(left, bleft)]
            if (re.search(r'[;.!?。？！]|아니라|\b(?:but|instead|then)\b', gap, re.IGNORECASE | re.ASCII)
                    or _TRANSFER_STOP_VERB.search(gap) or v1._REPEAT_NEGATION.search(gap)):
                continue
            accepted.append((left, right))
            break
    return accepted


def _repeat_stop_verbs(text, start, end, pattern):
    """Find affirmative repeat stops bound to the stated completion boundary."""
    evidence = text[start:end]
    boundaries = _affirmed_boundaries(text, start, end, pattern)
    if not boundaries or not _REPEAT_WORD.search(evidence):
        return []
    # These instructions require a condition that the program cannot express.
    clause_left = max((text.rfind(mark, 0, start) for mark in '.!?;。？！'), default=-1) + 1
    clause_right = min((pos for mark in '.!?;。？！' if (pos := text.find(mark, end)) >= 0),
                       default=len(text))
    # A rejected alternative completion time remains a temporal clause, not
    # an external sensor condition. Its polarity is handled separately.
    allowed = [(start + item.start(), start + item.end())
               for boundary_pattern in (_TRANSFER_BOUNDARY, _TASK_BOUNDARY, _ROUND_BOUNDARY)
               for item in _affirmed_boundaries(text, start, end, boundary_pattern)]
    if _unrepresented_repeat_condition(text, clause_left, clause_right, allowed):
        return []
    repeats = list(_REPEAT_WORD.finditer(evidence))
    accepted = []
    for verb in _TRANSFER_STOP_VERB.finditer(evidence):
        verb_start, verb_end = start + verb.start(), start + verb.end()
        if (re.search(r'\b(?:said|says|asked|asks|told|wrote|wants|wanted|requested|suggested|mentioned|quoted)\b'
                      r'[^.!?;。？！]{0,80}$', text[clause_left:verb_start], re.IGNORECASE | re.ASCII)
                or re.match(r'[가-힣]{0,3}(?:라고|라는|다며|다고|자는)', text[verb_end:])):
            continue
        repeat = min(repeats, key=lambda item: abs(item.start() - verb.start()))
        boundary = min(boundaries, key=lambda item: abs(item.start() - verb.start()))
        if _boundary_preceded_by_repeat_control(text, start, boundary, verb_start):
            continue
        boundary_gap = evidence[min(verb.end(), boundary.end()):max(verb.start(), boundary.start())]
        if re.search(r'[;.!?。？！]|말고|아니라|그리고|그다음|\b(?:but|instead|then)\b',
                     boundary_gap, re.IGNORECASE | re.ASCII) or _TRANSFER_STOP_VERB.search(boundary_gap):
            continue
        between = evidence[min(verb.end(), repeat.end()):max(verb.start(), repeat.start())]
        if (re.search(r'[;,.!?。？！]|그리고|그다음|(?:하고|고)\s+|\b(?:and|then|but)\b',
                      between, re.IGNORECASE | re.ASCII) or _TRANSFER_STOP_VERB.search(between)):
            continue
        if re.fullmatch(r'\b(?:stop|end|cancel)\b', verb.group(), re.IGNORECASE | re.ASCII):
            # A nearby "repeating" does not make "stop the robot while
            # repeating" a repeat stop; bind the verb to its actual object.
            if (verb.start() > repeat.start() or
                    re.fullmatch(r'\s+(?:the\s+)?', between, re.IGNORECASE | re.ASCII) is None):
                continue
        elif verb.start() > repeat.start():
            # Remove the actual boundary from the intervening text before
            # looking for a competing object of the stop command.
            begin, finish = repeat.end(), verb.start()
            relevant = evidence[begin:finish]
            if begin <= boundary.start() and boundary.end() <= finish:
                relevant = evidence[begin:boundary.start()] + evidence[boundary.end():finish]
            if re.search(r'하며|하면서|중에|동안|하는|동작|로봇|작업|팔(?:을|이|은|도)', relevant):
                continue
        if _explicit_span(text, verb_start, verb_end, verb.group()):
            accepted.append((verb_start, verb_end))
    return (accepted + _corrected_korean_repeat_stops(text, start, end, boundaries)
            + _negative_repeat_commands(text, start, end, boundaries))


def _transfer_stop_verbs(text, start, end):
    return _repeat_stop_verbs(text, start, end, _TRANSFER_BOUNDARY)


_KO_STOP_STATUS_SUFFIX = re.compile(
    r'(?:춰야|추어야|해야|하여야)\s*하는\s*지|'
    r'(?:췄|추었|추고\s*있|춰\s*있|추|춘|출|됐|되었|하고\s*있|했|하였|하|된|한|할|될)'
    r'(?:는지|은지|는\s*지|지\s*(?:알려|확인|봐)|나요|습니까|는가|'
    r'다고|다는|어(?:요)?(?=\s*[.!?。？！]|\s*$)|다(?=\s*[.!?。？！]|\s*$)|'
    r'는\s*(?:방법|법|것|게)|(?:을|는)?\s*때|'
    r'\s*(?:상태|상황|여부|시점|시간|건지|것인지))')
_EN_STOP_STATUS_QUESTION = re.compile(
    r'\b(?:check|tell|ask|show|report|explain|know|wonder)\b[^.!?;,]{0,65}'
    r'\b(?:whether|if|when|how|that)\b[^.!?;,]{0,65}$', re.IGNORECASE | re.ASCII)
_EN_STOP_REPORT_VERB = (
    r'(?:said|say(?:s|ing)?|shout(?:s|ed|ing)?|yell(?:s|ed|ing)?|scream(?:s|ed|ing)?|'
    r'call(?:s|ed|ing)?\s+out|(?:told|asked)\s+(?:me|us|you|him|her|them|the\s+robot)\s+to)'
    r'(?:\s+(?:out|aloud|the\s+(?:words?|phrase|command)))?')
_EN_REPORTED_STOP_PREFIX = re.compile(
    r'(?:\b' + _EN_STOP_REPORT_VERB + r'(?:\s*:\s*|\s+)|'
    r'(?:^|[.!?;。？！])\s*(?:(?:i|we)\s+(?:hear|heard)\s+)?'
    r'(?:someone|somebody|he|she|they|him|her|them|(?:a|an|the|my|our)\s+[a-z-]+|'
    r'(?!(?:a|an|the|my|our)\b)(?-i:[A-Z][a-z]+))\s+'
    + _EN_STOP_REPORT_VERB + r'\s*,\s*)(?:to\s+)?(?:please\s+)?$',
    re.IGNORECASE | re.ASCII)
_KO_REPORTED_STOP_SUFFIX = re.compile(
    r'(?:추|춰|추어|하|해)?\s*(?:달|주)?(?:라고|라는|자고)\s*'
    r'(?:말|외치|외쳤|소리치|소리쳤|했|하던|하더|들었|요청)')


def _korean_dependent_action(text, start, end):
    """Bind Korean prohibitions and hypothetical endings to their own verb."""
    if re.search(r'[가-힣]', text[start:end]) is None:
        return False
    return re.match(
        r'(?:[가-힣]{0,5}(?:(?:서는|선)\s*(?:안\s*(?:돼|되|됨)|곤란|위험)'
        r'|지(?:는|도)?\s*(?:마|말|않)|기\s*싫|고\s*싶지\s*않'
        r'|(?:으?면|거든)(?!서)(?=\s|[,，.!?。？！]|[가-힣]|$))'
        r'|[가-힣]{0,5}\s+(?:주|해)면\s*(?:곤란|위험|안\s*(?:돼|되))'
        r'|[가-힣]{0,5}\s*(?:주|드리|놓|두|보|버리|내)(?:시)?지(?:는|도)?\s*(?:마|말|않)'
        r'|[가-힣]{0,5}\s*(?:달라는\s*)?(?:말|뜻|의도)(?:이|은|는)?\s*아니)',
        text[end:]) is not None


def _reject_dependent_action(step, text, start, end):
    """Require a proposed action to contain a non-dependent action verb."""
    op = step['op']
    if op == 'task':
        pattern = v1._TASK_WORDS.get(step['kind'])
    else:
        pattern = {
            'stop_motion': r'멈|정지|중단|그만|\b(?:stop|halt|cancel)\b',
            'stop_repeat': r'멈|정지|중단|그만|끝내|종료|\b(?:stop|end|cancel)\b',
            'return': r'돌아가|돌아오|돌려|복귀|복원|원위치|\b(?:return|restore)\b',
            'repeat': r'반복|되풀이|\b(?:repeat|repeating)\b',
            'continue': r'계속|이어|유지|\b(?:continue|resume)\b',
        }.get(op)
    if not pattern:
        return
    verbs = list(re.finditer(pattern, text[start:end], re.IGNORECASE | re.ASCII))
    if verbs and all(_korean_dependent_action(text, start + verb.start(), start + verb.end())
                     for verb in verbs):
        v1._fail('unresolved_reference', 'a prohibition or hypothetical does not authorize this action')


def _stop_span_is_mentioned(text, start, end):
    """Bind a status query or reported speech to this occurrence of the stop verb."""
    if _KO_STOP_STATUS_SUFFIX.match(text, end) or _KO_REPORTED_STOP_SUFFIX.match(text, end):
        return True
    left = max((text.rfind(mark, 0, start) for mark in '.!?;,。？！'), default=-1) + 1
    return bool(_EN_STOP_STATUS_QUESTION.search(text, left, start)
                or _EN_REPORTED_STOP_PREFIX.search(text, 0, start))


_REPORT_SENTENCE_END = re.compile(r'[.!?;。？！]')
_EN_REPORT_HEAD = re.compile(
    r'\b(?:said|says|told|tells|asked|asks|instructed|instructs|ordered|orders|'
    r'advised|advises|urged|urges|requested|requests|suggested|suggests|'
    r'reminded|reminds|wrote|writes|shouted|shouts|yelled|yells|wanted|wants|want)\b', re.I | re.ASCII)
_EN_REPORT_COMPLEMENT = re.compile(
    r'\s+(?:[a-z]+[\s-]+){0,6}?(?:to|that)\b|\s*[:,]?\s*(?:please\s+)?'
    r'(?:move|transfer|put|place|stack|unstack|swap|exchange|remember|save|'
    r'register|arrange|stop|halt|repeat|resume|continue|return|restore)\b', re.I | re.ASCII)
_EN_NEW_REQUEST = re.compile(
    r'\bbut\s+(?:now\s+)?i\s+(?:want|ask|am\s+asking)\s+you\s+to\b', re.I | re.ASCII)
_EN_WISH_SUBJECT = re.compile(r'\s*(?:i|we)\b(?P<modifiers>(?:[\s,]+[a-z]+)*[\s,]*)', re.I | re.ASCII)
_EN_WISH_UNCERTAIN = re.compile(
    r'\b(?:would|could|might|may|should|shall|will|can|must|perhaps|maybe|possibly|potentially|'
    r'presumably|supposedly|allegedly|apparently|almost|nearly|barely|hardly|scarcely|rarely|'
    r'sometimes|occasionally|formerly|previously|once|used|if|unless|when|whether|that|'
    r'i|we|you|he|she|they|it|me|us|him|her|them)\b', re.I | re.ASCII)
_EN_WISH_MODIFIERS = re.compile(
    r'(?:(?:[a-z]+ly|just|now|still|do|did|also|both|too|myself|ourselves|'
    r'(?:very|so|quite|rather|most)(?:\s+much)?|as\s+well)(?:\s+|$))*', re.I | re.ASCII)
_EN_ADDRESSED_PREDICATE = re.compile(r'\s+you\s+to\s+(?:(?:please|first|now|[a-z]+ly)\s+)*', re.I | re.ASCII)
_POSSIBLE_DIRECT_PREFIX = re.compile(
    '|'.join(v1._TASK_WORDS.values()) + r'|멈|정지|중단|반복|재개|복귀|복원|'
    r'\b(?:stop|halt|repeat|resume|continue|return|restore)\b',
    re.I | re.ASCII)
_KO_OPEN_SEQUENCE = re.compile(r'(?:고|며|면서|서|자마자|(?:한|긴|은|된)\s*(?:뒤|후|다음))\s*$')
_KO_REPORT_COMPLEMENT = re.compile(
    r'(?:라|자|다|냐)(?:고|며|는)\s*(?:(?:말|부탁|요청|지시|명령|얘기|이야기|전달)(?:을|를)?\s*)?'
    r'(?:했|하였|하더|한대|한다|하는데|하셨|말했|말하|말씀하|전했|전하|들었|들려|부탁했|요청했|그러더|그랬|그러셨)'
    r'[가-힣]*|(?:달라|으라|하라|기라)(?:더라|더군)[가-힣]*|'
    r'(?P<short>[가-힣]+?)(?:래(?:요|서|도|지만|더라)?|랬(?:어(?:요)?|대(?:요)?|다|지만))(?=$|[\s,，.!?;。？！])')


def _possible_korean_direct_prefix(prefix):
    # A finished instruction before a comma can belong to the current
    # speaker. A coordinating verb ending ("기억하고, ...") stays open.
    pieces = re.split(r'[,，]', prefix)
    if any(_POSSIBLE_DIRECT_PREFIX.search(piece) and not _KO_OPEN_SEQUENCE.search(piece)
           for piece in pieces[:-1]):
        return True
    # Unpunctuated speech can also start with a complete imperative before
    # the report's own predicate. Do not drop the earlier requirement.
    return any(_KOREAN_IMPERATIVE.fullmatch(word.group()) and
               _POSSIBLE_DIRECT_PREFIX.search(prefix, word.end())
               for word in re.finditer(r'[가-힣]+', prefix))


def _direct_english_wish(sentence, head):
    """Require a current first-person matrix subject and the addressed actor."""
    addressed = _EN_ADDRESSED_PREDICATE.match(sentence, head.end())
    if (head.group().lower() != 'want' or not addressed or
            not _POSSIBLE_DIRECT_PREFIX.match(sentence, addressed.end())):
        return False
    # A preceding subject inside a report, question or conditional is not the
    # current speaker. An explicit "but I want you to" can start a new request.
    clause = re.split(r'\bbut\b', sentence[:head.start()], flags=re.I | re.ASCII)[-1]
    subject = _EN_WISH_SUBJECT.fullmatch(clause)
    if not subject:
        return False
    modifiers = ' '.join(re.findall(r'[a-z]+', subject['modifiers'], re.I | re.ASCII))
    if v1._EN_NEGATION_CUE.search(modifiers) or _EN_WISH_UNCERTAIN.search(modifiers):
        return False
    # Additive particles, degree phrases and ordinary adverbs can modify the
    # matrix predicate. Free content cannot: it may introduce another subject
    # or a noncommittal attitude. Unknown word sequences stay unresolved.
    return _EN_WISH_MODIFIERS.fullmatch(modifiers) is not None


def _direct_speech_view(text):
    """Exclude unquoted reported complements at sentence scope, keeping offsets.

    Coordinated verbs can share one report head (English) or one final
    quotation ending (Korean). A comma or 'then' inside that complement
    cannot grant the second verb its own authority.
    """
    if not _EN_REPORT_HEAD.search(text) and not _KO_REPORT_COMPLEMENT.search(text):
        return text, False
    quotes = v1._quote_spans(text)
    # Check grammar outside paired quotations. A visible marker prevents a
    # report verb before a quotation from matching words after it.
    visible = list(text)
    for left, right in quotes:
        visible[left:right] = '\ufffc' * (right - left)
    outside_quotes = ''.join(visible)
    output = list(text)
    unclear_boundary = False
    left = 0
    for right in [m.start() for m in _REPORT_SENTENCE_END.finditer(outside_quotes)] + [len(text)]:
        sentence = outside_quotes[left:right]
        report_end = 0
        for head in _EN_REPORT_HEAD.finditer(sentence):
            prior = sentence[:head.start()]
            complement = _EN_REPORT_COMPLEMENT.match(sentence, head.end())
            if head.group().lower() in ('want', 'wants', 'wanted'):
                if _direct_english_wish(sentence, head):
                    continue
                # An unresolved actor may have a longer noun phrase. Do not
                # fall through to direct execution because a fixed word budget
                # did not reach its infinitive.
                complement = re.search(r'\bto\b', sentence[head.end():], re.I | re.ASCII)
            # "As I said, stop now" contains the speaker's own new request.
            if re.search(r'\bas\s+(?:i|we|you|he|she|they)\s+(?:just\s+)?$', prior,
                         re.I | re.ASCII) and re.match(r'\s*,', sentence[head.end():]):
                continue
            if complement:
                if _POSSIBLE_DIRECT_PREFIX.search(prior):
                    unclear_boundary = True
                new_request = _EN_NEW_REQUEST.search(sentence, head.end())
                # An adversative alone can be part of the reported request.
                # Only an overt request by the current speaker ends its scope.
                if new_request and not _EN_CONDITIONAL.search(sentence[:new_request.start()]):
                    report_end = max(report_end, new_request.end())
                else:
                    report_end = len(sentence)
        for ending in _KO_REPORT_COMPLEMENT.finditer(sentence):
            # Explicit quotation marks already isolate the quoted material.
            if re.search(r'\ufffc\s*(?:이)?$', sentence[:ending.start()]):
                continue
            short = ending.group('short')
            if short is not None:
                # -(으)래 contracts an imperative quotation. Contrast the
                # prospective -(으)ㄹ래 ("줄래") and nouns such as "원래".
                final_consonant = (ord(short[-1]) - ord('가')) % 28
                if not short.endswith('달') and (final_consonant == 8 or not _POSSIBLE_DIRECT_PREFIX.search(short)):
                    continue
            predicate_end = ending.end('short') if short is not None else ending.start()
            if _possible_korean_direct_prefix(sentence[:predicate_end]):
                unclear_boundary = True
            # Korean marks the contrast on the reporting predicate itself.
            # A conditional ending cannot license the rest of its sentence.
            boundary = ending.end() if ending.group().endswith('지만') else len(sentence)
            report_end = max(report_end, boundary)
        if report_end:
            output[left:left + report_end] = ' ' * report_end
        left = right + 1
    return ''.join(output), unclear_boundary


def _current_stop_spans(text):
    """Do not turn past, descriptive or queried stop states into new stops."""
    text, _ = _direct_speech_view(text)
    repeat_stop_starts = {start for start, _ in (_transfer_stop_verbs(text, 0, len(text)) +
                          _repeat_stop_verbs(text, 0, len(text), _TASK_BOUNDARY) +
                          _repeat_stop_verbs(text, 0, len(text), _ROUND_BOUNDARY))}
    return [(start, end) for start, end in v1._affirmed_motion_stops(text)
            if start not in repeat_stop_starts and not _stop_span_is_mentioned(text, start, end) and
            not _korean_dependent_action(text, start, end) and
            re.match(r'(?:한|했던|됐던|된|췄던|춘)\s*(?:직전|이전|지난|방금|작업|동작)', text[end:]) is None]


_KOREAN_HEIGHT = re.compile(
    r'(\d+|[영공일이삼사오육칠팔구십백천만]+|'
    r'(?:열|스물|서른|마흔|쉰|예순|일흔|여든|아흔)(?:하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉)?|'
    r'스무|다섯|여섯|일곱|여덟|아홉|하나|둘|셋|넷|한|두|세|네)\s*층')
_EN_HEIGHT_WORD = (r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
                   r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
                   r'twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|'
                   r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)')
_EN_HEIGHT_NUMBER = r'(\d+|' + _EN_HEIGHT_WORD + r'(?:[-\u2010-\u2015\s]+(?:and\s+)?' + _EN_HEIGHT_WORD + r')*)'
_EN_HEIGHT_UNIT = r'(?:levels?|layers?|floors?|stor(?:y|ies)|storeys?|tiers?)'
_EN_HEIGHT_BEFORE = re.compile(r'\b(?:height|level|floor|layer|story|storey|tier)\s*'
                               r'(?:number\s*|no\.?\s*|#\s*)?' + _EN_HEIGHT_NUMBER +
                               r'(?:st|nd|rd|th)?\b', re.IGNORECASE | re.ASCII)
_EN_HEIGHT_AFTER = re.compile(r'\b' + _EN_HEIGHT_NUMBER + r'(?:st|nd|rd|th)?'
                              r'(?:\s*[-\u2010-\u2015]\s*|\s+)' + _EN_HEIGHT_UNIT + r'\b',
                              re.IGNORECASE | re.ASCII)
_EN_HEIGHT_HIGH = re.compile(r'\b' + _EN_HEIGHT_NUMBER + r'(?:\s*[-\u2010-\u2015]\s*|\s+)'
                             r'(?:(?:cylinders?|objects?|pieces?)\s+)?(?:high|tall)\b',
                             re.IGNORECASE | re.ASCII)
_EN_HEIGHT_VALUES = {'zero': 0, 'one': 1, 'two': 2, 'first': 1, 'second': 2}


def _excluded_height_mention(text, match):
    """Ignore explicitly rejected heights; do not invent a target."""
    start, end = match.span()
    if v1._negated_span(text, start, end):
        return True
    for quoted in re.finditer(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<![A-Za-z])\'[^\']*\'', text):
        if quoted.start() <= start and end <= quoted.end():
            start, end = quoted.span()
            if v1._negated_span(text, start, end):
                return True
            break
    prefix = text[max(0, start - 60):start]
    suffix = text[end:min(len(text), end + 50)]
    if re.match(r'\s*(?:이|가|은|는|을|를)?\s*대신', suffix):
        return True
    if re.search(r"\b(?:not|except|instead\s+of|rather\s+than|other\s+than)\s+"
                 r"(?:(?:a|an|the)\s+)?$", prefix, re.IGNORECASE | re.ASCII):
        return True
    if re.search(r"\b(?:do\s+not|don['’]?t|never)\s+(?:make|build|create|form|stack)\s+"
                 r"(?:(?:a|an|the)\s+)?$", prefix, re.IGNORECASE | re.ASCII):
        return True
    return re.match(r'\s*(?:으로|로|을|를)?\s*(?:쌓|만들|올리|놓)[가-힣]{0,8}'
                    r'(?:지\s*(?:마|말|않)|(?:으)?면\s*안\s*(?:돼|되))', suffix) is not None


def _validate_explicit_height_limit(text):
    """Reject an affirmed height over two anywhere in the original request."""
    for match in _KOREAN_HEIGHT.finditer(text):
        token = match.group(1)
        value = int(token) if token.isdecimal() else v1.common._NUMBERS.get(token, 3)
        if value > 2 and not _excluded_height_mention(text, match):
            v1._fail('unsupported_action', 'the request explicitly exceeds the supported two-level height')
    for pattern in (_EN_HEIGHT_BEFORE, _EN_HEIGHT_AFTER, _EN_HEIGHT_HIGH):
        for match in pattern.finditer(text):
            token = match.group(1).lower()
            value = int(token) if token.isdecimal() else _EN_HEIGHT_VALUES.get(token, 3)
            if value > 2 and not _excluded_height_mention(text, match):
                v1._fail('unsupported_action', 'the request explicitly exceeds the supported two-level height')


def _repeat_reference_negated(text, start, end):
    """Bind a nearby repeat verb to this reference's contrast clause."""
    if v1._negated_span(text, start, end):
        return True
    boundaries = list(re.finditer(r'[;.!?。？！]|말고|아니고|\b(?:but|instead)\b', text, re.IGNORECASE | re.ASCII))
    left = max((match.end() for match in boundaries if match.end() <= start), default=0)
    right = min((match.start() for match in boundaries if match.start() >= end), default=len(text))
    verbs = list(re.finditer(r'반복|되풀이|\b(?:repeat|repeating|again)\b', text[left:right], re.IGNORECASE | re.ASCII))
    return bool(verbs) and all(v1._negated_action(text, left + verb.start(), left + verb.end()) for verb in verbs)


def _repeat_reference_roles(text):
    positive, negative = set(), set()
    for match in v1._TASK_REFERENCE.finditer(text):
        word = match.group(0).lower()
        if match.group('current') or match.group('en_current'):
            role = 'active_task'
        else:
            role = 'previous_program' if ('프로그램' in word or 'program' in word) else 'previous_task'
        (negative if _repeat_reference_negated(text, match.start(), match.end()) else positive).add(role)
    return positive, negative


def _repeat_reference_clause(text, reference, affirmative_verbs):
    """Locate the repeat clause and whether the proposed reference occurs in it."""
    boundaries = list(re.finditer(r'[;.!?。？！]|말고|아니고|\b(?:but|instead)\b',
                                  text, re.IGNORECASE | re.ASCII))
    occurrences = [match.start() for match in re.finditer(re.escape(reference), text)]
    clauses = []
    for position in affirmative_verbs:
        left = max((match.end() for match in boundaries if match.end() <= position), default=0)
        right = min((match.start() for match in boundaries if match.start() >= position), default=len(text))
        if (left, right) not in clauses:
            clauses.append((left, right))
        for ref_start in occurrences:
            if (left <= ref_start < right
                    and not _repeat_reference_negated(text, ref_start, ref_start + len(reference))):
                return left, right, ref_start
    if len(clauses) == 1:
        return *clauses[0], None
    v1._fail('unresolved_reference', 'repeat reference is not associated with one affirmative repeat clause')


def _mask_quoted(fragment):
    """Remove quoted contents without changing offsets into the transcript."""
    chars = list(fragment)
    for left, right in v1._quote_spans(fragment):
        chars[left:right] = ' ' * (right - left)
    return ''.join(chars)


def _repeat_clause_quantities(text, clause, offset):
    """Read repetition numbers from one clause, excluding numbered slots."""
    clause = _mask_quoted(clause)
    mentioned = set()
    for match in v1.common._NUMBER_UNIT.finditer(clause):
        if not match.group(2).startswith('번'):
            continue
        left, right = offset + match.start(), offset + match.end()
        if (v1._quoted(text, left, right) or v1._negated_span(text, left, right) or
                re.match(r'\s*(?:자리|위치|에서|으로|로|에(?:는|서)?|의|원통|물체|더미|덤이|탑)', clause[match.end():])):
            continue
        token = match.group(1)
        mentioned.add(int(token) if token.isdecimal() else v1.common._NUMBERS[token])
    # Keep a negated count from supplying either a value or the "more" mode.
    # Offsets refer to the original utterance, even after quoted text is masked.
    mode_clause = list(clause)
    for match in _EN_REPEAT_QUANTITY.finditer(clause):
        left, right = offset + match.start(), offset + match.end()
        negated = (v1._quoted(text, left, right) or v1._negated_span(text, left, right) or
                   _NEGATED_QUANTITY_PREFIX.search(text[max(0, left - 60):left]))
        if negated:
            mode_clause[match.start():match.end()] = ' ' * (match.end() - match.start())
            continue
        token = match.group('number')
        mentioned.add(v1.bilingual._value(token) if token else
                      (1 if match.group('short').lower() == 'once' else 2))
    has_more = v1._contains_any(''.join(mode_clause), r'더(?!미)|추가|\b(?:more|additional)\b')
    continuous = v1._contains_any(clause, r'계속|끊임|무한|\b(?:keep|continuous(?:ly)?|indefinitely)\b')
    return mentioned, has_more, continuous


def _validate_repeat(step, text, context, index, steps):
    target, count, evidence = step['target'], step['count'], step['evidence']
    if target['reference'] not in text or not v1._contains_any(evidence, r'반복|되풀이|\b(?:repeat|repeating|again)\b'):
        v1._fail('ungrounded_value', 'repeat reference is not supported by evidence')
    start = text.find(evidence)
    verbs = list(re.finditer(r'반복|되풀이|\b(?:repeat|repeating|again)\b', evidence, re.IGNORECASE | re.ASCII))
    affirmative_verbs = [start + verb.start() for verb in verbs
                         if not v1._quoted(text, start + verb.start(), start + verb.end())
                         and not v1._negated_action(text, start + verb.start(), start + verb.end())
                         and not v1._question_only(text, start + verb.start(), start + verb.end())]
    if not affirmative_verbs:
        v1._fail('unsupported_action', 'repeat is negated or questioned in its clause')
    kind, indices = target['kind'], target['step_indices']
    if kind in ('active_task', 'previous_task'):
        reference_starts = [match.start() for match in re.finditer(re.escape(target['reference']), text)]
        if all(_repeat_reference_negated(text, ref_start, ref_start + len(target['reference']))
               for ref_start in reference_starts):
            v1._fail('unsupported_action', 'the selected repeat reference is negated')
        left, right, local_reference = _repeat_reference_clause(text, target['reference'], affirmative_verbs)
        selected, _ = _repeat_reference_roles(target['reference'])
        if selected and kind not in selected:
            v1._fail('unresolved_reference', 'reference points to a different task')
        if len(selected) > 1:
            v1._fail('unresolved_reference', 'repeat reference names multiple tasks')
        affirmative, _ = _repeat_reference_roles(text[left:right])
        _, negated = _repeat_reference_roles(text)
        if local_reference is not None and _repeat_reference_negated(
                text, local_reference, local_reference + len(target['reference'])):
            v1._fail('unsupported_action', 'the selected repeat reference is negated')
        if kind in negated and kind not in affirmative:
            v1._fail('unsupported_action', 'the selected repeat target is negated')
        previous = context['previous_task']
        candidates = set()
        if context['execution_state'] == 'running':
            candidates.add('active_task')
        if previous is not None and previous['status'] == 'unique':
            candidates.add('previous_task')
        candidates -= negated - affirmative
        # A name in another clause is not evidence that the deictic repeat
        # selected it. It can still resolve when context leaves one candidate.
        local_selected = selected if local_reference is not None else set()
        roles = local_selected or (affirmative & {'active_task', 'previous_task'})
        if roles:
            if kind not in roles:
                v1._fail('unresolved_reference', 'repeat target differs from the supported reference')
            candidates &= roles
        if len(candidates) > 1:
            v1._fail('unresolved_reference', 'current and previous tasks are both plausible repeat targets')
        if candidates and kind not in candidates:
            v1._fail('unresolved_reference', 'repeat target differs from the supported reference')
    elif kind == 'previous_program':
        v1._validate_task_reference(kind, target['reference'], text)
        ref_start = text.find(target['reference'])
        if _repeat_reference_negated(text, ref_start, ref_start + len(target['reference'])):
            v1._fail('unsupported_action', 'the selected repeat reference is negated')
        affirmative, negated = _repeat_reference_roles(text)
        if kind in negated and kind not in affirmative:
            v1._fail('unsupported_action', 'the selected repeat target is negated')
        if affirmative and kind not in affirmative:
            v1._fail('unresolved_reference', 'repeat target differs from the affirmed reference')
    if kind == 'prior_steps':
        if indices != sorted(indices) or any(i >= index or steps[i]['op'] not in ('task', 'return') for i in indices):
            v1._fail('invalid_repeat_reference', 'repeat must reference earlier task or return steps in order')
    if kind in ('previous_task', 'previous_program'):
        ref = context[kind]
        if ref is None or ref['status'] != 'unique':
            v1._fail('unresolved_reference', 'previous reference is not unique and current')
    if kind == 'active_task' and context['execution_state'] != 'running':
        v1._fail('unresolved_reference', 'repeat needs the currently running task')
    if count['mode'] == 'continuous' and not v1._contains_any(evidence, r'계속|끊임|무한|\b(?:keep|continuous(?:ly)?|indefinitely)\b'):
        v1._fail('ungrounded_value', 'continuous repetition was not stated')
    if count['mode'] in ('total', 'additional'):
        # Evidence can cover the whole utterance. Slot ordinals in an earlier
        # move are not a repeat count, even when they occur in that evidence.
        clauses = list(dict.fromkeys(_repeat_clause(text, position, position + 1)
                                     for position in affirmative_verbs))
        if len(clauses) > 1:
            bound = []
            for clause, offset in clauses:
                for reference in re.finditer(re.escape(target['reference']), clause):
                    left, right = offset + reference.start(), offset + reference.end()
                    if not v1._quoted(text, left, right) and not _repeat_reference_negated(text, left, right):
                        bound.append((clause, offset))
                        break
            if len(bound) != 1:
                v1._fail('unresolved_reference', 'repeat count is ambiguous across repeat clauses')
            clauses = bound
        mentioned, has_more = set(), False
        for clause, offset in clauses:
            values, additional, _ = _repeat_clause_quantities(text, clause, offset)
            mentioned.update(values)
            has_more |= additional
        if len(mentioned) != 1 or count['value'] not in mentioned:
            v1._fail('ungrounded_value', 'repeat count was not stated')
        if (count['mode'] == 'additional') != has_more:
            v1._fail('invalid_repeat_count', 'total and additional counts disagree with the transcript')


def _validate_previous_continue(step, text, context, start, end):
    target, evidence = step['target'], step['evidence']
    if target['reference'] not in evidence or not v1._contains_any(evidence, r'이어|계속|재개|\b(?:continue|resume|carry\s+on)\b'):
        v1._fail('ungrounded_value', 'previous continuation reference is not supported by evidence')
    if not v1._affirmed_control(text, start, end,
                                r'이어|계속|재개|\b(?:continue|resume|carry\s+on)\b',
                                target['reference'], target_kind=target['kind']):
        v1._fail('unsupported_action', 'previous continuation is negated or questioned')
    v1._validate_task_reference('previous_task', target['reference'], text)
    ref = context['previous_task']
    if ref is None or ref['status'] != 'unique':
        v1._fail('unresolved_reference', 'previous task is not uniquely identified')


def _validate_continue_reference_context(step, text, context):
    kind = step['target']['kind']
    if kind not in ('active_task', 'previous_task'):
        return
    positive, _ = v1._reference_roles(text)
    if v1._contains_any(text, r'\b(?:previously|previous|prior|last)\s+(?:stopped\s+|interrupted\s+)?task\b'):
        positive.add('previous_task')
    if v1._contains_any(text, r'지금\s*(?:하던|하는|진행\s*중인)?\s*작업|\b(?:current|active)\s+task\b'):
        positive.add('active_task')
    if positive and kind not in positive:
        v1._fail('unresolved_reference', 'continuation target disagrees with the stated task')
    previous = context['previous_task']
    both = context['execution_state'] == 'running' and previous is not None and previous['status'] == 'unique'
    if both and not positive:
        v1._fail('unresolved_reference', 'current and previous tasks are both plausible')


_MEMORY_RETURN_EN = re.compile(
    r'\b(?P<verb>return|restore|go\s+back|bring\s+back)\s+(?:to\s+)?(?:the\s+)?'
    r'(?:remembered|memorized)\s+(?:arrangement|layout)\b', re.IGNORECASE | re.ASCII)
_MEMORY_RETURN_KO = re.compile(
    r'(?:기억(?:한|했던|해\s*둔)|저장(?:한|했던|해\s*둔))\s*배치'
    r'(?:로|를|을|에)?\s*(?:다시\s*)?(?P<verb>복원|복귀|돌려|돌리|되돌|돌아)')


def _memory_return_uses_prior_step(step, text, steps, index, validated_memories):
    """Allow one exact, unique, most-recent verified remember evidence as a reference."""
    reference, evidence = step['target']['reference'], step['evidence']
    matches = [i for i, prior in enumerate(steps)
               if prior['op'] == 'task' and prior['kind'] == 'remember' and prior['evidence'] == reference]
    if matches or reference not in evidence:
        if (len(matches) != 1 or matches[0] >= index or not validated_memories or
                validated_memories[-1] != matches[0] or text.count(reference) != 1):
            v1._fail('unresolved_reference', 'memory reference must identify the unique most recent verified remember step')
    return reference not in evidence


def _validate_cross_step_memory_return(step, text, evidence_start):
    """Keep the original return step intact while verifying its explicit target."""
    evidence = step['evidence']
    if v1._contains_any(evidence, r'팔|그리퍼|집게|다른\s*(?:물체|원통|블록)|'
                                  r'\b(?:arm|gripper|robot|another|other)\b|'
                                  r'\b(?:of|for)\s+(?:\w+\s+){0,3}(?:object|cylinder|cube|block)\b'):
        v1._fail('unresolved_reference', 'return evidence names an individual or different target')
    for kind in ('arm_home', 'registered_initial', 'task_start'):
        if v1._contains_any(evidence, v1._RETURN_REFERENCE[kind]):
            v1._fail('unresolved_reference', 'return evidence also names a different return target')
    if v1._contains_any(text, r'(?:복원하|복귀하|돌아가|원위치하)\s*지\s*(?:마|말|않)|'
                             r'\b(?:do\s+not|don[\'’]?t|never)\s+(?:return|restore|go\s+back)\b'):
        v1._fail('unsupported_action', 'return is negated')
    for pattern in (_MEMORY_RETURN_EN, _MEMORY_RETURN_KO):
        for match in pattern.finditer(evidence):
            start, end = evidence_start + match.start(), evidence_start + match.end()
            verb_start = evidence_start + match.start('verb')
            verb_end = evidence_start + match.end('verb')
            if (not v1._quoted(text, start, end) and not v1._negated_action(text, verb_start, verb_end)
                    and not v1._question_only(text, start, end)):
                return
    v1._fail('ungrounded_value', 'return evidence must explicitly request restoration of the remembered arrangement')


def _validate_return_scope(step, text, start, end, *, cross_memory=False):
    """Keep an explicit return target from being replaced by the focused memory.

    The focus identifies an existing memory; it is not evidence that a user
    requested that memory when the utterance names another target. This is a
    bounded language guard, not a general semantic parser.
    """
    kind = step['target']['kind']
    if kind == 'unresolved':
        return
    clause, offset = _task_clause(text, start, end)
    if not cross_memory:
        control = (r'원위치|복귀|돌아|돌려|돌리|되돌|복원|'
                   r'\b(?:return|restore|go\s+back|home|back)\b')
        if kind == 'arm_home':
            control += r'|보내|\b(?:send|bring|take|put|move)\b'
        local = v1._affirmed_control_clauses(text, start, end, control, step['target']['reference'])
        if len(local) == 1:
            offset, right = local[0]
            clause = text[offset:right]
    if v1._question_only(text, offset, offset + len(clause)):
        v1._fail('unsupported_action', 'a question does not request a return')
    if re.search(r'(?:복원|복귀|돌아가|되돌리)(?:하는|할|할\s*수\s*있는)\s*'
                 r'(?:방법|법|설명|절차)|'
                 r'\b(?:explain|describe|tell\s+me|show\s+me)\s+(?:how|what)\b',
                 clause, re.IGNORECASE | re.ASCII):
        v1._fail('unsupported_action', 'a request to explain a return is not a return')
    if kind == 'arm_home':
        _validate_arm_home_scope(step, text, start, end)
    def rejected_mention(left, right):
        return (v1._quoted(text, left, right) or v1._negated_span(text, left, right)
                or re.match(r'\s*(?:은|는|을|를|이|가)?\s*대신', text[right:]) is not None)

    for other, pattern in v1._RETURN_REFERENCE.items():
        if other == kind:
            continue
        for match in re.finditer(pattern, clause, re.IGNORECASE | re.ASCII):
            left, right = offset + match.start(), offset + match.end()
            if not rejected_mention(left, right):
                v1._fail('unresolved_reference', 'return proposal conflicts with the explicit target')
    if kind != 'learned_memory':
        return
    # Reject explicit partial objects, foreign places and arm targets even if
    # the model shortens its evidence to only "restore" or "아까대로".
    other_target = re.compile(
        r'(?:[1-4일이삼사]\s*번(?:째)?(?:의|에|에서|을|를|은|는|이|가)?\s*'
        r'(?:(?:있는|놓인|놓여\s*있는)\s*)?(?:원통|물체|블록|자리|위치)|'
        r'[1-4일이삼사]\s*번(?:째)?(?:자리)?\s*(?:만|을|를)|'
        r'(?:처음|맨\s*처음|초기|원래)(?:에|의|으로|로)?\s*'
        r'(?:(?:등록한|등록된|놓은|놓인|정한|설정한)\s*)?(?:배치|상태|위치)|'
        r'(?:원통|물체|블록)\s*[1-4일이삼사]\s*번|'
        r'(?:내|우리|다른)\s*(?:방|집|책상|휴대폰|핸드폰|전화기|물건|물체|원통|곳)|'
        r'(?:로봇\s*)?(?:팔|그리퍼|집게)|'
        r'\b(?:arm|gripper|phone|bedroom|kitchen|another\s+(?:room|place|object|cylinder)|'
        r'(?:initial|starting|original)\s+(?:arrangement|layout|state|position)|'
        r'(?:cylinder|object|slot)\s*(?:[1-4]|one|two|three|four)|'
        r'(?:first|second|third|fourth|one|two|three)\s+(?:cylinders?|objects?|slots?))\b)',
        re.IGNORECASE | re.ASCII)
    for match in other_target.finditer(clause):
        left, right = offset + match.start(), offset + match.end()
        if not rejected_mention(left, right):
            v1._fail('unresolved_reference', 'memory return specifies another or partial target')
    if cross_memory:
        return  # The separate cross-step guard requires an explicit remembered arrangement.
    memory_reference = re.compile(
        r'아까\s*대로|기억(?:한|했던|해\s*둔|해둔|을|된)?|'
        r'저장(?:한|했던|해\s*둔|해둔|된)|'
        r'\b(?:remembered|memorized|memory|previously\s+saved|as\s+(?:i|we|you)\s+remember)\b',
        re.IGNORECASE | re.ASCII)
    if not any(not v1._quoted(text, offset + match.start(), offset + match.end())
               and not v1._negated_span(text, offset + match.start(), offset + match.end())
               for match in memory_reference.finditer(clause)):
        v1._fail('unresolved_reference', 'the focused memory is not named by this return')


# The patterns below deliberately cover only direct quantity expressions; they
# reject a conflicting proposal and never produce or repair an executable step.
# A sentence or another action's clause is not evidence for this task.
_TASK_CLAUSE_BOUNDARY = re.compile(
    r'[;.!?。？！]|(?<=옮)기고|(?<=옮)긴\s*(?:뒤|후|다음)|'
    r'(?<=쌓)고(?:\s*나서)?|(?<=쌓)은\s*(?:뒤|후|다음)|(?<=올려)놓고|(?<=기억)하고|'
    r'(?<=기억)한\s*(?:뒤|후|다음)|(?<=바꾼)\s*(?:뒤|후|다음)|'
    r'\b(?:and\s+then|then|followed\s+by|but|instead)\b|'
    r',\s*(?=(?:please\s+)?(?:move|transfer|put|place|arrange|stack|unstack|swap|exchange|'
    r'remember|register|repeat|restore|return|stop|continue|resume|finish|complete)\b)|'
    r'\band\s+(?=(?:keep\s+)?(?:move|transfer|put|place|arrange|stack|unstack|swap|exchange|'
    r'remember|register|repeat|restore|stop)\b)|그리고|그\s*다음|말고|아니고',
    re.IGNORECASE | re.ASCII)
_EN_CARDINAL = r'(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)'
_EN_MULTIPLE_DESTINATIONS = re.compile(
    r'\b(?:to|into|onto|in|at|on(?:\s+top\s+of)?)\s+(?:the\s+)?(?:slots?|positions?)\s*(?:number\s*)?'
    + _EN_CARDINAL + r'\s*(?:and|or|&)\s*(?:(?:slots?|positions?)\s*)?' + _EN_CARDINAL + r'\b',
    re.IGNORECASE | re.ASCII)
_KO_STACK_TARGET_MENTION = re.compile(
    v1._KO_NUMBERED_TOWER + r'\s*(?:위(?:에다가|에다|에|로|나|와|과|이랑|랑)|(?:으)?로|에(?!서)|'
    r'(?=(?:또는|혹은|아니면|그리고|및|이나|나|와|과|이랑|랑)))'
    r'(?!\s*(?:있는|있던|놓인|놓여\s*있는|올려진))')
_KO_TARGET_JOIN = re.compile(r'\s*(?:(?:또는|혹은|아니면|그리고|및|이나|나|와|과|이랑|랑|,)\s*)*')
_KO_REPEATED_TRANSFER = re.compile(v1.common._NUMBER + r'\s*번(?:씩|만)?\s*(?=쌓|올려|옮|이동|내려|풀)')
_EN_REPEAT_CARDINAL = r'(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
_EN_REPEAT_QUANTITY = re.compile(
    r'\b(?:(?P<number>' + _EN_REPEAT_CARDINAL + r')\s+(?:more\s+)?times?|(?P<short>once|twice))\b',
    re.IGNORECASE | re.ASCII)
_EN_REPEAT_CORRECTION_LEFT = re.compile(
    r'\bnot\s+(?:(?:exactly|precisely|only|just)\s+)*' + _EN_REPEAT_CARDINAL +
    r'(?:\s+(?:more\s+)?times?)?\s*,?\s*$', re.IGNORECASE | re.ASCII)
_EN_REPEAT_CORRECTION_RIGHT = re.compile(
    r'but\s+(?:(?:exactly|precisely|only|just)\s+)*' + _EN_REPEAT_CARDINAL +
    r'\s+(?:more\s+)?times?\b', re.IGNORECASE | re.ASCII)
_KO_CARDINAL = r'(?:\d+|다섯|여섯|일곱|여덟|아홉|하나|둘|셋|넷|영|공|일|이|삼|사|오|육|칠|팔|구|십|한|두|세|네|열)'
_OBJECT_QUANTITY = re.compile(
    r'\bboth\s+(?:of\s+)?(?:(?:the|these|those)\s+)?(?:cylinders?|objects?|pieces?)\b|'
    r'\b' + _EN_CARDINAL + r'\s+(?:cylinders?|objects?|pieces?)\b|'
    r'(?:원통|물체)(?:을|를|이|가)?\s*(?:둘\s*(?:다|모두)|모두\s*두\s*개)|'
    r'(?:원통|물체)(?:은|는|을|를)?\s*(?:하나|둘|셋|넷)(?:을|를|이|가|만|씩)?(?=\s|[.,!?]|$)|'
    r'(?<![가-힣])' + _KO_CARDINAL + r'\s*(?:개(?:의)?\s*)?(?:원통|물체)|'
    r'(?<![가-힣])' + _KO_CARDINAL + r'\s*개(?:씩)?', re.IGNORECASE | re.ASCII)
_EN_SLOT_NAME = r'(?:slot|spot|position|place)\s*(?:number\s*|no\.?\s*|#\s*)?'
_EN_LAYOUT_VALUE = (r'(?:(?:exactly|precisely|only|just)\s+)?'
                    r'(?:empty|vacant|none|no\s+(?:cylinders?|objects?|pieces?)|'
                    r'both\s+(?:of\s+)?(?:the\s+)?(?:cylinders?|objects?|pieces?)|'
                    + _EN_CARDINAL + r'\s+(?:cylinders?|objects?|pieces?))')
_KO_LAYOUT_VALUE = (r'(?:(?:원통|물체)(?:을|를|이|가)?\s*)?(?:(?:정확히|딱|꼭|총)\s*)?(?:' + _KO_CARDINAL +
                    r'\s*개(?:씩)?|(?:하나|둘|셋|넷)(?:이|가)?\s*(?:있|놓|되)|비워|비우|비어|없|빈\s*자리)')
_EN_SLOT_FIRST = re.compile(r'\b' + _EN_SLOT_NAME + r'(?P<slot>' + _EN_CARDINAL + r')\s*'
                            r'(?:(?:is|has|contains|with|should\s+be|should\s+have|should\s+contain|'
                            r'must\s+have|must\s+contain|has\s+to\s+have|has\s+to\s+contain|'
                            r'to\s+have|to\s+contain)\s+|[:=]\s*)?'
                            r'(?P<value>' + _EN_LAYOUT_VALUE + r')\b', re.IGNORECASE | re.ASCII)
_EN_VALUE_FIRST = re.compile(r'\b(?P<value>' + _EN_LAYOUT_VALUE + r')\s+(?:in|at|on|to)\s+'
                             r'(?:the\s+)?' + _EN_SLOT_NAME + r'(?P<slot>' + _EN_CARDINAL + r')\b',
                             re.IGNORECASE | re.ASCII)
_EN_EMPTY_SLOT = re.compile(r'\b(?P<value>empty)\s+(?:the\s+)?' + _EN_SLOT_NAME +
                            r'(?P<slot>' + _EN_CARDINAL + r')\b', re.IGNORECASE | re.ASCII)
_KO_SLOT_FIRST = re.compile(r'(?P<slot>' + _KO_CARDINAL + r')\s*번(?:째)?\s*(?:자리|위치)?'
                            r'(?:에는|에서|으로|에게|은|는|을|를|이|가|엔|에|로)?\s*'
                            r'(?P<value>' + _KO_LAYOUT_VALUE + r')')
_KO_VALUE_FIRST = re.compile(r'(?P<value>' + _KO_LAYOUT_VALUE + r')(?:을|를|이|가)?\s*'
                             r'(?P<slot>' + _KO_CARDINAL + r')\s*번(?:째)?\s*(?:자리|위치)?'
                             r'(?:에는|으로|은|는|에|로)')
_EN_SLOT_CORRECTION = re.compile(
    r'\b' + _EN_SLOT_NAME + r'(?P<slot>' + _EN_CARDINAL + r')\s*'
    r'(?:(?:is|has|contains|should\s+be|should\s+have)\s+)?not\s+'
    r'(?:' + _EN_LAYOUT_VALUE + r')\s*,?\s*(?:but|instead)\s+'
    r'(?P<value>' + _EN_LAYOUT_VALUE + r')\b', re.IGNORECASE | re.ASCII)
_KO_SLOT_CORRECTION = re.compile(
    r'(?P<slot>' + _KO_CARDINAL + r')\s*번(?:째)?\s*(?:자리|위치)?'
    r'(?:에는|은|는|을|를|이|가|에)?\s*(?:' + _KO_LAYOUT_VALUE + r')'
    r'(?:이|가|은|는|을|를)?\s*(?:말고|아니고|대신)\s*(?P<value>' + _KO_LAYOUT_VALUE + r')')
_KO_TRANSFER_VALUE = r'(?:' + _KO_CARDINAL + r'\s*개(?:씩)?|하나|둘|셋|넷)'
_KO_TRANSFER_CORRECTION = re.compile(
    r'(?:원통|물체)(?:은|는|을|를|이|가)?\s*'
    r'(?P<negative>' + _KO_TRANSFER_VALUE + r')(?:을|를|이|가|은|는)?\s*'
    r'(?:말고|아니고|대신)\s*'
    r'(?P<positive>' + _KO_TRANSFER_VALUE + r')(?:을|를|이|가|은|는|만)?'
    r'(?=\s|[.,!?]|$)')
_OTHER_ACTION = re.compile(r'기억|저장|등록|복원|반복|되풀이|\b(?:remember|register|restore|repeat)\b',
                           re.IGNORECASE | re.ASCII)
_LAYOUT_ZERO = re.compile(r'\b(?:empty|vacant|none|no\s+(?:cylinders?|objects?|pieces?))\b|'
                          r'비워|비우|비어|없|빈\s*자리', re.IGNORECASE | re.ASCII)
_NEGATED_QUANTITY_PREFIX = re.compile(
    r'\b(?:not|except|without|instead\s+of|rather\s+than)\s+'
    r'(?:(?:the|a|an|exactly|precisely|only|just)\s+)*$', re.IGNORECASE | re.ASCII)


def _restatement_comma(text, position):
    return re.search(r'(?:\b(?:in\s+other\s+words|that\s+is)|즉|다시\s*말해(?:서|면)?)\s*$',
                     text[max(0, position - 40):position], re.IGNORECASE | re.ASCII) is not None


def _task_clause(text, start, end):
    """Retain the whole local clause even when the model clipped evidence."""
    boundaries = [match for match in _TASK_CLAUSE_BOUNDARY.finditer(text)
                  if not (match.group().startswith(',') and _restatement_comma(text, match.start()))]
    left = max((match.end() for match in boundaries if match.end() <= start), default=0)
    right = min((match.start() for match in boundaries if match.start() >= end), default=len(text))
    return text[left:right], left


def _quantity_value(value):
    lower = value.lower().strip()
    lower = re.sub(r'^(?:원통|물체)(?:을|를|이|가)?\s*', '', lower)
    lower = re.sub(r'^(?:(?:exactly|precisely|only|just)\s+|(?:정확히|딱|꼭|총)\s*)', '', lower)
    if re.match(r'(?:both|둘\s*(?:다|모두)|모두\s*두)', lower):
        return 2
    if re.match(r'(?:empty|vacant|none|no\b|비워|비우|비어|없|빈)', lower):
        return 0
    token = re.search(_EN_CARDINAL + '|' + _KO_CARDINAL, lower)
    if token is None:
        raise ValueError('quantity expression does not contain a number')
    number = token.group(0)
    if number.isdecimal():
        return int(number)
    return v1.bilingual._EN_VALUES[number] if number in v1.bilingual._EN_VALUES else v1.common._NUMBERS[number]


def _affirmed_quantities(text, *, offset=0, whole=None):
    whole = text if whole is None else whole
    values = set()
    for match in _OBJECT_QUANTITY.finditer(text):
        if not v1._negated_span(whole, offset + match.start(), offset + match.end()):
            word = match.group(0)
            # Korean object-first both expressions have an object prefix.
            if re.search(r'둘\s*(?:다|모두)|모두\s*두', word):
                values.add(2)
            else:
                values.add(_quantity_value(word))
    return values


def _layout_quantities(text, *, offset=0, whole=None, include_spans=False):
    """Find direct slot/count pairs in either stated order, not global sets."""
    whole = text if whole is None else whole
    result = {}
    spans = []
    for pattern in (_EN_SLOT_FIRST, _EN_VALUE_FIRST, _EN_EMPTY_SLOT, _KO_SLOT_FIRST, _KO_VALUE_FIRST,
                    _EN_SLOT_CORRECTION, _KO_SLOT_CORRECTION):
        for match in pattern.finditer(text):
            start, end = match.span('value')
            if v1._negated_span(whole, offset + start, offset + end):
                continue
            slot = _quantity_value(match.group('slot'))
            count = _quantity_value(match.group('value'))
            result.setdefault(slot, set()).add(count)
            spans.append((offset + start, offset + end))
    return (result, spans) if include_spans else result


def _layout_slot_mentions(text):
    """Find slot anchors, excluding bare numbered objects."""
    result = []
    for match in v1.bilingual._EN_SLOT.finditer(text):
        result.append((match.start(), v1.bilingual._value(match.group(1))))
    for match in v1.common._REGISTRATION_SLOT.finditer(text):
        token = match.group(1) or match.group(2)
        result.append((match.start(), int(token) if token.isdecimal() else v1.common._NUMBERS[token]))
    for match in re.finditer(r'(?P<slot>' + _KO_CARDINAL + r')\s*번(?:째)?\s*엔', text):
        result.append((match.start(), _quantity_value(match.group('slot'))))
    return sorted(set(result))


def _validate_unbound_layout_quantities(clause, offset, whole, pairs, spans):
    """Require clarification when a stated count cannot be bound safely."""
    if not _layout_slot_mentions(clause):
        return
    for pattern in (_OBJECT_QUANTITY, _LAYOUT_ZERO):
        for match in pattern.finditer(clause):
            start, end = offset + match.start(), offset + match.end()
            if any(a <= start and end <= b for a, b in spans):
                continue
            local, local_offset = _task_clause(whole, start, end)
            anchors = _layout_slot_mentions(local)
            if not anchors:
                continue
            prefix = whole[max(0, start - 60):start]
            negated = (v1._negated_span(whole, start, end) or
                       _NEGATED_QUANTITY_PREFIX.search(prefix) is not None)
            earlier = [(local_offset + pos, slot) for pos, slot in anchors if local_offset + pos < start]
            if negated and earlier and max(earlier)[1] in pairs:
                # A separately parsed affirmative correction is still checked.
                continue
            v1._fail('unresolved_reference', 'explicit layout quantity cannot be bound to a slot')


def _transfer_clause(step, text, start, end):
    # Evidence may include other operations. Bind the quantity to the
    # affirmative verb's local clause, never to an earlier remember/repeat.
    clauses = []
    for verb in re.finditer(v1._TASK_WORDS[step['kind']], step['evidence'], re.IGNORECASE | re.ASCII):
        left, right = start + verb.start(), start + verb.end()
        if not _explicit_span(text, left, right, verb.group()):
            continue
        candidate = _task_clause(text, left, right)
        if candidate not in clauses:
            clauses.append(candidate)
    if len(clauses) > 1:
        matching = []
        for candidate, offset in clauses:
            _, sources, destinations, objects, _, _ = v1._task_supports(
                _mask_quoted(candidate), whole=text, offset=offset)
            if (step['source_slot'] is not None and step['source_slot'] not in sources or
                    step['destination_slot'] is not None and step['destination_slot'] not in destinations or
                    step['object_id'] is not None and step['object_id'] not in objects):
                continue
            matching.append((candidate, offset))
        if len(matching) != 1:
            v1._fail('unresolved_reference', 'task evidence spans ambiguous transfer clauses')
        clauses = matching
    return clauses[0] if clauses else _task_clause(text, start, end)


def _reject_multiple_korean_stack_targets(step, text, start):
    """Reject paired alternatives even when one equals the source slot."""
    if step['kind'] != 'stack':
        return
    evidence = step['evidence']
    mentions = sorted((m for pattern in (_KO_STACK_TARGET_MENTION, v1._KO_PLACE_STACK_DESTINATION)
                       for m in pattern.finditer(evidence)
                       if not v1._quoted(text, start + m.start(), start + m.end())
                       and not v1._negated_span(text, start + m.start(), start + m.end())),
                      key=lambda m: m.start())
    for first, second in zip(mentions, mentions[1:]):
        if _KO_TARGET_JOIN.fullmatch(evidence[first.end():second.start()]):
            v1._fail('unresolved_reference', 'multiple stack destinations require clarification')


def _transfer_corrections(text, clause, offset):
    """Associate a bare Korean correction only with its following transfer."""
    right = offset + len(clause)
    verbs = list(re.finditer(v1._TASK_WORDS['move'] + '|' + v1._TASK_WORDS['stack'] + '|' +
                             v1._TASK_WORDS['unstack'], clause, re.IGNORECASE | re.ASCII))
    result = []
    for match in _KO_TRANSFER_CORRECTION.finditer(text):
        start, end = match.span('positive')
        if (end <= offset and v1._negated_span(text, start, end)
                and re.fullmatch(r'\s*(?:말고|아니고|대신)\s*', text[match.end():offset])):
            v1._fail('unresolved_reference', 'multiple quantity corrections need clarification')
        if not (offset <= start and end <= right) or v1._negated_span(text, start, end):
            continue
        following = [offset + verb.start() for verb in verbs if offset + verb.start() >= match.end()]
        if not following or _OTHER_ACTION.search(text[match.end():min(following)]):
            continue
        result.append(match)
    return result


def _validate_explicit_task_quantities(step, text, start, end):
    if step['kind'] in ('move', 'stack', 'unstack'):
        clause, offset = _transfer_clause(step, text, start, end)
        counts = _affirmed_quantities(clause, offset=offset, whole=text)
        counts.update(_quantity_value(match.group('positive'))
                      for match in _transfer_corrections(text, clause, offset))
        if counts and (len(counts) != 1 or step['count'] not in counts):
            v1._fail('ungrounded_value', 'an explicit object quantity was omitted, changed or ambiguous')
    elif step['kind'] in ('arrange', 'register') and step['layout'] is not None:
        clause, offset = _task_clause(text, start, end)
        pairs, spans = _layout_quantities(clause, offset=offset, whole=text, include_spans=True)
        _validate_unbound_layout_quantities(clause, offset, text, pairs, spans)
        all_ids = [object_id for entry in step['layout'] for object_id in (entry['object_ids'] or [])]
        total = sum(entry['count'] if entry['count'] is not None else len(entry['object_ids'] or [])
                    for entry in step['layout'])
        if (len({entry['slot'] for entry in step['layout']}) != len(step['layout']) or
                len(set(all_ids)) != len(all_ids) or total > 4):
            v1._fail('invalid_task', 'duplicate layout slot/object or more than four objects')
        entries = {entry['slot']: entry for entry in step['layout']}
        for slot, counts in pairs.items():
            if len(counts) != 1 or slot not in entries:
                v1._fail('ungrounded_value', 'an explicit slot quantity was omitted or ambiguous')
            entry = entries[slot]
            actual = entry['count'] if entry['count'] is not None else len(entry['object_ids'] or [])
            if actual not in counts:
                v1._fail('ungrounded_value', 'layout count differs from its stated slot')
        return pairs
    return {}


# These guards cover only recognisable, explicitly linked operations. They
# check a proposal against the request; they do not infer or add operations.
_REQUIRED_MEMORY = re.compile(
    r'기억(?=해|하(?:려|시기|기|고|세|라|자)|한\s*(?:뒤|후|다음))|'
    r'저장(?=해|하(?:려|시기|기|고|세|라|자)|한\s*(?:뒤|후|다음))|'
    r'\b(?:remember|remembering|memorize|memorizing|save|saving)\b', re.IGNORECASE | re.ASCII)
_REQUIRED_MOVE = re.compile(r'옮|이동|\b(?:move|moving|transfer|transferring)\b', re.IGNORECASE | re.ASCII)
_COVERAGE_MOVE = re.compile(
    r'옮|이동|놓(?=아|으세요|으라|고)|'
    r'\b(?:move|moving|transfer|transferring|put|place|bring|carry|send)\b', re.IGNORECASE | re.ASCII)
_REQUIRED_SWAP = re.compile(r'바꾸|바꿔|바꾼|교환|맞바꿔|\b(?:swap|swapping|exchange|exchanging|switch|switching)\b',
                            re.IGNORECASE | re.ASCII)
_REQUIRED_STACK = re.compile(v1._TASK_WORDS['stack'], re.IGNORECASE | re.ASCII)
_REQUIRED_UNSTACK = re.compile(v1._TASK_WORDS['unstack'], re.IGNORECASE | re.ASCII)
_REQUIRED_RETURN = re.compile(r'원위치(?=해|하|시켜)|복귀(?=해|하)|돌아가|복원(?=해|하)|'
                              r'\b(?:return|restore|go\s+back)\b', re.IGNORECASE | re.ASCII)
_EN_SCENE_SLOT = r'(?:[1-4]|one|two|three|four)'
_EN_TOP_TOWER_MOVE = re.compile(
    r'\b(?P<verb>move)\s+the\s+top\s+cylinder\s+from\s+the\s+tower\s+at\s+slot\s+'
    r'(?P<source>' + _EN_SCENE_SLOT + r')\s+onto\s+the\s+tower\s+at\s+slot\s+'
    r'(?P<destination>' + _EN_SCENE_SLOT + r')\b', re.IGNORECASE | re.ASCII)
_EN_CURRENT_LAYOUT_SENTENCE = r'remember\s+the\s+current\s+layout'
_EN_REMEMBERED_LAYOUT_SENTENCE = (r'(?:return|restore|go\s+back)\s+(?:to\s+)?(?:the\s+)?'
                                  r'(?:remembered|memorized)\s+(?:arrangement|layout)')
_EN_STACK_ON = re.compile(
    r'\b(?P<verb>stack)\s+(?:the|a|(?P<one>one|1))\s+cylinder\s+from\s+slot\s+'
    r'(?P<source>' + _EN_SCENE_SLOT + r')\s+' + v1._EN_STACK_SLOT_PREFIX
    + r'(?P<destination>' + _EN_SCENE_SLOT + r')\b',
    re.IGNORECASE | re.ASCII)

# The source and the supporting object are different grammatical roles. Bind
# the roles to the utterance here; their slots depend on the execution scene.
_KO_CYLINDER = r'(?:원통|물체)'
_KO_SCENE_NUMBER = r'[1-4일이삼사한두세네]'
_KO_NUMBERED_CYLINDER = (r'(?:' + _KO_SCENE_NUMBER + r'\s*번\s*' + _KO_CYLINDER + r'|' +
                         _KO_CYLINDER + r'\s*(?:번호\s*)?' + _KO_SCENE_NUMBER + r'\s*번)')
_KO_STACK_SOURCE = (r'(?:(?P<slot>' + _KO_SCENE_NUMBER + r')\s*번\s*(?:(?:자리|위치|더미|덤이|탑)\s*(?:의|에서|에\s*있는)|에서)\s*)?'
                    r'(?P<source>' + _KO_NUMBERED_CYLINDER + '|' + _KO_CYLINDER + r')'
                    r'(?:\s*(?P<one>하나|한\s*개|1\s*개))?(?:을|를|은|는)?')
_KO_STACK_SUPPORT = r'(?P<destination>' + _KO_NUMBERED_CYLINDER + r')\s*(?:의\s*)?위(?:에다가|에다|에|로)'
_EN_CYLINDER = r'(?:cylinder|object)'
_EN_NUMBERED_CYLINDER = _EN_CYLINDER + r'\s*(?:(?:number|id|no)\s*|#\s*)?' + _EN_SCENE_SLOT + r'\b'
_EN_STACK_SUPPORT = (r'(?:on\s+top\s+of|onto|on|atop)\s+(?:the\s+)?(?P<destination>' +
                     _EN_NUMBERED_CYLINDER + r')')
_OBJECT_STACK_PATTERNS = (
    re.compile(_KO_STACK_SOURCE + r'\s*(?:가져다가\s*)?' + _KO_STACK_SUPPORT + r'\s*(?P<verb>쌓|올려)'),
    re.compile(_KO_STACK_SUPPORT + r'\s*' + _KO_STACK_SOURCE + r'\s*(?P<verb>쌓|올려)'),
    re.compile(r'\b(?P<verb>stack|pile|put|place)\s+(?:the\s+)?(?:(?P<one>one|1)\s+)?'
               r'(?P<source>' + _EN_NUMBERED_CYLINDER + '|' + _EN_CYLINDER + r')'
               r'(?:\s+(?:from|at|in)\s+(?:the\s+)?(?:slot|position)\s+(?P<slot>' + _EN_SCENE_SLOT + r'))?'
               r'\s+' + _EN_STACK_SUPPORT, re.IGNORECASE | re.ASCII),
)
_OBJECT_SUPPORTS = (re.compile(_KO_STACK_SUPPORT), re.compile(_EN_STACK_SUPPORT, re.IGNORECASE | re.ASCII))
_OBJECT_NUMBER = re.compile(_KO_SCENE_NUMBER + r'|\b(?:one|two|three|four)\b', re.IGNORECASE | re.ASCII)
_KO_PLACE_SOURCE = (r'(?P<source>' + _KO_SCENE_NUMBER + r')\s*번(?:째)?\s*' + v1._KO_PLACE
                    + r'\s*' + v1._KO_SLOT_OCCUPANT + r'(?:\s*(?P<one>하나|한\s*개|1\s*개))?'
                    r'\s*(?:을|를|은|는|이|가)?')
_KO_PLACE_SUPPORT = r'(?P<destination>' + _KO_SCENE_NUMBER + r')\s*번(?:째)?\s*' + v1._KO_PLACE_ABOVE
_SLOT_STACK_PATTERNS = (
    re.compile(_KO_PLACE_SOURCE + r'\s*' + _KO_PLACE_SUPPORT + r'\s*(?P<verb>쌓|올려)'),
    re.compile(_KO_PLACE_SUPPORT + r'\s*' + _KO_PLACE_SOURCE + r'\s*(?P<verb>쌓|올려)'),
    _EN_STACK_ON,
)


def _stack_object_roles(match):
    def number(argument):
        token = _OBJECT_NUMBER.search(argument or '')
        return _quantity_value(token.group()) if token else None
    return (number(match.group('slot')), number(match.group('source')), number(match.group('destination')),
            1 if match.group('one') else None)


def _explicit_object_stacks(text):
    for pattern in _OBJECT_STACK_PATTERNS:
        for match in pattern.finditer(text):
            if _explicit_match(text, match):
                yield match


def _complete_object_stack(text, match, *, standalone=False):
    clause, offset = (text, 0) if standalone else _task_clause(text, *match.span())
    prefix, suffix = clause[:match.start() - offset], clause[match.end() - offset:]
    return (re.fullmatch(r'\s*[,]?\s*(?:(?:please|now|먼저|우선|지금|그다음에?)\s*)*', prefix, re.I)
            and re.fullmatch(r'(?:아|아라|으세요|아요|놓아|놔)?\s*'
                             r'(?:줘요|줘|주세요|주십시오|please)?[\s,.!。！?？]*', suffix, re.I))


def explicit_object_stack_step(text, *, evidence=None):
    """Bind two named cylinders in one direct stack; their slots depend on the execution scene."""
    if not isinstance(text, str) or len(text) > v1.MAX_TEXT_CHARS:
        return None
    evidence = text if evidence is None else evidence
    start = text.find(evidence)
    matches = [m for m in _explicit_object_stacks(text) if 0 <= start <= m.start()
               and m.end() <= start + len(evidence) and _complete_object_stack(text, m, standalone=True)]
    if len(matches) != 1:
        return None
    source_slot, moved, support, count = _stack_object_roles(matches[0])
    if moved is None or support is None or moved == support:
        return None
    return {'op': 'task', 'kind': 'stack', 'source_slot': source_slot, 'object_id': moved,
            'destination_slot': None, 'destination_object_id': support, 'count': count,
            'height': None, 'layout': None, 'evidence': evidence}


def _object_stack_roles_match(step, stated):
    """Allow the two stated participants in slot fields; never supply a missing participant."""
    source_slot = stated['source_slot']
    if source_slot is not None and step['source_slot'] != source_slot:
        return False
    moved = {step['object_id'], step['source_slot'] if source_slot is None else None} - {None}
    support = {step['destination_slot'], step.get('destination_object_id')} - {None}
    return (moved == {stated['object_id']} and support == {stated['destination_object_id']}
            and (step['count'] == stated['count'] or (stated['count'] is None and step['count'] == 1)))


def _numbered_tower_citation(step, text):
    """Identify the cited destination in the same affirmative stacking clause."""
    evidence = step['evidence']
    start = text.find(evidence)
    if start < 0:
        return None
    citations = set()
    for verb in _REQUIRED_STACK.finditer(evidence):
        left, right = start + verb.start(), start + verb.end()
        if not _explicit_span(text, left, right, verb.group()):
            continue
        clause, offset = _task_clause(text, left, right)
        targets = list(_KO_STACK_TARGET_MENTION.finditer(clause))
        if len(targets) != 1 or any(pattern.search(clause) for pattern in _OBJECT_SUPPORTS):
            return None
        target = targets[0]
        if (v1._KO_SLOT_STACK_DESTINATION.fullmatch(target.group()) is None
                or re.search(r'위(?:에다가|에다|에|로)$', target.group()) is None
                or _quantity_value(target.group(1)) != step['destination_object_id']
                or not start <= offset + target.start() < offset + target.end() <= start + len(evidence)
                or v1._quoted(text, offset + target.start(), offset + target.end())
                or v1._negated_span(text, offset + target.start(), offset + target.end())):
            return None
        citations.add((offset + target.start(), target.group()))
    return next(iter(citations))[1] if len(citations) == 1 else None


def _slot_stack_match(text, start, end, *, standalone=False):
    """Find one fully qualified slot-to-slot stack in the original clause."""
    matches = []
    for pattern in _SLOT_STACK_PATTERNS:
        for match in pattern.finditer(text, start, end):
            if not _explicit_match(text, match):
                continue
            clause, offset = (text, 0) if standalone else _task_clause(text, *match.span('verb'))
            prefix = clause[:match.start() - offset]
            suffix = clause[match.end() - offset:]
            if (not re.fullmatch(r'\s*[,]?\s*(?:(?:please|now|먼저|우선|지금)\s*)*', prefix, re.I)
                    or not re.fullmatch(r'(?:아|아라|으세요|아요|놓아|놔)?\s*'
                                        r'(?:줘요|줘|주세요|주십시오|please)?[\s,.!。！?？]*', suffix, re.I)):
                continue
            if _quantity_value(match['source']) != _quantity_value(match['destination']):
                matches.append(match)
    return matches[0] if len(matches) == 1 else None


def explicit_slot_stack_roles(text):
    """Return (source slot, destination slot, stated count) for one whole direct stack, or None."""
    if not isinstance(text, str) or len(text) > v1.MAX_TEXT_CHARS:
        return None
    match = _slot_stack_match(text, 0, len(text), standalone=True)
    if match is None:
        return None
    return _quantity_value(match['source']), _quantity_value(match['destination']), 1 if match['one'] else None


def _explicit_slot_stack_citation(step, text):
    evidence = step['evidence']
    start = text.find(evidence)
    match = _slot_stack_match(text, start, start + len(evidence)) if start >= 0 else None
    if match is None:
        return None
    source, destination = _quantity_value(match['source']), _quantity_value(match['destination'])
    if (step['source_slot'] not in (None, source) or step['object_id'] not in (None, source)
            or (step['source_slot'] is None and step['object_id'] is None)
            or step['destination_slot'] not in (None, destination)
            or step.get('destination_object_id') not in (None, destination)):
        return None
    return source, destination, match.group()


def normalize_stack_proposal(candidate, text):
    """Ground explicitly worded stack and unstack roles, retaining the raw proposal."""
    steps = list(candidate['steps'])
    changes = []
    for index, step in enumerate(steps):
        if (len(steps) == 1 and step['op'] == 'task' and step['kind'] in ('move', 'stack')
                and step['height'] is None and step['layout'] is None):
            stated = explicit_object_stack_step(text, evidence=step['evidence'])
            if stated is not None and _object_stack_roles_match(step, stated):
                if stated != step:
                    steps[index] = stated
                    changes.append({'step_index': index, 'reason': 'explicit_object_stack_roles',
                                    'evidence': step['evidence'], 'object_id': stated['object_id'],
                                    'destination_object_id': stated['destination_object_id']})
                continue
        if (len(steps) == 1 and step['op'] == 'task' and step['kind'] in ('move', 'unstack')
                and step['height'] is None and step['layout'] is None):
            roles = explicit_top_unstack_roles(text, evidence=step['evidence'])
            if roles is not None:
                source, destination, stated_count = roles
                if (step['source_slot'] in (None, source) and step['object_id'] in (None, source)
                        and (step['source_slot'] is not None or step['object_id'] is not None)
                        and step['destination_slot'] == destination
                        and (step['count'] == stated_count or (stated_count is None and step['count'] == 1))):
                    unstack = {**step, 'kind': 'unstack', 'source_slot': source,
                               'object_id': None, 'count': stated_count}
                    if unstack != step:
                        steps[index] = unstack
                        changes.append({'step_index': index, 'reason': 'explicit_top_unstack_roles',
                                        'evidence': step['evidence'], 'source_slot': source,
                                        'destination_slot': destination})
        if step['op'] != 'task' or step['kind'] != 'stack':
            continue
        citation = _explicit_slot_stack_citation(step, text)
        if citation is not None:
            source, destination, cited_text = citation
            stack = dict(step)
            stack['source_slot'] = source
            stack['object_id'] = None
            stack['destination_slot'] = destination
            stack.pop('destination_object_id', None)
            if stack != step:
                steps[index] = stack
                changes.append({'step_index': index, 'reason': 'explicit_slot_stack_roles',
                                'evidence': cited_text, 'source_slot': source,
                                'destination_slot': destination})
            continue
        if (step.get('destination_object_id') is None
                or step['destination_slot'] is not None):
            continue
        citation = _numbered_tower_citation(step, text)
        if citation is None:
            continue
        stack = dict(step)
        stack['destination_slot'] = stack.pop('destination_object_id')
        steps[index] = stack
        changes.append({'step_index': index, 'reason': 'numbered_tower_names_slot',
                        'evidence': citation, 'destination_slot': stack['destination_slot']})
    return ({**candidate, 'steps': steps} if changes else candidate), changes


def _validate_object_stack(step, text, assigned_spans):
    expected = (step['source_slot'], step['object_id'], step['destination_object_id'], step['count'])
    if expected[:2] == (None, None) or step['object_id'] == step['destination_object_id']:
        v1._fail('invalid_task', 'a stack needs a named source and a different supporting cylinder')
    evidence_spans = [m.span() for m in re.finditer(re.escape(step['evidence']), text)]
    matches = [m for m in _explicit_object_stacks(text) if _stack_object_roles(m) == expected
               and any(start <= m.start() and m.end() <= end for start, end in evidence_spans)]
    if not matches or v1._contains_any(text, v1._INJECTED_INSTRUCTION):
        v1._fail('ungrounded_value', 'the moved cylinder and its support need distinct, explicit roles')
    match = next((m for m in matches if m.span('verb') not in assigned_spans), None)
    if match is None:
        v1._fail('duplicate_unproven_step', 'a second stack needs its own stated action')
    # Never authorize a clipped match inside a choice, a correction, a
    # description, or another target qualification. Other actions are checked
    # independently by the whole-program guards.
    if not _complete_object_stack(text, match):
        v1._fail('unresolved_reference', 'the stack contains another qualification or ambiguous reference')
    assigned_spans.add(match.span('verb'))


_EN_UNSTACK_TOP = re.compile(
    r'\b(?P<verb>unstack)\s+the\s+top\s+cylinder\s+in\s+slot\s+(?P<source>' + _EN_SCENE_SLOT + r')\s+'
    r'and\s+(?P<put>put)\s+(?P<anaphora>it)\s+in\s+the\s+empty\s+slot\s+'
    r'(?P<destination>' + _EN_SCENE_SLOT + r')\b', re.IGNORECASE | re.ASCII)
_KO_EMPTY_DESTINATION = re.compile(r'([1-4일이삼사한두세네])\s*번\s*빈\s*자리(?:로|에)')
_KO_ONE_CYLINDER = r'(?:한\s*개|1\s*개|하나)'
_KO_TOP_CYLINDER = (r'(?:탑(?:의)?\s*)?맨\s*위(?:의|에(?:\s*(?:있는|놓인))?)?\s*'
                    r'원통(?:\s*' + _KO_ONE_CYLINDER + r')?(?:을|를)?')
_KO_TOP_UNSTACK_ACTION = re.compile(r'내려(?:놓아|놔|서)?(?:\s*탑(?:을)?\s*풀어)?')
_KO_UNSTACK_TO_EMPTY = re.compile(
    r'\s*(?P<source>[1-4일이삼사한두세네])\s*번\s*(?:(?:자리|위치)\s*(?:에서|의)?|에서|'
    + v1._KO_TOWER + r'(?:에서|의)?)\s*'
    r'(?:' + _KO_EMPTY_DESTINATION.pattern + r'\s*' + _KO_TOP_CYLINDER + r'|'
    + _KO_TOP_CYLINDER + r'\s*' + _KO_EMPTY_DESTINATION.pattern + r')\s*'
    r'(?P<verb>' + _KO_TOP_UNSTACK_ACTION.pattern + r')'
    r'(?:\s*(?:줘요|줘|주세요|라))?\s*[.!。！?？]?\s*')


def explicit_top_unstack_roles(text, *, evidence=None):
    """Bind the top cylinder and empty destination of one complete Korean lowering request."""
    if not isinstance(text, str) or len(text) > v1.MAX_TEXT_CHARS:
        return None
    command = _KO_UNSTACK_TO_EMPTY.fullmatch(text)
    if command is None or not _explicit_span(text, *command.span('verb'), command['verb']):
        return None
    if evidence is not None:
        start = text.find(evidence)
        if start < 0 or start > command.start('source') or start + len(evidence) < command.end('verb'):
            return None
    source = _quantity_value(command['source'])
    destination = _quantity_value(_KO_EMPTY_DESTINATION.search(text)[1])
    if source == destination:
        return None
    count = 1 if re.search(r'원통\s*' + _KO_ONE_CYLINDER, text) else None
    return source, destination, count


_EN_SWAP_POSITIONS = re.compile(
    r'\b(?P<verb>swap)\s+the\s+cylinders\s+in\s+positions\s+(?P<source>' + _EN_SCENE_SLOT + r')\s+'
    r'and\s+(?P<destination>' + _EN_SCENE_SLOT + r')\b', re.IGNORECASE | re.ASCII)
_EN_NOW_SCENE_MEMORY = re.compile(
    r'\b(?P<verb>remember)\s+how\s+the\s+four\s+positions\s+are\s+arranged\s+'
    r'(?P<anaphora>right)\s+now\b', re.IGNORECASE | re.ASCII)
_KO_HOME_SEND = re.compile(
    r'(?:로봇\s*)?(?P<arm>팔)(?:을|를)?\s*대기\s*자세(?:로|에)\s*(?P<verb>보내)(?:줘|주세요|라)')
# A return's evidence may cover an entire utterance.  Check the actual home
# action and target rather than treating quotation/negation of that whole span
# as evidence that every word within it is an affirmative command.
_ARM_HOME_CUE = re.compile(r'대기\s*자세|원위치|\b(?:home|back)\b', re.IGNORECASE | re.ASCII)
_ARM_HOME_VERB = re.compile(
    r'보내|돌려|돌리|되돌|돌아가|복귀|복원|이동하|원위치(?=해|하|시켜)|'
    r'\b(?:send|return|move|bring|take|put|go|restore)\b', re.IGNORECASE | re.ASCII)
_ARM_HOME_CONDITION = re.compile(
    r'\b(?:if|unless|when|whenever|provided)\b|'
    r'[가-힣]{2,}(?:으면|면|다면)(?=\s|[,，])|'
    r'(?:경우|때)(?:에|에는|만)?(?=\s|[,，])', re.IGNORECASE | re.ASCII)
_ARM_HOME_FOREIGN_TARGET = re.compile(
    r'(?:내|나의|제|저의|사람(?:의)?|인간(?:의)?|인체(?:의)?|사용자(?:의)?|'
    r'다른\s*사람(?:의)?|친구(?:의)?)\s*(?:왼쪽|오른쪽|왼|오른|좌측|우측)?\s*팔|'
    r'(?:다른|옆의|저쪽)\s*로봇(?:의)?\s*팔|'
    r'(?:원통|물체|사람|인간|블록)(?:을|를)\s*(?:대기\s*자세|원위치)|'
    r'\b(?:my|his|her|their|human|person[\'’]s|user[\'’]s)\s+'
    r'(?:(?:left|right|upper|lower)\s+)?arm\b|'
    r'\b(?:other|another)\s+robot(?:ic)?\s+arm\b|'
    r'\b(?:send|return|move|bring|take|put)\s+(?:the\s+|a\s+|my\s+)?'
    r'(?:cylinder|object|person|human|block)\b[^.;!?]{0,40}\b(?:home|back)\b',
    re.IGNORECASE | re.ASCII)
_ARM_HOME_FRAGMENT_BOUNDARY = re.compile(
    r'[,，;.!?。？！]|\b(?:then|but|instead)\b|'
    r'\band\s+(?=(?:send|return|move|bring|take|put|go|restore)\b)|'
    r'그리고|그\s*다음|말고|아니고', re.IGNORECASE | re.ASCII)
_MEMORY_TARGET = re.compile(r'배치|장면|상태|원통|물체|\b(?:layout|arrangement|scene|cylinders?|objects?)\b',
                            re.IGNORECASE | re.ASCII)
_OTHER_MEMORY_TARGET_EN = re.compile(
    r'\b(?P<verb>remember|memorize|save)\b\s+'
    r'(?:(?:the|this|that|a|an|my|our)\s+)?'
    r'(?:(?:previous|prior|old|original|initial|past|different|former|earlier|last)\s+'
    r'(?:layout|arrangement|scene|state|positions?|locations?|places?|slots?)|'
    r'(?:route|path|trajectory|movement|move|sequence|command|instruction|task|action))\b',
    re.IGNORECASE | re.ASCII)
_OTHER_MEMORY_TARGET_KO = re.compile(
    r'(?:(?:이전|과거|예전|원래|다른|시작\s*전|이동(?:하기)?\s*전|옮기기\s*전)(?:의)?\s*'
    r'(?:배치|장면|상태|위치|자리)|'
    r'경로|동작|작업|움직임|이동|순서|명령|행동)(?:을|를|도|만)?\s*'
    r'(?P<verb>기억(?=해|하(?:기|고|세|라|자))|저장(?=해|하(?:기|고|세|라|자)))')
# A remember step stores the whole registered robot scene.  These checks use
# the request clause, not the model-selected evidence, so clipping evidence to
# just "remember" cannot turn a different object into the robot scene.
_FOREIGN_MEMORY_PLACE = re.compile(
    r'(?:내|우리|제|저희|다른)\s*(?:방|집|거실|부엌|주방|침실|서랍|책장|곳|장소)(?:의|에|에서|에\s*있는|에\s*놓인)?|'
    r'\b(?:(?:my|our|your|his|her|their|another|different)\s+)?'
    r'(?:room|bedroom|kitchen|house|drawer|closet|bookshelf)\b', re.IGNORECASE | re.ASCII)
_KO_PERSONAL_PLACE = (
    r'(?<![가-힣])(?:(?:내\s*|나의\s*|제\s+|저의\s*|우리(?:의)?\s*|저희(?:의)?\s*)'
    r'(?:자리|좌석|위치)|'
    r'(?P<personal_subject>내가|제가|우리가|저희가|나는|저는|우리는|저희는)\s*'
    r'(?P<personal_descriptor>(?:[가-힣0-9]{1,16}\s+){0,3}[가-힣0-9]{1,16})\s*(?:자리|좌석|위치)|'
    r'(?:(?:네가|당신이|사람이)\s*'
    r'(?:(?:지금|방금|자주|늘|항상|가끔|잠시|잠깐)\s*){0,2})?'
    r'앉(?:은|는|을|던|았던|아\s*있(?:는|던|었던|을)|고\s*있는)\s*(?:자리|좌석|위치))')
_KO_PERSONAL_MEMORY_PLACE = re.compile(_KO_PERSONAL_PLACE)
# The user confirmed this specific Whisper rendering of "놓인 네 자리의 배치".
# Anchor the entire memory target so another seat or a partial layout cannot
# accidentally become the four robot slots.
_WHISPER_FOUR_SLOT_TARGET = (
    r'\s*(?:(?:지금|현재|방금|먼저|우선|이제)\s+)*(?:(?:원통|물체)(?:들)?[이가]\s+)?'
    r'(?P<scene>(?:노인|놓인)\s+내\s*자리(?:에|의)?\s*배치(?:를)?)\s*')
_WHISPER_FOUR_SLOT_REQUEST = re.compile(
    _WHISPER_FOUR_SLOT_TARGET + r'(?:기억|저장)(?:해(?:\s*(?:줘(?:요)?|주세요|주십시오|라|요))?|하세요)'
    r'[\s.!。！]*')


def is_whisper_four_slot_memory_request(text):
    return _WHISPER_FOUR_SLOT_REQUEST.fullmatch(text) is not None


_KO_PERSONAL_ROBOT_WHOLE_SCENE = re.compile(
    _KO_PERSONAL_PLACE + r'(?:의|에\s*(?:있는|놓인))\s*'
    r'로봇(?:의)?\s*(?:작업\s*공간(?:의)?\s*)?'
    r'(?:(?:현재|지금)(?:의)?\s*)?(?:(?:네|4)\s*자리(?:의)?\s*)?'
    r'(?:(?:전체|모든)\s*(?:(?:원통|물체|자리)(?:의)?\s*)?(?:배치|장면)|'
    r'(?:배치|장면)\s*전체)')
_PARTIAL_MEMORY_KO = re.compile(
    r'(?:[1-4]|일|이|삼|사)\s*번(?:째)?\s*(?:자리|위치)?\s*'
    r'(?:(?:만(?:의)?|의)\s*(?:배치|상태|위치|자리|원통|물체)?(?:을|를)?|'
    r'(?:자리|위치)?(?:을|를|도|은|는)|자리|위치)\s*$|'
    r'(?:일부|하나|한\s*자리|왼쪽|오른쪽|위쪽|아래쪽)\s*'
    r'(?:자리|위치|원통|물체)?\s*만\s*$|'
    r'(?:왼쪽|오른쪽|위쪽|아래쪽|좌측|우측)\s*'
    r'(?:원통|원기둥|물체)(?:의)?\s*(?:위치|자리|배치)(?:만|을|를|도|은|는)?\s*$')
_PARTIAL_MEMORY_EN_SLOT = re.compile(
    r'\b(?:slot|position|place)\s*(?:number\s*)?(?P<slot>[1-4]|one|two|three|four)\b',
    re.IGNORECASE | re.ASCII)
_PARTIAL_MEMORY_EN_ORDINAL = re.compile(
    r'\b(?:first|second|third|fourth|[1-4](?:st|nd|rd|th))\s+(?:slot|position|place)\b|'
    r'\b(?:only|just)\s+(?:(?:the|this|that|a)\s+)?(?:\w+\s+){0,2}'
    r'(?:slot|position|cylinder|object)\b|\b(?:slot|cylinder|object)\s+only\b',
    re.IGNORECASE | re.ASCII)
_KO_MEMORY_NUMBERED_SLOT = re.compile(r'(?P<slot>[1-4]|일|이|삼|사)\s*번(?:째)?')
_KO_MEMORY_ALL_SLOT_RANGE = re.compile(r'(?:1|일)\s*번\s*(?:부터|에서)\s*(?:4|사)\s*번\s*까지')
_KO_MEMORY_PRECEDING_ACTION = re.compile(
    r'(?:쌓|풀|놓|올리|내리|옮기|바꾸|교환하)(?:고|기\s*전(?:에)?|은\s*(?:뒤|후)|한\s*(?:뒤|후))|'
    r'(?:올려|내려|옮겨|바꿔)(?:서|놓고)')
_KO_MEMORY_REFERENT = re.compile(
    r'(?P<referent>[가-힣A-Za-z0-9]+?)(?:의)?\s+'
    r'(?:위치|배치|자리|장면|상태)(?:을|를|만|도|은|는)?\s*$')
_KO_SCENE_REFERENT = re.compile(
    r'^(?:현재|지금|여기|그|이|저|원통|원기둥|물체|로봇|장면|배치|상태|자리|자리별|위치|각|네|전체|모든)$')
_EN_MEMORY_OBJECT_IDS = re.compile(
    r'\b(?:cylinders?|objects?|pieces?)\s*(?:numbers?\s*)?'
    r'(?P<ids>(?:[1-4]|one|two|three|four)(?:\s*(?:,|and|&)\s*'
    r'(?:[1-4]|one|two|three|four))*)\b', re.IGNORECASE | re.ASCII)
_EN_MEMORY_REFERENT = re.compile(
    r'\b(?:layout|arrangement|scene|state|positions?|locations?)\s+of\s+'
    r'(?:(?:the|my|our|your|this|all|four|both|[1-4])\s+)*'
    r'(?P<referent>[a-z]+)\b', re.IGNORECASE | re.ASCII)
_EN_SCENE_REFERENT = re.compile(
    r'^(?:cylinders?|objects?|pieces?|slots?|positions?|places?|stacks?|scene|robot)$',
    re.IGNORECASE | re.ASCII)
_KO_MEMORY_OBJECT = re.compile(r'(?P<object>[가-힣A-Za-z0-9]+?)(?:을|를|만|도|은|는)\s*$')
_KO_MEMORY_BARE_OBJECT = re.compile(r'(?P<object>[가-힣A-Za-z0-9]+)\s*$')
_KO_SCENE_MEMORY_OBJECT = re.compile(
    r'(?:배치|장면|상태|위치|자리|모습|배열|원통|물체|그것|이것|요것|저것|그거|이거|저거|전부|모두|전체|'
    r'그대로)$|^(?:지금|현재|여기|다|응|네)$')
_EN_SCENE_MEMORY_HEAD = re.compile(
    r'^(?:(?:the|this|that|a|an|current|present|whole|entire|all|every|four|both|our|robot[\'’]s)\s+)*'
    r'(?:layout|arrangement|scene|state|slots?|positions?|locations?|placement|configuration|'
    r'cylinders?|objects?|things?|it|this|that)\b', re.IGNORECASE | re.ASCII)
_EN_MEMORY_SCENE_EXPLANATION = re.compile(
    r'^(?:how|where)\b.*\b(?:positions?|slots?|cylinders?|objects?|arranged|layout|scene)\b',
    re.IGNORECASE | re.ASCII)
_EN_RESULTING_SCENE = re.compile(
    r'(?P<target>(?:the\s+)?resulting\s+(?:arrangement|layout))(?:\s+afterwards?)?',
    re.IGNORECASE | re.ASCII)
_EN_PAST_MEMORY_TARGET = re.compile(
    r'\bfrom\s+(?:yesterday|earlier|last\s+(?:night|week|month|year))\b',
    re.IGNORECASE | re.ASCII)
_COMPOSITE_CONNECTOR = re.compile(
    r'기\s*(?:[가-힣0-9]{1,8}\s*){0,3}(?:전(?:에)?|에\s*앞서)|(?:한|은|인|긴|꾼)\s*(?:뒤|후|다음)|'
    r'(?:기억하|기억해|옮기|옮겨|이동하|바꾸|바꿔|교환하)고|'
    r'\b(?:before|after|then|and)\b|[;.!?。？！]', re.IGNORECASE | re.ASCII)
_KOREAN_IMPERATIVE = re.compile(
    r'(?:기억|저장)(?:해|하세요|하라)|옮(?:겨|기세요)|이동(?:해|하세요)|'
    r'바꿔|바꾸세요|교환(?:해|하세요)', re.IGNORECASE | re.ASCII)
_EN_SPEAKER_OR_POSSIBILITY = re.compile(
    r'\b(?:(?:i|we|he|she|they)\s+(?:(?:will|would|may|might|can|could|shall)\s+)?'
    r'(?:(?:personally|myself|ourselves)\s+)?|'
    r'(?:i\s+am|we\s+are|he\s+is|she\s+is|they\s+are)\s+going\s+to\s+|'
    r'i[\'’]ll\s+|(?:will|would|may|might|can|could|should)\s+|'
    r'(?:how|way|ability|plan)\s+to\s+|(?:say|quote|mention|explain)\s+)$',
    re.IGNORECASE | re.ASCII)
_KO_PERSON = re.compile(r'내가|제가|나는|저는|우리가|저희가|너가|네가|당신이|로봇이')
_KOREAN_CONDITIONAL = re.compile(
    r'(?:바꾸|바뀌|옮기|이동하|쌓|풀|기억하|저장하|비어\s*있|비|있|없)(?:면|으면|다면)')
_EN_CONDITIONAL = re.compile(r'\b(?:if|unless|when)\b', re.IGNORECASE | re.ASCII)
_SLOT_PRESERVE_EN = re.compile(
    r'\b(?P<verb>leave|keep|preserve)\s+(?:the\s+)?'
    r'(?:contents\s+of\s+)?' + _EN_SLOT_NAME + _EN_CARDINAL +
    r'\s+(?:unchanged|intact|untouched|as\s+(?:it\s+is|is)|in\s+place)\b|'
    r'\b' + _EN_SLOT_NAME + _EN_CARDINAL + r'\s+(?:must|should)\s+'
    r'(?P<verb2>remain|stay)\s+(?:unchanged|intact|untouched)\b', re.IGNORECASE | re.ASCII)
_SLOT_PRESERVE_KO = re.compile(
    _KO_CARDINAL + r'\s*번(?:째)?\s*(?:자리|위치)?(?:은|는|을|를|이|가|에)?\s*'
    r'(?:(?:원통|물체|내용물)(?:은|는|을|를)?\s*)?'
    r'(?:원래|원상태|그대로|변경\s*없이)\s*(?P<verb>둬|두|놔|유지|보존)|'
    + _KO_CARDINAL + r'\s*번(?:째)?\s*(?:자리|위치)?(?:은|는|을|를|이|가|에)?\s*'
    r'(?P<verb2>건드리지\s*(?:마|말)|바꾸지\s*(?:마|말)|변경하지\s*(?:마|말))')


def _explicit_span(text, start, end, word):
    """Exclude quotations, negation, questions, other speakers and descriptions."""
    if (_korean_dependent_action(text, start, end) or v1._quoted(text, start, end) or v1._negated_action(text, start, end)
            or v1._reported_action(text, start)
            or v1._question_only(text, start, end) or v1._negated_span(text, start, end)):
        return False
    prefix = text[max(0, start - 80):start]
    suffix = text[end:min(len(text), end + 35)]
    if _EN_SPEAKER_OR_POSSIBILITY.search(prefix) or re.search(
            r'\b(?:not|without|instead\s+of|rather\s+than)\s*$', prefix, re.IGNORECASE | re.ASCII):
        return False
    if re.match(r'(?:는|는\s*방법|는\s*것|ㄹ\s*수|할\s*수|할게|겠습니다)', suffix):
        return False
    if re.search(r'(?:방법|설명|질문|표현)\s*(?:을|를)?\s*$', prefix):
        return False
    # A newly stated second-person subject overrides an earlier first-person
    # subject. A first-person action itself is not a robot instruction.
    people = list(_KO_PERSON.finditer(prefix))
    if people and people[-1].group() in ('내가', '제가', '나는', '저는', '우리가', '저희가'):
        return False
    if word.lower().endswith('ing'):
        # The supported action vocabulary includes base verbs such as
        # "bring".  Recognise those whole tokens before treating an -ing
        # form as a description; "bringing" remains a participle.
        base_action = (_ARM_HOME_VERB.fullmatch(word) is not None or any(
            re.fullmatch(pattern, word, re.IGNORECASE | re.ASCII)
            for pattern in v1._TASK_WORDS.values()))
        if not base_action and not re.search(r'\b(?:before|after)\s*$', prefix, re.IGNORECASE | re.ASCII):
            return False
    return True


def _explicit_action(text, match):
    return _explicit_span(text, *match.span(), match.group())


def _validate_move_scope(step, text, start, end):
    """Bind a proposed transfer to one affirmative, unquoted action clause.

    Evidence may span a quotation and another instruction.  Numbers from the
    quoted description must not provide the source or destination for the
    latter instruction, and a nominal discussion of moving is not a command.
    """
    kind = step['kind']
    if kind not in ('move', 'stack', 'unstack', 'swap'):
        return
    verbs = list(re.finditer(v1._TASK_WORDS[kind], step['evidence'], re.IGNORECASE | re.ASCII))
    quoted = ''.join(step['evidence'][left:right] for left, right in v1._quote_spans(step['evidence']))
    quoted_slots, quoted_sources, quoted_destinations, quoted_objects, _, _ = v1._task_supports(quoted)
    rejected_english = any(re.search(r'[A-Za-z]', verb.group()) and
                           (v1._negated_action(text, start + verb.start(), start + verb.end()) or
                            v1._quoted(text, start + verb.start(), start + verb.end()) or
                            v1._reported_action(text, start + verb.start())) for verb in verbs)
    rejected_korean = any(_korean_dependent_action(text, start + verb.start(), start + verb.end())
                          for verb in verbs)
    bind_local_values = bool(quoted_slots or quoted_sources or quoted_destinations or quoted_objects
                             or rejected_english or rejected_korean)
    if kind != 'move' and not bind_local_values:
        return
    for verb in verbs:
        left, right = start + verb.start(), start + verb.end()
        if not _explicit_span(text, left, right, verb.group()):
            continue
        suffix = text[right:]
        if ((verb.group() == '옮' and re.match(r'(?:기는|긴(?!\s*(?:뒤|후|다음))|길|김|기\s*(?:를|가|는|것|방법|횟수))', suffix))
                or (verb.group() == '이동' and re.match(r'(?:하는|할\s*(?:동작|횟수|방법)|하기\s*(?:를|의))', suffix))):
            continue
        # With a rejected or quoted action, whole-utterance numbers cannot
        # supply values to a different affirmative verb.
        if not bind_local_values:
            return
        # These supported phrasings bind their two slots as one expression,
        # including the "unstack ... and put it" anaphora.
        direct = {'stack': _EN_STACK_ON, 'unstack': _EN_UNSTACK_TOP,
                  'swap': _EN_SWAP_POSITIONS}.get(kind)
        if direct is not None:
            for match in direct.finditer(text, start, end):
                if match.start('verb') != left or not _explicit_match(text, match):
                    continue
                values = tuple(v1.bilingual._value(match.group(name))
                               for name in ('source', 'destination'))
                proposed = (step['source_slot'], step['destination_slot'])
                if (set(values) == set(proposed) if kind == 'swap' else values == proposed):
                    return
        clause, offset = _task_clause(text, left, right)
        slots, sources, destinations, objects, _, _ = v1._task_supports(
            _mask_quoted(clause), whole=text, offset=offset)
        if kind == 'swap':
            sources = destinations = slots
        if (step['source_slot'] is not None and step['source_slot'] not in sources or
                step['destination_slot'] is not None and step['destination_slot'] not in destinations or
                step['object_id'] is not None and step['object_id'] not in objects):
            continue
        return
    v1._fail('unresolved_reference', 'move values need the same unquoted affirmative action clause')


def _personal_memory_places(clause):
    """Find personal places, including a first-person relative clause.

    A Korean verb modifying a following noun commonly ends in ㄴ or ㄹ.
    Quantifiers such as "모든" and "네" do not by themselves make a
    place personal; an earlier personal relative predicate still can.
    This bounded check does not attempt general Korean morphological parsing.
    """
    places = []
    for place in _KO_PERSONAL_MEMORY_PLACE.finditer(clause):
        descriptor = place.group('personal_descriptor')
        if descriptor is not None:
            words = descriptor.split()
            while words and words[-1] in ('내', '나의', '제', '저의', '우리', '우리의',
                                          '저희', '저희의', '모든', '네', '4'):
                words.pop()
            if not words:
                continue
            last = ord(words[-1][-1])
            if not (0xAC00 <= last <= 0xD7A3 and (last - 0xAC00) % 28 in (4, 8)):
                continue
        places.append(place)
    return places


def _resulting_scene_follows_action(text, memory, reference, steps):
    """Bind the requested result to one explicit scene change in this program."""
    actions = _required_actions(text)
    if len(actions) != 1:
        return False
    kind, action = actions[0]
    if [(step['op'], step.get('kind')) for step in steps] != [('task', kind), ('task', 'remember')]:
        return False
    target_end = text.index(reference, memory.end()) + len(reference)
    if text[memory.start():target_end] not in steps[1]['evidence']:
        return False
    motion_first = action.start() < memory.start()
    events = sorted([(kind, *action.span()), ('remember', *memory.span())], key=lambda item: item[1])
    order = _composite_before(text, *events)
    if order is not None:
        return order == motion_first
    # In “move … and remember the resulting layout”, the result itself refers
    # to that move. Mentioning the memory first without an order is ambiguous.
    return motion_first and re.search(r'\band\s+(?:please\s+)?$', text[action.end():memory.start()],
                                      re.IGNORECASE | re.ASCII) is not None


def _korean_resulting_scene_follows_action(text, memory, target, steps):
    """Bind a complete Korean result target to one preceding scene change."""
    actions = _required_actions(text)
    if len(actions) != 1:
        return False
    kind, action = actions[0]
    if [(step['op'], step.get('kind')) for step in steps] != [('task', kind), ('task', 'remember')]:
        return False
    if not action.end() <= target.start() < target.end() <= memory.start():
        return False
    if not re.fullmatch(r'(?:긴|한|은)\s*(?:뒤|후)\s*', text[action.end():target.start()]):
        return False
    # This narrowly supported form refers to the whole resulting scene.
    # Reject additional suffixes: a later qualifier such as "단 3번만"
    # can change that scope, and must not be silently ignored.
    if not re.fullmatch(r'(?:해(?:\s*(?:주세요|줘|주라|두세요|둬))?|하세요|하라|하자|해요)[\s.!?]*',
                        text[memory.end():]):
        return False
    return text[target.start():memory.end()] in steps[1]['evidence']


def _unsupported_memory_scope(text, match, steps):
    """Detect explicit non-scene or partial targets in the memory verb's clause.

    This is a guard, not an ASR confidence test or a general language parser.
    It never supplies a target and does not inspect private simulator state.
    """
    clause, offset = _task_clause(text, *match.span())
    target_end = match.start() - offset
    target = clause[:target_end]
    alias = re.fullmatch(_WHISPER_FOUR_SLOT_TARGET, target)
    if alias and any(step['op'] == 'task' and step['kind'] == 'remember'
                     and clause[alias.start('scene'):match.end() - offset] in step['evidence'] for step in steps):
        # Equal-width substitutions on this private view preserve every
        # original evidence position and the independent direct-speech guards.
        target = target.replace('노인', '놓인').replace('내', '네').replace('자리에', '자리의')
        clause = target + clause[target_end:]
    places = _personal_memory_places(clause)
    if not _explicit_action(text, match):
        # "내가 쓰는 자리" has a first-person subject for the place, not for
        # the later memory request. Mask just that relative-clause subject
        # while checking whether the memory verb is affirmative; keep all
        # offsets, quotation and negation and validate the original text.
        masked = list(text)
        masked_subject = False
        for place in places:
            if place.group('personal_subject') is not None:
                start, end = offset + place.start('personal_subject'), offset + place.end('personal_subject')
                masked[start:end] = ' ' * (end - start)
                masked_subject = True
        if not masked_subject or not _explicit_span(''.join(masked), *match.span(), match.group()):
            return False
    before = clause[:match.start() - offset]
    after = clause[match.end() - offset:]
    # A personal or seated place is not evidence of all four robot slots.
    # Only an explicitly connected whole robot scene can disambiguate it.
    for place in places:
        start, end = offset + place.start(), offset + place.end()
        if v1._quoted(text, start, end) or v1._negated_span(text, start, end):
            continue
        whole = _KO_PERSONAL_ROBOT_WHOLE_SCENE.match(clause, place.start())
        if whole is None or v1._negated_span(text, offset + whole.start(), offset + whole.end()):
            return True
    # An explicit unrelated place changes the referent even if the request
    # happens to contain the word "layout". Ignore independently negated or
    # quoted mentions (including a rejected target in a correction).
    for place in _FOREIGN_MEMORY_PLACE.finditer(clause):
        start, end = offset + place.start(), offset + place.end()
        if not v1._quoted(text, start, end) and not v1._negated_span(text, start, end):
            return True
    if re.search(r'[가-힣]', match.group()):
        # Treat exactly this result expression separately; do not mask the
        # preceding transfer for other targets, which could be partial scenes.
        resulting = re.search(r'바뀐\s+배치(?:를|을)?\s*$', text[:match.start()])
        if resulting:
            return not _korean_resulting_scene_follows_action(text, match, resulting, steps)
        target = re.sub(
            r'\s*(?:(?:우선|먼저|꼭|잘|다시|한\s*번|이제|바로|일단|계속|당장|그냥|제발|'
            r'새로|확실히|정확히|잠깐|잠시|좀)\s*)+$', '', before).strip()
        target = re.sub(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<![A-Za-z])\'[^\']*\'',
                        lambda quoted: ' ' * len(quoted.group()), target)
        # Numbered sources in an earlier completed motion are not the object
        # of a following bare "remember" command.
        actions = list(_KO_MEMORY_PRECEDING_ACTION.finditer(target))
        if actions:
            target = target[actions[-1].end():].strip()
        if _PARTIAL_MEMORY_KO.search(target):
            return True
        numbered = {item.group('slot') for item in _KO_MEMORY_NUMBERED_SLOT.finditer(target)}
        slot_values = {int(item) if item.isdecimal() else v1.common._NUMBERS[item] for item in numbered}
        if slot_values and slot_values != {1, 2, 3, 4} and not _KO_MEMORY_ALL_SLOT_RANGE.search(target):
            return True
        referent = _KO_MEMORY_REFERENT.search(target)
        if (referent and not _KO_SCENE_REFERENT.fullmatch(referent.group('referent')) and
                not re.search(r'(?:인|된|던|진|한)$', referent.group('referent'))):
            return True
        object_match = _KO_MEMORY_OBJECT.search(target)
        if object_match and not _KO_SCENE_MEMORY_OBJECT.search(object_match.group('object')):
            start, end = offset + object_match.start('object'), offset + object_match.end('object')
            if not v1._quoted(text, start, end) and not v1._negated_span(text, start, end):
                return True
        # Korean often omits the object particle ("점심 메뉴 기억해").
        # Do not treat a preceding action/connective as an object.
        if not object_match:
            bare = _KO_MEMORY_BARE_OBJECT.search(target)
            if bare:
                word = bare.group('object')
                if (not _KO_SCENE_MEMORY_OBJECT.search(word)
                        and not re.search(r'(?:고|서|뒤|후|다음|전|때|면|그리고|부터)$', word)
                        and not any(pattern.search(word) for pattern in
                                    (_REQUIRED_MOVE, _REQUIRED_SWAP, _REQUIRED_RETURN))):
                    start, end = offset + bare.start('object'), offset + bare.end('object')
                    if not v1._quoted(text, start, end) and not v1._negated_span(text, start, end):
                        return True
        return False
    target = after.strip()
    # Slot numbers in a later transfer are not part of the memory target.
    target = re.split(
        r',\s*(?:then\s+)?(?:move|transfer|stack|unstack|swap|exchange|place|put)\b|'
        r'\b(?:before|after)\s+(?:you\s+)?(?:move|transfer|stack|unstack|swap|exchange|place|put)\b',
        target, maxsplit=1, flags=re.IGNORECASE | re.ASCII)[0].strip()
    if not target:
        return False  # Bare "Remember." means the current registered scene.
    target_offset = match.end() + len(after) - len(after.lstrip())
    if any(not v1._quoted(text, target_offset + item.start(), target_offset + item.end())
           and not v1._negated_span(text, target_offset + item.start(), target_offset + item.end())
           for item in _EN_PAST_MEMORY_TARGET.finditer(target)):
        return True
    slots = {item.group('slot').lower() for item in _PARTIAL_MEMORY_EN_SLOT.finditer(target)}
    # Explicitly naming only one to three slots is not a request to save all
    # four.  A request naming all four remains eligible for the normal guard.
    values = {v1.bilingual._EN_VALUES.get(slot, int(slot) if slot.isdecimal() else None) for slot in slots}
    if values and values != {1, 2, 3, 4}:
        return True
    if _PARTIAL_MEMORY_EN_ORDINAL.search(target):
        return True
    for match_ids in _EN_MEMORY_OBJECT_IDS.finditer(target):
        ids = re.findall(r'\b(?:[1-4]|one|two|three|four)\b', match_ids.group('ids'), re.IGNORECASE | re.ASCII)
        values = {int(item) if item.isdecimal() else v1.bilingual._EN_VALUES[item.lower()] for item in ids}
        if values != {1, 2, 3, 4}:
            return True
    if any(not _EN_SCENE_REFERENT.fullmatch(item.group('referent'))
           for item in _EN_MEMORY_REFERENT.finditer(target)):
        return True
    target = re.sub(r'^(?:please\s+|just\s+|now\s+)*', '', target, flags=re.IGNORECASE | re.ASCII)
    if not target:
        return False
    resulting = _EN_RESULTING_SCENE.fullmatch(target)
    if resulting:
        return not _resulting_scene_follows_action(text, match, resulting.group('target'), steps)
    # A named non-scene object such as "the lunch menu" cannot be silently
    # replaced with a whole-scene memory. Common direct and deictic scene
    # wordings remain supported. Generic ambiguous language is not repaired.
    return not (_EN_SCENE_MEMORY_HEAD.search(target) or _EN_MEMORY_SCENE_EXPLANATION.search(target))


def _explicit_match(text, match):
    if v1._quoted(text, *match.span()):
        return False
    for name in ('verb', 'put'):
        if name in match.re.groupindex and match.group(name) is not None:
            if not _explicit_span(text, *match.span(name), match.group(name)):
                return False
    for name in ('source', 'destination'):
        if name in match.re.groupindex and v1._negated_span(text, *match.span(name)):
            return False
    return True


def _validate_arm_home_scope(step, text, start, end):
    """Require an affirmative home action for the robot in the source clause.

    The reference itself remains an exact substring of the evidence.  This
    guard only rejects unsupported proposals; it neither creates a home step
    nor claims that the controller can execute one.
    """
    reference = step['target']['reference']
    evidence = step['evidence']
    if reference not in evidence:
        v1._fail('ungrounded_value', 'return reference is not supported by evidence')
    reference_spans = [(start + match.start(), start + match.end())
                       for match in re.finditer(re.escape(reference), evidence)]
    for cue in _ARM_HOME_CUE.finditer(text, start, end):
        if v1._quoted(text, *cue.span()):
            continue
        # Keep a conditional that precedes a comma in scope: "if it drops,
        # send the arm home" must not become an unconditional home command.
        sentence_left = max((text.rfind(mark, 0, cue.start()) for mark in '.!?;。？！'), default=-1) + 1
        sentence_right = min((pos for mark in '.!?;。？！' if (pos := text.find(mark, cue.end())) >= 0),
                             default=len(text))
        if any(not v1._quoted(text, *condition.span())
               for condition in _ARM_HOME_CONDITION.finditer(text, sentence_left, sentence_right)):
            continue
        verbs = list(_ARM_HOME_VERB.finditer(text, sentence_left, sentence_right))
        verbs.sort(key=lambda item: min(abs(item.start() - cue.end()), abs(cue.start() - item.end())))
        for verb in verbs:
            if (abs(verb.start() - cue.start()) > 100 or
                    not _explicit_span(text, *verb.span(), verb.group())):
                continue
            first, last = min(verb.start(), cue.start()), max(verb.end(), cue.end())
            boundaries = list(_ARM_HOME_FRAGMENT_BOUNDARY.finditer(text, sentence_left, sentence_right))
            if any(first < boundary.start() and boundary.end() < last for boundary in boundaries):
                continue
            left = max((boundary.end() for boundary in boundaries if boundary.end() <= first),
                       default=sentence_left)
            right = min((boundary.start() for boundary in boundaries if boundary.start() >= last),
                        default=sentence_right)
            # A second, valid home command elsewhere must not make a
            # reference to a different person's arm in an earlier clause
            # appear grounded.  The selected reference must actually overlap
            # this action's local fragment.
            if not any(ref_left < right and left < ref_right
                       for ref_left, ref_right in reference_spans):
                continue
            foreign = [target for target in _ARM_HOME_FOREIGN_TARGET.finditer(text, left, right)
                       if not v1._quoted(text, *target.span())
                       and not v1._negated_span(text, *target.span())]
            if not foreign:
                return
    v1._fail('unsupported_action', 'arm home needs an affirmative, unconditional robot action')


def _memory_instruction(text, match):
    if not _explicit_action(text, match):
        return False
    if any(candidate.start('verb') == match.start() and _explicit_match(text, candidate)
           for candidate in _EN_NOW_SCENE_MEMORY.finditer(text)):
        return True
    clause, _ = _task_clause(text, *match.span())
    if not _MEMORY_TARGET.search(clause):
        return False
    if re.match(r'\s+to\s+(?:move|transfer|swap|exchange|return|restore)\b', text[match.end():],
                re.IGNORECASE | re.ASCII):
        return False
    return re.match(r'\s+(?:that\s+)?(?:the\s+)?(?:layout|arrangement|scene)\s+'
                    r'(?:can|could|may|might|will|is|was)\b', text[match.end():],
                    re.IGNORECASE | re.ASCII) is None


def _bare_memory_imperative(text, match):
    if not _explicit_action(text, match):
        return False
    suffix = text[match.end():]
    if re.match(r'(?:해|하세요|하라)', suffix):
        return True
    return bool(re.match(r'\s*(?:please\s*)?(?:[.,!?。？！]|$)', suffix, re.IGNORECASE | re.ASCII))


def _required_actions(text):
    # The explicit task must name its slots; mere discussion of a motion is
    # insufficient to create a requirement on the proposed program.
    tower = _KO_UNSTACK_TO_EMPTY.fullmatch(text)
    if tower:
        # “내려 탑을 풀어” describes a single top-cylinder transfer.
        return [('unstack', _KO_TOP_UNSTACK_ACTION.match(text, tower.start('verb')))]
    object_stacks = list(_explicit_object_stacks(text))
    actions = [('stack', match) for match in object_stacks]
    for kind, pattern in (('move', _COVERAGE_MOVE), ('stack', _REQUIRED_STACK),
                          ('unstack', _REQUIRED_UNSTACK), ('swap', _REQUIRED_SWAP)):
        for match in pattern.finditer(text):
            if any(named.start('verb') == match.start() for named in object_stacks):
                continue
            if not _explicit_action(text, match):
                continue
            if (match.group().lower() == 'place' and re.search(
                    r'\b(?:a|the|this|that|in|at|from|to)\s+$',
                    text[max(0, match.start() - 25):match.start()], re.IGNORECASE | re.ASCII)):
                continue  # "place" can also be a location noun.
            clause, offset = _task_clause(text, *match.span())
            slots, sources, destinations, objects, _, _ = v1._task_supports(clause, whole=text, offset=offset)
            if kind == 'swap' and len(slots) >= 2:
                actions.append((kind, match))
            elif kind in ('move', 'stack', 'unstack') and destinations and (sources or objects):
                actions.append((kind, match))
    for kind, pattern in (('move', _EN_TOP_TOWER_MOVE), ('stack', _EN_STACK_ON), ('unstack', _EN_UNSTACK_TOP),
                          ('swap', _EN_SWAP_POSITIONS)):
        for match in pattern.finditer(text):
            if (_explicit_match(text, match) and
                    not any(existing_kind == kind and existing.start() == match.start()
                            for existing_kind, existing in actions)):
                actions.append((kind, match))
    return actions


def _required_intents(text):
    # The supported standalone "기억해/Remember" command means writing the
    # registered current arrangement; keep that meaning after another action.
    memories = [match for match in _REQUIRED_MEMORY.finditer(text)
                if _memory_instruction(text, match) or _bare_memory_imperative(text, match)]
    if not memories or not _COMPOSITE_CONNECTOR.search(text):
        return None
    actions = _required_actions(text)
    if not actions:
        return None
    # A Korean subordinate phrase alone is descriptive; one of the paired
    # actions must contain an actual imperative ending.
    if re.search(r'[가-힣]', text) and not _KOREAN_IMPERATIVE.search(text):
        return None
    if len(memories) != 1:
        v1._fail('unresolved_reference', 'multiple explicit memory operations require clarification')
    return actions, memories[0]


def _required_pair(text):
    intents = _required_intents(text)
    if intents is None or len(intents[0]) != 1:
        return None
    (kind, action), memory = intents[0][0], intents[1]
    return kind, action, memory


def _require_pair_order(text, program, pair, *, action_index=None, memory_index=None):
    kind, action, memory = pair
    action_steps = [i for i, step in enumerate(program['steps'])
                    if step['op'] == 'task' and step['kind'] == kind]
    memory_steps = [i for i, step in enumerate(program['steps'])
                    if step['op'] == 'task' and step['kind'] == 'remember']
    if not action_steps or not memory_steps:
        v1._fail('unresolved_reference', 'an explicit operation in the composite request was omitted')
    if action_index is not None:
        action_steps = [action_index]
    if memory_index is not None:
        memory_steps = [memory_index]
    if len(action_steps) != 1 or len(memory_steps) != 1:
        v1._fail('unresolved_reference', 'composite operation identity is ambiguous')
    # Only unambiguous temporal markers are checked here; ambiguous order
    # continues through the existing context and execution validation.
    # The coverage matcher records the stem "옮", so "옮기기 전에" has
    # two 기 characters after that match; "이동하기 전에" has 하기도.
    before_action = re.match(r'(?:기기|하기|기)?\s*전', text[action.end():].lstrip()) is not None
    before_memory = re.match(r'(?:하)?기\s*전', text[memory.end():].lstrip()) is not None
    earlier, later = sorted((action, memory), key=lambda match: match.start())
    between = text[earlier.end():later.start()]
    prefix = text[:earlier.start()]
    before = re.search(r'\bbefore\b', prefix + between, re.IGNORECASE | re.ASCII)
    after = re.search(r'\bafter\b', prefix + between, re.IGNORECASE | re.ASCII)
    action_first = None
    def korean_after(match):
        suffix = text[match.end():]
        if match.group() == '바꾼' and re.match(r'\s*(?:뒤|후|다음)', suffix):
            return True
        return re.match(r'(?:긴|한|은)\s*(?:뒤|후|다음)|(?:기고|고\s*나서|겨서|고\s)', suffix) is not None

    if before_action:
        action_first = False
    elif before_memory:
        action_first = True
    elif korean_after(action):
        action_first = True
    elif korean_after(memory):
        action_first = False
    elif before:
        action_first = ((earlier is action) if before.start() >= len(prefix) else (later is action))
    elif after:
        action_first = ((later is action) if after.start() >= len(prefix) else (earlier is action))
    elif re.search(r'\bthen\b|(?:하고|옮기고|옮겨서|바꾸고|바꿔서)', between, re.IGNORECASE | re.ASCII):
        action_first = earlier is action
    elif re.search(r'[.!。！]', between):
        # Separate positive instructions are issued in their spoken order.
        action_first = earlier is action
    if action_first is not None and ((action_steps[0] < memory_steps[0]) != action_first):
        v1._fail('unresolved_reference', 'composite operations are in the wrong stated order')


def _validate_completed_swap(step, text, start, exc):
    """Cover v1's missing Korean past-link form without modifying its input."""
    if (step.get('op') != 'task' or step.get('kind') != 'swap' or exc.code != 'unsupported_action'
            or str(exc) != 'task evidence does not affirm this action'):
        return False
    intents = _required_intents(text)
    if intents is None:
        return False
    matches = [action for kind, action in intents[0]
               if kind == 'swap' and action.group() == '바꾼'
               and start <= action.start() and action.end() <= start + len(step['evidence'])]
    if len(matches) != 1:
        return False
    action = matches[0]
    if (not (start <= action.start() and action.end() <= start + len(step['evidence']))
            or not re.match(r'\s*(?:뒤|후|다음)', text[action.end():])):
        return False
    if v1._contains_any(step['evidence'], v1._TASK_UNRESOLVED) or v1._contains_any(text, v1._INJECTED_INSTRUCTION):
        v1._fail('unresolved_reference', 'task includes an unknown description or an instruction override')
    slots = v1._task_supports(step['evidence'], whole=text, offset=start)[0]
    if (step['source_slot'] not in slots or step['destination_slot'] not in slots
            or step['source_slot'] == step['destination_slot']
            or any(step[key] is not None for key in ('object_id', 'count', 'height', 'layout'))):
        v1._fail('ungrounded_value', 'completed swap needs the two explicitly stated slots')
    return True


def _top_tower_context_is_supported(text, match):
    """Accept the direct move alone or with one explicit separate memory step."""
    prefix, suffix = text[:match.start()], text[match.end():]
    end = r'\s*[.!]?\s*'
    if not prefix.strip() and re.fullmatch(end, suffix):
        return True
    flags = re.IGNORECASE | re.ASCII
    companion = r'(?:' + _EN_CURRENT_LAYOUT_SENTENCE + r'|' + _EN_REMEMBERED_LAYOUT_SENTENCE + r')'
    before = re.fullmatch(r'\s*' + companion + r'\s*[.!]\s*', prefix, flags)
    after = re.fullmatch(r'\s*[.!]\s*' + companion + end, suffix, flags)
    return bool((before and re.fullmatch(end, suffix)) or (not prefix.strip() and after))


def _validate_direct_scene_expression(step, text, start, exc):
    """Recognise specific wordings v1 cannot parse without changing its input."""
    if step.get('op') != 'task':
        return False
    kind = step['kind']
    patterns = {'move': _EN_TOP_TOWER_MOVE, 'stack': _EN_STACK_ON, 'unstack': _EN_UNSTACK_TOP,
                'swap': _EN_SWAP_POSITIONS, 'remember': _EN_NOW_SCENE_MEMORY}
    pattern = patterns.get(kind)
    expected_error = ((kind in ('move', 'stack') and exc.code == 'ungrounded_value' and
                       str(exc) == 'destination_slot is not supported by its evidence') or
                      (kind == 'swap' and exc.code == 'ungrounded_value' and
                       str(exc) in ('source_slot is not supported by its evidence',
                                    'destination_slot is not supported by its evidence')) or
                      (kind in ('unstack', 'remember') and exc.code == 'unresolved_reference' and
                       str(exc) == 'task includes an unknown description or an instruction override'))
    if pattern is None or not expected_error:
        return False
    evidence = step['evidence']
    matches = [match for match in pattern.finditer(text, start, start + len(evidence))
               if _explicit_match(text, match)]
    if len(matches) != 1 or v1._contains_any(text, v1._INJECTED_INSTRUCTION):
        return False
    match = matches[0]
    if kind == 'move' and not _top_tower_context_is_supported(text, match):
        # Do not extrapolate a recognized phrase into other clauses.
        return False
    # Only the provably bound "put it" and temporal "right now" are
    # exempt from v1's unknown-reference/direction guard; inspect the
    # unmodified evidence and reject every other unknown mention.
    permitted = ({match.span('anaphora')} if 'anaphora' in match.re.groupindex else set())
    if any((start + unknown.start(), start + unknown.end()) not in permitted
           for unknown in re.finditer(v1._TASK_UNRESOLVED, evidence, re.IGNORECASE | re.ASCII)):
        return False
    if any(step[key] is not None for key in ('object_id', 'count', 'height', 'layout')):
        return False
    if kind == 'remember':
        return step['source_slot'] is None and step['destination_slot'] is None
    source, destination = (v1.bilingual._value(match.group(name)) for name in ('source', 'destination'))
    if source == destination:
        return False
    if kind == 'swap':
        return {step['source_slot'], step['destination_slot']} == {source, destination}
    clause, offset = _task_clause(text, *match.span('verb'))
    counts, heights = v1._task_supports(clause, whole=text, offset=offset)[4:]
    return (not counts and not heights and step['source_slot'] == source and
            step['destination_slot'] == destination)


def _validate_move_with_bound_unstack_anaphora(step, text, start, exc):
    """Validate a later move against its own clause while retaining full evidence."""
    if (step.get('op') != 'task' or step['kind'] != 'move' or
            exc.code != 'unresolved_reference' or
            str(exc) != 'task includes an unknown description or an instruction override' or
            v1._contains_any(text, v1._INJECTED_INSTRUCTION)):
        return False
    evidence = step['evidence']
    bound = [match for match in _EN_UNSTACK_TOP.finditer(text, start, start + len(evidence))
             if _explicit_match(text, match)]
    if len(bound) != 1:
        return False
    permitted = bound[0].span('anaphora')
    if any((start + unknown.start(), start + unknown.end()) != permitted
           for unknown in re.finditer(v1._TASK_UNRESOLVED, evidence, re.IGNORECASE | re.ASCII)):
        return False
    clause, offset = _transfer_clause(step, text, start, start + len(evidence))
    if not any(_explicit_action(text, match) for match in _REQUIRED_MOVE.finditer(text, offset, offset + len(clause))):
        return False
    _, sources, destinations, objects, counts, heights = v1._task_supports(clause, whole=text, offset=offset)
    source, destination = step['source_slot'], step['destination_slot']
    if (destination is None or (source is None and step['object_id'] is None) or
            source == destination or step['layout'] is not None or
            (source is not None and source not in sources) or destination not in destinations or
            (step['object_id'] is not None and step['object_id'] not in objects) or
            (step['count'] is not None and step['count'] not in counts) or
            (counts and step['count'] not in counts) or
            (step['height'] is not None and step['height'] not in heights) or
            (heights and step['height'] not in heights)):
        return False
    return True


def _explicit_empty_slot_unstack(step, text, start, exc):
    """Verify a proposed single top-cylinder transfer on the original Korean evidence."""
    if (step.get('op') != 'task' or step['kind'] != 'unstack' or exc.code != 'ungrounded_value'
            or str(exc) != 'destination_slot is not supported by its evidence'):
        return False
    roles = explicit_top_unstack_roles(text, evidence=step['evidence'])
    return (roles == (step['source_slot'], step['destination_slot'], step['count'])
            and all(step[key] is None for key in ('object_id', 'height', 'layout')))


def _validate_explicit_arm_home_send(step, text, start, exc):
    if (step.get('op') != 'return' or step['target']['kind'] != 'arm_home' or
            exc.code != 'ungrounded_value' or str(exc) != 'return reference is not supported by evidence'):
        return False
    evidence, reference = step['evidence'], step['target']['reference']
    # The original exact reference must still be in evidence.  The expression
    # ends at the spoken verb, so a final sentence full stop may be outside
    # the regex match even though it is part of the exact transcript span.
    match_reference = reference.rstrip('.。')
    matches = [match for match in _KO_HOME_SEND.finditer(text, start, start + len(evidence))
               if (_explicit_match(text, match) and match_reference in match.group() and
                   ('대기' in match_reference or '팔' in match_reference))]
    if len(matches) != 1 or reference not in evidence or v1._negated_reference(evidence, reference):
        return False
    return True


def _require_multiple_actions(text, program, actions, memory=None):
    memory_steps = [i for i, step in enumerate(program['steps'])
                    if step['op'] == 'task' and step['kind'] == 'remember']
    if memory is not None and len(memory_steps) != 1:
        v1._fail('unresolved_reference', 'the explicit memory operation is missing or ambiguous')
    used, located = set(), []
    for kind, action in actions:
        clause, offset = _task_clause(text, *action.span())
        slots, sources, destinations, objects, _, _ = v1._task_supports(clause, whole=text, offset=offset)
        choices = []
        for index, step in enumerate(program['steps']):
            if kind == 'arm_home':
                if step['op'] != 'return' or step['target']['kind'] != 'arm_home':
                    continue
            elif step['op'] != 'task' or step['kind'] != kind:
                continue
            starts = (found.start() for found in re.finditer(re.escape(step['evidence']), text))
            action_start = action.start('arm') if action.re is _KO_HOME_SEND else action.start()
            if not any(evidence_start <= action_start and
                       action.end() <= evidence_start + len(step['evidence']) for evidence_start in starts):
                continue
            if kind == 'arm_home':
                pass
            elif action.re in _OBJECT_STACK_PATTERNS:
                if (_stack_object_roles(action) !=
                        (step['source_slot'], step['object_id'], step.get('destination_object_id'), step['count'])):
                    continue
            elif action.re in (_EN_TOP_TOWER_MOVE, _EN_STACK_ON, _EN_UNSTACK_TOP, _EN_SWAP_POSITIONS):
                source, destination = (v1.bilingual._value(action.group(name))
                                       for name in ('source', 'destination'))
                expected = (step['source_slot'], step['destination_slot'])
                if (set(expected) != {source, destination} if kind == 'swap'
                        else expected != (source, destination)):
                    continue
            elif kind == 'swap':
                if step['source_slot'] not in slots or step['destination_slot'] not in slots:
                    continue
            elif ((step['source_slot'] is not None and step['source_slot'] not in sources) or
                  (step['destination_slot'] is not None and step['destination_slot'] not in destinations) or
                  (step['object_id'] is not None and step['object_id'] not in objects)):
                continue
            choices.append(index)
        choices = [index for index in choices if index not in used]
        # Multiple identical steps are interchangeable only when each has
        # separately stated evidence, already checked by validate_program.
        if len(choices) > 1:
            meanings = {json.dumps({key: value for key, value in program['steps'][index].items()
                                    if key != 'evidence'}, ensure_ascii=False, sort_keys=True)
                        for index in choices}
            if len(meanings) == 1:
                choices = [min(choices)]
        if len(choices) != 1:
            v1._fail('unresolved_reference', 'an explicit action is missing or cannot be uniquely matched')
        index = choices[0]
        used.add(index)
        located.append((action, index))
        if memory is not None:
            _require_pair_order(text, program, (kind, action, memory),
                                action_index=index, memory_index=memory_steps[0])
    located.sort(key=lambda item: item[0].start())
    for (first, first_index), (second, second_index) in zip(located, located[1:]):
        between = text[first.end():second.start()]
        if (re.search(r'\b(?:then|followed\s+by)\b|\band\s*$|[.!。！]|'
                      r'(?:기고|긴\s*(?:뒤|후|다음)|꾼\s*(?:뒤|후|다음))',
                      between, re.IGNORECASE | re.ASCII) and first_index >= second_index):
            v1._fail('unresolved_reference', 'explicit successive actions are in the wrong order')
    return located


def _affirmed_preservation(text):
    for pattern in (_SLOT_PRESERVE_EN, _SLOT_PRESERVE_KO):
        for match in pattern.finditer(text):
            verb = next(name for name in ('verb', 'verb2') if match.group(name) is not None)
            start, end = match.span(verb)
            # "건드리지 마" is itself a positive preservation directive,
            # unlike "그대로 두지 마", which rejects preservation.
            negative_preserve = match.group(verb).startswith(('건드리지', '바꾸지', '변경하지'))
            if (v1._quoted(text, match.start(), match.end()) or v1._question_only(text, start, end)):
                continue
            if negative_preserve:
                if (re.search(r'(?:내가|제가|나는|저는)\s*.{0,25}$', text[max(0, start - 50):start]) or
                        re.match(r'\s*라고\s*(?:말|써|적|설명)', text[end:])):
                    continue
            elif not _explicit_span(text, start, end, match.group(verb)):
                continue
            return True
    return False


def _conditional_instruction(text, match):
    """A described hypothetical is not itself a conditional robot command."""
    if v1._quoted(text, *match.span()) or v1._question_only(text, *match.span()):
        return False
    if re.match(r'[^.!?;]{0,30}?(?:라는|라고)\s*(?:말|문구|표현|문장|단어)', text[match.end():]):
        return False
    left = max((text.rfind(marker, 0, match.start()) for marker in ('.', '!', '?', ';', '。', '？', '！')),
               default=-1) + 1
    right = min((position for marker in ('.', '!', '?', ';', '。', '？', '！')
                 if (position := text.find(marker, match.end())) >= 0), default=len(text))
    found = []
    for pattern in (_REQUIRED_MEMORY, _REQUIRED_MOVE, _REQUIRED_SWAP, _REQUIRED_STACK,
                    _REQUIRED_UNSTACK, _REQUIRED_RETURN, _KO_HOME_SEND):
        for action in pattern.finditer(text, left, right):
            if match.start() <= action.start() and action.end() <= match.end():
                continue
            directed = ((_memory_instruction(text, action) or _bare_memory_imperative(text, action))
                        if pattern is _REQUIRED_MEMORY else
                        _explicit_match(text, action) if pattern is _KO_HOME_SEND else _explicit_action(text, action))
            if directed:
                found.append(action)
    if match.re is _KOREAN_CONDITIONAL:
        return bool(found)
    before = [action for action in found if action.end() <= match.start()]
    if before:
        return True
    after = sorted((action for action in found if action.start() >= match.end()), key=lambda item: item.start())
    separator = re.search(r',|\bthen\b', text[match.end():right], re.IGNORECASE | re.ASCII)
    if separator:
        consequence_start = match.end() + separator.end()
        return any(action.start() >= consequence_start for action in after)
    # Without a separator, do not confuse an antecedent verb for a command.
    # Two distinct actions reveal a consequence; a nominal state condition
    # (e.g. "if slot 1 is empty move...") can also precede one action.
    if len(after) >= 2:
        return True
    if len(after) == 1:
        antecedent = text[match.end():after[0].start()]
        return bool(re.search(r'\b(?:is|are|was|were|has|contains|empty|occupied|full)\b', antecedent,
                              re.IGNORECASE | re.ASCII))
    return False


_REPEAT_INSTRUCTION = re.compile(
    r'(?:반복|되풀이)(?=해|하(?:세요|라|여|자|고))|\b(?:repeat|repeating)\b',
    re.IGNORECASE | re.ASCII)
_REPEAT_RESTATEMENT = re.compile(
    r'^\s*[,，]?\s*(?:즉|다시\s*말해(?:서|면)?|달리\s*말하면|in\s+other\s+words|that\s+is)\s*[,，:]?',
    re.IGNORECASE | re.ASCII)
_REPEAT_CHAIN_BOUNDARY = re.compile(
    r'(?<=반복)하고|(?<=되풀이)하고|(?<=반복)한\s*(?:뒤|후|다음)|'
    r'(?<=되풀이)한\s*(?:뒤|후|다음)')


def _repeat_clause(text, start, end):
    """Keep separate Korean repeat-and-repeat requests in separate spans."""
    clause, offset = _task_clause(text, start, end)
    # A numeric "not three times but four times" is one correction, not two
    # action clauses. Keep the remainder of that clause as well, so any
    # conflicting count after the correction cannot be silently discarded.
    if _EN_REPEAT_CORRECTION_LEFT.search(clause):
        correction = _EN_REPEAT_CORRECTION_RIGHT.match(text, offset + len(clause))
        if correction:
            tail, tail_offset = _task_clause(text, correction.end() - 1, correction.end())
            clause = text[offset:tail_offset + len(tail)]
    boundaries = [(offset + match.start(), offset + match.end())
                  for match in _REPEAT_CHAIN_BOUNDARY.finditer(clause)
                  if not v1._quoted(text, offset + match.start(), offset + match.end())]
    left = max((right for _, right in boundaries if right <= start), default=offset)
    right = min((left for left, _ in boundaries if left >= end), default=offset + len(clause))
    return text[left:right], left


def _repeat_restatement_matches(text, prior, prior_offset, current, current_offset):
    """Treat a reformulation as one action only when its scope agrees."""
    prior_values, prior_more, prior_continuous = _repeat_clause_quantities(text, prior, prior_offset)
    current_values, current_more, current_continuous = _repeat_clause_quantities(text, current, current_offset)
    finite = len(prior_values) == len(current_values) == 1 and prior_values == current_values
    continuous = not prior_values and not current_values and prior_continuous and current_continuous
    if (not (finite or continuous) or prior_more != current_more
            or prior_continuous != current_continuous):
        return False

    def targets(clause, offset):
        clause = _mask_quoted(clause)
        _, sources, destinations, objects, _, _ = v1._task_supports(clause, whole=text, offset=offset)
        roles, _ = _repeat_reference_roles(clause)
        named = set()
        for match in re.finditer(
                r'(?:이전|앞선|지난|직전|현재|지금|이번|다른|첫째|둘째|첫\s*번째|두\s*번째|마지막)\s*'
                r'(?:작업|동작|이동|프로그램)|'
                r'\b(?:first|second|third|fourth|previous|current|last|other|another)\s+'
                r'(?:task|step|move|action|operation|program)\b', clause, re.IGNORECASE | re.ASCII):
            named.add(re.sub(r'\s+', ' ', match.group().lower()))
        return sources, destinations, objects, roles, named

    prior_targets = targets(prior, prior_offset)
    current_targets = targets(current, current_offset)
    # An explicit new target cannot be inferred from an unnamed previous one,
    # and a different explicitly named target is not a restatement of it.
    return all(not current_part or (prior_part and prior_part == current_part)
               for prior_part, current_part in zip(prior_targets, current_targets))


def _explicit_repeat_clauses(text):
    """Find distinct affirmative repeat requests without inventing a plan."""
    result = []
    seen = set()
    for match in _REPEAT_INSTRUCTION.finditer(text):
        prefix = text[max(0, match.start() - 30):match.start()]
        keep = re.search(r'\bkeep\s+$', prefix, re.IGNORECASE | re.ASCII)
        keep_repeating = (match.group().lower() == 'repeating' and keep is not None
                          and _explicit_span(text, match.start() - len(prefix) + keep.start(),
                                             match.start() - len(prefix) + keep.start() + 4, 'keep')
                          and not v1._negated_action(text, *match.span())
                          and not v1._quoted(text, *match.span()))
        if (re.search(r'\b(?:stop|end|cancel)\s+(?:the\s+)?$', prefix, re.IGNORECASE | re.ASCII)
                or not (_explicit_action(text, match) or keep_repeating)):
            continue
        clause, offset = _repeat_clause(text, *match.span())
        key = (offset, offset + len(clause))
        if key in seen:
            continue
        seen.add(key)
        # A restatement describes one request only when count and target do
        # not contradict the preceding request.
        if result and _REPEAT_RESTATEMENT.match(clause):
            prior_start, prior_end = result[-1]
            prior_clause, prior_offset = _repeat_clause(text, prior_start, prior_end)
            if not _repeat_restatement_matches(text, prior_clause, prior_offset, clause, offset):
                v1._fail('unresolved_reference', 'repeat restatement has a conflicting or unclear scope')
            continue
        result.append((match.start(), match.end()))
    return result


# A directive with a scene object but no verifiable target is not a command
# that another, unrelated step can silently cover. This is a bounded guard,
# not a claim that arbitrary natural-language instructions can be recognised.
_SCENE_DIRECTIVE_VERB = re.compile(
    r'(?:' + '|'.join(pattern.pattern for pattern in
                       (_COVERAGE_MOVE, _REQUIRED_STACK, _REQUIRED_UNSTACK, _REQUIRED_SWAP)) + r')|'
    r'회전|기울|들어|\b(?:rotate|twist|tilt|lift|pick\s+up)\b', re.IGNORECASE | re.ASCII)
_SCENE_DIRECTIVE_OBJECT = re.compile(
    r'원통|물체|자리|슬롯|\b(?:cylinders?|objects?|pieces?|slots?|towers?)\b',
    re.IGNORECASE | re.ASCII)
_REGISTER_INSTRUCTION = re.compile(r'등록(?=해|하(?:세요|라|고|여))|\bregister\b', re.IGNORECASE | re.ASCII)
_ARRANGE_INSTRUCTION = re.compile(r'배치(?=해|하(?:세요|라|고|여))|\barrange\b', re.IGNORECASE | re.ASCII)
_CONTINUE_INSTRUCTION = re.compile(
    r'이어(?=서|가|해)|계속(?=해|하(?:세요|라|고|여))|재개(?=해|하(?:세요|라|고|여))|'
    r'\b(?:continue|resume|finish|complete|carry\s+on)\b', re.IGNORECASE | re.ASCII)
_CONTINUE_TARGET = re.compile(
    r'작업|동작|프로그램|\b(?:tasks?|actions?|steps?|programs?)\b', re.IGNORECASE | re.ASCII)
_STATUS_INSTRUCTION = re.compile(
    r'(?P<ko>(?:현재\s*)?(?:상태|진행\s*상황)(?:을|를)?\s*(?:알려|보여|말해))|'
    r'(?P<en>\b(?:(?:what\s+is|what[\'’]s|report|show|tell)\s+'
    r'(?:(?:me|us)\s+)?(?:the\s+)?(?:(?:current|robot|task)\s+)?(?:status|state|progress)))\b',
    re.IGNORECASE | re.ASCII)
_RETURN_COVERAGE_VERB = re.compile(
    r'원위치(?=해|하|시켜)|복귀(?=해|하)|돌아가|복원(?=해|하)|돌려|돌리|되돌|보내|'
    r'\b(?:return|restore|go\s+back|bring\s+back|send|bring|take|put|move)\b',
    re.IGNORECASE | re.ASCII)
_UNRESOLVED_RETURN_TARGET = re.compile(
    r'배치|장면|상태|위치|자리|원위치|대기|팔|'
    r'\b(?:arrangement|layout|scene|state|positions?|slots?|tasks?|home|back|origin)\b',
    re.IGNORECASE | re.ASCII)
_UNPARSED_SCENE_DIRECTIVE = re.compile(
    r'(?:^|[;.!?。？！]|\bthen\b)\s*(?:please\s+)?(?P<verb>[a-z][a-z-]*)\s+'
    r'(?:(?:the|a|an|this|that|one|two|three|four|top|bottom|each|other)\s+)*'
    r'(?:cylinders?|objects?|pieces?|slots?|towers?|grippers?)\b', re.IGNORECASE | re.ASCII)
_NONVERB_SCENE_PREFIX = frozenset(('the', 'a', 'an', 'this', 'that', 'these', 'those', 'each', 'every',
                                  'one', 'two', 'three', 'four', 'all', 'no', 'not', 'my', 'your', 'our',
                                  'their', 'its', 'if', 'when', 'unless', 'where', 'how', 'what', 'which'))
_AND_CLAUSE = re.compile(r'\band\s+(?:also\s+)?(?P<word>[a-z][a-z-]*)\b', re.IGNORECASE | re.ASCII)
_COMMA_CLAUSE = re.compile(r',\s*(?:and\s+)?(?:also\s+)?(?:instead\s+)?(?P<word>[a-z][a-z-]*)\b',
                           re.IGNORECASE | re.ASCII)
_AND_NONVERB = _NONVERB_SCENE_PREFIX | frozenset((
    'then', 'slot', 'slots', 'position', 'positions', 'cylinder', 'cylinders', 'object', 'objects',
    'piece', 'pieces', 'both', 'some', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'arm', 'arms',
    'if', 'unless', 'when', 'where', 'while', 'with', 'without', 'before', 'after', 'because',
    'as', 'using', 'via', 'by', 'from', 'to', 'in', 'at', 'on', 'near', 'through', 'for', 'very',
    'under', 'including', 'excluding', 'namely', 'especially', 'such'))
_SCENE_TASK_TARGET = re.compile(
    r'장면|배치|상태|자리|원통|물체|\b(?:scene|layout|arrangement|slots?|cylinders?|objects?)\b',
    re.IGNORECASE | re.ASCII)


def _explicit_returns(text):
    """Bind a return verb to the same local target vocabulary as v1."""
    targets = {}
    for pattern in (_MEMORY_RETURN_EN, _MEMORY_RETURN_KO):
        for match in pattern.finditer(text):
            if _explicit_match(text, match):
                start, end = match.span('verb')
                targets.setdefault(start, set()).add(('learned_memory', end))
    for match in _RETURN_COVERAGE_VERB.finditer(text):
        if not _explicit_action(text, match):
            continue
        clause, offset = _task_clause(text, *match.span())
        if _ARM_HOME_FOREIGN_TARGET.search(_mask_quoted(clause)):
            continue
        for kind, reference_pattern in v1._RETURN_REFERENCE.items():
            if (kind != 'arm_home' and match.group().lower() in ('send', 'bring', 'take', 'put', 'move', '보내')):
                continue
            for reference in re.finditer(reference_pattern, clause, re.IGNORECASE | re.ASCII):
                start, end = offset + reference.start(), offset + reference.end()
                if not v1._quoted(text, start, end) and not v1._negated_span(text, start, end):
                    targets.setdefault(match.start(), set()).add((kind, match.end()))
                    break
        if (match.start() not in targets and _REQUIRED_RETURN.fullmatch(match.group())
                and _UNRESOLVED_RETURN_TARGET.search(_mask_quoted(clause))):
            # A separately stated return still needs a return proposal even
            # when its exact target requires context or clarification.
            targets.setdefault(match.start(), set()).add(('unresolved', match.end()))
    result = []
    for start, candidates in targets.items():
        kinds = {kind for kind, _ in candidates} - {'unresolved'}
        kind = next(iter(kinds)) if len(kinds) == 1 else ('unresolved' if not kinds else 'ambiguous')
        result.append((kind, start, max(end for _, end in candidates)))
    return sorted(result, key=lambda item: (item[1], item[2]))


def _explicit_repeat_stops(text):
    spans = set()
    for match in v1._EXPLICIT_REPEAT_STOP.finditer(text):
        name = next(name for name in ('ko', 'ko_first', 'en') if match.group(name) is not None)
        start, end = match.span(name)
        if (v1._has_global_repeat_stop(text, match.start(), match.end())
                and _explicit_span(text, start, end, match.group(name))):
            spans.add((start, end))
    for match in v1._REPEAT_NEGATION.finditer(text):
        if v1._has_global_repeat_stop(text, *match.span()):
            spans.add(match.span())
    return sorted(spans)


def _sentence_breaks_between(text, start, end):
    """Ignore quoted punctuation, decimal points and common abbreviations."""
    quotes = v1._quote_spans(text)
    for pos in range(start, end):
        if text[pos] not in '.;!?。？！' or any(left <= pos < right for left, right in quotes):
            continue
        if text[pos] == '.':
            if pos and pos + 1 < len(text) and text[pos - 1].isdigit() and text[pos + 1].isdigit():
                continue
            if (pos and pos + 2 < len(text) and text[pos - 1].isalpha() and text[pos + 1].isalpha()
                    and (text[pos + 2] == '.' or (text[pos - 1].isupper() and text[pos + 1].isupper()))):
                continue
            if re.search(r'\b(?:dr|mr|mrs|ms|prof|no|vs|e\.g|i\.e)$',
                         text[max(0, pos - 8):pos], re.IGNORECASE | re.ASCII):
                continue
        return True
    return False


def _composite_before(text, earlier, later):
    """Return a stated ordering, or None when the linkage is ambiguous."""
    if {earlier[0], later[0]} == {'stop_motion', 'stop_repeat'}:
        # An immediate motion stop preempts a repeat-boundary control. The
        # latter's "after the current transfer" is not the order of these ops.
        return None
    left, left_end = earlier[1:3]
    right, _ = later[1:3]
    between = text[left_end:right]
    if earlier[0] == 'stack' and re.match(r'(?:고\s|은\s*(?:뒤|후|다음)|놓고\s)', between):
        return True
    sentence_start = max((pos + 1 for pos in range(left) if text[pos] in '.;!?。？！'
                          and _sentence_breaks_between(text, pos, pos + 1)), default=0)
    prefix = text[sentence_start:left]
    before = re.search(r'\bbefore\b', prefix + between, re.IGNORECASE | re.ASCII)
    after = re.search(r'\bafter\b', prefix + between, re.IGNORECASE | re.ASCII)
    if before and after:
        v1._fail('unresolved_reference', 'composite temporal order is ambiguous')
    if before:
        return before.start() >= len(prefix)
    if after:
        return after.start() < len(prefix)
    if re.search(r'\b(?:then|followed\s+by)\b|그리고|그\s*다음|(?:기고|겨서|고\s*나서)', between,
                 re.IGNORECASE | re.ASCII) or _sentence_breaks_between(text, left_end, right):
        return True
    if any(match.start() < right <= match.end() and not _clause_boundary_is_modifier(text, match, [])
           for match in _COMMA_CLAUSE.finditer(text, left_end, later[2])):
        return True
    return None


def _unrepresented_sentence_is_directive(text, left, right):
    """Treat an unrecognised independent imperative as needing clarification."""
    original = text[left:right]
    sentence = _mask_quoted(original).strip(' \t\r\n.,;!?。？！:')
    if not sentence:
        return False
    # Quoting or explicitly describing a phrase does not itself authorize the
    # quoted action. Preserve these existing non-action forms.
    if v1._quote_spans(original) and re.fullmatch(
            r'(?:say|quote|mention|read|repeat\s+the\s+words?|'
            r'(?:라고\s*)?(?:말해주세요|말해줘|말하세요|말해|인용해))\s*',
            sentence, re.IGNORECASE | re.ASCII):
        return False
    # A label can contain or report words without imposing those words on the
    # robot. Restrict unquoted reports to an identity statement so a later
    # imperative in the same sentence is never discarded as label text.
    identity = (r'(?:cylinder|object|piece)\s*(?:number\s*|no\.?\s*|#\s*)?'
                r'(?:[1-4]|one|two|three|four)')
    label_head = r'(?:the|a)\s+label\s+(?:says|reads|states)'
    if (v1._quote_spans(original) and re.fullmatch(label_head, sentence, re.IGNORECASE | re.ASCII)
            or re.fullmatch(label_head + r'\s+(?:that\s+)?it\s+'
                            r'(?:must|should)\s+(?:not\s+)?be\s+' + identity,
                            sentence, re.IGNORECASE | re.ASCII)):
        return False
    # "Need not" removes a requirement; unlike "must not", it imposes no
    # restriction on the target and does not require a separate action.
    if re.fullmatch(r'it\s+need\s+not\s+(?:be|equal|match)\s+' + identity,
                    sentence, re.IGNORECASE | re.ASCII):
        return False
    if (re.match(r"^(?:do\s+not|don['’]?t|never)\b", sentence, re.IGNORECASE | re.ASCII)
            or re.match(r'^not\b', sentence, re.IGNORECASE | re.ASCII)
            or re.search(r'(?:하지|지)\s*(?:마|말|않)(?:아|아요|세요|고)?\s*$', sentence)):
        return False
    if re.match(r'^(?:i|we|he|she|they)\s+(?:will|would|may|might|can|could|shall|am|are|is)\b',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    if re.match(r'^(?:when|if)\s+(?:i|we|he|she|they)\b[^,]*,\s*'
                r'(?:the|a|an|i|we|he|she|it|they)\b.*\b(?:may|might|will|would|can|could|is|are)\b',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    if re.match(r'^(?:before|after)\s+(?:(?:i|we|he|she|they)\b|'
                r'(?:the|this|current|a)\s+(?:(?:current|this)\s+)?(?:transfer|task|move|action)\b)',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    if re.match(r'^[a-z]+ing\b.+\b(?:is|are|was|were|has|have|[a-z]+(?:s|ed))\b',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    if re.match(r'^remember\s+that\b.+\b(?:can|could|may|might|will|is|are|was|were)\b',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    action_patterns = (_COVERAGE_MOVE, _REQUIRED_STACK, _REQUIRED_UNSTACK, _REQUIRED_SWAP,
                       _REQUIRED_RETURN, _REQUIRED_MEMORY)
    actions = [match for pattern in action_patterns for match in pattern.finditer(text, left, right)]
    if actions and all(v1._negated_action(text, *match.span()) or v1._quoted(text, *match.span())
                       for match in actions):
        return False
    # A complete prohibition or a plain statement of risk can be background
    # for a separate imperative. A conditional consequence that is itself an
    # instruction remains unrepresented and must still be clarified.
    if (actions and all(_korean_dependent_action(text, *match.span()) for match in actions)
            and (re.search(r'(?:서는|선)\s*안\s*(?:돼|되|됨)\s*$', sentence)
                 or re.search(r'지(?:는|도)?\s*(?:마|말|않)(?:아|아요|세요)?\s*$', sentence)
                 or re.search(r'위험(?:해|하다|합니다|해요)\s*$', sentence))):
        return False
    stop_mentions = list(re.finditer(r'멈|정지|중단|그만', text[left:right]))
    if (not actions and stop_mentions
            and all(_korean_dependent_action(text, left + item.start(), left + item.end())
                    for item in stop_mentions)
            and re.search(r'(?:서는|선)\s*안\s*(?:돼|되|됨)\s*$', sentence)):
        return False
    if re.search(r'\b(?:must|should|shall|(?:is|are|has|have)\s+to)\b',
                 sentence, re.IGNORECASE | re.ASCII):
        return True
    if re.match(r'^(?:if|unless|when|provided|assuming)\b|^(?:만약|만일)', sentence,
                re.IGNORECASE | re.ASCII):
        return True
    # A numbered object's colour or identity is a description, not another
    # transfer. The full match cannot hide a following instruction.
    if re.fullmatch(identity + r'\s+(?:is|was)\s+(?:not\s+)?(?:' + identity +
                    r'|red|blue|green|yellow|white|black|orange|purple|brown|gray|grey|pink)',
                    sentence, re.IGNORECASE | re.ASCII):
        return False
    # Clearly declarative background is not a separate operation. Other
    # unrecognised sentence forms are ambiguous and are rejected conservatively.
    if re.match(r'^(?:(?:dr|mr|mrs|ms|prof)\.\s+[a-z][a-z\'-]*\s+'
                r'(?:is|was|has|does|[a-z]+(?:s|ed))\b|'
                r'(?:the|a|an|this|that|these|those|my|your|our|their)\s+'
                r'(?:[a-z][a-z\'-]*\s+){1,4}(?:is|are|was|were|has|have|does|did)\b|'
                r'(?:i|we|he|she|it|they)\s+(?:am|is|are|was|were|has|have|do|does|did)\b)',
                sentence, re.IGNORECASE | re.ASCII):
        return False
    if re.search(r'(?:입니다|습니다|였어|했어|이었다|있었다|있다|없다)\s*$', sentence):
        return False
    return bool(re.search(r'[A-Za-z가-힣]', sentence))


def _clause_boundary_is_modifier(text, match, events):
    word = match.group('word').lower()
    if (word in _AND_NONVERB or v1._quoted(text, *match.span()) or
            v1._negated_action(text, *match.span('word')) or
            (word in ('place', 'places') and re.match(r'\s*\d', text[match.end():]))):
        return True
    # A terminal comma and standalone polite 'please' introduce no second operation.
    if (match.re is _COMMA_CLAUSE and word == 'please'
            and re.fullmatch(r',[ \t]*please[ \t]*\.?[ \t]*', text[match.start():], re.IGNORECASE | re.ASCII)):
        return True
    if match.re is _COMMA_CLAUSE and _restatement_comma(text, match.start()):
        return True
    if match.re is _COMMA_CLAUSE:
        # Commas around a certified subject modifier ("I, too, want you to
        # ...") cannot separate robot operations. The complete subject and
        # directly addressed predicate must be recognized before excluding it.
        left = max((text.rfind(mark, 0, match.start()) for mark in '.!?;。？！'), default=-1) + 1
        sentence = text[left:]
        head = re.search(r'\bwant\b', sentence, re.I | re.ASCII)
        if (head and match.start() - left < head.start() and
                _EN_WISH_SUBJECT.fullmatch(sentence[:head.start()]) and _direct_english_wish(sentence, head)):
            return True
    if word in ('e', 'i') and re.match(r'\.(?:g|e)\.', text[match.end():], re.IGNORECASE | re.ASCII):
        return True
    if word.endswith('ly'):
        following = re.match(r'\s+(?P<word>[a-z][a-z-]*)\b', text[match.end():], re.IGNORECASE | re.ASCII)
        if following is None or following.group('word').lower() in _AND_NONVERB:
            return True
    if word == 'empty' and re.match(r'\s+(?:slot|position)\s*\d', text[match.end():],
                                    re.IGNORECASE | re.ASCII):
        clause, offset = _task_clause(text, *match.span())
        if any(kind == 'arrange' and offset <= start < offset + len(clause)
               for kind, start, _, _ in events):
            return True
    # A recognised complete action can itself contain "and put", e.g.
    # the supported unstack expression. Do not split that single action.
    return any(start < match.start() and match.end() <= end for _, start, end, _ in events)


def _validate_unrepresented_sentences(text, events, program):
    """Check independent sentences and explicit sequential clause boundaries."""
    boundaries = []
    for pos in range(len(text)):
        if text[pos] in '.;!?。？！' and _sentence_breaks_between(text, pos, pos + 1):
            boundaries.append((pos, pos + 1))
    boundaries.extend(match.span() for match in re.finditer(r'\b(?:then|followed\s+by)\b', text,
                                                            re.IGNORECASE | re.ASCII)
                      if not v1._quoted(text, *match.span()))
    for pattern in (_AND_CLAUSE, _COMMA_CLAUSE):
        for match in pattern.finditer(text):
            if not _clause_boundary_is_modifier(text, match, events):
                boundaries.append((match.start(), match.start('word')))
    spans, left = [], 0
    for start, end in sorted(boundaries):
        if start < left:
            continue
        if text[left:start].strip(' \t\r\n,'):
            spans.append((left, start))
        left = end
    if text[left:].strip(' \t\r\n,'):
        spans.append((left, len(text)))
    if len(spans) < 2:
        return
    for left, right in spans:
        if any(left <= start < right for _, start, _, _ in events):
            continue
        if (_EN_EXPLICIT_FOLLOWING_MOVE.fullmatch(text) and
                re.fullmatch(r'\s*make\s+that\s+move\s*', text[left:right], re.IGNORECASE | re.ASCII)):
            # This one exact construction restates its sole earlier transfer;
            # the normal composite matcher still has to bind that transfer.
            continue
        if (any(step['op'] == 'stop_repeat' and step['boundary'] == 'after_task'
                for step in program['steps']) and
                re.fullmatch(r'\s*(?:finish|complete)\s+(?:the\s+)?(?:current|this)\s+task\s*',
                             text[left:right], re.IGNORECASE | re.ASCII)):
            # The supported after-task boundary already expresses finishing
            # the in-flight task before repetition stops.
            continue
        previous_repeats = [(start, end) for kind, start, end, _ in events if kind == 'repeat' and end <= left]
        if previous_repeats and _REPEAT_RESTATEMENT.match(text[left:right]):
            for match in _REPEAT_INSTRUCTION.finditer(text, left, right):
                prior, prior_offset = _repeat_clause(text, *previous_repeats[-1])
                current, current_offset = _repeat_clause(text, *match.span())
                if _repeat_restatement_matches(text, prior, prior_offset, current, current_offset):
                    break
            else:
                match = None
            if match is not None:
                continue
        if _unrepresented_sentence_is_directive(text, left, right):
            v1._fail('unresolved_reference', 'a separate request needs clarification before work')


def _validate_composite_coverage(program, text, stop_spans, requested_repeats):
    """Bind each recognised instruction to a distinct validated program step."""
    only_stop = len(program['steps']) == 1 and program['steps'][0]['op'] == 'stop_motion'
    if (only_stop and stop_spans and _unrepresented_repeat_condition(
            text, stop_spans[0][1], len(text), [])):
        # Preserve the immediate-stop path. The existing coordinator reports
        # a conditional follow-up as deferred; it does not perform that work.
        return
    actions = _required_actions(text)
    events = [(kind, match.start(), match.end(), match) for kind, match in actions]
    for match in _REQUIRED_MEMORY.finditer(text):
        if _memory_instruction(text, match) or _bare_memory_imperative(text, match):
            events.append(('remember', match.start(), match.end(), None))
    for kind, pattern in (('register', _REGISTER_INSTRUCTION), ('arrange', _ARRANGE_INSTRUCTION)):
        for match in pattern.finditer(text):
            if not _explicit_action(text, match):
                continue
            clause, _ = _task_clause(text, *match.span())
            if (_SCENE_TASK_TARGET.search(_mask_quoted(clause)) or
                    re.fullmatch(r'\s*(?:please\s+)?' + re.escape(match.group()) + r'\s*',
                                 clause, re.IGNORECASE | re.ASCII)):
                events.append((kind, match.start(), match.end(), None))
    for match in _CONTINUE_INSTRUCTION.finditer(text):
        if not _explicit_action(text, match):
            continue
        clause, _ = _task_clause(text, *match.span())
        if _REPEAT_WORD.search(_mask_quoted(clause)):
            continue
        if (match.group().lower() in ('finish', 'complete') and
                any(step['op'] == 'stop_repeat' and step['boundary'] == 'after_task'
                    for step in program['steps']) and
                re.fullmatch(r'\s*(?:finish|complete)\s+(?:the\s+)?(?:current|this)\s+task\s*',
                             clause, re.IGNORECASE | re.ASCII)):
            continue
        if (_CONTINUE_TARGET.search(_mask_quoted(clause)) or
                re.fullmatch(r'\s*(?:please\s+)?' + re.escape(match.group()) + r'\s*',
                             clause, re.IGNORECASE | re.ASCII)):
            events.append(('continue', match.start(), match.end(), None))
    for match in _STATUS_INSTRUCTION.finditer(text):
        start, end = match.span()
        if not (v1._quoted(text, start, end) or v1._reported_action(text, start)
                or v1._negated_action(text, start, end)):
            target = re.search(r'상태|진행\s*상황|\b(?:status|state|progress)\b', match.group(),
                               re.IGNORECASE | re.ASCII)
            events.append(('status', start + target.start(), start + target.end(), None))
    events.extend(('return:' + kind, left, right, None) for kind, left, right in _explicit_returns(text)
                  if not (only_stop and kind == 'arm_home'))
    events.extend(('stop_motion', left, right, None) for left, right in stop_spans)
    events.extend(('repeat', left, right, None) for left, right in requested_repeats)
    events.extend(('stop_repeat', left, right, None) for left, right in _explicit_repeat_stops(text))
    for match in _SCENE_DIRECTIVE_VERB.finditer(text):
        if only_stop:
            # A direct emergency stop may be dispatched independently; it is
            # not evidence that an unidentified follow-up was performed.
            break
        if not _explicit_action(text, match):
            continue
        if any(boundary.start() <= match.start() and match.end() <= boundary.end()
               for pattern in (_TRANSFER_BOUNDARY, _TASK_BOUNDARY, _ROUND_BOUNDARY)
               for boundary in pattern.finditer(text)):
            continue  # A stated completion boundary is not another transfer.
        if any(action.start() <= match.start() and match.end() <= action.end() for _, action in actions):
            continue
        if (match.group().lower() == 'place' and re.search(
                r'\b(?:a|the|this|that|in|at|from|to)\s+$',
                text[max(0, match.start() - 25):match.start()], re.IGNORECASE | re.ASCII)):
            continue
        clause, _ = _task_clause(text, *match.span())
        if _SCENE_DIRECTIVE_OBJECT.search(_mask_quoted(clause)):
            events.append(('unresolved_scene_action', match.start(), match.end(), None))
    if not only_stop:
        for match in _UNPARSED_SCENE_DIRECTIVE.finditer(text):
            left, right = match.span('verb')
            verb = match.group('verb')
            if (verb.lower() in _NONVERB_SCENE_PREFIX or not _explicit_span(text, left, right, verb)
                    or any(start <= left < end for _, start, end, _ in events)):
                continue
            events.append(('unresolved_scene_action', left, right, None))
    events.sort(key=lambda item: (item[1], item[2]))
    # Consecutive repeated stop words can restate the same urgent stop.
    events = [event for index, event in enumerate(events) if not (
        index and event[0] in ('stop_motion', 'stop_repeat') and events[index - 1][0] == event[0])]
    if not only_stop:
        _validate_unrepresented_sentences(text, events, program)
    if len(events) < 2:
        return
    if any(event[0] == 'unresolved_scene_action' for event in events):
        v1._fail('unresolved_reference', 'a separate scene directive needs clarification')
    if len(events) > len(program['steps']):
        v1._fail('unresolved_reference', 'an explicit operation in the composite request was omitted')
    located = _require_multiple_actions(text, program, actions) if actions else []
    action_indices = {id(match): index for match, index in located}

    def candidates(event):
        kind, left, right, match = event
        if match is not None:
            return [action_indices[id(match)]]
        result = []
        for index, step in enumerate(program['steps']):
            target_kind = kind.removeprefix('return:') if kind.startswith('return:') else None
            compatible = ((kind in ('remember', 'register', 'arrange') and step['op'] == 'task'
                           and step['kind'] == kind) or
                          (target_kind is not None and target_kind != 'ambiguous' and step['op'] == 'return' and
                           (target_kind == 'unresolved' or step['target']['kind'] in (target_kind, 'unresolved'))) or
                          (kind in ('stop_motion', 'stop_repeat', 'repeat', 'continue', 'status')
                           and step['op'] == kind))
            if compatible and any(found.start() <= left and right <= found.end()
                                  for found in re.finditer(re.escape(step['evidence']), text)):
                result.append(index)
        return result

    choices = [candidates(event) for event in events]
    if any(not indices for indices in choices):
        v1._fail('unresolved_reference', 'a composite operation has no corresponding supported step')
    order = [_composite_before(text, first, second) for first, second in zip(events, events[1:])]

    def bind(position, used, previous):
        if position == len(events):
            return True
        for index in choices[position]:
            if index in used:
                continue
            if position and order[position - 1] is not None:
                if (previous < index) != order[position - 1]:
                    continue
            if bind(position + 1, used | {index}, index):
                return True
        return False

    if not bind(0, set(), None):
        v1._fail('unresolved_reference', 'composite operations are missing, ambiguous or in the wrong order')


def _english_future_before(text, position):
    """Conservatively identify an unquoted before-clause before a verb."""
    match = re.search(r'\bbefore\b(?P<tail>[^,.!?;]{0,80})$', text[:position],
                      re.IGNORECASE | re.ASCII)
    if not match or v1._quoted(text, match.start(), match.start() + len('before')):
        return False
    # These openers mean to do the following command first, including when
    # speech transcription omits the comma.
    return re.fullmatch(r'\s*(?:(?:doing\s+)?anything\s+else|(?:you\s+)?(?:continue|proceed))\s*', match.group('tail'),
                        re.IGNORECASE | re.ASCII) is None


def _reject_ambiguous_before_clause(text):
    """Do not infer an execution from a merely future before-clause."""
    intents = _required_intents(text)
    if intents is None:
        return
    actions, memory = intents
    if (_english_future_before(text, memory.start()) or
            re.match(r'(?:하기|기|하시기|하려\s*하기|해\s*(?:두기|놓기))\s*'
                     r'(?:[가-힣0-9]{1,8}\s*){0,3}(?:전(?:에)?|에\s*앞서)', text[memory.end():])):
        v1._fail('unresolved_reference', 'a future before-clause does not clearly request the later memory')
    for kind, action in actions:
        english = _english_future_before(text, action.start())
        korean = re.match(r'(?:기기|하기|기|겨\s*(?:놓기|두기))?\s*'
                          r'(?:[가-힣0-9]{1,8}\s*){0,3}(?:전(?:에)?|에\s*앞서)', text[action.end():])
        if not english and not korean:
            continue
        if len(actions) == 1 and kind == 'move':
            if english and _EN_EXPLICIT_FOLLOWING_MOVE.fullmatch(text):
                continue
            if (korean and action.start() < memory.start() and re.fullmatch(
                    r'하고\s*,?\s*(?:그\s*다음(?:에)?|그\s*후(?:에)?|이후에)\s*'
                    r'(?:옮겨|이동해)(?:\s*(?:줘|주세요))?\s*[.!。！]?\s*', text[memory.end():])):
                continue
        v1._fail('unresolved_reference', 'a future before-clause does not clearly request the later action')


def _validate_complete_request(program, text, stop_spans):
    """Reject recognised omitted operations or conditions the schema cannot express."""
    if all(step['op'] == 'clarify' for step in program['steps']):
        return
    if (any(step['op'] == 'clarify' for step in program['steps']) and
            all(step['op'] in ('stop_motion', 'stop_repeat', 'clarify', 'status') for step in program['steps'])):
        return
    _reject_ambiguous_before_clause(text)
    # A model cannot silently reduce an explicitly repeated operation to one
    # move.  This check recognises affirmative repeat instructions only; a
    # quoted or negated repeat is not added to the proposal.
    requested_repeats = _explicit_repeat_clauses(text)
    _validate_composite_coverage(program, text, stop_spans, requested_repeats)
    repeat_steps = [step for step in program['steps'] if step['op'] == 'repeat']
    if len(repeat_steps) < len(requested_repeats):
        v1._fail('unresolved_reference', 'an explicit repeat instruction was omitted')
    if len(requested_repeats) > 1:
        # Bind successive steps to distinct stated clauses. Evidence may span
        # the whole utterance, but one clause cannot justify two executions.
        next_request = 0
        for step in repeat_steps:
            spans = [match.span() for match in re.finditer(re.escape(step['evidence']), text)]
            matching = [i for i in range(next_request, len(requested_repeats))
                        if any(left <= requested_repeats[i][0] and requested_repeats[i][1] <= right
                               for left, right in spans)]
            if not matching:
                v1._fail('unresolved_reference', 'repeat steps are not bound in request order')
            next_request = matching[0] + 1
    for pattern in (_OTHER_MEMORY_TARGET_EN, _OTHER_MEMORY_TARGET_KO):
        for match in pattern.finditer(text):
            if _explicit_span(text, *match.span('verb'), match.group('verb')):
                v1._fail('unresolved_reference', 'the requested memory target is not the current arrangement')
    if any(_unsupported_memory_scope(text, match, program['steps']) and not re.match(
            r'\s+to\s+(?:move|transfer|swap|exchange|return|restore)\b', text[match.end():],
            re.IGNORECASE | re.ASCII) for match in _REQUIRED_MEMORY.finditer(text)):
        v1._fail('unresolved_reference', 'the requested memory target is not the whole current robot scene')
    can_change_layout = any((step['op'] == 'task' and step['kind'] != 'remember') or
                            step['op'] in ('return', 'repeat', 'continue') for step in program['steps'])
    if can_change_layout and _affirmed_preservation(text):
        v1._fail('unresolved_reference', 'explicit slot preservation is not represented by this program schema')
    for pattern in (_KOREAN_CONDITIONAL, _EN_CONDITIONAL):
        for match in pattern.finditer(text):
            if _conditional_instruction(text, match):
                v1._fail('unresolved_reference', 'conditional operation requires clarification')
    if stop_spans:
        followups = [(kind, action) for kind, action in _required_actions(text)
                     if any(end <= action.start() and re.search(
                         r'\bthen\b|[.!。！]|멈추고|멈춰서|중단하고|정지하고',
                         text[start:action.start()], re.IGNORECASE | re.ASCII)
                            for start, end in stop_spans)]
        if followups:
            located = _require_multiple_actions(text, program, followups)
            stops = [i for i, step in enumerate(program['steps']) if step['op'] == 'stop_motion']
            if not stops or any(min(stops) >= index for _, index in located):
                v1._fail('unresolved_reference', 'the follow-up action must come after the requested stop')
    intents = _required_intents(text)
    if intents is not None:
        actions, memory = intents
        if len(actions) == 1:
            kind, action = actions[0]
            _require_pair_order(text, program, (kind, action, memory))
        else:
            _require_multiple_actions(text, program, actions, memory)
    else:
        actions = _required_actions(text)
        actions.extend(('arm_home', match) for match in _KO_HOME_SEND.finditer(text)
                       if _explicit_match(text, match))
        actions.sort(key=lambda item: item[1].start())
        linked = any(re.search(r'\b(?:then|followed\s+by)\b|\band\s*$|[.!。！]',
                               text[first[1].end():second[1].start()], re.IGNORECASE | re.ASCII)
                     for first, second in zip(actions, actions[1:]))
        if linked:
            _require_multiple_actions(text, program, actions)
    for match in _KO_HOME_SEND.finditer(text):
        if not _explicit_match(text, match):
            continue
        home_steps = [i for i, step in enumerate(program['steps'])
                      if step['op'] == 'return' and step['target']['kind'] == 'arm_home'
                      and (start := text.find(step['evidence'])) <= match.start('arm')
                      and match.end() <= start + len(step['evidence'])]
        if len(home_steps) != 1:
            v1._fail('unresolved_reference', 'the explicit arm-home operation was omitted or ambiguous')
        if any(end <= match.start() for _, end in stop_spans):
            stops = [i for i, step in enumerate(program['steps']) if step['op'] == 'stop_motion']
            if not stops or min(stops) >= home_steps[0]:
                v1._fail('unresolved_reference', 'arm home must follow the requested stop')
    if stop_spans and any(_explicit_action(text, match) for match in _REQUIRED_RETURN.finditer(text)):
        if not any(step['op'] == 'return' for step in program['steps']):
            v1._fail('unresolved_reference', 'the return requested after the stop was omitted')


_EN_EXPLICIT_FOLLOWING_MOVE = re.compile(
    r'\s*before\s+you\s+(?P<transfer>move\s+one\s+cylinder\s+from\s+slot\s+'
    r'(?:[1-4]|one|two|three|four)\s+to\s+slot\s+(?:[1-4]|one|two|three|four))\s*,\s*'
    r'remember\s+the\s+arrangement\s*,\s*(?P<follow>then\s+make\s+that\s+move)\s*[.!]?\s*',
    re.IGNORECASE | re.ASCII)


def _bound_explicit_following_move(program, text):
    """Recognize one exact, explicit follow-up referring to its sole transfer.

    This deliberately does not resolve generic "that move" references. The
    whole utterance must contain just one described transfer, a memory, and
    the explicit instruction to make that transfer afterwards.
    """
    if [(step['op'], step.get('kind')) for step in program['steps']] != [
            ('task', 'remember'), ('task', 'move')]:
        return None
    match = _EN_EXPLICIT_FOLLOWING_MOVE.fullmatch(text)
    if match is None or program['steps'][1]['evidence'] != match.group('follow'):
        return None
    if program['steps'][0]['evidence'] not in text[:match.start('follow')]:
        return None
    proposal = copy.deepcopy(program)
    proposal['steps'][1]['evidence'] = match.group('transfer')
    return proposal


def validate_program(program, text, reference_context=None, *, on_valid_step=None, _independent_stop=False):
    validate_schema(program)
    context = v1.normalize_context(reference_context)
    if not isinstance(text, str) or not text.strip():
        v1._fail('invalid_input', 'transcript is empty')
    direct, unclear_boundary = _direct_speech_view(text)
    if direct == text or all(s['op'] in ('clarify', 'status') for s in program['steps']):
        return _validate_direct_program(program, text, context, on_valid_step, _independent_stop, {})
    if v1._contains_any(text, v1._INJECTED_INSTRUCTION):
        v1._fail('unresolved_reference', 'task includes an instruction override')
    # Keep the actual input and model output on the ticket. Only the private
    # validation view masks report content. A broad citation must support all
    # of its fields in the remaining direct speech. It cannot borrow a field
    # or an extra step from the report.
    shadow = copy.deepcopy(program)
    errors = {}
    for index, step in enumerate(shadow['steps']):
        evidence = step['evidence']
        starts = [m.start() for m in re.finditer(re.escape(evidence), text)]
        if not starts:
            errors[index] = ('ungrounded_evidence', 'evidence is not an exact transcript substring')
            continue
        # A sentence containing both a possible direct request and a report
        # must not disappear and silently permit a later, partial program.
        # The independent immediate stop still receives its usual checks.
        if unclear_boundary and step['op'] not in ('stop_motion', 'clarify', 'status'):
            errors[index] = ('unresolved_reference', 'direct and reported instructions need clarification')
            continue
        if evidence in direct and any(direct[pos:pos + len(evidence)] == evidence for pos in starts):
            continue
        views = {direct[pos:pos + len(evidence)].strip() for pos in starts}
        supported = [view for view in views if re.search(r'\w', view)]
        if len(supported) == 1:
            step['evidence'] = supported[0]
        elif step['op'] == 'clarify':
            # A clarification cannot cause a robot action. Retain its literal
            # occurrence in this private view, even when its words are masked.
            pos = starts[0]
            step['evidence'] = direct[pos:pos + len(evidence)]
        else:
            errors[index] = ('unsupported_action', 'reported speech does not authorize this action')
    callback = (None if on_valid_step is None else
                lambda index, _step: on_valid_step(index, program['steps'][index]))
    _validate_direct_program(shadow, direct, context, callback, _independent_stop, errors)
    return program


def _validate_direct_program(program, text, context, on_valid_step, _independent_stop, source_errors):
    bound_followup = None if source_errors else _bound_explicit_following_move(program, text)
    if bound_followup is not None:
        # Validate the unique antecedent against all existing field, scope,
        # quantity and ordering guards. The original evidence and returned
        # proposal remain unchanged; no operation is added.
        proposal = bound_followup
        callback = (None if on_valid_step is None else
                    lambda index, _step: on_valid_step(index, program['steps'][index]))
        validate_program(proposal, text, context, on_valid_step=callback,
                         _independent_stop=_independent_stop)
        return program
    same_step_evidence = Counter()
    assigned_object_stacks = set()
    covered_stops = []
    validated_memories = []
    stop_spans = _current_stop_spans(text)
    for index, step in enumerate(program['steps']):
        if index in source_errors:
            v1._fail(*source_errors[index])
        meaning = json.dumps({key: value for key, value in step.items() if key != 'evidence'},
                             ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        evidence_key = (meaning, step['evidence'])
        same_step_evidence[evidence_key] += 1
        if same_step_evidence[evidence_key] > 1 and same_step_evidence[evidence_key] > text.count(step['evidence']):
            v1._fail('duplicate_unproven_step', 'identical steps require distinct occurrences of their evidence')
        start = text.find(step['evidence'])
        if start < 0:
            v1._fail('ungrounded_evidence', 'evidence is not an exact transcript substring')
        end = start + len(step['evidence'])
        _reject_dependent_action(step, text, start, end)
        if step['op'] not in ('clarify', 'status') and v1._quoted(text, start, end):
            v1._fail('quoted_action', 'quoted words do not authorize an action')
        if step['op'] == 'task' and step['kind'] in ('move', 'stack', 'unstack', 'swap'):
            _validate_move_scope(step, text, start, end)
        slot_stack = None
        if step['op'] == 'task' and step['kind'] in ('move', 'stack', 'unstack'):
            _reject_multiple_korean_stack_targets(step, text, start)
            clause, offset = _transfer_clause(step, text, start, end)
            direct_clause = _mask_quoted(clause)
            korean_support = (v1._KO_SLOT_STACK_DESTINATION.search(direct_clause)
                              or v1._KO_PLACE_STACK_DESTINATION.search(direct_clause))
            if (step['kind'] == 'move' and korean_support
                    and (re.search(r'가져다(?:가)?\s*.*(?:올리|올려|쌓)', direct_clause)
                         or any(_explicit_span(text, offset + m.start(), offset + m.end(), m.group())
                                for m in _REQUIRED_STACK.finditer(direct_clause)))):
                v1._fail('unresolved_reference', 'move values need the same unquoted affirmative action clause')
            if step['kind'] == 'stack':
                slot_stack = _slot_stack_match(text, start, end)
                english_support = v1._EN_STACK_SLOT_DESTINATION.search(direct_clause)
                if slot_stack is not None:
                    slots = (_quantity_value(slot_stack['source']), _quantity_value(slot_stack['destination']))
                    if (slots != (step['source_slot'], step['destination_slot'])
                            or step['object_id'] is not None or 'destination_object_id' in step):
                        v1._fail('ungrounded_value', 'the explicit places name source and destination slots')
                elif korean_support or english_support:
                    named = v1._task_supports(direct_clause, whole=text, offset=offset)[3]
                    if named:
                        source_part = direct_clause[:english_support.start()] if english_support else direct_clause
                        source_ids = v1._task_supports(source_part, whole=text, offset=offset)[3]
                        if named != source_ids or source_ids != {step['object_id']}:
                            v1._fail('ungrounded_value', 'object_id is not supported by its evidence')
            for repeated in _KO_REPEATED_TRANSFER.finditer(direct_clause):
                if (_quantity_value(repeated.group(1)) > 1
                        and not re.search(r'(?:원통|물체|자리|위치)\s*$', direct_clause[:repeated.start()])
                        and not v1._negated_span(text, offset + repeated.start(), offset + repeated.end())):
                    v1._fail('unresolved_reference', 'a repeated transfer must not be reduced to one movement')
            if ('destination_object_id' not in step and any(
                    not v1._quoted(text, offset + match.start(), offset + match.end())
                    and not v1._negated_span(text, offset + match.start(), offset + match.end())
                    for pattern in _OBJECT_SUPPORTS for match in pattern.finditer(clause))):
                v1._fail('ungrounded_value', 'destination_slot is not supported by its evidence')
            slots, _, _, _, _, _ = v1._task_supports(_mask_quoted(clause), whole=text, offset=offset)
            bound = {step['source_slot'], step['destination_slot']} - {None}
            if 'destination_object_id' not in step and slots - bound:
                v1._fail('unresolved_reference', 'an explicit slot is not represented by this transfer')
            for match in _EN_MULTIPLE_DESTINATIONS.finditer(clause):
                left, right = offset + match.start(), offset + match.end()
                if not v1._quoted(text, left, right) and not v1._negated_span(text, left, right):
                    v1._fail('unresolved_reference', 'multiple destination slots require clarification')
        if (step['op'] == 'task' or (step['op'] in ('return', 'continue', 'repeat')
                                     and step['target']['kind'] != 'unresolved')):
            _validate_explicit_height_limit(text)
        explicit_layout = (_validate_explicit_task_quantities(step, text, start, end)
                           if step['op'] == 'task' else {})
        cross_memory = (step['op'] == 'return' and step['target']['kind'] == 'learned_memory' and
                        _memory_return_uses_prior_step(step, text, program['steps'], index, validated_memories))
        if step['op'] == 'return':
            _validate_return_scope(step, text, start, end, cross_memory=cross_memory)
        if step['op'] == 'stop_repeat' and step['boundary'] != 'unresolved':
            clause_left = max((text.rfind(mark, 0, start) for mark in '.!?;。？！'), default=-1) + 1
            clause_right = min((pos for mark in '.!?;。？！' if (pos := text.find(mark, end)) >= 0),
                               default=len(text))
            allowed = [(item.start(), item.end()) for pattern in (_TRANSFER_BOUNDARY, _TASK_BOUNDARY, _ROUND_BOUNDARY)
                       for item in _affirmed_boundaries(text, 0, len(text), pattern)]
            if _unrepresented_repeat_condition(text, clause_left, clause_right, allowed):
                v1._fail('unresolved_reference', 'conditional repeat stop requires clarification')
        if 'destination_object_id' in step:
            # v1 only accepts a destination slot. Validate the typed relation
            # directly without inventing a slot or altering the original text.
            _validate_object_stack(step, text, assigned_object_stacks)
        elif cross_memory:
            _validate_cross_step_memory_return(step, text, start)
        elif step['op'] == 'stop_repeat' and step['boundary'] == 'after_transfer':
            if not _transfer_stop_verbs(text, start, end):
                v1._fail('ungrounded_value', 'a current transfer boundary and an affirmative repeat stop are required')
        elif (step['op'] == 'stop_repeat' and step['boundary'] in ('after_task', 'after_round')
              and _transfer_stop_verbs(text, 0, len(text))):
            v1._fail('ungrounded_value', 'the explicitly earlier transfer boundary cannot be delayed')
        elif (step['op'] == 'stop_repeat' and step['boundary'] == 'after_round'
              and _repeat_stop_verbs(text, 0, len(text), _TASK_BOUNDARY)):
            v1._fail('ungrounded_value', 'the explicitly earlier task boundary cannot be delayed')
        elif (step['op'] == 'stop_repeat' and step['boundary'] == 'after_task'
              and _repeat_stop_verbs(text, start, end, _TASK_BOUNDARY)):
            pass  # The v2 guard checked the affirmative, explicit task boundary.
        elif (step['op'] == 'stop_repeat' and step['boundary'] == 'after_round'
              and _repeat_stop_verbs(text, start, end, _ROUND_BOUNDARY)):
            pass  # A current round is distinct from a task or a transfer.
        elif step['op'] == 'repeat':
            _validate_repeat(step, text, context, index, program['steps'])
        elif step['op'] == 'continue' and step['target']['kind'] == 'previous_task':
            _validate_previous_continue(step, text, context, start, end)
            _validate_continue_reference_context(step, text, context)
        else:
            # Reuse the unchanged v1 per-step semantic guards without adopting
            # its cross-step evidence ordering or its whole-program stop check.
            old_step = copy.deepcopy(step)
            if slot_stack is not None:
                # A broad citation can include another valid move. Its count or
                # IDs must not become constraints on this independently bound stack.
                clause, offset = _task_clause(text, *slot_stack.span('verb'))
                old_step['evidence'] = text[max(start, offset):min(end, offset + len(clause))]
            if step['op'] == 'task' and step['kind'] in ('move', 'stack', 'unstack'):
                clause, clause_offset = _transfer_clause(step, text, start, end)
                corrections = _transfer_corrections(text, clause, clause_offset)
                v1_counts = v1._task_supports(step['evidence'])[4]
                if len(corrections) == 1:
                    correction = corrections[0]
                    rejected = _quantity_value(correction.group('negative'))
                    affirmed = _quantity_value(correction.group('positive'))
                    if (start <= correction.start('negative') and correction.end('positive') <= end
                            and rejected != affirmed and step['count'] == affirmed
                            and v1_counts == {rejected}):
                        # v1 sees only the explicitly rejected cardinal here.
                        # Validate every other field and the unchanged original
                        # text through v1; v2 checked the real count above.
                        old_step['count'] = rejected
            if (step['op'] == 'task' and step['kind'] in ('move', 'stack', 'unstack')
                    and step['count'] is not None
                    and step['count'] in _affirmed_quantities(step['evidence'], offset=start, whole=text)
                    and step['count'] not in v1._task_supports(step['evidence'])[4]
                    and not v1._task_supports(step['evidence'])[4]):
                # v1 cannot parse "both cylinders" or "두 원통". Validate all
                # other fields with v1, retaining the original v2 proposal.
                old_step['count'] = None
            if (step['op'] == 'task' and step['kind'] in ('arrange', 'register')
                    and old_step['layout'] is not None):
                evidence_layout = _layout_quantities(step['evidence'], offset=start, whole=text)
                v1_counts = v1._task_supports(step['evidence'])[4]
                for entry in old_step['layout']:
                    count = entry['count']
                    v1_zero = (count == 0 and
                               v1._contains_any(step['evidence'], r'없|\b(?:empty|zero|none)\b'))
                    if (count is not None and entry['object_ids'] is None and count not in v1_counts
                            and not v1_zero and evidence_layout.get(entry['slot']) == {count}):
                        # Validate other v1 fields with a neutral placeholder;
                        # the explicit original count was checked above, and
                        # this private value is never returned or executed.
                        entry['count'], entry['object_ids'] = None, []

            def validate_old(value):
                old = {'schema': 'task_program.v1', 'request_id': program['request_id'], 'steps': [value]}
                seen = []
                try:
                    v1.validate_program(old, text, context, on_valid_step=lambda _index, item: seen.append(item))
                except v1.ProgramError as exc:
                    if exc.code != 'missing_explicit_stop' or len(seen) != 1:
                        raise

            try:
                validate_old(old_step)
            except v1.ProgramError as exc:
                completed_swap = _validate_completed_swap(step, text, start, exc)
                direct_scene = _validate_direct_scene_expression(step, text, start, exc)
                bound_move = _validate_move_with_bound_unstack_anaphora(step, text, start, exc)
                empty_unstack = len(program['steps']) == 1 and _explicit_empty_slot_unstack(step, text, start, exc)
                explicit_arm_home = _validate_explicit_arm_home_send(step, text, start, exc)
                entries = old_step.get('layout') if old_step.get('op') == 'task' else None
                # v1's multi-slot scanner assumes every count follows its slot.
                # A fully parsed value-before-slot request is checked entry by
                # entry, after v1 has checked duplicates and the combined total.
                if not (completed_swap or direct_scene or bound_move or empty_unstack or explicit_arm_home):
                    if (exc.code != 'ungrounded_value' or
                            str(exc) not in ('layout count is not bound to its stated slot',
                                             'layout object ID is not bound to its stated slot') or
                            not entries or len(entries) < 2 or
                            any(entry['slot'] not in explicit_layout for entry in entries)):
                        raise
                    for entry in entries:
                        one = copy.deepcopy(old_step)
                        one['layout'] = [entry]
                        validate_old(one)
            if step['op'] == 'continue' and step['target']['kind'] == 'active_task' and any(s['op'] == 'stop_motion' for s in program['steps'][:index]):
                v1._fail('unresolved_reference', 'current continuation cannot cancel an earlier stop')
            if step['op'] == 'continue':
                _validate_continue_reference_context(step, text, context)
            if step['op'] == 'stop_motion':
                matches = [(a, b) for a, b in stop_spans if start <= a and b <= end]
                if not matches:
                    v1._fail('unsupported_action', 'stop does not refer to a new affirmative stop')
                covered_stops.extend(matches)
        if step['op'] == 'task' and step['kind'] == 'register':
            validated_memories.clear()
        elif step['op'] == 'task' and step['kind'] == 'remember':
            validated_memories.append(index)
        if on_valid_step is not None:
            on_valid_step(index, step)
    # A diagnostic first-stop check validates only its own evidence. The full
    # program still has to cover every stop; the coordinator separately checks
    # the original utterance before it dispatches an independently valid stop.
    if not _independent_stop and any(span not in covered_stops for span in stop_spans):
        v1._fail('missing_explicit_stop', 'an affirmed motion stop was omitted')
    if not _independent_stop:
        _validate_complete_request(program, text, stop_spans)
    return program


def _validated_initial_stop(candidate, text, context, request_id):
    """Independently audit a proposed first stop, even if a later step is malformed.

    This is diagnostic only: it does not repair the schema or dispatch a stop.
    """
    if not isinstance(candidate, dict) or candidate.get('schema') != 'task_program.v2':
        return []
    steps = candidate.get('steps')
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict) or steps[0].get('op') != 'stop_motion':
        return []
    first = {'schema': 'task_program.v2', 'request_id': request_id, 'steps': [steps[0]]}
    try:
        validate_program(first, text, context, _independent_stop=True)
    except v1.ProgramError:
        return []
    return [{'step_index': 0, 'step': copy.deepcopy(steps[0])}]


class TaskProgramChatClient(v1.TaskProgramChatClient):
    def complete(self, messages, *, timeout_seconds):
        return self._complete(messages, timeout_seconds, proposal_schema())

    def complete_local_model(self, messages, *, timeout_seconds):
        return self._complete(messages, timeout_seconds, local_model_schema())

    def complete_local_audio_model(self, messages, *, timeout_seconds, cancel_event=None):
        return self._complete(messages, timeout_seconds, local_audio_model_schema(), cancel_event=cancel_event)

    def complete_pending_audio_model(self, messages, *, timeout_seconds, cancel_event=None):
        return self._complete(messages, timeout_seconds, pending_audio_model_schema(), cancel_event=cancel_event)

    def complete_pending_text_model(self, messages, *, timeout_seconds, cancel_event=None):
        return self._complete(messages, timeout_seconds, pending_audio_model_schema(voice_output=False),
                              cancel_event=cancel_event)

    def _complete(self, messages, timeout_seconds, schema, *, cancel_event=None):
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('timeout must be positive and finite')
        request = {'model': self.model, 'messages': messages, **GENERATION,
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'task_program_v2', 'strict': True, 'schema': schema}}}
        v1.common._raise_if_cancelled(cancel_event)
        body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
        v1.common._raise_if_cancelled(cancel_event)
        if cancel_event is None:
            raw = self.transport(self.url, body, timeout_seconds)
        else:
            raw = self.transport(self.url, body, timeout_seconds, cancel_event=cancel_event)
        v1.common._raise_if_cancelled(cancel_event)
        if not isinstance(raw, bytes) or len(raw) > v1.common.MAX_REPLY_BYTES:
            raise RuntimeError('invalid server response size or type')
        try:
            envelope = json.loads(raw, object_pairs_hook=v1.common._no_duplicate_keys, parse_constant=v1.common._bad_constant)
            choice = envelope['choices'][0]
            content = choice['message']['content']
            if not isinstance(content, str):
                raise TypeError('content must be a string')
            if choice.get('finish_reason') not in (None, 'stop'):
                raise v1.IncompleteResponse(choice.get('finish_reason'), content)
            return content
        except v1.IncompleteResponse:
            raise
        except (ValueError, KeyError, IndexError, TypeError, RecursionError) as exc:
            raise RuntimeError('malformed chat-completions response') from exc


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
    proposal_only: bool
    execution_authorized: bool
    required_resume_checks: list
    required_repeat_checks: list
    repeat_reference_bindings: list
    normalizations: list = field(default_factory=list)
    voice_output: str | None = None


class TaskProgramInterpreter:
    def __init__(self, client):
        self.client = client

    def interpret(self, text, *, request_id, reference_context=None, timeout_seconds=30.0):
        return self._interpret(text, request_id, reference_context, timeout_seconds, validate_program)

    def interpret_audio(self, audio, *, request_id, reference_context=None, active_transfer_at_onset=False,
                        language='auto', timeout_seconds=180, cancel_event=None):
        from .voice import NoSpeechDetected
        v1.common._raise_if_cancelled(cancel_event)
        messages = build_local_audio_messages(audio, reference_context, language=language,
                                              active_transfer_at_onset=active_transfer_at_onset)
        v1.common._raise_if_cancelled(cancel_event)
        if cancel_event is None:
            raw = self.client.complete_local_audio_model(messages, timeout_seconds=timeout_seconds)
        else:
            raw = self.client.complete_local_audio_model(messages, timeout_seconds=timeout_seconds,
                                                         cancel_event=cancel_event)
        v1.common._raise_if_cancelled(cancel_event)
        response = v1.common.parse_json(raw)
        if not isinstance(response, dict) or not isinstance(response.get('transcript'), str):
            raise ValueError('음성 모델 응답에 인식 문장이 없습니다.')
        text = response.pop('transcript').strip()
        if not text:
            raise NoSpeechDetected('녹음에서 알아들을 수 있는 음성을 찾지 못했습니다.')
        declared_output = response.get('voice_output')
        clear_stop = (declared_output in ('mute', 'unmute')
                      and _clear_motion_stop(text, reference_context))
        voice_output = _audio_voice_output(text, response, reference_context)
        preserved_stop = _preserve_direct_motion_stop(text, response)
        response.pop('voice_output', None)
        if voice_output is not None:
            # The model's robot program is never admitted for a voice-only turn.
            response = _voice_only_proposal(text)
        elif preserved_stop is not None:
            response = preserved_stop
        elif clear_stop and response.get('steps', [{}])[0].get('op') != 'stop_motion':
            response = {'schema': 'task_program.v2', 'steps': [
                {'op': 'stop_motion', 'scope': 'active_task', 'evidence': text[:300]}]}
        elif (clear_stop and len(response.get('steps', [])) > 1
              and response['steps'][0].get('op') == 'stop_motion'
              and all(step.get('op') == 'clarify' and step.get('reason') == 'question_only'
                      for step in response['steps'][1:])):
            # The output-control field already carries these non-executable
            # voice-only clarifications; do not delay the stop as a compound.
            response['steps'] = response['steps'][:1]
        result = self._interpret(text, request_id, reference_context, timeout_seconds, None,
                                 supplied_json=response)
        v1.common._raise_if_cancelled(cancel_event)
        if result.status != 'ok':
            raise ValueError(result.detail or result.status)
        composite_output = (declared_output if clear_stop and not _direct_robot_stop_for_audio(text) else None)
        return text, replace(result, voice_output=voice_output or composite_output)

    def interpret_pending_audio(self, audio, *, request_id, reference_context, pending, language='auto',
                                timeout_seconds=180, cancel_event=None):
        """Classify a follow-up and preserve the old plan for a single-step edit."""
        v1.common._raise_if_cancelled(cancel_event)
        messages = build_pending_audio_messages(audio, reference_context, pending, language=language)
        raw = self.client.complete_pending_audio_model(messages, timeout_seconds=timeout_seconds,
                                                        cancel_event=cancel_event)
        v1.common._raise_if_cancelled(cancel_event)
        return self._parse_pending_result(raw, request_id=request_id, reference_context=reference_context,
                                          pending=pending, timeout_seconds=timeout_seconds)

    def interpret_pending_text(self, text, *, request_id, reference_context, pending,
                               timeout_seconds=180, cancel_event=None):
        """Classify a typed follow-up without allowing the model to alter its text."""
        v1.common._raise_if_cancelled(cancel_event)
        messages = build_pending_text_messages(text, reference_context, pending)
        complete = getattr(self.client, 'complete_pending_text_model', None)
        raw = (complete(messages, timeout_seconds=timeout_seconds, cancel_event=cancel_event)
               if complete is not None else self.client.complete(messages, timeout_seconds=timeout_seconds))
        v1.common._raise_if_cancelled(cancel_event)
        return self._parse_pending_result(raw, request_id=request_id, reference_context=reference_context,
                                          pending=pending, supplied_text=text, timeout_seconds=timeout_seconds)

    def interpret_standalone_text(self, text, *, request_id, reference_context, timeout_seconds=180,
                                  cancel_event=None):
        """Interpret text after a prior spoken turn that contained no task."""
        v1.common._raise_if_cancelled(cancel_event)
        context = v1.normalize_context(reference_context)
        context.update(active_request_id=None, execution_state='idle', repeat_state='inactive',
                       previous_task=None, previous_program=None)
        messages = build_local_model_messages(text, context)
        messages[0]['content'] += (
            '\nThe preceding spoken turn was only a question or conversation, not a robot task. '
            'Interpret this text as a separate request. If it changes or redirects an unstated '
            'previous task, or omits the source of a transfer, return a clarification; '
            'do not invent the missing source or destination from the scene.')
        complete = getattr(self.client, 'complete_local_model', None) or self.client.complete
        raw = complete(messages, timeout_seconds=timeout_seconds)
        v1.common._raise_if_cancelled(cancel_event)
        result = self._interpret(text, request_id, context, timeout_seconds, None,
                                 supplied_json=v1.common.parse_json(raw))
        if result.status == 'ok' and any(step.get('op') == 'retarget' for step in result.program['steps']):
            clarification = {'schema': 'task_program.v2', 'steps': [
                {'op': 'clarify', 'reason': 'missing_target',
                 'question': '진행 중인 작업이 없어요. 어느 원통을 어디로 옮길지 알려 주세요.',
                 'evidence': text[:300]}]}
            return self._interpret(text, request_id, context, timeout_seconds, None, supplied_json=clarification)
        return result

    def _parse_pending_result(self, raw, *, request_id, reference_context, pending, timeout_seconds, supplied_text=None):
        response = v1.common.parse_json(raw)
        # Older injected clients did not return the audio-only field.
        if isinstance(response, dict) and supplied_text is None:
            response.setdefault('voice_output', 'none')
        v1._schema_check(response, pending_audio_model_schema(voice_output=supplied_text is None))
        disposition = response.pop('pending_disposition')
        model_text = response.pop('transcript').strip()
        text = model_text if supplied_text is None else supplied_text.strip()
        declared_output = response.get('voice_output') if supplied_text is None else None
        voice_output = _audio_voice_output(text, response, reference_context) if supplied_text is None else None
        preserved_stop = _preserve_direct_motion_stop(text, response) if supplied_text is None else None
        response.pop('voice_output', None)
        clear_stop = (supplied_text is None and declared_output in ('mute', 'unmute')
                      and _clear_motion_stop(text, reference_context))
        if clear_stop:
            disposition = 'stop'
        elif direct_voice_output_control(text) is not None and voice_output is not None:
            return text, None, 'voice_output_' + voice_output
        if (disposition == 'unchanged' and declared_output in ('mute', 'unmute')
                and not _voice_output_is_question_or_quote(text) and _mentions_pending_action(text)):
            return text, None, 'uncertain'
        if voice_output is not None and disposition == 'unchanged':
            if _mentions_pending_action(text):
                return text, None, 'uncertain'
            return text, None, 'voice_output_' + voice_output
        if preserved_stop is not None:
            response, disposition = preserved_stop, 'stop'
        if disposition == 'stop' and response['steps'][0].get('op') != 'stop_motion':
            # A contradictory model proposal must never release a held task.
            # Promote only a stop prefix independently validated against the
            # original utterance; otherwise the existing error cancels it.
            from .demo_service import _validated_pending_stop_prefix
            if _validated_pending_stop_prefix(text, v1.normalize_context(reference_context)):
                response = {'schema': 'task_program.v2', 'steps': [
                    {'op': 'stop_motion', 'scope': 'active_task', 'evidence': text[:300]}]}
        if disposition == 'unchanged':
            steps = response['steps']
            if any(step.get('op') in ('stop_motion', 'stop_repeat', 'retarget') for step in steps):
                raise ValueError('후속 발화의 분류와 작업 내용이 서로 다릅니다.')
            if any(step.get('op') == 'clarify' and step.get('reason') not in ('question_only', 'negated_only')
                   for step in steps):
                raise ValueError('후속 발화에 확인이 필요한 내용이 있습니다.')
            if not all(step.get('op') in ('status', 'clarify') for step in steps):
                original = (pending.get('program') or {}).get('steps')
                without_evidence = lambda values: [{k: v for k, v in step.items() if k != 'evidence'}
                                                   for step in values]
                if not isinstance(original, list) or without_evidence(steps) != without_evidence(original):
                    raise ValueError('후속 발화의 분류와 작업 내용이 서로 다릅니다.')
            return text, None, disposition
        if not text:
            raise ValueError('후속 발화의 내용을 확인하지 못했습니다.')
        edit_record = None
        model_proposal = copy.deepcopy(response)
        if disposition == 'edit':
            if len(response['steps']) != 1 or response['steps'][0]['op'] != 'edit_step':
                raise ValueError('변경할 작업 단계를 확인하지 못했습니다.')
            response, edit_record = _apply_pending_step_edit(pending, response['steps'][0], text)
        result = self._interpret(text, request_id, reference_context, timeout_seconds, None,
                                 supplied_json=response)
        if result.status != 'ok':
            raise ValueError(result.detail or result.status)
        if edit_record is not None:
            # The assembled plan feeds the existing coordinator; keep the
            # actual model proposal and both versions for diagnostics.
            result = replace(result, raw_text=raw, raw_json=model_proposal,
                             normalizations=[*result.normalizations, edit_record])
            disposition = 'replace'
        if disposition == 'stop' and result.program['steps'][0]['op'] != 'stop_motion':
            raise ValueError('중단 요청을 확인하지 못했습니다.')
        if (disposition == 'stop' and declared_output in ('mute', 'unmute')
                and not _direct_robot_stop_for_audio(text)):
            disposition = 'stop_voice_output_' + declared_output
        if disposition == 'replace' and (result.unresolved or any(
                step['op'] in ('retarget', 'stop_motion', 'stop_repeat') for step in result.program['steps'])):
            disposition = 'uncertain'
        return text, result, disposition

    def _interpret(self, text, request_id, reference_context, timeout_seconds, check_program, *,
                   active_transfer_at_onset=False, supplied_json=None):
        start = time.perf_counter()
        def result(status, program=None, raw_text=None, raw_json=None, json_valid=None, schema_valid=None,
                   unresolved=None, validated_stop_prefix=None, detail=None, required_resume_checks=None, required_repeat_checks=None,
                   repeat_reference_bindings=None, normalizations=None):
            return TaskProgramResult(status, request_id if v1._valid_id(request_id) else None, program, raw_text, raw_json,
                                     json_valid, schema_valid, unresolved, validated_stop_prefix or [], detail,
                                     (time.perf_counter() - start) * 1000, True, False, required_resume_checks or [],
                                     required_repeat_checks or [], repeat_reference_bindings or [], normalizations or [])
        if not v1._valid_id(request_id) or not isinstance(text, str) or not text.strip() or len(text) > v1.MAX_TEXT_CHARS:
            return result('invalid_input', detail='invalid request ID or transcript')
        try:
            context = v1.normalize_context(reference_context)
        except v1.ProgramError as exc:
            return result(exc.code, detail=str(exc))
        try:
            if supplied_json is None:
                messages = (build_local_model_messages(text, context, active_transfer_at_onset=active_transfer_at_onset)
                            if check_program is None else build_messages(text, context))
                complete = (getattr(self.client, 'complete_local_model', self.client.complete)
                            if check_program is None else self.client.complete)
                raw = complete(messages, timeout_seconds=timeout_seconds)
            else:
                raw = json.dumps(supplied_json, ensure_ascii=False)
        except v1.IncompleteResponse as exc:
            try:
                parsed = v1.common.parse_json(exc.raw_text)
                json_valid = True
            except v1.common.IntentValidationError:
                parsed, json_valid = None, False
            prefix = _validated_initial_stop(parsed, text, context, request_id) if json_valid and check_program is not None else []
            return result('incomplete_response', raw_text=exc.raw_text, raw_json=parsed, json_valid=json_valid,
                          validated_stop_prefix=prefix, detail=str(exc))
        except (TimeoutError, socket.timeout):
            return result('timeout', detail='local inference timed out')
        except Exception as exc:
            return result('transport_error', detail=f'{type(exc).__name__}: {exc}')
        try:
            candidate = v1.common.parse_json(raw)
        except v1.common.IntentValidationError as exc:
            return result('invalid_json', raw_text=raw if isinstance(raw, str) else None, json_valid=False, schema_valid=False, detail=str(exc))
        try:
            v1._schema_check(candidate, local_model_schema() if check_program is None else proposal_schema())
            normalized, changes = (candidate, []) if check_program is None else normalize_stack_proposal(candidate, text)
            program = {'schema': normalized['schema'], 'request_id': request_id, 'steps': normalized['steps']}
            if check_program is None:
                v1._schema_check(program, local_model_schema(with_request_id=True))
            else:
                validate_schema(program)
        except v1.ProgramError as exc:
            prefix = _validated_initial_stop(candidate, text, context, request_id) if check_program is not None else []
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=False,
                          validated_stop_prefix=prefix, detail=str(exc))
        stops = []
        def keep_stop(index, step):
            if step['op'] == 'stop_motion':
                stops.append({'step_index': index, 'step': copy.deepcopy(step)})
        try:
            if check_program is not None:
                check_program(program, text, context, on_valid_step=keep_stop)
        except v1.ProgramError as exc:
            return result(exc.code, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True,
                          validated_stop_prefix=stops, detail=str(exc))
        unresolved = any(step['op'] == 'clarify' or (step['op'] in ('return', 'repeat', 'continue') and step['target']['kind'] == 'unresolved')
                         or (step['op'] == 'repeat' and step['count']['mode'] == 'unresolved')
                         or (step['op'] == 'stop_repeat' and step['boundary'] == 'unresolved') for step in program['steps'])
        resume = any(step['op'] == 'continue' and step['target']['kind'] == 'previous_task' for step in program['steps'])
        required = ['previous_task_identity_and_interruption_verified', 'stop_confirmed',
                    'current_state_and_layout_verified', 'replan_from_current_state'] if resume else []
        repeat_checks, repeat_bindings = [], []
        for index, step in enumerate(program['steps']):
            if step['op'] != 'repeat':
                continue
            if any(previous['op'] == 'stop_motion' for previous in program['steps'][:index]):
                for check in ('stop_confirmed', 'repeat_target_revalidated_after_stop'):
                    if check not in repeat_checks:
                        repeat_checks.append(check)
                if step['target']['kind'] == 'active_task':
                    repeat_bindings.append({'step_index': index, 'request_id': context['active_request_id'],
                                            'state_revision': context['state_revision']})
            if step['target']['kind'] == 'unresolved' and 'repeat_target_resolved' not in repeat_checks:
                repeat_checks.append('repeat_target_resolved')
            if step['count']['mode'] == 'additional' and 'existing_repeat_schedule_verified_or_clarified' not in repeat_checks:
                repeat_checks.append('existing_repeat_schedule_verified_or_clarified')
            for check in ('repeat_plan_checked', 'current_state_and_layout_verified_before_each_round'):
                if check not in repeat_checks:
                    repeat_checks.append(check)
        return result('ok', program=program, raw_text=raw, raw_json=candidate, json_valid=True, schema_valid=True,
                      unresolved=unresolved, validated_stop_prefix=stops, required_resume_checks=required,
                      required_repeat_checks=repeat_checks, repeat_reference_bindings=repeat_bindings,
                      normalizations=changes)
