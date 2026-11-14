"""Headless v2 admission and bounded execution through one SceneModelSession.

Admission validates the complete original program exactly once. Execution is a
separate explicit call. This module does not start models, simulators or a GUI.
"""
from __future__ import annotations

import copy
import hashlib
import re
import secrets
import threading
from contextlib import nullcontext

from . import scene_recall, task_program, task_program_v2
from .concurrency import CombinedEvent as _RetargetCancellation
from .scene_model_session import (DuplicateRequest, RepeatBoundaryReached, StackReferenceUnavailable, UnsupportedStep,
                                  _LocalModelProgram, _ValidatedTopUnstackSource, _reject_unsupported_target_constraint,
                                  _validated_single_top_unstack, plan_task)


class StaleReference(RuntimeError):
    """The captured context no longer names the current state."""


class Busy(RuntimeError):
    pass


class SessionNeedsRecovery(RuntimeError):
    """The registered scene or stopped state is not known after interruption."""


class PendingTextPlanChanged(RuntimeError):
    """An unstarted admission must be replaced before it can execute."""

    def __init__(self, result, *, context, stamp, revision, cleanup_bumps):
        super().__init__('the unstarted request has a replacement plan')
        self.result = result
        self.context = context
        self.stamp = stamp
        self.revision = revision
        self.cleanup_bumps = cleanup_bumps


class RepeatUnavailable(UnsupportedStep):
    """The next repetition cannot be performed from the known current scene."""


class StepUnavailable(UnsupportedStep):
    """A planned step cannot start from the scene and issued no transfer."""

    def __init__(self, message, *, not_started_reason=None):
        super().__init__(message)
        self.not_started_reason = not_started_reason


class NoMotionObserved(RuntimeError):
    """A transfer stopped before any observed motion or gripper action."""


def _finished_retarget(run, events):
    """Summarize only executor evidence, including an unfinished switch."""
    if not (run.get('retarget_requested') or run.get('retarget_dispatch_pending')
            or run.get('retarget_cutoff')):
        return None
    transfers = [event for event in events if event.get('event') == 'transfer']
    transfer = transfers[-1] if transfers else {}
    execution = transfer.get('result') or {}
    source_return = execution.get('source_return') if isinstance(execution, dict) else None
    redirect = execution.get('retarget') if isinstance(execution, dict) else None
    if (isinstance(redirect, dict)
            and redirect.get('source_slot') == run.get('retarget_source_slot')
            and redirect.get('destination_slot') == run.get('retarget_destination_slot')):
        index = redirect.get('activation_decision_index')
        applied = type(index) is int and index >= 0
        state = ('completed_expected' if transfer.get('committed') is True
                 and transfer.get('retarget_expected') is True else
                 'incomplete' if applied else 'not_applied')
        return {'status': state, 'applied': applied, 'source_slot': run.get('retarget_source_slot'),
                'destination_slot': run.get('retarget_destination_slot'),
                'activation_decision_index': index if applied else None, 'physical_success': None}
    if not isinstance(source_return, dict):
        source_return = None
    index = source_return.get('activation_decision_index') if source_return is not None else None
    applied = True if type(index) is int and index >= 0 else False if source_return is not None and index is None else None
    if transfer.get('committed') is True and transfer.get('source_return_expected') is True:
        state = 'completed_expected'
    elif applied is True:
        state = 'incomplete'
    elif applied is False:
        state = 'not_applied'
    else:
        state = 'unconfirmed'
    return {'status': state, 'applied': applied, 'source_slot': run.get('retarget_source_slot'),
            'destination_slot': run.get('retarget_destination_slot'),
            'activation_decision_index': index if applied is True else None, 'physical_success': None}


def _repeatable(step):
    return ((step['op'] == 'task' and step['kind'] in ('move', 'stack', 'unstack', 'swap', 'arrange'))
            or (step['op'] == 'return' and step['target']['kind'] in ('registered_initial', 'learned_memory')))


# A proposed first step does not establish that the *utterance* asked for the
# stop first.  In particular, the interpreter's diagnostic stop prefix can be
# valid in isolation for "move, then stop" or a conditional stop.  Recognise
# temporal and sequential clauses before permitting immediate dispatch.
_STOP_NOMINAL_CONDITION = (
    r'\b(?:in\s+case\s+of|in\s+the\s+event\s+of|upon|during)\b|'
    r'\bin\s+(?:an?\s+|the\s+)?emergenc(?:y|ies)\b|'
    r'(?<![가-힣])[가-힣]+(?:\s+[가-힣]+){0,3}\s+시(?:에(?:만|는|도)?|만|는|도)?(?![가-힣])|'
    r'(?:비상|긴급)(?:\s*(?:상황|사태|상태|사고))?\s*시(?:에(?:만|는|도)?|만|는|도)?|'
    r'[가-힣]+\s*(?:때|경우)(?:에(?:서)?)?(?:만|라도|는|도)?'
)
_STOP_CLOCK_DELAY = (
    r'\b(?:at|by)\s+(?:[01]?\d|2[0-3])(?::[0-5]\d)?(?:\s*[ap](?:\.?\s*m)?)?\b|'
    r'(?<![가-힣0-9])(?:(?:오전|오후|새벽|저녁|아침)\s*)?'
    r'(?:\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열(?:한|두)?)\s*시'
    r'(?:\s*\d{1,2}\s*분)?(?:에|까지|부터)?(?![가-힣])'
)
_KO_CONDITIONAL = r'(?<![가-힣])[가-힣]{2,}면(?![가-힣])'
_PRECEDING_STOP_BOUNDARY = re.compile(
    r'\b(?:after|afterwards|then|once|when|whenever|if|unless|later|following|tomorrow)\b|'
    r'\bnext\s+(?:day|week|month|year)\b|\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|'
    r'(?:나중에|이후|후(?:에)?|전에\s*끝|뒤(?:에)?|다음(?:에)?|끝나면|완료되면|하고\s*나서|'
    r'내일|모레|다음\s*(?:주|달|해)|[가-힣]+(?:다면|라면|으면|하면|지면|이면))|'
    + _KO_CONDITIONAL + '|' + _STOP_CLOCK_DELAY + '|' + _STOP_NOMINAL_CONDITION, re.IGNORECASE | re.ASCII)
_PRECEDING_STOP_ACTION = re.compile(
    r'(?:옮기|이동하|쌓|올리|내리|풀|교환하|바꾸|복원하|복귀하|돌아가|기억하|저장하|등록하|보내|놓|끝내|마치|완료하|완료되)'
    r'(?:고|한\s*뒤|한\s*후|기\s*전에)|'
    r'\b(?:move|transfer|stack|unstack|swap|exchange|remember|memorize|save|register|restore|return|'
    r'place|put|send|carry|finish|complete)\b', re.IGNORECASE | re.ASCII)
_STOP_FOLLOWUP_EN = re.compile(
    r'\b(?:and\s+then|then|and\s+(?:if|when|unless|once|wait|pause|ask|tell|say|show|report|'
    r'send|go|move|return|bring|take|put|park|restore|repeat|remember|memorize|save)|'
    r'but\s+(?:do\s+not|don[\'’]?t|never)\s+(?:ask|tell|say|show|report|check|'
    r'send|go|move|return|bring|take|put|park|restore|repeat|remember))\b',
    re.IGNORECASE | re.ASCII)
_STOP_POST_CONDITION = re.compile(
    r'\b(?:after|when|whenever|if|unless|once|until|later|following|tomorrow)\b|'
    r'\bnext\s+(?:day|week|month|year)\b|\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|'
    r'\bas\s+soon\s+as\b|\b(?:at|on|upon)\s+(?:the\s+)?(?:end|completion)\b|'
    r'\bat\s+(?:the\s+)?next\s+(?:move|transfer|task)\b|'
    r'\b(?:in|within)\s+(?:\d+|one|two|three|four|five|ten|a|an)\s+'
    r'(?:milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b|'
    r'(?:나중에|이후|후(?:에)?|뒤(?:에)?|다음(?:에)?|끝나면|끝난|끝날\s*때|완료\s*(?:시|때)|완료되면|하고\s*나서|'
    r'내일|모레|다음\s*(?:주|달|해)|[가-힣]+(?:다면|라면|으면|하면|지면|이면))|'
    + _KO_CONDITIONAL + '|' + _STOP_CLOCK_DELAY + '|' + _STOP_NOMINAL_CONDITION, re.IGNORECASE | re.ASCII)
_STOP_CALENDAR_DELAY = re.compile(
    r'내일|모레|다음\s*(?:주|달|해)|\btomorrow\b|\bnext\s+(?:day|week|month|year)\b|'
    r'\bon\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|' + _STOP_CLOCK_DELAY, re.I | re.ASCII)
_EN_ARM_HOME_REQUEST = re.compile(
    r'\b(?P<verb>return|send|bring|move|park|put|take)\b[^.!?;]{0,60}'
    r'\b(?P<arm>arm)\b[^.!?;]{0,30}\b(?P<home>home)\b',
    re.IGNORECASE | re.ASCII)
_KO_STATUS_FOLLOWUP = re.compile(
    r'(?:상태|여부|어디|(?:멈췄|멈추었|멈추는|멈춰\s*있|정지했|정지됐|중단했|중단됐)(?:는지|지))'
    r'[^.!?;。？！]{0,48}(?P<verb>알려|보여|확인|말해|봐)')
_EN_STATUS_FOLLOWUP = re.compile(
    r'\b(?P<verb>check|tell|report|show|verify)\b[^.!?;]{0,55}'
    r'\b(?:status|state|whether|if|where)\b', re.IGNORECASE | re.ASCII)
_KO_HYPOTHETICAL_STOP = re.compile(
    r'(?:멈추|멈춰주|멈춰|정지하|중단하|중지하|그만두)'
    r'(?:고|기|길|는\s*(?:것|지))?\s*,?\s*'
    r'(?:싶|원하|바라|생각|수\s*있|수\s*없)')
_STOP_NOMINAL_MATCH = re.compile(_STOP_NOMINAL_CONDITION, re.IGNORECASE | re.ASCII)
_POST_STOP_EXPLANATION = re.compile(
    r'(?:\b(?:because|since)\b[^,.!?;]{0,80}|'
    r'(?:^|,)\s*(?:i\s+am|(?:we|you|they)\s+are|(?:he|she|it|there|the\s+robot)\s+is)'
    r'(?:\s+[a-z]+){0,5})\s*$',
    re.IGNORECASE | re.ASCII)


def _immediate_initial_stop(text, step):
    """Confirm that an independently validated stop is not a later action.

    This uses only the captured utterance, not the model's step ordering.  A
    conditional or sequential stop stays pending for interpretation rather
    than being silently advanced ahead of the user's requested action.
    """
    evidence = step['evidence']
    start = text.find(evidence)
    end = start + len(evidence)
    spans = [(a, b) for a, b in task_program_v2._current_stop_spans(text)
             if start <= a and b <= end]
    return any(_immediate_stop_span(text, a, b) for a, b in spans)


def _immediate_stop_span(text, stop_start, stop_end):
    """Check each stop mention in the captured utterance in its own scope."""
    # "멈추고 싶으면 알려줘" describes a wish or condition; the
    # grammatical "-고" is not an imperative stop followed by a new action.
    if _KO_HYPOTHETICAL_STOP.match(text, stop_start):
        return False
    # A stop mentioned only as a future reference is not a request to stop
    # immediately; don't advance it ahead of the user's preceding action.
    if (task_program_v2._english_future_before(text, stop_start) or
            re.match(r'(?:추기|추시기|(?:추어|춰)\s*(?:두기|놓기)|지하기|기|하기)\s*'
                     r'(?:[가-힣0-9]{1,8}\s*){0,3}(?:전(?:에)?|에\s*앞서)',
                     text[stop_end:])):
        return False
    earlier = text[:stop_start]
    sentence_start = max((i + 1 for i, char in enumerate(earlier)
                          if char in '.!?;。？！' and not task_program._quoted(text, i, i + 1)),
                         default=0)
    # A new sentence explicitly asking for an immediate stop is independent
    # of earlier commands. In the same sentence, an earlier action remains a
    # sequencing constraint and cannot be silently skipped.
    separate_immediate = (sentence_start > 0 and (
        re.search(r'(?:지금|즉시|바로)\s*$', text[sentence_start:stop_start]) or
        re.match(r'\s+(?:(?:moving|(?:the\s+)?(?:robot|arm|motion))\s+)?(?:now|immediately)\b',
                 text[stop_end:], re.IGNORECASE | re.ASCII)))
    action_start = sentence_start if separate_immediate else 0
    if any(action_start <= match.start() < stop_start
           for _, match in task_program_v2._required_actions(text)):
        return False
    # A completed status notification followed by a comma and an explicit
    # "지금 즉시" begins a separate stop request. Do not carry the status
    # clause's "때" over as the stop's condition.
    separate = re.search(r',\s*지금\s*즉시\s*$', text[sentence_start:stop_start])
    if separate:
        comma = sentence_start + separate.start()
        for query_start, query_end in task_program._affirmed_motion_stops(text):
            if not sentence_start <= query_start < query_end <= comma:
                continue
            suffix = task_program_v2._KO_STOP_STATUS_SUFFIX.match(text, query_end)
            if (suffix is not None and suffix.end() <= comma and
                    re.fullmatch(r'\s*알려\s*(?:줘|주세요)\s*', text[suffix.end():comma])):
                sentence_start = comma + 1
                break
    # An explicit "stop now" directly after a complete repeat-boundary
    # correction is a separate immediate request. The correction helper keeps
    # conditional, reported, quoted and negated stops out of this exception.
    corrected_immediate = (re.match(
        r'\s+(?:(?:moving|(?:the\s+)?(?:robot|arm|motion))\s+)?(?:now|immediately)\b',
        text[stop_end:], re.IGNORECASE | re.ASCII) and any(
            task_program._corrected_repeat_stop_ends_negation(text, cue, stop_start)
            for cue in task_program._EN_NEGATION_CUE.finditer(text, 0, stop_start)))
    boundary_start = stop_start if corrected_immediate else sentence_start
    for match in _PRECEDING_STOP_BOUNDARY.finditer(text, boundary_start, stop_start):
        if task_program._quoted(text, *match.span()):
            continue
        if _STOP_CALENDAR_DELAY.fullmatch(match.group()):
            before, after_delay = text[boundary_start:match.start()], text[match.end():stop_start]
            has_now = (re.search(r'(?:지금|즉시|바로)\s*$', after_delay)
                       or re.match(r'\s+(?:(?:the\s+)?(?:robot|motion|arm)\s+)?(?:now|immediately)\b',
                                   text[stop_end:], re.I | re.ASCII))
            if has_now and (re.search(r'\bnot\s+$', before, re.I | re.ASCII)
                            or re.match(r'\s*(?:은|이|에)?\s*(?:말고|아니고)', after_delay)):
                continue
        return False
    for match in _PRECEDING_STOP_ACTION.finditer(text, action_start, stop_start):
        if (not task_program._quoted(text, *match.span())
                and not task_program._reported_action(text, match.start())
                and not task_program._negated_action(text, *match.span())):
            return False
    after = text[stop_end:]
    # Korean "멈추고/정지하고" immediately joins a *following* action;
    # conditions attached to that later action must not suppress the stop.
    korean_followup = re.match(r'(?:추고|추어서|춰서|춰주고|춰\s*주고|하고|해\s*주고)', after)
    if korean_followup:
        return True
    clause_end = min((position for mark in '.!?;。？！' if (position := after.find(mark)) >= 0),
                     default=len(after))
    followup = _STOP_FOLLOWUP_EN.search(after, 0, clause_end)
    if followup:
        clause_end = min(clause_end, followup.start())
    for match in _STOP_POST_CONDITION.finditer(after, 0, clause_end):
        if task_program._quoted(text, stop_end + match.start(), stop_end + match.end()):
            continue
        # "now, not later" explicitly rejects a delay; other conditions in
        # the same clause must still be checked independently.
        if ((match.group().lower() == 'later' or _STOP_CALENDAR_DELAY.fullmatch(match.group()))
                and re.search(r'\bnot\s+$', after[:match.start()], re.IGNORECASE | re.ASCII)
                and re.search(r'\b(?:now|immediately)\b', after[:match.start()], re.IGNORECASE | re.ASCII)):
            continue
        if (_STOP_CALENDAR_DELAY.fullmatch(match.group())
                and re.match(r'\s*(?:은|이|에)?\s*(?:말고|아니고)', after[match.end():])
                and re.search(r'(?:지금|즉시|바로)', text[sentence_start:stop_end] + after[:match.start()])):
            continue
        # A separate present-state or reason clause ("we are in an
        # emergency") does not make the preceding stop conditional.
        if (_STOP_NOMINAL_MATCH.fullmatch(match.group())
                and _POST_STOP_EXPLANATION.search(after[:match.start()])):
            continue
        return False
    return True


def _explicit_arm_home_request(text, *, start=0):
    if any(task_program_v2._explicit_match(text, match)
           for match in task_program_v2._KO_HOME_SEND.finditer(text, start)):
        return True
    return any(task_program_v2._explicit_span(text, *match.span('verb'), match.group('verb'))
               and not task_program._quoted(text, *match.span('arm'))
               and not task_program._quoted(text, *match.span('home'))
               and not task_program_v2._ARM_HOME_FOREIGN_TARGET.search(match.group())
               for match in _EN_ARM_HOME_REQUEST.finditer(text, start))


def _explicit_arm_home_followup(text, step):
    """Identify an omitted home request only after the admitted immediate stop."""
    start = text.find(step['evidence'])
    end = start + len(step['evidence'])
    return any(_explicit_arm_home_request(text, start=stop_end)
               for stop_start, stop_end in task_program_v2._current_stop_spans(text)
               if start <= stop_start and stop_end <= end
               and _immediate_stop_span(text, stop_start, stop_end))


def _explicit_status_followup(text, step):
    """Keep a separate status request visible when the proposal omits it."""
    start = text.find(step['evidence'])
    end = start + len(step['evidence'])
    stops = [(a, b) for a, b in task_program_v2._current_stop_spans(text)
             if start <= a and b <= end]
    if not stops:
        return False
    stop_end = stops[0][1]
    return any(task_program_v2._explicit_span(text, *match.span('verb'), match.group('verb'))
               for pattern in (_KO_STATUS_FOLLOWUP, _EN_STATUS_FOLLOWUP)
               for match in pattern.finditer(text, stop_end))


def _explicit_memory_followup(text, step):
    """Report an omitted memory instruction without performing the write."""
    start = text.find(step['evidence'])
    end = start + len(step['evidence'])
    stops = [(a, b) for a, b in task_program_v2._current_stop_spans(text)
             if start <= a and b <= end]
    if not stops:
        return False
    return any(task_program_v2._memory_instruction(text, match)
               or task_program_v2._bare_memory_imperative(text, match)
               for match in task_program_v2._REQUIRED_MEMORY.finditer(text, stops[0][1]))


def _requires_registered_top_scene(step, validation_source):
    return (isinstance(validation_source, _ValidatedTopUnstackSource)
            and validation_source.requires_registered_scene(step))


def _registered_top_scene(session, scene):
    """Use explicit registration, never treat an expected scene as confirmed."""
    if session.status != 'registered' or scene is None:
        return False
    view = getattr(session, 'scene_state', None)
    if not callable(view):
        # The older assessed session's registered current scene is its known
        # scene; oracle-free sessions expose the stricter state below.
        return session.current == scene
    state = view()
    if (not isinstance(state, dict) or state.get('current_registered_scene') is None
            or state.get('registration_source') != 'user_declared'
            or state.get('confirmation') != 'explicit_registration'):
        return False
    try:
        return (scene_recall.scene(state['current_registered_scene']) == scene_recall.scene(scene)
                and state.get('unresolved_transfer') is None)
    except (TypeError, ValueError):
        return False


def _unique_unspecified_target(step, scene, *, validation_source=None):
    if step['op'] != 'task' or step['kind'] not in ('move', 'stack', 'unstack'):
        return True
    _reject_unsupported_target_constraint(step, validation_source)
    if (step['count'] is not None or step['object_id'] is not None
            or step.get('destination_object_id') is not None):
        # The named-support planner checks the source together with the
        # support, including when a preceding recall first exposes the scene.
        return True
    source = step['source_slot']
    if source is None:
        return False
    if scene is None:
        # A learned-memory return supplies its scene only during execution.
        # Defer the source check for model proposals until that scene exists;
        # _step repeats this check against the actual expected scene.
        return isinstance(validation_source, _LocalModelProgram)
    source_count = len(scene[source - 1])
    return (source_count == 1 or
            (source_count == 2 and _validated_single_top_unstack(step, validation_source)))


def _preview_scene(step, scene, registration_scene=None, *, registered_initial=None, task_start=None,
                   train_groups, validation_source=None):
    """Predict only for preflight; execution must still validate each real result."""
    if step['op'] == 'task' and step['kind'] == 'register':
        return scene_recall.scene(registration_scene)
    if step['op'] == 'return':
        kind = step['target']['kind']
        target = registered_initial if kind == 'registered_initial' else task_start if kind == 'task_start' else None
        if target is None or scene is None:
            return None
        try:
            return scene_recall.scene(target, scene_recall.identities(scene))
        except ValueError:
            return None
    if step['op'] != 'task' or step['kind'] not in ('move', 'stack', 'unstack', 'swap', 'arrange'):
        return scene
    if scene is None:
        return None
    return plan_task(scene, step, train_groups, validation_source=validation_source)[0]


class SceneProgramCoordinator:
    """A single-owner coordinator with an independent stop/control path."""

    def __init__(self, session):
        self.session = session
        self._lock = threading.RLock()
        self._revision = 0
        self._generation = 0
        self._serial = 0
        self._pending = {}
        self._interpretations = {}
        self._voice_holds = {}
        self._voice_turn_started = {}
        self._pending_text_overrides = {}
        self._pending_model_results = {}
        self._voice_hold_changed = threading.Condition(self._lock)
        self._active = None
        self._previous_task = None
        self._previous_program = None
        self._last_execution = None
        self._registered_initial = None
        self._registration_revision = None
        self._memory_revision = None
        self._stop_unconfirmed = False
        self._arm_home_reservation = None
        self.events = []
        # Reuse an existing explicit registration event, never guess that the
        # current scene is the initial scene after prior movement.
        if session.status == 'registered':
            registrations = [(i, event) for i, event in enumerate(session.events)
                             if event.get('event') == 'registered' and event.get('memory_cleared') is not False]
            if registrations:
                registration_index, registration = registrations[-1]
                self._registered_initial = scene_recall.scene(registration['scene'])
                self._registration_revision = self._new_id('registration')
                if self._recoverable_memory_event(session.events[registration_index + 1:]):
                    self._memory_revision = self._new_id('memory')
        self._known_session_stamp = self._session_stamp()
        self._expected_memory_stamp = self._memory_stamp()

    def _memory_stamp(self):
        memory = self.session.memory
        return id(memory), getattr(memory, 'state_version', None)

    def _recoverable_memory_event(self, events):
        """Reuse a preexisting memory only if its live generation is proven.

        Each completed transfer contributes one non-write GRU observation.
        Legacy events without an identity/version cannot certify the live
        state, and are intentionally not promoted into a memory reference.
        """
        memory = self.session.memory
        if not self._memory_exists():
            return False
        identity, version = getattr(memory, 'state_identity', None), getattr(memory, 'state_version', None)
        writes = [(index, event) for index, event in enumerate(events) if event.get('event') == 'remember']
        if (not isinstance(identity, str) or not identity or not isinstance(version, int)
                or isinstance(version, bool) or not writes):
            return False
        index, event = writes[-1]
        write_version = event.get('memory_state_version')
        if (event.get('memory_state_identity') != identity or not isinstance(write_version, int)
                or isinstance(write_version, bool) or write_version < 0):
            return False
        committed = sum(item.get('event') == 'transfer' and item.get('committed') is True
                        for item in events[index + 1:])
        return version == write_version + committed

    def _memory_exists(self):
        memory = self.session.memory
        value = getattr(memory, 'has_memory', getattr(memory, 'written', None))
        # Legacy doubles without a versioned state still use the recorded
        # successful remember event. Production SceneRecall exposes both.
        return value is not False

    def _session_stamp(self):
        return (self.session.status, copy.deepcopy(self.session.current), len(self.session.events),
                self._memory_stamp())

    def _refresh_external_state(self):
        """Invalidate our references if the shared session changed outside this coordinator."""
        stamp = self._session_stamp()
        if self._active is None and stamp != self._known_session_stamp:
            self._previous_task = self._previous_program = None
            self._last_execution = None
            # A direct memory clear/observe invalidates references but does not
            # erase a separately registered starting layout.
            if stamp[:3] != self._known_session_stamp[:3]:
                self._registered_initial = self._registration_revision = None
            self._memory_revision = None
            self._known_session_stamp = stamp
            self._expected_memory_stamp = self._memory_stamp()
            self._bump()
        elif self._active is None and self._memory_stamp() != self._expected_memory_stamp:
            # A recovery subclass can explicitly register a new scene and
            # update its session stamp after clearing memory. Reconcile the
            # memory generation without restoring any old memory reference.
            self._memory_revision = None
            self._expected_memory_stamp = self._memory_stamp()
            self._bump()

    def _new_id(self, prefix):
        self._serial += 1
        return f'{prefix}-{self._serial}'

    def _bump(self):
        self._revision += 1

    @staticmethod
    def _reference(record):
        return None if record is None else {'status': 'unique', 'reference_id': record['id'],
                                             'revision': record['revision']}

    def _context(self):
        active = self._active
        if active is not None:
            state = 'stopping' if active['stop_requested'] else 'running'
        else:
            state = ('uncertain' if self.session.status == 'uncertain' or self._stop_unconfirmed
                     else 'unknown' if self.session._lock.locked() else 'idle')
        repeat = ('ending' if active and active['repeat_active'] and active['stop_repeat']
                  else 'active' if active and active['repeat_active'] else 'inactive')
        focus = ({'kind': 'learned_memory', 'source': 'explicit_recent', 'revision': self._memory_revision}
                 if state == 'idle' and self.session.status == 'registered'
                 and self._memory_revision is not None and self._memory_exists() else None)
        return {'execution_state': state, 'repeat_state': repeat,
                'active_request_id': active['request_id'] if active else None,
                'state_revision': f'state-{self._revision}',
                'previous_task': self._reference(self._previous_task),
                'previous_program': self._reference(self._previous_program), 'return_focus': focus}

    def snapshot(self):
        with self._lock:
            self._refresh_external_state()
            context = self._context()
            previous_task = copy.deepcopy(self._previous_task)
            previous_program = copy.deepcopy(self._previous_program)
            # Internal validation capabilities never enter the public snapshot.
            if previous_task is not None:
                previous_task.pop('validation_source', None)
            if previous_program is not None:
                previous_program.pop('validation_source', None)
            active = None if self._active is None else {key: copy.deepcopy(self._active[key]) for key in
                ('request_id', 'program_id', 'program_revision', 'round', 'step_index',
                 'task_id', 'task_revision', 'start_scene', 'task_start_scene',
                 'repeat_active', 'stop_repeat', 'stop_requested', 'atomic_inflight',
                 'transfer_id', 'transfer_sequence', 'transfer_phase', 'last_committed_transfer_id')}
            if active is not None and 'kind' in self._active:
                active['kind'] = self._active['kind']
            result = {'reference_context': context, 'active_program': active,
                      'previous_task': previous_task,
                      'previous_program': previous_program,
                      'last_execution': copy.deepcopy(self._last_execution),
                      'registered_initial': copy.deepcopy(self._registered_initial),
                      'registration_revision': self._registration_revision,
                      'memory_revision': self._memory_revision, 'session_status': self.session.status,
                      'stop_confirmation': 'unconfirmed' if self._stop_unconfirmed else 'not_requested'}
            scene_state = getattr(self.session, 'scene_state', None)
            if callable(scene_state):
                result['scene_state'] = scene_state()
            return result

    def _owner(self):
        if self._active is None:
            return None
        return {key: self._active[key] for key in
                ('program_id', 'program_revision', 'task_id', 'task_sequence', 'round',
                 'transfer_id', 'transfer_sequence', 'transfer_phase')}

    def _same_program(self, owner):
        return bool(owner and self._active and owner['program_id'] == self._active['program_id']
                    and owner['program_revision'] == self._active['program_revision'])

    @staticmethod
    def _invalidate_retarget_dispatches(run):
        if run is not None:
            for event in run.pop('retarget_cancellations', []):
                event.set()

    @staticmethod
    def _transfer_stop_boundary(run):
        return ('after_transfer' if run['stop_transfer_id'] is not None
                and run['stop_transfer_id'] == run['last_committed_transfer_id']
                else 'before_next_transfer')

    def _planning_snapshot(self):
        """Copy admission inputs under the lock; planning must not hold it."""
        return {'active': self._active is not None, 'session_busy': self.session._lock.locked(),
                'session_status': self.session.status, 'scene': copy.deepcopy(self.session.current),
                'cancelled': self.session.cancel.is_set(), 'stop_unconfirmed': self._stop_unconfirmed,
                'registered_initial': copy.deepcopy(self._registered_initial),
                'memory_revision': self._memory_revision, 'previous_task': copy.deepcopy(self._previous_task),
                'previous_program': copy.deepcopy(self._previous_program),
                'train_groups': frozenset(self.session.train_groups)}

    def _preview_scene(self, step, scene, registration_scene=None, *, registered_initial=None,
                       task_start=None, train_groups, validation_source=None):
        return _preview_scene(step, scene, registration_scene, registered_initial=registered_initial,
                              task_start=task_start, train_groups=train_groups, validation_source=validation_source)

    def _preflight(self, program, registration_scene, state, *, validation_source=None):
        steps = program['steps']
        if not state['active'] and state['session_busy']:
            raise Busy('the shared session is executing outside this coordinator')
        if len(steps) == 1 and steps[0]['op'] in ('stop_motion', 'stop_repeat'):
            if registration_scene is not None:
                raise ValueError('control cannot register a scene')
            if steps[0]['op'] == 'stop_repeat' and steps[0]['boundary'] == 'unresolved':
                raise UnsupportedStep('repeat-stop boundary needs clarification')
            return {'kind': 'control'}
        if state['active']:
            raise Busy('work is already active; only a separate control may be admitted')
        if state['session_status'] == 'uncertain' or state['stop_unconfirmed'] or state['cancelled']:
            raise SessionNeedsRecovery('session needs recovery or stop confirmation')
        registrations = [i for i, step in enumerate(steps)
                         if step['op'] == 'task' and step['kind'] == 'register']
        if len(registrations) > 1 or (registrations and registrations != [0]):
            raise UnsupportedStep('a registration must be the single first registration')
        if bool(registrations) != (registration_scene is not None):
            raise ValueError('registration needs exactly one explicit registration scene')
        if state['session_status'] != 'registered' and not registrations:
            raise UnsupportedStep('an explicit registration is required before work')
        if registrations:
            scene_recall.scene(registration_scene)
        repeat_index = None
        known_scene = state['scene'] if state['session_status'] == 'registered' else None
        known_memory = state['memory_revision'] is not None and not registrations
        prior_task_available = state['previous_task'] is not None and not registrations
        registered_initial = scene_recall.scene(registration_scene) if registrations else state['registered_initial']
        known_task_start = state['previous_task']['start_scene'] if prior_task_available else None
        for index, step in enumerate(steps):
            op = step['op']
            if op == 'task':
                if step['kind'] not in ('register', 'remember', 'move', 'stack', 'unstack', 'swap', 'arrange'):
                    raise UnsupportedStep(f"unsupported task: {step['kind']}")
                if step['kind'] == 'register' and step['layout'] is not None:
                    raise UnsupportedStep('registration uses the separately supplied scene')
                if (step['kind'] in ('move', 'stack', 'unstack') and step['count'] is not None
                        and step['count'] > 1 and step['object_id'] is not None):
                    raise UnsupportedStep('a single object ID cannot specify a multiple-object bundle')
                if (step['kind'] in ('move', 'stack', 'unstack') and step['count'] is not None
                        and (step['count'] > 2 or (step['height'] is not None and step['height'] < step['count']))):
                    raise UnsupportedStep('the quantity cannot fit the requested destination height')
                if step['kind'] == 'arrange' and any(
                        entry['count'] is None or entry['object_ids'] is not None for entry in step['layout']):
                    raise UnsupportedStep('only count-only arrangement constraints are implemented')
                if (_requires_registered_top_scene(step, validation_source)
                        and not state.get('top_scene_registered', False)):
                    raise UnsupportedStep('the current top-of-tower scene needs explicit registration')
                if not _unique_unspecified_target(step, known_scene, validation_source=validation_source):
                    raise UnsupportedStep('single-cylinder target is not unique in the known source')
                if step['kind'] == 'remember':
                    known_memory = True
            elif op == 'return':
                kind = step['target']['kind']
                if kind not in ('registered_initial', 'task_start', 'learned_memory'):
                    raise UnsupportedStep(f'unsupported return target: {kind}')
                if kind == 'registered_initial' and state['registered_initial'] is None and not registrations:
                    raise UnsupportedStep('registered initial snapshot is unavailable')
                if kind == 'learned_memory' and not known_memory:
                    raise UnsupportedStep('learned memory has not been written in the current registration')
                if kind == 'task_start' and not prior_task_available and not any(
                        prior['op'] in ('task', 'return') and not (prior['op'] == 'task' and prior['kind'] == 'register')
                        for prior in steps[:index]):
                    raise UnsupportedStep('a unique task-start snapshot is unavailable')
            elif op == 'repeat':
                if repeat_index is not None or index != len(steps) - 1:
                    raise UnsupportedStep('only one final, flat repeat is implemented')
                if step['count']['mode'] not in ('total', 'continuous'):
                    raise UnsupportedStep('additional or unresolved repeat plans require clarification')
                repeat_index = index
            else:
                raise UnsupportedStep(f'unsupported work step: {op}')
            before = known_scene
            known_scene = self._preview_scene(step, known_scene, registration_scene,
                                              registered_initial=registered_initial, task_start=known_task_start,
                                              train_groups=state['train_groups'], validation_source=validation_source)
            if op in ('task', 'return') and not (op == 'task' and step['kind'] == 'register'):
                known_task_start = before
        plan = {'kind': 'work', 'repeat_index': repeat_index, 'body': [], 'prior_body': False}
        if repeat_index is not None:
            repeat = steps[repeat_index]
            kind = repeat['target']['kind']
            if kind == 'prior_steps':
                indices = repeat['target']['step_indices']
                if indices != list(range(repeat_index)) or not indices:
                    raise UnsupportedStep('repeat body must be the complete preceding sequence')
                body = [steps[i] for i in indices]
                plan['prior_body'] = True
                body_validation_source = validation_source
            elif kind in ('previous_task', 'previous_program'):
                if repeat_index != 0:
                    raise UnsupportedStep('a previous reference must be the sole repeat step')
                record = state['previous_task'] if kind == 'previous_task' else state['previous_program']
                if record is None:
                    raise StaleReference('previous reference is unavailable')
                if kind == 'previous_program' and record['status'] != 'completed':
                    raise UnsupportedStep('an incomplete previous program cannot be replayed')
                body = [record['step']] if kind == 'previous_task' else record['steps']
                body_validation_source = record.get('validation_source')
            else:
                raise UnsupportedStep('active or unresolved repeat target is not implemented')
            if not body or not all(_repeatable(step) for step in body):
                raise UnsupportedStep('repeat target contains a non-repeatable step')
            if not plan['prior_body']:
                for step in body:
                    if not _unique_unspecified_target(step, known_scene, validation_source=body_validation_source):
                        raise UnsupportedStep('repeat target is not unique in the known source')
                    before = known_scene
                    known_scene = self._preview_scene(step, known_scene, registered_initial=registered_initial,
                                                      task_start=known_task_start, train_groups=state['train_groups'],
                                                      validation_source=body_validation_source)
                    known_task_start = before
            plan['body'] = copy.deepcopy(body)
            plan['body_validation_source'] = copy.deepcopy(body_validation_source)
            plan['count'] = copy.deepcopy(repeat['count'])
        return plan

    def _begin_captured_request(self, *, request_id, transcript, registration_scene=None):
        """Reserve a unique request ID and snapshot context/owner under the coordinator lock."""
        with self._lock:
            self._refresh_external_state()
            with self.session._request_lock:
                if request_id in self.session._requests:
                    raise DuplicateRequest('request ID was already used')
                self.session._requests.add(request_id)
            receipt = secrets.token_urlsafe(24)
            context = self._context()
            self._interpretations[receipt] = {
                'request_id': request_id, 'transcript': transcript, 'context': copy.deepcopy(context),
                'owner': self._owner(), 'registration_scene': copy.deepcopy(registration_scene)}
            captured = {'interpretation_id': receipt, 'request_id': request_id,
                        'reference_context': copy.deepcopy(context)}
            if transcript is not None:
                captured['text'] = transcript
            return captured

    def begin_interpretation(self, text, *, request_id, registration_scene=None):
        """Capture the input and owner before calling the interpreter.

        The returned context is a copy for the interpreter. Admission uses only
        the stored input, never a caller-supplied replacement context or text.
        A request ID is reserved as soon as interpretation begins.
        """
        if (not task_program._valid_id(request_id) or not isinstance(text, str)
                or not text.strip() or len(text) > task_program.MAX_TEXT_CHARS):
            raise ValueError('invalid request ID or transcript')
        return self._begin_captured_request(
            request_id=request_id, transcript=text, registration_scene=registration_scene)

    def begin_voice_processing(self, *, request_id):
        """Reserve a request and capture its owner before processing an audio file.

        This is the file-processing boundary, not the beginning of speech.
        The transcript is attached later without replacing the saved context or
        owner, and an unfinished capture cannot be admitted.
        """
        if not task_program._valid_id(request_id):
            raise ValueError('invalid request ID')
        return self._begin_captured_request(request_id=request_id, transcript=None)

    def recapture_pending_text_plan(self, request_id, text, change, *, cancel_event=None):
        """Re-admit the same still-unstarted request after its admission changed type."""
        with self._lock:
            if (cancel_event is not None and cancel_event.is_set()):
                raise InterruptedError('요청을 취소했습니다.')
            if (self._voice_turn_started.get(request_id) is not False or self._active is not None
                    or request_id not in self.session._requests):
                raise StaleReference('the pending request can no longer be replaced')
            self._refresh_external_state()
            context = self._context()
            expected = copy.deepcopy(change.context)
            expected['state_revision'] = context['state_revision']
            if (self._revision != change.revision + change.cleanup_bumps
                    or self._session_stamp() != change.stamp or context != expected):
                raise StaleReference('the scene changed while replacing the pending request')
            receipt = secrets.token_urlsafe(24)
            self._interpretations[receipt] = {
                'request_id': request_id, 'transcript': text, 'context': copy.deepcopy(context),
                'owner': None, 'registration_scene': None}
            return {'interpretation_id': receipt, 'request_id': request_id,
                    'text': text, 'reference_context': copy.deepcopy(context)}

    def hold_pending_voice(self, interpretation_id, target_request_id):
        """Bind a new utterance to an unstarted voice request at its onset."""
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if (captured is None or captured['owner'] is not None or self._active is not None
                    or self._voice_turn_started.get(target_request_id) is not False):
                return None
            hold = self._voice_holds.get(target_request_id)
            if hold is None:
                hold = {'current': target_request_id, 'tokens': set(), 'order': [], 'cancelled': False}
                reservation = self._arm_home_reservation
                if reservation is not None and reservation['request_id'] == target_request_id:
                    hold['home_cleanup'] = {'revision': self._revision, 'context': copy.deepcopy(self._context()),
                                            'session_stamp': self._session_stamp()}
                self._voice_holds[target_request_id] = hold
            if hold['current'] != target_request_id or hold['cancelled']:
                return None
            hold['tokens'].add(interpretation_id)
            hold.setdefault('order', []).append(interpretation_id)
            return hold

    def register_pending_voice(self, request_id):
        with self._lock:
            self._voice_turn_started[request_id] = False

    def cancel_pending_voice(self, request_id):
        with self._voice_hold_changed:
            hold = self._voice_holds.get(request_id)
            if hold is not None:
                hold['cancelled'] = True
                self._voice_hold_changed.notify_all()

    def resolve_pending_voice(self, hold, interpretation_id, *, replacement=None, cancel=False):
        with self._voice_hold_changed:
            if interpretation_id not in hold['tokens']:
                raise StaleReference('the pending voice follow-up is no longer reserved')
            if replacement is not None:
                if hold['cancelled']:
                    raise StaleReference('the pending voice request was cancelled')
                hold['current'] = replacement
                self._voice_holds[replacement] = hold
                self._voice_turn_started[replacement] = False
            if cancel:
                hold['cancelled'] = True
            hold['tokens'].remove(interpretation_id)
            if interpretation_id in hold.get('order', ()):
                hold['order'].remove(interpretation_id)
            self._voice_hold_changed.notify_all()

    def wait_pending_followup_turn(self, hold, interpretation_id, cancel_event):
        """Interpret follow-ups against plans updated by earlier inputs."""
        with self._voice_hold_changed:
            while True:
                if (cancel_event.is_set() or hold.get('cancelled')
                        or interpretation_id not in hold.get('tokens', ())):
                    raise InterruptedError('후속 요청을 취소했습니다.')
                order = hold.get('order', ())
                if not order or order[0] == interpretation_id:
                    return
                self._voice_hold_changed.wait(.1)

    def forget_pending_voice(self, request_id):
        with self._voice_hold_changed:
            self._voice_turn_started.pop(request_id, None)
            self._pending_text_overrides.pop(request_id, None)
            self._pending_model_results.pop(request_id, None)
            hold = self._voice_holds.get(request_id)
            if hold is not None and (hold['current'] != request_id or not hold['tokens'] or hold['cancelled']):
                self._voice_holds.pop(request_id, None)

    def replace_pending_text_plan(self, hold, interpretation_id, result):
        """Replace an unstarted text plan while its execution boundary is held."""
        with self._voice_hold_changed:
            target = hold.get('current')
            if (interpretation_id not in hold.get('tokens', ()) or hold.get('cancelled')
                    or self._voice_holds.get(target) is not hold
                    or self._voice_turn_started.get(target) is not False or self._active is not None
                    or not isinstance(result, task_program_v2.TaskProgramResult)
                    or result.status != 'ok' or result.unresolved or result.request_id != target):
                raise StaleReference('the pending text request is no longer available')
            self._pending_text_overrides[target] = copy.deepcopy(result)
            self.resolve_pending_voice(hold, interpretation_id)

    def wait_pending_voice(self, request_id, cancel_event):
        with self._lock:
            self._wait_pending_voice_locked(request_id, cancel_event)

    def pending_voice_held(self, request_id):
        with self._lock:
            hold = self._voice_holds.get(request_id)
            return bool(hold and hold['current'] == request_id and hold['tokens'])

    def rebase_voice_after_cancelled_home(self, capture, hold):
        """Accept only the revision caused by clearing this unstarted home reservation."""
        saved = hold.get('home_cleanup')
        if saved is None:
            return
        with self._lock:
            captured = self._interpretations.get(capture['interpretation_id'])
            if captured is None:
                raise DuplicateRequest('voice interpretation was discarded')
            if captured['context'] != saved['context']:
                return
            current = self._context()
            prior = copy.deepcopy(saved['context'])
            prior['state_revision'] = current['state_revision']
            if (self._revision != saved['revision'] + 1 or self._active is not None
                    or self._arm_home_reservation is not None or self._session_stamp() != saved['session_stamp']
                    or current != prior):
                raise StaleReference('the pending home or scene changed before the correction')
            captured['context'] = copy.deepcopy(current)
            capture['reference_context'] = copy.deepcopy(current)

    def _wait_pending_voice_locked(self, request_id, cancel_event=None):
        """Recheck under the same lock that changes a pending request to active."""
        while True:
            hold = self._voice_holds.get(request_id)
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError('요청을 취소했습니다.')
            if hold is None:
                return
            if hold['cancelled'] or hold['current'] != request_id:
                raise InterruptedError('후속 발화에 따라 이전 요청을 취소했습니다.')
            if not hold['tokens']:
                return
            self._voice_hold_changed.wait(.1)

    def request_transfer_retarget(self, interpretation_id, active_request_id, destination_slot, *, cancel_event=None,
                                  expected_transcript=None):
        """Bind a spoken redirect to the transfer captured when speech began.

        A successful return means only that the active executor accepted a
        request. It is never evidence that a physical target changed.
        """
        return self._request_transfer_retarget(interpretation_id, active_request_id, destination_slot,
                                               cancel_event=cancel_event, expected_transcript=expected_transcript)

    def _request_transfer_retarget(self, interpretation_id, active_request_id, destination_slot, *, cancel_event=None,
                                   expected_transcript=None, model_step=None):

        if (not isinstance(interpretation_id, str) or not isinstance(active_request_id, str)
                or (destination_slot is not None and (type(destination_slot) is not int or not 1 <= destination_slot <= 4))):
            raise ValueError('invalid retarget request')
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if captured is None or captured.get('transcript') is None:
                return {'status': 'retarget_rejected', 'reason': 'captured_utterance_unavailable'}
            if expected_transcript is not None and captured['transcript'] != expected_transcript:
                return {'status': 'retarget_rejected', 'reason': 'captured_transcript_changed'}
            if captured.get('retarget_dispatched'):
                return {'status': 'retarget_rejected', 'reason': 'already_dispatched'}
            if model_step is None:
                from .voice_retarget import parse_voice_retarget_details
                kind, proposed_destination, mentioned_previous = parse_voice_retarget_details(captured['transcript'])
            else:
                proposed_destination, mentioned_previous = model_step['destination_slot'], model_step['previous_slot']
                kind = 'origin' if proposed_destination is None else 'valid'
            parsed = kind, proposed_destination
            if parsed != ('valid', destination_slot) and not (parsed == ('origin', None) and destination_slot is None):
                return {'status': 'retarget_rejected', 'reason': 'destination_not_grounded'}
            owner, active = captured.get('owner'), self._active
            if (captured['context'].get('active_request_id') != active_request_id
                    or not owner or active is None or active.get('request_id') != active_request_id
                    or owner.get('transfer_phase') != 'in_progress' or owner.get('transfer_id') is None
                    or active.get('transfer_phase') != 'in_progress'):
                return {'status': 'retarget_rejected', 'reason': 'captured_transfer_not_active'}
            identity_keys = ('program_id', 'program_revision', 'task_id', 'task_sequence', 'round',
                             'transfer_id', 'transfer_sequence')
            if any(owner.get(key) != active.get(key) for key in identity_keys):
                return {'status': 'retarget_rejected', 'reason': 'captured_transfer_changed'}
            if (active.get('stop_requested') or active.get('stop_repeat') or self._stop_unconfirmed
                    or self.session.cancel.is_set() or (cancel_event is not None and cancel_event.is_set())):
                return {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
            if active.get('retarget_dispatch_pending'):
                return {'status': 'retarget_rejected', 'reason': 'retarget_request_pending'}
            scene_reader = getattr(self.session, 'scene_state', None)
            scene_state = scene_reader() if callable(scene_reader) else None
            transfer = scene_state.get('unresolved_transfer') if isinstance(scene_state, dict) else None
            slots = scene_state.get('active_slots') if isinstance(scene_state, dict) else None
            if not isinstance(transfer, dict) or not isinstance(slots, (list, tuple)):
                reason = 'return_to_source_unsupported' if parsed[0] == 'origin' else 'active_transfer_unavailable'
                return {'status': 'retarget_unsupported', 'reason': reason}
            source, previous_destination = transfer.get('source_slot'), transfer.get('destination_slot')
            if mentioned_previous is not None and mentioned_previous != previous_destination:
                return {'status': 'retarget_rejected', 'reason': 'original_destination_mismatch'}
            if source not in slots or previous_destination not in slots:
                return {'status': 'retarget_rejected', 'reason': 'active_transfer_outside_scope'}
            try:
                scene = scene_recall.scene(self.session.current)
                expected_object = transfer.get('expected_object_id')
                if (type(source) is not int or not 1 <= source <= 4 or not scene[source - 1]
                        or type(expected_object) is not int or scene[source - 1][-1] != expected_object):
                    return {'status': 'retarget_rejected', 'reason': 'active_transfer_mismatch'}
            except (ValueError, TypeError, IndexError, KeyError):
                return {'status': 'retarget_rejected', 'reason': 'active_transfer_mismatch'}
            controller = getattr(self.session, 'controller', None)
            config = getattr(controller, 'config', None)
            mock_return = (getattr(self.session, 'supports_mock_source_return', False) is True
                           and scene_state.get('execution_profile') == 'simulation_four_slot'
                           and getattr(config, 'backend', None) == 'mock'
                           and getattr(config, 'hardware_enabled', None) is False
                           and getattr(config, 'port', None) is None)
            # Newer executors declare the destinations they can actually
            # handle.  Older sessions retain the previous mock-only source
            # return behavior and still decide ordinary redirects themselves.
            capabilities = getattr(self.session, 'retarget_capabilities', None)
            if callable(capabilities):
                try:
                    declared = capabilities()
                except Exception:
                    declared = None
                destinations = declared.get('destinations') if isinstance(declared, dict) else None
                if (not isinstance(declared, dict) or declared.get('available') is not True
                        or type(declared.get('return_to_source')) is not bool
                        or not isinstance(destinations, (list, tuple))
                        or any(type(slot) is not int or slot not in slots for slot in destinations)):
                    return {'status': 'retarget_unsupported', 'reason': 'executor_not_available'}
                source_return = declared['return_to_source']
                supported_destinations = set(destinations)
            else:
                source_return = mock_return
                supported_destinations = set(slots)
            if parsed[0] == 'origin':
                if not source_return or type(source) is not int or source not in slots:
                    return {'status': 'retarget_unsupported', 'reason': 'return_to_source_unsupported'}
                destination_slot = source
            if destination_slot == previous_destination:
                return {'status': 'retarget_rejected', 'reason': 'destination_unchanged'}
            if active.get('retarget_requested') and destination_slot == active.get('retarget_destination_slot'):
                return {'status': 'retarget_rejected', 'reason': 'retarget_already_requested'}
            if (destination_slot not in slots or destination_slot not in supported_destinations
                    or (destination_slot == source and not source_return)):
                return {'status': 'retarget_unsupported', 'reason': 'destination_outside_supported_scope'}
            if destination_slot != source:
                try:
                    from . import scene_features, scene_policy, task_catalog
                    action = {'source_slot': source, 'destination_slot': destination_slot}
                    before, task = scene_features.transfer_task(self.session.current, action)
                    task_catalog.transfer(before, source, destination_slot)
                    if scene_policy.split_group_for(before, task) not in self.session.train_groups:
                        return {'status': 'retarget_unsupported', 'reason': 'outside_trained_scope'}
                except (ValueError, TypeError, IndexError, KeyError):
                    return {'status': 'retarget_rejected', 'reason': 'destination_infeasible'}
            handler = getattr(self.session, 'request_transfer_retarget', None)
            if not callable(handler):
                return {'status': 'retarget_unsupported', 'reason': 'executor_not_available'}
            if (active.get('stop_requested') or self.session.cancel.is_set()
                    or (cancel_event is not None and cancel_event.is_set())):
                return {'status': 'retarget_rejected', 'reason': 'stop_or_cancel_pending'}
            captured['retarget_dispatched'] = True
            transfer_cancel = threading.Event()
            active.setdefault('retarget_cancellations', []).append(transfer_cancel)
            active['retarget_dispatch_pending'] = True
            active['retarget_source_slot'] = source
            active['retarget_destination_slot'] = destination_slot
            cancellation = _RetargetCancellation(transfer_cancel, self.session.cancel, cancel_event)
        # The executor can wait for an in-flight command.  Never retain the
        # coordinator lock here: stop and transfer completion must proceed.
        try:
            result = handler(destination_slot, active_request_id=active_request_id,
                             program_id=owner['program_id'], task_id=owner['task_id'],
                             transfer_id=owner['transfer_id'], transfer_sequence=owner['transfer_sequence'],
                             cancel_event=cancellation)
        except Exception:
            result = None
        with self._lock:
            active['retarget_dispatch_pending'] = False
            if not isinstance(result, dict) or result.get('status') not in (
                    'retarget_requested', 'retarget_rejected', 'retarget_unsupported'):
                # An exception gives no evidence about whether the executor
                # accepted the request. Invalidate it and never expose the old
                # task as a repeatable reference.
                transfer_cancel.set()
                active['retarget_requested'] = True
                if self._active is active:
                    self._previous_task = self._previous_program = None
                    self._bump()
                return {'status': 'retarget_unknown', 'reason': 'executor_result_unavailable',
                        'remaining_program_stopped': self._active is active or bool(active.get('retarget_cutoff'))}
            if result['status'] == 'retarget_requested':
                active['retarget_requested'] = True
                # The original destination must not be offered later as the
                # task or program to repeat, even if the change is interrupted.
                if self._active is active:
                    self._previous_task = self._previous_program = None
                    self._bump()
                if self._active is not active or transfer_cancel.is_set():
                    return {'status': 'retarget_unknown', 'reason': 'transfer_changed_after_dispatch',
                            'remaining_program_stopped': bool(active.get('retarget_cutoff'))}
                result = {**result, 'remaining_program_stopped': True}
            else:
                transfer_cancel.set()
                if active.get('retarget_cutoff'):
                    result = {**result, 'remaining_program_stopped': True}
            return copy.deepcopy(result)

    def interpret_voice_audio(self, interpretation_id, interpreter, audio, *, language, cancel_event):
        """Bind the local model's single audio response to its original speech onset."""
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if (captured is None or captured['transcript'] is not None or captured.get('audio_pending')
                    or 'audio_result' in captured):
                raise DuplicateRequest('voice interpretation is no longer available')
            captured['audio_pending'] = True
            context = copy.deepcopy(captured['context'])
            owner = captured['owner'] or {}
            owned = bool(context['active_request_id'] and owner.get('transfer_id')
                         and owner.get('transfer_phase') == 'in_progress')
        try:
            text, result = interpreter.interpret_audio(
                audio, request_id=captured['request_id'], reference_context=context,
                active_transfer_at_onset=owned, language=language, cancel_event=cancel_event)
            with self._lock:
                if cancel_event.is_set():
                    raise InterruptedError('음성 해석이 취소되었습니다.')
                if self._interpretations.get(interpretation_id) is not captured:
                    raise DuplicateRequest('voice interpretation was discarded')
                captured['audio_text'], captured['audio_result'] = text, result
            return text, result
        finally:
            with self._lock:
                captured.pop('audio_pending', None)

    def interpret_pending_voice_audio(self, interpretation_id, interpreter, audio, pending, *, language, cancel_event,
                                      rebase=False):
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if (captured is None or captured['transcript'] is not None or captured.get('audio_pending')
                    or ('audio_result' in captured and not rebase)):
                raise DuplicateRequest('voice interpretation is no longer available')
            captured['audio_pending'] = True
            context = copy.deepcopy(captured['context'])
        try:
            text, result, disposition = interpreter.interpret_pending_audio(
                audio, request_id=captured['request_id'], reference_context=context,
                pending=pending, language=language, cancel_event=cancel_event)
            with self._lock:
                if cancel_event.is_set():
                    raise InterruptedError('음성 해석이 취소되었습니다.')
                if self._interpretations.get(interpretation_id) is not captured:
                    raise DuplicateRequest('voice interpretation was discarded')
                if result is not None:
                    captured['audio_text'], captured['audio_result'] = text, result
                elif rebase:
                    captured.pop('audio_text', None)
                    captured.pop('audio_result', None)
            return text, result, disposition
        finally:
            with self._lock:
                captured.pop('audio_pending', None)

    def complete_voice_transcription(self, interpretation_id, text):
        """Attach one transcript to the original capture, if its target survives."""
        if not isinstance(text, str) or not text.strip() or len(text) > task_program.MAX_TEXT_CHARS:
            raise ValueError('invalid transcript')
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if captured is None:
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            if captured['transcript'] is not None:
                raise DuplicateRequest('the interpretation already has a transcript')
            if captured.get('audio_pending') or ('audio_result' in captured and captured['audio_text'] != text):
                raise ValueError('voice transcript does not match its captured response')
            self._refresh_external_state()
            current = self._context()
            try:
                # Before interpretation the kind is unknown. Allow progress in
                # the same program; admit() later applies its stricter check to
                # ordinary work and the program check to controls.
                self._check_admission_context(True, captured['context'], captured['owner'])
            except StaleReference as exc:
                # Diagnostic only: this context is never used for admission.
                exc.current_context = copy.deepcopy(current)
                raise
            captured['transcript'] = text
            result = {'interpretation_id': interpretation_id, 'request_id': captured['request_id'],
                    'text': text, 'reference_context': copy.deepcopy(captured['context']),
                    'current_reference_context': copy.deepcopy(current)}
            if 'audio_result' in captured:
                result.update(direct_audio=True, **task_program_v2.audio_stop_flags(captured['audio_result']))
            return result

    def bind_pending_voice_target(self, interpretation_id, target_request_id):
        """Bind a captured pending request only if that same request became active."""
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if (captured is None or captured['transcript'] is not None or captured['owner'] is not None
                    or captured['context']['active_request_id'] is not None or self._active is None
                    or self._active['request_id'] != target_request_id):
                raise StaleReference('the captured pending target is no longer the active request')
            captured['context'] = copy.deepcopy(self._context())
            captured['owner'] = self._owner()

    def discard_interpretation(self, interpretation_id):
        """Release a capture whose interpreter timed out or produced no proposal."""
        with self._lock:
            return self._interpretations.pop(interpretation_id, None) is not None

    def _captured_input(self, interpretation_id):
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if captured is None:
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            if captured['transcript'] is None:
                raise ValueError('interpretation has no transcript')
            return copy.deepcopy(captured)

    def interpret_and_admit(self, interpretation_id, client, *, timeout_seconds, before_admission=None):
        """Read and independently ground a proposal for research and strict callers."""
        captured = self._captured_input(interpretation_id)
        validation_source = None

        def check_program(program, text, context, *, on_valid_step):
            nonlocal validation_source
            validation_source = _ValidatedTopUnstackSource(program, text, context, on_valid_step=on_valid_step)

        interpreter = task_program_v2.TaskProgramInterpreter(client)
        result = interpreter._interpret(captured['transcript'], captured['request_id'], captured['context'],
                                        timeout_seconds, check_program)
        if before_admission is not None:
            # UI observers can report and cancel. They cannot edit what will be admitted.
            before_admission(copy.deepcopy(result))
        if result.status != 'ok':
            # A failed or truncated response may still contain an independent immediate stop.
            admission = self.admit_interpretation(interpretation_id, result)
        else:
            admission = self._admit_candidate(interpretation_id, result, captured, result.raw_json, validation_source)
        return result, admission

    def interpret_local_model_and_admit(self, interpretation_id, client, *, timeout_seconds,
                                        before_admission=None, followup_target=None, cancel_event=None, stop_only=False):
        """Let the local model interpret speech; only its structured command goes to planning."""
        captured = self._captured_input(interpretation_id)
        owner = captured['owner'] or {}
        owned_transfer = bool(captured['context']['active_request_id'] and owner.get('transfer_id')
                              and owner.get('transfer_phase') == 'in_progress')
        result = captured.get('audio_result')
        if result is None:
            result = task_program_v2.TaskProgramInterpreter(client)._interpret(
                captured['transcript'], captured['request_id'], captured['context'], timeout_seconds, None,
                active_transfer_at_onset=owned_transfer)
        if before_admission is not None:
            before_admission(copy.deepcopy(result))
        return self.admit_local_model_result(interpretation_id, result, followup_target=followup_target,
                                             cancel_event=cancel_event, stop_only=stop_only)

    def admit_local_model_result(self, interpretation_id, result, *, followup_target=None,
                                 cancel_event=None, stop_only=False):
        """Admit an already interpreted local plan, honoring later held edits."""
        captured = self._captured_input(interpretation_id)
        # A typed correction can arrive while the model is still producing
        # the original proposal. Hold that request at its execution boundary
        # and use its corrected proposal without creating a second command.
        with self._lock:
            self._wait_pending_voice_locked(captured['request_id'], cancel_event)
            result = self._pending_text_overrides.pop(captured['request_id'], result)
        if result.status != 'ok':
            self.discard_interpretation(interpretation_id)
            return result, {'kind': 'rejected', 'status': result.status, 'program_executed': False}
        steps = result.program['steps']
        if stop_only and steps[0]['op'] != 'stop_motion':
            self.discard_interpretation(interpretation_id)
            return result, {'kind': 'rejected', 'status': 'stop_only', 'program_executed': False}
        if result.unresolved and steps[0]['op'] != 'stop_motion':
            if not self.discard_interpretation(interpretation_id):
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            questions = [{'reason': s['reason'], 'question': s['question']} for s in steps if s['op'] == 'clarify']
            return result, {'kind': 'rejected', 'status': 'ok', 'unresolved': True,
                            'clarifications': questions, 'program_executed': False}
        for step in steps:
            for field in ('source_slot', 'destination_slot', 'previous_slot'):
                slot = step.get(field)
                if slot is not None and not 1 <= slot <= 4:
                    self.discard_interpretation(interpretation_id)
                    return result, {'kind': 'rejected', 'status': 'slot_unavailable',
                                    'slot': slot, 'program_executed': False}
            if (step['op'] == 'task' and step['kind'] == 'swap'
                    and step['source_slot'] == step['destination_slot']):
                self.discard_interpretation(interpretation_id)
                return result, {'kind': 'rejected', 'status': 'infeasible_scene', 'program_executed': False}
        if followup_target is not None and captured['context']['active_request_id'] != followup_target:
            self.discard_interpretation(interpretation_id)
            return result, {'kind': 'rejected', 'status': 'active_followup_only', 'program_executed': False}
        if any(step['op'] == 'retarget' for step in steps):
            try:
                active = captured['context']['active_request_id']
                if len(steps) != 1 or active is None or (followup_target is not None and active != followup_target):
                    changed = {'status': 'retarget_rejected', 'reason': 'captured_transfer_not_active'}
                else:
                    changed = self._request_transfer_retarget(
                        interpretation_id, active, steps[0]['destination_slot'], cancel_event=cancel_event,
                        expected_transcript=captured['transcript'], model_step=steps[0])
                return result, {'kind': 'retarget', 'status': changed['status'], 'retarget': changed}
            finally:
                self.discard_interpretation(interpretation_id)
        if (followup_target is not None and not result.unresolved
                and steps[0]['op'] not in ('stop_motion', 'stop_repeat', 'status')):
            self.discard_interpretation(interpretation_id)
            return result, {'kind': 'rejected', 'status': 'active_followup_only', 'program_executed': False}
        source = _LocalModelProgram(result.program)
        return result, self._admit_candidate(interpretation_id, result, captured, result.program, source)

    @staticmethod
    def _validated_unresolved_clarifications(candidate, program, captured):
        """Validate structure and evidence, not the relevance of a question."""
        task_program._schema_check(candidate, task_program_v2.proposal_schema())
        validated = task_program_v2.validate_program(program, captured['transcript'], captured['context'])
        steps = validated['steps']
        unresolved = any(
            step['op'] == 'clarify'
            or (step['op'] in ('return', 'repeat', 'continue') and step['target']['kind'] == 'unresolved')
            or (step['op'] == 'repeat' and step['count']['mode'] == 'unresolved')
            or (step['op'] == 'stop_repeat' and step['boundary'] == 'unresolved')
            for step in steps)
        if not unresolved:
            return None
        return [{'reason': step['reason'], 'question': step['question']}
                for step in steps if step['op'] == 'clarify']

    def admit_interpretation(self, interpretation_id, interpretation):
        """Handle a v2 interpreter result, preserving a valid initial stop.

        The model's stop-prefix diagnostics are deliberately not an execution
        capability.  Revalidate its raw first step against the coordinator's
        captured transcript and context, then bind the immediate stop to the
        captured program.  Follow-up steps are *not* admitted or executed;
        in particular, stopping has not confirmed a safe arm-home movement.

        Ordinary proposals still follow the explicit ``admit``/``execute``
        ticket path.  An unresolved result may include its original reason and
        question as non-executing diagnostics.  Validation does not establish
        whether the question is relevant or correct.
        """
        if not isinstance(interpretation, task_program_v2.TaskProgramResult):
            raise TypeError('a v2 TaskProgramResult is required')
        captured = self._captured_input(interpretation_id)
        if interpretation.request_id != captured['request_id']:
            raise ValueError('interpreter request ID does not match the captured input')
        # Reparse the preserved model response; neither the derived JSON nor
        # the diagnostic prefix is an independent authorization capability.
        # Incomplete responses may still contain a complete, parseable first
        # stop, but the model text and the supplied parsed object must agree.
        try:
            parsed = (task_program.common.parse_json(interpretation.raw_text)
                      if isinstance(interpretation.raw_text, str) else None)
        except task_program.common.IntentValidationError:
            parsed = None
        if parsed != interpretation.raw_json:
            with self._lock:
                if self._interpretations.pop(interpretation_id, None) is None:
                    raise DuplicateRequest('interpretation ID is unknown or already consumed')
            return {'kind': 'rejected', 'status': 'inconsistent_interpretation', 'program_executed': False}
        candidate = copy.deepcopy(parsed)
        if isinstance(candidate, dict) and 'request_id' in candidate and candidate['request_id'] != captured['request_id']:
            raise ValueError('proposal request ID does not match the captured input')
        if interpretation.status == 'ok':
            try:
                task_program._schema_check(candidate, task_program_v2.proposal_schema())
                normalized, changes = task_program_v2.normalize_stack_proposal(candidate, captured['transcript'])
                expected_program = {**normalized, 'request_id': captured['request_id']}
            except task_program.ProgramError:
                expected_program, changes = None, []
            if (expected_program is None or interpretation.program != expected_program
                    or interpretation.normalizations != changes):
                with self._lock:
                    if self._interpretations.pop(interpretation_id, None) is None:
                        raise DuplicateRequest('interpretation ID is unknown or already consumed')
                return {'kind': 'rejected', 'status': 'inconsistent_interpretation', 'program_executed': False}
        return self._admit_candidate(interpretation_id, interpretation, captured, candidate)

    def _admit_candidate(self, interpretation_id, interpretation, captured, candidate, validation_source=None):
        steps = candidate.get('steps') if isinstance(candidate, dict) else None
        local_model = isinstance(validation_source, _LocalModelProgram)
        # Product voice and text use the model's structured result directly.
        # Do not reclassify the transcript with the research grammar here.
        if local_model and interpretation.status == 'ok' and not interpretation.unresolved:
            home = lambda step: (step.get('op') == 'return'
                                 and (step.get('target') or {}).get('kind') == 'arm_home')
            if ((len(steps) == 1 and home(steps[0])) or
                    (len(steps) == 2 and steps[0].get('op') == 'stop_motion' and home(steps[1]))):
                return self._admit_local_arm_home(interpretation_id, captured, len(steps) == 2)
        if local_model and all(step['op'] == 'status' for step in steps):
            with self._lock:
                if self._interpretations.pop(interpretation_id, None) is None:
                    raise DuplicateRequest('interpretation ID is unknown or already consumed')
                self._refresh_external_state()
            return {'kind': 'status', 'status': 'reported', 'subjects': [s['subject'] for s in steps],
                    'program_executed': False}
        has_first_stop = bool(isinstance(steps, list) and steps and isinstance(steps[0], dict)
                              and steps[0].get('op') == 'stop_motion')
        if not has_first_stop:
            if (interpretation.status == 'ok' and interpretation.program is not None
                    and not interpretation.unresolved):
                try:
                    ticket = self._admit(interpretation_id, interpretation.program, validation_source)
                except StackReferenceUnavailable as exc:
                    return {'kind': 'rejected', 'status': 'target_unavailable', 'detail': str(exc),
                            'program_executed': False}
                return {'kind': 'ticket', 'status': 'admitted', 'ticket': ticket}
            if (interpretation.status == 'ok' and interpretation.program is not None
                    and interpretation.unresolved):
                # An unresolved proposal is never a ticket.  Its diagnostic
                # fields are still untrusted: revalidate the complete raw
                # proposal and program against the captured input and state.
                with self._lock:
                    if self._interpretations.pop(interpretation_id, None) is None:
                        raise DuplicateRequest('interpretation ID is unknown or already consumed')
                    self._refresh_external_state()
                    self._check_admission_context(False, captured['context'], captured['owner'])
                    try:
                        if validation_source is None:
                            clarifications = self._validated_unresolved_clarifications(
                                candidate, interpretation.program, captured)
                        else:
                            clarifications = [{'reason': step['reason'], 'question': step['question']}
                                              for step in interpretation.program['steps'] if step['op'] == 'clarify']
                    except task_program.ProgramError as exc:
                        return {'kind': 'rejected', 'status': exc.code, 'program_executed': False}
                    self._refresh_external_state()
                    self._check_admission_context(False, captured['context'], captured['owner'])
                    if clarifications is None:
                        return {'kind': 'rejected', 'status': 'inconsistent_interpretation',
                                'program_executed': False}
                    return {'kind': 'rejected', 'status': interpretation.status,
                            'program_executed': False, 'unresolved': True,
                            'clarifications': clarifications}
            with self._lock:
                if self._interpretations.pop(interpretation_id, None) is None:
                    raise DuplicateRequest('interpretation ID is unknown or already consumed')
            return {'kind': 'rejected', 'status': interpretation.status, 'program_executed': False}

        # Independently validate the first raw step, never a caller-supplied
        # ``validated_stop_prefix``.  A malformed later step cannot erase a
        # clear immediate stop, but a later/conditional stop cannot be advanced.
        if local_model:
            validated = [{'step_index': 0, 'step': steps[0]}]
            valid_first = True
        else:
            validated = task_program_v2._validated_initial_stop(
                candidate, captured['transcript'], captured['context'], captured['request_id'])
            valid_first = bool(validated and validated[0]['step_index'] == 0
                               and _immediate_initial_stop(captured['transcript'], validated[0]['step']))
        with self._lock:
            current = self._interpretations.pop(interpretation_id, None)
            if current is None:
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            if not valid_first:
                return {'kind': 'rejected', 'status': 'stop_not_immediate_or_ungrounded',
                        'program_executed': False}
            self._refresh_external_state()
            owner, context = captured['owner'], captured['context']
            self._check_admission_context(True, context, owner)
            wants_clarification = (interpretation.status == 'ok' and interpretation.program is not None
                                   and interpretation.unresolved)
            follow_up_context_current = False
            follow_up_context_unavailable = False
            if wants_clarification:
                try:
                    follow_up_context_current = context == self._context()
                except Exception:
                    # This is an additional diagnostic check, never a reason
                    # to withhold an independently validated immediate stop.
                    follow_up_context_unavailable = True
            target_request_id = context['active_request_id']
            # A rejected/unfinished interpretation may have omitted a requested
            # follow-up entirely.  Never label the whole request complete just
            # because the raw proposal only supplied its first stop.
            explicit_home = not local_model and _explicit_arm_home_request(captured['transcript'])
            explicit_status = not local_model and _explicit_status_followup(captured['transcript'], validated[0]['step'])
            explicit_memory = not local_model and _explicit_memory_followup(captured['transcript'], validated[0]['step'])
            multiple_stops = not local_model and len(task_program_v2._current_stop_spans(captured['transcript'])) > 1
            follow_up = (len(steps) > 1 or interpretation.status != 'ok'
                         or bool(interpretation.unresolved) or explicit_home or explicit_status or explicit_memory
                         or multiple_stops)
            response = {'kind': 'stop_prefix', 'request_id': captured['request_id'],
                        'target_request_id': target_request_id, 'interpreter_status': interpretation.status,
                        'program_executed': False,
                        'follow_up_status': 'deferred' if follow_up else 'none'}
            if explicit_home or any(isinstance(step, dict) and step.get('op') == 'return'
                   and isinstance(step.get('target'), dict) and step['target'].get('kind') == 'arm_home'
                   for step in steps[1:]):
                response['arm_home_status'] = 'deferred_unsupported'
            if explicit_status or any(isinstance(step, dict) and step.get('op') == 'status' for step in steps[1:]):
                response['status_query_status'] = 'deferred_unsupported'
            if owner is None:
                # No coordinator-owned task was present at capture.  Never
                # infer an external session task as the intended stop target.
                response.update(status='no_active_task', stop_confirmation='not_requested')
            elif (target_request_id is None or self._active is None
                  or self._active['request_id'] != target_request_id):
                raise StaleReference('stop target is no longer active')
            elif self._active['stop_requested']:
                response.update(status='already_stopping', stop_confirmation='unconfirmed')
            else:
                control_id = 'stop-prefix-' + hashlib.sha256(interpretation_id.encode('utf-8')).hexdigest()[:32]
                stopped = self.request_motion_stop(control_id, target_request_id=target_request_id)
                response.update(status=stopped['status'], stop_confirmation=stopped['stop_confirmation'],
                                control_request_id=control_id)
                self.events.append({'event': 'interpreter_stop_prefix_dispatched',
                                    'request_id': captured['request_id'], 'control_request_id': control_id,
                                    'target_request_id': target_request_id,
                                    'follow_up_status': response['follow_up_status']})
            # Dispatch the independently valid stop first.  A follow-up is
            # merely diagnostic and must separately pass full validation on a
            # still-current captured context; it cannot delay the stop request.
            if follow_up_context_unavailable:
                response['clarification_status'] = 'validation_unavailable'
            elif wants_clarification and not follow_up_context_current:
                response['clarification_status'] = 'stale_context'
            elif wants_clarification and follow_up_context_current:
                try:
                    after_stop_context, after_stop_stamp = self._context(), self._session_stamp()
                    try:
                        if local_model:
                            clarifications = [{'reason': s['reason'], 'question': s['question']}
                                              for s in steps if s['op'] == 'clarify']
                        else:
                            clarifications = self._validated_unresolved_clarifications(
                                candidate, interpretation.program, captured)
                    except task_program.ProgramError:
                        clarifications = None
                    self._refresh_external_state()
                    if clarifications is None:
                        response['clarification_status'] = 'validation_failed'
                    elif (self._context() == after_stop_context
                          and self._session_stamp() == after_stop_stamp):
                        response.update(unresolved=True, clarifications=clarifications)
                    else:
                        response['clarification_status'] = 'stale_context'
                except Exception:
                    # The stop has already been dispatched.  An unexpected
                    # diagnostic failure must not hide its unconfirmed result.
                    response['clarification_status'] = 'validation_unavailable'
            return response

    def _admit_local_arm_home(self, interpretation_id, captured, stop_first):
        """Reserve one model-requested home operation, bound to its captured task."""
        with self._lock:
            # An already active task must receive its stop immediately. When
            # the preceding voice request has not started, however, the whole
            # stop-and-home proposal is still pending and may be corrected.
            if not stop_first or captured['owner'] is None:
                self._wait_pending_voice_locked(captured['request_id'])
            if self._interpretations.pop(interpretation_id, None) is None:
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            self._refresh_external_state()
            owner, context = captured['owner'], captured['context']
            self._check_admission_context(stop_first, context, owner)
            if self._arm_home_reservation is not None:
                if stop_first:
                    reservation = self._arm_home_reservation
                    reservation['cancelled'].set()
                    active = self._active
                    if active is not None and owner is not None and self._same_program(owner):
                        if active['stop_requested']:
                            stopped = {'status': 'already_stopping', 'stop_confirmation': 'unconfirmed'}
                        else:
                            stopped = self.request_motion_stop(
                                'stop-home-' + hashlib.sha256(interpretation_id.encode('utf-8')).hexdigest()[:32],
                                target_request_id=active['request_id'])
                    else:
                        stopped = {'status': 'cancelled_pending_request', 'stop_confirmation': 'not_requested'}
                    return {'kind': 'stop_prefix', **stopped, 'follow_up_status': 'deferred',
                            'arm_home_status': 'cancelled', 'program_executed': False}
                return {'kind': 'rejected', 'status': 'arm_home_pending', 'program_executed': False}
            target = context['active_request_id'] if stop_first else None
            active = self._active
            if stop_first and owner is not None:
                if (active is None or active['request_id'] != target or active['stop_requested']
                        or active.get('kind') in ('arm_home', 'recovery_registration')):
                    return {'kind': 'rejected', 'status': 'already_stopping', 'program_executed': False}
            elif active is not None or self.session._lock.locked():
                return {'kind': 'rejected', 'status': 'active_followup_only', 'program_executed': False}
            if ((not stop_first or active is None) and
                    (self.session.status != 'registered' or self._stop_unconfirmed or self.session.cancel.is_set())):
                raise SessionNeedsRecovery('session needs recovery or stop confirmation')
            # A virtual scene has no resident arm controller. Preserve the
            # independent stop even when its follow-up cannot be performed.
            supported = getattr(self.session, 'controller', None) is not None
            if stop_first and active is not None:
                stopped = self.request_motion_stop(
                    'stop-home-' + hashlib.sha256(interpretation_id.encode('utf-8')).hexdigest()[:32],
                    target_request_id=target)
            elif stop_first:
                stopped = self._stop_motion(captured['request_id'])
            else:
                stopped = None
            if not supported:
                return {'kind': 'stop_prefix' if stop_first else 'rejected',
                        'status': stopped['status'] if stopped else 'arm_home_unavailable',
                        'stop_confirmation': stopped['stop_confirmation'] if stopped else 'not_requested',
                        'follow_up_status': 'deferred' if stop_first else 'none',
                        'arm_home_status': 'unavailable', 'program_executed': False}
            reservation = {'id': self._new_id('arm-home'), 'request_id': captured['request_id'],
                           'target_request_id': target,
                           'target_finished': active.get('_completion_event') if active is not None else None,
                           'claimed': False, 'cancelled': threading.Event()}
            if active is not None and reservation['target_finished'] is None:
                return {'kind': 'stop_prefix', 'status': stopped['status'],
                        'stop_confirmation': stopped['stop_confirmation'],
                        'follow_up_status': 'deferred', 'arm_home_status': 'completion_unavailable',
                        'program_executed': False}
            self._arm_home_reservation = reservation
            self._bump()
            return {'kind': 'arm_home', 'status': 'reserved', 'reservation': reservation['id'],
                    'request_id': captured['request_id'], 'target_request_id': target,
                    'stop_request': stopped, 'program_executed': False}

    def execute_arm_home(self, reservation_id, executor, *, cancel_event):
        """Wait for the stopped task to release control, then execute this reservation once."""
        with self._lock:
            reservation = self._arm_home_reservation
            if reservation is None or reservation['id'] != reservation_id or reservation['claimed']:
                raise DuplicateRequest('arm-home reservation is unknown or already consumed')
            reservation['claimed'] = True
        run = None
        acquired = False
        try:
            finished = reservation['target_finished']
            while finished is not None and not finished.wait(.1):
                if cancel_event.is_set() or reservation['cancelled'].is_set():
                    raise InterruptedError('원위치 요청을 취소했습니다.')
            with self._lock:
                if cancel_event.is_set() or reservation['cancelled'].is_set():
                    raise InterruptedError('원위치 요청을 취소했습니다.')
                self._wait_pending_voice_locked(reservation['request_id'], cancel_event)
                if (self._arm_home_reservation is not reservation or self._active is not None
                        or (reservation['target_request_id'] is not None
                            and (self._last_execution or {}).get('request_id') != reservation['target_request_id'])):
                    raise StaleReference('stopped task did not finish as reserved')
                replacement = self._pending_text_overrides.pop(reservation['request_id'], None)
                if replacement is not None:
                    raise PendingTextPlanChanged(replacement, context=copy.deepcopy(self._context()),
                        stamp=self._session_stamp(), revision=self._revision, cleanup_bumps=1)
                if not self.session._lock.acquire(blocking=False):
                    raise Busy('the stopped task still owns the session')
                acquired = True
                self._generation += 1
                run = {'kind': 'arm_home', 'request_id': reservation['request_id'],
                       'program_id': self._new_id('program'), 'program_revision': self._new_id('revision'),
                       'generation': self._generation, 'round': 1, 'step_index': 0,
                       'task_id': None, 'task_revision': None, 'task_sequence': 0,
                       'start_scene': copy.deepcopy(self.session.current), 'task_start_scene': None,
                       'repeat_active': False, 'stop_repeat': None, 'stop_requested': False,
                       'stop_requests': [], 'atomic_inflight': False, 'transfer_id': None,
                       'transfer_sequence': 0, 'transfer_phase': 'between', 'last_committed_transfer_id': None,
                       'cancel_home': threading.Event()}
                if reservation['request_id'] in self._voice_turn_started:
                    self._voice_turn_started[reservation['request_id']] = True
                self._active = run
                self._bump()
            result = executor(run['cancel_home'], run)
            with self._lock:
                if run['stop_requested'] or run['cancel_home'].is_set():
                    raise InterruptedError('원위치 동작이 중단되었습니다.')
            return result
        finally:
            with self._lock:
                if run is not None and self._active is run:
                    self._active = None
                    self._known_session_stamp = self._session_stamp()
                    self._bump()
                if self._arm_home_reservation is reservation:
                    self._arm_home_reservation = None
                    self._bump()
                if acquired:
                    self.session._lock.release()

    def _check_admission_context(self, is_control, context, owner):
        if is_control and ((owner is not None and not self._same_program(owner))
                           or (owner is None and context != self._context())):
            raise StaleReference('control refers to an earlier program or context')
        if not is_control and context != self._context():
            raise StaleReference('input context is already stale')

    def admit(self, interpretation_id, program):
        """Consume a captured input and validate its proposal without executing it."""
        return self._admit(interpretation_id, program, None)

    def _admit(self, interpretation_id, program, validation_source):
        program = copy.deepcopy(program)
        with self._lock:
            captured = self._interpretations.get(interpretation_id)
            if captured is None:
                raise DuplicateRequest('interpretation ID is unknown or already consumed')
            if captured['transcript'] is None:
                raise ValueError('interpretation has no transcript')
            self._interpretations.pop(interpretation_id)
        text, context = captured['transcript'], captured['context']
        registration_scene, owner = captured['registration_scene'], captured['owner']
        if not isinstance(program, dict):
            raise task_program.ProgramError('invalid_schema', 'the proposal must be a program object')
        if program.get('request_id') != captured['request_id']:
            raise ValueError('proposal request ID does not match the captured input')
        steps = program.get('steps') if isinstance(program, dict) else None
        stop_home_control = (owner is not None and isinstance(steps, list) and len(steps) == 2
                             and isinstance(steps[0], dict) and steps[0].get('op') == 'stop_motion'
                             and isinstance(steps[1], dict) and steps[1].get('op') == 'return'
                             and isinstance(steps[1].get('target'), dict)
                             and steps[1]['target'].get('kind') == 'arm_home')
        is_control = (stop_home_control or
                      (isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], dict)
                       and steps[0].get('op') in ('stop_motion', 'stop_repeat')))
        with self._lock:
            self._refresh_external_state()
            if self._arm_home_reservation is not None and not is_control:
                raise Busy('an arm-home operation has reserved the session')
            if not is_control:
                owner = None
            self._check_admission_context(is_control, context, owner)
        # The capability validates the entire captured request exactly once and
        # binds any supported implicit top target to its original step.
        if validation_source is None:
            validation_source = _ValidatedTopUnstackSource(program, text, context)
        # The direct ticket API is also used by callers that already have a
        # parsed program.  It must not advance a conditional or deferred stop
        # merely because it starts with a stop step. Share the same
        # original-utterance immediacy check as admit_interpretation().
        if is_control and steps[0]['op'] == 'stop_motion' and not _immediate_initial_stop(text, steps[0]):
            raise task_program.ProgramError('unresolved_reference', 'the requested stop is conditional or deferred')
        with self._lock:
            self._refresh_external_state()
            self._check_admission_context(is_control, context, owner)
            state = self._planning_snapshot()
            if any(_requires_registered_top_scene(step, validation_source) for step in steps):
                state['top_scene_registered'] = _registered_top_scene(self.session, state['scene'])
        try:
            if stop_home_control:
                if registration_scene is not None:
                    raise ValueError('control cannot register a scene')
                # The complete two-step proposal was validated above, but
                # only its initial stop is executable through this ticket.
                plan = {'kind': 'control'}
            else:
                plan = self._preflight(program, registration_scene, state, validation_source=validation_source)
            error = None
        except Exception as exc:
            plan, error = None, exc
        with self._lock:
            self._refresh_external_state()
            self._check_admission_context(is_control, context, owner)
            if not is_control and state['train_groups'] != self.session.train_groups:
                raise StaleReference('allowed training groups changed while planning')
            if error is not None:
                raise error
            ticket = self._new_id('ticket')
            self._pending[ticket] = {'program': program, 'transcript': text, 'context': context, 'plan': plan,
                                     'registration_scene': copy.deepcopy(registration_scene), 'owner': owner,
                                     'validation_source': validation_source}
            return ticket

    def _stop_motion(self, request_id):
        active = self._active
        reservation = self._arm_home_reservation
        if (reservation is not None and active is not None
                and active['request_id'] == reservation['target_request_id']):
            # A new stop while waiting also cancels the queued home motion.
            reservation['cancelled'].set()
        if active is None:
            self._generation += 1
            self._bump()
            self.events.append({'event': 'stop_without_active_task', 'request_id': request_id})
            return {'request_id': request_id, 'status': 'no_active_task', 'stop_confirmation': 'not_requested'}
        active['stop_requested'] = True
        self._invalidate_retarget_dispatches(active)
        self._stop_unconfirmed = True
        self._generation += 1
        if active.get('kind') == 'arm_home':
            active['cancel_home'].set()
            cancel_home = active.get('cancel_home_action')
            if cancel_home is not None:
                cancel_home()
        else:
            self.session.cancel.set()
        self._bump()
        self.events.append({'event': 'stop_requested', 'request_id': request_id,
                            'target_request_id': active['request_id'], 'stop_confirmation': 'unconfirmed'})
        return {'request_id': request_id, 'status': 'stop_requested', 'stop_confirmation': 'unconfirmed'}

    def request_motion_stop(self, request_id, *, target_request_id=None):
        """Trusted immediate button-like stop, independent of any model proposal."""
        if not task_program._valid_id(request_id) or (target_request_id is not None
                                                        and not task_program._valid_id(target_request_id)):
            raise ValueError('control and target IDs must be valid opaque IDs')
        with self._lock:
            if target_request_id is not None and (self._active is None
                                                   or self._active['request_id'] != target_request_id):
                raise StaleReference('stop target is no longer active')
            with self.session._request_lock:
                if request_id in self.session._requests:
                    raise DuplicateRequest('request ID was already admitted')
                self.session._requests.add(request_id)
            if self._active is None and self.session._lock.locked():
                self._generation += 1
                self._stop_unconfirmed = True
                self.session.cancel.set()
                self._bump()
                self.events.append({'event': 'stop_requested', 'request_id': request_id,
                                    'target_request_id': None, 'stop_confirmation': 'unconfirmed'})
                return {'request_id': request_id, 'status': 'stop_requested', 'stop_confirmation': 'unconfirmed'}
            return self._stop_motion(request_id)

    def _control(self, entry):
        step, request_id = entry['program']['steps'][0], entry['program']['request_id']
        if step['op'] == 'stop_motion':
            stop_home_control = len(entry['program']['steps']) == 2
            if stop_home_control and self._active is not None and self._active['stop_requested']:
                result = {'request_id': request_id, 'status': 'already_stopping',
                          'stop_confirmation': 'unconfirmed'}
            else:
                result = self._stop_motion(request_id)
            # A direct ticket can contain only the stop even though the
            # captured utterance also asked for a home return or status.  The
            # omitted follow-up was not executed and must remain visible just
            # as it does through admit_interpretation().
            explicit_home = stop_home_control or _explicit_arm_home_followup(entry['transcript'], step)
            explicit_status = _explicit_status_followup(entry['transcript'], step)
            if (explicit_home or explicit_status or _explicit_memory_followup(entry['transcript'], step)
                    or len(task_program_v2._current_stop_spans(entry['transcript'])) > 1):
                result.update(follow_up_status='deferred', program_executed=False)
            if explicit_home:
                result['arm_home_status'] = 'deferred_unsupported'
            if explicit_status:
                result['status_query_status'] = 'deferred_unsupported'
            return result
        active = self._active
        if active is None or not active['repeat_active']:
            self._bump()
            return {'request_id': request_id, 'status': 'no_active_repeat'}
        if active['stop_requested']:
            return {'request_id': request_id, 'status': 'already_stopping', 'stop_confirmation': 'unconfirmed'}
        boundary = step['boundary']
        owner = entry['owner']
        missed = bool(owner and ((boundary == 'after_task'
                                  and (owner['task_id'] != active['task_id'] if owner['task_id'] is not None
                                       else owner['task_sequence'] != active['task_sequence']))
                                 or (boundary == 'after_round' and owner['round'] != active['round'])))
        if boundary == 'after_transfer' and owner:
            if owner['transfer_phase'] == 'in_progress':
                missed = not (owner['transfer_id'] == active['transfer_id']
                              and active['transfer_phase'] == 'in_progress')
            else:
                # No transfer was in progress at interpretation start. If a new
                # one began meanwhile, its safe commit is now the earliest stop.
                missed = active['transfer_sequence'] > owner['transfer_sequence']
        effective = 'after_task' if missed else boundary
        if boundary == 'after_transfer':
            effective = 'after_transfer'
        # A later request cannot weaken an already requested earlier boundary.
        priority = {None: 3, 'after_round': 2, 'after_task': 1, 'after_transfer': 0}
        if priority[effective] < priority[active['stop_repeat']]:
            active['stop_repeat'] = effective
            if effective == 'after_transfer':
                active['stop_transfer_id'] = (active['transfer_id']
                                              if active['transfer_phase'] == 'in_progress' else None)
        active['stop_boundary_missed'] |= missed
        self._bump()
        record = {'request_id': request_id, 'requested_boundary': boundary, 'boundary': active['stop_repeat'],
                  'requested_boundary_missed': missed, 'captured_round': owner['round'] if owner else None,
                  'captured_task_id': owner['task_id'] if owner else None,
                  'captured_transfer_id': owner['transfer_id'] if owner and owner['transfer_phase'] == 'in_progress' else None,
                  'captured_transfer_sequence': owner['transfer_sequence'] if owner else None,
                  'captured_transfer_phase': owner['transfer_phase'] if owner else None,
                  'transfer_id_at_acceptance': active['transfer_id'],
                  'transfer_phase_at_acceptance': active['transfer_phase'],
                  'atomic_inflight_at_acceptance': active['atomic_inflight']}
        active['stop_requests'].append(record)
        self.events.append({'event': 'repeat_stop_requested', 'target_request_id': active['request_id'], **record})
        return {'request_id': request_id, 'status': 'repeat_stop_requested', 'boundary': active['stop_repeat'],
                'requested_boundary': boundary, 'requested_boundary_missed': missed}

    def _step(self, run, step, index, registration_scene, *, repeating=False, validation_source=None):
        with self._lock:
            if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                raise InterruptedError('request was stopped or replaced')
            if run.get('retarget_requested') or run.get('retarget_dispatch_pending'):
                run['retarget_cutoff'] = True
                return None, 'retarget_cutoff'
            if run['stop_repeat'] in ('after_task', 'after_transfer'):
                return None, ('before_next_transfer' if run['stop_repeat'] == 'after_transfer' else 'after_task')
            live_scene = self.session.current if self.session.status == 'registered' else None
            if (_requires_registered_top_scene(step, validation_source)
                    and not _registered_top_scene(self.session, live_scene)):
                raise UnsupportedStep('the current top-of-tower scene needs explicit registration')
            if not _unique_unspecified_target(step, live_scene, validation_source=validation_source):
                if repeating:
                    raise RepeatUnavailable('repeat source is empty or its target is not unique')
                raise StepUnavailable('single-cylinder target is no longer unique in the current source')
            task_id, task_revision = self._new_id('task'), self._new_id('revision')
            run['task_sequence'] += 1
            start = copy.deepcopy(self.session.current)
            task_start = None if self._previous_task is None else copy.deepcopy(self._previous_task['start_scene'])
            run.update(step_index=index, task_id=task_id, task_revision=task_revision, task_start_scene=start)
            run['atomic_inflight'] = False
            self._bump()
        event_start = len(self.session.events)
        def before_transfer():
            with self._lock:
                if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                    raise InterruptedError('motion stop reached before the next transfer')
                if run.get('retarget_requested') or run.get('retarget_dispatch_pending'):
                    run['retarget_cutoff'] = True
                    return False
                if run['stop_boundary_missed'] or run['stop_repeat'] == 'after_transfer':
                    return False
                run['atomic_inflight'] = True
                run['transfer_sequence'] += 1
                run['transfer_id'] = self._new_id('transfer')
                run['transfer_phase'] = 'in_progress'
                self._bump()
                return True
        def on_transfer_commit():
            # Called by the session while holding this coordinator's RLock,
            # in the same critical section as the scene/event commit.
            run['transfer_phase'] = 'committed'
            self._invalidate_retarget_dispatches(run)
            run['last_committed_transfer_id'] = run['transfer_id']
            run['last_committed_transfer_sequence'] = run['transfer_sequence']
            self._bump()
        def after_transfer():
            with self._lock:
                self._invalidate_retarget_dispatches(run)
                run['atomic_inflight'] = False
                run['transfer_id'] = None
                run['transfer_phase'] = 'between'
        try:
            with self.session._memory_binding_lock:
                memory = self.session.memory
                guard = getattr(memory, 'synchronized', None)
                with guard() if callable(guard) else nullcontext():
                    if self._memory_stamp() != self._expected_memory_stamp:
                        raise StaleReference('memory changed outside the active coordinator')
                    try:
                        self.session._execute_validated_step(
                            step, registration_scene=registration_scene, registered_initial=self._registered_initial,
                            task_start=task_start, commit_lock=self._lock,
                            is_current=lambda: self._active is run and run['generation'] == self._generation,
                            before_transfer=before_transfer, after_transfer=after_transfer,
                            on_transfer_commit=on_transfer_commit, planned_tasks=True,
                            validation_source=validation_source)
                    finally:
                        self._expected_memory_stamp = self._memory_stamp()
        except UnsupportedStep as exc:
            # A repeat can become infeasible as its scene changes. Planning is
            # done without the coordinator lock so a stop can arrive promptly.
            # Once any transfer was attempted, the ordinary uncertain path wins.
            attempted = any(event.get('event') == 'transfer' for event in self.session.events[event_start:])
            if attempted and isinstance(exc, StackReferenceUnavailable):
                raise RuntimeError('stack reference failed after a transfer was attempted') from exc
            if attempted:
                raise
            if not repeating:
                raise StepUnavailable(str(exc)) from exc
            with self._lock:
                if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                    raise InterruptedError('motion stop took precedence over unavailable repeat') from exc
                run.update(step_index=None, task_id=None, task_revision=None, task_start_scene=None)
                self._bump()
            raise RepeatUnavailable(str(exc)) from exc
        except RepeatBoundaryReached:
            with self._lock:
                if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                    raise InterruptedError('motion stop took precedence over repeat stop')
                if self._memory_revision is not None and any(
                        event.get('committed') for event in self.session.events[event_start:]
                        if event.get('event') == 'transfer'):
                    self._memory_revision = self._new_id('memory')
                self.events.append({'event': 'task_stopped_before_next_transfer', 'request_id': run['request_id'],
                                    'round': run['round'], 'step_index': index, 'task_id': task_id})
                run.update(step_index=None, task_id=None, task_revision=None, task_start_scene=None)
                self._bump()
                boundary = ('retarget_cutoff' if run.get('retarget_cutoff') else
                            self._transfer_stop_boundary(run) if run['stop_repeat'] == 'after_transfer' else 'before_next_transfer')
                return None, boundary
        except Exception:
            # A physical release can commit an expected scene before a later
            # retreat fails. Preserve the already advanced memory reference.
            with self._lock:
                if self._memory_revision is not None and any(
                        event.get('committed') for event in self.session.events[event_start:]
                        if event.get('event') == 'transfer'):
                    self._memory_revision = self._new_id('memory')
            raise
        finally:
            with self._lock:
                run['atomic_inflight'] = False
        with self._lock:
            if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                raise InterruptedError('late step result after a stop')
            if self._memory_stamp() != self._expected_memory_stamp:
                raise StaleReference('memory changed after the active step')
            if step['op'] == 'task' and step['kind'] == 'register':
                self._registered_initial = copy.deepcopy(self.session.current)
                self._registration_revision = self._new_id('registration')
                self._memory_revision = None
                self._previous_task = None
                self._previous_program = None
            else:
                self._previous_task = (None if (run.get('retarget_requested') or run.get('retarget_dispatch_pending')
                                               or run.get('retarget_cutoff')) else
                    {'id': task_id, 'revision': task_revision, 'start_scene': start,
                     'step': copy.deepcopy(step), 'program_id': run['program_id'],
                     'program_revision': run['program_revision'], 'round': run['round'],
                     'step_index': index, 'validation_source': copy.deepcopy(validation_source)})
                if step['op'] == 'task' and step['kind'] == 'remember':
                    self._memory_revision = self._new_id('memory')
                elif self._memory_revision is not None and any(
                        event.get('committed') for event in self.session.events[event_start:]
                        if event.get('event') == 'transfer'):
                    self._memory_revision = self._new_id('memory')
            self.events.append({'event': 'step_complete', 'request_id': run['request_id'],
                                'round': run['round'], 'step_index': index, 'task_id': task_id})
            run['last_completed'] = {'round': run['round'], 'step_index': index, 'task_id': task_id,
                                     'task_revision': task_revision}
            run.update(step_index=None, task_id=None, task_revision=None, task_start_scene=None)
            self._bump()
            progress = sum(bool(event.get('committed')) for event in self.session.events[event_start:]
                           if event.get('event') == 'transfer')
            if run.get('retarget_requested') or run.get('retarget_dispatch_pending'):
                run['retarget_cutoff'] = True
                return progress, 'retarget_cutoff'
            return progress, run['stop_repeat']

    def execute(self, ticket, *, cancel_event=None):
        """Explicitly execute an admitted ticket; a stale or duplicate ticket cannot run."""
        with self._lock:
            entry = self._pending.get(ticket)
            if entry is None:
                raise DuplicateRequest('ticket is unknown or already consumed')
            try:
                self._wait_pending_voice_locked(entry['program']['request_id'], cancel_event)
            except BaseException:
                self._pending.pop(ticket, None)
                raise
            entry = self._pending.pop(ticket, None)
            if entry is None:
                raise InterruptedError('요청을 취소했습니다.')
            self._refresh_external_state()
            if entry['plan']['kind'] == 'control' and entry['owner'] is not None:
                if not self._same_program(entry['owner']):
                    raise StaleReference('control target is no longer the active program')
            elif entry['context'] != self._context():
                raise StaleReference('request context changed after admission')
            replacement = self._pending_text_overrides.pop(entry['program']['request_id'], None)
            if replacement is not None:
                raise PendingTextPlanChanged(replacement, context=copy.deepcopy(self._context()),
                    stamp=self._session_stamp(), revision=self._revision, cleanup_bumps=0)
            if entry['plan']['kind'] == 'control':
                return self._control(entry)
            if self._arm_home_reservation is not None:
                raise Busy('an arm-home operation has reserved the session')
            if self._active is not None or not self.session._lock.acquire(blocking=False):
                raise Busy('session is already executing')
            if self.session.status == 'uncertain' or self._stop_unconfirmed or self.session.cancel.is_set():
                self.session._lock.release()
                raise RuntimeError('session needs recovery or stop confirmation')
            self._generation += 1
            run = {'request_id': entry['program']['request_id'], 'program_id': self._new_id('program'),
                   'program_revision': self._new_id('revision'), 'generation': self._generation,
                   'round': 1, 'step_index': None, 'task_id': None, 'task_revision': None, 'task_sequence': 0,
                   'start_scene': copy.deepcopy(self.session.current), 'task_start_scene': None,
                   'repeat_active': entry['plan']['repeat_index'] is not None,
                   'stop_repeat': None, 'stop_requested': False, 'stop_requests': [], 'last_completed': None,
                   'stop_boundary_missed': False, 'atomic_inflight': False,
                   'transfer_id': None, 'transfer_sequence': 0, 'transfer_phase': 'between',
                   'last_committed_transfer_id': None, 'last_committed_transfer_sequence': 0,
                   'stop_transfer_id': None, 'retarget_cutoff': False,
                   '_completion_event': threading.Event()}
            if entry['program']['request_id'] in self._voice_turn_started:
                self._voice_turn_started[entry['program']['request_id']] = True
            self._active = run
            self._bump()
        plan, program = entry['plan'], entry['program']
        validation_source = entry['validation_source']
        start_events, begin_scene = len(self.session.events), copy.deepcopy(run['start_scene'])
        status, detail, actual_boundary, finished_scene_state = 'completed', None, None, None
        not_started_reason = None
        finished_events, finished_retarget = [], None
        try:
            initial = program['steps'] if plan['repeat_index'] is None else program['steps'][:plan['repeat_index']]
            progress = 0
            def check_step_boundary(moved, pending):
                if pending == 'retarget_cutoff':
                    return 'retarget_cutoff', 'after_transfer'
                if moved is None:
                    return 'repeat_stopped', (pending if pending in ('before_next_transfer', 'after_transfer')
                                              else 'before_next_task')
                if pending in ('after_task', 'after_transfer'):
                    return 'repeat_stopped', (self._transfer_stop_boundary(run)
                                              if pending == 'after_transfer' else pending)
                return None

            for index, step in enumerate(initial):
                try:
                    moved, pending = self._step(run, step, index, entry['registration_scene'],
                                                validation_source=validation_source)
                except StackReferenceUnavailable as exc:
                    # A GRU return can reveal the next step's scene only at
                    # execution. Failed symbolic binding issues no movement.
                    self.session._ready()
                    status, detail = 'target_unavailable', str(exc)
                    break
                boundary = check_step_boundary(moved, pending)
                if moved is not None:
                    progress += moved
                if boundary is not None:
                    status, actual_boundary = boundary
                    break
            if plan['repeat_index'] is not None and status == 'completed':
                with self._lock:
                    pending = run['stop_repeat']
                if pending:
                    status, actual_boundary = 'repeat_stopped', (self._transfer_stop_boundary(run)
                                                                 if pending == 'after_transfer' else 'after_round')
                elif plan['prior_body'] and progress == 0:
                    status = 'no_progress'
                else:
                    rounds = 0
                    while plan['count']['mode'] == 'continuous' or rounds < plan['count']['value']:
                        with self._lock:
                            if run['stop_repeat']:
                                status, actual_boundary = 'repeat_stopped', (self._transfer_stop_boundary(run)
                                    if run['stop_repeat'] == 'after_transfer' else 'before_next_round')
                                break
                            if run['generation'] != self._generation or self.session.cancel.is_set():
                                raise InterruptedError('request stopped before the next round')
                            run['round'] = rounds + (2 if plan['prior_body'] else 1)
                            self._bump()
                        progress = 0
                        for index, step in enumerate(plan['body']):
                            try:
                                moved, pending = self._step(run, step, index, None, repeating=True,
                                                            validation_source=plan['body_validation_source'])
                            except RepeatUnavailable as exc:
                                self.session._ready()
                                status, detail = 'repeat_unavailable', str(exc)
                                with self._lock:
                                    self.events.append({'event': 'repeat_unavailable', 'request_id': run['request_id'],
                                                        'round': run['round'], 'step_index': index, 'detail': detail})
                                break
                            boundary = check_step_boundary(moved, pending)
                            if moved is not None:
                                progress += moved
                            if boundary is not None:
                                status, actual_boundary = boundary
                                break
                        rounds += 1
                        if status != 'completed':
                            break
                        with self._lock:
                            pending = run['stop_repeat']
                        if pending:
                            status, actual_boundary = 'repeat_stopped', (self._transfer_stop_boundary(run)
                                                                         if pending == 'after_transfer' else 'after_round')
                            break
                        if progress == 0:
                            status = 'no_progress'
                            break
            with self._lock:
                if self._active is not run or run['generation'] != self._generation or self.session.cancel.is_set():
                    raise InterruptedError('late program result after a stop')
                if status in ('completed', 'no_progress', 'repeat_unavailable') and run['stop_repeat']:
                    actual_boundary = (self._transfer_stop_boundary(run) if run['stop_repeat'] == 'after_transfer' else
                                       'before_next_task' if status == 'repeat_unavailable' else 'before_next_round')
                    status = 'repeat_stopped'
                run['repeat_active'] = False
                saved_steps = plan['body'] if plan['repeat_index'] is not None else initial
                if run.get('retarget_requested') or run.get('retarget_dispatch_pending') or run.get('retarget_cutoff'):
                    # A retarget request may be accepted yet never applied.
                    # Session events distinguish an expected return from that
                    # outcome without claiming observed object motion.
                    if status in ('completed', 'retarget_cutoff'):
                        evidence = _finished_retarget(run, self.session.events[start_events:])
                        status = {'completed_expected': 'retargeted_expected', 'incomplete': 'retarget_incomplete',
                                  'not_applied': 'retarget_not_applied', 'unconfirmed': 'retarget_unconfirmed'}[evidence['status']]
                    self._previous_program = None
                else:
                    self._previous_program = {'id': run['program_id'], 'revision': run['program_revision'],
                                              'request_id': run['request_id'], 'start_scene': begin_scene,
                                              'transcript': entry['transcript'], 'admission_context': copy.deepcopy(entry['context']),
                                              'steps': copy.deepcopy(saved_steps), 'status': status,
                                              'validation_source': copy.deepcopy(
                                                  plan['body_validation_source'] if plan['repeat_index'] is not None
                                                  else validation_source)}
                self.events.append({'event': 'program_finished', 'request_id': run['request_id'], 'status': status})
                if status == 'repeat_stopped':
                    transfers = [event for event in self.session.events[start_events:]
                                 if event.get('event') == 'transfer' and event.get('committed')]
                    last_transfer = None if not transfers else {key: copy.deepcopy(transfers[-1][key])
                                                               for key in ('before', 'planned_after', 'task')}
                    if last_transfer is not None:
                        last_transfer['transfer_id'] = run['last_committed_transfer_id']
                        last_transfer['transfer_sequence'] = run['last_committed_transfer_sequence']
                    if last_transfer is not None and 'commit_kind' in transfers[-1]:
                        last_transfer['commit_kind'] = transfers[-1]['commit_kind']
                    run['last_committed_transfer'] = last_transfer
                    self.events.append({'event': 'repeat_stopped', 'request_id': run['request_id'],
                                        'actual_boundary': actual_boundary,
                                        'last_completed': copy.deepcopy(run['last_completed']),
                                        'last_committed_transfer': last_transfer,
                                        'requests': copy.deepcopy(run['stop_requests'])})
        except NoMotionObserved as exc:
            if self.session.cancel.is_set() or self.session.status == 'uncertain':
                self.session._halt(f'{type(exc).__name__}: {exc}')
                status = 'uncertain'
            else:
                status = 'motion_not_observed'
            detail = str(exc)
            with self._lock:
                self.events.append({'event': 'program_stopped_no_motion', 'request_id': run['request_id'],
                                    'detail': detail})
        except StepUnavailable as exc:
            # A preceding step may have completed (e.g. learned-memory
            # restore), but no transfer was attempted by this unavailable
            # step. Keep the last committed expected scene and memory.
            if self.session.cancel.is_set() or self.session.status == 'uncertain':
                self.session._halt(f'{type(exc).__name__}: {exc}')
                status = 'uncertain'
            else:
                status = 'step_unavailable'
                not_started_reason = exc.not_started_reason
            detail = str(exc)
            with self._lock:
                self.events.append({'event': 'program_step_unavailable', 'request_id': run['request_id'],
                                    'detail': detail, 'not_started_reason': not_started_reason})
        except Exception as exc:
            self.session._halt(f'{type(exc).__name__}: {exc}')
            status, detail = 'uncertain', f'{type(exc).__name__}: {exc}'
            with self._lock:
                self.events.append({'event': 'program_stopped_uncertain', 'request_id': run['request_id'],
                                    'detail': detail})
        except BaseException:
            self.session._halt('request interrupted')
            status = 'uncertain'
            raise
        finally:
            try:
                with self._lock:
                    try:
                        scene_state = getattr(self.session, 'scene_state', None)
                        if callable(scene_state):
                            # Capture the result before releasing the session for another request.
                            finished_scene_state = scene_state()
                    finally:
                        if self._active is run:
                            try:
                                self._invalidate_retarget_dispatches(run)
                                if self._memory_stamp() != self._expected_memory_stamp:
                                    self._memory_revision = None
                                    self._expected_memory_stamp = self._memory_stamp()
                                self._last_execution = {'request_id': run['request_id'], 'program_id': run['program_id'],
                                                        'program_revision': run['program_revision'], 'status': status,
                                                        'start_scene': copy.deepcopy(run['start_scene']),
                                                        'transcript': entry['transcript'], 'admission_context': copy.deepcopy(entry['context']),
                                                        'last_completed': copy.deepcopy(run['last_completed']),
                                                        'stop_confirmation': 'unconfirmed' if run['stop_requested'] else 'not_requested'}
                                if not_started_reason is not None:
                                    self._last_execution['not_started_reason'] = not_started_reason
                                finished_events = copy.deepcopy(self.session.events[start_events:])
                                failed_transfer = next((event for event in reversed(finished_events)
                                    if event.get('event') == 'transfer' and isinstance(event.get('result'), dict)), None)
                                if failed_transfer is not None and status not in ('completed', 'retargeted_expected'):
                                    self._last_execution['failure_termination'] = failed_transfer['result'].get('termination')
                                    self._last_execution['failure_previous_termination'] = failed_transfer['result'].get('previous_termination')
                                    self._last_execution['failure_error'] = copy.deepcopy(failed_transfer['result'].get('error'))
                                    self._last_execution['release_expected'] = failed_transfer.get('release_expected') is True
                                finished_retarget = _finished_retarget(run, finished_events)
                                if finished_retarget is not None:
                                    self._last_execution['retarget'] = copy.deepcopy(finished_retarget)
                                if finished_scene_state is not None:
                                    self._last_execution['scene_state'] = copy.deepcopy(finished_scene_state)
                                    self._last_execution['completion_semantics'] = 'execution_only_no_physical_success_claim'
                            finally:
                                self._active = None
                                self._bump()
                            self._known_session_stamp = self._session_stamp()
            finally:
                self.session._lock.release()
                run['_completion_event'].set()
        result = {'request_id': run['request_id'], 'program_id': run['program_id'], 'status': status,
                  'detail': detail, 'stop_confirmation': 'unconfirmed' if run['stop_requested'] else 'not_requested',
                  'repeat_stop': ({'actual_boundary': actual_boundary, 'last_completed': copy.deepcopy(run['last_completed']),
                                   'last_committed_transfer': copy.deepcopy(run.get('last_committed_transfer')),
                                   'requests': copy.deepcopy(run['stop_requests'])}
                                  if status == 'repeat_stopped' else None),
                  'events': finished_events}
        if not_started_reason is not None:
            result['not_started_reason'] = not_started_reason
        if finished_retarget is not None:
            result['retarget'] = finished_retarget
        if finished_scene_state is not None:
            result.update(scene_state=finished_scene_state, physical_success=None,
                          completion_semantics='execution_only_no_physical_success_claim')
        return result
