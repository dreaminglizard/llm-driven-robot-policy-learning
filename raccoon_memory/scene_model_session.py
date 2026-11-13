"""Headless, simulation-only connection of validated steps, recall and a student.

This is a development harness, not the application or a learned scene planner.
The evaluator, not this session, owns any remembered-scene answer key.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
from functools import wraps
import json
from pathlib import Path
import re
import threading

from . import scene_demo_data, scene_features, scene_policy, scene_recall, task_catalog, task_program
from scripts import evaluate_scene_policy

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PLAN = ROOT / "configs/scene_split_plan.v1.json"


class UnsupportedStep(ValueError):
    """The bounded development adapter does not implement this control."""


class StackReferenceUnavailable(UnsupportedStep):
    """A named support cannot be used from the current symbolic scene, before motion."""


class DuplicateRequest(ValueError):
    pass


class RepeatBoundaryReached(RuntimeError):
    """A repeat stop was reached before the next atomic transfer."""


def _train_path_to(start, is_goal, train_groups):
    """Find a shortest allowed path to a scene satisfying a trusted predicate."""
    groups = frozenset(train_groups)
    queue, previous = deque([start]), {start: None}
    while queue:
        before = queue.popleft()
        if is_goal(before):
            path = []
            while previous[before] is not None:
                parent, action = previous[before]
                path.append(action)
                before = parent
            return list(reversed(path))
        for source in range(1, 5):
            for destination in range(1, 5):
                try:
                    after = task_catalog.transfer(before, source, destination)
                except ValueError:
                    continue
                action = {"source_slot": source, "destination_slot": destination}
                if after not in previous and scene_policy.split_group_for(before, action) in groups:
                    previous[after] = (before, action)
                    queue.append(after)
    raise UnsupportedStep("no path using only the student's allowed training groups")


def train_path(start, goal, train_groups):
    """Find a shortest legal symbolic path, keeping every edge in train groups."""
    start = scene_recall.scene(start)
    goal = scene_recall.scene(goal, scene_recall.identities(start))
    return _train_path_to(start, lambda value: value == goal, train_groups)


_TOP_UNSTACK_PHRASES = (
    re.compile(r'\bunstack\s+(?:the\s+)?top(?:most)?\s+(?:cylinder|object)\s+'
               r'(?:in|from|at|of)\s+(?:the\s+)?slot\s+'
               r'(?P<source>[1-4]|one|two|three|four)\b', re.IGNORECASE | re.ASCII),
    re.compile(r'(?P<source>[1-4일이삼사한두세네])\s*번(?:째)?\s*(?:(?:자리|위치)?\s*(?:에서|의)|'
               + task_program._KO_TOWER + r'(?:에서|의)?)\s*'
               r'(?:맨|가장)\s*위(?:에\s*(?:있는|놓인))?\s*(?:원통|물체)'
               r'(?!\s*(?:[1-4]|[일이삼사])\s*번)'),
)

# A target reference plus an identity relation and a concrete identity is an
# additional condition, whether positive or negative.  We do not infer that
# condition into the model's step: until it is represented and checked, the
# originating move must be held.  A separately named object's description,
# e.g. "cylinder 1 is not cylinder 2", is not such a target reference.
_EN_TARGET_IDENTITY = (
    r"\b(?:it|(?:this|that)\s+(?:one|cylinder|object)|"
    r"the\s+(?:(?:top|topmost|selected|moved)\s+)?(?:one|cylinder|object))\s+"
    r"(?:is(?:\s+not)?|isn['’]?t|was(?:\s+not)?|wasn['’]?t|"
    r"(?:(?:must|should|may|can|could)\s+(?:not\s+)?|cannot\s+|"
    r"(?:can|mustn|shouldn|couldn)['’]t\s+)(?:be|equal|match)|"
    r"equals?|does\s+not\s+equal)\s+"
    r"(?:(?:the|a|an)\s+)?(?:"
    r"(?:cylinder|object)\s*(?:(?:number|id|no\.?)\s*|#\s*)?(?:[1-4]|one|two|three|four)|"
    r"(?:number|no\.?)\s*(?:[1-4]|one|two|three|four)|"
    r"(?:red|blue|green|yellow|first|second|third|fourth)\s+(?:cylinder|object|one))\b"
)
_KO_TARGET_IDENTITY = (
    r"(?:(?:그|이)\s*(?:원통|물체|것|거)(?:은|는|이|가)?|그건|그게)\s*"
    r"(?:[1-4]|[일이삼사])\s*번\s*(?:원통|물체)(?:이|가|은|는)?\s*"
    r"(?:아니|아닌|이면\s*안|여야|이어야|맞아|맞다)"
)

# These are unsupported *directives/qualifications*, not a parser that
# resolves exclusions or revises a destination.  Scanning the whole captured
# utterance matters: a model can otherwise supply evidence ending just before
# a later qualification.  Descriptions and quoted words are not directives.
_UNSUPPORTED_TARGET_CONSTRAINTS = tuple(re.compile(pattern, re.IGNORECASE | re.ASCII) for pattern in (
    _EN_TARGET_IDENTITY,
    _KO_TARGET_IDENTITY,
    r"\b(?:except(?:\s+for)?|excluding|but\s+not|other\s+than)\s+"
    r"(?:the\s+)?(?:(?:red|blue|green|yellow|top)\s+)?(?:cylinder|object|block|piece|one|it|them)s?\b",
    r"\b(?:do\s+not|don['’]?t|never|must\s+not|should\s+not)\s+(?:ever\s+)?"
    r"(?:move|unstack|stack|transfer|pick(?:\s+up)?|touch|put|place|carry)\b",
    r"(?:움직이|옮기|이동하|건드리|내리|내려놓|집|들|쌓)[가-힣]{0,6}지\s*(?:마|말)",
    r"(?:움직이|옮기|이동하|건드리|내리|내려놓|집|들|쌓)[가-힣]{0,6}면\s*안\s*(?:돼|되)",
    r"(?:[1-4]|[일이삼사])\s*번\s*(?:원통|물체)?(?:은|는|을|를|이|가)?\s*"
    r"(?:말고|빼고|제외(?:하고|해|한)?)",
    r"(?:[1-4]|[일이삼사])\s*번\s*(?:원통|물체)(?:이|가)?\s*(?:면|라면|이라면|인\s*경우)",
    r"\b(?:unless|only\s+if|only\s+when|provided(?:\s+that)?)\b[^.!?。？！]{0,80}"
    r"\b(?:it|they|them|cylinder|object|one)\b",
    r"\b(?:leave|keep)\s+(?:it|them|(?:the\s+)?(?:cylinder|object|piece)\s*(?:[1-4])?)\s+"
    r"(?:where\s+(?:it|they)\s+(?:is|are)|in\s+place|untouched|alone|as\s+is|stationary)\b",
    r"\b(?:cylinder|object|piece)\s*[1-4]\s+(?:(?:must|should)\s+)?"
    r"(?:remain|stay|be\s+left)\s+(?:in\s+place|untouched|alone|stationary)\b",
    r"(?:[1-4]|[일이삼사])\s*번\s*(?:원통|물체)(?:은|는|을|를|이|가)?\s*"
    r"그대로\s*(?:두|둬|놔|놓|유지)",
    r"\b(?:actually|instead|rather|correction)\b[^.!?。？！]{0,100}"
    r"\b(?:put|place|move|transfer|unstack|stack|send)\b[^.!?。？！]{0,100}\bslot\s+"
    r"(?:[1-4]|one|two|three|four)\b",
    r"\b(?:put|place|move|transfer|send)\b[^.!?。？！]{0,100}\bslot\s+"
    r"(?:[1-4]|one|two|three|four)\b[^.!?。？！]{0,60}\binstead\b",
    r"\b(?:change|correct)\s+(?:the\s+)?(?:destination|target)\b[^.!?。？！]{0,80}"
    r"\bslot\s+(?:[1-4]|one|two|three|four)\b",
    r"(?:아니|정정|대신)[^.!?。？！]{0,80}(?:[1-4]|[일이삼사])\s*번"
    r"[^.!?。？！]{0,60}(?:옮|내려|놓|보내)",
))


def _quoted_span(text, start, end):
    if task_program._quoted(text, start, end):
        return True
    # Apostrophes in contractions must not terminate an enclosing quote.
    for match in re.finditer(r"(?<![A-Za-z0-9])'(?:[^']|'(?=[A-Za-z]))*'(?![A-Za-z0-9])"
                             r"|‘(?:[^’]|’(?=[A-Za-z]))*’", text):
        if match.start() <= start and end <= match.end():
            return True
    return False


def _has_unsupported_target_constraint(text):
    """Hold rather than silently drop a target prohibition or correction."""
    for pattern in _UNSUPPORTED_TARGET_CONSTRAINTS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if _quoted_span(text, start, end):
                continue
            # Reported speech and descriptions of the speaker's own habits do
            # not ask this executor to restrict its motion.
            prefix = text[max(0, start - 60):start]
            suffix = text[end:min(len(text), end + 35)]
            if (re.search(r"\b(?:says?|reads?|states?|quotes?|writes?|i|we)\s*:?\s*$",
                          prefix, re.IGNORECASE | re.ASCII)
                    or re.match(r"\s*(?:라고|라는)\s*(?:쓰|적|표시|말|읽)", suffix)):
                continue
            # "except cylinder 2 is blue" describes another object rather
            # than removing it from the requested target set.
            if re.match(r"\s+(?:[1-4]\s+)?(?:is|was|looks|has)\b", suffix, re.IGNORECASE | re.ASCII):
                continue
            return True
    return False


def _top_mention_binds_unstack(evidence, match):
    """Do not bind a mention used by inspection, negation or preservation."""
    tail = evidence[match.end():]
    # Korean puts the unstack verb after the object.  It must be the next
    # relevant action, not an action on a different object after inspection.
    if re.match(r'\s*(?:이|가|은|는|을|를)?\s*(?:아닌|아니|말고|빼고|제외)', tail):
        return False
    if re.search(r'\b(?:unstack)\b', match.group(), re.IGNORECASE | re.ASCII):
        return not re.search(r'\b(?:leave|keep)\s+(?:it|the\s+(?:top\s+)?(?:cylinder|object))\s+'
                             r'(?:where\s+it\s+is|in\s+place|untouched)\b', tail,
                             re.IGNORECASE | re.ASCII)
    movement = re.search(r'내려|풀|분리', tail)
    if movement is None:
        return False
    between = tail[:movement.start()]
    if re.search(r'아닌|아니|말고|빼고|제외|그대로\s*(?:두|놔|놓)|확인|살펴|점검|검사', between):
        return False
    start, end = match.end() + movement.start(), match.end() + movement.end()
    return not task_program._negated_action(evidence, start, end)


def _step_key(step):
    return json.dumps(step, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


_EMPTY_DESTINATIONS = re.compile(
    r'(?P<ko>[1-4일이삼사한두세네])\s*번\s*빈\s*자리(?:로|에(?!서))|'
    r'\b(?:to|into|onto|in|at)\s+(?:the\s+)?empty\s+(?:slot|position)\s+'
    r'(?P<en>[1-4]|one|two|three|four)\b', re.IGNORECASE | re.ASCII)


def _explicit_empty_destination(step, text):
    if not _EMPTY_DESTINATIONS.search(text):
        return False
    from . import task_program_v2
    evidence = step['evidence']
    start = text.find(evidence)
    # In this validated English form “and put it” completes the same unstack,
    # although it starts a new clause for the general transfer parser.
    if step['kind'] == 'unstack':
        for pair in task_program_v2._EN_UNSTACK_TOP.finditer(text):
            if (start <= pair.start('verb') and pair.end('verb') <= start + len(evidence)
                    and task_program_v2._explicit_match(text, pair)
                    and task_program.bilingual._value(pair.group('source')) == step['source_slot']
                    and task_program.bilingual._value(pair.group('destination')) == step['destination_slot']):
                return True
    clause, offset = task_program_v2._transfer_clause(step, text, start, start + len(evidence))
    for match in _EMPTY_DESTINATIONS.finditer(clause):
        left, right = offset + match.start(), offset + match.end()
        if _quoted_span(text, left, right) or task_program._negated_span(text, left, right):
            continue
        word = match.group('ko') or match.group('en').lower()
        number = int(word) if word.isdecimal() else (
            task_program.common._NUMBERS.get(word) if match.group('ko') else task_program.bilingual._value(word))
        if number == step['destination_slot']:
            return True
    return False


class _ValidatedTopUnstackSource:
    """A capability minted only after validating the complete request.

    It binds exact step values to the captured original request. A plain step,
    transcript fragment, or caller-supplied Boolean cannot authorize the
    implicit-top interpretation. The program is not rewritten. The historic
    name also covers the explicitly worded, standalone top-of-tower move.
    """
    def __init__(self, program, text, context, *, v1=False, on_valid_step=None):
        from . import task_program_v2
        if v1:
            task_program.validate_program(program, text, context, on_valid_step=on_valid_step)
        else:
            task_program_v2.validate_program(program, text, context, on_valid_step=on_valid_step)
        allowed = []
        registered_scene_required = []
        empty_destination_required = []
        blocked = []
        has_target_constraint = _has_unsupported_target_constraint(text)
        for step in program['steps']:
            if step.get('op') != 'task' or step.get('kind') not in ('move', 'stack', 'unstack'):
                continue
            if has_target_constraint:
                blocked.append(_step_key(step))
            if _explicit_empty_destination(step, text):
                empty_destination_required.append(_step_key(step))
            if step['kind'] == 'move' and not v1 and not has_target_constraint:
                evidence = step['evidence']
                start = text.find(evidence)
                matches = [match for match in task_program_v2._EN_TOP_TOWER_MOVE.finditer(
                    text, start, start + len(evidence)) if task_program_v2._explicit_match(text, match)]
                if (len(matches) == 1 and all(step[key] is None for key in ('object_id', 'count', 'height', 'layout'))):
                    match = matches[0]
                    source = task_program.bilingual._value(match.group('source'))
                    destination = task_program.bilingual._value(match.group('destination'))
                    if (task_program_v2._top_tower_context_is_supported(text, match)
                            and source != destination and step['source_slot'] == source
                            and step['destination_slot'] == destination):
                        allowed.append(_step_key(step))
                        registered_scene_required.append(_step_key(step))
            if step['kind'] != 'unstack':
                continue
            evidence = step['evidence']
            top_to_empty = task_program_v2.explicit_top_unstack_roles(text, evidence=evidence)
            if (not v1 and step['object_id'] is None and step['count'] is None
                    and top_to_empty == (step['source_slot'], step['destination_slot'], None)):
                allowed.append(_step_key(step))
                continue
            evidence_start = text.find(evidence)
            # A stated object ID is an additional constraint. It must not be
            # silently discarded by resolving only the top of the source.
            _, _, _, objects, _, _ = task_program._task_supports(
                evidence, whole=text, offset=evidence_start)
            matches = [match for pattern in _TOP_UNSTACK_PHRASES for match in pattern.finditer(evidence)]
            # A top mention attached to a different action or explicitly
            # excluded cannot authorize even an explicitly counted transfer.
            if any(not _top_mention_binds_unstack(evidence, match)
                   or _quoted_span(text, evidence_start + match.start(), evidence_start + match.end())
                   for match in matches):
                blocked.append(_step_key(step))
            if step.get('count') is not None or step.get('object_id') is not None:
                continue
            def matches_source(match):
                word = match.group('source').lower()
                source = (int(word) if word.isdecimal() else
                          task_program.bilingual._EN_VALUES.get(word, task_program.common._NUMBERS.get(word)))
                start, end = evidence_start + match.start(), evidence_start + match.end()
                return (source == step['source_slot'] and not task_program._negated_span(text, start, end)
                        and not task_program._negated_action(text, start, end)
                        and not _quoted_span(text, start, end)
                        and not task_program._question_only(text, start, end)
                        and _top_mention_binds_unstack(evidence, match))
            if not objects and any(matches_source(match) for match in matches):
                allowed.append(_step_key(step))
        self._allowed = frozenset(allowed)
        self._blocked = frozenset(blocked)
        self._registered_scene_required = frozenset(registered_scene_required)
        self._empty_destination_required = frozenset(empty_destination_required)

    def allows(self, step):
        return _step_key(step) in self._allowed and not self.blocks(step)

    def blocks(self, step):
        return _step_key(step) in self._blocked

    def requires_empty_destination(self, step):
        return _step_key(step) in self._empty_destination_required

    def requires_registered_scene(self, step):
        return _step_key(step) in self._registered_scene_required and self.allows(step)


class _LocalModelProgram:
    """The local model chose these typed transfers. A slot-only transfer takes one exposed top."""
    def __init__(self, program):
        self._top_steps = frozenset(_step_key(s) for s in program['steps']
                                    if s['op'] == 'task' and s['kind'] in ('move', 'stack', 'unstack')
                                    and s['source_slot'] is not None and s['object_id'] is None
                                    and s['count'] is None)

    def allows(self, step):
        return _step_key(step) in self._top_steps


def _validated_single_top_unstack(step, validation_source):
    """Only a capability from whole-program validation can bind the top."""
    return (isinstance(step, dict) and isinstance(validation_source, (_ValidatedTopUnstackSource, _LocalModelProgram))
            and validation_source.allows(step))


def _reject_unsupported_target_constraint(step, validation_source, scene=None):
    if isinstance(validation_source, _LocalModelProgram):
        # The model expresses the required resulting height in the structured step.
        # plan_task still checks it against the actual planning scene.
        return
    if (isinstance(step, dict) and isinstance(validation_source, _ValidatedTopUnstackSource)
            and validation_source.blocks(step)):
        raise UnsupportedStep('target prohibition or correction needs clarification')
    destination = step.get('destination_slot')
    if scene is not None and destination is not None and scene[destination - 1]:
        required = (validation_source.requires_empty_destination(step)
                    if isinstance(validation_source, _ValidatedTopUnstackSource)
                    else _explicit_empty_destination(step, step.get('evidence', '')))
        if required:
            raise UnsupportedStep('the requested empty destination is occupied in the expected layout')


def _transfer_slots(scene, step, validation_source=None):
    source, moved = step['source_slot'], step['object_id']
    support = step.get('destination_object_id')
    failure = UnsupportedStep if support is None else StackReferenceUnavailable
    if moved is not None and moved == support:
        raise failure('the moved cylinder cannot support itself')
    if source is None:
        source = next((i for i, stack in enumerate(scene, 1) if stack and stack[-1] == moved), None)
        if source is None:
            raise failure('specified cylinder is not available at the top of a stack')
    if support is None:
        if source == step['destination_slot']:
            raise failure('source and destination resolve to the same slot')
        return source, step['destination_slot']
    source_stack = scene[source - 1]
    if (not source_stack or (moved is not None and source_stack[-1] != moved)
            or (moved is None and step['count'] is None and len(source_stack) != 1
                and not (isinstance(validation_source, _LocalModelProgram) and validation_source.allows(step)))):
        raise failure('the requested source cylinder is not uniquely available at its stated slot')
    # A named support must be exposed AND have room above it. scene() already
    # checks globally unique IDs, bottom-to-top order and the two-level limit.
    destination = next((i for i, stack in enumerate(scene, 1) if stack == (support,)), None)
    if destination is None or source == destination:
        raise failure('the named supporting cylinder is unavailable for stacking in the expected layout')
    return source, destination


def plan_task(start, step, train_groups, *, validation_source=None):
    """Plan a complete symbolic task, preserving active IDs and stack order.

    Counts in an arrangement constrain only their specified slots. They do not
    assert an identity target. The returned scene is one reachable outcome.
    """
    start = scene_recall.scene(start)
    kind = step['kind']
    if kind in ('move', 'stack', 'unstack'):
        _reject_unsupported_target_constraint(step, validation_source, start)
        source, destination = _transfer_slots(start, step, validation_source)
        obj, count = step['object_id'], step['count']
        if count is not None and count > 1 and obj is not None:
            raise UnsupportedStep('a single object ID cannot specify a multiple-object bundle')
        source_stack, destination_stack = start[source - 1], start[destination - 1]
        if obj is not None and (not source_stack or source_stack[-1] != obj):
            raise UnsupportedStep('specified cylinder is not the source top object')
        if count is None:
            if (obj is None and len(source_stack) != 1 and
                    not (len(source_stack) == 2 and _validated_single_top_unstack(step, validation_source))):
                raise UnsupportedStep('single-cylinder target is not unique in the known source')
            count = 1
        if count > len(source_stack) or len(destination_stack) + count > 2:
            raise UnsupportedStep('the requested quantity does not fit the source and destination')
        final_height = len(destination_stack) + count
        if step['height'] is not None and step['height'] != final_height:
            raise UnsupportedStep('destination height differs from the requested height')
        if kind == 'stack' and final_height != 2:
            raise UnsupportedStep('stack requires a two-level destination')
        if kind == 'unstack' and len(source_stack) != 2:
            raise UnsupportedStep('unstack requires a two-level source')
        goal = list(start)
        goal[source - 1] = source_stack[:-count]
        goal[destination - 1] = destination_stack + source_stack[-count:]
        goal = scene_recall.scene(goal, scene_recall.identities(start))
        if step.get('destination_object_id') is not None:
            action = {'source_slot': source, 'destination_slot': destination}
            _, task = scene_features.transfer_task(start, action)
            if scene_policy.split_group_for(start, task) not in train_groups:
                raise StackReferenceUnavailable('the direct stack is outside the allowed transfer groups')
            return goal, [action]
        return goal, train_path(start, goal, train_groups)
    if kind == 'swap':
        source, destination = step['source_slot'] - 1, step['destination_slot'] - 1
        goal = list(start)
        goal[source], goal[destination] = goal[destination], goal[source]
        goal = scene_recall.scene(goal, scene_recall.identities(start))
        return goal, train_path(start, goal, train_groups)
    if kind == 'arrange':
        entries = step['layout']
        if not entries or any(entry['count'] is None or entry['object_ids'] is not None for entry in entries):
            raise UnsupportedStep('only count-only arrangement constraints are implemented')
        counts = {entry['slot'] - 1: entry['count'] for entry in entries}
        active = len(scene_recall.identities(start))
        if sum(counts.values()) > active or sum(counts.values()) + 2 * (4 - len(counts)) < active:
            raise UnsupportedStep('the requested counts cannot preserve all active cylinders')
        path = _train_path_to(start, lambda value: all(len(value[slot]) == n for slot, n in counts.items()),
                              train_groups)
        goal = start
        for action in path:
            goal = task_catalog.transfer(goal, action['source_slot'], action['destination_slot'])
        return goal, path
    raise UnsupportedStep(f'no symbolic task planner for {kind}')


def _check_supported(steps):
    for step in steps:
        if step["op"] == "task":
            kind = step["kind"]
            if kind not in ("register", "remember", "move", "stack", "unstack"):
                raise UnsupportedStep(f"unsupported task: {kind}")
            if kind == "register" and step["layout"] is not None:
                raise UnsupportedStep("register needs the separately supplied explicit registered scene")
            if kind in ("move", "stack", "unstack") and step["count"] not in (None, 1):
                raise UnsupportedStep("only a single cylinder transfer is supported")
        elif step["op"] != "return" or step["target"]["kind"] != "learned_memory":
            raise UnsupportedStep(f"unsupported control: {step['op']}")


def _with_memory_binding(method):
    """Do not let a memory reference be replaced during a session step."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._memory_binding_lock:
            return method(self, *args, **kwargs)
    return guarded


class SceneModelSession:
    """One continuous simulator and one latent memory; never resets the simulator."""

    def __init__(self, sim, memory, student, *, device="cpu", max_decisions=24,
                 cancel=None, transfer_runner=None, checkpoint_train_groups=None):
        self._memory_binding_lock = threading.RLock()
        plan, plan_hash, _ = scene_demo_data.read_split_plan(SPLIT_PLAN)
        self.plan_sha256 = plan_hash
        groups = set(plan["train_groups"])
        if checkpoint_train_groups is not None:
            actual = set(checkpoint_train_groups)
            if not actual or not actual.issubset(groups):
                raise ValueError("student training groups do not match the fixed split plan")
            groups &= actual
        self.train_groups = frozenset(groups)
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("max_decisions must be a positive integer")
        self.sim, self.memory, self.student = sim, memory, student
        self.device, self.max_decisions = device, max_decisions
        self.cancel = cancel if cancel is not None else threading.Event()
        self.transfer_runner = transfer_runner or evaluate_scene_policy.run_transfer
        self.current = None
        self.status = "unregistered"
        self.events = []
        self._requests = set()
        self._request_lock = threading.Lock()
        self._lock = threading.Lock()

    @property
    def memory(self):
        # Reads must not block the separate stop/control path while a transfer
        # owns the binding lock. Replacement is serialized by the setter.
        return self._memory

    @memory.setter
    def memory(self, value):
        with self._memory_binding_lock:
            session_lock = getattr(self, '_lock', None)
            if session_lock is not None and session_lock.locked():
                raise RuntimeError('cannot replace memory while a session is executing')
            self._memory = value

    def _halt(self, reason):
        self.status = "uncertain"
        try:
            self.sim.mark_layout_uncertain(reason)
        except Exception:
            # Even when the simulator cannot accept a marker, this session stays blocked.
            pass

    def _registered(self, expected):
        observed = self.sim.observation()
        if (observed.get("layout_status") != "registered"
                or scene_recall.scene(observed.get("registered_layout")) != expected):
            raise RuntimeError("simulator registered state differs from the session")
        return observed

    def _ready(self):
        if self.status != "registered" or self.current is None:
            raise RuntimeError("session has no certain registered scene")
        self._registered(self.current)
        if self.cancel.is_set():
            raise InterruptedError("request cancelled")

    def _register(self, value):
        if self.status == "uncertain":
            raise RuntimeError("uncertain sessions cannot be recovered automatically")
        if self.cancel.is_set():
            raise InterruptedError("request cancelled")
        registered = scene_recall.scene(value)
        self._registered(registered)
        self.memory.clear()
        self.memory.observe(registered, write=False)
        self.current = registered
        self.status = "registered"
        self.events.append({"event": "registered", "scene": registered, "memory_cleared": True})

    def _move(self, action, *, commit_lock=None, is_current=None, on_transfer_commit=None):
        self._ready()
        before, task = scene_features.transfer_task(self.current, action)
        after = task_catalog.transfer(before, task["source_slot"], task["destination_slot"])
        group = scene_policy.split_group_for(before, task)
        if group not in self.train_groups:
            raise UnsupportedStep(f"transfer is outside allowed train groups: {group}")
        event = {"event": "transfer", "before": before, "planned_after": after,
                 "task": task, "split_group": group, "result": None, "committed": False}
        self.events.append(event)
        result = self.transfer_runner(self.student, self.sim, before, task,
                                      max_decisions=self.max_decisions, device=self.device, cancel=self.cancel)
        event["result"] = result
        if (not isinstance(result, dict) or result.get("success") is not True
                or result.get("physical_success") is not True or result.get("termination") != "model_done"
                or result.get("layout_status") != "registered"
                or type(result.get("teacher_assistance_count")) is not int
                or result.get("teacher_assistance_count") != 0 or result.get("actual_hardware") is not False):
            raise RuntimeError("transfer did not complete with a registered physical success")
        evaluate_scene_policy.validate_assessment(result.get("assessment"), before, task)
        self._registered(after)
        if self.cancel.is_set():
            raise InterruptedError("request cancelled during transfer")
        # Update both only after student success, the assessment and registered state.
        self.memory.observe(after, write=False)
        if self.cancel.is_set():
            raise InterruptedError("request cancelled while updating memory")
        # The coordinator can serialize this final commit with a stop request.
        # It never holds that lock during the student or memory model calls.
        with commit_lock if commit_lock is not None else nullcontext():
            if self.cancel.is_set() or (is_current is not None and not is_current()):
                raise InterruptedError("request invalidated before transfer commit")
            self.current = after
            event["committed"] = True
            # The coordinator must see this commit before another thread can
            # bind a repeat-stop request to a transfer that has already ended.
            if on_transfer_commit is not None:
                on_transfer_commit()

    def _prepare_command_move(self, step, *, validation_source=None):
        """Check one direct transfer using only the already-registered scene.

        This does not call the simulator, a runner or the memory model and
        does not modify the session or its events. The caller checks readiness.
        """
        _reject_unsupported_target_constraint(step, validation_source, self.current)
        source, destination = _transfer_slots(self.current, step, validation_source)
        obj = step['object_id']
        source_stack = self.current[source - 1]
        if (obj is None and step['count'] is None and len(source_stack) != 1
                and not (len(source_stack) == 2 and _validated_single_top_unstack(step, validation_source))):
            raise UnsupportedStep('single-cylinder target is not unique in the known source')
        _, task = scene_features.transfer_task(self.current,
            {"source_slot": source, "destination_slot": destination})
        if obj is not None and task["object_id"] != obj:
            raise UnsupportedStep("specified cylinder is not the source's top object")
        if step["height"] is not None and task["destination_level"] != step["height"]:
            raise UnsupportedStep("destination height differs from the requested height")
        if step["kind"] == "stack" and task["destination_level"] != 2:
            raise UnsupportedStep("stack requires an occupied destination")
        if step["kind"] == "unstack" and task["source_level"] != 2:
            raise UnsupportedStep("unstack requires a two-level source")
        return task

    def _command_move(self, step, *, commit_lock=None, is_current=None, validation_source=None):
        self._ready()
        task = self._prepare_command_move(step, validation_source=validation_source)
        self._move(task, commit_lock=commit_lock, is_current=is_current)

    @_with_memory_binding
    def _execute_validated_step(self, step, *, registration_scene=None, registered_initial=None,
                                task_start=None, commit_lock=None, is_current=None,
                                before_transfer=None, after_transfer=None, on_transfer_commit=None,
                                planned_tasks=False,
                                validation_source=None):
        """Execute a trusted step after its entire originating program was validated.

        The caller owns the session lock. This boundary does not validate a model
        proposal or convert a v2 step into a synthetic v1 utterance.
        """
        if self.status == "uncertain":
            raise RuntimeError("uncertain session cannot execute another step")
        if self.cancel.is_set():
            raise InterruptedError("request cancelled")
        if step["op"] == "task" and step["kind"] == "register":
            self._register(registration_scene)
        elif step["op"] == "task" and step["kind"] == "remember":
            self._ready()
            self.memory.observe(self.current, write=True)
            event = {"event": "remember", "goal_source": "model_hidden_state"}
            version = getattr(self.memory, 'state_version', None)
            identity = getattr(self.memory, 'state_identity', None)
            if (isinstance(version, int) and not isinstance(version, bool)
                    and isinstance(identity, str) and identity):
                # Runtime metadata only: never persist a remembered target.
                event['memory_state_version'] = version
                event['memory_state_identity'] = identity
            self.events.append(event)
        elif step["op"] == "task" and planned_tasks:
            self._ready()
            goal, path = plan_task(self.current, step, self.train_groups, validation_source=validation_source)
            self.events.append({"event": "symbolic_plan", "steps": len(path),
                                "planner": "breadth_first_search_not_learned",
                                "goal_source": "count_constraints" if step['kind'] == 'arrange' else "explicit_task",
                                "planned_goal": goal})
            self._execute_path(path, commit_lock, is_current, before_transfer, after_transfer, on_transfer_commit)
        elif step["op"] == "task":
            self._command_move(step, commit_lock=commit_lock, is_current=is_current,
                               validation_source=validation_source)
        elif step["op"] == "return":
            self._ready()
            kind = step["target"]["kind"]
            if kind == "learned_memory":
                goal = scene_recall.scene(self.memory.recall(), scene_recall.identities(self.current))
                self.events.append({"event": "recall", "decoded_goal": goal,
                                    "goal_source": "decoded_gru_hidden_state"})
            elif kind in ("registered_initial", "task_start"):
                source = registered_initial if kind == "registered_initial" else task_start
                if source is None:
                    raise UnsupportedStep(f"{kind} snapshot is not available")
                goal = scene_recall.scene(source, scene_recall.identities(self.current))
                self.events.append({"event": "restore_target", "target": goal, "goal_source": kind})
            else:
                raise UnsupportedStep(f"unsupported return target: {kind}")
            path = train_path(self.current, goal, self.train_groups)
            self.events.append({"event": "symbolic_plan", "steps": len(path),
                                "planner": "breadth_first_search_not_learned"})
            self._execute_path(path, commit_lock, is_current, before_transfer, after_transfer, on_transfer_commit)
        else:
            raise UnsupportedStep(f"unsupported trusted step: {step['op']}")

    def _execute_path(self, path, commit_lock, is_current, before_transfer, after_transfer, on_transfer_commit=None):
        for action in path:
            if before_transfer is not None and not before_transfer():
                raise RepeatBoundaryReached("repeat stop reached before the next transfer")
            try:
                self._move(action, commit_lock=commit_lock, is_current=is_current,
                           on_transfer_commit=on_transfer_commit)
            finally:
                if after_transfer is not None:
                    after_transfer()

    def execute(self, program, text, *, registration_scene=None):
        """Validate a whole program before side effects and execute each ID at most once."""
        program = copy.deepcopy(program)
        validation_source = _ValidatedTopUnstackSource(program, text, None, v1=True)
        _check_supported(program["steps"])
        has_registration = any(s["op"] == "task" and s["kind"] == "register" for s in program["steps"])
        if (registration_scene is None) == has_registration:
            raise ValueError("an explicit scene is required exactly for registration programs")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a request is already in progress")
        start = len(self.events)
        try:
            request_id = program["request_id"]
            with self._request_lock:
                if request_id in self._requests:
                    raise DuplicateRequest("request ID was already used; movement will not be repeated")
                self._requests.add(request_id)
            # Reserve the ID before the pure preflight, preserving at-most-once
            # behavior even if an unsupported request is rejected. Only a
            # first direct transfer can use this path: no registration,
            # memory operation, execution step or transfer has started yet.
            rejected = None
            try:
                first = program['steps'][0]
                if first['op'] == 'task' and first['kind'] in ('move', 'stack', 'unstack'):
                    # Readiness may call an external simulator, so its errors
                    # are handled by the normal uncertain path, not by the
                    # side-effect-free UnsupportedStep handler below.
                    self._ready()
                    try:
                        self._prepare_command_move(first, validation_source=validation_source)
                    except UnsupportedStep as exc:
                        rejected = exc
                    if self.cancel.is_set():
                        raise InterruptedError("request cancelled during preflight")
                if rejected is None:
                    for index, step in enumerate(program["steps"]):
                        self._execute_validated_step(step, registration_scene=registration_scene,
                                                     validation_source=validation_source)
                        if self.cancel.is_set():
                            raise InterruptedError("request cancelled during step")
                        self.events.append({"event": "step_complete", "step_index": index})
                    return {"request_id": request_id, "status": "completed", "events": copy.deepcopy(self.events[start:])}
            except Exception as exc:
                self._halt(f"{type(exc).__name__}: {exc}")
                self.events.append({"event": "stopped_uncertain", "error_type": type(exc).__name__,
                                    "message": str(exc)})
                return {"request_id": request_id, "status": "uncertain", "events": copy.deepcopy(self.events[start:])}
            except BaseException:
                self._halt("request interrupted")
                raise
            raise rejected
        finally:
            self._lock.release()
