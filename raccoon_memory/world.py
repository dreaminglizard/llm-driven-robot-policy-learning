"""Finite task world; no device I/O and no learning dependencies.

Stacks are bottom to top. IDs identify cylinders, not slots or sensed colors.
"""
from itertools import permutations, product

SLOTS = ("A", "B", "C", "D")
MAX_HEIGHT = 4
N_TOKENS = 4 * MAX_HEIGHT
N_FEATURES = N_TOKENS * 5
MOVES = tuple((src, dst) for src in range(4) for dst in range(4) if src != dst)
DONE = len(MOVES)
STOP = DONE + 1
N_ACTIONS = STOP + 1
INITIAL = ((1,), (2,), (3,), (4,))


def state(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("A state must contain exactly four stacks")
    stacks = tuple(tuple(items) for items in value)
    if any(len(items) > MAX_HEIGHT for items in stacks):
        raise ValueError("A stack cannot contain more than four cylinders")
    flat = [obj for items in stacks for obj in items]
    if any(type(obj) is not int for obj in flat) or sorted(flat) != [1, 2, 3, 4]:
        raise ValueError("Each cylinder ID must occur exactly once")
    return stacks


def key(value):
    return "/".join("".join(str(obj) for obj in stack) or "-" for stack in state(value))


def tokens(value):
    return tuple(obj for stack in state(value) for obj in (*stack, *(0 for _ in range(MAX_HEIGHT-len(stack)))))


def from_tokens(values):
    if len(values) != N_TOKENS:
        raise ValueError("Sixteen layer tokens are required")
    stacks = []
    for i in range(0, N_TOKENS, MAX_HEIGHT):
        items = tuple(int(item) for item in values[i:i+MAX_HEIGHT])
        if any(items[j] != 0 and items[j-1] == 0 for j in range(1, MAX_HEIGHT)):
            raise ValueError("A cylinder cannot float over an empty lower layer")
        stacks.append(tuple(obj for obj in items if obj != 0))
    return state(stacks)


def features(value):
    return [float(token == label) for token in tokens(value) for label in range(5)]


def legal(value, include_terminal=True):
    value = state(value)
    mask = [bool(value[src]) and len(value[dst]) < MAX_HEIGHT for src, dst in MOVES]
    # These are syntax/safety masks only. DONE is not masked using the goal.
    return mask + [bool(include_terminal), bool(include_terminal)]


def step(value, action):
    value = state(value)
    if type(action) is not int or not 0 <= action < len(MOVES):
        raise ValueError("Only an explicit cylinder move changes world state")
    if not legal(value)[action]:
        raise ValueError("An empty source or full destination is not a legal move")
    src, dst = MOVES[action]
    changed = [list(items) for items in value]
    obj = changed[src].pop()
    changed[dst].append(obj)
    return state(changed)


def describe(action, value=None):
    if action == DONE:
        return {"kind": "done"}
    if action == STOP:
        return {"kind": "stop"}
    if type(action) is not int or not 0 <= action < len(MOVES):
        raise ValueError("Unknown action")
    src, dst = MOVES[action]
    result = {"kind": "transfer", "from": SLOTS[src], "to": SLOTS[dst]}
    if value is not None:
        value = state(value)
        if not value[src]:
            raise ValueError("Source is empty")
        obj = value[src][-1]
        result.update(object_id=obj, source_level=len(value[src]), destination_level=len(value[dst])+1)
    return result


def action_id(src, dst):
    return MOVES.index((SLOTS.index(src), SLOTS.index(dst)))


def all_states():
    result = []
    for lengths in product(range(MAX_HEIGHT+1), repeat=4):
        if sum(lengths) != 4:
            continue
        for order in permutations(range(1, 5)):
            cursor, stacks = 0, []
            for length in lengths:
                stacks.append(order[cursor:cursor+length])
                cursor += length
            result.append(tuple(stacks))
    return tuple(sorted(result))


def relabel(value, object_permutation, slot_permutation):
    """Documented symbolic augmentation, never presented as a new real trial."""
    result = [()] * 4
    for src, items in enumerate(state(value)):
        result[slot_permutation[src]] = tuple(object_permutation[obj-1] for obj in items)
    return state(result)
