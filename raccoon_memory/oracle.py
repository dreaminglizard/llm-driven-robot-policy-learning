"""Exact comparison/label source. Never called by the learned runtime to repair it."""
from collections import deque
from functools import lru_cache

from .world import DONE, MOVES, N_ACTIONS, all_states, legal, state, step

STATES = all_states()
INDEX = {value: i for i, value in enumerate(STATES)}
EDGES = tuple(tuple((a, INDEX[step(value, a)]) for a, allowed in enumerate(legal(value)[:len(MOVES)]) if allowed) for value in STATES)


@lru_cache(maxsize=840)
def distances(goal):
    goal = state(goal)
    values = [-1] * len(STATES)
    values[INDEX[goal]] = 0
    pending = deque([INDEX[goal]])
    while pending:
        source = pending.popleft()
        for _, destination in EDGES[source]:
            if values[destination] == -1:
                values[destination] = values[source] + 1
                pending.append(destination)
    if -1 in values:
        raise RuntimeError("Task graph is unexpectedly disconnected")
    return tuple(values)


def optimal_actions(current, goal):
    current, goal = state(current), state(goal)
    if current == goal:
        return (DONE,)
    values = distances(goal)
    index = INDEX[current]
    return tuple(action for action, nxt in EDGES[index] if values[nxt] == values[index]-1)


def plan(current, goal):
    current, goal = state(current), state(goal)
    actions = []
    while current != goal:
        action = optimal_actions(current, goal)[0]
        actions.append(action)
        current = step(current, action)
    return actions


def label(current, goal):
    targets = [0.0] * N_ACTIONS
    actions = optimal_actions(current, goal)
    for action in actions:
        targets[action] = 1.0 / len(actions)
    return targets
