"""Local inference. Learned ranking plus explicit bounded symbolic lookahead."""
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
from pathlib import Path
import threading

import torch

from .neural import load_model
from .scope import enabled_slots as validate_slots
from .world import DONE, MOVES, all_states, features, from_tokens, legal, state, step


@dataclass(frozen=True)
class Plan:
    actions: tuple
    method: str
    expanded_states: int
    direct_failure: str | None = None


class LocalPlanner:
    def __init__(self, model_dir, max_expansions=840, max_height=4, enabled_slots=(1,2,3,4)):
        if type(max_height) is not int or not 1 <= max_height <= 4:
            raise ValueError("허용 층수는 1~4 사이의 정수여야 합니다.")
        torch.set_num_threads(2)
        self.policy = load_model(Path(model_dir)/"policy.pt")
        self.max_expansions = max_expansions
        self.max_height = max_height
        self.enabled_slots = validate_slots(enabled_slots)
        self.states = tuple(s for s in all_states() if max(map(len,s)) <= max_height)
        self.index = {value:i for i,value in enumerate(self.states)}
        self.encoded = torch.tensor([features(value) for value in self.states])
        self.masks = torch.tensor([self.allowed(value) for value in self.states], dtype=torch.bool)

    def allowed(self,value):
        mask = legal(value)
        for action,(source,destination) in enumerate(MOVES):
            mask[action] = (mask[action] and len(value[destination]) < self.max_height
                            and source+1 in self.enabled_slots and destination+1 in self.enabled_slots)
        return mask

    @lru_cache(maxsize=128)
    @torch.inference_mode()
    def scores(self, goal):
        goal = state(goal)
        if goal not in self.index:
            raise ValueError("목표 배치가 허용된 최대 층수를 넘습니다.")
        goal_features = self.encoded[self.index[goal]][None].expand(len(self.states),-1)
        logits, values = self.policy.predict(self.encoded, goal_features)
        logits = logits.masked_fill(~self.masks, -1e9)
        return torch.log_softmax(logits, -1).numpy(), values.numpy()

    def plan(self, start, goal, *, search=True, cancel=None):
        start, goal = state(start), state(goal)
        if start not in self.index or goal not in self.index:
            raise ValueError("현재 또는 목표 배치가 허용된 최대 층수를 넘습니다.")
        if any(start[i]!=goal[i] for i in range(4) if i+1 not in self.enabled_slots):
            raise ValueError("현재 사용하지 않는 자리의 배치를 바꾸는 요청입니다.")
        cancel = cancel or threading.Event()
        probabilities, values = self.scores(goal)
        current, seen, actions = start, {start}, []
        failure = None
        for _ in range(20):
            if cancel.is_set():
                raise InterruptedError("작업을 정지했습니다.")
            if current == goal:
                return Plan(tuple(actions), "direct_learned_policy", 0)
            action = int(probabilities[self.index[current]].argmax())
            if action >= DONE:
                failure = "premature_terminal"
                break
            current = step(current, action)
            actions.append(action)
            if current in seen:
                failure = "cycle"
                break
            seen.add(current)
        failure = failure or "step_limit"
        if not search:
            raise ValueError(f"학습 모델의 직접 계획 실패: {failure}")
        # This is reported as search assistance. No oracle labels/paths are read.
        frontier = [(float(values[self.index[start]]), 0, 0.0, start)]
        cost = {start:0.0}
        parents = {}
        closed = set()
        serial = 0
        while frontier and len(closed) < self.max_expansions:
            if cancel.is_set():
                raise InterruptedError("작업을 정지했습니다.")
            _, _, distance, current = heapq.heappop(frontier)
            if current in closed or distance != cost[current]:
                continue
            closed.add(current)
            if current == goal:
                actions = []
                while current != start:
                    current, action = parents[current]
                    actions.append(action)
                return Plan(tuple(reversed(actions)), "learned_policy_with_search", len(closed), failure)
            row = probabilities[self.index[current]]
            choices = sorted((a for a, ok in enumerate(self.allowed(current)[:DONE]) if ok), key=lambda a:-row[a])
            for action in choices:
                nxt = step(current, action)
                candidate = distance + 1.0 + 0.15*min(30.0, max(0.0, -float(row[action])))
                if nxt not in closed and candidate < cost.get(nxt, math.inf):
                    cost[nxt] = candidate
                    parents[nxt] = (current,action)
                    serial += 1
                    priority = candidate + 0.5*float(values[self.index[nxt]])
                    heapq.heappush(frontier, (priority,serial,candidate,nxt))
        raise ValueError("탐색 한도 안에서 실행 가능한 계획을 찾지 못했습니다.")


class LearnedRecall:
    """The saved arrangement is represented only by the GRU hidden state."""
    def __init__(self, model_dir):
        self.model = load_model(Path(model_dir)/"memory.pt")
        self.clear()

    def clear(self):
        self.hidden = None
        self.saved = False
        self.logits = None
        self.steps_since_write = 0

    @torch.inference_mode()
    def observe(self, current, write=False):
        if write:
            # A new explicit "remember" replaces the one saved episode.
            # Clearing the old hidden state avoids mixing two distinct targets.
            self.hidden = None
        sequence = torch.tensor([features(current)+[float(write)]])[None]
        self.logits, self.hidden = self.model(sequence, self.hidden)
        if write:
            self.saved = True
            self.steps_since_write = 0
        elif self.saved:
            self.steps_since_write += 1

    def recall(self):
        if not self.saved:
            raise ValueError("먼저 ‘지금 배치 기억해’라고 말해 주세요.")
        if self.steps_since_write > 47:
            raise ValueError("검증한 기억 길이 48단계를 넘었습니다. 새로 기억할 배치를 지정해 주세요.")
        predicted = self.logits[0,-1].argmax(-1).tolist()
        try:
            result = from_tokens(predicted)
        except ValueError as exc:
            raise ValueError("기억 모델의 출력이 올바른 4개 원통 배치가 아닙니다. 실행을 중단했습니다.") from exc
        return result
