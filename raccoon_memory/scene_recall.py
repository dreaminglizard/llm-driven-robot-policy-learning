"""Offline learned recall for one to four registered cylinders.

This module does not modify the older four-cylinder world or call a device.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import permutations, product
import random

import torch
from torch import nn
from torch.nn import functional as F

SLOT_COUNT = 4
MAX_HEIGHT = 2
TOKEN_COUNT = SLOT_COUNT * MAX_HEIGHT
VOCAB_SIZE = 5
INPUT_SIZE = TOKEN_COUNT * VOCAB_SIZE + 1
MIN_HISTORY_LENGTH = 8
SPLIT_VERSION = "canonical_full_history_v1_sha256_80_10_10"
SPLITS = ("train", "validation", "test")


def scene(value, active_ids=None):
    """Return four bottom-to-top stacks after strict identity/height validation."""
    if not isinstance(value, (tuple, list)) or len(value) != SLOT_COUNT:
        raise ValueError("배치는 네 자리여야 합니다.")
    if any(not isinstance(stack, (tuple, list)) for stack in value):
        raise ValueError("각 자리는 아래층부터 나열한 목록이어야 합니다.")
    result = tuple(tuple(stack) for stack in value)
    if any(len(stack) > MAX_HEIGHT for stack in result):
        raise ValueError("각 자리는 최대 2층입니다.")
    objects = [item for stack in result for item in stack]
    if not 1 <= len(objects) <= 4 or any(type(item) is not int or item not in (1, 2, 3, 4) for item in objects):
        raise ValueError("활성 원통은 ID 1~4 중 1~4개여야 합니다.")
    if len(set(objects)) != len(objects):
        raise ValueError("원통 ID가 중복됐습니다.")
    if active_ids is not None and frozenset(objects) != frozenset(active_ids):
        raise ValueError("현재 활성 원통 집합이 기억 당시와 다릅니다.")
    return result


def identities(value):
    return frozenset(item for stack in scene(value) for item in stack)


def tokens(value):
    value = scene(value)
    return tuple(item for stack in value for item in (*stack, *((0,) * (MAX_HEIGHT-len(stack)))))


def from_tokens(values, active_ids=None):
    if not isinstance(values, (tuple, list)) or len(values) != TOKEN_COUNT:
        raise ValueError("아래층·위층 토큰 8개가 필요합니다.")
    if any(type(item) is not int or not 0 <= item < VOCAB_SIZE for item in values):
        raise ValueError("토큰은 0~4 사이 정수여야 합니다.")
    stacks = []
    for index in range(0, TOKEN_COUNT, MAX_HEIGHT):
        lower, upper = values[index:index+2]
        if lower == 0 and upper != 0:
            raise ValueError("원통이 빈 아래층 위에 떠 있습니다.")
        stacks.append(tuple(item for item in (lower, upper) if item))
    return scene(stacks, active_ids)


def features(value, write):
    if type(write) is not bool:
        raise ValueError("쓰기 신호는 불리언이어야 합니다.")
    return [float(item == label) for item in tokens(value) for label in range(VOCAB_SIZE)] + [float(write)]


@lru_cache(maxsize=15)
def all_scenes(active_ids):
    """Enumerate only valid symbolic scenes for a specific active set."""
    if not isinstance(active_ids, tuple) or tuple(sorted(set(active_ids))) != active_ids:
        raise ValueError("활성 ID는 중복 없이 정렬된 튜플이어야 합니다.")
    if not 1 <= len(active_ids) <= 4 or any(type(obj) is not int or obj not in (1, 2, 3, 4) for obj in active_ids):
        raise ValueError("활성 ID는 1~4개여야 합니다.")
    result = []
    for lengths in product(range(MAX_HEIGHT+1), repeat=SLOT_COUNT):
        if sum(lengths) != len(active_ids):
            continue
        for order in permutations(active_ids):
            stacks, cursor = [], 0
            for length in lengths:
                stacks.append(order[cursor:cursor+length])
                cursor += length
            result.append(tuple(stacks))
    return tuple(sorted(result))


@dataclass(frozen=True)
class History:
    """A labeled synthetic sequence, used for training and evaluation only."""
    scenes: tuple
    writes: tuple

    def __post_init__(self):
        if not isinstance(self.scenes, tuple) or not isinstance(self.writes, tuple) or len(self.scenes) != len(self.writes) or not self.scenes:
            raise ValueError("같은 길이의 장면과 쓰기 이력이 필요합니다.")
        if any(type(flag) is not bool for flag in self.writes) or not self.writes[0]:
            raise ValueError("첫 관측에서 쓰고 쓰기 신호는 불리언이어야 합니다.")
        active = identities(self.scenes[0])
        for observed in self.scenes:
            scene(observed, active)

    @property
    def target(self):
        # This label is never retained by SceneRecall or used to repair model output.
        return self.targets_by_step[-1]

    @property
    def targets_by_step(self):
        """Latest written scene at each step; never labels a past with a future write."""
        target = None
        result = []
        for value, flag in zip(self.scenes, self.writes):
            if flag:
                target = value
            result.append(target)
        return tuple(result)

    @property
    def active_ids(self):
        return identities(self.scenes[0])


def canonical_history(history):
    """Canonicalize the full episode under slot and identity renamings.

    Renaming identities by their first occurrence for each slot permutation is
    equivalent to considering every identity permutation, with less work.
    """
    candidates = []
    for slots in permutations(range(SLOT_COUNT)):
        mapping = {}
        encoded = bytearray()
        for observed, write in zip(history.scenes, history.writes):
            encoded.append(int(write))
            for slot in slots:
                for item in (*observed[slot], *((0,) * (MAX_HEIGHT-len(observed[slot])))):
                    if item and item not in mapping:
                        mapping[item] = len(mapping) + 1
                    encoded.append(mapping.get(item, 0))
        candidates.append(bytes(encoded))
    return min(candidates)


def history_split(history):
    digest = hashlib.sha256(canonical_history(history)).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _random_active(rng, count):
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError("원통 수는 1~4여야 합니다.")
    return tuple(sorted(rng.sample((1, 2, 3, 4), count)))


def sample_history(rng, split, count, length=8, *, overwrite=None, present_mode="any", max_attempts=10000):
    """Generate only the requested partition; other candidate labels are discarded."""
    if split not in SPLITS or type(length) is not int or length < MIN_HISTORY_LENGTH:
        raise ValueError("분할과 8 이상의 이력 길이가 필요합니다.")
    if overwrite is not None and type(overwrite) is not bool:
        raise ValueError("덮어쓰기 설정이 잘못됐습니다.")
    if present_mode not in ("any", "distinct", "target"):
        raise ValueError("현재 배치의 생성 조건이 잘못됐습니다.")
    for _ in range(max_attempts):
        active = _random_active(rng, count)
        states = all_scenes(active)
        first = rng.choice(states)
        do_overwrite = rng.random() < 0.5 if overwrite is None else overwrite
        second = rng.choice([item for item in states if item != first]) if do_overwrite else first
        if present_mode == "distinct":
            present = rng.choice([item for item in states if item != first and item != second])
        elif present_mode == "target":
            present = second
        else:
            present = rng.choice(states)
        values = [first] + [rng.choice(states) for _ in range(length-2)] + [present]
        writes = [True] + [False] * (length-1)
        if do_overwrite:
            position = rng.randrange(1, length-1)
            values[position], writes[position] = second, True
        history = History(tuple(values), tuple(writes))
        if history_split(history) == split:
            return history
    raise RuntimeError("지정된 분할에서 이력을 생성하지 못했습니다.")


def sample_pair(rng, split, count, length=8, *, reverse_layers=False, max_attempts=100000):
    """Two different remembered scenes followed by the same observations."""
    if split not in SPLITS or type(length) is not int or length < MIN_HISTORY_LENGTH:
        raise ValueError("분할과 8 이상의 이력 길이가 필요합니다.")
    if reverse_layers and count < 2:
        raise ValueError("층 순서 비교에는 원통이 2개 이상 필요합니다.")
    for _ in range(max_attempts):
        active = _random_active(rng, count)
        states = all_scenes(active)
        if reverse_layers:
            first = rng.choice([item for item in states if any(len(stack) == 2 for stack in item)])
            slot = rng.choice([i for i, stack in enumerate(first) if len(stack) == 2])
            changed = list(first)
            changed[slot] = tuple(reversed(changed[slot]))
            second = tuple(changed)
        else:
            first, second = rng.sample(states, 2)
        present = rng.choice([item for item in states if item != first and item != second])
        suffix = tuple([rng.choice(states) for _ in range(length-2)] + [present])
        writes = (True,) + (False,) * (length-1)
        left, right = History((first,) + suffix, writes), History((second,) + suffix, writes)
        if history_split(left) == split and history_split(right) == split:
            return left, right
    raise RuntimeError("같은 현재 배치의 비교 이력을 생성하지 못했습니다.")


class SceneMemory(nn.Module):
    """A GRU hidden state is updated at each observation and reset on writes."""
    def __init__(self, hidden_size=128):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("은닉 차원은 양의 정수여야 합니다.")
        self.hidden_size = hidden_size
        self.encoder = nn.GRUCell(INPUT_SIZE, hidden_size)
        self.decoder = nn.Linear(hidden_size, TOKEN_COUNT * VOCAB_SIZE)

    def advance(self, observed, hidden=None):
        if hidden is None:
            hidden = observed.new_zeros((observed.shape[0], self.hidden_size))
        # The explicit overwrite operation discards the previous episode.
        previous = torch.where(observed[:, -1:].bool(), torch.zeros_like(hidden), hidden)
        return self.encoder(observed, previous)

    def decode(self, hidden):
        return self.decoder(hidden).reshape(-1, TOKEN_COUNT, VOCAB_SIZE)

    def forward_steps(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != INPUT_SIZE or sequence.shape[1] == 0:
            raise ValueError("입력은 배치 × 시간 × 41이어야 합니다.")
        hidden = sequence.new_zeros((sequence.shape[0], self.hidden_size))
        hidden_steps = []
        for observed in sequence.unbind(dim=1):
            hidden = self.advance(observed, hidden)
            hidden_steps.append(hidden)
        hidden_steps = torch.stack(hidden_steps, dim=1)
        logits = self.decoder(hidden_steps).reshape(sequence.shape[0], sequence.shape[1], TOKEN_COUNT, VOCAB_SIZE)
        return logits, hidden

    def forward(self, sequence):
        logits, hidden = self.forward_steps(sequence)
        return logits[:, -1], hidden


def load_scene_memory(path, device="cpu"):
    """Load only this model's explicitly versioned, weights-only checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "variable_scene_memory_v1":
        raise ValueError("지원하지 않는 장면 기억 모델입니다.")
    model = SceneMemory(payload.get("hidden_size")).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


class SceneRecall:
    """Runtime state contains a latent vector and the ID set, not a scene label."""
    def __init__(self, model):
        import secrets
        import threading
        self.model = model.eval()
        # This generation is process-local and contains no remembered scene.
        # It lets a coordinator detect external clear/observe calls between
        # interpretation and use, including read-only scene observations that
        # advance the GRU hidden state.
        self._state_lock = threading.RLock()
        self._state_identity = secrets.token_urlsafe(18)
        self._state_version = 0
        self._hidden = None
        self._active_ids = None
        self.clear()

    @property
    def state_version(self):
        return self._state_version

    @property
    def state_identity(self):
        return self._state_identity

    @property
    def has_memory(self):
        return self._hidden is not None

    @property
    def hidden(self):
        # A clone prevents in-place edits from silently changing the live GRU
        # state without advancing the revision.
        with self._state_lock:
            return None if self._hidden is None else self._hidden.clone()

    @hidden.setter
    def hidden(self, value):
        with self._state_lock:
            self._hidden = None if value is None else value.detach().clone()
            self._state_version += 1

    @property
    def active_ids(self):
        return self._active_ids

    @active_ids.setter
    def active_ids(self, value):
        with self._state_lock:
            self._active_ids = None if value is None else tuple(value)
            self._state_version += 1

    def synchronized(self):
        """Serialize a coordinator step with external public memory updates."""
        return self._state_lock

    def clear(self):
        with self._state_lock:
            self._hidden = None
            self._active_ids = None
            self._state_version += 1

    @torch.inference_mode()
    def observe(self, current, *, write=False):
        with self._state_lock:
            if type(write) is not bool:
                raise ValueError("쓰기 신호는 불리언이어야 합니다.")
            current = scene(current, self._active_ids)
            ids = identities(current)
            if write or self._hidden is not None:
                device = next(self.model.parameters()).device
                observed = torch.tensor([features(current, write)], dtype=torch.float32, device=device)
                hidden = self.model.advance(observed, self._hidden)
                self._hidden = hidden.detach()
            self._active_ids = ids
            self._state_version += 1

    @torch.inference_mode()
    def recall(self):
        with self._state_lock:
            if self._hidden is None:
                raise ValueError("먼저 기억할 배치를 지정해 주세요.")
            logits = self.model.decode(self._hidden)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("기억 모델의 출력이 유한하지 않습니다.")
            predicted = logits[0].argmax(-1).tolist()
            try:
                return from_tokens(predicted, self._active_ids)
            except ValueError as exc:
                raise ValueError("기억 모델이 유효하지 않은 배치를 출력했습니다.") from exc


def histories_tensor(histories, device="cpu"):
    if not histories or len({len(item.scenes) for item in histories}) != 1:
        raise ValueError("길이가 같은 이력이 필요합니다.")
    inputs = torch.tensor([[features(value, flag) for value, flag in zip(item.scenes, item.writes)] for item in histories], dtype=torch.float32, device=device)
    labels = torch.tensor([tokens(item.target) for item in histories], dtype=torch.long, device=device)
    return inputs, labels


def loss_for_histories(model, histories, device="cpu"):
    inputs, _ = histories_tensor(histories, device)
    targets = torch.tensor([[tokens(target) for target in item.targets_by_step] for item in histories],
                           dtype=torch.long, device=device)
    logits, _ = model.forward_steps(inputs)
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
