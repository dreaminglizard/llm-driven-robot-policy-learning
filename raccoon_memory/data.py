"""Split before augmentation. Action supervision is from explicit GPT traces only."""
from collections import defaultdict
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random

import numpy as np

from .teacher import validated_cases
from .world import DONE, STOP, MOVES, N_ACTIONS, all_states, key, legal, relabel, tokens
from .learning_io import write_new_json

SEED = 20261026


def validate_split(split):
    states=all_states()
    if type(split) is not dict or type(split.get('seed')) is not int:
        raise ValueError('분리 자료와 정수 seed가 필요합니다.')
    if split.get('unit')!='complete_goal_configuration_before_augmentation':
        raise ValueError('학습 분리 단위가 목표 배치와 다릅니다.')
    if split.get('state_keys')!=[key(value) for value in states]:
        raise ValueError('분리 자료의 상태 순서가 현재 표현과 다릅니다.')
    combined=[]
    for name in ('train','validation','test'):
        values=split.get(name)
        if type(values) is not list or not values or any(type(i) is not int or not 0<=i<len(states) for i in values):
            raise ValueError(f'{name}: 유효한 상태 번호 목록이 필요합니다.')
        combined.extend(values)
    if sorted(combined)!=list(range(len(states))):
        raise ValueError('학습·검증·시험 분리가 겹치거나 상태가 누락됐습니다.')
    return states


def validate_arrays(split,pairs,targets,remaining):
    states=validate_split(split)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)==0 or not np.issubdtype(pairs.dtype,np.integer):
        raise ValueError('현재·목표 번호 배열은 비어 있지 않은 정수 Nx2여야 합니다.')
    if np.any(pairs<0) or np.any(pairs>=len(states)) or len(np.unique(pairs,axis=0))!=len(pairs):
        raise ValueError('상태 번호가 범위 밖이거나 현재·목표 쌍이 중복됐습니다.')
    if not np.isin(pairs[:,1],split['train']).all():
        raise ValueError('학습 배열에 검증 또는 시험 목표가 포함됐습니다.')
    if targets.shape!=(len(pairs),N_ACTIONS) or not np.isfinite(targets).all() or np.any(targets<0):
        raise ValueError('행동 정답의 크기·값이 유효하지 않습니다.')
    if not np.allclose(targets.sum(axis=1),1,rtol=0,atol=1e-6):
        raise ValueError('행동 정답 확률의 합은 1이어야 합니다.')
    allowed=np.asarray([legal(states[int(i)]) for i in pairs[:,0]],dtype=bool)
    if np.any(targets[~allowed]>0) or np.any(targets[:,STOP]>0):
        raise ValueError('GPT 이동 자료에 불법 행동 또는 출처 없는 정지 정답이 있습니다.')
    terminal=pairs[:,0]==pairs[:,1]
    if np.any(targets[~terminal,DONE]>0) or np.any(targets[terminal,DONE]!=1):
        raise ValueError('종료 정답은 현재와 목표가 같은 배치에만 허용됩니다.')
    if remaining.shape!=(len(pairs),) or not np.isfinite(remaining).all() or np.any(remaining<0):
        raise ValueError('남은 이동 수의 크기·값이 유효하지 않습니다.')
    if np.any(remaining!=np.floor(remaining)) or np.any(remaining[terminal]!=0) or np.any(remaining[~terminal]<1):
        raise ValueError('남은 이동 수와 종료 배치가 일치하지 않습니다.')
    return {'pairs':len(pairs),'train_goals':len(set(map(int,pairs[:,1]))),
            'terminal_pairs':int(terminal.sum()),'goal_split_leakage':False,'illegal_labels':0}


def make_split(seed=SEED):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('seed는 0 이상 2**32 미만의 정수여야 합니다.')
    states = all_states()
    shuffled = list(range(len(states)))
    random.Random(seed).shuffle(shuffled)
    # Both same-present/different-past demonstration goals remain held out.
    reserved = [states.index(value) for value in (((1,2), (), (3,4), ()), ((1,4), (), (3,2), ()))]
    shuffled = [i for i in shuffled if i not in reserved] + reserved
    train_end, valid_end = int(len(states)*0.8), int(len(states)*0.9)
    return {"seed": seed, "unit": "complete_goal_configuration_before_augmentation",
            "train": shuffled[:train_end], "validation": shuffled[train_end:valid_end],
            "test": shuffled[valid_end:], "state_keys": [key(value) for value in states]}


def teacher_dataset(split):
    states = validate_split(split)
    index = {value: i for i, value in enumerate(states)}
    train_goals = set(split["train"])
    counts = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    remaining = {}
    augmented_transitions = 0
    completed_episodes = 0
    sources = defaultdict(int)
    cases = validated_cases()
    for case in cases:
        for objects in permutations(range(1,5)):
            for slots in permutations(range(4)):
                goal = relabel(case["goal"], objects, slots)
                gi = index[goal]
                if gi not in train_goals:
                    continue
                completed_episodes += 1
                for position, item in enumerate(case["steps"]):
                    current = relabel(item["current"], objects, slots)
                    src, dst = MOVES[item["action"]]
                    action = MOVES.index((slots[src], slots[dst]))
                    pair = (index[current], gi)
                    counts[pair][action] += 1
                    remaining[pair] = min(remaining.get(pair, 1e9), len(case["steps"])-position)
                    augmented_transitions += 1
                    sources[case["name"]] += 1
                counts[(gi,gi)][DONE] += 1
                remaining[(gi,gi)] = 0
    pairs = sorted(counts)
    labels = np.stack([counts[pair]/counts[pair].sum() for pair in pairs])
    values = np.array([remaining[pair] for pair in pairs], dtype=np.float32)
    pair_array = np.array(pairs, dtype=np.int64)
    digest = hashlib.sha256(pair_array.tobytes()+labels.tobytes()+values.tobytes()).hexdigest()
    return pair_array, labels, values, {
        "source": "gpt_authored_symbolic_traces_and_identity_slot_permutations",
        "base_gpt_episodes": len(cases), "base_gpt_moves": sum(len(c["steps"]) for c in cases),
        "augmented_training_episodes": completed_episodes,
        "augmented_training_transitions": augmented_transitions,
        "unique_state_goal_training_pairs": len(pairs),
        "rule_planner_training_labels": 0,
        "verified_physical_success_training_labels": 0,
        "source_counts": dict(sources), "dataset_sha256": digest,
        "note": "Programmatic permutations are augmentation, not new GPT calls or physical trials.",
    }


def write_data(output, split):
    output = Path(output)
    names=('split.json','teacher_examples.json','data_provenance.json','training_pairs.npz')
    if any((output/name).exists() for name in names):
        raise FileExistsError('기존 학습 자료를 덮어쓰지 않습니다. 빈 학습 실행 폴더를 사용해 주세요.')
    output.mkdir(parents=True, exist_ok=True)
    pairs, labels, values, provenance = teacher_dataset(split)
    validate_arrays(split,pairs,labels,values)
    for name, value in (("split.json", split), ("teacher_examples.json", validated_cases()), ("data_provenance.json", provenance)):
        write_new_json(output/name,value)
    with (output/'training_pairs.npz').open('xb') as stream:
        np.savez_compressed(stream,pairs=pairs,targets=labels,remaining=values)
    return pairs, labels, values, provenance
