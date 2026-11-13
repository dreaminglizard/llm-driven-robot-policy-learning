"""Train locally, without connecting to the robot or calling an external model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import SEED, make_split, write_data
from .neural import ActionPolicy, EpisodicMemory, RelationalActionPolicy, parameter_count, load_model
from .learning_io import file_sha256,reserve_training_directory,write_new_json
from .world import DONE, INITIAL, MOVES, N_ACTIONS, N_FEATURES, N_TOKENS, STOP, all_states, features, legal, step, tokens


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def tensors(device):
    states = all_states()
    encoded = torch.tensor([features(value) for value in states], device=device)
    allowed = torch.tensor([legal(value) for value in states], dtype=torch.bool, device=device)
    labels = torch.tensor([tokens(value) for value in states], dtype=torch.long, device=device)
    return states, encoded, allowed, labels


def memory_batch(encoded, token_labels, goal_ids, rng, batch_size, length, device):
    count = len(encoded)
    current = rng.integers(0, count, (batch_size,length))
    selected = rng.choice(goal_ids, batch_size)
    first = rng.integers(0, 4, batch_size)
    second = rng.choice(goal_ids, batch_size)
    rewrite = rng.random(batch_size) < 0.35
    rewrite_at = rng.integers(length//2, length-2, batch_size)
    target = np.zeros((batch_size,length), dtype=np.int64)
    writes = np.zeros((batch_size,length,1), dtype=np.float32)
    mask = np.zeros((batch_size,length), dtype=np.float32)
    for b in range(batch_size):
        current[b,first[b]] = selected[b]
        target[b,:] = selected[b]
        writes[b,first[b],0] = 1
        mask[b,first[b]:] = 1
        if rewrite[b]:
            current[b,rewrite_at[b]] = second[b]
            writes[b,rewrite_at[b],0] = 1
            target[b,rewrite_at[b]:] = second[b]
    current = torch.as_tensor(current, device=device)
    sequence = torch.cat((encoded[current], torch.as_tensor(writes, device=device)), dim=-1)
    targets = token_labels[torch.as_tensor(target, device=device)]
    return sequence, targets, torch.as_tensor(mask, device=device)


@torch.inference_mode()
def evaluate_memory(model, encoded, token_labels, goal_ids, seed=601, length=48, repeats=8):
    device = encoded.device
    rng = np.random.default_rng(seed)
    goals = np.repeat(goal_ids, repeats)
    currents = rng.integers(0, len(encoded), (len(goals),length))
    currents[:,2] = goals
    currents[:,-1] = all_states().index(INITIAL)
    writes = torch.zeros((len(goals),length,1), device=device)
    writes[:,2,0] = 1
    inputs = torch.cat((encoded[torch.tensor(currents, device=device)], writes), dim=-1)
    outputs = []
    model.eval()
    for batch in inputs.split(128):
        logits, _ = model(batch)
        outputs.append(logits[:,-1].argmax(-1))
    predicted = torch.cat(outputs)
    expected = token_labels[torch.tensor(goals, device=device)]
    return {"histories": len(goals), "steps_per_history": length,
            "exact_recall": float((predicted==expected).all(-1).float().mean()),
            "token_accuracy": float((predicted==expected).float().mean())}


@torch.inference_mode()
def direct_rollouts(model, states, encoded, allowed, goals, seed=730, count=2000, max_moves=20):
    model.eval()
    device = encoded.device
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(states), count)
    goal_ids = rng.choice(goals, count)
    current = starts.copy()
    finished = current == goal_ids
    failed = np.zeros(count, dtype=bool)
    moves = np.zeros(count, dtype=np.int32)
    index = {value:i for i,value in enumerate(states)}
    transitions = np.full((len(states),N_ACTIONS), -1, dtype=np.int64)
    for si, value in enumerate(states):
        for action, ok in enumerate(legal(value)[:len(MOVES)]):
            if ok:
                transitions[si,action] = index[step(value,action)]
    raw_illegal, selected = 0, 0
    visited = [{int(value)} for value in current]
    for _ in range(max_moves):
        active = np.flatnonzero(~finished & ~failed)
        if len(active) == 0:
            break
        ci = torch.as_tensor(current[active], device=device)
        gi = torch.as_tensor(goal_ids[active], device=device)
        raw = model(encoded[ci], encoded[gi])
        raw_choices = raw.argmax(-1)
        raw_illegal += int((~allowed[ci].gather(1,raw_choices[:,None]).squeeze(1)).sum())
        selected += len(active)
        choices = raw.masked_fill(~allowed[ci],-1e9).argmax(-1).cpu().numpy()
        for row, action in zip(active,choices):
            if action >= DONE:
                failed[row] = True
                continue
            nxt = int(transitions[current[row],action])
            moves[row] += 1
            current[row] = nxt
            finished[row] = nxt == goal_ids[row]
            if not finished[row] and nxt in visited[row]:
                failed[row] = True
            visited[row].add(nxt)
    return {"episodes": count, "successes": int(finished.sum()), "success_rate": float(finished.mean()),
            "raw_illegal_proposals": raw_illegal, "decisions": selected,
            "mean_moves_for_success": float(moves[finished].mean()) if finished.any() else None,
            "max_moves": max_moves, "search_used": False}


def save_model(path, model, kind, config, split):
    torch.save({"kind": kind, "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "config": config, "schema": "numbered_slots_v1", "seed": split["seed"]}, path)


def selection_score(metrics):
    return (metrics['success_rate'],
            -metrics['raw_illegal_proposals']/max(1,metrics['decisions']),
            -(metrics['mean_moves_for_success'] or 0))


def train(args):
    for name in ('policy_epochs','memory_steps','hidden_size','width'):
        value=getattr(args,name)
        if type(value) is not int or value<1:
            raise ValueError(f'{name}은 1 이상의 정수여야 합니다.')
    split = make_split(args.seed)
    policy_lr=getattr(args,'policy_lr',.0015)
    illegal_weight=getattr(args,'illegal_proposal_weight',0.0)
    if not 0<policy_lr<=.01 or not 0<=illegal_weight<=1:
        raise ValueError('학습률 또는 불법 제안 손실 가중치가 범위를 벗어났습니다.')
    initial_policy=getattr(args,'initial_policy',None)
    memory_source=getattr(args,'memory_source',None)
    for source in (initial_policy,memory_source):
        if source and not Path(source).is_file():raise FileNotFoundError(source)
    output = reserve_training_directory(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    pairs, target, remaining, provenance = write_data(output, split)
    states, encoded, allowed, token_labels = tensors(device)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
              "data": provenance, "policy_epochs_requested": args.policy_epochs,
              "memory_steps_requested": args.memory_steps, "training_labels_from_oracle": False,
              "policy_log": [], "memory_log": [], "test_set_used_for_selection": False}
    report.update(policy_lr=policy_lr,illegal_proposal_weight=illegal_weight,
                  source_sha256={str(p.name):file_sha256(p) for p in
                      (Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('teacher.py'),
                       Path(__file__).with_name('world.py'),Path(__file__).with_name('neural.py'))},
                  split_sha256=file_sha256(output/'split.json'))
    emit({"phase": "data_ready", **provenance})
    policy_class = RelationalActionPolicy if args.architecture == "relational" else ActionPolicy
    policy_kind = "relational_policy" if args.architecture == "relational" else "action_policy"
    report["architecture"] = args.architecture
    policy = (load_model(initial_policy) if initial_policy else policy_class(args.width)).to(device)
    if not isinstance(policy,policy_class) or policy.width!=args.width:
        raise ValueError('초기 정책의 구조·너비가 요청한 후보와 다릅니다.')
    optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_lr, weight_decay=0.0001)
    pair_t = torch.as_tensor(pairs, device=device)
    targets = torch.as_tensor(target, device=device)
    values = torch.as_tensor(remaining, device=device)
    report['policy_selection_rule']='validation success, then raw illegal proposal rate, then successful path length'
    best_score = (-1.0,-1e9,-1e9)
    if initial_policy:
        baseline=direct_rollouts(policy,states,encoded,allowed,split['validation'],count=2000)
        best_score=selection_score(baseline)
        shutil.copyfile(initial_policy,output/'policy.pt')
        report.update(initial_policy={'path':str(Path(initial_policy).resolve()),'sha256':file_sha256(initial_policy)},
                      policy_selected_epoch=0,policy_validation=baseline)
        emit({'phase':'initial_policy_validation',**baseline})
    began = time.monotonic()
    for epoch in range(1,args.policy_epochs+1):
        policy.train()
        order = torch.randperm(len(pairs), device=device)
        total_loss = 0.0
        for batch in order.split(1024):
            current_ids, goal_ids = pair_t[batch].unbind(1)
            logits, prediction = policy.predict(encoded[current_ids], encoded[goal_ids])
            invalid_mass=(logits.softmax(-1)*(~allowed[current_ids])).sum(-1).mean()
            logits = logits.masked_fill(~allowed[current_ids], -1e9)
            imitation = -(targets[batch]*F.log_softmax(logits,dim=-1)).sum(-1).mean()
            value_loss = F.smooth_l1_loss(prediction, values[batch])
            loss = imitation + 0.15*value_loss + illegal_weight*invalid_mass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach())*len(batch)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.policy_epochs:
            metrics = direct_rollouts(policy, states, encoded, allowed, split["validation"], count=2000)
            score = selection_score(metrics)
            if score > best_score:
                best_score = score
                save_model(output/"policy.pt", policy, policy_kind, {"width":args.width}, split)
                report["policy_selected_epoch"] = epoch
                report["policy_validation"] = metrics
            item = {"epoch": epoch, "loss": total_loss/len(pairs), "elapsed_seconds": round(time.monotonic()-began,2), **metrics}
            report["policy_log"].append(item)
            emit({"phase": "policy", **item})
    report["policy_parameters"] = parameter_count(policy)
    if memory_source:
        reused=load_model(memory_source)
        if not isinstance(reused,EpisodicMemory):raise ValueError('재사용할 기억 모델 형식이 다릅니다.')
        shutil.copyfile(memory_source,output/'memory.pt')
        report.update(memory_parameters=parameter_count(reused),memory_selected_step=None,
                      memory_validation={'reused':True,'evaluation_repeated':False},
                      memory_source={'path':str(Path(memory_source).resolve()),'sha256':file_sha256(memory_source)})
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        report['model_sha256']={name:file_sha256(output/name) for name in ('policy.pt','memory.pt')}
        write_new_json(output/'training.json',report)
        emit({'phase':'training_complete','output':str(output),'policy_validation':report['policy_validation'],
              'memory_retrained':False})
        return report
    memory = EpisodicMemory(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(memory.parameters(), lr=0.002, weight_decay=0.00001)
    rng = np.random.default_rng(args.seed+17)
    best_memory = -1.0
    started_memory = time.monotonic()
    weights = torch.tensor([0.3,1,1,1,1],device=device)
    for number in range(1,args.memory_steps+1):
        memory.train()
        length = int(rng.choice((16,24,32,48)))
        sequence, expected, mask = memory_batch(encoded, token_labels, split["train"], rng, 128, length, device)
        logits, _ = memory(sequence)
        loss_items = F.cross_entropy(logits.reshape(-1,5), expected.reshape(-1), weight=weights, reduction="none").reshape(128,length,N_TOKENS).mean(-1)
        loss = (loss_items*mask).sum()/mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
        optimizer.step()
        if number % 100 == 0 or number == 1 or number == args.memory_steps:
            metrics = evaluate_memory(memory, encoded, token_labels, split["validation"])
            if metrics["exact_recall"] > best_memory:
                best_memory = metrics["exact_recall"]
                save_model(output/"memory.pt", memory, "episodic_memory", {"hidden_size":args.hidden_size}, split)
                report["memory_selected_step"] = number
                report["memory_validation"] = metrics
            item = {"step":number,"loss":float(loss.detach()),"elapsed_seconds":round(time.monotonic()-started_memory,2),**metrics}
            report["memory_log"].append(item)
            emit({"phase":"memory",**item})
            if best_memory >= 0.999 and number >= 800:
                break
    report["memory_parameters"] = parameter_count(memory)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report['model_sha256']={name:file_sha256(output/name) for name in ('policy.pt','memory.pt')}
    write_new_json(output/'training.json',report)
    emit({"phase":"training_complete","output":str(output),"policy_validation":report["policy_validation"],"memory_validation":report["memory_validation"]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",help="비어 있는 새 실행 폴더. 생략하면 artifacts/training_runs 아래에 새로 만듭니다.")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--device",default="auto",choices=("auto","cpu","cuda"))
    parser.add_argument("--policy-epochs",type=int,default=140)
    parser.add_argument("--memory-steps",type=int,default=1800)
    parser.add_argument("--hidden-size",type=int,default=128)
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--architecture", choices=("dense", "relational"), default="relational")
    parser.add_argument('--initial-policy',help='기존 정책을 복사해 별도 후보에서 미세 조정')
    parser.add_argument('--memory-source',help='기억 모델을 재학습하지 않고 새 실행 폴더에 복사')
    parser.add_argument('--policy-lr',type=float,default=.0015)
    parser.add_argument('--illegal-proposal-weight',type=float,default=0.0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
