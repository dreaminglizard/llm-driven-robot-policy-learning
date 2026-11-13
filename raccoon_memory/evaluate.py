"""Frozen held-out symbolic evaluation. The exact solver is baseline only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .inference import LearnedRecall, LocalPlanner
from .neural import load_model
from .oracle import plan as exact_plan
from .train import direct_rollouts, evaluate_memory, tensors
from .world import INITIAL, features, from_tokens, step, tokens
from .data import validate_split
from .learning_io import new_report_path,write_new_json


def evaluate(model_dir, count=2000, *, output=None):
    if type(count) is not int or count<1:
        raise ValueError('평가 에피소드 수는 1 이상의 정수여야 합니다.')
    root = Path(model_dir)
    output=new_report_path(root,'evaluation',output)
    split = json.loads((root/"split.json").read_text(encoding="utf-8"))
    validate_split(split)
    torch.set_num_threads(2)
    states, encoded, allowed, token_labels = tensors("cpu")
    policy = load_model(root/"policy.pt")
    memory = load_model(root/"memory.pt")
    seed = 20261027
    report = {"at":datetime.now(timezone.utc).isoformat(), "domain":"symbolic_four_cylinders_four_numbered_slots",
        "physical_trials":0, "verified_physical_successes":0, "test_goal_configurations":len(split["test"]),
        "goal_split":"exact configurations; not unseen task families or unseen object counts",
        "model_selection":"validation selection reported by the saved training run; not reselected here", "seed":seed,
        "evaluation_kind":"repeat_on_saved_test_goals","new_blind_test":False,
        "memory_overwrite_semantics":"raw GRU sequence; deployed replace-on-write wrapper has separate regression evidence",
        "model_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("policy.pt","memory.pt")}}
    report["direct_policy"] = direct_rollouts(policy,states,encoded,allowed,split["test"],seed=seed,count=count)
    planner = LocalPlanner(root)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,len(states),count)
    goals = rng.choice(split["test"],count)
    success, assisted, expansions, lengths, optimum = 0,0,[],[],[]
    failures = []
    for si, gi in zip(starts,goals):
        current,goal = states[si],states[gi]
        try:
            result = planner.plan(current,goal)
            for action in result.actions:
                current = step(current,action)
            success += current==goal
            assisted += result.method=="learned_policy_with_search"
            expansions.append(result.expanded_states)
            lengths.append(len(result.actions))
            optimum.append(len(exact_plan(states[si],goal)))
        except (ValueError,RuntimeError) as exc:
            failures.append({"start":int(si),"goal":int(gi),"error":str(exc)})
    report["policy_with_explicit_search"] = {"episodes":count,"successes":success,"success_rate":success/count,
        "search_assisted_episodes":assisted,"max_expanded_states":max(expansions,default=0),
        "mean_moves":float(np.mean(lengths)),"mean_extra_moves_vs_exact_baseline":float(np.mean(np.array(lengths)-optimum)),
        "exact_solver_used_for_action_selection":False,"failures":failures}
    report["memory_by_length"] = {str(length):evaluate_memory(memory,encoded,token_labels,split["test"],seed=seed+1,length=length)
                                  for length in (8,24,48)}
    # Identical final observation across every history, with a later overwrite.
    rng = np.random.default_rng(seed+2)
    targets = np.repeat(split["test"],8)
    histories = rng.integers(0,len(states),(len(targets),48))
    histories[:,2] = rng.choice(split["train"],len(targets))
    histories[:,22] = targets
    histories[:,-1] = states.index(INITIAL)
    writes = torch.zeros(len(targets),48,1)
    writes[:,2,0] = 1
    writes[:,22,0] = 1
    seq = torch.cat((encoded[torch.tensor(histories)],writes),-1)
    with torch.inference_mode():
        full,_ = memory(seq)
        reset,_ = memory(seq[:,-1:])
    expected = token_labels[torch.tensor(targets)]
    recalled = full[:,-1].argmax(-1)
    report["memory_overwrite"] = {"histories":len(targets),"exact_recall":float((recalled==expected).all(-1).float().mean())}
    report["memory_hidden_reset"] = {"histories":len(targets),"exact_recall":float((reset[:,-1].argmax(-1)==expected).all(-1).float().mean()),
                                     "same_final_observation":True}
    end_to_end = 0
    for predicted, gi in zip(recalled.tolist(),targets):
        try:
            recovered = from_tokens(predicted)
            if recovered != states[gi]:
                continue
            result = planner.plan(INITIAL,recovered)
            current = INITIAL
            for action in result.actions:
                current = step(current,action)
            end_to_end += current==states[gi]
        except ValueError:
            pass
    report["remember_then_restore"] = {"histories":len(targets),"successes":int(end_to_end),"success_rate":end_to_end/len(targets),
        "start_for_restore":"same flat state", "target_input":"GRU decode only", "physical_success":None}
    paired = []
    for past in (((1,2),(),(3,4),()),((1,4),(),(3,2),())):
        recall = LearnedRecall(root)
        recall.observe(INITIAL)
        current = INITIAL
        for action in planner.plan(current,past).actions:
            current = step(current,action)
            recall.observe(current)
        recall.observe(current,write=True)
        for action in planner.plan(current,INITIAL).actions:
            current = step(current,action)
            recall.observe(current)
        target = recall.recall()
        result = planner.plan(current,target)
        for action in result.actions:
            current = step(current,action)
        paired.append({"remembered_past":past,"same_present":INITIAL,"decoded_target":target,
            "actions":result.actions,"restored":current,"correct":current==past,"method":result.method})
    report["same_present_different_pasts"] = paired
    report["baseline"] = {"kind":"explicit_saved_goal_and_exact_symbolic_search",
                          "purpose":"grading and shortest-path comparison only"}
    report['output']=str(output.resolve())
    write_new_json(output,report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",default="artifacts/memory_v2")
    parser.add_argument("--episodes",type=int,default=2000)
    parser.add_argument('--output',help='새 평가 결과 경로. 기존 evaluation.json은 보존합니다.')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model_dir,args.episodes,output=args.output),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
