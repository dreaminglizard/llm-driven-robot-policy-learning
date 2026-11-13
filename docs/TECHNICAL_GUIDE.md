# Technical entry guide / 기술 구조와 첫 실행

**LLM-assisted development of learned robot policies, not just an LLM calling robot functions.**
The research question is how language models can help create training data and learning software so that a smaller policy acquires behavior through changes to its weights. RaccoonBot is the application case, not the scope of the idea.

**핵심은 LLM을 이용한 로봇 조작 모델의 개발·데이터 구성과 학습, 그리고 학습으로 바뀌는 가중치입니다.**
다만 “LLM이 개발을 돕는다”, “학습기가 가중치를 갱신한다”, “실행 중 Gemma가 명령을 해석한다”는 서로 다른 단계입니다.

## 1. Two paths, not one / 개발과 실행 구분

The diagrams are conceptual maps of the published source, not proof that every experimental variant uses the same runtime.

```mermaid
flowchart LR
    A["LLM-assisted design and data preparation"] --> B["Recorded examples and provenance"]
    B --> C["Offline policy training"]
    C --> D["Loss → gradients → optimizer step"]
    D --> E["Updated policy weights / checkpoint"]
    E --> F["Separate recorded, simulated and physical evaluation"]
    F --> A
```

```mermaid
flowchart LR
    U["User instruction"] --> G["Gemma: interpretation / task proposal"]
    G --> V["Validated task and reference context"]
    V --> P["Learned manipulation policy"]
    O["Current observation and allowed history"] --> P
    W["Saved learned weights"] --> P
    P --> C["Controller / execution checks"]
    C --> S["Robot SDK"]
    S --> R["Robot"]
```

**한국어 요약:** 개발 단계는 데이터 → 학습 → 가중치 갱신 → 평가입니다. 실행 단계는 Gemma → 조작 모델 → 제어기·SDK입니다. 이 저장소에는 여러 실험 경로가 있으므로, 위 그림만으로 모든 실행이 순수 신경망 제어이거나 매번 온라인 학습한다고 해석하면 안 됩니다.

## 2. Where the weight update actually happens

| Question | Source entry | What it shows |
|---|---|---|
| Does training modify model weights? | [scene_trajectory_training.py](../raccoon_memory/scene_trajectory_training.py), `train` | Computes a loss, calls `loss.backward()` and `optimizer.step()`, then saves `trajectory_student.pt`. |
| What does that trajectory learner optimize? | Same module | Joint-position MSE + velocity MSE + weighted gripper cross-entropy + terminal binary cross-entropy. |
| What is a saved model? | [neural.py](../raccoon_memory/neural.py), `load_model`; [train.py](../raccoon_memory/train.py), `save_model` | The numbered-slot models store architecture/configuration and `state_dict`; loading restores weights and switches to evaluation mode. This is a different checkpoint family from the trajectory student. |
| What can a manipulation model output? | [scene_policy.py](../raccoon_memory/scene_policy.py), `ScenePolicy.forward` | History-window input; action logits and four normalized absolute joint targets. |
| Where is remembered state represented? | [inference.py](../raccoon_memory/inference.py), `LearnedRecall` | GRU hidden state is retained across observations; an explicit new write resets the old hidden state. |
| Is every plan purely neural? | Same module, `LocalPlanner.plan` | First attempts a learned policy; optional bounded symbolic search is reported separately as search assistance. |
| Does language parsing itself move the robot? | [task_program.py](../raccoon_memory/task_program.py) | Produces and validates bounded task-program proposals; does not dispatch robot commands. |
| Is controller completion an observed success? | [robot_bridge.py](../raccoon_memory/robot_bridge.py) | Explicitly distinguishes expected object state from observed physical success. |

The optimizer changes learned policy parameters. The published code does **not** establish that Gemma directly writes arbitrary neural-network weight values. Likewise, the presence of LLM-assisted development does not mean all labels come from an LLM: the trajectory module documents replay-derived labels and direct-command variants with distinct provenance.

**가중치 수정의 의미:** 학습 데이터와 손실을 바탕으로 학습기가 모델 파라미터를 갱신합니다. “Gemma가 가중치 숫자를 직접 써 넣는다”는 주장과는 다릅니다. 데이터 출처도 경로별로 구분해야 합니다.

## 3. First runnable check: no robot and no model download

This is deliberately a small **symbolic representation smoke check**, not a learned-policy demonstration. It gives a newcomer a safe first executable entry point without installing the SDK, downloading weights, or starting training.

From the repository root, with Python 3.11 or newer:

```sh
python -m examples.offline_world_smoke
```

Expected output:

```text
PASS: symbolic move, restoration, 80-value encoding, invalid-state rejection
1/2/3/4 -> -/21/3/4 -> 1/2/3/4
Not tested: learned weights, LLM inference, physics, or hardware.
```

Each slash separates a registered slot. Stack IDs are written bottom to top: `21` means cylinder 1 is above cylinder 2. The example moves cylinder 1 from A to B, reverses that move, verifies the 80-value representation, and checks rejection of duplicate object IDs.

**검증 범위:** 2026-11-13, Python 3.12.14의 별도 클라우드 작업공간에서 공개된 `world.py`와 `__init__.py`를 사용해 위 출력을 확인했습니다. 모델 가중치·LLM·물리 시뮬레이터·실물 성공률 검사는 아닙니다. 이 확인에는 모터 작동이나 새 학습이 없습니다.

## 4. What the evidence does and does not establish

- **Recorded-observation metrics:** next-action accuracy / target error on saved observations. They do not establish closed-loop or physical success. See `RECORDED_METRIC_SCOPE` in [scene_policy.py](../raccoon_memory/scene_policy.py).
- **Trajectory validation:** grouped by original source, but explicitly not an independent final holdout. Small errors on these records are not evidence of generalization to new robots or environments.
- **Memory boundary:** `LearnedRecall` rejects recall beyond the documented 48-step sequence scope. A new explicit write replaces the remembered episode.
- **Terminal boundary:** trajectory training documents `apply_then_finish`: stopping before applying the last output creates a one-tick mismatch.
- **Physical demonstration:** a successful demonstration is valuable, but it must not be presented as a measured success rate without a trial count, conditions and observations.
- **Current verification here:** only the small symbolic smoke check above was executed. Full clean-environment installation, pretrained inference, training reproduction and physical testing were not performed for this guide.

No performance number or demo video has been invented. This guide intentionally does not turn source-code assertions into measured benchmark results.

## 5. Reading order / 권장 탐색 순서

1. [Offline example](../examples/offline_world_smoke.py): understand the four-slot task representation.
2. [World representation](../raccoon_memory/world.py): states, moves, encoding and constraints.
3. [Model definitions](../raccoon_memory/neural.py) and [scene-policy definitions](../raccoon_memory/scene_policy.py): distinguish the model families.
4. [Trajectory training](../raccoon_memory/scene_trajectory_training.py): data labels, weight updates, saved checkpoint and stated limitations.
5. [Inference](../raccoon_memory/inference.py): learned inference, retained memory and optional search.
6. [Task-program interface](../raccoon_memory/task_program.py) and [robot bridge](../raccoon_memory/robot_bridge.py): language-to-task boundary and execution boundary.

This guide grants no new license and changes no third-party terms. It is a technical entry point, not a replacement for the repository's existing introduction or publication history.
