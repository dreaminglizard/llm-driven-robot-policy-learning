# task_program.v2 개발 계약

이 판본은 기존 `task_program.v1` 스키마·프롬프트·결과와 별도다. 모델의 출력은 실행기가 없는 **제안**이며 `proposal_only=true`, `execution_authorized=false`다. 기억한 목표 배치, 관절값이나 경로를 모델에 제공하거나 생성하지 않는다. [스키마](task_program.v2_1.schema.json), [프롬프트](task_program.v2.prompt.txt), [검사기](../raccoon_memory/task_program_v2.py)를 사용한다.

`steps`는 발화 순서가 아니라 **요청된 실행 순서**다. 각 단계의 `evidence`는 실제 발화의 연속 구절이어야 하지만, 구절 위치가 단계 순서대로 증가할 필요는 없다. 같은 단계를 같은 유일한 근거로 복제하면 차단한다. 서로 다른 단계가 한 구절에 근거하거나 같은 구절이 발화에 실제로 두 번 나오는 경우는 구분한다. 알려진 예제에서 정답 실행 순서는 평가기로 대조한다. 임의 문장의 전후 관계를 검사기가 완전히 이해한다고 보장하지 않는다.

`continue.active_task`는 실행 중 작업의 유지, `continue.previous_task`는 식별된 직전 작업을 이어가자는 제안, `repeat`는 별도 반복이다. 직전 작업 제안에는 이전 작업의 식별·중단 여부, 정지 확인, 현재 상태·배치 확인과 재계획이 필요하다는 표시가 남는다. 명시 정지 뒤 `repeat.active_task`가 나오면 당시 작업 ID와 상태 판본을 별도 메타데이터로 묶는다. 후속 작업이 이미 실행 가능하다는 뜻이 아니며 정지 확인, 참조·반복 계획 재검사와 각 회차의 상태 확인이 필요하다. 동일한 발화에 현재 작업과 직전 작업 모두 가능한 문맥이면 임의로 하나를 선택하지 않는다.

`repeat.count.mode=total`은 **새 반복 프로그램이 수행할 반복 횟수**다. 기준이 된 원래 작업이나 현재 진행 중인 원래 실행은 포함하지 않는다. 따라서 완료한 직전 작업을 한 번 반복하라는 제안은 새 반복 1회다. `additional`은 이미 정해진 반복 계획에 더할 횟수이며 기존 계획이 무엇인지 확인하거나 질문해야 한다. 범위가 모호하면 `unresolved` 또는 확인 요청으로 남긴다. 이 정의는 반복 실행기가 구현됐거나 확인됐다는 뜻이 아니다.

연속·미해결 반복의 `value`는 `null`, `prior_steps` 외 대상의 `step_indices`는 `null`, 기억 단계의 자리·원통·개수·높이·배치는 `null`이 되도록 생성 스키마가 분기한다. 이동에는 유효한 출발 또는 원통 ID와 다른 도착이 필요하다. 형식이 틀린 결과를 고쳐 성공으로 바꾸지 않는다. `uniqueItems`는 로컬 변환 문법에 반영되지 않는 한계가 있어 별도 스키마 검사도 유지한다. [오프라인 문법 감사](../artifacts/task_program/20261030_task_program_v2_preparation_final/schema_audit.json)는 동봉된 b11960 C++ 변환과 별도 작성한 문법 검사기 결과이며, 실제 서버 채팅 래퍼의 생성 문법을 덤프한 것은 아니다.

뒤 단계의 스키마 오류 또는 길이 제한이 있어도 응답이 엄격한 JSON으로 파싱되고 첫 단계가 실제 발화에 근거한 유효한 정지라면 독립 진단 접두부를 남긴다. 전체 응답은 실패로 유지하며, 부정·인용·형식이 틀린 정지를 억지로 추출하지 않는다. 이 기록은 실제 정지 요청이나 물리적 정지 확인이 아니다.

## 개발 입력과 평가

[실제 비교 입력](../data/task_program/v2_1/combined_manifest.json)은 이전 개발 자료의 원문 24건·전사 48건과 무음 1건, 새 한영 개발 대조 40건이다. 새 대조에는 전후 순서, 정지 후 모호한 복귀, 현재·직전 작업, 횟수·추가 횟수, 부정·정정·질문, 같은 발화의 다른 문맥을 넣었다. [검토 중 추가한 8건](../data/task_program/v2_1/combined_manifest_final.json)은 모의 검사에만 사용하며 이번 모델 요청에는 넣지 않는다. 서로 다른 입력군의 분모를 따로 기록하며 독립 최종 시험으로 부르지 않는다. [평가기](../scripts/evaluate_task_program_v2.py)는 이 입력의 텍스트만 읽고 요청·응답 원문, 오류·시간·토큰과 선택한 서버 프로세스의 요청 중 표본 메모리를 보존한다.

복귀 대상이 불명확할 때 직접 질문한 응답은 단어 일치로 정답을 확정하지 않는다. 질문 내용이 있는 후보는 사례별 검토 대기로 따로 두고, 질문 없음·빈 질문과 스키마 오류는 확정 실패로 분모에 남긴다. [검토 집계기](../scripts/review_task_program_v2_questions.py)는 판정과 근거를 명시한 기록만 반영하며, 미검토가 남으면 전체 입력 기준 정확도를 `null`로 둔다. 검토자와 검토 방식은 기록하고, 도구가 그 신원을 인증하지는 않는다.

아래 명령은 실행 방법이다. PID는 실행 직전에 확인하고 두 모델의 요청은 순차적으로 보낸다.

```powershell
python scripts/evaluate_task_program_v2.py --dataset data/task_program/v2_1/combined_development112.jsonl --base-url http://127.0.0.1:8799/v1 --model gemma4-e2b-intent --timeout 120 --server-pid <E2B_PID> --asset-manifest .models/gemma-4-e2b-qat-q4_0/verified_assets.json --output artifacts/task_program/v2_e2b_development
python scripts/evaluate_task_program_v2.py --dataset data/task_program/v2_1/combined_development112.jsonl --base-url http://127.0.0.1:8798/v1 --model gemma4-e4b-intent --timeout 120 --server-pid <E4B_PID> --asset-manifest .models/gemma-4-e4b-qat-q4_0/verified_assets.json --compare-against artifacts/task_program/v2_e2b_development --output artifacts/task_program/v2_e4b_development
```
