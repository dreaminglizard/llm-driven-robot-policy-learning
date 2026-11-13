# task_program.v1 텍스트 해석 계약

이 경로는 한국어·영어 전사 한 문장을 순서 있는 **프로그램 제안**으로 만든다. 기존 단일 의도 코드와 결과는 사용하거나 변경하지 않으며, 프로그램을 세션이나 로봇 실행기에 전달하지 않는다. 복합 지시를 일괄 거부하지 않고 [음성 복합 명령 계약](../docs/architecture_research/05_음성_복합명령과_실행제어_계약.md)에 따라 순서와 미해결 참조를 보존한다.

## 스키마와 참조

[정식 JSON 스키마](task_program.schema.json)는 `schema`, 호출자가 정하는 `request_id`, 순서 있는 `steps`를 가진다. `steps`는 1~8개이며 다음 여덟 종류다.

- `task`: 이동·쌓기·풀기·자리 교환·기억·등록·배치 목표. 명시된 자리, 원통 ID, 개수, 높이와 배치 항목을 보존한다. 언급하지 않은 값은 `null`이다.
- `repeat`: `prior_steps`, 현재 `active_task`, `previous_task`, `previous_program`, `unresolved` 중 하나를 참조한다. `prior_steps`는 앞선 `task` 또는 `return`만 순서대로 참조하고 중첩 반복이나 제어 단계는 참조하지 않는다. 횟수는 `continuous`, `total`, `additional`, `unresolved`를 구분한다.
- `stop_motion`: 현재 동작의 정지 요청을 나타낸다. 정지 확인이나 물리 성공의 증거가 아니다.
- `stop_repeat`: 별도 즉시 정지가 없으면 기본 `after_round`; 현재 작업까지만 마친다고 명시한 경우 `after_task`다. 경계가 불명확하면 `unresolved`다.
- `continue`: 실행 중인 같은 작업을 유지한다. 새 작업 시작이나 이미 요청된 정지의 취소가 아니다. 현재 작업을 식별할 수 없으면 참조가 `unresolved`다.
- `return`: `arm_home`, `registered_initial`, `task_start`, `learned_memory`, `unresolved`를 구분한다. 배치나 경로를 반환 단계에 넣지 않는다.
- `clarify`, `status`: 모호함·부정만 있는 말·상태 질문을 보존한다. 명시적으로 확인된 앞선 정지를 자동으로 지우지 않는다.

각 단계의 `evidence`는 실제 전사 문장에서 순서대로 인용한 비어 있지 않은 연속 문자열이다. `return`, `repeat`, `continue`의 `reference`도 실제 발화에 있어야 한다. 숫자·참조·단계 순서를 임의로 채우지 않으며, 스키마나 근거 검사가 실패하면 모델 출력을 다른 행동으로 수정하지 않고 실패와 원문을 기록한다.

모델에는 `request_id`를 요구하지 않는다. 스키마에서 그 필드만 제외한 동일한 단계 스키마로 제안을 받고, 호출자가 `request_id`를 덧붙여 정식 스키마를 검증한다. 이는 추적용 메타데이터이며 작업 단계를 생성하거나 보정하지 않는다.

## 실행 문맥

모델에 전달하는 문맥은 [해석기](../raccoon_memory/task_program.py)의 `normalize_context`가 허용하는 상태와 불투명 참조뿐이다. 예를 들어 현재 실행 중인 작업은 다음처럼 식별한다.

```json
{"execution_state":"running","active_request_id":"motion-1","state_revision":"state-3","repeat_state":"active"}
```

직전 작업이나 프로그램은 `status`, `reference_id`, `revision`을 가진 참조로만 제공한다. 원위치의 대화 초점도 종류와 `explicit_recent` 출처 및 판본만 제공한다. 기억한 목표 배치, 관절값, 경로와 임의의 추가 문맥 필드는 거부한다. `continue.active_task`는 실행 상태가 `running`이고 현재 요청 ID와 상태 판본이 있을 때만 유효하다. 앞 단계에 정지 요청이 있으면 같은 작업을 계속 실행 중이라고 확정하지 않는다.

“멈추지 말고 지금 작업을 계속해”는 `continue`이며 정지 요청이 아니다. “반복하지 말고 이번 작업만 마쳐”는 `stop_repeat`의 `after_task` 경계와 필요한 `continue`로 표현한다. 단순 부정어 검출만으로 이 발화를 버리지 않는다. “동작 중단하고 원위치해”에서 원위치가 불명확하면 `stop_motion`과 `return.unresolved`를 순서대로 보존한다.

뒤 단계의 참조 검사에 실패해도 앞에서 개별 검증된 정지 단계는 `validated_stop_prefix`에 진단 근거로 남는다. 전체 JSON 스키마가 깨졌거나 앞 정지 자체가 부정·인용·질문이면 이 접두부를 만들지 않는다. 이 값도 실행 권한이 아니며, 직접 정지 접수와 정지 확인·후속 실행은 아직 별도 실행 경로에서 구현해야 한다.

## ASR 텍스트 평가 입력

[평가기](../scripts/evaluate_task_program.py)는 JSONL을 읽는다. 각 줄에는 고유한 ASCII `id`, `language`(`ko`, `en`, `mixed`, `unknown`), `transcript`와 선택적인 `reference_context`, `expected`가 들어간다. `transcript`가 비었거나 `null`이면 모델을 호출하지 않고 실패로 남긴다. 음성 파일을 읽지 않는다. 다음은 형식 예시다.

```json
{"id":"ko-001","language":"ko","transcript":"동작 중단하고 원위치해","reference_context":{"execution_state":"running","active_request_id":"motion-1","state_revision":"state-3"},"expected":{"schema":"task_program.v1","steps":[{"op":"stop_motion","scope":"active_task","evidence":"동작 중단"},{"op":"return","target":{"kind":"unresolved","reference":"원위치"},"evidence":"원위치해"}]}}
```

두 서버를 사용할 때는 같은 JSONL과 [한영 공통 프롬프트](task_program.prompt.v1.txt), 스키마를 적용해 **순차적으로** 실행한다. 모델별 동일 준비 발화 1건이 기록되며 생성 설정은 온도 0, top-p 1, 최대 512토큰, 시드 20261030, 스트리밍 꺼짐이다. 예시는 실행 방법일 뿐이며 현재 준비 단계에서는 호출하지 않았다.

```powershell
python scripts/evaluate_task_program.py --dataset "ASR_TRANSCRIPTS.jsonl" --base-url "http://127.0.0.1:8799/v1" --model "gemma4-e2b-intent" --output "artifacts/task_program/e2b_run_01"
python scripts/evaluate_task_program.py --dataset "ASR_TRANSCRIPTS.jsonl" --base-url "http://127.0.0.1:8798/v1" --model "gemma4-e4b-intent" --output "artifacts/task_program/e4b_run_01"
```

새 출력 폴더에 요청과 응답 원문, 실패·시간, 토큰, 입력과 코드 지문을 남긴다. 모델 서버 PID를 주면 요청 구간의 프로세스 메모리를 표본으로 측정하며, 측정이 불가능하면 미측정으로 둔다. 시간 초과·잘린 응답은 성공이나 재시도로 바꾸지 않는다.

후속 출력 계약 대조는 [필드 설명 보강 프롬프트](task_program.prompt.v2_1.txt)를 `--prompt-file`로 선택한다. 미지정 시 기존 v1이 기본이다. 선택한 파일과 실제 시스템 메시지의 지문을 기록하고 사본을 저장한다. `--compare-baseline`에 이전 실행 폴더를 주면 모델 이름과 시스템 메시지를 제외한 입력·스키마·생성 설정이 같은지 검사한다. 두 번째 모델에 `--compare-against`로 첫 번째 모델의 새 실행 폴더를 추가하면 모델 이름을 제외한 전체 요청이 같은지도 검사한다. 이 설정 준비는 모델 호출이 아니며 FK 학습·물리 평가가 끝나기 전에는 후속 생성을 시작하지 않는다.

정답이 있는 경우 원시 의미와 검증 후 의미, 잘못된 작업·정지 제안, 검증된 정지 접두부와 미해결 프로그램을 별도로 집계한다. 의미 비교는 단계 순서·종류·대상 종류·자리·개수·경계를 비교하며 `evidence`와 원문 참조 문자열은 제외한다. 후자는 검사기가 실제 전사와 별도로 대조한다. 정답이 없으면 정확도는 계산하지 않는다. 이 평가는 전사된 텍스트 조건의 결과이지 실제 마이크 성능이나 물리 성공률이 아니다.
