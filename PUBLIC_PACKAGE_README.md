# Research package: 2026-11-13 snapshot

이 문서는 [고정 연구 릴리스](https://github.com/dreaminglizard/llm-driven-robot-policy-learning/releases/tag/v0.1.0-research)의 구성과 재현 조건을 설명합니다. 연구 개요는 [README](README.md), 현재 코드의 실행 안내는 [기술 안내](docs/TECHNICAL_GUIDE.md)를 보세요.

## 무엇을 내려받을까?

| 목적 | 자료 |
|---|---|
| 원래 소스·스크립트·테스트·선정 가중치 | 릴리스의 `LLM_robot_policy_source_and_selected_weights_20261113.zip` |
| 선정 정책의 저장 근거 | `LLM_selected_policy_evidence_20261113.zip` |
| 직접 시연 자료 | `LLM_direct_demonstrations_1000_20261113.zip` |
| 전체 산출물의 목록만 조사 | `LLM_research_artifacts_inventory_public_20261113.jsonl.gz` |

소스 ZIP은 8,751개 파일, 42,941,082바이트입니다. SHA-256:
`508826d83762dc602d783bee1c0c27aed9dd919d024a827b233a0b4f8abd2d3e`

인벤토리는 목록이지 모든 원본 파일의 공개가 아닙니다. 시연·선정 근거 자료 역시 독립 최종 평가 결과와 구분해야 합니다.

## 저장소와 ZIP의 차이

| 현재 GitHub 저장소 | 고정 릴리스 ZIP |
|---|---|
| 주요 코드와 `models/` 가중치, 이후 소스 수정 | 당시 원본 경로를 보존한 스냅샷 |
| Code → Download ZIP으로 다운로드 | Releases에서 자산 파일을 따로 다운로드 |
| 현재 루트에 `scripts/`, `tests/`, `artifacts/`, `모델_가중치/` 없음 | 해당 경로와 설정·소형 자료·실험 설명 포함 |
| 이후 버그 수정이 반영될 수 있음 | GitHub 수정이 자동으로 반영되지 않음 |

현재 `native_physical_session.py`와 `terra_native_close_evidence.py`에는 스냅샷 이후 수정이 있습니다. ZIP과 현재 소스를 섞는 경우 호환성을 별도로 확인해야 합니다.

## 실행 준비와 빠진 입력

Python 3.11 이상과 `pyproject.toml`의 용도별 선택 의존성(`learning`, `simulation`, `hardware`, `voice`)을 확인하세요. 일부 스크립트는 원래 로컬 경로를 사용하므로 설정이 필요합니다.

- 전체 실험 이력·모든 대용량 데이터·중간 체크포인트를 포함한 완전 재학습 패키지가 아닙니다.
- ZIP의 직접 연속 시연 converter와 테스트는 당시 버전입니다. converter는 다른 `scripts` 모듈에도 의존합니다.
- 관련 테스트에 필요한 지정 case-0007의 result·assignment·events·public trace·private checkpoints는 이 소스 ZIP에서 확인되지 않았습니다. 사례가 없으면 테스트가 건너뛰어지므로, 정상 종료를 실제 검사 통과로 해석하면 안 됩니다.
- 독립 환경의 전체 설치·재학습·실물 재현은 검증되지 않았습니다.

Gemma·Whisper 본체, llama.cpp, Python 가상환경과 장치 SDK 바이너리는 포함하지 않습니다. 원래 Gemma 경로는 `.models/gemma-4-e4b-qat-q4_0/gemma-4-E4B_q4_0-it.gguf`이며 projector `gemma-4-E4B-it-mmproj.gguf`를 함께 사용합니다. 외부 모델은 공식 배포처에서 실제 파일과 사용 조건을 확인하세요.

## 공개 범위와 사용 조건

인증정보가 포함될 수 있는 런타임·세션 로그, 개인 대화 링크와 내부 인계 문서는 제외하거나 정리했습니다. 코드·가중치의 별도 라이선스를 이 문서에서 부여하지 않습니다. 논문의 라이선스와 외부 모델·SDK의 조건은 각각 별개입니다.

실물 실행 스크립트는 로봇을 움직일 수 있습니다. 먼저 장치 없는 검사로 입출력을 확인하고, 실물 실행에는 별도 안전 준비와 승인이 필요합니다.

## English summary

The release is a frozen source-and-selected-weights snapshot, not a rolling repository backup or a complete retraining dataset. **Code → Download ZIP** and **Releases → asset ZIP** are different downloads. The inventory lists artifacts; it does not contain them all.

The original scripts and tests may require local paths and saved inputs not included in the source ZIP. In particular, the direct-demonstration converter tests skip without their designated case. A successful exit with skipped tests is not reproduction evidence. For small runnable checks, use the [technical guide](docs/TECHNICAL_GUIDE.md).
