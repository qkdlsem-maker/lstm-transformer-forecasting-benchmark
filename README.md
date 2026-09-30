# LSTM–Transformer Forecasting Benchmark

시계열 예측에서 LSTM과 Transformer를 재현 가능한 조건으로 비교하는 연구입니다.

**상태: 2026-09-30 본 실험 300회, 학습률 선택 48회, 기준 모델 44건 완료. 독립 통계 검증 통과.**

이전 원고의 수치를 검증된 신규 결과로 간주하지 않습니다. 실제 공개 데이터만 사용하고, 단순 기준 모델이 더 좋거나 가설과 반대인 결과도 함께 보고합니다.

## 재현 절차

Linux CUDA 환경에서 PyTorch 및 `requirements.txt`의 패키지를 준비합니다. 시스템 전체 패키지를 변경하기보다 전용 가상환경을 사용하십시오. 기존 환경으로 실행하면 정확한 버전을 기록하십시오.

```bash
python study.py --stage manifest
python study.py --stage tune --smoke --dataset ETTh1 --limit 2 --device cuda:0
bash run_pipeline.sh
python attribution.py --device cuda:0
python validate_results.py
python publish_results.py
python plot_results.py
```

`PROTOCOL.md`에 분할, 누출 방지, 평가 척도, 모델, 선택 예산, 통계 방법을 명시했습니다. 15개의 서로 다른 조건 × 2개 모델 × 10개 시드 = 300개 본 실험과 48개 검증 전용 학습률 선택 실행으로 구성됩니다. 실행 시간은 GPU와 조기 종료 시점에 따라 달라집니다.

## 변경 사항

- 원본 ETTh1, ETTm1, Exchange, 실제 862채널 Traffic을 다운로드하고 해시를 기록합니다. Traffic은 훈련 구간 분산으로 선택한 7개 채널만 모델링하며, 합성 데이터로 대체하지 않습니다.
- 채널 선택과 학습용 정규화는 훈련 데이터만 사용합니다.
- 입력 길이와 예측 길이가 달라도 검증/테스트 예측 시작점을 동일하게 유지합니다.
- 데이터 비율별 학습 scaler와 별개로, 공통 평가 척도를 사용합니다.
- Persistence, Seasonal Naive, 검증셋으로 선택한 Ridge를 포함합니다.
- 테스트 정보 없이 학습률을 선택하고, 최저 검증 손실의 모델을 복원합니다.
- 실행별 독립 파일, 원자적 저장, smoke 분리, 코드/데이터 해시를 적용합니다.
- 전체 10개 시드의 대응 검정·효과크기·신뢰구간과 전체 검정군의 Holm 보정을 보고합니다.

## 파일

- `kci_verified_research_v5.zip`: 코드, 실행별 매니페스트, 전체 결과 CSV, 데이터 해시, 그림. 논문 파일은 포함하지 않습니다. 압축 해제 후 루트에서 재현 명령을 실행합니다. 원자료와 체크포인트는 포함하지 않습니다.
- `manuscript_model_summary.csv`, `manuscript_baselines.csv`, `paired_statistics.csv`: 원고 수치와 모든 조건의 통계.
- `validate_results.py`, `publish_results.py`: 실행 완전성·평가 일치·통계 및 Holm 보정 재계산.
- `attribution.py`, `plot_results.py`: 같은 방식의 입력 기울기 분석과 그림.

- `data_pipeline.py`: 원본 데이터, 시간순 분할, 정규화와 공통 평가 척도
- `models.py`: 기본 LSTM 및 Transformer
- `study.py`: 검증 전용 학습률 선택과 본 실험
- `baselines.py`: 단순 기준 및 Ridge 평가
- `analyze.py`: 결과 검증·통계 및 훈련 구간 정상성 진단
- `run_pipeline.sh`: 순차 실행

대용량 데이터·체크포인트·접속정보는 Git에 포함하지 않습니다. 데이터의 권리와 사용조건은 원 제공자를 따릅니다. 현재 저장소에 별도 오픈소스 라이선스를 부여하지 않았습니다.

## 해석 범위

15개 고유 조건에서 Holm 보정 후 두 검정 모두 Transformer 우위 10개, LSTM 우위 1개, 유의하지 않은 비교 4개였습니다. ETTh1·ETTm1·Exchange 기본 조건에서 Ridge가 두 신경망보다 낮은 MAE를 보였으며, Traffic에서는 LSTM이 가장 낮았습니다. 192단계에서 LSTM 평균은 낮았으나 차이는 유의하지 않았습니다. 모든 결과를 포함하고 반대 결과도 보존했습니다.

검증된 실행은 RTX 4090, FP32에서 수행했습니다. 학습 코드 해시는 `dcf877b9e5cdcfa78ada26c542d5e9f3eb64e911188a704ac4b2c6b8e9ecc0f6`입니다. 6개 프로토콜 단위 검사가 통과했고, 300개 실행의 고유성·예상 시드·데이터 해시·평가 시작점·기준 모델 척도를 확인했습니다. 대응 통계와 다중 비교 보정을 별도로 재계산했습니다.

시드 반복은 한 시간 분할에서 학습 변동을 측정하며 미래 기간 일반화를 보장하지 않습니다. 네 데이터셋의 차이로 정상성이나 계절성의 인과 효과를 단정하지 않습니다. 입력 gradient와 attention은 서로 다른 측정이며, Integrated Gradients 또는 설명 충실성 검증으로 부르지 않습니다.
