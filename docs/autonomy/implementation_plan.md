# Spiking Bi-Physformer 교차 검증 및 벤치마크 구현 계획

본 계획은 사용자가 직접 데이터를 제공하는 환경에서 **UBFC-rPPG**와 **PURE** 데이터셋 간의 교차 검증을 수행하고, 이를 기존 모델(Physformer 등)과 비교 분석하는 프로세스를 정의합니다.

## 사용자의 역할 (Action Required)
> [!IMPORTANT]
> 1. **데이터 준비**: `data/PURE.zip` 및 `data/UBFC.zip` 파일명으로 `C:\Users\user\SNNBiformer\data\` 폴더에 압축 파일을 배치해 주세요.
> 2. **환경 대기**: 압축 파일이 감지되면 제가 자율적으로 압축 해제 및 학습 루프를 시작합니다.

## Proposed Changes

### 1. 교차 검증 시퀀스 정의 (Sequential Pipeline)
- **Phase 1: UBFC → PURE**
  - UBFC-rPPG 학습 (Train 80% / Val 20% 분할)
  - 최적 체크포인트(`best_ubfc.pth`) 저장
  - PURE 전체 데이터셋에 대해 평가 및 성능 기록
- **Phase 2: PURE → UBFC**
  - PURE 학습 (Train 80% / Val 20% 분할)
  - 최적 체크포인트(`best_pure.pth`) 저장
  - UBFC-rPPG 전체 데이터셋에 대해 평가 및 성능 기록

### 2. 베이치마크 비교 보고서 생성
`results/cross_dataset_comparison.md` 파일을 생성하여 다음과 같은 비교표를 작성합니다.
> [!NOTE]
> 연구 문헌에 보고된 Physformer의 교차 검증 Baseline 수치를 대조군으로 사용합니다.

| Train | Test | Model | MAE | RMSE | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UBFC** | **PURE** | **Spiking Bi-Physformer** | (측정 예정) | (측정 예정) | (측정 예정) |
| UBFC | PURE | Physformer (Baseline) | ~13.19 | ~24.57 | ~0.47 |
| **PURE** | **UBFC** | **Spiking Bi-Physformer** | (측정 예정) | (측정 예정) | (측정 예정) |
| PURE | UBFC | Physformer (Baseline) | ~1.69 | ~6.64 | ~0.93 |

### 3. 핵심 컴포넌트 고도화

#### [NEW] `scripts/benchmark_runner.py`
- 데이터 압축 파일이 존재하는지 1분 주기로 폴링(Polling)합니다.
- 파일 발견 시 `agent_data.py` -> `train.py` -> `metrics.py`를 순차 실행합니다.

#### [MODIFY] `src/data/datasets.py`
- PURE: PNG 시퀀스 및 JSON 라벨 파싱 로직 완성.
- UBFC: AVI 비디오 및 TXT 라벨 파싱 로직 완성.
