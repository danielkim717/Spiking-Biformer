# 📊 실시간 학습 진행 보고서 및 성능 비교

**마지막 업데이트**: 2026-05-01 04:46:21

## 🎯 현재 학습 상태
- **현재 실험**: `PURE → UBFC-rPPG` (1/2)
- **진행 단계**: **Training** ⏳
- **현재 실험 진행률**: `[....................]` 0.3%
- **전체 진행률**: `[....................]` 0.1%
- **Epoch**: 1 / 30
- **Step**: 30 / 375
- **예상 남은 시간(ETA)**: 약 계산 중...
- **현재 Loss**: `1.329958`

---

## 🔬 SNN 스파이크 모니터링 (Spike Firing Rate)
본 지표는 각 LIF 층의 발화율입니다 (0%면 소실, 5~25%가 건강한 영역).
> **현재 발화율**: `[9.64%, 9.36%, 9.25%, 8.95%, 8.98%, 8.67%]`  (평균 9.14%)

---

## 🏆 rPPG 모델 간 성능 비교 (Cross-Dataset)

| 모델 (Model)             | Train/Test | MAE ↓ | RMSE ↓ | Pearson r ↑ | 비고                  |
| :---                     | :---       | :---: | :---: | :---:       | :---                  |
| **DeepPhys (CNN)**       | UBFC/PURE  | 3.45  | 4.56  | 0.54        | Baseline 2018         |
| **Physformer (ViT)**     | UBFC/PURE  | 2.37  | 3.12  | 0.82        | High Power            |
| **Spiking Physformer**   | UBFC/PURE  | 2.21  | 2.98  | **0.85**    | SNN SOTA              |
| **Spiking Bi-Physformer**| PURE/UBFC-rPPG  | (대기 중) | (대기 중) | (대기 중) | **Proposed (SDLA+MS)**|
| **Spiking Bi-Physformer**| UBFC-rPPG/PURE  | (대기 중) | (대기 중) | (대기 중) | **Proposed (SDLA+MS)**|

---

## 🛠️ 최근 작업 타임라인
- `[00:03]` Cross-dataset 실험 시작 (2개)
- `[진행]` **PURE → UBFC-rPPG** Training (Epoch 1/30, Step 30/375)

---
*본 보고서는 5분마다 자동 갱신됩니다.*
