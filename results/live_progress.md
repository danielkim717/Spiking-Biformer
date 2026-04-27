# 📊 실시간 학습 진행 보고서 및 성능 비교

**마지막 업데이트**: 2026-04-28 05:59:05

## 🎯 현재 학습 상태
- **현재 실험**: `UBFC-rPPG->PURE`
- **진행 단계**: **평가(Evaluation) 중...** ⏳
- **진행률**: `[###################.]` 98.7%
- **Epoch**: 1 / 1
- **Step**: 370 / 375
- **예상 남은 시간(ETA)**: 약 0분
- **마지막 Loss**: `0.000000`

---

## 🔬 SNN 스파이크 모니터링 (Spike Firing Rate)
본 지표는 각 층의 뉴런이 얼마나 활발하게 발화하는지 나타냅니다 (0%면 소실된 것).
> **현재 발화율**: `계산 중...`

---

## 🏆 rPPG 모델 간 성능 비교 (Cross-Dataset)

| 모델 (Model)             | Train/Test | MAE ↓ | RMSE ↓ | Pearson r ↑ | 비고                  |
| :---                     | :---       | :---: | :---: | :---:       | :---                  |
| **DeepPhys (CNN)**       | UBFC/PURE  | 3.45  | 4.56  | 0.54        | Baseline 2018         |
| **Physformer (ViT)**     | UBFC/PURE  | 2.37  | 3.12  | 0.82        | High Power            |
| **Spiking Physformer**   | UBFC/PURE  | 2.21  | 2.98  | **0.85**    | SNN SOTA              |
| **Spiking Bi-Physformer**| UBFC/PURE  | **(TBD)**| **(TBD)**| **(TBD)**   | **Proposed (SDLA+MS)**|

---

## 🛠️ 최근 작업 타임라인 (5분 단위 갱신)
- `[00:55]` 아키텍처 전면 개정 (Spiking Shortcut & Learnable Scale 적용)
- `[01:00]` 학습 재시작 (Iteration 1: Vth=1.0)
- `[01:05]` 스파이크 발화율 정상화 확인 중

---
*본 보고서는 5분마다 자동으로 갱신됩니다.*
