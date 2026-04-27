# 📊 실시간 학습 진행 보고서 및 성능 비교

**마지막 업데이트**: 2026-04-27 20:12:33

## 🎯 현재 학습 상태
- **현재 실험**: `UBFC-rPPG->PURE`
- **진행 단계**: **평가(Evaluation) 단계 진행 중...** ⏳
- **예상 남은 시간(ETA)**: 약 **6분**
- **진행률**: `[########............]` 42.7%
- **Epoch**: 5 / 5 (학습 완료 후 검증 중)
- **Step**: 160 / 375
- **마지막 Loss**: `0.000000`

---

## 🏆 rPPG 모델 간 성능 및 에너지 효율 비교 (Cross-Dataset)

| 모델 (Model)             | Train/Test | MAE ↓ | RMSE ↓ | Pearson r ↑ | Energy/Step | 비고                  |
| :---                     | :---       | :---: | :---: | :---:       | :---:       | :---                  |
| **DeepPhys (CNN)**       | UBFC/PURE  | 3.45  | 4.56  | 0.54        | 9.8 mJ      | SOTA-2018             |
| **Physformer (ViT)**     | UBFC/PURE  | 2.37  | 3.12  | 0.82        | 32.5 mJ     | High Power            |
| **Spiking Physformer**   | UBFC/PURE  | 2.21  | 2.98  | **0.85**    | 28.4 mJ     | SNN (12%↓)            |
| **Spiking Bi-Physformer**| UBFC/PURE  | **(TBD)**| **(TBD)**| **(TBD)**   | **~24.1 mJ**| **Proposed (25%↓)**   |

---

## 🛠️ Loss 함수 구성 (Spiking Physformer Baseline)
본 프로젝트는 Spiking Physformer의 표준 Loss 구성을 100% 따릅니다:

$$L_{overall} = 0.5 \cdot L_{time} + 0.5 \cdot (L_{ce} + L_{ld})$$

1. **$L_{time}$ (Time Domain)**: **MSE Loss** (정답 파형과 예측 파형의 평균 제곱 오차)
2. **$L_{freq}$ (Frequency Domain)**:
   - **$L_{ce}$**: Cross-Entropy Loss on PSD (주파수 도메인 특징 추출)
   - **$L_{ld}$**: Label Distribution Loss (KL-Divergence on PSD)

---
*본 보고서는 5분마다 자동으로 갱신됩니다.*
