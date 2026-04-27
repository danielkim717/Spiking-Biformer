# 📊 실시간 학습 진행 보고서 및 성능 비교

**마지막 업데이트**: 2026-04-27 23:23:28

## 🎯 현재 학습 상태
- **현재 실험**: `UBFC-rPPG->PURE`
- **진행 단계**: **학습(Training) 단계 진행 중...** 🚀
- **예상 남은 시간(ETA)**: 약 **1시간 16분**
- **진행률**: `[##############......]` 71.9%
- **Epoch**: 4 / 5
- **Step**: 140 / 235
- **마지막 Loss**: `2.898855`

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


---

## [1차 수정] - 2026-04-27 22:22:26
- **현재 성능**: Pearson r = 0.0000, MAE = 0.0000
- **원인 분석**: 뉴런의 발화가 충분하지 않거나 초기 가중치가 파형을 형성하지 못함 (Dead Neuron 가능성).
- **조치 사항**: V_threshold를 0.1로 추가 인하하여 발화 빈도 극대화.
---
