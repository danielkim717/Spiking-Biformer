# Spiking Bi-Physformer 교차 검증 벤치마크 결과

본 보고서는 **Spiking Bi-Physformer** 모델의 도메인 일반화 능력을 측정하기 위해 PURE 및 UBFC-rPPG 데이터셋 간의 교차 검증을 수행하고, 기존 SOTA 모델인 Physformer와 비교한 결과를 담고 있습니다.

## 1. 실험 요약 및 비교표
| Train | Test | Model | MAE (bpm) | RMSE (bpm) | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UBFC** | **PURE** | **Spiking Bi-Physformer** | (대기 중) | (대기 중) | (대기 중) |
| UBFC | PURE | Physformer (Baseline) | 13.19 | 24.57 | 0.47 |
| **PURE** | **UBFC** | **Spiking Bi-Physformer** | (대기 중) | (대기 중) | (대기 중) |
| PURE | UBFC | Physformer (Baseline) | 1.69 | 6.64 | 0.93 |

> [!NOTE]
> - **PURE → UBFC** 방향은 일반적으로 도메인 간 유사도가 높아 높은 성능이 기대됩니다.
> - **UBFC → PURE** 방향은 조명 및 카메라 조건 차이로 인해 대다수 모델에서 성능 저하가 발생하는 고난도 구간입니다.

## 2. 실험 로그 및 체크포인트
- `checkpoints/best_ubfc.pth`: UBFC 학습 모델
- `checkpoints/best_pure.pth`: PURE 학습 모델

---
*학습이 완료되면 위 표의 (대기 중) 항목이 자동으로 업데이트됩니다.*
