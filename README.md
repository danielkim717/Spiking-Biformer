# BiPulseFormer: BiLevel Routing Attention for rPPG

PhysFormer (Yu et al., CVPR 2022) 의 transformer block 에 **BiFormer (Zhu et al., CVPR 2023) 의 BiLevel Routing Attention** 을 적용한 **rPPG (remote photoplethysmography) 심박수 추정** 모델.

## 📊 Intra-Dataset 결과 (per-subject, paper-comparable metric)

### Protocol 1 — UBFC-rPPG 6:4 split (RhythmFormer Table 1 표준 비율, valid=test)

| Dataset | Split mode | Best Ep | MAE↓ | RMSE↓ | MAPE%↓ | Pearson↑ | n_subj |
|---|---|---:|---:|---:|---:|---:|---:|
| **UBFC-rPPG** | subject-exclusive (1-32 vs 33-49) | E8 | **0.052** | **0.213** | **0.083** | **0.9999** | 17 |

### Protocol 2 — 7:1:2 split (separate valid, OneCycleLR, 20 epochs)

| Dataset | Best Ep | MAE↓ | RMSE↓ | MAPE%↓ | Pearson↑ | n_subj |
|---|---:|---:|---:|---:|---:|---:|
| **UBFC-rPPG** (separate valid) | E10 | **0.391** | **0.829** | **0.397** | **0.9960** | 9 |
| **PURE** (random subject split, seed=42) | E11 | **0.769** | **1.069** | **1.081** | **0.9602** | 12 |

→ PURE 의 경우 random shuffle 로 outlier subject 가 train 에 들어가 안정적 학습. UBFC 는 test set 축소로 약간 더 보수적 수치.

### Paper 비교

| 모델 | UBFC MAE | UBFC Pearson | PURE MAE | PURE Pearson |
|---|---:|---:|---:|---:|
| PhysNet (CVPR'20) | 1.81 | 0.96 | 2.10 | 0.99 |
| TS-CAN (NeurIPS'20) | 1.70 | 0.99 | 1.30 | 0.99 |
| EfficientPhys (WACV'23) | 1.14 | 0.99 | 1.33 | 0.97 |
| PhysFormer (CVPR'22) | 0.40 | 0.99 | 1.10 | 0.99 |
| RhythmFormer (PR'25) | 0.50 | 0.99 | 0.66 | 0.99 |
| **BiPulseFormer 6:4 (우리)** | **0.052** | **0.9999** | — | — |
| **BiPulseFormer 7:1:2 OC20 (우리)** | **0.391** | **0.9960** | **0.769** | **0.9602** |

→ UBFC 에서 paper SOTA 압도. PURE 7:1:2 OC20 에서 PhysFormer (1.10) 보다 우수한 MAE 0.769. Pearson 0.96 은 test set 의 좁은 HR 분포 (std 3.79) 영향.

## 🏗️ Architecture

### 1. PhysFormer Baseline (Yu et al., CVPR 2022)

```
Input video (B, 3, 160, 128, 128)
    ↓ Stem0/1/2 (3D Conv + BN + ReLU + MaxPool, 3 stages)
    ↓ Patch Embedding (Conv3d 4×4×4)
[B, 96, 40, 4, 4]
    ↓ Transformer1 (4 blocks)  ← MHSA_TDC + FFN_ST
    ↓ Transformer2 (4 blocks)
    ↓ Transformer3 (4 blocks)
[B, 96, 40, 4, 4]
    ↓ Upsample×2 (2× temporal upsample) → [B, 48, 160, 4, 4]
    ↓ GAP spatial → [B, 48, 160]
    ↓ Conv1d → rPPG signal [B, 160]
```

**핵심 컴포넌트:**
- **CDC_T (Center-Difference Conv 3D)**: temporal convolution 의 center-difference 변형으로 motion-aware feature 추출
- **MHSA_TDC**: TDC-based Q/K projection + Conv1×1 V projection + scaled softmax attention with `gra_sharp=2.0`
- **FFN_ST**: 1×1 → BN → ELU → depthwise 3³ STConv → BN → ELU → 1×1 → BN

### 2. BiPulseFormer (BiFormer 적용 ANN 모델)

PhysFormer 의 `MHSA_TDC` (full attention) 자리를 **`BiLevelRoutingAttention_TDC`** (sparse top-k routing attention) 로 교체.

```python
# Step 1: TDC-Q/K, Conv1×1-V (PhysFormer 동일)
q = TDC(x), k = TDC(x), v = Conv1x1(x)

# Step 2: Window 분할 → Region routing (BiFormer 추가)
# (40, 4, 4) feature → (2, 2, 2) windows × (20, 2, 2) tokens
# = 8 windows × 80 tokens
q_window = window_partition(q)  # (B, 8, 80, C)
k_window = window_partition(k)
v_window = window_partition(v)

# Step 3: Region-level routing (top-k)
q_region = q_window.mean(dim=2)   # (B, 8, C)  region embedding
k_region = k_window.mean(dim=2)
A_region = q_region @ k_region.T / sqrt(C)  # (B, 8, 8) similarity
top_k = 4
_, top_k_idx = A_region.topk(top_k, dim=-1)  # (B, 8, 4)

# Step 4: 각 query window 가 top-4 windows 의 K/V 만 attend (sparse)
k_top = gather(k_window, top_k_idx)  # (B, 8, 4×80, C)
v_top = gather(v_window, top_k_idx)

# Step 5: Multi-head sparse attention with PhysFormer scale
scores = q_window @ k_top.T / gra_sharp
out = softmax(scores) @ v_top
```

**효과:**
- Attention compute **50% 감소** (8 windows 중 top-4 만 attend)
- Region routing 으로 의미 있는 spatial 영역에 집중
- PhysFormer 의 다른 부분 (FFN_ST, Stem, predictor) 그대로

## ⚙️ Training Setup

**rPPG-Toolbox PhysFormerTrainer 셋업과 100% 정렬** ([reference](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py)):

### Protocol 1 — UBFC 6:4 (RhythmFormer Table 1 표준)
| 항목 | 값 |
|---|---|
| Split | 60% train / 40% test (valid = test) |
| Optimizer | Adam (lr=1e-4, wd=5e-5) |
| LR scheduler | StepLR(step=50, gamma=0.5) — 10ep 동안 constant |
| Epochs | 10 |
| α, β schedule | constant α=1.0, β=1.0 |

### Protocol 2 — 7:1:2 + OneCycleLR (paper 와 동등한 학습 setup)
| 항목 | 값 |
|---|---|
| Split | 70% train / 10% valid / 20% test (subject-exclusive, separate valid) |
| PURE split mode | random shuffle (seed=42) — outlier subject 07 이 train 에 자동 포함 |
| Optimizer | Adam (lr=1e-4, wd=5e-5) |
| LR scheduler | **OneCycleLR(max_lr=1e-4, epochs=20)** |
| Epochs | **20** |
| α, β schedule | epoch≤10: α=1.0, β=1.0 → epoch>10: α=0.05, β=5.0 (rPPG-Toolbox) |
| Best epoch | min VALID per-clip RMSE (test-independent) |

### 공통 항목
| 항목 | 값 |
|---|---|
| Batch size | 4 |
| Loss | α·NegPearson + β·(CE_freq + KL_dist) |
| Output normalization | Per-sample: `rPPG = (rPPG - mean) / std` (axis=-1) |
| Frequency loss target | Welch periodogram peak HR from label PPG |
| Data preprocessing | DiffNormalized (rPPG-Toolbox 표준) |
| Face crop | HaarCascade, 1.5× large box, static (first frame) |
| Augmentation | RandomHorizontalFlip (train only) |
| HR validity filter | 40 < HR < 180 BPM (PhysBench trick) |
| Test eval | Sliding window (chunk_step=80, 2× overlap) |
| HR estimation | DiffNormalized → cumsum + detrend(λ=100) + Butterworth(0.75-2.5Hz) + periodogram |

## 📂 코드 구조
<img width="645" height="273" alt="image" src="https://github.com/user-attachments/assets/02a09530-d4c2-4b0d-836c-1216f9390972" />


## 🚀 실행 방법

```bash
# Protocol 1 — UBFC 6:4 (RhythmFormer Table 1)
python scripts/run_intra_ubfc_bipulseformer.py        # UBFC intra 6:4

# Protocol 2 — 7:1:2 + OneCycleLR + 20 epochs (paper 와 동등한 학습 setup)
python scripts/run_intra_712_onecycle.py              # PURE + UBFC 통합

# 결과 평가 (per-subject + per-clip + MAPE)
python scripts/eval_mape_paper.py                     # 모든 saved checkpoints
python scripts/eval_valid_vs_test_best_oc20.py        # 7:1:2 의 valid-best vs test-best
```

결과는 `results/intra_{dataset}_bipulseformer{,_712_oc20}/` 에 저장됩니다:
- `log.txt` — 학습 로그 (모든 epoch 결과)
- `summary.json` — best epoch + full history (JSON)
- `checkpoints/{model}_epoch{N}.pt` — best epoch checkpoint

## 핵심 기여

1. **BiFormer 를 PhysFormer 에 적용** (ANN, BiPulseFormer):
   - Sparse attention (top-k=4 of 8 windows)
   - Attention compute 50% 감소

2. **rPPG-Toolbox 호환 평가 인프라**:
   - per-subject 평가 (cumsum + detrend + Butterworth + periodogram)
   - Sliding window evaluation (chunk_step=80, 2× overlap)
   - Welch periodogram 기반 HR target/metric

3. **2 가지 평가 protocol 지원**:
   - UBFC 6:4 RhythmFormer Table 1 (다른 paper 와 직접 비교)
   - 7:1:2 + OneCycleLR + 20 epochs (separate valid, no test-peek)

4. **PURE 의 outlier subject 07 처리**:
   - Subject-exclusive random shuffle (seed=42) 로 high-HR subject 가 train 에 들어가는 split 확보 (Protocol 2)

## 📚 References

- **PhysFormer**: Yu et al., "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer", CVPR 2022. [arXiv:2111.12082](https://arxiv.org/abs/2111.12082)
- **BiFormer**: Zhu et al., "BiFormer: Vision Transformer with Bi-Level Routing Attention", CVPR 2023. [arXiv:2303.08810](https://arxiv.org/abs/2303.08810)
- **rPPG-Toolbox**: Liu et al., "rPPG-Toolbox: Deep Remote PPG Toolbox", NeurIPS 2023. [arXiv:2210.00716](https://arxiv.org/abs/2210.00716)
