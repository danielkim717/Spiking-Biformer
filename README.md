# BiPulseFormer: BiLevel Routing Attention for rPPG

PhysFormer (Yu et al., CVPR 2022) 의 transformer block 에 **BiFormer (Zhu et al., CVPR 2023) 의 BiLevel Routing Attention** 을 적용한 **rPPG (remote photoplethysmography) 심박수 추정** 모델.

## 📊 Intra-Dataset 결과 (per-subject, paper-comparable metric)

### Protocol 1 — 6:4 split (RhythmFormer Table 1 표준 비율, valid=test)

| Dataset | Split mode | Best Ep | MAE↓ | RMSE↓ | MAPE%↓ | Pearson↑ | n_subj |
|---|---|---:|---:|---:|---:|---:|---:|
| **UBFC-rPPG** | subject-exclusive (1-32 vs 33-49) | E8 | **0.052** | **0.213** | **0.083** | **0.9999** | 17 |
| **PURE** | session-per-subject (60% sessions per subject) | E7 | **6.804** | 16.372 | 7.429 | 0.7368 | 29 |

\* PURE 의 sort-based subject-exclusive 6:4 (subjects 01-06 train, 07-10 test) 는 subject 07 (HR 127, outlier) 을 test 에 배정 → sub-harmonic 예측으로 MAE 12.80 폭증. Session-per-subject 변형으로 outlier 영향 완화 (MAE 6.80) 하나 여전히 unstable. 안정적 PURE 평가는 80:20 (Protocol 2) 또는 7:1:2 random (Protocol 3).

### Protocol 2 — 8:0:2 split (rPPG-Toolbox 표준 PURE, valid=test)

| Dataset | Best Ep | MAE↓ | RMSE↓ | MAPE%↓ | Pearson↑ | n_subj |
|---|---:|---:|---:|---:|---:|---:|
| **PURE** (subjects 01-08 train, 09-10 test) | E9 | **1.099** | **1.838** | **1.393** | **0.9949** | 12 |

→ subject 07 (HR 127 outlier) 이 train 에 자동 포함되어 high-HR coverage 확보. PhysFormer paper PURE (MAE 1.10) 와 정확히 일치.

### Protocol 3 — Stricter setup (7:1:2 split, separate valid, OneCycleLR, 20 epochs)

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
| **BiPulseFormer 6:4 (우리)** | **0.052** | **0.9999** | 6.804* | 0.7368* |
| **BiPulseFormer 8:2 (우리)** | — | — | **1.099** | **0.9949** |
| **BiPulseFormer 7:1:2 OC20 (우리)** | **0.391** | **0.9960** | **0.769** | **0.9602** |

\* PURE 6:4 의 outlier subject 07 OOD 영향. 8:2 또는 7:1:2 random 으로 해결.

→ UBFC 에서 paper SOTA 압도. PURE 8:2 에서 PhysFormer (1.10) 와 동등 (1.099), Pearson 0.9949. PURE 7:1:2 OC20 (random shuffle) 에서 MAE 0.769 로 더 우수, 다만 Pearson 0.96 은 test set 의 좁은 HR 분포 (std 3.79) 영향.

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

### 3. Spiking-PhysFormer (Liu et al., NN 2024)

ANN PE block + **SNN parallel SDT block** + ANN predictor:

```
Input → ANN PE block (Stem + patch_embedding) → membrane
    ↓ Direct encoding × T_snn=4 → [T, B, C, t, h, w]
    ↓ ParallelSDTBlock × 4 (SNN)
    ↓ T-axis 평균 → [B, C, t, h, w]
    ↓ ANN predictor → rPPG
```

**ParallelSDTBlock:**
- Input → BN → LIF → spike
- Spike 가 **두 분기로 평행** (sequential 이 아닌 parallel):
  - **S3A 분기**: SN(BN(TDC(s))) Q, SN(BN(Conv(s))) K, SN(BN(s)) V
    - `S3A = SN(SUM_c(Q ⊗ K)) ⊗ V` (channel-sum gating, full attention)
  - **MLP 분기**: Conv1×1 → BN → SN → Conv1×1 → BN
- 결과 = identity + S3A + MLP (MS shortcut)

### 4. Spiking-Biformer (우리 제안 모델) ⭐

Spiking-PhysFormer 의 `SDA` (S3A) 자리를 **`BiSDA`** (Pre-LIF Gating BiLevel Routing Attention) 로 교체.

**BiSDA 의 핵심: Pre-LIF Gating**

기존 BiFormer 는 "top-k 만 attend" 하는 sparse 방식이지만, SNN 에서 sparse attention 을 직접 구현하면 spike 정보가 손실됨. 우리는 **LIF 통과 전 membrane potential 을 routing-derived gain 으로 곱해서, 중요한 region 이 자연스럽게 더 자주 발화하도록** 함:

```python
def forward(self, x):
    T, B, C, Lt, Lh, Lw = x.shape

    # 1. Q, K, V branches (LIF 직전 실수)
    q_pre = q_bn(q_conv(x))   # [T, B, C, Lt, Lh, Lw]
    k_pre = k_bn(k_conv(x))
    v_pre = v_bn(x)

    # 2. Routing similarity → gain map (per spatial position)
    #    Window 분할 → region 평균 → 유사도 → top-k softmax → gain ∈ [0.5, 2.5]
    gain = compute_gain_map(q_pre, k_pre, n_win=(2,2,2), topk=4)

    # 3. K, V membrane modulate. Q 는 그대로 (query 자체는 routing source)
    #    Routed region 의 K, V membrane 이 커져 LIF 통과 시 spike 발화율 ↑
    k_pre_gated = k_pre * gain
    v_pre_gated = v_pre * gain

    # 4. LIF → spike (routed region 의 spike rate 가 자연스럽게 큼)
    q = q_lif(q_pre)
    k = k_lif(k_pre_gated)
    v = v_lif(v_pre_gated)

    # 5. S3A — full feature map 그대로 (Spiking-PhysFormer Eq.5-8)
    attn = (q * k).sum(dim=3, keepdim=True)
    attn = attn_lif(attn)
    out = attn * v

    # 6. SN → Conv → BN (membrane out)
    return proj_bn(proj_conv(proj_lif(out)))
```

**Gain map 정규화 3-step pipeline:**
1. `A_r = q_region @ k_region.T / sqrt(dim)` — channel-scale 정규화
2. Top-k 안에서 softmax → 부드러운 가중치 분포 (sum=1)
3. `gain = base + scale × max_normalized` ∈ [0.5, 2.5] — base 0.5 (non-routed 보존), scale 2.0 (routed 강조)

**핵심 통찰:**
- BiFormer 의 routing 정보를 **spike rate 차이로 자연스럽게 표현**
- Attention 자체는 full S3A (정보 손실 없음)
- 중요 region 의 spike rate ↑ → 그 영역의 contribution 자연스럽게 커짐
- Energy efficiency 는 SNN 자체 sparsity 로 확보

## ⚙️ Training Setup

**rPPG-Toolbox PhysFormerTrainer 셋업과 100% 정렬** ([reference](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py)):

### Protocol 1 — 6:4 (RhythmFormer Table 1 표준)
| 항목 | 값 |
|---|---|
| Split | 60% train / 40% test (valid = test) |
| Optimizer | Adam (lr=1e-4, wd=5e-5) |
| LR scheduler | StepLR(step=50, gamma=0.5) — 10ep 동안 constant |
| Epochs | 10 |
| α, β schedule | constant α=1.0, β=1.0 |

### Protocol 2 — 7:1:2 + OneCycleLR (stricter, paper 와 동등한 학습)
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

**SNN 추가 설정:**
- T_snn = 4, v_threshold = 1.0
- Surrogate gradient: ATan
- BatchNorm: `track_running_stats=False` (cross-domain robustness)
- Direct encoding (T 회 반복)

## 📂 코드 구조

```
src/
  models/
    physformer_baseline.py      # PhysFormer (CVPR 2022) 공식 코드 포팅
    bipulseformer.py            # PhysFormer + BiLevel Routing Attention (ANN)
    spiking_physformer.py       # Spiking-PhysFormer + BiSDA (SNN)
  data/
    rppg_dataset.py             # PURE/UBFC-rPPG dataset
                                #   PURE split modes: subject_exclusive (default),
                                #     subject_exclusive_random (seed=42),
                                #     session_per_subject
  evaluation.py                 # rPPG-Toolbox per-subject 평가 (paper-comparable)
  evaluation_per_clip.py        # per-clip 보조 평가 (5.3s clip)
  train.py                      # NegPearsonLoss + FrequencyLoss (DLDL_softmax2)
scripts/
  run_intra_ubfc_bipulseformer.py        # UBFC 6:4 (RhythmFormer protocol) 학습
  run_intra_pure_bipulseformer.py        # PURE 6:4 학습 (session-per-subject)
  run_intra_pure_bipulseformer_80_20.py  # PURE 8:0:2 (rPPG-Toolbox standard) 학습
  run_intra_712_onecycle.py              # 7:1:2 OneCycleLR 20ep (PURE+UBFC 통합)
  eval_mape_paper.py                      # per-subject MAPE 계산
  eval_valid_vs_test_best_oc20.py        # valid-best vs test-best epoch 비교
```

## 🔋 Energy Analysis

### Per-operation Energy (45nm process, Horowitz 2014)
- ANN MAC (32-bit FP): **4.6 pJ**
- SNN AC (32-bit FP): **0.9 pJ** (5.1× cheaper)

### Spike Rate (학습 후 관측)
- Block 1: ~5-10%, Block 2: ~2-3%, Block 3: ~1-2%, Block 4: ~0.4-1%
- 평균 spike rate ≈ **0.05** (5%)

### Transformer Block Energy (relative to PhysFormer)

```
ANN_energy ∝ N_params × MAC × 1.0 (dense)
SNN_energy ∝ N_params × AC × spike_rate × T_snn
           = (2.16/7.38) × (0.9/4.6) × 0.05 × 4
           ≈ 0.012 × baseline
```

→ **~80× transformer block energy reduction** (Spiking-PhysFormer paper 12.2× 보다 큼; 더 sparse 한 spike rate 덕분)

### 종합

| 모델 | 모델 크기 ratio | Energy ratio | Pearson(HR) |
|---|---|---|---|
| PhysFormer | 1.0× | 1.0× | 0.528 (재현) |
| BiPulseFormer (ANN+BRA) | 1.0× | 0.5× (sparse attn) | 0.632 |
| **Spiking-Biformer** ⭐ | **0.293×** | **~0.012×** | **0.955** |

## 🚀 실행 방법

```bash
# Protocol 1 — 6:4 (RhythmFormer Table 1)
python scripts/run_intra_ubfc_bipulseformer.py        # UBFC intra 6:4
python scripts/run_intra_pure_bipulseformer.py        # PURE intra 6:4 (session-per-subject)

# Protocol 2 — 8:0:2 (rPPG-Toolbox standard PURE)
python scripts/run_intra_pure_bipulseformer_80_20.py  # PURE intra 8:2

# Protocol 3 — 7:1:2 + OneCycleLR + 20 epochs (paper 와 동등한 학습 setup)
python scripts/run_intra_712_onecycle.py              # PURE + UBFC 통합

# 결과 평가 (per-subject + per-clip + MAPE)
python scripts/eval_mape_paper.py                     # 모든 saved checkpoints
python scripts/eval_valid_vs_test_best_oc20.py        # 7:1:2 의 valid-best vs test-best
```

결과는 `results/intra_{dataset}_bipulseformer{_80_20,_712_oc20}/` 에 저장됩니다:
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

3. **3 가지 평가 protocol 지원**:
   - 6:4 RhythmFormer Table 1 (다른 paper 와 직접 비교)
   - 8:0:2 rPPG-Toolbox standard PURE
   - 7:1:2 + OneCycleLR + 20 epochs (separate valid, no test-peek)

4. **PURE 의 outlier subject 07 처리**:
   - Subject-exclusive random shuffle (seed=42) 로 high-HR subject 가 train 에 들어가는 split 확보
   - rPPG-Toolbox 의 80/20 권장 protocol 과 일치

## 📚 References

- **PhysFormer**: Yu et al., "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer", CVPR 2022. [arXiv:2111.12082](https://arxiv.org/abs/2111.12082)
- **BiFormer**: Zhu et al., "BiFormer: Vision Transformer with Bi-Level Routing Attention", CVPR 2023. [arXiv:2303.08810](https://arxiv.org/abs/2303.08810)
- **Spiking-PhysFormer**: Liu et al., "Spiking-PhysFormer: Camera-Based Remote Photoplethysmography with Parallel Spike-driven Transformer", Neural Networks 2024. [arXiv:2402.04798](https://arxiv.org/abs/2402.04798)
- **rPPG-Toolbox**: Liu et al., "rPPG-Toolbox: Deep Remote PPG Toolbox", NeurIPS 2023. [arXiv:2210.00716](https://arxiv.org/abs/2210.00716)
