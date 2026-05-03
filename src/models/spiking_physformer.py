"""
Spiking-PhysFormer 공식 논문 구조의 직접 구현 (Liu et al., 2024, arXiv:2402.04798).

논문 핵심:
- ANN Patch Embedding (PhysFormer 의 Stem0+Stem1+Stem2 + patch_embedding)
- 4 개 parallel spike-driven transformer block
  - 입력 spike → S3A 경로 + MLP 경로 (병렬) → MS shortcut
  - S3A: Q = LIF(BN(TDC(S))), K = LIF(BN(Conv3D(S))), V = LIF(BN(S))
         attention = sum_c(Q ⊙ K) ⊙ V   (element-wise mask + column sum + column mask)
- T_snn = 4, dim = 96, v_threshold = 1, **v_reset = 0 (hard reset)**, surrogate = ATan
- ANN Predictor Head (PhysFormer 의 upsample×2 + spatial GAP + Conv1d)
- Biformer routing 은 사용하지 않음 (사용자 지시)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional


# -----------------------------------------------------------------------------
# Temporal Center-Difference Conv (PhysFormer 의 CDC_T)
# -----------------------------------------------------------------------------
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        if self.conv.weight.shape[2] > 1:
            kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + \
                          self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
        return out_normal


# -----------------------------------------------------------------------------
# Multi-step wrapper: applies an ANN module per-timestep on [T, B, ...] input
# -----------------------------------------------------------------------------
class _MS(nn.Module):
    """Wrap a 4D-conv/BN module to operate on [T, B, C, ...] by reshape."""
    def __init__(self, module):
        super().__init__()
        self.m = module

    def forward(self, x):
        T = x.shape[0]
        rest = x.shape[1:]
        y = self.m(x.reshape((T * rest[0],) + tuple(rest[1:])))
        return y.reshape((T,) + (rest[0],) + y.shape[1:])


# -----------------------------------------------------------------------------
# Spike-Driven Self Attention (S3A) — Spiking Physformer / Spike-Driven Tx V2
# -----------------------------------------------------------------------------
class SDA(nn.Module):
    """Spiking-PhysFormer S3A (paper Eq. 5-8):
        Q = SN(BN(TDC(S))),  K = SN(BN(Conv3D(S))),  V = SN(BN(S))
        S3A'(Q,K,V) = SN(SUM_c(Q ⊗ K)) ⊗ V
        S3A(Q,K,V)  = BN(Conv(SN(S3A'(Q,K,V))))    ← 출력은 membrane (마지막 LIF 없음)
    """
    def __init__(self, dim, num_heads=4, theta=0.7, v_threshold=1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q: TDC + BN + SN
        self.q_conv = _MS(CDC_T(dim, dim, kernel_size=3, padding=1, bias=False, theta=theta))
        self.q_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.q_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)

        # K: vanilla Conv3D + BN + SN
        self.k_conv = _MS(nn.Conv3d(dim, dim, kernel_size=1, bias=False))
        self.k_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.k_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)

        # V: BN + SN (no conv)
        self.v_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.v_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)

        # Inner-attention SN: SN(SUM_c(Q⊗K))   (Eq. 6 의 g(Q,K))
        self.attn_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                       surrogate_function=surrogate.ATan(), detach_reset=True)

        # Output projection: SN(·) → Conv → BN  (Eq. 8). 출력은 BN, 즉 membrane.
        self.proj_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                       surrogate_function=surrogate.ATan(), detach_reset=True)
        self.proj_conv = _MS(nn.Conv3d(dim, dim, kernel_size=1, bias=False))
        self.proj_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """x: [T, B, C, Lt, Lh, Lw]   (spike input from previous block's LIF)"""
        T, B, C, Lt, Lh, Lw = x.shape

        q = self.q_lif(self.q_bn(self.q_conv(x)))
        k = self.k_lif(self.k_bn(self.k_conv(x)))
        v = self.v_lif(self.v_bn(x))

        # Reshape with heads: [T, B, H, head_dim, Lt*Lh*Lw]
        N = Lt * Lh * Lw
        H = self.num_heads
        d = self.head_dim
        q = q.reshape(T, B, H, d, N)
        k = k.reshape(T, B, H, d, N)
        v = v.reshape(T, B, H, d, N)

        # S3A' = SN(SUM_c(Q⊗K)) ⊗ V
        attn = (q * k).sum(dim=3, keepdim=True)   # [T, B, H, 1, N]
        attn = self.attn_lif(attn)                 # SN(SUM_c(Q⊗K))
        out = attn * v                             # [T, B, H, head_dim, N] (broadcast)
        out = out.reshape(T, B, C, Lt, Lh, Lw)

        # S3A = BN(Conv(SN(out)))  — 마지막은 membrane (LIF 없음)
        out = self.proj_lif(out)
        out = self.proj_bn(self.proj_conv(out))
        return out


# -----------------------------------------------------------------------------
# BiLevel Routing Spike-Driven Attention (BiSDA) — Option A 통합
#   - Spiking-PhysFormer S3A 의 channel-sum gating 식 (`SN(SUM_c(Q⊙K_agg)) ⊙ V_agg`)
#     을 그대로 유지 → linear complexity, 토큰간 표준 attention 없음.
#   - Biformer routing 은 K, V 를 query window 별 top-k routed window 들의
#     position-wise 평균으로 만들어 routing 정보를 전달.
#   - 즉 attention 식은 SDA 와 동일, K/V 의 source 만 routed-aggregated.
# -----------------------------------------------------------------------------
class BiSDA(nn.Module):
    def __init__(self, dim, num_heads=4, theta=0.7, v_threshold=1.0,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_win = n_win
        self.topk = topk

        self.q_conv = _MS(CDC_T(dim, dim, kernel_size=3, padding=1, bias=False, theta=theta))
        self.q_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.q_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)
        self.k_conv = _MS(nn.Conv3d(dim, dim, kernel_size=1, bias=False))
        self.k_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.k_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)
        self.v_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.v_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                    surrogate_function=surrogate.ATan(), detach_reset=True)
        # Inner-attention SN
        self.attn_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                       surrogate_function=surrogate.ATan(), detach_reset=True)
        # Output: SN → Conv → BN  (membrane out)
        self.proj_lif = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                       surrogate_function=surrogate.ATan(), detach_reset=True)
        self.proj_conv = _MS(nn.Conv3d(dim, dim, kernel_size=1, bias=False))
        self.proj_bn = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.attn_scale = self.head_dim ** -0.5
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """x: [T, B, C, Lt, Lh, Lw]"""
        T, B, C, Lt, Lh, Lw = x.shape
        wt, wh, ww = self.n_win

        # 0. padding (안전망)
        pad_t = (wt - Lt % wt) % wt
        pad_h = (wh - Lh % wh) % wh
        pad_w = (ww - Lw % ww) % ww
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
            _, _, _, Lt, Lh, Lw = x.shape

        # 1. Q, K (실수, pre-LIF) — routing 신호
        q_real = self.q_bn(self.q_conv(x))
        k_real = self.k_bn(self.k_conv(x))

        # 2. 윈도우 분할 후 region 임베딩 (실수)
        lt, lh, lw = Lt // wt, Lh // wh, Lw // ww
        win_size = lt * lh * lw
        num_wins = wt * wh * ww

        def to_wins(t):  # [T,B,C,Lt,Lh,Lw] → [T,B,num_wins,win_size,C]
            t = t.view(T, B, C, wt, lt, wh, lh, ww, lw)
            t = t.permute(0, 1, 3, 5, 7, 4, 6, 8, 2).contiguous()
            t = t.view(T, B, num_wins, win_size, C)
            return t

        q_real_w = to_wins(q_real)
        k_real_w = to_wins(k_real)
        q_region = q_real_w.mean(dim=(0, 3))  # [B, num_wins, C]
        k_region = k_real_w.mean(dim=(0, 3))
        a_r = (q_region @ k_region.transpose(-2, -1)) * self.attn_scale
        topk = min(self.topk, num_wins)
        routing_indices = torch.topk(a_r, k=topk, dim=-1).indices   # [B, num_wins, topk]

        # 3. Spike Q, K, V
        q = self.q_lif(q_real)
        k = self.k_lif(k_real)
        v = self.v_lif(self.v_bn(x))
        q_w = to_wins(q)
        k_w = to_wins(k)
        v_w = to_wins(v)
        # [T, B, num_wins, win_size, C]

        # 4. Top-k K, V gather: 각 query window 마다 topk routed window 의 K,V 모음
        # k_src: [T, B, num_wins (query), num_wins (key), win_size, C]
        k_src = k_w.unsqueeze(2).expand(T, B, num_wins, num_wins, win_size, C)
        v_src = v_w.unsqueeze(2).expand(T, B, num_wins, num_wins, win_size, C)
        idx = routing_indices.view(1, B, num_wins, topk, 1, 1).expand(
            T, B, num_wins, topk, win_size, C)
        # [T, B, num_wins, topk, win_size, C] — query window 마다 topk routed window 의 토큰
        k_routed = torch.gather(k_src, dim=3, index=idx)
        v_routed = torch.gather(v_src, dim=3, index=idx)

        # 5. Position-wise aggregation across routed windows.
        #    각 intra-window position p 에서 routed top-k window 의 K,V 를 평균:
        #      K_agg[t,b,w,p] = mean_k k_routed[t,b,w,k,p]   ← [T, B, num_wins, win_size, C]
        k_agg = k_routed.mean(dim=3)
        v_agg = v_routed.mean(dim=3)

        # 6. S3A channel-sum gating (SDA 와 동일 식, 다만 K,V 가 routed-aggregated):
        #      attn[p] = SN(SUM_c(Q[p] ⊙ K_agg[p]))     scalar per token
        #      out[p]  = attn[p] ⊙ V_agg[p]              token-wise mask, 토큰간 상호작용 없음
        H = self.num_heads
        d = self.head_dim
        q_h = q_w.view(T, B, num_wins, win_size, H, d)
        k_h = k_agg.view(T, B, num_wins, win_size, H, d)
        v_h = v_agg.view(T, B, num_wins, win_size, H, d)

        attn = (q_h * k_h).sum(dim=5, keepdim=True)   # SUM_d, [T, B, W, win, H, 1]
        attn = self.attn_lif(attn)                     # SN(SUM_c(Q ⊙ K_agg))
        out = attn * v_h                               # [T, B, W, win, H, d]
        out = out.reshape(T, B, num_wins, win_size, C)

        # 6. Reverse window partition → [T, B, C, Lt, Lh, Lw]
        out = out.view(T, B, wt, wh, ww, lt, lh, lw, C)
        out = out.permute(0, 1, 8, 2, 5, 3, 6, 4, 7).contiguous()
        out = out.view(T, B, C, Lt, Lh, Lw)

        if pad_t or pad_h or pad_w:
            out = out[:, :, :, :Lt - pad_t, :Lh - pad_h, :Lw - pad_w]

        # 7. SN(out) → Conv → BN  (membrane out, 마지막 LIF 없음)
        out = self.proj_lif(out)
        out = self.proj_bn(self.proj_conv(out))
        return out


# -----------------------------------------------------------------------------
# MLP block — Spike-Driven V2 / Spiking-PhysFormer parallel block 의 MLP 분기.
#   입력: spike (이전 block 의 lif_in 출력)
#   출력: membrane (BN 출력) — paper Eq.9 의 잔차 합산을 위해 LIF 없이 BN 으로 끝.
# -----------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, v_threshold=1.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = _MS(nn.Conv3d(dim, hidden_dim, kernel_size=1, bias=False))
        self.bn1 = _MS(nn.BatchNorm3d(hidden_dim, track_running_stats=False))
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                   surrogate_function=surrogate.ATan(), detach_reset=True)
        self.fc2 = _MS(nn.Conv3d(hidden_dim, dim, kernel_size=1, bias=False))
        self.bn2 = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # Conv → BN → SN → Conv → BN  (출력 membrane)
        x = self.lif1(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


# -----------------------------------------------------------------------------
# Parallel Spike-Driven Transformer Block
# -----------------------------------------------------------------------------
class ParallelSDTBlock(nn.Module):
    """y = x + S3A(BN(LIF(x))) + MLP(BN(LIF(x)))   (parallel, MS shortcut)"""

    def __init__(self, dim, num_heads=4, theta=0.7, v_threshold=1.0):
        super().__init__()
        # 입력을 spike 로 변환
        self.bn_in = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.lif_in = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                     surrogate_function=surrogate.ATan(), detach_reset=True)
        self.sda = SDA(dim, num_heads=num_heads, theta=theta, v_threshold=v_threshold)
        self.mlp = MLPBlock(dim, hidden_dim=dim * 4, v_threshold=v_threshold)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        s = self.lif_in(self.bn_in(x))
        # Firing-rate trace (디버깅): lif_in 출력의 평균 발화율
        with torch.no_grad():
            self.last_firing_rate = float(s.mean().item())
        sda_out = self.sda(s)
        mlp_out = self.mlp(s)
        # Parallel MS shortcut
        return identity + sda_out + mlp_out


# -----------------------------------------------------------------------------
# Parallel Spike-Driven Transformer Block — Biformer 변형
# -----------------------------------------------------------------------------
class ParallelBiSDTBlock(nn.Module):
    """Spiking-PhysFormer 의 SDA 자리에 BiSDA (BiLevel Routing 적용) 사용."""
    def __init__(self, dim, num_heads=4, theta=0.7, v_threshold=1.0,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.bn_in = _MS(nn.BatchNorm3d(dim, track_running_stats=False))
        self.lif_in = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0,
                                     surrogate_function=surrogate.ATan(), detach_reset=True)
        self.sda = BiSDA(dim, num_heads=num_heads, theta=theta, v_threshold=v_threshold,
                         n_win=n_win, topk=topk)
        self.mlp = MLPBlock(dim, hidden_dim=dim * 4, v_threshold=v_threshold)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        s = self.lif_in(self.bn_in(x))
        with torch.no_grad():
            self.last_firing_rate = float(s.mean().item())
        return identity + self.sda(s) + self.mlp(s)


# -----------------------------------------------------------------------------
# Spiking-PhysFormer (no Biformer)
# -----------------------------------------------------------------------------
class SpikingPhysformer(nn.Module):
    def __init__(self, dim=96, num_blocks=4, num_heads=4, frame=160, image_size=128,
                 v_threshold=1.0, T_snn=4, theta=0.7,
                 use_biformer=False, n_win=(2, 2, 2), topk=4,
                 pretrained_pe_path=None):
        super().__init__()
        self.dim = dim
        self.frame = frame
        self.T_snn = T_snn

        # ---- ANN Patch Embedding (PhysFormer 의 Stem0+Stem1+Stem2 + patch_embedding) ----
        # ANN BN 은 PhysFormer default 그대로 (track_running_stats=True).
        # 이전엔 SNN 호환 위해 False 로 두었으나, ANN 영역까지 적용하면 eval 시
        # batch_size=2 mini-batch 통계 사용으로 cross-dataset 평가가 불안정해짐 →
        # PhysFormer 와 동일하게 running stats 사용.
        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, kernel_size=[1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),  # 128 -> 64
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=[3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),  # 64 -> 32
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=[3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),  # 32 -> 16
        )
        # Patch embed: (4, 4, 4) stride → 160→40 temporal, 16→4 spatial
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(4, 4, 4), stride=(4, 4, 4))

        # ---- Parallel SDT blocks ----
        if use_biformer:
            self.blocks = nn.ModuleList([
                ParallelBiSDTBlock(dim, num_heads=num_heads, theta=theta,
                                   v_threshold=v_threshold, n_win=n_win, topk=topk)
                for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                ParallelSDTBlock(dim, num_heads=num_heads, theta=theta, v_threshold=v_threshold)
                for _ in range(num_blocks)
            ])

        # ---- ANN Predictor Head (PhysFormer 와 동일, BN running stats 사용) ----
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2),
            nn.ELU(),
        )
        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        # 모든 spikingjelly 모듈을 multi-step 으로 설정
        functional.set_step_mode(self, step_mode='m')
        self._init_weights()

        # 사전학습된 PE block weight 로딩 (paper 의 PhysFormer pretraining 적용)
        if pretrained_pe_path is not None:
            self.load_pretrained_pe(pretrained_pe_path)

    def load_pretrained_pe(self, path):
        """PhysFormer pretrain 으로 학습된 Stem0/1/2 + patch_embedding weight 로드.

        SpikingPhysformer 의 PE block 구조와 PhysFormer baseline 의 PE block 구조가
        완전히 동일해야 한다 (BN track_running_stats=True 포함).
        """
        sd = torch.load(path, map_location='cpu')
        own_sd = self.state_dict()
        loaded, missing = 0, []
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded += 1
            else:
                missing.append(k)
        self.load_state_dict(own_sd)
        print(f"[SpikingPhysformer] Loaded pretrained PE block: {loaded} tensors from {path}")
        if missing:
            print(f"  missing/shape-mismatch: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: [B, 3, T_video, H, W]"""
        B = x.shape[0]

        # 1. ANN patch embedding -> [B, dim, 40, 4, 4]
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)

        # 2. Direct encoding: T_snn 만큼 반복 → [T, B, dim, Lt, Lh, Lw]
        x = x.unsqueeze(0).expand(self.T_snn, -1, -1, -1, -1, -1).contiguous()

        # 3. SDT blocks
        for block in self.blocks:
            x = block(x)
        # 각 block lif_in 의 firing rate 수집 (forward pass 직후의 last_firing_rate)
        self.last_firing_rates = [
            getattr(b, 'last_firing_rate', 0.0) for b in self.blocks
        ]

        # 4. T 평균 → [B, dim, 40, 4, 4]
        x = x.mean(dim=0)

        # 5. ANN predictor head
        x = self.upsample(x)        # [B, dim, 80, 4, 4]
        x = self.upsample2(x)       # [B, dim/2, 160, 4, 4]
        x = torch.mean(x, dim=3)    # [B, dim/2, 160, 4]
        x = torch.mean(x, dim=3)    # [B, dim/2, 160]
        rppg = self.ConvBlockLast(x).squeeze(1)  # [B, 160]
        return rppg
