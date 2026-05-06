"""
BiPhysFormer — PhysFormer (ANN) + BiLevel Routing Attention (Zhu et al., CVPR 2023).

목적:
  Spiking 효과를 제외한 상태에서 BiFormer 의 BiLevel Routing Attention 을
  PhysFormer 에 단독 적용했을 때의 기여도를 검증하기 위한 ablation 모델.

차이점 (vs. PhysFormer):
  - MHSA_TDC  →  BiLevelRoutingAttention_TDC  (region top-k routing + sparse attn)
  - 그 외 (Stem0/1/2, patch_embedding, TDC-Q/K, FFN_ST, predictor head, gra_sharp)
    는 PhysFormer 원본과 동일.

Reference:
  https://github.com/rayleizhu/BiFormer/blob/public_release/ops/bra_legacy.py
  Zhu et al., "BiFormer: Vision Transformer with Bi-Level Routing Attention",
  CVPR 2023.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Temporal Center-Difference Conv (PhysFormer 동일)
# -----------------------------------------------------------------------------
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
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
# BiLevel Routing Attention with TDC-Q/K (BiFormer paper, Zhu et al., CVPR 2023)
#
# Algorithm:
#   1) Q = TDC(x), K = TDC(x), V = Conv1x1(x)             # PhysFormer-style
#   2) Window partition: feature map → (wt × wh × ww) windows
#   3) Region routing:
#        q_r = mean_token(Q in each window)   # [B, S, C]
#        k_r = mean_token(K in each window)   # [B, S, C]
#        A_r = q_r @ k_r.T   →  top-k key-windows per query-window
#   4) Token-level sparse attention:
#        각 query-token 은 top-k 개 routed window 안의 token (k * win_size) 만
#        key/value 로 사용 (full attention 대비 sparsity)
#   5) Multi-head softmax (gra_sharp 적용, PhysFormer recipe)
# -----------------------------------------------------------------------------
class BiLevelRoutingAttention_TDC(nn.Module):
    def __init__(self, dim, num_heads, dropout, theta,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_win = n_win
        self.topk = topk

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop = nn.Dropout(dropout)

    def _window_partition(self, x, t, h, w):
        """[B, C, t, h, w] → [B, S, win, C], S = wt*wh*ww, win = lt*lh*lw."""
        B, C = x.shape[:2]
        wt, wh, ww = self.n_win
        lt, lh, lw = t // wt, h // wh, w // ww
        # reshape into (B, C, wt, lt, wh, lh, ww, lw)
        x = x.view(B, C, wt, lt, wh, lh, ww, lw)
        # permute → (B, wt, wh, ww, lt, lh, lw, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        S = wt * wh * ww
        win = lt * lh * lw
        return x.view(B, S, win, C), (lt, lh, lw)

    def _window_reverse(self, x, t, h, w, lt, lh, lw):
        """[B, S, win, C] → [B, C, t, h, w]."""
        B = x.shape[0]
        C = x.shape[-1]
        wt, wh, ww = self.n_win
        x = x.view(B, wt, wh, ww, lt, lh, lw, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()  # (B, C, wt, lt, wh, lh, ww, lw)
        return x.view(B, C, t, h, w)

    def forward(self, x, gra_sharp, t, h, w):
        # x: (B, P, C), P = t*h*w
        B, P, C = x.shape
        x_3d = x.transpose(1, 2).reshape(B, C, t, h, w)

        # Step 1: Q, K, V via PhysFormer projections
        q_3d = self.proj_q(x_3d)            # (B, C, t, h, w)
        k_3d = self.proj_k(x_3d)
        v_3d = self.proj_v(x_3d)

        # Step 2: window partition
        q_w, (lt, lh, lw) = self._window_partition(q_3d, t, h, w)   # [B, S, win, C]
        k_w, _ = self._window_partition(k_3d, t, h, w)
        v_w, _ = self._window_partition(v_3d, t, h, w)
        S = q_w.shape[1]
        win = q_w.shape[2]

        # Step 3: region routing
        q_r = q_w.mean(dim=2)               # [B, S, C]
        k_r = k_w.mean(dim=2)               # [B, S, C]
        a_r = q_r @ k_r.transpose(-2, -1) / math.sqrt(C)   # [B, S, S]
        topk = min(self.topk, S)
        _, topk_idx = torch.topk(a_r, k=topk, dim=-1)       # [B, S, topk]

        # Step 4: gather K, V from top-k routed windows for each query window
        # gather index: [B, S, topk] → expand to [B, S, topk, win, C]
        idx = topk_idx.view(B, S, topk, 1, 1).expand(B, S, topk, win, C)
        # k_w / v_w: [B, S, win, C] → expand source dim along query:
        #   source = [B, 1, S, win, C]   gather dim=2 → [B, S, topk, win, C]
        k_src = k_w.unsqueeze(1).expand(B, S, S, win, C)
        v_src = v_w.unsqueeze(1).expand(B, S, S, win, C)
        k_g = torch.gather(k_src, 2, idx)   # [B, S, topk, win, C]
        v_g = torch.gather(v_src, 2, idx)
        # flatten topk*win into a single key/value dim
        k_g = k_g.view(B, S, topk * win, C)
        v_g = v_g.view(B, S, topk * win, C)

        # Step 5: multi-head sparse attention within each query window
        H = self.n_heads
        d = self.head_dim
        # reshape per-head: [B, S, win/(topk*win), H, d]
        q_h = q_w.view(B, S, win, H, d).permute(0, 1, 3, 2, 4)            # [B, S, H, win, d]
        k_h = k_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)     # [B, S, H, topk*win, d]
        v_h = v_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)

        # PhysFormer recipe: scores = q@k^T / gra_sharp  (note: not /sqrt(d))
        scores = (q_h @ k_h.transpose(-2, -1)) / gra_sharp                # [B, S, H, win, topk*win]
        scores = self.drop(F.softmax(scores, dim=-1))
        out = scores @ v_h                                                # [B, S, H, win, d]

        # merge heads, then window-reverse to (B, C, t, h, w)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, win, C)  # [B, S, win, C]
        out_3d = self._window_reverse(out, t, h, w, lt, lh, lw)           # [B, C, t, h, w]
        out = out_3d.flatten(2).transpose(1, 2)                           # [B, P, C]
        return out


# -----------------------------------------------------------------------------
# Position-wise FeedForward with depthwise STConv (PhysFormer 동일)
# -----------------------------------------------------------------------------
class PositionWiseFeedForward_ST(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, kernel_size=3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
        )

    def forward(self, x, t, h, w):
        B, P, C = x.shape
        x_3d = x.transpose(1, 2).reshape(B, C, t, h, w)
        x_3d = self.fc1(x_3d)
        x_3d = self.STConv(x_3d)
        x_3d = self.fc2(x_3d)
        return x_3d.flatten(2).transpose(1, 2)


# -----------------------------------------------------------------------------
# BiFormer-style transformer block (BRA + FFN_ST)
# -----------------------------------------------------------------------------
class BiTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout, theta, n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.attn = BiLevelRoutingAttention_TDC(dim, num_heads, dropout, theta,
                                                n_win=n_win, topk=topk)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp, t, h, w):
        attn_out = self.attn(self.norm1(x), gra_sharp, t, h, w)
        x = x + self.drop(self.proj(attn_out))
        ff_out = self.pwff(self.norm2(x), t, h, w)
        x = x + self.drop(ff_out)
        return x


# -----------------------------------------------------------------------------
# BiPhysFormer (ANN, BiFormer 적용)
# -----------------------------------------------------------------------------
class BiPhysFormer(nn.Module):
    """PhysFormer + BiLevel Routing Attention (Spiking 미적용)."""
    def __init__(self, dim=96, ff_dim=144, num_heads=4, num_layers=12,
                 frame=160, image_size=128, dropout=0.1, theta=0.7,
                 patches=(4, 4, 4), n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.dim = dim
        self.frame = frame
        self.gra_sharp = 2.0

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        ft, fh, fw = patches
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        # 12 BiTransformer layers in 3 stages of 4 (PhysFormer 와 동일한 layer 수)
        layers_per_stage = num_layers // 3
        self.transformer1 = nn.ModuleList([
            BiTransformerBlock(dim, num_heads, ff_dim, dropout, theta, n_win=n_win, topk=topk)
            for _ in range(layers_per_stage)
        ])
        self.transformer2 = nn.ModuleList([
            BiTransformerBlock(dim, num_heads, ff_dim, dropout, theta, n_win=n_win, topk=topk)
            for _ in range(layers_per_stage)
        ])
        self.transformer3 = nn.ModuleList([
            BiTransformerBlock(dim, num_heads, ff_dim, dropout, theta, n_win=n_win, topk=topk)
            for _ in range(layers_per_stage)
        ])

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

    def forward(self, x):
        B = x.shape[0]
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)            # (B, dim, t, h, w)
        _, C, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, P, C)

        for blk in self.transformer1:
            x = blk(x, self.gra_sharp, t, h, w)
        for blk in self.transformer2:
            x = blk(x, self.gra_sharp, t, h, w)
        for blk in self.transformer3:
            x = blk(x, self.gra_sharp, t, h, w)

        features_last = x.transpose(1, 2).reshape(B, C, t, h, w)
        features_last = self.upsample(features_last)
        features_last = self.upsample2(features_last)
        features_last = torch.mean(features_last, dim=3)
        features_last = torch.mean(features_last, dim=3)
        rppg = self.ConvBlockLast(features_last).squeeze(1)
        return rppg

    def load_pretrained_pe(self, path):
        """PhysFormer 사전학습 PE block (Stem0/1/2 + patch_embedding) weight 로드."""
        sd = torch.load(path, map_location='cpu')
        own_sd = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded += 1
        self.load_state_dict(own_sd)
        print(f"[BiPhysFormer] Loaded pretrained PE: {loaded} tensors from {path}")
