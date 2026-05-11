"""
BiPhysFormer ??PhysFormer (CVPR 2022, Yu et al.) ??怨듭떇 援ы쁽??嫄곗쓽 洹몃?濡??ы똿?섎릺, MultiHeadedSelfAttention_TDC_gra_sharp 留?BiLevelRoutingAttention_TDC_gra_sharp 濡?援먯껜??ablation 紐⑤뜽.

?먮낯:
  https://github.com/ZitongYu/PhysFormer/blob/main/model/transformer_layer.py
  https://github.com/ZitongYu/PhysFormer/blob/main/model/physformer.py

李⑥씠??(PhysFormer ?먮낯 ?鍮?:
  - Block_ST_TDC_gra_sharp.attn  ?? BiLevelRoutingAttention_TDC_gra_sharp
  - 洹???(CDC_T, FFN_ST, Block ??norm/proj/pwff, Stem0/1/2,
    patch_embedding, transformer1/2/3, upsample, ConvBlockLast,
    init_weights, forward signature) 紐⑤몢 PhysFormer ?먮낯 洹몃?濡?

BiLevel Routing Attention (Zhu et al., CVPR 2023) ?곸슜 諛⑹떇:
  Q,K,V 異붿텧? ?숈씪 (TDC-Q, TDC-K, Conv1횞1-V), ??attention ?④퀎?먯꽌
    1) Q,K 瑜?window ?⑥쐞濡??됯퇏 ??q_region, k_region
    2) q_region @ k_region.T 濡?region similarity ??媛?query window 媛
       ?곸쐞 k 媛쒖쓽 key window 留?李몄“ (top-k routing)
    3) 洹?k횞win_size ?좏겙留?key/value 濡??ъ슜??multi-head softmax ?섑뻾
  PhysFormer ??gra_sharp (=2.0) scale ??洹몃?濡??좎??섏뿬 fair comparison.
"""
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CDC_T ??PhysFormer ?먮낯 洹몃?濡?# =============================================================================
class CDC_T(nn.Module):
    """Temporal Center-difference based 3D Convolution (CDC_T)."""
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


# =============================================================================
# split_last / merge_last ??PhysFormer ?먮낯 洹몃?濡?# =============================================================================
def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


# =============================================================================
# BiLevelRoutingAttention_TDC_gra_sharp
#   PhysFormer MHSA_TDC_gra_sharp ??drop-in replacement.
#   - ?낅젰/異쒕젰 shape, return signature, forward(x, gra_sharp) ?쒓렇?덉쿂
#     紐⑤몢 ?먮낯 MHSA ? ?숈씪.
#   - ?댄뀗???대?留?BiLevel Routing ?쇰줈 援먯껜.
# =============================================================================
class BiLevelRoutingAttention_TDC_gra_sharp(nn.Module):
    """BiFormer-style BRA, drop-in for PhysFormer MHSA_TDC_gra_sharp."""
    def __init__(self, dim, num_heads, dropout, theta,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim, track_running_stats=False),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim, track_running_stats=False),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.dim = dim
        self.n_win = n_win
        self.topk = topk
        self.scores = None  # for visualization (PhysFormer ?먮낯 ?명솚)

    def _window_partition(self, feat_3d, t, h, w):
        """feat_3d: (B, C, t, h, w) ??(B, S, win, C),  S=?n_win, win=?len."""
        B, C = feat_3d.shape[:2]
        wt, wh, ww = self.n_win
        lt, lh, lw = t // wt, h // wh, w // ww
        feat = feat_3d.view(B, C, wt, lt, wh, lh, ww, lw)
        feat = feat.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()  # (B, wt, wh, ww, lt, lh, lw, C)
        S = wt * wh * ww
        win = lt * lh * lw
        return feat.view(B, S, win, C), (lt, lh, lw)

    def _window_reverse(self, x_w, t, h, w, lt, lh, lw):
        """(B, S, win, C) ??(B, t*h*w, C)."""
        B = x_w.shape[0]
        C = x_w.shape[-1]
        wt, wh, ww = self.n_win
        x = x_w.view(B, wt, wh, ww, lt, lh, lw, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # (B, wt, lt, wh, lh, ww, lw, C)
        x = x.view(B, t * h * w, C)
        return x

    def forward(self, x, gra_sharp):
        """x: (B, P, C), P = t*h*w. PhysFormer ?몄텧 ?⑦꽩 洹몃?濡??ъ슜.
        Return: (h, scores) ??scores ??region routing 寃곌낵 (B, S, S) 濡?諛섑솚."""
        B, P, C = x.shape
        # PhysFormer ?먮낯? P = 16*t (h=w=4) 濡?reshape ???곕━???숈씪.
        x_3d = x.transpose(1, 2).view(B, C, P // 16, 4, 4)
        t, h, w = P // 16, 4, 4

        q_3d = self.proj_q(x_3d)
        k_3d = self.proj_k(x_3d)
        v_3d = self.proj_v(x_3d)

        # Window partition
        q_w, (lt, lh, lw) = self._window_partition(q_3d, t, h, w)   # (B, S, win, C)
        k_w, _ = self._window_partition(k_3d, t, h, w)
        v_w, _ = self._window_partition(v_3d, t, h, w)
        S = q_w.shape[1]
        win = q_w.shape[2]

        # Region routing
        q_r = q_w.mean(dim=2)                                       # (B, S, C)
        k_r = k_w.mean(dim=2)
        a_r = q_r @ k_r.transpose(-2, -1) / math.sqrt(C)            # (B, S, S)
        topk = min(self.topk, S)
        _, topk_idx = torch.topk(a_r, k=topk, dim=-1)               # (B, S, topk)

        # Gather K, V from top-k routed windows for each query window
        idx = topk_idx.view(B, S, topk, 1, 1).expand(B, S, topk, win, C)
        k_src = k_w.unsqueeze(1).expand(B, S, S, win, C)
        v_src = v_w.unsqueeze(1).expand(B, S, S, win, C)
        k_g = torch.gather(k_src, 2, idx).view(B, S, topk * win, C)  # (B, S, k*win, C)
        v_g = torch.gather(v_src, 2, idx).view(B, S, topk * win, C)

        # Multi-head sparse attention within each query window
        H = self.n_heads
        d = C // H
        q_h = q_w.view(B, S, win, H, d).permute(0, 1, 3, 2, 4)             # (B, S, H, win, d)
        k_h = k_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)      # (B, S, H, k*win, d)
        v_h = v_g.view(B, S, topk * win, H, d).permute(0, 1, 3, 2, 4)
        # PhysFormer recipe: scores = q @ k.T / gra_sharp  (NOT /sqrt(d))
        scores = (q_h @ k_h.transpose(-2, -1)) / gra_sharp                  # (B, S, H, win, k*win)
        scores = self.drop(F.softmax(scores, dim=-1))
        out = scores @ v_h                                                  # (B, S, H, win, d)

        # Merge heads + window-reverse
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, win, C)    # (B, S, win, C)
        h_out = self._window_reverse(out, t, h, w, lt, lh, lw)              # (B, P, C)
        # Score (region routing) for visualization compatibility
        self.scores = a_r
        return h_out, a_r


# =============================================================================
# PositionWiseFeedForward_ST ??PhysFormer ?먮낯 洹몃?濡?# =============================================================================
class PositionWiseFeedForward_ST(nn.Module):
    """1횞1 Conv ??BN ??ELU ??depthwise 3쨀 STConv ??BN ??ELU ??1횞1 Conv ??BN."""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim, track_running_stats=False),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim, track_running_stats=False),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim, track_running_stats=False),
        )

    def forward(self, x):
        B, P, C = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)
        x = self.fc1(x)
        x = self.STConv(x)
        x = self.fc2(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =============================================================================
# Block_ST_TDC_gra_sharp_Bi ??PhysFormer ?먮낯 Block ?먯꽌 attn 留?BRA 濡?援먯껜
# =============================================================================
class Block_ST_TDC_gra_sharp_Bi(nn.Module):
    """Transformer Block (BiLevel Routing Attention ?곸슜)."""
    def __init__(self, dim, num_heads, ff_dim, dropout, theta,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.attn = BiLevelRoutingAttention_TDC_gra_sharp(
            dim, num_heads, dropout, theta, n_win=n_win, topk=topk
        )
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_ST(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gra_sharp):
        Atten, Score = self.attn(self.norm1(x), gra_sharp)
        h = self.drop(self.proj(Atten))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x, Score


# =============================================================================
# Transformer_ST_TDC_gra_sharp_Bi ??PhysFormer ?먮낯 Transformer ?먯꽌
#   Block 留?Bi 濡?援먯껜
# =============================================================================
class Transformer_ST_TDC_gra_sharp_Bi(nn.Module):
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta,
                 n_win=(2, 2, 2), topk=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp_Bi(dim, num_heads, ff_dim, dropout, theta,
                                      n_win=n_win, topk=topk)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


# =============================================================================
# ViT_BiPhysFormer ??PhysFormer ?먮낯 ViT_ST_ST_Compact3_TDC_gra_sharp ?
#   Stem/PE/upsample/init_weights/forward ?꾨? ?숈씪.
#   transformer1/2/3 留?Bi 踰꾩쟾 ?ъ슜.
# =============================================================================
def _as_tuple(x):
    return x if isinstance(x, tuple) else (x, x, x)


class ViT_BiPhysFormer(nn.Module):
    """PhysFormer + BiLevel Routing Attention. ?숈뒿/forward ?명꽣?섏씠?ㅻ뒗
    ?먮낯 PhysFormer ? 100% ?숈씪 (forward(x, gra_sharp) ??rPPG, S1, S2, S3)."""

    def __init__(
        self,
        patches=(4, 4, 4),
        dim: int = 96,
        ff_dim: int = 144,
        num_heads: int = 4,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        in_channels: int = 3,
        frame: int = 160,
        theta: float = 0.7,
        image_size=(160, 128, 128),
        n_win=(2, 2, 2),
        topk: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.frame = frame
        self.dim = dim

        ft, fh, fw = patches if isinstance(patches, tuple) else (patches, patches, patches)
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        self.transformer1 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk,
        )
        self.transformer2 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk,
        )
        self.transformer3 = Transformer_ST_TDC_gra_sharp_Bi(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
            n_win=n_win, topk=topk,
        )

        self.Stem0 = nn.Sequential(
            nn.Conv3d(3, dim // 4, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(dim // 4, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem1 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim // 2, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )
        self.Stem2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim, track_running_stats=False),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(dim, dim // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(dim // 2, track_running_stats=False),
            nn.ELU(),
        )
        self.ConvBlockLast = nn.Conv1d(dim // 2, 1, 1, stride=1, padding=0)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        """PhysFormer ?먮낯: Linear xavier_uniform + bias normal_(std=1e-6)."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, gra_sharp=2.0):
        """x: (B, 3, T, H, W). Returns (rPPG, Score1, Score2, Score3) ??PhysFormer ?숈씪."""
        b, c, t, fh, fw = x.shape

        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)        # (B, dim, t/4, 4, 4)
        x = x.flatten(2).transpose(1, 2)   # (B, t/4*4*4, dim)

        T1, S1 = self.transformer1(x, gra_sharp)
        T2, S2 = self.transformer2(T1, gra_sharp)
        T3, S3 = self.transformer3(T2, gra_sharp)

        features_last = T3.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)
        features_last = self.upsample(features_last)
        features_last = self.upsample2(features_last)
        features_last = torch.mean(features_last, 3)
        features_last = torch.mean(features_last, 3)
        rPPG = self.ConvBlockLast(features_last).squeeze(1)
        return rPPG, S1, S2, S3

    def load_pretrained_pe(self, path):
        """Optional: load Stem0/1/2 + patch_embedding weights."""
        sd = torch.load(path, map_location='cpu')
        own_sd = self.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded += 1
        self.load_state_dict(own_sd)
        print(f"[BiPhysFormer] Loaded pretrained PE: {loaded} tensors from {path}")


# Backward-compat alias (older training scripts import BiPhysFormer)
BiPhysFormer = ViT_BiPhysFormer
