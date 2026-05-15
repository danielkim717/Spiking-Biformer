"""
PhysFormer (Yu et al., CVPR 2022) ??怨듭떇 GitHub 肄붾뱶??吏곸젒 ?ы똿.

?먮낯:
  https://github.com/ZitongYu/PhysFormer/blob/main/model/transformer_layer.py
  https://github.com/ZitongYu/PhysFormer/blob/main/model/physformer.py

?⑸룄:
  1) PE block (Stem0+Stem1+Stem2 + patch_embedding) ?ъ쟾?숈뒿??  2) BiPulseFormer (BiFormer ?곸슜) ???fair baseline 鍮꾧탳??
?먮낯 ?鍮?李⑥씠??import ? type-hint ?뺣━肉? 紐⑤뜽 援ъ“/init/forward ???숈씪.
"""
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CDC_T ??怨듭떇 肄붾뱶 洹몃?濡?# =============================================================================
class CDC_T(nn.Module):
    """Temporal Center-difference based 3D Convolution."""
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
# MultiHeadedSelfAttention_TDC_gra_sharp ??怨듭떇 肄붾뱶 洹몃?濡?# =============================================================================
class MultiHeadedSelfAttention_TDC_gra_sharp(nn.Module):
    """Multi-Headed Dot Product Attention with depth-wise Conv3d (PhysFormer)."""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Sequential(
            nn.Conv3d(dim, dim, 1, stride=1, padding=0, groups=1, bias=False),
        )
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None

    def forward(self, x, gra_sharp):
        B, P, C = x.shape
        x = x.transpose(1, 2).view(B, C, P // 16, 4, 4)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        q, k, v = (split_last(t_, (self.n_heads, -1)).transpose(1, 2) for t_ in [q, k, v])
        scores = q @ k.transpose(-2, -1) / gra_sharp
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


# =============================================================================
# PositionWiseFeedForward_ST ??怨듭떇 肄붾뱶 洹몃?濡?# =============================================================================
class PositionWiseFeedForward_ST(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.STConv = nn.Sequential(
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv3d(ff_dim, dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(dim),
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
# Block_ST_TDC_gra_sharp ??怨듭떇 肄붾뱶 洹몃?濡?# =============================================================================
class Block_ST_TDC_gra_sharp(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MultiHeadedSelfAttention_TDC_gra_sharp(dim, num_heads, dropout, theta)
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


class Transformer_ST_TDC_gra_sharp(nn.Module):
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block_ST_TDC_gra_sharp(dim, num_heads, ff_dim, dropout, theta)
            for _ in range(num_layers)
        ])

    def forward(self, x, gra_sharp):
        for block in self.blocks:
            x, Score = block(x, gra_sharp)
        return x, Score


# =============================================================================
# ViT_ST_ST_Compact3_TDC_gra_sharp ??怨듭떇 肄붾뱶 洹몃?濡?# =============================================================================
def _as_tuple(x):
    return x if isinstance(x, tuple) else (x, x, x)


class ViT_ST_ST_Compact3_TDC_gra_sharp(nn.Module):
    """PhysFormer 怨듭떇 紐⑤뜽 (CVPR 2022)."""

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
    ):
        super().__init__()
        self.image_size = image_size
        self.frame = frame
        self.dim = dim

        ft, fh, fw = patches if isinstance(patches, tuple) else (patches, patches, patches)
        self.patch_embedding = nn.Conv3d(dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw))

        self.transformer1 = Transformer_ST_TDC_gra_sharp(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
        )
        self.transformer2 = Transformer_ST_TDC_gra_sharp(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
        )
        self.transformer3 = Transformer_ST_TDC_gra_sharp(
            num_layers=num_layers // 3, dim=dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout_rate, theta=theta,
        )

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

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, gra_sharp=2.0):
        b, c, t, fh, fw = x.shape
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

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

    def export_pe_state_dict(self):
        """Stem0/1/2 + patch_embedding state_dict (PE pretraining ??."""
        sd = {}
        for prefix in ['Stem0', 'Stem1', 'Stem2', 'patch_embedding']:
            module = getattr(self, prefix)
            for k, v in module.state_dict().items():
                sd[f'{prefix}.{k}'] = v.clone().detach()
        return sd


# Backward-compat alias
PhysFormer = ViT_ST_ST_Compact3_TDC_gra_sharp
