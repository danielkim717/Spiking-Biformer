"""
PhysFormer (Yu et al., CVPR 2022) ANN baseline 구현.

핵심: Spiking-PhysFormer 의 PE block (Stem0+Stem1+Stem2+patch_embedding)
초기 가중치를 사전학습으로 얻기 위해 사용.

아키텍처:
  - Stem0/1/2 (3D Conv + BN + ReLU + MaxPool)
  - patch_embedding (4,4,4) Conv3d
  - 12 transformer layer (3 stage × 4)
    Block: LayerNorm → MHSA_TDC(gra_sharp) → +res → LayerNorm → PositionWiseFFN_ST → +res
    MHSA_TDC: Q=CDC_T, K=CDC_T, V=Conv1x1, scaled-dot-product attention with gra_sharp
    PositionWiseFFN_ST: Conv1x1 → BN → ELU → depthwise 3×3×3 STConv → BN → ELU → Conv1x1 → BN
  - Predictor head (Upsample×2 + ConvBN + ELU + GAP + Conv1d)

Reference:
  https://github.com/ZitongYu/PhysFormer/blob/main/model/Physformer.py
  https://github.com/ZitongYu/PhysFormer/blob/main/model/transformer_layer.py
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _split_last(x, shape):
    shape = list(shape)
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def _merge_last(x, n_dims):
    s = x.size()
    return x.view(*s[:-n_dims], -1)


class MHSA_TDC(nn.Module):
    """Multi-head Self-Attention with TDC-Q, TDC-K, Conv1x1-V."""
    def __init__(self, dim, num_heads, dropout, theta):
        super().__init__()
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
        self.n_heads = num_heads

    def forward(self, x, gra_sharp, t, h, w):
        # x: (B, P, C)  →  reshape to (B, C, t, h, w)
        B, P, C = x.shape
        x_3d = x.transpose(1, 2).reshape(B, C, t, h, w)
        q = self.proj_q(x_3d).flatten(2).transpose(1, 2)   # (B, P, C)
        k = self.proj_k(x_3d).flatten(2).transpose(1, 2)
        v = self.proj_v(x_3d).flatten(2).transpose(1, 2)
        q, k, v = (_split_last(t_, (self.n_heads, -1)).transpose(1, 2) for t_ in (q, k, v))
        scores = q @ k.transpose(-2, -1) / gra_sharp
        scores = self.drop(F.softmax(scores, dim=-1))
        h_out = (scores @ v).transpose(1, 2).contiguous()
        return _merge_last(h_out, 2)   # (B, P, C)


class PositionWiseFeedForward_ST(nn.Module):
    """FFN with depth-wise 3x3x3 STConv between two 1x1 projections."""
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout, theta):
        super().__init__()
        self.attn = MHSA_TDC(dim, num_heads, dropout, theta)
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


class PhysFormer(nn.Module):
    """PhysFormer ANN baseline (CVPR 2022)."""
    def __init__(self, dim=96, ff_dim=144, num_heads=4, num_layers=12,
                 frame=160, image_size=128, dropout=0.1, theta=0.7,
                 patches=(4, 4, 4)):
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

        # 12 transformer layers in 3 stages of 4
        layers_per_stage = num_layers // 3
        self.transformer1 = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout, theta)
            for _ in range(layers_per_stage)
        ])
        self.transformer2 = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout, theta)
            for _ in range(layers_per_stage)
        ])
        self.transformer3 = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout, theta)
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

    def export_pe_state_dict(self):
        """Return state_dict of Stem0+Stem1+Stem2+patch_embedding (PE block).
        Used to transfer pretrained weights to SpikingPhysformer."""
        sd = {}
        for prefix in ['Stem0', 'Stem1', 'Stem2', 'patch_embedding']:
            module = getattr(self, prefix)
            for k, v in module.state_dict().items():
                sd[f'{prefix}.{k}'] = v.clone().detach()
        return sd
