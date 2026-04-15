"""
Spiking Bi-Physformer: rPPG 신호 추출을 위한 SNN 기반 Physformer + Biformer 아키텍처

입력: 비디오 클립 [B, T, C, H, W]
출력: rPPG 신호 [B, T] (BVP waveform)

Note: LIF 뉴런만 spikingjelly에서 사용, 나머지는 PyTorch 기본 레이어.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional
from .spiking_biformer import SpikingBiLevelRoutingAttention, SpikingMLP


class SpikingPatchEmbed3D(nn.Module):
    """
    시공간(Spatio-Temporal) Patch Embedding
    비디오 입력 [T, B, C, H, W] → 토큰 [T, B, N, D]
    시간축(T)을 배치에 합쳐 Conv2d를 통과시킨 후 다시 분리합니다.
    """
    def __init__(self, img_size=128, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        """x: [T, B, C, H, W] → [T, B, N, D]"""
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.proj(x)              # [T*B, D, H', W']
        _, D, Hp, Wp = x.shape
        x = x.reshape(T, B, D, Hp, Wp)
        x = self.lif(x)
        x = x.flatten(3).transpose(2, 3)  # [T, B, N, D]
        return x


class SpikingBiPhysformerBlock(nn.Module):
    """Spiking Bi-Physformer 블록: Norm → Bi-Attention → Residual → Norm → MLP → Residual"""
    def __init__(self, dim, num_heads=8, topk=4, n_win=4, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpikingBiLevelRoutingAttention(
            dim, num_heads=num_heads, topk=topk, n_win=n_win
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SpikingMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpikingBiPhysformer(nn.Module):
    """
    Spiking Bi-Physformer: rPPG 신호 추정 모델
    """
    def __init__(self, img_size=128, patch_size=8, in_chans=3,
                 embed_dim=128, depth=4, num_heads=4, topk=4, n_win=4):
        super().__init__()

        self.patch_embed = SpikingPatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )

        self.blocks = nn.ModuleList([
            SpikingBiPhysformerBlock(
                dim=embed_dim, num_heads=num_heads,
                topk=topk, n_win=n_win
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # rPPG Head: 각 time step 별로 Spatial Average → 스칼라 출력
        self.rppg_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W]  비디오 클립 입력
        returns: [B, T]  rPPG 신호 파형
        """
        B, T, C, H, W = x.shape

        # [B, T, C, H, W] → [T, B, C, H, W]  (SNN time step = video frames)
        x = x.permute(1, 0, 2, 3, 4)

        # Patch Embedding
        x = self.patch_embed(x)  # [T, B, N, D]

        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Spatial average (N 차원 축약)
        x = x.mean(dim=2)  # [T, B, D]

        # rPPG 예측
        x = self.rppg_head(x).squeeze(-1)  # [T, B]

        # [T, B] → [B, T]
        x = x.permute(1, 0)

        return x
