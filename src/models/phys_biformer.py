import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from src.models.biformer_attention import BiLevelRoutingAttention

class PhysBiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, v_threshold, n_win=(2, 4, 4)):
        super().__init__()
        # SNN-friendly BN: track_running_stats=False so train/eval 모두 batch stats 사용.
        # Spike-Driven Transformer / Spiking Physformer 류에서 train-eval 발화율 불일치를
        # 막기 위한 표준 처리.
        self.bn1 = layer.BatchNorm3d(dim, track_running_stats=False)
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.attn = BiLevelRoutingAttention(dim, num_heads=num_heads, n_win=n_win, v_threshold=v_threshold)

        self.bn2 = layer.BatchNorm3d(dim, track_running_stats=False)
        self.lif2 = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.ffn = nn.Sequential(
            layer.Conv3d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            layer.Conv3d(dim * 4, dim, kernel_size=1)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

        # Ensure all sub-modules use multi-step mode
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, C, Lt, Lh, Lw] (Potential Path - Real Valued)
        """
        identity = x

        # Sub-block 1: Attention
        x_norm = self.bn1(x)
        spikes = self.lif1(x_norm)
        self.last_firing_rate1 = spikes.mean().item()

        # Attention expects [T, B, Lt, Lh, Lw, C]
        spikes_attn = spikes.permute(0, 1, 3, 4, 5, 2)
        attn_out = self.attn(spikes_attn)
        attn_out = attn_out.permute(0, 1, 5, 2, 3, 4) # [T, B, C, Lt, Lh, Lw]

        # MS Shortcut: Identity (Potential) + Attn(Spikes)
        x = identity + attn_out * self.scale

        # Sub-block 2: FFN
        identity = x
        x_norm = self.bn2(x)
        spikes = self.lif2(x_norm)
        self.last_firing_rate2 = spikes.mean().item()

        x = identity + self.ffn(spikes) * self.scale
        return x

class PhysBiformer(nn.Module):
    def __init__(self, dim=64, num_blocks=3, num_heads=4, v_threshold=1.0, frame=160, patches=(4, 4, 4), T_snn=4):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.frame = frame
        self.T_snn = T_snn

        # Stem (ANN). track_running_stats=False — train/eval 일관된 BN 동작.
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=patches, padding=(1, 1, 1)),
            nn.BatchNorm3d(16, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(16, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(dim, track_running_stats=False),
            nn.ReLU()
        )

        # Temporal Positional Embedding (truncated normal init — zeros 시 학습 초기 collapse 유발)
        self.pos_embed = nn.Parameter(torch.empty(1, dim, 40, 1, 1))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Removed lif_input to preserve MS Shortcut benefits from the start

        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold, n_win=(2, 4, 4)) for _ in range(num_blocks)
        ])

        # Prediction head — identical to PhysFormer (baseline/Physformer.py)
        # Two stages of (Upsample 2x temporal + Conv3d + BN + ELU), then spatial GAP, then Conv1d.
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

        # Set entire model to multi-step mode
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """
        x: [B, 3, 160, 128, 128]
        """
        b, c, t, h, w = x.shape

        # 1. Stem (ANN) -> Real-valued features
        x = self.stem(x)
        x = x + self.pos_embed[:, :, :x.shape[2], :, :]

        # 2. SNN Potential Path Initialization
        # Repeat real-valued input for T_snn steps.
        # This acts as a constant current I injected into the first SNN block.
        x = x.unsqueeze(0).repeat(self.T_snn, 1, 1, 1, 1, 1) # [T, B, C, Lt, Lh, Lw]

        firing_rates = []
        for block in self.blocks:
            x = block(x)
            firing_rates.append(block.last_firing_rate1)
            firing_rates.append(block.last_firing_rate2)

        # 3. Head — PhysFormer 동일 구조
        # T_snn 평균 (membrane-potential 시간 평균) → upsample 2단계 → 공간 GAP → Conv1d
        x_out = x.mean(dim=0)                    # [B, C, Lt, Lh, Lw]   ex) [B, 64, 40, 32, 32]
        x_out = self.upsample(x_out)             # [B, C, 2*Lt, Lh, Lw] ex) [B, 64, 80, 32, 32]
        x_out = self.upsample2(x_out)            # [B, C/2, 4*Lt, Lh, Lw] ex) [B, 32, 160, 32, 32]
        x_out = torch.mean(x_out, 3)             # [B, C/2, T, Lw]
        x_out = torch.mean(x_out, 3)             # [B, C/2, T]
        rppg = self.ConvBlockLast(x_out)         # [B, 1, T]
        rppg = rppg.squeeze(1)                   # [B, T]

        self.last_firing_rates = firing_rates
        return rppg
