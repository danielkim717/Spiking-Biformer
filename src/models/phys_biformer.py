import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from src.models.biformer_attention import BiLevelRoutingAttention

class PhysBiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, v_threshold, n_win=(2, 4, 4)):
        super().__init__()
        # Use spikingjelly's layer wrappers and set multi-step mode
        self.bn1 = layer.BatchNorm3d(dim)
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.attn = BiLevelRoutingAttention(dim, num_heads=num_heads, n_win=n_win, v_threshold=v_threshold)

        self.bn2 = layer.BatchNorm3d(dim)
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

        # Stem (ANN)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=patches, padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )

        # Temporal Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 40, 1, 1))

        # Removed lif_input to preserve MS Shortcut benefits from the start

        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold, n_win=(2, 4, 4)) for _ in range(num_blocks)
        ])

        self.head = nn.Linear(dim, 1)

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

        # 3. Head (ANN)
        # Final output is the mean of the membrane potential across T
        x_out = x.mean(dim=0)

        # Spatial Global Average Pooling
        x_out = x_out.mean(dim=(3, 4)) # [B, C, Lt]

        # Upsample Lt to Target (160)
        x_out = F.interpolate(x_out, size=self.frame, mode='linear', align_corners=True)

        x_out = x_out.transpose(1, 2) # [B, 160, C]
        rppg = self.head(x_out).squeeze(-1) # [B, 160]

        self.last_firing_rates = firing_rates
        return rppg
