import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer
from src.models.biformer_attention import BiLevelRoutingAttention

class PhysBiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, v_threshold):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.attn = BiLevelRoutingAttention(dim, num_heads=num_heads, v_threshold=v_threshold)
        
        self.bn2 = nn.BatchNorm1d(dim)
        self.lif2 = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, Lt, b, L, dim):
        """
        x: [T_snn, B, Lt*L_spatial, dim]
        """
        T_snn = x.shape[0]
        L_total = Lt * L
        identity = x
        
        # Sub-block 1: Attention
        # Flatten time and batch for BatchNorm: [T_snn*B, dim, L_total]
        x_norm = self.bn1(x.reshape(-1, L_total, dim).transpose(-1, -2)).transpose(-1, -2).reshape(T_snn, b, L_total, dim)
        spikes = self.lif1(x_norm)
        self.last_firing_rate1 = spikes.mean().item()
        
        x = identity + self.attn(spikes) * self.scale
        
        # Sub-block 2: FFN
        identity = x
        x_norm = self.bn2(x.reshape(-1, L_total, dim).transpose(-1, -2)).transpose(-1, -2).reshape(T_snn, b, L_total, dim)
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
        
        # Stem (ANN) - Spiking-PhysFormer Configuration
        # Stride=(4, 4, 4) -> Lt = 160/4 = 40, H' = 128/4 = 32, W' = 128/4 = 32
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=patches, padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        
        self.lif_input = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold) for _ in range(num_blocks)
        ])
            
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        
        # 1. Stem (ANN)
        x = self.stem(x) # [B, dim, Lt, Lh, Lw]
        _, dim, Lt, Lh, Lw = x.shape
        L_spatial = Lh * Lw
        
        # 2. SNN Transition
        # Prepare sequence: [B, Lt, Lh, Lw, dim] -> [B, Lt*Lh*Lw, dim]
        x = x.permute(0, 2, 3, 4, 1).reshape(b, -1, dim)
        
        # Spiking-PhysFormer SNN Loop (Repeat T times)
        # [T_snn, B, L_total, dim]
        x = x.unsqueeze(0).repeat(self.T_snn, 1, 1, 1)
        x = self.lif_input(x)
        
        firing_rates = []
        for block in self.blocks:
            x = block(x, Lt, b, L_spatial, dim)
            firing_rates.append(block.last_firing_rate1)
            firing_rates.append(block.last_firing_rate2)
            
        # 3. Head (ANN)
        # Average across SNN time dimension: [T_snn, B, L_total, dim] -> [B, L_total, dim]
        x_out = x.mean(dim=0)
        
        # Reshape to extract temporal dimension: [B, Lt, L_spatial, dim]
        x_out = x_out.reshape(b, Lt, L_spatial, dim)
        
        # Average across spatial dimension: [B, Lt, dim]
        x_out = x_out.mean(dim=2) # [B, Lt, dim]
        
        # Upsample Lt to Target Frames (e.g. 40 -> 160)
        x_out = x_out.transpose(1, 2) # [B, dim, Lt]
        x_out = F.interpolate(x_out, size=self.frame, mode='linear', align_corners=True)
        
        # Final Projection
        x_out = x_out.transpose(1, 2) # [B, 160, dim]
        rppg = self.head(x_out).squeeze(-1) # [B, 160]
        
        self.last_firing_rates = firing_rates
        return rppg
