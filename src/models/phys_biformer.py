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
        identity = x
        
        # Sub-block 1: Attention
        x_norm = self.bn1(x.transpose(-1, -2).reshape(-1, dim, L)).reshape(Lt, b, dim, L).transpose(-1, -2)
        spikes = self.lif1(x_norm)
        self.last_firing_rate1 = spikes.mean().item()
        
        x = identity + self.attn(spikes) * self.scale
        
        # Sub-block 2: FFN
        identity = x
        x_norm = self.bn2(x.transpose(-1, -2).reshape(-1, dim, L)).reshape(Lt, b, dim, L).transpose(-1, -2)
        spikes = self.lif2(x_norm)
        self.last_firing_rate2 = spikes.mean().item()
        
        x = identity + self.ffn(spikes) * self.scale
        return x

class PhysBiformer(nn.Module):
    def __init__(self, dim=64, num_blocks=3, num_heads=4, v_threshold=1.0, frame=160, patches=(40, 4, 4)):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.frame = frame
        
        # Stem (ANN)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        self.lif_input = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold) for _ in range(num_blocks)
        ])
            
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = self.stem(x)
        _, dim, Lt, Lh, Lw = x.shape
        L = Lh * Lw
        
        x = x.permute(2, 0, 3, 4, 1).reshape(Lt, b, L, dim)
        x = self.lif_input(x)
        
        firing_rates = []
        for block in self.blocks:
            x = block(x, Lt, b, L, dim)
            firing_rates.append(block.last_firing_rate1)
            firing_rates.append(block.last_firing_rate2)
            
        x_out = x.mean(dim=2).permute(1, 2, 0) # [B, dim, Lt]
        x_out = F.interpolate(x_out, size=160, mode='linear', align_corners=True) # [B, dim, 160]
        
        x_out = x_out.transpose(1, 2) # [B, 160, dim]
        rppg = self.head(x_out).squeeze(-1) # [B, 160]
        
        self.last_firing_rates = firing_rates
        return rppg
        
        self.last_firing_rates = firing_rates
        return rppg
