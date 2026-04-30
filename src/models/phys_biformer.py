import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer
from src.models.biformer_attention import BiLevelRoutingAttention

class PhysBiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, v_threshold, n_win=(4, 4, 4)):
        super().__init__()
        # Use spikingjelly's BatchNorm for better SNN compatibility
        self.bn1 = layer.BatchNorm3d(dim)
        self.lif1 = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.attn = BiLevelRoutingAttention(dim, num_heads=num_heads, n_win=n_win, v_threshold=v_threshold)
        
        self.bn2 = layer.BatchNorm3d(dim)
        self.lif2 = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.ffn = nn.Sequential(
            layer.Conv3d(dim, dim * 4, kernel_size=1),
            layer.GELU(),
            layer.Conv3d(dim * 4, dim, kernel_size=1)
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """
        x: [T, B, C, Lt, Lh, Lw] (Potential Path)
        """
        # Membrane Shortcut (MS): x = x + attn(lif(x))
        # Note: In MS, identity is the potential/float signal.
        identity = x
        
        # Sub-block 1: Attention
        x_norm = self.bn1(x)
        spikes = self.lif1(x_norm) # [T, B, C, Lt, Lh, Lw]
        self.last_firing_rate1 = spikes.mean().item()
        
        # Attention expects [T, B, Lt, Lh, Lw, C]
        spikes_attn = spikes.permute(0, 1, 3, 4, 5, 2)
        attn_out = self.attn(spikes_attn)
        attn_out = attn_out.permute(0, 1, 5, 2, 3, 4) # [T, B, C, Lt, Lh, Lw]
        
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
        
        # 3. Temporal Positional Embedding
        # 160 / 4 = 40
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, 40, 1, 1))
        
        self.lif_input = neuron.LIFNode(v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        # Set n_win for attention (Lt=40, Lh=32, Lw=32 -> wins of 4x4x4)
        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold, n_win=(4, 4, 4)) for _ in range(num_blocks)
        ])
            
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: [B, 3, 160, 128, 128]
        """
        b, c, t, h, w = x.shape
        
        # 1. Stem (ANN)
        x = self.stem(x) # [B, dim, Lt, Lh, Lw]
        x = x + self.pos_embed[:, :, :x.shape[2], :, :]
        
        # 2. SNN Transition (Constant Current I)
        # Repeat input for T_snn steps to simulate SNN dynamics properly
        x = x.unsqueeze(0).repeat(self.T_snn, 1, 1, 1, 1, 1) # [T, B, C, Lt, Lh, Lw]
        
        # Potential Path starts here
        x = self.lif_input(x) # Initial spikes conversion
        
        firing_rates = []
        for block in self.blocks:
            x = block(x)
            firing_rates.append(block.last_firing_rate1)
            firing_rates.append(block.last_firing_rate2)
            
        # 3. Head (ANN)
        # Average across SNN time: [T, B, C, Lt, Lh, Lw] -> [B, C, Lt, Lh, Lw]
        x_out = x.mean(dim=0)
        
        # Spatial Global Average Pooling
        x_out = x_out.mean(dim=(3, 4)) # [B, C, Lt]
        
        # Upsample to target temporal length (160)
        x_out = F.interpolate(x_out, size=self.frame, mode='linear', align_corners=True)
        
        # Final Projection
        x_out = x_out.transpose(1, 2) # [B, 160, C]
        rppg = self.head(x_out).squeeze(-1) # [B, 160]
        
        self.last_firing_rates = firing_rates
        return rppg
