import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.baseline.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from src.models.biformer_attention import BiLevelRoutingAttention

from spikingjelly.activation_based import neuron, surrogate, functional

class PhysBiformer(nn.Module):
    def __init__(self, frame=160, patches=(4, 4, 4), dim=64, num_heads=4, n_win=8, topk=4, v_threshold=1.0, T=4):
        super().__init__()
        self.T = T # SNN simulation timesteps
        self.dim = dim
        self.patches = patches
        
        # 1. ANN Stem & Patch Embedding
        self.baseline = ViT_ST_ST_Compact3_TDC_gra_sharp(
            frame=frame, patches=patches, dim=dim, num_heads=num_heads, image_size=(160, 128, 128)
        )
        
        # 2. SNN Transformer Blocks
        # We replace the original transformer layers with our Spiking Biformer blocks
        self.lif_input = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, topk=topk, v_threshold=v_threshold),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True),
                nn.Linear(dim * 4, dim),
                neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
            ]) for _ in range(3)
        ])
        
    def forward(self, x):
        # x: [B, 3, 160, 128, 128]
        b, c, t, h, w = x.shape
        
        # ANN Stem Phase
        x = self.baseline.Stem0(x)
        x = self.baseline.Stem1(x)
        x = self.baseline.Stem2(x)
        x = self.baseline.patch_embedding(x) # [B, dim, Lt, Lh, Lw] e.g. [B, 64, 40, 4, 4]
        
        Lt, Lh, Lw = x.shape[2], x.shape[3], x.shape[4]
        # [B, dim, Lt, Lh, Lw] -> [B, Lt, Lh, Lw, dim]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        # --- SNN Simulation Phase ---
        # Repeat input for T simulation steps: [T, B, Lt, Lh, Lw, dim]
        x_snn = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1) 
        x_snn = x_snn.flatten(2, 4) # [T, B, L_total, dim] where L_total = Lt*Lh*Lw
        
        # Initial Spike conversion
        x_snn = self.lif_input(x_snn)
        
        for ln1, attn, ln2, ffn1, lif1, ffn2, lif2 in self.blocks:
            # SNN Block 1: Attention
            identity = x_snn
            x_snn = ln1(x_snn)
            x_snn = attn(x_snn, T=Lt, H=Lh, W=Lw) # BRA processes [T, B, L, D]
            x_snn = identity + x_snn
            
            # SNN Block 2: FFN
            identity = x_snn
            x_snn = ln2(x_snn)
            x_snn = ffn1(x_snn)
            x_snn = lif1(x_snn)
            x_snn = ffn2(x_snn)
            x_snn = lif2(x_snn)
            x_snn = identity + x_snn
            
        # --- Integration & Head Phase ---
        # Average spikes over simulation time T: [B, L_total, dim]
        x_out = torch.mean(x_snn, dim=0) 
        
        # Reshape for Upsampling: [B, dim, Lt, Lh, Lw]
        x_out = x_out.view(b, Lt, Lh, Lw, self.dim).permute(0, 4, 1, 2, 3)
        
        # Baseline Upsamplers (ANN)
        # Lt=40 -> 80 -> 160
        features_up = self.baseline.upsample(x_out)
        features_up2 = self.baseline.upsample2(features_up)
        
        # Global Pooling and Prediction
        features_mean = torch.mean(features_up2, dim=3) # Spatially
        features_mean = torch.mean(features_mean, dim=3)
        
        rPPG = self.baseline.ConvBlockLast(features_mean).squeeze(1) # [B, 160]
        
        return rPPG, None, None, None
