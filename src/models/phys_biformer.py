import torch
import torch.nn as nn
from src.models.baseline.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from src.models.biformer_attention import BiLevelRoutingAttention
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate

class PhysBiformer(nn.Module):
    def __init__(self, frame=160, patches=(4, 4, 4), dim=64, num_heads=4, n_win=8, topk=4):
        super().__init__()
        self.baseline = ViT_ST_ST_Compact3_TDC_gra_sharp(frame=frame, patches=patches, dim=dim, num_heads=num_heads, image_size=(160, 128, 128))
        
        # SNN components
        self.lif_input = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.biformer_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, topk=topk),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True), # GELU -> LIF
                nn.Linear(dim * 4, dim),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)  # Added LIF for post-FFN spikes
            ]) for _ in range(3)
        ])

    def forward(self, x, gra_sharp=None):
        b, c, t, h, w = x.shape
        
        # Stem and Patch Embedding (Real-valued initial layers)
        x = self.baseline.Stem0(x)
        x = self.baseline.Stem1(x)
        x = self.baseline.Stem2(x)
        x = self.baseline.patch_embedding(x) # [B, dim, T', H', W']
        
        # 1. Convert to Spikes and Reshape for SNN [T, B, N, C]
        # Preserve time dimension (t_p) instead of flattening it
        t_p, h_p, w_p = x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(2, 0, 3, 4, 1).reshape(t_p, b, h_p * w_p, -1) # [T, B, N_spatial, C]
        
        # Initial LIF to ensure spike format
        x = self.lif_input(x)
        
        for block in self.biformer_blocks:
            norm1, attn, norm2, fc1, lif1, fc2, lif2 = block
            
            # Block 1: Attention (Spike-In -> Spike-Out)
            attn_out = attn(norm1(x), T=t_p, H=h_p, W=w_p)
            x = x + attn_out
            
            # Block 2: FFN (Spike-In -> Spike-Out)
            ffn_out = lif2(fc2(lif1(fc1(norm2(x)))))
            x = x + ffn_out
            
        # 2. Output Head Processing
        # Aggregate back for rPPG extraction
        # x is [T, B, N_spatial, C]
        seq_len = x.shape[2]
        # Reshape to [B, C, T, H, W] for baseline upsampling
        out = x.permute(1, 3, 0, 2).reshape(b, -1, t_p, h_p, w_p)
        
        # Upsampling and global pooling
        features_last = self.baseline.upsample(out)
        features_last = self.baseline.upsample2(features_last)
        features_last = torch.sum(features_last, 3)
        features_last = torch.sum(features_last, 3)
        
        rPPG = self.baseline.ConvBlockLast(features_last)
        rPPG = rPPG.squeeze(1)
        return rPPG, None, None, None

