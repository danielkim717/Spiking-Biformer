import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.baseline.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from src.models.biformer_attention import BiLevelRoutingAttention
from spikingjelly.activation_based import neuron, surrogate, functional

class PhysBiformer(nn.Module):
    def __init__(self, frame=160, patches=(4, 4, 4), dim=64, num_heads=4, n_win=8, topk=4, v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.patches = patches
        
        # 1. ANN Stem & Patch Embedding
        self.baseline = ViT_ST_ST_Compact3_TDC_gra_sharp(
            frame=frame, patches=patches, dim=dim, num_heads=num_heads, image_size=(160, 128, 128)
        )
        
        # 2. SNN Transformer Blocks
        # Using LIFNode for Spike Generation
        self.lif_input = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm1d(dim), # Hardware friendly compared to LayerNorm
                BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, topk=topk, v_threshold=v_threshold),
                nn.BatchNorm1d(dim),
                nn.Linear(dim, dim * 4),
                neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True),
                nn.Linear(dim * 4, dim),
                neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
            ]) for _ in range(3)
        ])
        
    def forward(self, x):
        # x: [B, 3, 160, 128, 128]
        b, c, t, h, w = x.shape
        
        # --- ANN Stem Phase ---
        x = self.baseline.Stem0(x)
        x = self.baseline.Stem1(x)
        x = self.baseline.Stem2(x)
        x = self.baseline.patch_embedding(x) # [B, dim, Lt, Lh, Lw]
        
        Lt, Lh, Lw = x.shape[2], x.shape[3], x.shape[4]
        
        # --- SNN Simulation Phase (Sequence Mapping) ---
        # Direct Mapping: Lt (temporal dimension) becomes SNN Timestep T
        # [B, dim, Lt, Lh, Lw] -> [Lt, B, Lh*Lw, dim]
        x_snn = x.permute(2, 0, 3, 4, 1).contiguous()
        x_snn = x_snn.view(Lt, b, Lh * Lw, self.dim)
        
        # Initial Spike conversion
        x_snn = self.lif_input(x_snn)
        
        for bn1, attn, bn2, ffn1, lif1, ffn2, lif2 in self.blocks:
            # SNN Block 1: Attention
            identity = x_snn
            
            # BatchNorm1d expects [N, C, L] or [N, C]
            # Here we have [T, B, L, C]. We reshpae to apply BN across tokens
            x_snn = x_snn.permute(0, 1, 3, 2) # [T, B, C, L]
            orig_shape = x_snn.shape
            x_snn = bn1(x_snn.reshape(-1, self.dim, Lh*Lw))
            x_snn = x_snn.view(orig_shape).permute(0, 1, 3, 2) # [T, B, L, C]
            
            x_snn = attn(x_snn, T=Lt, H=Lh, W=Lw)
            x_snn = identity + x_snn
            
            # SNN Block 2: FFN
            identity = x_snn
            
            x_snn = x_snn.permute(0, 1, 3, 2)
            x_snn = bn2(x_snn.reshape(-1, self.dim, Lh*Lw))
            x_snn = x_snn.view(orig_shape).permute(0, 1, 3, 2)
            
            x_snn = ffn1(x_snn)
            x_snn = lif1(x_snn)
            x_snn = ffn2(x_snn)
            x_snn = lif2(x_snn)
            x_snn = identity + x_snn
            
        # --- Integration & Head Phase ---
        # SNN의 최종 출력 [T, B, L_spatial, dim]을 다시 ANN 업샘플러가 요구하는
        # [B, dim, T, H, W] 차원으로 돌려놓습니다.
        x_out = x_snn.permute(1, 3, 0, 2) # [B, dim, Lt, L_spatial]
        x_out = x_out.view(b, self.dim, Lt, Lh, Lw) # [B, dim, Lt, Lh, Lw]
        
        # Baseline Upsamplers (ANN)
        # Lt=4 -> 8 -> 160 (configured via patches=(40, 4, 4))
        features_up = self.baseline.upsample(x_out)
        features_up2 = self.baseline.upsample2(features_up)
        
        # Global Pooling and Prediction
        features_mean = torch.mean(features_up2, dim=3) # Spatially
        features_mean = torch.mean(features_mean, dim=3)
        
        rPPG = self.baseline.ConvBlockLast(features_mean).squeeze(1) # [B, 160]
        
        return rPPG, None, None, None
