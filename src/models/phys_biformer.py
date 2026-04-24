import torch
import torch.nn as nn
from src.models.baseline.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from src.models.biformer_attention import BiLevelRoutingAttention
import torch.nn.functional as F

class PhysBiformer(nn.Module):
    def __init__(self, frame=160, patches=(4, 4, 4), dim=64, num_heads=4, n_win=8, topk=4):
        super().__init__()
        self.baseline = ViT_ST_ST_Compact3_TDC_gra_sharp(frame=frame, patches=patches, dim=dim, num_heads=num_heads, image_size=(160, 128, 128))
        
        self.biformer_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, topk=topk),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(3)
        ])

    def forward(self, x, gra_sharp=None):
        b, c, t, h, w = x.shape
        x = self.baseline.Stem0(x)
        x = self.baseline.Stem1(x)
        x = self.baseline.Stem2(x)
        x = self.baseline.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Calculate temporal and spatial dimensions for routing
        t_p = t // 4 
        h_p = 4
        w_p = 4
        
        for block in self.biformer_blocks:
            # block[1] is BiLevelRoutingAttention
            attn_out = block[1](block[0](x), T=t_p, H=h_p, W=w_p)
            x = x + attn_out
            ffn_out = block[5](block[4](block[3](block[2](x))))
            x = x + ffn_out
            
        seq_len = x.shape[1]
        temp_dim = int(seq_len / 16)
        features_last = x.transpose(1, 2).view(b, self.baseline.dim, temp_dim, 4, 4)
        features_last = self.baseline.upsample(features_last)
        features_last = self.baseline.upsample2(features_last)
        features_last = torch.sum(features_last, 3)
        features_last = torch.sum(features_last, 3)
        rPPG = self.baseline.ConvBlockLast(features_last)
        rPPG = rPPG.squeeze(1)
        return rPPG, None, None, None
