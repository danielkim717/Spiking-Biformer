import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from .spiking_biformer import SpikingBiLevelRoutingAttention, SpikingMLP

class SpikingPatchEmbed(nn.Module):
    """
    2D 이미지를 Spiking Token 형태로 변환하는 Patch Embedding 레이어
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = layer.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan())
        
    def forward(self, x):
        """ x: [T, B, C, H, W] -> out: [T, B, N, C] """
        x = self.proj(x)
        x = self.lif(x)
        T, B, C, H, W = x.shape
        # Spatial 차원을 Flatten하고 채널 축을 Sequence C로 변경 (N = H*W)
        x = x.flatten(3).transpose(2, 3) 
        return x

class SpikingBiPhysformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, topk=4, mlp_ratio=4.):
        super().__init__()
        self.norm1 = layer.LayerNorm(dim)
        self.attn = SpikingBiLevelRoutingAttention(dim, num_heads=num_heads, topk=topk)
        
        self.norm2 = layer.LayerNorm(dim)
        self.mlp = SpikingMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SpikingBiPhysformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=128, depth=4, num_heads=8, T=4):
        super().__init__()
        self.T = T
        self.patch_embed = SpikingPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        self.blocks = nn.ModuleList([
            SpikingBiPhysformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.norm = layer.LayerNorm(embed_dim)
        self.head = layer.Linear(embed_dim, num_classes)
        self.lif_head = neuron.LIFNode(surrogate_function=surrogate.ATan())
        
    def forward(self, x):
        """ [B, C, H, W] 이미지 입력을 Time Step만큼 반복 후 전달 """
        # 단일 이미지 입력을 Spiking 처리를 위해 시간 축(T)으로 복제
        # 입력이 이미 [T, B, C, H, W] 형태라면 생략 가능.
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            
        x = self.patch_embed(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # SNN Global Average Pooling
        x = x.mean(dim=2) # Token dimension (N) 평균
        
        x = self.head(x)
        x = self.lif_head(x)
        
        # Time step (T)에 대한 출력 평균 계산 후 예측
        return x.mean(0)
