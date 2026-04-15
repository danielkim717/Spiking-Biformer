import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate

class SpikingMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = layer.Linear(in_features, hidden_features)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = layer.Linear(hidden_features, out_features)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.drop = layer.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.drop(x)
        return x

class SpikingBiLevelRoutingAttention(nn.Module):
    """
    Spiking 기반 Bi-level Routing Attention의 핵심 구조
    Q, K, V 매트릭스 계산 전후에 LIF 뉴런을 배치하여 Spiking 동작 수행
    (일반 Attention 구조를 SNN-Biformer 형태로 축약)
    """
    def __init__(self, dim, num_heads=8, topk=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.topk = topk # routing할 region 개수
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = layer.Linear(dim, dim * 3, bias=qkv_bias)
        self.lif_qkv = neuron.LIFNode(surrogate_function=surrogate.ATan())
        
        self.attn_drop = layer.Dropout(attn_drop)
        
        self.proj = layer.Linear(dim, dim)
        self.lif_proj = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.proj_drop = layer.Dropout(proj_drop)

    def forward(self, x):
        """
        x 입력 형태: [T, B, N, C]
        T: Time step (SNN용)
        B: Batch size
        N: Sequence Length
        C: Channels (dim)
        """
        # SpikingJelly의 activation_based layer를 통과
        qkv = self.lif_qkv(self.qkv(x))
        # qkv 분리 후 Routing 로직 추가 가능 (현재는 기초 Attention 뼈대)
        
        T, B, N, C3 = qkv.shape
        qkv_reshaped = qkv.view(T, B, N, 3, self.num_heads, (C3 // 3) // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # q, k, v의 shape: [T, B, num_heads, N, head_dim]
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        # Time step을 고려한 SNN 행렬 곱 (Q * K)
        # 실제 최적화 된 Bi-level Routing 적용 시 region 별 top-k 마스킹 수행
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # dropout for time step dimensions
        
        x_out = (attn @ v).transpose(3, 4).reshape(T, B, N, self.dim)
        
        # Projection
        x_out = self.lif_proj(self.proj(x_out))
        x_out = self.proj_drop(x_out)
        
        return x_out
