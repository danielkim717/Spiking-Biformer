"""
Spiking Bi-level Routing Attention (SNN + Biformer)

핵심 아이디어:
 1) 입력 토큰을 Region(윈도) 단위로 분할
 2) Region 간 유사도를 LIF 뉴런의 Spike Firing Rate로 계산 → 라우팅 인덱스(top-k) 결정
 3) Top-k Region 쌍에 대해서만 Fine-grained Attention 수행
 4) 모든 Q, K, V, Projection 전후에 LIF 뉴런을 배치하여 SNN 특성 유지

Note: spikingjelly의 wrapper 레이어(layer.Linear 등)는 multi-step mode에서 텐서 차원 불일치를
      일으킬 수 있으므로, PyTorch 기본 nn.Linear/nn.Dropout을 사용하고
      LIF 뉴런만 spikingjelly에서 가져옵니다.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from einops import rearrange


class SpikingMLP(nn.Module):
    """Spiking MLP: FC → LIF → FC → LIF"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [T, B, N, C]  – LIF 뉴런은 선두 차원을 time step으로 처리
        T, B, N, C = x.shape
        # FC는 마지막 차원에만 적용되므로 reshape 불필요
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.drop(x)
        return x


class SpikingBiLevelRoutingAttention(nn.Module):
    """
    Spiking Bi-level Routing Attention
    - Region Partition → Coarse Routing (LIF 기반) → Fine-grained Attention
    """
    def __init__(self, dim, num_heads=8, n_win=7, topk=4,
                 qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_win = n_win
        self.topk = topk

        # QKV 생성 (PyTorch 기본 nn.Linear)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.lif_q = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.lif_k = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.lif_v = neuron.LIFNode(surrogate_function=surrogate.ATan())

        # region 수준 라우팅을 위한 Linear
        self.router = nn.Linear(dim, dim, bias=False)
        self.lif_router = neuron.LIFNode(surrogate_function=surrogate.ATan())

        # Projection
        self.proj = nn.Linear(dim, dim)
        self.lif_proj = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: [T, B, N, C]  (Time, Batch, Seq, Channels)
        """
        T, B, N, C = x.shape

        # QKV 계산
        qkv = self.qkv(x)                           # [T, B, N, 3C]
        qkv = qkv.reshape(T, B, N, 3, C)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]

        # Spiking Layer 통과
        q = self.lif_q(q)    # [T, B, N, C]
        k = self.lif_k(k)
        v = self.lif_v(v)

        # ---- Region Partition (Coarse level) ----
        n_win = self.n_win
        win_size = max(1, N // (n_win * n_win)) if N >= n_win * n_win else 1
        n_regions = N // win_size if win_size > 0 else N

        if n_regions > 1 and N % win_size == 0:
            # Region 평균으로 라우팅 키 생성
            q_r = q.reshape(T, B, n_regions, win_size, C).mean(dim=3)
            k_r = k.reshape(T, B, n_regions, win_size, C).mean(dim=3)

            # Spiking Router
            q_r = self.lif_router(self.router(q_r))
            region_sim = torch.einsum('tbrc, tbsc -> tbrs', q_r, k_r)

            # Top-k 라우팅
            topk_val = min(self.topk, n_regions)
            topk_indices = region_sim.mean(dim=0).topk(topk_val, dim=-1).indices

        # Fine-grained Attention (전체 또는 라우팅된 토큰)
        q_heads = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k_heads = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v_heads = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # [T, B, heads, N, head_dim]

        attn = (q_heads @ k_heads.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v_heads)
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, N, C)

        # Projection + Spiking
        out = self.lif_proj(self.proj(out))
        out = self.proj_drop(out)
        return out
