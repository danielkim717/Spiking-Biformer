"""
Spike-Driven BiLevel Routing Attention.

이 모듈은 Spiking Physformer 의 MS-SA(Membrane-Shortcut Spike-driven Self-Attention)
연산 자리에 그대로 들어가는 Biformer(BiLevel Routing) 어텐션이다.

원칙
----
1. Spiking Physformer 의 spike-driven attention 흐름은 그대로 유지한다.
   - 입력: BN→LIF 후 binary spike  ([T, B, Lt, Lh, Lw, C])
   - QKV 는 Linear 실수 투영 → LIF → 스파이크 Q/K/V
   - 어텐션 본체:  O = (Q · K.T · V) * scale       (softmax 없음, per-token 정규화 없음)
   - 출력: 실수 텐서 (다음 BN/LIF/MS-shortcut 에서 처리)

2. Biformer 메커니즘은 다음 두 단계로 attention 안에서만 수행된다.
   (a) 3D Window Partitioning  : [T,B,Lt,Lh,Lw,C] → [T,B,num_wins,win_size,C]
   (b) Region-level Top-k Routing :
       - Routing 점수는 Biformer 원논문과 동일하게 **실수값(dense) region feature**
         로 계산한다. (스파이크 평균은 발화율이 균일하여 region 간 변별력이 사라짐)
       - 각 query window 가 affinity 상위 top-k key window 만 참조하도록 K, V 를
         gather 한 뒤, 그 위에서 spike-driven attention 을 수행한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate


class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, n_win=(2, 4, 4), topk=4, v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads

        self.n_win = n_win
        self.topk = topk

        # QKV 실수 투영 (Spiking Physformer MS-SA 와 동일한 자리)
        self.qkv = nn.Linear(dim, dim * 3)

        # Spike encoding for attention path
        self.lif_q = neuron.LIFNode(v_threshold=v_threshold,
                                    surrogate_function=surrogate.ATan(),
                                    detach_reset=True)
        self.lif_k = neuron.LIFNode(v_threshold=v_threshold,
                                    surrogate_function=surrogate.ATan(),
                                    detach_reset=True)
        self.lif_v = neuron.LIFNode(v_threshold=v_threshold,
                                    surrogate_function=surrogate.ATan(),
                                    detach_reset=True)

        self.proj = nn.Linear(dim, dim)
        self.attn_scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        x: [T, B, Lt, Lh, Lw, C]  (binary spikes from BN→LIF)
        returns: [T, B, Lt, Lh, Lw, C]  (real-valued, fed back into MS-shortcut)
        """
        T, B, Lt, Lh, Lw, C = x.shape
        wt, wh, ww = self.n_win

        # 0. 3D padding (Lt/Lh/Lw 가 wt/wh/ww 의 배수가 아닐 때 안전망)
        pad_t = (wt - Lt % wt) % wt
        pad_h = (wh - Lh % wh) % wh
        pad_w = (ww - Lw % ww) % ww
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
            _, _, Lt, Lh, Lw, _ = x.shape

        # 1. 3D Window Partitioning -> [T, B, num_wins, win_size, C]
        x_win = x.view(T, B, wt, Lt // wt, wh, Lh // wh, ww, Lw // ww, C)
        x_win = x_win.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
        x_win = x_win.flatten(5, 7)        # [T, B, wt, wh, ww, win_size, C]
        x_win = x_win.flatten(2, 4)        # [T, B, num_wins, win_size, C]
        num_wins = wt * wh * ww
        win_size = x_win.shape[3]
        topk = min(self.topk, num_wins)

        # 2. QKV 실수 투영
        qkv = self.qkv(x_win)              # [T, B, num_wins, win_size, 3C]
        q_real, k_real, v_real = qkv.chunk(3, dim=-1)

        # 3. BiLevel Region Routing — Biformer 원논문과 동일하게
        #    *실수값* region feature 로 affinity 를 계산한다.
        #    (스파이크 평균은 모든 region 이 ~동일한 발화율로 수렴해 변별력 상실)
        q_region = q_real.mean(dim=(0, 3))   # [B, num_wins, C]
        k_region = k_real.mean(dim=(0, 3))   # [B, num_wins, C]
        a_r = (q_region @ k_region.transpose(-2, -1)) * self.attn_scale   # [B, num_wins, num_wins]
        routing_indices = torch.topk(a_r, k=topk, dim=-1).indices         # [B, num_wins, topk]

        # 4. Spike-Driven path: LIF 로 binary Q/K/V 생성
        q = self.lif_q(q_real)
        k = self.lif_k(k_real)
        v = self.lif_v(v_real)

        # head 분리
        q = q.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        k = k.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        v = v.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)

        # 5. Top-k 라우팅된 K, V 만 gather
        #    k_src: 각 query 윈도우 q_i 가 임의의 key 윈도우 k_j 를 선택할 수 있도록 확장
        k_src = k.unsqueeze(2).expand(T, B, num_wins, num_wins,
                                      win_size, self.num_heads, self.head_dim)
        v_src = v.unsqueeze(2).expand(T, B, num_wins, num_wins,
                                      win_size, self.num_heads, self.head_dim)
        idx = routing_indices.view(1, B, num_wins, topk, 1, 1, 1).expand(
            T, B, num_wins, topk, win_size, self.num_heads, self.head_dim)
        k_g = torch.gather(k_src, dim=3, index=idx).flatten(3, 4)
        v_g = torch.gather(v_src, dim=3, index=idx).flatten(3, 4)
        # k_g, v_g: [T, B, num_wins, topk*win_size, head, head_dim]

        # 6. Spike-Driven Attention  (Spiking Physformer / Spike-Driven Transformer 표준)
        #    O = (Q · K.T · V) * 1/√d_h          (softmax X, per-token 정규화 X)
        q  = q .permute(0, 1, 2, 4, 3, 5)   # [T, B, num_wins, head, win_size,         head_dim]
        k_g = k_g.permute(0, 1, 2, 4, 3, 5) # [T, B, num_wins, head, topk*win_size,    head_dim]
        v_g = v_g.permute(0, 1, 2, 4, 3, 5) # [T, B, num_wins, head, topk*win_size,    head_dim]

        kv = k_g.transpose(-2, -1) @ v_g    # [T, B, num_wins, head, head_dim, head_dim]
        out = (q @ kv) * self.attn_scale    # [T, B, num_wins, head, win_size,  head_dim]

        # 7. 출력 정렬 + projection
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(T, B, num_wins, win_size, C)
        out = self.proj(out)

        # 8. Reverse window partitioning -> [T, B, Lt, Lh, Lw, C]
        out = out.view(T, B, wt, wh, ww, Lt // wt, Lh // wh, Lw // ww, C)
        out = out.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).reshape(T, B, Lt, Lh, Lw, C)

        if pad_t or pad_h or pad_w:
            out = out[:, :, :Lt - pad_t, :Lh - pad_h, :Lw - pad_w, :]

        return out
