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
        
        self.n_win = n_win # (Lt_win, Lh_win, Lw_win)
        self.topk = topk
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.lif_q = neuron.LIFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lif_k = neuron.LIFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lif_v = neuron.LIFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.proj = nn.Linear(dim, dim)
        self.attn_scale = self.head_dim ** -0.5

    def forward(self, x):
        """
        x: [T, B, Lt, Lh, Lw, C]
        """
        T, B, Lt, Lh, Lw, C = x.shape
        wt, wh, ww = self.n_win
        
        # 0. Padding check for robustness
        pad_t = (wt - Lt % wt) % wt
        pad_h = (wh - Lh % wh) % wh
        pad_w = (ww - Lw % ww) % ww
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
            _, _, Lt, Lh, Lw, _ = x.shape
            
        # 1. 3D Window Partitioning
        x_win = x.view(T, B, wt, Lt//wt, wh, Lh//wh, ww, Lw//ww, C)
        x_win = x_win.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).flatten(5, 7) # [T, B, num_wins, win_size, C]
        num_wins = wt * wh * ww
        win_size = x_win.shape[5]
        
        # 2. Routing Phase
        # Use mean of spikes over T to decide routing
        region_feat = x_win.mean(dim=(0, 4)) # [B, num_wins, C]
        routing_scores = (region_feat @ region_feat.transpose(-2, -1)) * self.attn_scale
        routing_indices = torch.topk(routing_scores, k=self.topk, dim=-1).indices # [B, num_wins, topk]
        
        # 3. Attention Phase (SDLA)
        qkv = self.qkv(x_win)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)
        
        # Split heads
        q = q.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        k = k.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        v = v.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        
        # --- FIXED Vectorized Gathering using torch.gather ---
        # k: [T, B, num_wins_k, win_size, head, head_dim]
        # routing_indices: [B, num_wins_q, topk]
        
        # Expand k to allow each query window to select from any key window
        k_src = k.unsqueeze(2).expand(T, B, num_wins, num_wins, win_size, self.num_heads, self.head_dim)
        v_src = v.unsqueeze(2).expand(T, B, num_wins, num_wins, win_size, self.num_heads, self.head_dim)
        
        # Prepare index for gather: [T, B, num_wins_q, topk, win_size, head, head_dim]
        idx = routing_indices.view(1, B, num_wins, self.topk, 1, 1, 1).expand(T, B, num_wins, self.topk, win_size, self.num_heads, self.head_dim)
        
        k_g = torch.gather(k_src, dim=3, index=idx).flatten(3, 4) # [T, B, num_wins, topk*win_size, head, head_dim]
        v_g = torch.gather(v_src, dim=3, index=idx).flatten(3, 4)
        
        # --- SDLA with Correct Normalization ---
        q = q.permute(0, 1, 2, 4, 3, 5) # [T, B, num_wins, head, win_size, head_dim]
        k_g = k_g.permute(0, 1, 2, 4, 3, 5)
        v_g = v_g.permute(0, 1, 2, 4, 3, 5)
        
        kv = k_g.transpose(-2, -1) @ v_g 
        out = q @ kv
        
        # Standard Linear Attention Normalization: q @ (sum_j k_j)
        # k_g.sum(dim=-2) is [T, B, num_wins, head, head_dim]
        den = q @ k_g.sum(dim=-2, keepdim=True).transpose(-2, -1) # [T, B, num_wins, head, win_size, 1]
        out = out / (den + 1e-6)
        
        # Final projection
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(T, B, num_wins, win_size, C)
        out = self.proj(out)
        
        # 4. Reverse Window Partitioning
        out = out.view(T, B, wt, wh, ww, Lt//wt, Lh//wh, Lw//ww, C)
        out = out.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).reshape(T, B, Lt, Lh, Lw, C)
        
        # Unpad if necessary
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :Lt-pad_t, :Lh-pad_h, :Lw-pad_w, :]
            
        return out
