import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, n_win=(4, 4, 4), topk=4, v_threshold=1.0):
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
        x: [T, B, Lt, Lh, Lw, C] (SNN Spikes)
        """
        T, B, Lt, Lh, Lw, C = x.shape
        wt, wh, ww = self.n_win
        
        # 1. 3D Window Partitioning
        # Ensure divisible
        assert Lt % wt == 0 and Lh % wh == 0 and Lw % ww == 0
        
        # [T, B, wt, Lt//wt, wh, Lh//wh, ww, Lw//ww, C]
        x_win = x.view(T, B, wt, Lt//wt, wh, Lh//wh, ww, Lw//ww, C)
        # Permute to [T, B, wt, wh, ww, (Lt//wt * Lh//wh * Lw//ww), C]
        x_win = x_win.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).flatten(5, 7)
        num_wins = wt * wh * ww
        win_size = x_win.shape[5]
        
        # 2. Vectorized Routing Phase
        # Calculate regional features (mean spikes over time and window)
        # [B, num_wins, C]
        region_feat = x_win.sum(dim=0).mean(dim=4) 
        
        # Routing scores [B, num_wins, num_wins]
        routing_scores = (region_feat @ region_feat.transpose(-2, -1)) * self.attn_scale
        # topk indices [B, num_wins, topk]
        routing_indices = torch.topk(routing_scores, k=self.topk, dim=-1).indices
        
        # 3. Attention Phase (SDLA)
        qkv = self.qkv(x_win) # [T, B, wt*wh*ww, L_win, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)
        
        # Split heads
        q = q.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        k = k.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        v = v.view(T, B, num_wins, win_size, self.num_heads, self.head_dim)
        
        # Vectorized Gathering
        # routing_indices: [B, num_wins, topk]
        # We need to gather K and V for each window from its topk regions.
        # This can be done via index_select or advanced indexing.
        
        # Expand routing_indices for gathering: [T, B, num_wins, topk, win_size, head, head_dim]
        # For simplicity, we cat the topk regions' K and V.
        # [T, B, num_wins, win_size * topk, head, head_dim]
        
        # Efficient gathering without Python loop
        B_idx = torch.arange(B, device=x.device).view(1, B, 1, 1)
        # routing_indices is [B, num_wins, topk]
        k_gathered = k[:, B_idx, routing_indices] # [T, B, num_wins, topk, win_size, head, head_dim]
        v_gathered = v[:, B_idx, routing_indices] 
        
        k_gathered = k_gathered.flatten(3, 4) # [T, B, num_wins, topk*win_size, head, head_dim]
        v_gathered = v_gathered.flatten(3, 4)
        
        # --- Normalized SDLA ---
        # q: [T, B, num_wins, win_size, head, head_dim]
        # k_g: [T, B, num_wins, topk*win_size, head, head_dim]
        
        q = q.permute(0, 1, 2, 4, 3, 5) # [T, B, num_wins, head, win_size, head_dim]
        k_g = k_gathered.permute(0, 1, 2, 4, 3, 5)
        v_g = v_gathered.permute(0, 1, 2, 4, 3, 5)
        
        # KV = K.T @ V
        kv = k_g.transpose(-2, -1) @ v_g # [T, B, num_wins, head, head_dim, head_dim]
        
        # out = Q @ KV
        out = q @ kv # [T, B, num_wins, head, win_size, head_dim]
        
        # Normalization (Q @ K^T @ 1)
        # In SDLA, we normalize by the sum of keys to keep magnitude stable.
        D = q @ k_g.transpose(-2, -1).sum(dim=-1, keepdim=True) # [T, B, num_wins, head, win_size, 1]
        out = out / (D + 1e-6)
        
        # Final projection
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(T, B, num_wins, win_size, C)
        out = self.proj(out)
        
        # 4. Reverse Window Partitioning
        # [T, B, wt, wh, ww, Lt//wt, Lh//wh, Lw//ww, C]
        out = out.view(T, B, wt, wh, ww, Lt//wt, Lh//wh, Lw//ww, C)
        out = out.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).reshape(T, B, Lt, Lh, Lw, C)
        
        return out
