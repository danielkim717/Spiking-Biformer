import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, n_win=8, topk=4, qk_scale=None, v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.n_win = n_win
        self.topk = topk
        
        # SNN Q, K, V layers
        self.qkv = nn.Linear(dim, dim * 3)
        self.lif_q = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.lif_k = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.lif_v = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.proj = nn.Linear(dim, dim)
        self.lif_proj = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)

    def forward(self, x, T=40, H=4, W=4):
        """
        x: [T_snn, B, L, C]
        """
        T_step, B, L, C = x.shape
        n_win = self.n_win
        win_size = L // n_win
        
        # 1. Routing Phase (Spike Summation for Rate Coding)
        # Using sum(dim=0) as requested for hardware-friendly routing
        x_spikes_sum = x.sum(dim=0) # [B, L, C] - Total spikes over time
        
        # Region features
        x_win = x_spikes_sum.view(B, n_win, win_size, C)
        region_feat = x_win.sum(dim=2) # [B, n_win, C] - Total spikes per region
        
        # Calculate Routing Map
        attn_r = (region_feat @ region_feat.transpose(-2, -1)) * (C ** -0.5)
        # routing_indices: [B, n_win, topk]
        routing_indices = torch.topk(attn_r, k=self.topk, dim=-1).indices
        
        # 2. Attention Phase (Sparse via Gather)
        qkv = self.qkv(x) # [T_snn, B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)
        
        # Split into windows: [T, B, n_win, win_size, head, head_dim]
        q = q.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        k = k.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        v = v.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        
        # --- Real Biformer Gather Logic ---
        # For each window in Q, we gather top-k windows from K and V
        # routing_indices: [B, n_win, topk]
        
        # We need to expand routing_indices to match the shape for gathering
        # indices_expanded: [T, B, n_win, topk, win_size, head, head_dim]
        idx = routing_indices.view(1, B, n_win, self.topk, 1, 1, 1)
        idx = idx.expand(T_step, B, n_win, self.topk, win_size, self.num_heads, self.head_dim)
        
        # Gather K and V: [T, B, n_win, topk, win_size, head, head_dim]
        # We gather from the 'window' dimension (dim=2)
        # Note: torch.gather doesn't support complex indexing easily for this, 
        # so we use advanced indexing or a loop for clarity and correctness in SNN.
        
        k_gathered = []
        v_gathered = []
        for i in range(self.topk):
            # routing_indices[:, :, i] gives the i-th best window for each query window
            r_idx = routing_indices[:, :, i] # [B, n_win]
            # Gather windows: [T, B, n_win, win_size, head, head_dim]
            k_i = torch.stack([k[:, b, r_idx[b]] for b in range(B)], dim=1)
            v_i = torch.stack([v[:, b, r_idx[b]] for b in range(B)], dim=1)
            k_gathered.append(k_i)
            v_gathered.append(v_i)
            
        k_g = torch.cat(k_gathered, dim=3) # [T, B, n_win, topk * win_size, head, head_dim]
        v_g = torch.cat(v_gathered, dim=3) # [T, B, n_win, topk * win_size, head, head_dim]
        
        # Now perform attention: Q [T, B, n_win, win_size, head, head_dim]
        # K_g [T, B, n_win, L_g, head, head_dim] where L_g = topk * win_size
        
        q = q.permute(0, 1, 2, 4, 3, 5) # [T, B, n_win, head, win_size, head_dim]
        k_g = k_g.permute(0, 1, 2, 4, 3, 5) # [T, B, n_win, head, L_g, head_dim]
        v_g = v_g.permute(0, 1, 2, 4, 3, 5) # [T, B, n_win, head, L_g, head_dim]
        
        # Attention: [T, B, n_win, head, win_size, L_g]
        attn = (q @ k_g.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Aggregate: [T, B, n_win, head, win_size, head_dim]
        out = (attn @ v_g)
        
        # Back to original shape
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(T_step, B, L, C)
        out = self.lif_proj(self.proj(out))
        
        return out
