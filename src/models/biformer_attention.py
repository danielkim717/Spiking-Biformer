import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, n_win=8, topk=4, v_threshold=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = dim // num_heads
        
        self.n_win = n_win
        self.topk = topk
        
        # Learnable scaling factor (Anti-sparsity)
        self.scale = nn.Parameter(torch.ones(1) * 0.125) # Initial scale
        
        # SNN Q, K, V layers
        self.qkv = nn.Linear(dim, dim * 3)
        self.lif_q = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.lif_k = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        self.lif_v = neuron.LIFNode(v_threshold=v_threshold, v_reset=0.0, surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True)
        
        self.proj = nn.Linear(dim, dim)
        # Note: No internal LIF for projection output. 
        # The result will be real-valued and added to the residual.

    def forward(self, x, T=40, H=4, W=4):
        """
        x: [T_snn, B, L, C] (Spikes)
        """
        T_step, B, L, C = x.shape
        n_win = self.n_win
        win_size = L // n_win
        
        # 1. Routing Phase (Spike Summation)
        x_spikes_sum = x.sum(dim=0) # [B, L, C]
        x_win = x_spikes_sum.view(B, n_win, win_size, C)
        region_feat = x_win.sum(dim=2) # [B, n_win, C]
        
        attn_r = (region_feat @ region_feat.transpose(-2, -1))
        routing_indices = torch.topk(attn_r, k=self.topk, dim=-1).indices
        
        # 2. Attention Phase (Spike-driven Linear Attention)
        # Apply learnable scaling before QKV
        qkv = self.qkv(x * self.scale) # [T_snn, B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # SNN Activation
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)
        
        # Split into windows
        q = q.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        k = k.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        v = v.view(T_step, B, n_win, win_size, self.num_heads, self.head_dim)
        
        # Gather K and V from top-k regions
        k_gathered = []
        v_gathered = []
        for i in range(self.topk):
            r_idx = routing_indices[:, :, i] # [B, n_win]
            k_i = torch.stack([k[:, b, r_idx[b]] for b in range(B)], dim=1)
            v_i = torch.stack([v[:, b, r_idx[b]] for b in range(B)], dim=1)
            k_gathered.append(k_i)
            v_gathered.append(v_i)
            
        k_g = torch.cat(k_gathered, dim=3) # [T, B, n_win, L_g, head, head_dim]
        v_g = torch.cat(v_gathered, dim=3) # [T, B, n_win, L_g, head, head_dim]
        
        # --- Spike-driven Linear Attention (SDLA) ---
        q = q.permute(0, 1, 2, 4, 3, 5)
        k_g = k_g.permute(0, 1, 2, 4, 3, 5)
        v_g = v_g.permute(0, 1, 2, 4, 3, 5)
        
        # Step 1: KV = K.T @ V
        kv = k_g.transpose(-2, -1) @ v_g
        # Step 2: Out = Q @ KV
        out = q @ kv
        
        # Back to original shape
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(T_step, B, L, C)
        out = self.proj(out) # Real-valued output for residual
        
        return out
