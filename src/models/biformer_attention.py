import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, surrogate

class BiLevelRoutingAttention(nn.Module):
    """
    Spiking Bi-Level Routing Attention (S-BRA)
    1. Actual Sparse Attention: Uses gather for efficient sparse attention.
    2. Dynamic Routing over Time: Calculates routing indices per timestep.
    3. Spike-Driven Region Features: Uses sum instead of mean.
    4. SNN Integration: Q, K, V pass through LIF neurons after projection.
    """
    def __init__(self, dim, num_heads=4, n_win=8, topk=4, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.n_win = n_win  # Number of spatial regions (windows)
        self.topk = topk    # Number of regions to attend to

        self.qkv = nn.Linear(dim, dim * 3)
        
        # SNN LIF Neurons for Q, K, V projections
        self.lif_q = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lif_k = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.lif_v = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.proj = nn.Linear(dim, dim)
        self.lif_proj = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        
    def forward(self, x, T=40, H=4, W=4):
        """
        x: [T, B, N_spatial, C] for SNN multi-step
        """
        # If x is 3D [B, N, C], treat it as single step or reshape from [B, T*H*W, C]
        if x.dim() == 3:
            B, N, C = x.shape
            # Reshape to [T, B, H*W, C]
            x = x.view(B, T, H*W, C).transpose(0, 1).contiguous()
        
        T_step, B, N_spatial, C = x.shape
        
        # QKV Projection
        qkv = self.qkv(x).reshape(T_step, B, N_spatial, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [T, B, H_head, N_spatial, D_head]
        
        # Pass through LIF Neurons (Spike-driven)
        # LIFNode expects [T, B, ...]
        q = self.lif_q(q)
        k = self.lif_k(k)
        v = self.lif_v(v)

        # Region Partitioning Logic (H*W assumed to be N_spatial)
        n_win = self.n_win
        win_size = N_spatial // n_win

        # Reshape for regions: [T, B, H_head, n_win, win_size, D_head]
        q_r = q.reshape(T_step, B, self.num_heads, n_win, win_size, self.head_dim)
        k_r = k.reshape(T_step, B, self.num_heads, n_win, win_size, self.head_dim)
        v_r = v.reshape(T_step, B, self.num_heads, n_win, win_size, self.head_dim)

        # 3. Spike-Driven Region Features (Sum)
        q_region = q_r.sum(dim=4)  # [T, B, H_head, n_win, D_head]
        k_region = k_r.sum(dim=4)  # [T, B, H_head, n_win, D_head]

        # 2. Dynamic Routing Matrix (Per timestep)
        routing_sim = (q_region @ k_region.transpose(-2, -1)) * self.scale
        
        # Masking inactive regions (Zero spikes)
        region_activity = k_r.abs().sum(dim=(4, 5))  # [T, B, H_head, n_win]
        activity_mask = (region_activity > 1e-5).float()
        routing_sim = routing_sim + (1.0 - activity_mask.unsqueeze(-2)) * -1e9
        
        topk = min(self.topk, n_win)
        routing_weights, routing_indices = torch.topk(routing_sim, k=topk, dim=-1)
        routing_weights = routing_weights.softmax(dim=-1)

        # 1. Actual Sparse Attention (Gather)
        k_v_flat_shape = (T_step, B, self.num_heads, n_win * win_size, self.head_dim)
        k_flat = k_r.reshape(k_v_flat_shape)
        v_flat = v_r.reshape(k_v_flat_shape)
        
        # Index generation
        gather_indices = routing_indices.unsqueeze(-1) * win_size + torch.arange(win_size, device=x.device)
        gather_indices = gather_indices.view(T_step, B, self.num_heads, n_win, topk * win_size)
        
        # Reshaping for advanced indexing
        # Combine T and B and Head
        flat_B = T_step * B * self.num_heads
        k_flat_trans = k_flat.view(flat_B, n_win * win_size, self.head_dim)
        v_flat_trans = v_flat.view(flat_B, n_win * win_size, self.head_dim)
        gather_indices_trans = gather_indices.view(flat_B, n_win, topk * win_size)
        
        batch_idx = torch.arange(flat_B, device=x.device).view(-1, 1, 1)
        k_gathered = k_flat_trans[batch_idx, gather_indices_trans] 
        v_gathered = v_flat_trans[batch_idx, gather_indices_trans] 
        
        # Fine-grained Attention
        q_r_trans = q_r.reshape(flat_B, n_win, win_size, self.head_dim)
        attn = (q_r_trans @ k_gathered.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v_gathered 
        
        # Reshape back to [T, B, N_spatial, C]
        out = out.reshape(T_step, B, self.num_heads, n_win * win_size, self.head_dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(T_step, B, N_spatial, C)
        
        # Projection and final LIF
        out = self.lif_proj(self.proj(out))
        
        return out


