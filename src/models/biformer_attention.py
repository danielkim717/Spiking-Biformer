import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLevelRoutingAttention(nn.Module):
    """
    Spike-Driven Bi-Level Routing Attention (S-BRA) for SpikingBiformer.
    1. Actual Sparse Attention: Uses gather/index_select for efficient sparse attention.
    2. Dynamic Routing over Time: Calculates routing indices per timestep instead of averaging.
    3. Spike-Driven Region Features: Uses sum instead of mean to reflect firing rate.
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
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, T=40, H=4, W=4):
        """
        x: [B, N, C] where N = T * H * W
        """
        B, N, C = x.shape
        # Reshape to separate Time and Space to ensure Dynamic Routing over Time
        # x is [B, T, H*W, C]
        x = x.view(B, T, H*W, C)
        
        qkv = self.qkv(x).reshape(B, T, H*W, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, T, H_head, H*W, D_head]

        # Ensure spatial_N is divisible by n_win
        spatial_N = H * W
        if spatial_N % self.n_win != 0:
            # Fallback or adjust
            win_size = 1
            n_win = spatial_N
        else:
            n_win = self.n_win
            win_size = spatial_N // n_win

        # Region Partition
        q_r = q.reshape(B, T, self.num_heads, n_win, win_size, self.head_dim)
        k_r = k.reshape(B, T, self.num_heads, n_win, win_size, self.head_dim)
        v_r = v.reshape(B, T, self.num_heads, n_win, win_size, self.head_dim)

        # 3. Spike-Driven Region Features (Sum instead of Mean)
        # This reflects the firing rate importance in SNN
        q_region = q_r.sum(dim=4)  # [B, T, H_head, n_win, D_head]
        k_region = k_r.sum(dim=4)  # [B, T, H_head, n_win, D_head]

        # 2. Dynamic Routing Index (Per Timestep)
        # routing_sim: [B, T, H_head, n_win, n_win]
        routing_sim = (q_region @ k_region.transpose(-2, -1)) * self.scale
        
        # Filtering regions with low spike activity (Exclude regions with zero spikes)
        # Calculate spike activity per region
        region_activity = k_r.abs().sum(dim=(4, 5))  # [B, T, H_head, n_win]
        activity_mask = (region_activity > 1e-5).float()
        # Apply mask to routing_sim (set low score for inactive regions)
        routing_sim = routing_sim + (1.0 - activity_mask.unsqueeze(-2)) * -1e9
        
        topk = min(self.topk, n_win)
        routing_weights, routing_indices = torch.topk(routing_sim, k=topk, dim=-1)  # [B, T, H_head, n_win, topk]
        routing_weights = routing_weights.softmax(dim=-1)

        # 1. Actual Sparse Attention (Using Gather-like indexing for efficiency)
        # We need to gather keys and values from the routed regions
        # Expand k_r and v_r for gathering
        # k_r: [B, T, H_head, n_win, win_size, D_head]
        
        # Flatten k_r to [B, T, H_head, n_win * win_size, D_head] to use efficient indexing
        k_v_flat_shape = (B, T, self.num_heads, n_win * win_size, self.head_dim)
        k_flat = k_r.reshape(k_v_flat_shape)
        v_flat = v_r.reshape(k_v_flat_shape)
        
        # Create indices for all tokens in the top-k regions
        # routing_indices: [B, T, H, n_win, topk]
        # For each window, we want (topk * win_size) tokens
        
        # Shift indices to point to the correct tokens in the flattened k_flat
        # idx_offset = [0, win_size, 2*win_size, ...]
        # This part is a bit complex for a single step, but we can do it:
        
        # Use a more readable gather implementation
        # (B, T, H, n_win, topk, win_size)
        gather_indices = routing_indices.unsqueeze(-1) * win_size + torch.arange(win_size, device=x.device)
        gather_indices = gather_indices.view(B, T, self.num_heads, n_win, topk * win_size)
        
        # Expand gather_indices for the last dimension (head_dim)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.head_dim)
        
        # Gather k and v: [B, T, H_head, n_win, topk * win_size, D_head]
        # Since torch.gather is 1D on the specified dim, we need to be careful.
        # Alternatively, we can use advanced indexing if we reshape carefully.
        
        # Reshaping for advanced indexing
        k_flat_trans = k_flat.view(B * T * self.num_heads, n_win * win_size, self.head_dim)
        v_flat_trans = v_flat.view(B * T * self.num_heads, n_win * win_size, self.head_dim)
        gather_indices_trans = gather_indices.view(B * T * self.num_heads, n_win, topk * win_size)
        
        # Batch indexing
        batch_idx = torch.arange(B * T * self.num_heads, device=x.device).view(-1, 1, 1)
        k_gathered = k_flat_trans[batch_idx, gather_indices_trans] # [BT H, n_win, topk*win_size, D]
        v_gathered = v_flat_trans[batch_idx, gather_indices_trans] # [BT H, n_win, topk*win_size, D]
        
        # Fine-grained Attention
        # q_r: [B, T, H_head, n_win, win_size, D_head]
        q_r_trans = q_r.reshape(B * T * self.num_heads, n_win, win_size, self.head_dim)
        
        attn = (q_r_trans @ k_gathered.transpose(-2, -1)) * self.scale # [..., n_win, win_size, topk*win_size]
        attn = attn.softmax(dim=-1)
        
        out = attn @ v_gathered # [..., n_win, win_size, D_head]
        
        # Reshape back
        out = out.reshape(B, T, self.num_heads, n_win * win_size, self.head_dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N // T, C)
        out = out.view(B, N, C)
        
        out = self.proj(out)
        return out

