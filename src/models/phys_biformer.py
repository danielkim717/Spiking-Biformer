import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from src.models.biformer_attention import BiLevelRoutingAttention


class PhysBiformerBlock(nn.Module):
    def __init__(self, dim, num_heads, v_threshold, n_win=(2, 4, 4), topk=4):
        super().__init__()
        # Sub-block 1: Spike-driven attention (BN -> LIF -> Attn).
        self.bn1 = layer.BatchNorm3d(dim)
        self.lif1 = neuron.LIFNode(
            v_threshold=v_threshold, v_reset=None,
            surrogate_function=surrogate.ATan(), detach_reset=True,
        )
        self.attn = BiLevelRoutingAttention(
            dim, num_heads=num_heads, n_win=n_win, topk=topk, v_threshold=v_threshold,
        )

        # Sub-block 2: FFN (BN -> LIF -> Conv-GELU-Conv).
        self.bn2 = layer.BatchNorm3d(dim)
        self.lif2 = neuron.LIFNode(
            v_threshold=v_threshold, v_reset=None,
            surrogate_function=surrogate.ATan(), detach_reset=True,
        )
        self.ffn = nn.Sequential(
            layer.Conv3d(dim, dim * 4, kernel_size=1),
            layer.BatchNorm3d(dim * 4),
            nn.GELU(),
            layer.Conv3d(dim * 4, dim, kernel_size=1),
            layer.BatchNorm3d(dim),
        )

        # MS-Shortcut residual scale. Fixed at 1.0 to avoid the trainable scalar
        # collapsing toward zero and disabling the attention path.
        self.register_buffer("scale", torch.tensor(1.0))

        # Ensure spikingjelly modules run in multi-step mode.
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        """
        x: [T, B, C, Lt, Lh, Lw] — real-valued membrane-potential path.
        """
        # ----- Spike-driven attention -----
        identity = x
        x_norm = self.bn1(x)
        spikes = self.lif1(x_norm)
        self.last_firing_rate1 = spikes.detach().mean().item()

        spikes_attn = spikes.permute(0, 1, 3, 4, 5, 2)            # [T, B, Lt, Lh, Lw, C]
        attn_out = self.attn(spikes_attn)
        attn_out = attn_out.permute(0, 1, 5, 2, 3, 4)             # [T, B, C, Lt, Lh, Lw]
        x = identity + attn_out * self.scale

        # ----- Spike-driven FFN -----
        identity = x
        x_norm = self.bn2(x)
        spikes = self.lif2(x_norm)
        self.last_firing_rate2 = spikes.detach().mean().item()
        x = identity + self.ffn(spikes) * self.scale
        return x


class PhysBiformer(nn.Module):
    """
    Spiking Bi-Physformer for rPPG.

    Input  : [B, 3, T_video, H, W]    (e.g. T_video=160, H=W=128)
    Output : [B, T_video]             (rPPG waveform)
    """

    def __init__(
        self,
        dim=64,
        num_blocks=3,
        num_heads=4,
        v_threshold=0.5,
        frame=160,
        patches=(4, 4, 4),
        T_snn=4,
        topk=4,
        n_win=(4, 4, 4),
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.frame = frame
        self.T_snn = T_snn
        self.patches = patches
        self.n_win = n_win

        ft, fh, fw = patches

        # ---- ANN Stem -----------------------------------------------------
        # Two-stage 3D conv. NOTE: do NOT end with ReLU here — feeding a
        # non-negative tensor straight into a BN→LIF stack leaves half of the
        # neurons silent after BN, which was a major contributor to the
        # ~5% firing rate observed earlier.
        self.stem = nn.Sequential(
            nn.Conv3d(3, dim // 2, kernel_size=(3, 3, 3), stride=(1, fh // 2, fw // 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(dim // 2),
            nn.GELU(),
            nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3), stride=(ft, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(dim),
        )

        # The temporal length after the stem is frame // ft.
        Lt_feat = frame // ft
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, Lt_feat, 1, 1))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            PhysBiformerBlock(dim, num_heads, v_threshold, n_win=n_win, topk=topk)
            for _ in range(num_blocks)
        ])

        # ANN regression head.
        self.head_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

        functional.set_step_mode(self, step_mode='m')
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv3d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: [B, 3, T_video, H, W]
        """
        b = x.shape[0]

        # 1. ANN Stem -> [B, C, Lt, Lh, Lw] real-valued features.
        feat = self.stem(x)
        feat = feat + self.pos_embed[:, :, : feat.shape[2], :, :]

        # 2. SNN potential-path initialisation.
        #    Repeat across T_snn timesteps. With v_reset=None (soft reset) the
        #    LIF neurons accumulate non-trivial dynamics over T even with the
        #    same input, and the linear projections in attention diversify per
        #    timestep through batch-norm running stats.
        x_snn = feat.unsqueeze(0).expand(self.T_snn, -1, -1, -1, -1, -1).contiguous()

        firing_rates = []
        for block in self.blocks:
            x_snn = block(x_snn)
            firing_rates.append(block.last_firing_rate1)
            firing_rates.append(block.last_firing_rate2)
        self.last_firing_rates = firing_rates

        # 3. Aggregate over SNN timesteps and spatial dims.
        x_out = x_snn.mean(dim=0)              # [B, C, Lt, Lh, Lw]
        x_out = x_out.mean(dim=(3, 4))         # [B, C, Lt]

        # 4. Upsample temporal axis back to the target frame count.
        x_out = F.interpolate(x_out, size=self.frame, mode='linear', align_corners=True)
        x_out = x_out.transpose(1, 2)          # [B, T, C]
        x_out = self.head_norm(x_out)
        rppg = self.head(x_out).squeeze(-1)    # [B, T]

        # Mean-centre per-sample so the loss focuses on waveform shape rather
        # than absolute offset (the BVP target is z-normalised in the dataset).
        rppg = rppg - rppg.mean(dim=1, keepdim=True)
        return rppg
