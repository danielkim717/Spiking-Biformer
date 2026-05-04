"""
Loss functions used by the active training pipeline.

NegPearsonLoss + FrequencyLoss (PhysFormer/Spiking-PhysFormer recipe).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        preds_mean = torch.mean(preds, dim=1, keepdim=True)
        labels_mean = torch.mean(labels, dim=1, keepdim=True)

        preds_std = preds - preds_mean
        labels_std = labels - labels_mean

        cov = torch.sum(preds_std * labels_std, dim=1)
        var_preds = torch.sqrt(torch.sum(preds_std ** 2, dim=1) + 1e-8)
        var_labels = torch.sqrt(torch.sum(labels_std ** 2, dim=1) + 1e-8)

        pearson = cov / (var_preds * var_labels)
        return 1.0 - torch.mean(pearson)


class FrequencyLoss(nn.Module):
    """PhysFormer / Spiking-PhysFormer Frequency loss (TorchLossComputer 재현).

    bpm range 40-180 (140 bins). Hanning window, 정규화된 complex_absolute (sum=1).
    target BPM index 는 labels 에서 argmax(complex_absolute) 로 추출.

    Returns (loss_ce, loss_ld):
      - loss_ce: F.cross_entropy(complex_absolute, target_idx)  (PhysFormer recipe 그대로)
      - loss_ld: KL(softmax(complex_absolute) || Gaussian(target_idx, std)) — DLDL_softmax2

    Reference:
      https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py
    """

    def __init__(self, fps=30, bpm_low=40, bpm_high=180, std=1.0):
        super().__init__()
        self.fps = fps
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high
        self.num_bpm = bpm_high - bpm_low
        self.std = std

    @staticmethod
    def _compute_complex_absolute_given_k(signal, k, N):
        """Hanning-windowed DFT magnitude squared at frequencies k.
        signal: (B, T)  k: (num_bpm,)  N: int  →  (B, num_bpm)"""
        device = signal.device
        two_pi_n_over_N = 2.0 * math.pi * torch.arange(0, N, dtype=torch.float32, device=device) / N
        hanning = torch.from_numpy(np.hanning(N).astype(np.float32)).to(device)
        windowed = signal * hanning.unsqueeze(0)
        phase = k.view(-1, 1) * two_pi_n_over_N.view(1, -1)
        sin_basis = torch.sin(phase)
        cos_basis = torch.cos(phase)
        sin_part = windowed @ sin_basis.t()
        cos_part = windowed @ cos_basis.t()
        return sin_part * sin_part + cos_part * cos_part

    def _complex_absolute(self, signal):
        B, N = signal.shape
        unit_per_hz = self.fps / N
        bpm_range = torch.arange(self.bpm_low, self.bpm_high, dtype=torch.float32, device=signal.device)
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        ca = self._compute_complex_absolute_given_k(signal, k, N)
        ca = ca / (ca.sum(dim=1, keepdim=True) + 1e-7)
        return ca

    def _gaussian_target_distribution(self, target_idx):
        bins = torch.arange(self.num_bpm, dtype=torch.float32, device=target_idx.device).view(1, -1)
        mean = target_idx.view(-1, 1).float()
        gauss = torch.exp(-(bins - mean) ** 2 / (2.0 * self.std ** 2)) / (math.sqrt(2.0 * math.pi) * self.std)
        gauss = torch.clamp(gauss, min=1e-15)
        gauss = gauss / gauss.sum(dim=1, keepdim=True)
        return gauss

    def forward(self, preds, labels):
        pred_ca = self._complex_absolute(preds)
        with torch.no_grad():
            label_ca = self._complex_absolute(labels)
            target_idx = torch.argmax(label_ca, dim=1)

        loss_ce = F.cross_entropy(pred_ca, target_idx)

        gauss_target = self._gaussian_target_distribution(target_idx)
        pred_softmax = F.softmax(pred_ca, dim=1)
        log_pred_softmax = torch.log(pred_softmax + 1e-15)
        loss_ld = F.kl_div(log_pred_softmax, gauss_target, reduction='batchmean')

        return loss_ce, loss_ld
