"""
학습 로그 및 상태 상세 기록을 포함한 Train 스크립트.
에포크마다 평가를 수행하도록 개선되었습니다.
"""
import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from src.models.phys_biformer import PhysBiformer
from src.data.rppg_dataset import get_dataloader
from src.utils.metrics import calculate_metrics, update_experiment_summary

def save_status(status):
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

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

def run_experiment(train_ds, test_ds, epochs=30, batch_size=2, v_threshold=1.0, lr=3e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Experiment: {train_ds} -> {test_ds} on {device} (V_th={v_threshold}, LR={lr})")

    os.makedirs('results', exist_ok=True)

    dataset_paths = {
        'PURE': 'D:\\PURE',
        'UBFC-rPPG': 'D:\\UBFC-rPPG'
    }

    train_loader = get_dataloader(train_ds, dataset_paths[train_ds], batch_size, clip_len=160)
    test_loader = get_dataloader(test_ds, dataset_paths[test_ds], batch_size, clip_len=160)

    if len(train_loader) == 0 or len(test_loader) == 0:
        print(f"[!] {train_ds} 또는 {test_ds} 샘플을 찾지 못했습니다.")
        return

    model = PhysBiformer(frame=160, patches=(4, 4, 4), v_threshold=v_threshold).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_pearson = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)

            # PhysFormer/Spiking-PhysFormer 류:
            #   L = α · L_NegPearson + β · (L_CE + L_LD)
            # 우리 freq loss 가 raw PSD 를 logits 으로 사용해 학습 초기 magnitude 가
            # 매우 큼 (epoch 1 ≈ 22). 결합 시 NegPearson 이 freq 에 묻혀 phase 학습이
            # 안되는 현상이 진단으로 확인됨 → β 를 작게 (0.1) 하여 NegPearson 우세.
            loss = loss_pearson + 0.1 * (loss_ce + loss_ld)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            functional.reset_net(model)

            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.6f}")
                save_status({
                    'experiment': f"{train_ds}->{test_ds}",
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'step': i,
                    'total_steps': len(train_loader),
                    'loss': loss.item(),
                    'phase': 'Training',
                    'firing_rates': getattr(model, 'last_firing_rates', [])
                })
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.6f}")
        
        # Intermediate Evaluation every epoch
        print(f"[*] Starting Evaluation for Epoch {epoch+1}...")
        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_preds.append(outputs.cpu())
                all_gts.append(labels.cpu())
                
                if j % 10 == 0:
                    save_status({
                        'experiment': f"{train_ds}->{test_ds}",
                        'epoch': epoch + 1,
                        'total_epochs': epochs,
                        'step': j,
                        'total_steps': len(test_loader),
                        'loss': 0.0,
                        'phase': 'Evaluation'
                    })
                    
        all_preds = torch.cat(all_preds)
        all_gts = torch.cat(all_gts)
        metrics = calculate_metrics(all_preds, all_gts)
        print(f"[*] Epoch {epoch+1} Results: {metrics}")
        update_experiment_summary(f"{train_ds}->{test_ds}", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ds', type=str, required=True)
    parser.add_argument('--test_ds', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--v_threshold', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-3)
    args = parser.parse_args()
    run_experiment(args.train_ds, args.test_ds, args.epochs, v_threshold=args.v_threshold, lr=args.lr)
