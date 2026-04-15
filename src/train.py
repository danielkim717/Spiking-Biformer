"""
Spiking Bi-Physformer 학습 및 평가 메인 스크립트
- Negative Pearson 상관계수 Loss (rPPG 표준)
- MAE, RMSE, Pearson r 평가 지표
- RTX 4060 CUDA 활용
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from spikingjelly.activation_based import functional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.datasets import get_rppg_dataloaders


# ============ Loss Function ============
class NegPearsonLoss(nn.Module):
    """Negative Pearson 상관계수 Loss (rPPG 학습에 표준 사용됨)"""
    def forward(self, pred, target):
        # pred, target: [B, T]
        pred_mean = pred.mean(dim=-1, keepdim=True)
        target_mean = target.mean(dim=-1, keepdim=True)
        pred_c = pred - pred_mean
        target_c = target - target_mean
        num = (pred_c * target_c).sum(dim=-1)
        den = torch.sqrt((pred_c ** 2).sum(dim=-1) * (target_c ** 2).sum(dim=-1) + 1e-8)
        pearson = num / den
        return (1 - pearson).mean()


# ============ Evaluation Metrics ============
def compute_metrics(preds, targets):
    """MAE, RMSE, Pearson r 계산"""
    preds = preds.flatten()
    targets = targets.flatten()
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    if len(preds) > 2:
        r, _ = pearsonr(preds, targets)
    else:
        r = 0.0
    return {'MAE': mae, 'RMSE': rmse, 'Pearson_r': r}


# ============ Training Loop ============
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for videos, labels in loader:
        videos = videos.to(device)    # [B, T, C, H, W]
        labels = labels.to(device)    # [B, T]

        optimizer.zero_grad()
        outputs = model(videos)       # [B, T]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # SNN 뉴런 상태 리셋 (매 배치 후 필수)
        functional.reset_net(model)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============ Evaluation Loop ============
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

        functional.reset_net(model)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


# ============ Main ============
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 학습 장치: {device}")
    if device.type == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    # DataLoader
    train_loader, val_loader = get_rppg_dataloaders(
        data_dir=args.data_dir,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type
    )
    print(f"[*] 학습 샘플: {len(train_loader.dataset)}, 검증 샘플: {len(val_loader.dataset)}")

    # Model
    model = SpikingBiPhysformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        topk=args.topk
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 모델 파라미터 수: {n_params:,}")

    # Loss & optimizer
    criterion = NegPearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_val_loss = float('inf')
    results_log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_MAE': val_metrics['MAE'],
            'val_RMSE': val_metrics['RMSE'],
            'val_Pearson_r': val_metrics['Pearson_r'],
            'time_sec': elapsed
        }
        results_log.append(log_entry)

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"MAE: {val_metrics['MAE']:.4f} | "
              f"RMSE: {val_metrics['RMSE']:.4f} | "
              f"Pearson r: {val_metrics['Pearson_r']:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Best model 저장
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/best_model_{args.dataset_type}.pth')
            print(f"  → Best model saved! (Val Loss: {best_val_loss:.4f})")

    # 최종 결과 저장
    os.makedirs('results', exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(results_log)
    df.to_csv(f'results/training_log_{args.dataset_type}.csv', index=False)
    print(f"\n[*] 학습 완료! 결과 저장: results/training_log_{args.dataset_type}.csv")

    return results_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spiking Bi-Physformer Training')
    parser.add_argument('--data_dir', type=str, default='./data/PURE', help='데이터셋 경로')
    parser.add_argument('--dataset_type', type=str, default='PURE', choices=['PURE', 'UBFC', 'MMPD', 'UBFC-Phys'])
    parser.add_argument('--clip_len', type=int, default=64, help='비디오 클립 길이 (프레임 수)')
    parser.add_argument('--img_size', type=int, default=128, help='입력 이미지 크기')
    parser.add_argument('--patch_size', type=int, default=8, help='패치 크기')
    parser.add_argument('--embed_dim', type=int, default=128, help='임베딩 차원')
    parser.add_argument('--depth', type=int, default=4, help='Transformer 블록 수')
    parser.add_argument('--num_heads', type=int, default=4, help='어텐션 헤드 수')
    parser.add_argument('--topk', type=int, default=4, help='Biformer routing top-k')
    parser.add_argument('--batch_size', type=int, default=2, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=30, help='에폭 수')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.parse_args()
    args = parser.parse_args()
    main(args)
