"""
ANN baseline (PhysFormer 원본) 으로 overfit-one 진단.

목적: SNN 적응이 학습 실패의 원인인지 확인.
- 같은 9 clip (UBFC subject1) train + test
- 같은 손실 (NegPearson + CE + LD)
- 같은 30 epoch, lr=3e-3
- 모델만 ANN PhysFormer (src/models/baseline/Physformer.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from src.models.baseline.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss


SUBJECT = ['subject1']
EPOCHS = 30
BATCH_SIZE = 2
LR = 1e-4   # PhysFormer 원논문 lr
WD = 5e-5
DATA_ROOT = 'D:\\UBFC-rPPG'
GRA_SHARP = 2.0   # PhysFormer hyperparameter

os.makedirs('results/diag_overfit_baseline', exist_ok=True)
LOG = 'results/diag_overfit_baseline/log.txt'
SAMPLES = 'results/diag_overfit_baseline/output_samples.txt'


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)


def per_clip_pearson(pred, gt):
    pearsons = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean()
        g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        pearsons.append(float((p*g).sum() / denom))
    return float(np.mean(pearsons)) if pearsons else 0.0


def main():
    open(LOG, 'w', encoding='utf-8').close()
    open(SAMPLES, 'w', encoding='utf-8').close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[*] ANN-baseline overfit on {device}")

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True)
    test_loader  = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=False)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    # ANN PhysFormer 원본
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128),
        patches=(4, 4, 4),
        dim=64,
        ff_dim=64 * 4,
        num_heads=4,
        num_layers=12,        # 12 // 3 = 4 layers per transformer block
        dropout_rate=0.1,
        theta=0.7,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  baseline params: {n_params}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        epoch_loss = 0.0
        nb = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            rppg, _, _, _ = model(inputs, GRA_SHARP)
            loss_p = pearson_criterion(rppg, labels)
            loss_ce, loss_ld = freq_criterion(rppg, labels)
            loss = loss_p + (loss_ce + loss_ld)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        avg_loss = epoch_loss / max(1, nb)

        # EVAL
        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                rppg, _, _, _ = model(inputs, GRA_SHARP)
                all_p.append(rppg.cpu())
                all_g.append(labels.cpu())
        preds = torch.cat(all_p).numpy()
        gts = torch.cat(all_g).numpy()

        mae = float(np.mean(np.abs(preds - gts)))
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        pred_std = float(preds.std())
        per_clip_std = float(np.mean([preds[i].std() for i in range(preds.shape[0])]))

        log(f"\nEpoch {epoch+1}/{EPOCHS}:")
        log(f"  Train avg loss: {avg_loss:.4f}")
        log(f"  Eval  MAE {mae:.4f}  RMSE {rmse:.4f}  pooled-Pearson {pooled:.4f}  per-clip-Pearson {per_clip:.4f}")
        log(f"  Output  std {pred_std:.4e}  per-clip avg-std {per_clip_std:.4e}")

        if epoch + 1 in {1, 5, 15, 30}:
            with open(SAMPLES, 'a', encoding='utf-8') as f:
                f.write(f"\n=== Epoch {epoch+1} ===\n")
                f.write(f"pred[0] (first 40 frames): {preds[0][:40].tolist()}\n")
                f.write(f"gt[0]   (first 40 frames): {gts[0][:40].tolist()}\n")
                f.write(f"pred[0] full std: {preds[0].std():.4e}, gt[0] full std: {gts[0].std():.4e}\n")

    log("\n[*] ANN-baseline overfit 완료.")


if __name__ == '__main__':
    main()
