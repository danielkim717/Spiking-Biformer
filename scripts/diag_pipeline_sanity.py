"""
학습 파이프라인 sanity test:
1. GT 를 pred 로 그대로 사용 → Pearson 이 1.0 인지
2. NegPearson 만 사용해서 학습 → Pearson 이 빠르게 양수 영역으로 가는지
3. Freq loss 만 사용해서 학습 → freq loss 가 Pearson 을 도와주는지 vs 방해하는지
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from spikingjelly.activation_based import functional

from src.models.spiking_physformer import SpikingPhysformer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss


SUBJECT = ['subject1']
EPOCHS = 15
BATCH_SIZE = 2
LR = 3e-3
WD = 5e-5
DATA_ROOT = 'D:\\UBFC-rPPG'

os.makedirs('results/diag_pipeline', exist_ok=True)
LOG = 'results/diag_pipeline/log.txt'


def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)


def per_clip_pearson(pred, gt):
    out = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean(); g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        out.append(float((p*g).sum() / denom))
    return float(np.mean(out)) if out else 0.0


def collect_data(loader):
    all_g = []
    for _, labels in loader:
        all_g.append(labels)
    return torch.cat(all_g)


def test_1_gt_as_pred():
    log("\n" + "=" * 60)
    log("TEST 1: GT 를 pred 로 그대로 사용 (Pearson 1.0 기대)")
    log("=" * 60)

    loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                            face_crop=True, subjects_filter=SUBJECT, shuffle=False)
    gts = collect_data(loader).numpy()
    preds = gts.copy()

    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
    per_clip = per_clip_pearson(preds, gts)
    log(f"  pred = gt:  MAE {mae:.4f}  RMSE {rmse:.4f}  pooled-Pearson {pooled:.4f}  per-clip {per_clip:.4f}")
    log(f"  → 1.0 이 나와야 metric 정상")


def test_2_train_with(loss_combo):
    label = '+'.join(loss_combo)
    log("\n" + "=" * 60)
    log(f"TEST 2/3: 학습 (loss = {label})")
    log("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True)
    test_loader  = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=False)

    model = SpikingPhysformer(dim=96, num_blocks=4, num_heads=4, frame=160, v_threshold=1.0, T_snn=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, nb = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0.0
            if 'pearson' in loss_combo:
                loss = loss + pearson_criterion(outputs, labels)
            if 'freq' in loss_combo:
                ce, ld = freq_criterion(outputs, labels)
                loss = loss + ce + ld
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            epoch_loss += float(loss.item()); nb += 1

        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_p.append(outputs.cpu()); all_g.append(labels.cpu())
        preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        std = float(preds.std())
        log(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={epoch_loss/max(1,nb):.3f}  Pearson(pooled)={pooled:.4f}  "
            f"Pearson(per-clip)={per_clip:.4f}  std={std:.3e}")


def main():
    open(LOG, 'w', encoding='utf-8').close()
    test_1_gt_as_pred()
    test_2_train_with(['pearson'])           # NegPearson only
    test_2_train_with(['freq'])              # Freq only
    log("\n[*] Done.")


if __name__ == '__main__':
    main()
