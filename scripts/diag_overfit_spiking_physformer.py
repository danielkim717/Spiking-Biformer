"""
순수 Spiking-PhysFormer (Biformer 없음) 으로 overfit-one 테스트.
같은 9 clip (UBFC subject1) train + test.
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
EPOCHS = 100   # 길게 돌려서 Pearson 추이 확인 (0.5 도달 여부)
BATCH_SIZE = 2
LR = 3e-3   # Spiking Physformer 논문 lr (그대로)
WD = 5e-5
DATA_ROOT = 'D:\\UBFC-rPPG'

os.makedirs('results/diag_spiking_physformer', exist_ok=True)
LOG = 'results/diag_spiking_physformer/log.txt'
SAMPLES = 'results/diag_spiking_physformer/output_samples.txt'


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
    log(f"[*] Pure Spiking-PhysFormer overfit on {device}")

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True)
    test_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                 face_crop=True, subjects_filter=SUBJECT, shuffle=False)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    model = SpikingPhysformer(dim=96, num_blocks=4, num_heads=4, frame=160, v_threshold=1.0, T_snn=4,
                              use_biformer=True, n_win=(2, 2, 2), topk=4).to(device)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")

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
            loss_p = pearson_criterion(outputs, labels)
            # NegPearson only — freq loss 가 Pearson 학습 방해 확인됨
            loss = loss_p
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            epoch_loss += loss.item(); nb += 1
        avg_loss = epoch_loss / max(1, nb)

        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_p.append(outputs.cpu()); all_g.append(labels.cpu())
        preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()

        mae = float(np.mean(np.abs(preds - gts)))
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        pred_std = float(preds.std())

        log(f"\nEpoch {epoch+1}/{EPOCHS}:")
        log(f"  Train avg loss: {avg_loss:.4f}")
        log(f"  Eval  MAE {mae:.4f}  RMSE {rmse:.4f}  pooled-Pearson {pooled:.4f}  per-clip-Pearson {per_clip:.4f}")
        log(f"  Output std {pred_std:.4e}")

        if epoch + 1 in {1, 5, 15, 30, 50, 75, 100}:
            with open(SAMPLES, 'a', encoding='utf-8') as f:
                f.write(f"\n=== Epoch {epoch+1} ===\n")
                f.write(f"pred[0] (first 40): {preds[0][:40].tolist()}\n")
                f.write(f"gt[0]   (first 40): {gts[0][:40].tolist()}\n")

    log("\n[*] Done.")


if __name__ == '__main__':
    main()
