"""
PhysFormer / Spiking-PhysFormer 논문 셋업 검증:
  - DiffNormalized data + label
  - Dynamic face detection @30 frames
  - Loss = 0.1·NegPearson + β·(CE+LD), β = 1·5^((e-1)/E)
  - Adam, lr=3e-3
UBFC subject1 9-clip overfit, 10 epoch.
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
EPOCHS = 10
BATCH_SIZE = 2
LR = 3e-3
WD = 5e-5
ALPHA = 0.1
BETA0 = 1.0
ETA = 5.0
DATA_ROOT = 'D:\\UBFC-rPPG'

OUT_DIR = 'results/verify_paper_config'
os.makedirs(OUT_DIR, exist_ok=True)
LOG = os.path.join(OUT_DIR, 'log.txt')


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


def main():
    open(LOG, 'w', encoding='utf-8').close()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[*] Paper-config verify on {device} (subject={SUBJECT})")

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True,
                                  data_type='diff_normalized', dynamic_detection_freq=30)
    test_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                 face_crop=True, subjects_filter=SUBJECT, shuffle=False,
                                 data_type='diff_normalized', dynamic_detection_freq=30)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    model = SpikingPhysformer(dim=96, num_blocks=4, num_heads=4, frame=160,
                              v_threshold=1.0, T_snn=4,
                              use_biformer=True, n_win=(2, 2, 2), topk=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)
    log(f"  params: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(EPOCHS):
        beta = BETA0 * (ETA ** (epoch / EPOCHS))
        model.train()
        epoch_loss, nb = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-7)
            loss_p = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)
            loss = ALPHA * loss_p + beta * (loss_ce + loss_ld)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            epoch_loss += float(loss.item()); nb += 1
        avg_loss = epoch_loss / max(1, nb)

        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = (outputs - outputs.mean()) / (outputs.std() + 1e-7)
                functional.reset_net(model)
                all_p.append(outputs.cpu()); all_g.append(labels.cpu())
        preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        std = float(preds.std())
        log(f"  Epoch {epoch+1:2d}/{EPOCHS}: β={beta:.3f}  loss={avg_loss:.3f}  "
            f"Pearson(pooled)={pooled:.4f}  Pearson(per-clip)={per_clip:.4f}  std={std:.3e}")

    log("\n[*] Done.")


if __name__ == '__main__':
    main()
