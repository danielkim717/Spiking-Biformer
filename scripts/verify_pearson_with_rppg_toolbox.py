"""
rPPG-Toolbox 전처리 적용 후 Pearson 학습 sanity check.
- UBFC subject1, 9 clip overfit
- Spiking-PhysFormer + BiLevel Routing
- NegPearson only, 30 epoch
- 30 epoch 안에 Pearson 0.3+ 진입하면 정상 (이전 center-crop 버전과 비교용)
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
from src.train import NegPearsonLoss


SUBJECT = ['subject1']
EPOCHS = 30
BATCH_SIZE = 2
LR = 3e-3
WD = 5e-5
DATA_ROOT = 'D:\\UBFC-rPPG'

OUT_DIR = 'results/verify_rppgtb_preproc'
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
    log(f"[*] rPPG-Toolbox preproc verify on {device} (subject={SUBJECT})")

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True,
                                  standardize_input=True)
    test_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                 face_crop=True, subjects_filter=SUBJECT, shuffle=False,
                                 standardize_input=True)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    model = SpikingPhysformer(dim=96, num_blocks=4, num_heads=4, frame=160,
                              v_threshold=1.0, T_snn=4,
                              use_biformer=True, n_win=(2, 2, 2), topk=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    log(f"  params: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, nb = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = pearson_criterion(outputs, labels)
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
                functional.reset_net(model)
                all_p.append(outputs.cpu()); all_g.append(labels.cpu())
        preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        std = float(preds.std())
        log(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={avg_loss:.3f}  pooled-Pearson={pooled:.4f}  "
            f"per-clip-Pearson={per_clip:.4f}  pred_std={std:.3e}")

    log("\n[*] Done.")


if __name__ == '__main__':
    main()
