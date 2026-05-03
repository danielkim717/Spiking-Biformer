"""
PhysFormer ANN baseline 사전학습 → PE block (Stem+patch_embedding) weight 추출.

paper Spiking-PhysFormer (Liu et al., 2024) Section 4.2 인용:
  "we initialize our model by pretraining PhysFormer and extracting the weights
   of the PE block as pre-trained parameters"

순서:
  1) PhysFormer (12-layer, ANN) 를 지정 데이터셋에서 10 epoch 학습
  2) Stem0/1/2 + patch_embedding state_dict 만 추출
  3) checkpoints/pretrained_pe_{dataset}.pt 로 저장

사용:
  python scripts/pretrain_physformer_pe.py --dataset PURE
  python scripts/pretrain_physformer_pe.py --dataset UBFC-rPPG
"""
import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from src.models.physformer_baseline import PhysFormer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss


EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4              # PhysFormer 논문 default (SNN 보다 작음)
WD = 5e-5
ALPHA = 0.1
BETA0 = 1.0
ETA = 5.0
DETECTION_FREQ = 30
DATASET_PATHS = {'PURE': 'D:\\PURE', 'UBFC-rPPG': 'D:\\UBFC-rPPG'}
CKPT_DIR = 'checkpoints'


def per_clip_pearson(pred, gt):
    out = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean(); g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        out.append(float((p*g).sum() / denom))
    return float(np.mean(out)) if out else 0.0


def main(dataset_name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_dir = f'results/pretrain_{dataset_name}'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    open(log_path, 'w', encoding='utf-8').close()

    def log(msg):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        print(msg, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[*] PhysFormer pretrain on {dataset_name} ({device})")
    log(f"  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}  WD={WD}")

    # paper / rPPG-Toolbox 표준: source train 80%, valid 20%. PE pretraining 도
    # 80% train 사용 (target test 정보 leakage 방지).
    loader = get_dataloader(dataset_name, DATASET_PATHS[dataset_name], BATCH_SIZE,
                            clip_len=160, face_crop=True, shuffle=True,
                            data_type='diff_normalized',
                            dynamic_detection_freq=DETECTION_FREQ,
                            split_range=(0.0, 0.8))
    log(f"  train clips (80%): {len(loader.dataset)}")

    model = PhysFormer(dim=96, ff_dim=144, num_heads=4, num_layers=12,
                       frame=160, image_size=128, dropout=0.1, theta=0.7).to(device)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = EPOCHS * len(loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    for epoch in range(EPOCHS):
        beta = BETA0 * (ETA ** (epoch / EPOCHS))
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  α={ALPHA}, β={beta:.4f}")

        model.train()
        epoch_loss, nb = 0.0, 0
        t0 = time.time()
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_p = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)
            loss = ALPHA * loss_p + beta * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item()); nb += 1
            if i % 20 == 0:
                log(f"  step {i}/{len(loader)}  loss={loss.item():.3f}  lr={scheduler.get_last_lr()[0]:.2e}")
        avg_loss = epoch_loss / max(1, nb)
        elapsed = time.time() - t0
        log(f"  Epoch {epoch+1} avg loss {avg_loss:.4f}  ({elapsed:.0f}s)")

    # Save PE block weights only
    pe_sd = model.export_pe_state_dict()
    out_path = os.path.join(CKPT_DIR, f'pretrained_pe_{dataset_name}.pt')
    torch.save(pe_sd, out_path)
    log(f"\n[*] Saved PE block weights → {out_path}")
    log(f"  num tensors: {len(pe_sd)}")
    log(f"  total params: {sum(v.numel() for v in pe_sd.values())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PURE',
                        choices=list(DATASET_PATHS.keys()))
    args = parser.parse_args()
    main(args.dataset)
