"""
Overfit sanity test: 단 1 subject 로 train + 같은 subject test.

목적: 모델이 BVP 신호를 학습할 수 있는 fundamental capacity 가 있는지 확인.
- 데이터: UBFC subject1 만 사용 (train/test 동일)
- Epoch: 30 (overfit 까지)
- 매 epoch 평가 시 Pearson + 출력 분포 통계 출력
- 모델 출력의 첫 batch 첫 sample 의 시계열을 파일로 저장

판정:
- Train Pearson 이 +0.5 이상 도달 → 모델은 학습 가능. 다른 곳에 문제.
- Train Pearson 이 끝까지 0 근처 → 모델/학습 루프 자체에 결함.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from spikingjelly.activation_based import functional

from src.models.phys_biformer import PhysBiformer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.utils.metrics import calculate_metrics


SUBJECT = ['subject1']
EPOCHS = 30
BATCH_SIZE = 2
LR = 1e-4   # PhysFormer 원논문 lr (was 3e-3 — too high)
WD = 5e-5
DATA_ROOT = 'D:\\UBFC-rPPG'

os.makedirs('results/diag_overfit', exist_ok=True)
SAMPLE_FILE = 'results/diag_overfit/output_samples.txt'
LOG_FILE = 'results/diag_overfit/log.txt'


def log(msg, also_stdout=True):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    if also_stdout:
        print(msg, flush=True)


def per_clip_pearson(pred, gt):
    pearsons = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean()
        g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        pearsons.append(float((p*g).sum() / denom))
    return float(np.mean(pearsons)), pearsons


def main():
    open(LOG_FILE, 'w', encoding='utf-8').close()
    open(SAMPLE_FILE, 'w', encoding='utf-8').close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"[*] Overfit-one diagnostic on {device}, subject = {SUBJECT}")

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=SUBJECT, shuffle=True)
    test_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                 face_crop=True, subjects_filter=SUBJECT, shuffle=False)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    model = PhysBiformer(frame=160, patches=(4, 4, 4), v_threshold=1.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  model params: {n_params}")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        epoch_loss = 0.0
        nb = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_p = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)
            loss = loss_p + (loss_ce + loss_ld)
            loss.backward()

            # gradient norm
            gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gn += float(p.grad.detach().pow(2).sum().item())
            gn = gn ** 0.5

            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
            functional.reset_net(model)
        avg_train_loss = epoch_loss / max(1, nb)

        # --- EVAL ---
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_preds.append(outputs.cpu())
                all_gts.append(labels.cpu())
        preds = torch.cat(all_preds).numpy()
        gts = torch.cat(all_gts).numpy()

        # 메트릭
        mae = float(np.mean(np.abs(preds - gts)))
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip_mean, per_clip_list = per_clip_pearson(preds, gts)

        # 출력 분포
        pred_mean = float(preds.mean())
        pred_std = float(preds.std())
        pred_per_clip_std = [float(preds[i].std()) for i in range(preds.shape[0])]
        avg_clip_std = float(np.mean(pred_per_clip_std))

        # 발화율
        fr = getattr(model, 'last_firing_rates', [])
        fr_avg = float(np.mean(fr)) if fr else 0.0

        log(f"\nEpoch {epoch+1}/{EPOCHS}:")
        log(f"  Train avg loss: {avg_train_loss:.4f}   (last grad-norm {gn:.4f})")
        log(f"  Eval  MAE {mae:.4f}  RMSE {rmse:.4f}  pooled-Pearson {pooled:.4f}  per-clip-Pearson {per_clip_mean:.4f}")
        log(f"  Output  mean {pred_mean:.4e}  std {pred_std:.4e}  per-clip avg-std {avg_clip_std:.4e}")
        log(f"  Firing rate avg {fr_avg*100:.2f}%")

        # epoch 1, 5, 15, 30 일 때 출력 sample 저장
        if epoch + 1 in {1, 5, 15, 30}:
            with open(SAMPLE_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n=== Epoch {epoch+1} ===\n")
                f.write(f"pred[0] (first 40 frames): {preds[0][:40].tolist()}\n")
                f.write(f"gt[0]   (first 40 frames): {gts[0][:40].tolist()}\n")
                f.write(f"pred[0] full std: {preds[0].std():.4e}, gt[0] full std: {gts[0].std():.4e}\n")

    log("\n[*] Overfit-one diagnostic 완료.")


if __name__ == '__main__':
    main()
