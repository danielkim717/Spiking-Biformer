"""BiPulseFormer intra 7/1/2 — rPPG-Toolbox 표준 (OneCycleLR + 20 epochs).

PURE: subject-exclusive RANDOM (seed=42)
  TRAIN (0.0-0.7): 7 subjects (03,04,06,07,08,09,10)  - subject 07 outlier 포함
  VALID (0.7-0.8): 1 subject (05)
  TEST  (0.8-1.0): 2 subjects (01,02)

UBFC: subject-exclusive sort-based
  TRAIN (0.0-0.7): ~29 subjects
  VALID (0.7-0.8): ~4 subjects
  TEST  (0.8-1.0): ~9 subjects

Setup (rPPG-Toolbox PhysFormerTrainer 표준):
  - OneCycleLR(max_lr=1e-4, epochs=20, steps_per_epoch=N)
  - 20 epochs
  - Loss schedule: epoch≤10 a=1.0, b=1.0;  epoch>10 a=0.05, b=5.0
"""
import os, sys, io, json, time, random
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim
from scipy.signal import welch

from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.evaluation import evaluate_per_subject, get_subject_signals
from src.evaluation_per_clip import evaluate_per_clip


def get_hr_welch(y, sr=30, hr_min=30, hr_max=180):
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-9:
        return 0.0
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=int(np.min((len(y) - 1, 256))))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not mask.any():
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)


EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
WD = 5e-5
ALPHA_START = 1.0
BETA_START = 1.0
ALPHA_AFTER10 = 0.05
BETA_AFTER10 = 5.0
GRA_SHARP = 2.0
DETECTION_FREQ = 0
GRAD_CLIP = 1.0
SEED = 42
FPS = 30
PURE_PATH = 'D:\\PURE'
UBFC_PATH = 'D:\\UBFC-rPPG'


def per_clip_pearson(pred, gt):
    out = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean(); g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        out.append(float((p*g).sum() / denom))
    return float(np.mean(out)) if out else 0.0


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate_loader(model, loader, device):
    model.eval()
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=GRA_SHARP)
            rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                   (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    preds = np.concatenate(all_p); gts = np.concatenate(all_g)
    pc = evaluate_per_clip(preds, gts, fs=FPS, diff_flag=True,
                           low_pass=0.75, high_pass=2.5)
    ps = evaluate_per_subject(preds, gts, loader.dataset.samples,
                              fs=FPS, diff_flag=True, low_pass=0.75, high_pass=2.5)
    return {
        'MAE_clip': pc['MAE_bpm_clip'], 'RMSE_clip': pc['RMSE_bpm_clip'],
        'Pearson_clip': pc['Pearson_clip'], 'n_clips': pc['n_clips'],
        'MAE_subj': ps['MAE_bpm'], 'RMSE_subj': ps['RMSE_bpm'],
        'Pearson_subj': ps['Pearson'], 'signal_Pearson': ps['signal_Pearson_mean'],
        'n_subjects': ps['n_subjects'],
    }, preds, gts


def run(name, path, result_dir, use_random_subj=False):
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, 'log.txt')
    def log(msg):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        print(msg, flush=True)
    open(log_file, 'w', encoding='utf-8').close()

    log("=" * 70); log(f"[*] {name} intra 7/1/2 — rPPG-Toolbox setup"); log("=" * 70)
    log(f"  OneCycleLR(max_lr={LR}, epochs={EPOCHS})")
    log(f"  Loss schedule: epoch≤10 a={ALPHA_START},b={BETA_START}; "
        f"epoch>10 a={ALPHA_AFTER10},b={BETA_AFTER10}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    common = dict(face_crop=True, dynamic_detection_freq=DETECTION_FREQ,
                  data_type='diff_normalized')
    if name == 'PURE':
        if use_random_subj:
            common['pure_split_mode'] = 'subject_exclusive_random'
        else:
            common['pure_split_mode'] = 'subject_exclusive'

    train_loader = get_dataloader(name, path, BATCH_SIZE, clip_len=160, fps=FPS,
                                  shuffle=True, random_hflip=True, hr_filter=True,
                                  split_range=(0.0, 0.7), **common)
    valid_loader = get_dataloader(name, path, BATCH_SIZE, clip_len=160,
                                  shuffle=False, chunk_step=80,
                                  split_range=(0.7, 0.8), **common)
    test_loader = get_dataloader(name, path, BATCH_SIZE, clip_len=160,
                                 shuffle=False, chunk_step=80,
                                 split_range=(0.8, 1.0), **common)
    log(f"  train clips: {len(train_loader.dataset)}")
    log(f"  valid clips: {len(valid_loader.dataset)}")
    log(f"  test  clips: {len(test_loader.dataset)}")

    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    # OneCycleLR: max_lr=1e-4, total_steps=EPOCHS * batches_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")

    best_valid_rmse = float('inf')
    best_test = None
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        if epoch >= 10:  # epoch index 10 means actual 11th epoch (epoch>10 in 1-indexed)
            a, b = ALPHA_AFTER10, BETA_AFTER10
        else:
            a, b = ALPHA_START, BETA_START
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  alpha={a}, beta={b}")
        model.train()
        epoch_loss, nb = 0.0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            hr_target = torch.tensor(
                [get_hr_welch(labels[k].cpu().numpy(), sr=FPS) for k in range(labels.shape[0])],
                dtype=torch.float32, device=device,
            )
            optimizer.zero_grad()
            rPPG, _, _, _ = model(inputs, gra_sharp=GRA_SHARP)
            rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                   (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
            loss_p = pearson_criterion(rPPG, labels)
            loss_ce, loss_ld = freq_criterion(rPPG, hr_target)
            loss = a * loss_p + b * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()  # OneCycleLR: per-batch step
            epoch_loss += float(loss.item()); nb += 1
        avg_loss = epoch_loss / max(1, nb)
        current_lr = optimizer.param_groups[0]['lr']

        valid_m, _, _ = evaluate_loader(model, valid_loader, device)
        test_m, test_preds, test_gts = evaluate_loader(model, test_loader, device)

        ckpt_dir = os.path.join(result_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{name}_epoch{epoch+1}.pt'))

        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'lr': current_lr,
                        'valid': valid_m, 'test': test_m})
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train loss {avg_loss:.4f}  LR={current_lr:.2e}")
        log(f"  VALID per-clip MAE {valid_m['MAE_clip']:.3f}  RMSE {valid_m['RMSE_clip']:.3f}  "
            f"Pearson {valid_m['Pearson_clip']:.4f}")
        log(f"  TEST  per-clip MAE {test_m['MAE_clip']:.3f}  RMSE {test_m['RMSE_clip']:.3f}  "
            f"Pearson {test_m['Pearson_clip']:.4f}  (n_clips={test_m['n_clips']})")
        log(f"  TEST  per-subj MAE {test_m['MAE_subj']:.3f}  Pearson {test_m['Pearson_subj']:.4f}  "
            f"sig_P {test_m['signal_Pearson']:.4f}  (n_subj={test_m['n_subjects']})")

        if valid_m['RMSE_clip'] < best_valid_rmse:
            best_valid_rmse = valid_m['RMSE_clip']
            best_test = test_m
            best_epoch = epoch + 1

    log(f"\n-> Best for {name} intra 7/1/2: epoch {best_epoch}")
    log(f"   per-clip  MAE {best_test['MAE_clip']:.3f}  RMSE {best_test['RMSE_clip']:.3f}  "
        f"Pearson {best_test['Pearson_clip']:.4f}")
    log(f"   per-subj  MAE {best_test['MAE_subj']:.3f}  Pearson {best_test['Pearson_subj']:.4f}")

    with open(os.path.join(result_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'name': name, 'best_epoch': best_epoch,
                   'best': best_test, 'history': history}, f, indent=2)
    return best_test, best_epoch


def main():
    _seed_everything(SEED)
    print(f"[*] BiPulseFormer intra 7/1/2 — OneCycleLR + 20 epochs")
    results = []
    for name, path, out, rand in [
        ('PURE', PURE_PATH, 'results/intra_pure_bipulseformer_712_oc20', True),
        ('UBFC-rPPG', UBFC_PATH, 'results/intra_ubfc_bipulseformer_712_oc20', False),
    ]:
        try:
            best, ep = run(name, path, out, use_random_subj=rand)
            results.append((name, best, ep))
        except Exception as e:
            print(f"[!] {name} failed: {e}")
            import traceback; traceback.print_exc()
    print("\n" + "=" * 70)
    print("[*] Final intra 7/1/2 OC20 — per-clip metric (paper-comparable)")
    print("=" * 70)
    for n, b, e in results:
        print(f"  {n}: MAE={b['MAE_clip']:.3f}  RMSE={b['RMSE_clip']:.3f}  Pearson={b['Pearson_clip']:.4f}  (E{e})")


if __name__ == '__main__':
    main()
