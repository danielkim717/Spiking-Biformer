"""BiPulseFormer PURE intra — 80/20 subject-exclusive split (rPPG-Toolbox / PhysFormer standard).

PURE 10 subjects:
  TRAIN (0.0-0.8): subjects 01-08  (subject 07 high-HR 포함)
  TEST  (0.8-1.0): subjects 09-10  (normal HR range 44-107 BPM)

Subject 07 (mean HR 127 BPM, outlier) 이 train 에 포함되어 OOD 문제 해소.
UBFC intra 와 동일한 RhythmFormer protocol (no separate valid, valid=test).
"""
import os
import sys
import io
import json
import time
import random
import threading
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal


def save_waveform_plot(subject_signals, out_path, fs=30, n_max=4, title=''):
    items = sorted(subject_signals.items(),
                   key=lambda kv: abs(kv[1]['hr_pred'] - kv[1]['hr_gt']),
                   reverse=True)[:n_max]
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (vid, s) in enumerate(items):
        T = len(s['pred_filt'])
        t = np.arange(T) / fs
        ax = axes[i, 0]
        pf = (s['pred_filt'] - s['pred_filt'].mean()) / (s['pred_filt'].std() + 1e-9)
        gf = (s['gt_filt'] - s['gt_filt'].mean()) / (s['gt_filt'].std() + 1e-9)
        ax.plot(t, gf, 'g-', label='GT', linewidth=0.8, alpha=0.8)
        ax.plot(t, pf, 'r-', label='Pred', linewidth=0.8, alpha=0.8)
        ax.set_title(f"{vid}  HR_gt={s['hr_gt']:.1f}  HR_pred={s['hr_pred']:.1f}  err={abs(s['hr_pred']-s['hr_gt']):.1f}",
                     fontsize=9)
        ax.set_xlim(0, min(15, t[-1])); ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('time (s)', fontsize=7)
        ax = axes[i, 1]
        N = 2 ** int(np.ceil(np.log2(T)))
        f_pred, p_pred = scipy.signal.periodogram(s['pred_filt'], fs=fs, nfft=N)
        f_gt, p_gt = scipy.signal.periodogram(s['gt_filt'], fs=fs, nfft=N)
        m = (f_pred >= 0.5) & (f_pred <= 3.5)
        ax.plot(f_pred[m] * 60, p_gt[m] / (p_gt[m].max() + 1e-9), 'g-', linewidth=0.8)
        ax.plot(f_pred[m] * 60, p_pred[m] / (p_pred[m].max() + 1e-9), 'r-', linewidth=0.8)
        ax.axvline(s['hr_gt'], color='g', linestyle='--', alpha=0.5)
        ax.axvline(s['hr_pred'], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('BPM', fontsize=7); ax.set_xlim(30, 180)
    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(out_path, dpi=80); plt.close(fig)


def get_hr_welch(y, sr=30, hr_min=30, hr_max=180):
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-9:
        return 0.0
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=int(np.min((len(y) - 1, 256))))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not mask.any():
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)


EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
WD = 5e-5
STEP_SIZE = 50
GAMMA = 0.5
ALPHA = 1.0
BETA = 1.0
GRA_SHARP = 2.0
DETECTION_FREQ = 0
GRAD_CLIP = 1.0
SEED = 42
FPS = 30
PURE_PATH = 'D:\\PURE'

EXPERIMENTS = [('PURE', PURE_PATH, 'PURE', PURE_PATH)]
RESULT_DIR = 'results/intra_pure_bipulseformer_80_20'
LOG = os.path.join(RESULT_DIR, 'log.txt')
STATUS_FILE = os.path.join('results', 'current_status_intra_pure_80_20.json')

_state = {'started_at': None, 'stop': False, 'completed': []}


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


def _seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _save_status(d):
    os.makedirs('results', exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def run_experiment(train_name, train_path, test_name, test_path):
    label = f"{train_name} -> {test_name}"
    log("\n" + "=" * 70); log(f"[*] {label}  (80/20 subject-exclusive split)"); log("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 80/20 subject-exclusive split — rPPG-Toolbox / PhysFormer paper 표준
    #   TRAIN: subjects 01-08 (incl. subject 07 high-HR outlier)
    #   TEST:  subjects 09-10 (HR 44-107, normal range)
    train_loader = get_dataloader(train_name, train_path, BATCH_SIZE, clip_len=160,
                                  face_crop=True, shuffle=True,
                                  data_type='diff_normalized', random_hflip=True,
                                  hr_filter=True, fps=FPS,
                                  dynamic_detection_freq=DETECTION_FREQ,
                                  split_range=(0.0, 0.8),
                                  pure_split_mode='subject_exclusive')
    test_loader = get_dataloader(test_name, test_path, BATCH_SIZE, clip_len=160,
                                 face_crop=True, shuffle=False,
                                 data_type='diff_normalized', chunk_step=80,
                                 dynamic_detection_freq=DETECTION_FREQ,
                                 split_range=(0.8, 1.0),
                                 pure_split_mode='subject_exclusive')
    valid_loader = test_loader  # RhythmFormer protocol (UBFC 와 동일)
    log(f"  train clips: {len(train_loader.dataset)} (subjects 01-08, 80% subject-exclusive)")
    log(f"  test  clips: {len(test_loader.dataset)} (subjects 09-10, 20% subject-exclusive)")
    log(f"  NOTE: valid = test (RhythmFormer protocol, UBFC 와 동일)")
    log(f"  Subject 07 (high-HR outlier, mean 127 BPM) 이 TRAIN 에 포함됨")

    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    log(f"  PE pretraining: disabled")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")
    log(f"  loss: alpha={ALPHA} (NegPearson) + beta={BETA} (CE+LD)")
    log(f"  optim: Adam(lr={LR}, wd={WD})  grad_clip: max_norm={GRAD_CLIP}")
    log(f"  EPOCHS: {EPOCHS}")

    best_test = {'Pearson_per_clip': 0.0, 'Pearson_pooled': 0.0,
                 'MAE_bpm': 0.0, 'RMSE_bpm': 0.0, 'MAPE_pct': 0.0,
                 'MAE_sample': 0.0, 'RMSE_sample': 0.0}
    best_valid = {'Pearson_per_clip': 0.0, 'MAE_bpm': 0.0}
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        a, b = ALPHA, BETA
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  alpha={a}, beta={b}")

        model.train()
        epoch_loss, nb = 0.0, 0
        for i, (inputs, labels) in enumerate(train_loader):
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
            epoch_loss += float(loss.item()); nb += 1
        scheduler.step()
        avg_loss = epoch_loss / max(1, nb)

        def evaluate(loader):
            model.eval()
            all_p, all_g = [], []
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(device, non_blocking=True)
                    rPPG, _, _, _ = model(inputs, gra_sharp=GRA_SHARP)
                    rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                           (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
                    all_p.append(rPPG.cpu()); all_g.append(labels.cpu())
            preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
            mae_s = float(np.mean(np.abs(preds - gts)))
            rmse_s = float(np.sqrt(np.mean((preds - gts) ** 2)))
            pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
            per_clip = per_clip_pearson(preds, gts)
            subj_metrics = evaluate_per_subject(preds, gts, loader.dataset.samples,
                                                fs=FPS, diff_flag=True,
                                                low_pass=0.75, high_pass=2.5)
            return {
                'MAE_bpm': subj_metrics['MAE_bpm'],
                'RMSE_bpm': subj_metrics['RMSE_bpm'],
                'MAPE_pct': subj_metrics['MAPE_pct'],
                'Pearson': subj_metrics['Pearson'],
                'signal_Pearson': subj_metrics['signal_Pearson_mean'],
                'n_subjects': subj_metrics['n_subjects'],
                'Pearson_pooled': pooled, 'Pearson_per_clip': per_clip,
                'MAE_sample': mae_s, 'RMSE_sample': rmse_s,
            }, preds, gts

        test_metrics, test_preds, test_gts = evaluate(test_loader)
        valid_metrics = test_metrics

        safe_label = label.replace(' -> ', '_to_').replace(' ', '').replace(':', '_')
        viz_dir = os.path.join(RESULT_DIR, 'waveforms', f'epoch_{epoch+1}')
        os.makedirs(viz_dir, exist_ok=True)
        try:
            sigs = get_subject_signals(test_preds, test_gts, test_loader.dataset.samples,
                                       fs=FPS, diff_flag=True, low_pass=0.75, high_pass=2.5)
            save_waveform_plot(sigs, os.path.join(viz_dir, f'TEST_{safe_label}.png'),
                               fs=FPS, n_max=4, title=f"{label} TEST epoch {epoch+1}")
        except Exception as e:
            log(f"  [!] waveform viz failed: {e}")

        ckpt_dir = os.path.join(RESULT_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{safe_label}_epoch{epoch+1}.pt'))

        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss,
                        'valid': valid_metrics, 'test': test_metrics})
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train loss {avg_loss:.4f}")
        log(f"  TEST  HR MAE {test_metrics['MAE_bpm']:.3f} BPM  RMSE {test_metrics['RMSE_bpm']:.3f} BPM  "
            f"Pearson(HR) {test_metrics['Pearson']:.4f}  sig_Pearson {test_metrics['signal_Pearson']:.4f}  "
            f"MAPE {test_metrics['MAPE_pct']:.2f}%  (n_subjects={test_metrics['n_subjects']})")

        if best_epoch == 0 or test_metrics['RMSE_bpm'] < best_test.get('RMSE_bpm', float('inf')):
            best_valid = valid_metrics
            best_test = test_metrics
            best_epoch = epoch + 1

    log(f"\n-> Best for {label}: epoch {best_epoch} (valid HR RMSE {best_valid['RMSE_bpm']:.3f} BPM)")
    log(f"   TEST HR  MAE {best_test['MAE_bpm']:.3f}  RMSE {best_test['RMSE_bpm']:.3f}  "
        f"Pearson(HR) {best_test['Pearson']:.4f}")
    return best_test, best_epoch, history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    _state['started_at'] = datetime.now()
    _seed_everything(SEED)
    log(f"[*] Seed fixed to {SEED}")
    log(f"[*] BiPulseFormer PURE intra — 80/20 subject-exclusive (rPPG-Toolbox standard)")
    try:
        all_results = []
        for train_n, train_p, test_n, test_p in EXPERIMENTS:
            best, best_epoch, history = run_experiment(train_n, train_p, test_n, test_p)
            all_results.append({
                'name': f"{train_n} -> {test_n}",
                'best': best, 'best_epoch': best_epoch, 'history': history,
            })
        with open(os.path.join(RESULT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
    finally:
        _state['stop'] = True


if __name__ == '__main__':
    main()
