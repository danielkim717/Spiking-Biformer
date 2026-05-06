"""
Cross-dataset 학습-평가 — BiPhysFormer (PhysFormer + BiLevel Routing, ANN).

목적: Spiking 효과 없이 BiFormer 단독의 기여도를 검증.
구조: SpikingPhysformer + BiSDA 와 동일한 학습 셋업 (loss, optim, scheduler)
       을 사용하되, 모델만 ANN BiPhysFormer 로 교체.

순차 실행:
  1) PURE → UBFC-rPPG  (10 epoch)
  2) UBFC-rPPG → PURE  (10 epoch)

설정 (PhysFormer 2022 + rPPG-Toolbox PHYSFORMER_BASIC YAML):
  - 모델: BiPhysFormer(dim=96, ff_dim=144, num_heads=4, num_layers=12,
                       n_win=(2,2,2), topk=4, theta=0.7)
  - 손실: L = α·NegPearson + β·(L_CE + L_LD)
        α = 0.1, β = β₀·η^((e-1)/E), β₀=1.0, η=5.0, E=10
  - 옵티마이저: Adam, lr=1e-4, wd=5e-5  (PhysFormer paper default — Spiking 보다 작음)
  - LR scheduler: OneCycleLR
  - Gradient clipping: max_norm=1.0
  - Best 기준: source valid per-clip Pearson
  - HR metric: 2nd-order Butterworth (0.75-2.5 Hz) + FFT peak → BPM

출력:
  - results/cross_biphysformer/log.txt
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

# UTF-8 stdout (Windows cp949 redirect 회피)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim
from scipy.signal import butter, filtfilt

from src.models.biphysformer import BiPhysFormer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss


EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4              # PhysFormer paper default (ANN, Spiking 의 3e-3 보다 작음)
WD = 5e-5
ALPHA = 0.1
BETA0 = 1.0
ETA = 5.0
DETECTION_FREQ = 30
GRAD_CLIP = 1.0
SEED = 42
HR_LOW = 0.75
HR_HIGH = 2.5
FPS = 30
PURE_PATH = 'D:\\PURE'
UBFC_PATH = 'D:\\UBFC-rPPG'
PRETRAINED_PE = {'PURE': 'checkpoints/pretrained_pe_PURE.pt',
                 'UBFC-rPPG': 'checkpoints/pretrained_pe_UBFC-rPPG.pt'}

EXPERIMENTS = [
    ('PURE', PURE_PATH, 'UBFC-rPPG', UBFC_PATH),
    ('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH),
]

RESULT_DIR = 'results/cross_biphysformer'
LOG = os.path.join(RESULT_DIR, 'log.txt')
LIVE_FILE = os.path.join('results', 'live_progress_biphys.md')
STATUS_FILE = os.path.join('results', 'current_status_biphys.json')

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


def _butter_bandpass_filter(signal, low_hz, high_hz, fps, order=2):
    ny = fps / 2.0
    b, a = butter(order, [low_hz / ny, high_hz / ny], btype='bandpass')
    return filtfilt(b, a, signal)


def predict_hr(signal, fps=FPS, low_hz=HR_LOW, high_hz=HR_HIGH):
    s = np.asarray(signal, dtype=np.float64)
    if np.std(s) < 1e-9:
        return 0.0
    filtered = _butter_bandpass_filter(s, low_hz, high_hz, fps, order=2)
    N = len(filtered)
    win = np.hanning(N)
    fft = np.abs(np.fft.rfft(filtered * win))
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    valid = (freqs >= low_hz) & (freqs <= high_hz)
    if not valid.any():
        return 0.0
    return float(freqs[valid][np.argmax(fft[valid])] * 60.0)


def hr_metrics(preds, gts, fps=FPS):
    pred_hrs = np.array([predict_hr(p, fps) for p in preds])
    gt_hrs = np.array([predict_hr(g, fps) for g in gts])
    abs_err = np.abs(pred_hrs - gt_hrs)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    nz = gt_hrs != 0
    mape = float(np.mean(abs_err[nz] / gt_hrs[nz]) * 100.0) if nz.any() else 0.0
    return mae, rmse, mape, pred_hrs, gt_hrs


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_status(d):
    os.makedirs('results', exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def _live(now, status):
    lines = [
        '# 📊 BiPhysFormer (ANN, BiFormer) — Cross-Dataset',
        '',
        f"**마지막 업데이트**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        '',
        '## 🎯 진행 상황',
        f"- 실험: `{status.get('experiment', '-')}`  단계: **{status.get('phase', '-')}**",
        f"- Epoch: {status.get('epoch', 0)} / {status.get('total_epochs', EPOCHS)}",
        f"- Step: {status.get('step', 0)} / {status.get('total_steps', 1)}",
        f"- Loss: `{status.get('loss', 0):.4f}`",
        '',
    ]
    if _state['completed']:
        lines.append('## ✅ 완료된 실험')
        for c in _state['completed']:
            m = c['best']
            lines.append(f"- **{c['name']}**: per-clip Pearson **{m['Pearson_per_clip']:.4f}** "
                         f"(pooled {m['Pearson_pooled']:.4f}, HR MAE {m['MAE_bpm']:.2f} BPM, epoch {c['best_epoch']})")
        lines.append('')
    os.makedirs('results', exist_ok=True)
    with open(LIVE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _progress_loop():
    while not _state['stop']:
        try:
            try:
                with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                    s = json.load(f)
            except Exception:
                s = {}
            _live(datetime.now(), s)
        except Exception:
            pass
        for _ in range(300):
            if _state['stop']: break
            time.sleep(1)


def run_experiment(train_name, train_path, test_name, test_path):
    label = f"{train_name} -> {test_name}"
    log("\n" + "=" * 70)
    log(f"[*] {label}")
    log("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(train_name, train_path, BATCH_SIZE, clip_len=160,
                                  face_crop=True, shuffle=True,
                                  data_type='diff_normalized',
                                  dynamic_detection_freq=DETECTION_FREQ,
                                  split_range=(0.0, 0.8))
    valid_loader = get_dataloader(train_name, train_path, BATCH_SIZE, clip_len=160,
                                  face_crop=True, shuffle=False,
                                  data_type='diff_normalized',
                                  dynamic_detection_freq=DETECTION_FREQ,
                                  split_range=(0.8, 1.0))
    test_loader = get_dataloader(test_name, test_path, BATCH_SIZE, clip_len=160,
                                 face_crop=True, shuffle=False,
                                 data_type='diff_normalized',
                                 dynamic_detection_freq=DETECTION_FREQ)
    log(f"  train clips: {len(train_loader.dataset)} (80% of {train_name})")
    log(f"  valid clips: {len(valid_loader.dataset)} (20% of {train_name}) — used for best-epoch selection")
    log(f"  test  clips: {len(test_loader.dataset)} ({test_name})")

    model = BiPhysFormer(dim=96, ff_dim=144, num_heads=4, num_layers=12,
                         frame=160, image_size=128, dropout=0.1, theta=0.7,
                         n_win=(2, 2, 2), topk=4).to(device)

    pe_path = PRETRAINED_PE.get(train_name)
    if pe_path is not None and os.path.exists(pe_path):
        log(f"  pretrained PE block: {pe_path}")
        model.load_pretrained_pe(pe_path)
    else:
        log(f"  [!] pretrained PE 없음 — from scratch 학습")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")
    log(f"  loss: α={ALPHA} (NegPearson) + β·(CE+LD), β = {BETA0}·{ETA}^((e-1)/{EPOCHS})")
    log(f"  optim: Adam(lr={LR}, wd={WD}) + OneCycleLR(max_lr={LR}, steps={total_steps})")
    log(f"  grad_clip: max_norm={GRAD_CLIP}")

    best_valid_per_clip = -1.0
    best_test = {'Pearson_per_clip': 0.0, 'Pearson_pooled': 0.0,
                 'MAE_bpm': 0.0, 'RMSE_bpm': 0.0, 'MAPE_pct': 0.0,
                 'MAE_sample': 0.0, 'RMSE_sample': 0.0}
    best_valid = {'Pearson_per_clip': 0.0, 'MAE_bpm': 0.0}
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        beta = BETA0 * (ETA ** (epoch / EPOCHS))
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  α={ALPHA}, β={beta:.4f}")

        model.train()
        epoch_loss, nb = 0.0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_p = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)
            loss = ALPHA * loss_p + beta * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item()); nb += 1
            if i % 10 == 0:
                _save_status({
                    'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                    'step': i, 'total_steps': len(train_loader),
                    'loss': float(loss.item()), 'phase': 'Training',
                })
        avg_loss = epoch_loss / max(1, nb)
        log(f"  current_lr: {scheduler.get_last_lr()[0]:.2e}")

        def evaluate(loader, phase_name):
            model.eval()
            all_p, all_g = [], []
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    all_p.append(outputs.cpu()); all_g.append(labels.cpu())
                    if j % 10 == 0:
                        _save_status({
                            'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                            'step': j, 'total_steps': len(loader),
                            'loss': 0.0, 'phase': phase_name,
                        })
            preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
            mae_s = float(np.mean(np.abs(preds - gts)))
            rmse_s = float(np.sqrt(np.mean((preds - gts) ** 2)))
            pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
            per_clip = per_clip_pearson(preds, gts)
            mae_hr, rmse_hr, mape, _, _ = hr_metrics(preds, gts, fps=FPS)
            return {
                'Pearson_pooled': pooled, 'Pearson_per_clip': per_clip,
                'MAE_bpm': mae_hr, 'RMSE_bpm': rmse_hr, 'MAPE_pct': mape,
                'MAE_sample': mae_s, 'RMSE_sample': rmse_s,
            }

        valid_metrics = evaluate(valid_loader, 'Validation')
        test_metrics = evaluate(test_loader, 'Test')

        history.append({
            'epoch': epoch + 1, 'avg_loss': avg_loss,
            'valid': valid_metrics, 'test': test_metrics,
        })
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train avg loss {avg_loss:.4f}")
        log(f"  VALID (source 20%)   per-clip Pearson {valid_metrics['Pearson_per_clip']:.4f}  "
            f"pooled {valid_metrics['Pearson_pooled']:.4f}  "
            f"HR MAE {valid_metrics['MAE_bpm']:.3f} BPM")
        log(f"  TEST  (target full)  per-clip Pearson {test_metrics['Pearson_per_clip']:.4f}  "
            f"pooled {test_metrics['Pearson_pooled']:.4f}  "
            f"HR MAE {test_metrics['MAE_bpm']:.3f} BPM  "
            f"RMSE {test_metrics['RMSE_bpm']:.3f} BPM  MAPE {test_metrics['MAPE_pct']:.2f}%")

        if valid_metrics['Pearson_per_clip'] > best_valid_per_clip:
            best_valid_per_clip = valid_metrics['Pearson_per_clip']
            best_valid = valid_metrics
            best_test = test_metrics
            best_epoch = epoch + 1

    log(f"\n→ Best for {label}: epoch {best_epoch} (selected by valid per-clip Pearson {best_valid['Pearson_per_clip']:.4f})")
    log(f"   TEST per-clip Pearson {best_test['Pearson_per_clip']:.4f}  "
        f"pooled {best_test['Pearson_pooled']:.4f}")
    log(f"   TEST HR  MAE {best_test['MAE_bpm']:.3f} BPM  RMSE {best_test['RMSE_bpm']:.3f} BPM  "
        f"MAPE {best_test['MAPE_pct']:.2f}%")
    return best_test, best_epoch, history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    _state['started_at'] = datetime.now()
    _seed_everything(SEED)
    log(f"[*] Seed fixed to {SEED}")
    logger = threading.Thread(target=_progress_loop, daemon=True)
    logger.start()

    try:
        all_results = []
        for train_n, train_p, test_n, test_p in EXPERIMENTS:
            best, best_epoch, history = run_experiment(train_n, train_p, test_n, test_p)
            all_results.append({
                'name': f"{train_n} -> {test_n}",
                'best': best, 'best_epoch': best_epoch, 'history': history,
            })
            _state['completed'].append({
                'name': f"{train_n} -> {test_n}", 'best': best, 'best_epoch': best_epoch,
            })

        log("\n" + "=" * 70)
        log("[*] 종합 결과 — BiPhysFormer (ANN, BiFormer)")
        log("=" * 70)
        for r in all_results:
            b = r['best']
            log(f"  {r['name']}:")
            log(f"    Best @ epoch {r['best_epoch']}: per-clip Pearson = {b['Pearson_per_clip']:.4f}  "
                f"(pooled {b['Pearson_pooled']:.4f})")
            log(f"      HR  MAE {b['MAE_bpm']:.3f} BPM  RMSE {b['RMSE_bpm']:.3f} BPM  "
                f"MAPE {b['MAPE_pct']:.2f}%")

        with open(os.path.join(RESULT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

    finally:
        _state['stop'] = True
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                s = json.load(f)
            s['phase'] = 'Completed'
        except Exception:
            s = {'phase': 'Completed'}
        _live(datetime.now(), s)


if __name__ == '__main__':
    main()
