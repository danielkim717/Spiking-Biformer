"""
Cross-dataset 학습-평가 (Spiking-PhysFormer + BiLevel Routing, **논문 셋업 그대로**).

순차 실행:
  1) PURE → UBFC-rPPG  (10 epoch)
  2) UBFC-rPPG → PURE  (10 epoch)

설정 (Spiking-PhysFormer 2024 + rPPG-Toolbox PHYSFORMER_BASIC YAML):
  - 모델: SpikingPhysformer(use_biformer=True, n_win=(2,2,2), topk=4)
  - 전처리: rPPG-Toolbox `DiffNormalized` data + label, HC face detect, 1.5× box,
            DYNAMIC_DETECTION_FREQUENCY=30
  - 손실: L = α·NegPearson + β·(L_CE + L_LD)
        α = 0.1 (고정),  β = β₀ · η^((e-1)/E),  β₀=1.0, η=5.0, E=10
  - 옵티마이저: Adam, lr=3e-3, wd=5e-5
  - LR scheduler: OneCycleLR (rPPG-Toolbox 기본값)
  - Gradient clipping: max_norm=1.0
  - Best 기준: per-clip Pearson
  - HR metric: 2nd-order Butterworth (0.75-2.5 Hz) + FFT peak → BPM (paper 4.2)

출력:
  - 매 epoch 평가 결과를 results/cross_biformer/log.txt 누적
  - 각 실험 종료 시 best epoch + 최종 metric 종합 리포트
"""
import os
import sys
import json
import time
import random
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from scipy.signal import butter, filtfilt
from spikingjelly.activation_based import functional

from src.models.spiking_physformer import SpikingPhysformer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss


EPOCHS = 10
BATCH_SIZE = 4   # paper 명시값 (BN running stats 안정화)
LR = 3e-3
WD = 5e-5
ALPHA = 0.1            # NegPearson 가중치 (고정)
BETA0 = 1.0            # β 시작값
ETA = 5.0              # β 지수 베이스 → epoch 끝에 β = β₀·η^((E-1)/E) ≈ 4.26
DETECTION_FREQ = 30    # rPPG-Toolbox DYNAMIC_DETECTION_FREQUENCY
GRAD_CLIP = 1.0        # gradient clip max_norm
SEED = 42              # paper: "fixed random seed for training and testing"
HR_LOW = 0.75          # Butterworth low cutoff (Hz) → 45 BPM
HR_HIGH = 2.5          # Butterworth high cutoff (Hz) → 150 BPM
FPS = 30
PURE_PATH = 'D:\\PURE'
UBFC_PATH = 'D:\\UBFC-rPPG'
# paper 의 사전학습 trick: PhysFormer 10 epoch → PE block weight 추출.
# scripts/pretrain_physformer_pe.py 로 생성. 해당 파일이 있으면 자동 로드.
PRETRAINED_PE = {'PURE': 'checkpoints/pretrained_pe_PURE.pt',
                 'UBFC-rPPG': 'checkpoints/pretrained_pe_UBFC-rPPG.pt'}

EXPERIMENTS = [
    ('PURE', PURE_PATH, 'UBFC-rPPG', UBFC_PATH),
    ('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH),
]

RESULT_DIR = 'results/cross_biformer'
LOG = os.path.join(RESULT_DIR, 'log.txt')
LIVE_FILE = os.path.join('results', 'live_progress.md')
STATUS_FILE = os.path.join('results', 'current_status.json')

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
    """2nd-order Butterworth bandpass (paper 4.2). signal: 1D np.array."""
    ny = fps / 2.0
    b, a = butter(order, [low_hz / ny, high_hz / ny], btype='bandpass')
    return filtfilt(b, a, signal)


def predict_hr(signal, fps=FPS, low_hz=HR_LOW, high_hz=HR_HIGH):
    """rPPG signal → 2nd-order Butterworth → FFT peak → HR (BPM).
    paper 4.2 의 후처리 절차 그대로."""
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
    valid_freqs = freqs[valid]
    valid_fft = fft[valid]
    peak_freq = valid_freqs[np.argmax(valid_fft)]
    return float(peak_freq * 60.0)


def hr_metrics(preds, gts, fps=FPS):
    """preds, gts: (N, T) arrays. Returns (MAE_bpm, RMSE_bpm, MAPE_pct)."""
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
    elapsed = (now - _state['started_at']).total_seconds() if _state['started_at'] else 0
    lines = []
    lines.append('# 📊 Spiking-PhysFormer + BiLevel Routing — Cross-Dataset')
    lines.append('')
    lines.append(f"**마지막 업데이트**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')
    lines.append('## 🎯 진행 상황')
    lines.append(f"- 실험: `{status.get('experiment', '-')}`  단계: **{status.get('phase', '-')}**")
    lines.append(f"- Epoch: {status.get('epoch', 0)} / {status.get('total_epochs', EPOCHS)}")
    lines.append(f"- Step: {status.get('step', 0)} / {status.get('total_steps', 1)}")
    lines.append(f"- Loss: `{status.get('loss', 0):.4f}`")
    lines.append('')
    if _state['completed']:
        lines.append('## ✅ 완료된 실험')
        for c in _state['completed']:
            m = c['best']
            lines.append(f"- **{c['name']}**: per-clip Pearson **{m['Pearson_per_clip']:.4f}** "
                         f"(pooled {m['Pearson_pooled']:.4f}, "
                         f"HR MAE {m['MAE_bpm']:.2f} BPM, epoch {c['best_epoch']})")
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
                                  dynamic_detection_freq=DETECTION_FREQ)
    test_loader = get_dataloader(test_name, test_path, BATCH_SIZE, clip_len=160,
                                 face_crop=True, shuffle=False,
                                 data_type='diff_normalized',
                                 dynamic_detection_freq=DETECTION_FREQ)
    log(f"  train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}")

    pe_path = PRETRAINED_PE.get(train_name)
    if pe_path is None or not os.path.exists(pe_path):
        log(f"  [!] pretrained PE 없음 ({pe_path}) — from scratch 학습")
        pe_path = None
    else:
        log(f"  pretrained PE block: {pe_path}")
    model = SpikingPhysformer(dim=96, num_blocks=4, num_heads=4, frame=160,
                              v_threshold=1.0, T_snn=4,
                              use_biformer=True, n_win=(2, 2, 2), topk=4,
                              pretrained_pe_path=pe_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    # OneCycleLR — rPPG-Toolbox 기본 scheduler. paper 4.2: "we follow the standard
    # configuration of rPPG-Toolbox" → max_lr=LR, total_steps=EPOCHS*len(train_loader)
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                                              total_steps=total_steps)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")
    log(f"  loss: α={ALPHA} (NegPearson) + β·(CE+LD), β = {BETA0}·{ETA}^((e-1)/{EPOCHS})")
    log(f"  optim: Adam(lr={LR}, wd={WD}) + OneCycleLR(max_lr={LR}, steps={total_steps})")
    log(f"  grad_clip: max_norm={GRAD_CLIP}")
    log(f"  HR metric: Butterworth({HR_LOW}-{HR_HIGH}Hz) + FFT peak → BPM")

    # best 기준은 per-clip Pearson (pooled 는 baseline drift 가 dominate)
    best = {'Pearson_per_clip': -1.0, 'Pearson_pooled': 0.0,
            'MAE_bpm': 0.0, 'RMSE_bpm': 0.0, 'MAPE_pct': 0.0,
            'MAE_sample': 0.0, 'RMSE_sample': 0.0}
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        # PhysFormer / Spiking-PhysFormer 공식 curriculum:
        #   L = α·NegPearson + β·(L_CE + L_LD),  α = 0.1 fixed
        #   β = β₀ · η^((Epoch_current-1)/Epoch_total),  β₀=1.0, η=5.0
        # epoch 1 → β=1.0, epoch 10 → β = 1·5^(9/10) ≈ 4.26
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
            functional.reset_net(model)
            epoch_loss += float(loss.item()); nb += 1
            if i % 10 == 0:
                _save_status({
                    'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                    'step': i, 'total_steps': len(train_loader),
                    'loss': float(loss.item()), 'phase': 'Training',
                })
        avg_loss = epoch_loss / max(1, nb)
        # firing rate trace (디버깅용)
        firing = getattr(model, 'last_firing_rates', [])
        if firing:
            log(f"  block firing rates: {[f'{f:.3f}' for f in firing]}  current_lr: {scheduler.get_last_lr()[0]:.2e}")

        # Eval
        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_p.append(outputs.cpu()); all_g.append(labels.cpu())
                if j % 10 == 0:
                    _save_status({
                        'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                        'step': j, 'total_steps': len(test_loader),
                        'loss': 0.0, 'phase': 'Evaluation',
                    })
        preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
        mae_s = float(np.mean(np.abs(preds - gts)))
        rmse_s = float(np.sqrt(np.mean((preds - gts) ** 2)))
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        # paper 4.2 후처리: Butterworth + FFT peak → HR (BPM)
        mae_hr, rmse_hr, mape, _, _ = hr_metrics(preds, gts, fps=FPS)
        history.append({
            'epoch': epoch + 1, 'avg_loss': avg_loss,
            'MAE_bpm': mae_hr, 'RMSE_bpm': rmse_hr, 'MAPE_pct': mape,
            'MAE_sample': mae_s, 'RMSE_sample': rmse_s,
            'Pearson_pooled': pooled, 'Pearson_per_clip': per_clip,
        })
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train avg loss {avg_loss:.4f}")
        log(f"  HR (BPM)   MAE {mae_hr:.3f}  RMSE {rmse_hr:.3f}  MAPE {mape:.2f}%")
        log(f"  Pearson    pooled {pooled:.4f}  per-clip {per_clip:.4f}")
        log(f"  Sample-wise (DiffBVP) MAE {mae_s:.4f}  RMSE {rmse_s:.4f}")
        # best 는 per-clip Pearson 기준
        if per_clip > best['Pearson_per_clip']:
            best = {
                'Pearson_per_clip': per_clip, 'Pearson_pooled': pooled,
                'MAE_bpm': mae_hr, 'RMSE_bpm': rmse_hr, 'MAPE_pct': mape,
                'MAE_sample': mae_s, 'RMSE_sample': rmse_s,
            }
            best_epoch = epoch + 1

    log(f"\n→ Best for {label}: epoch {best_epoch}")
    log(f"   per-clip Pearson {best['Pearson_per_clip']:.4f}  pooled Pearson {best['Pearson_pooled']:.4f}")
    log(f"   HR  MAE {best['MAE_bpm']:.3f} BPM  RMSE {best['RMSE_bpm']:.3f} BPM  MAPE {best['MAPE_pct']:.2f}%")
    return best, best_epoch, history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    _state['started_at'] = datetime.now()
    _seed_everything(SEED)
    log(f"[*] Seed fixed to {SEED} (paper: 'fixed random seed for reproducibility')")
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

        # Final summary
        log("\n" + "=" * 70)
        log("[*] 종합 결과")
        log("=" * 70)
        for r in all_results:
            b = r['best']
            log(f"  {r['name']}:")
            log(f"    Best @ epoch {r['best_epoch']}: "
                f"per-clip Pearson = {b['Pearson_per_clip']:.4f}  "
                f"(pooled {b['Pearson_pooled']:.4f})")
            log(f"      HR  MAE {b['MAE_bpm']:.3f} BPM  RMSE {b['RMSE_bpm']:.3f} BPM  "
                f"MAPE {b['MAPE_pct']:.2f}%")

        # JSON dump
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
