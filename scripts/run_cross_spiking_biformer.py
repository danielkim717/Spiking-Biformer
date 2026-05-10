"""
Cross-dataset 학습-평가 — BiPhysFormer (PhysFormer + BiLevel Routing, ANN).

PhysFormer 공식 학습 셋업을 그대로 따름
(https://github.com/ZitongYu/PhysFormer/blob/main/train_Physformer_160_VIPL.py):
  - 모델: ViT_BiPhysFormer(dim=96, ff_dim=144, num_heads=4, num_layers=12,
                          dropout=0.1, theta=0.7, n_win=(2,2,2), topk=4)
  - Optimizer: Adam(lr=1e-4, wd=5e-5)
  - LR scheduler: StepLR(step_size=50, gamma=0.5)  (25 epoch 동안 사실상 constant)
  - Batch size: 4
  - Epochs: 25  (PhysFormer paper)
  - 출력 정규화 (공식 코드 line 190): rPPG = (rPPG - mean) / std  ← 매우 중요
  - 손실: L = α·NegPearson + β·(L_CE + L_LD)
        α = 0.1 (공식 코드는 epoch>25 일 때만 0.05; 우리는 25 epoch 학습이라 항상 0.1)
        β = β₀ · η^(epoch/25),  β₀=1.0, η=5.0  (공식 schedule)
  - gra_sharp = 2.0 (forward 인자)
  - Best 기준: source valid per-clip Pearson
  - HR metric: 2nd-order Butterworth (0.75-2.5 Hz) + FFT peak

순차 실행:
  1) PURE → UBFC-rPPG  (25 epoch)
  2) UBFC-rPPG → PURE  (25 epoch)
"""
import os
import sys
import io
import json
import math
import time
import random
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UTF-8 stdout for Windows cp949 redirect
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
import torch.optim as optim
from scipy.signal import welch

from src.models.spiking_physformer import SpikingPhysformer
from spikingjelly.activation_based import functional as snn_functional
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.evaluation import evaluate_per_subject, get_subject_signals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal


def save_waveform_plot(subject_signals, out_path, fs=30, n_max=4, title=''):
    """Top-N subjects (highest HR error) waveform plot 저장."""
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
        ax.plot(t, gf, 'g-', label='GT (filtered)', linewidth=0.8, alpha=0.8)
        ax.plot(t, pf, 'r-', label='Pred (filtered)', linewidth=0.8, alpha=0.8)
        ax.set_title(f"{vid}  HR_gt={s['hr_gt']:.1f}  HR_pred={s['hr_pred']:.1f}  err={abs(s['hr_pred']-s['hr_gt']):.1f}",
                     fontsize=9)
        ax.set_xlim(0, min(15, t[-1]))
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('time (s)', fontsize=7)
        ax = axes[i, 1]
        N = 2 ** int(np.ceil(np.log2(T)))
        f_pred, p_pred = scipy.signal.periodogram(s['pred_filt'], fs=fs, nfft=N)
        f_gt, p_gt = scipy.signal.periodogram(s['gt_filt'], fs=fs, nfft=N)
        m = (f_pred >= 0.5) & (f_pred <= 3.5)
        ax.plot(f_pred[m] * 60, p_gt[m] / (p_gt[m].max() + 1e-9), 'g-', label='GT spectrum', linewidth=0.8)
        ax.plot(f_pred[m] * 60, p_pred[m] / (p_pred[m].max() + 1e-9), 'r-', label='Pred spectrum', linewidth=0.8)
        ax.axvline(s['hr_gt'], color='g', linestyle='--', alpha=0.5)
        ax.axvline(s['hr_pred'], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('BPM', fontsize=7)
        ax.set_xlim(30, 180)
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def get_hr_welch(y, sr=30, hr_min=30, hr_max=180):
    """rPPG-Toolbox PhysFormerTrainer.get_hr 와 동일.
    Welch periodogram peak in [hr_min/60, hr_max/60] Hz → HR (BPM)."""
    y = np.asarray(y, dtype=np.float64)
    if np.std(y) < 1e-9:
        return 0.0
    p, q = welch(y, sr, nfft=1e5 / sr, nperseg=int(np.min((len(y) - 1, 256))))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not mask.any():
        return 0.0
    return float(p[mask][np.argmax(q[mask])] * 60)


# rPPG-Toolbox PhysFormerTrainer 셋업 그대로 (paper MAE 1.44 / 12.92 를 만든 셋업)
# https://github.com/ubicomplab/rPPG-Toolbox/blob/main/neural_methods/trainer/PhysFormerTrainer.py
EPOCHS = 10                # SNN: 사용자 지시 10 epoch (β shift 안 일어남, 안정적 phase 만 학습)
BATCH_SIZE = 4
LR = 1e-4
WD = 5e-5
STEP_SIZE = 50             # StepLR(50, 0.5) — 10 epoch 동안 사실상 constant LR
GAMMA = 0.5
ALPHA = 1.0                # rPPG-Toolbox 의 a_start=1.0 (PhysFormer 원논문 0.1 에서 변경)
BETA = 1.0                 # rPPG-Toolbox 의 b_start=1.0, exp_b=1.0 → epoch≤10 동안 b=1.0 constant
GRA_SHARP = 2.0            # PhysFormer attention sharpness
DETECTION_FREQ = 0         # rPPG-Toolbox: DO_DYNAMIC_DETECTION=False (static, 첫 프레임)
GRAD_CLIP = 1.0
SEED = 42
FPS = 30
HR_MIN_BPM = 30           # rPPG-Toolbox get_hr 의 min
HR_MAX_BPM = 180          # rPPG-Toolbox get_hr 의 max
PURE_PATH = 'D:\\PURE'
UBFC_PATH = 'D:\\UBFC-rPPG'
# rPPG-Toolbox 는 PE pretraining 사용 안 함. 비활성.

EXPERIMENTS = [
    ('PURE', PURE_PATH, 'UBFC-rPPG', UBFC_PATH),
    ('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH),
]

RESULT_DIR = 'results/cross_spiking_biformer'
LOG = os.path.join(RESULT_DIR, 'log.txt')
LIVE_FILE = os.path.join('results', 'live_progress_spiking.md')
STATUS_FILE = os.path.join('results', 'current_status_spiking.json')

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


def predict_hr(signal, fps=FPS, hr_min=30, hr_max=180):
    """rPPG-Toolbox PhysFormerTrainer.get_hr 와 동일 (Welch periodogram)."""
    return get_hr_welch(signal, sr=fps, hr_min=hr_min, hr_max=hr_max)


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

    # rPPG-Toolbox/PhysBench PhysFormerTrainer 셋업:
    #   - Train: DiffNormalized + RandomHorizontalFlip + HR validity filter (40<HR<180)
    #   - Valid/Test: sliding window (chunk_step=20, 8x overlap) + HR aggregation 평균
    train_loader = get_dataloader(train_name, train_path, BATCH_SIZE, clip_len=160,
                                  face_crop=True, shuffle=True,
                                  data_type='diff_normalized', random_hflip=True,
                                  hr_filter=True, fps=FPS,
                                  dynamic_detection_freq=DETECTION_FREQ,
                                  split_range=(0.0, 0.8))
    valid_loader = get_dataloader(train_name, train_path, BATCH_SIZE, clip_len=160,
                                  face_crop=True, shuffle=False,
                                  data_type='diff_normalized',
                                  dynamic_detection_freq=DETECTION_FREQ,
                                  split_range=(0.8, 1.0))
    test_loader = get_dataloader(test_name, test_path, BATCH_SIZE, clip_len=160,
                                 face_crop=True, shuffle=False,
                                 data_type='diff_normalized', chunk_step=80,
                                 dynamic_detection_freq=DETECTION_FREQ)
    log(f"  train clips: {len(train_loader.dataset)} (80% of {train_name})")
    log(f"  valid clips: {len(valid_loader.dataset)} (20% of {train_name}) - used for best-epoch selection")
    log(f"  test  clips: {len(test_loader.dataset)} ({test_name})")

    # Spiking-Biformer: SNN-based PhysFormer + BiLevel Routing Attention
    # T_snn=4, BiSDA (Pre-LIF gating), use_biformer=True
    model = SpikingPhysformer(
        dim=96, num_blocks=4, num_heads=4, frame=160,
        v_threshold=1.0, T_snn=4, theta=0.7,
        use_biformer=True, n_win=(2, 2, 2), topk=4,
        pretrained_pe_path=None,  # from scratch
    ).to(device)
    log(f"  SNN: T_snn=4, v_threshold=1.0, BiSDA pre-LIF gating, use_biformer=True")
    log(f"  PE pretraining: disabled (from scratch, rPPG-Toolbox 동일)")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=FPS)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")
    log(f"  loss: alpha={ALPHA} (NegPearson) + beta={BETA} (CE+LD)  [rPPG-Toolbox: alpha=1.0, beta=1.0 constant]")
    log(f"  optim: Adam(lr={LR}, wd={WD}) + StepLR(step={STEP_SIZE}, gamma={GAMMA})")
    log(f"  grad_clip: max_norm={GRAD_CLIP}")
    log(f"  output norm: rPPG = (rPPG - mean(axis=-1)) / std(axis=-1)  [per-sample, rPPG-Toolbox]")
    log(f"  data: DiffNormalized input/label, static face detection, no augmentation")
    log(f"  EPOCHS: {EPOCHS}  (PhysFormer paper, 30ep)  schedule: epoch<=10 -> a=1.0,b=1.0; epoch>10 -> a=0.05,b=5.0")

    best_valid_per_clip = -1.0
    best_test = {'Pearson_per_clip': 0.0, 'Pearson_pooled': 0.0,
                 'MAE_bpm': 0.0, 'RMSE_bpm': 0.0, 'MAPE_pct': 0.0,
                 'MAE_sample': 0.0, 'RMSE_sample': 0.0}
    best_valid = {'Pearson_per_clip': 0.0, 'MAE_bpm': 0.0}
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        # rPPG-Toolbox PhysFormerTrainer.train line 135-141:
        #   if epoch > 10:  a = 0.05, b = 5.0  (Pearson 약화, freq 강화)
        #   else:           a = 1.0,  b = 1.0 * 1.0^(epoch/10) = 1.0
        # 10 epoch 학습이면 항상 else 분기 (a=1.0, b=1.0). 그래도 명시적 분기 유지.
        if epoch > 10:
            a, b = 0.05, 5.0
        else:
            a, b = ALPHA, BETA
        log(f"\n[*] Epoch {epoch+1}/{EPOCHS}  alpha={a}, beta={b:.4f}")

        model.train()
        epoch_loss, nb = 0.0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            # rPPG-Toolbox: HR 은 label signal 에서 Welch periodogram 으로 추출
            # (PhysFormerTrainer.py line 106). Per-sample 호출.
            hr_target = torch.tensor(
                [get_hr_welch(labels[b].cpu().numpy(), sr=FPS) for b in range(labels.shape[0])],
                dtype=torch.float32, device=device,
            )
            optimizer.zero_grad()
            rPPG = model(inputs); snn_functional.reset_net(model)
            # rPPG-Toolbox PhysFormerTrainer line 113: per-sample 정규화
            #   rPPG = (rPPG - mean(axis=-1)) / std(axis=-1)
            rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                   (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
            loss_p = pearson_criterion(rPPG, labels)
            loss_ce, loss_ld = freq_criterion(rPPG, hr_target)
            loss = a * loss_p + b * (loss_ce + loss_ld)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += float(loss.item()); nb += 1
            if i % 10 == 0:
                _save_status({
                    'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                    'step': i, 'total_steps': len(train_loader),
                    'loss': float(loss.item()), 'phase': 'Training',
                })
        scheduler.step()
        avg_loss = epoch_loss / max(1, nb)
        log(f"  current_lr: {scheduler.get_last_lr()[0]:.2e}")

        def evaluate(loader, phase_name):
            model.eval()
            all_p, all_g = [], []
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(loader):
                    inputs = inputs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                    rPPG = model(inputs); snn_functional.reset_net(model)
                    # rPPG-Toolbox: eval 시에도 per-sample 정규화 (PhysFormerTrainer line 199)
                    rPPG = (rPPG - torch.mean(rPPG, dim=-1, keepdim=True)) / \
                           (torch.std(rPPG, dim=-1, keepdim=True) + 1e-8)
                    all_p.append(rPPG.cpu()); all_g.append(labels.cpu())
                    if j % 10 == 0:
                        _save_status({
                            'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                            'step': j, 'total_steps': len(loader),
                            'loss': 0.0, 'phase': phase_name,
                        })
            preds = torch.cat(all_p).numpy(); gts = torch.cat(all_g).numpy()
            # 기존 per-clip metric (모니터링 용)
            mae_s = float(np.mean(np.abs(preds - gts)))
            rmse_s = float(np.sqrt(np.mean((preds - gts) ** 2)))
            pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
            per_clip = per_clip_pearson(preds, gts)
            # rPPG-Toolbox 호환 per-subject 평가 (paper 수치 비교용)
            subj_metrics = evaluate_per_subject(
                preds, gts, loader.dataset.samples,
                fs=FPS, diff_flag=True,
                low_pass=0.75, high_pass=2.5,
            )
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

        valid_metrics, valid_preds, valid_gts = evaluate(valid_loader, 'Validation')
        test_metrics, test_preds, test_gts = evaluate(test_loader, 'Test')

        # Waveform visualization: reuse preds from evaluate
        viz_dir = os.path.join(RESULT_DIR, 'waveforms', f'epoch_{epoch+1}')
        os.makedirs(viz_dir, exist_ok=True)
        safe_label = label.replace(' -> ', '_to_').replace(' ', '').replace(':', '_')
        try:
            from src.evaluation import get_subject_signals
            for loader, name, p_arr, g_arr in [
                (valid_loader, 'VALID', valid_preds, valid_gts),
                (test_loader, 'TEST', test_preds, test_gts),
            ]:
                sigs = get_subject_signals(p_arr, g_arr, loader.dataset.samples,
                                           fs=FPS, diff_flag=True, low_pass=0.75, high_pass=2.5)
                save_waveform_plot(
                    sigs, os.path.join(viz_dir, f'{name}_{safe_label}.png'),
                    fs=FPS, n_max=4, title=f"{label} {name} epoch {epoch+1}",
                )
            log(f"  waveforms saved → {viz_dir}")
        except Exception as e:
            log(f"  [!] waveform viz failed: {e}")

        # Save checkpoint
        ckpt_dir = os.path.join(RESULT_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'{safe_label}_epoch{epoch+1}.pt')
        torch.save(model.state_dict(), ckpt_path)

        history.append({
            'epoch': epoch + 1, 'avg_loss': avg_loss,
            'valid': valid_metrics, 'test': test_metrics,
        })
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train avg loss {avg_loss:.4f}")
        log(f"  VALID  HR MAE {valid_metrics['MAE_bpm']:.3f} BPM  RMSE {valid_metrics['RMSE_bpm']:.3f} BPM  "
            f"Pearson(HR) {valid_metrics['Pearson']:.4f}  "
            f"signal_pearson {valid_metrics['signal_Pearson']:.4f}  "
            f"(n_subjects={valid_metrics['n_subjects']})")
        log(f"  TEST   HR MAE {test_metrics['MAE_bpm']:.3f} BPM  RMSE {test_metrics['RMSE_bpm']:.3f} BPM  "
            f"Pearson(HR) {test_metrics['Pearson']:.4f}  "
            f"signal_pearson {test_metrics['signal_Pearson']:.4f}  "
            f"MAPE {test_metrics['MAPE_pct']:.2f}%  (n_subjects={test_metrics['n_subjects']})")
        log(f"  [per-clip ref] VALID Pearson_per_clip {valid_metrics['Pearson_per_clip']:.4f}  "
            f"TEST Pearson_per_clip {test_metrics['Pearson_per_clip']:.4f}")

        # rPPG-Toolbox PhysFormerTrainer: best epoch = min RMSE on valid
        # (PhysFormerTrainer.py line 167-177). 우리는 valid HR RMSE 사용.
        valid_rmse = valid_metrics['RMSE_bpm']
        if best_epoch == 0 or valid_rmse < best_valid.get('RMSE_bpm', float('inf')):
            best_valid_per_clip = valid_metrics['Pearson_per_clip']
            best_valid = valid_metrics
            best_test = test_metrics
            best_epoch = epoch + 1

    log(f"\n-> Best for {label}: epoch {best_epoch} (selected by valid HR RMSE {best_valid['RMSE_bpm']:.3f} BPM)")
    log(f"   TEST HR  MAE {best_test['MAE_bpm']:.3f} BPM  RMSE {best_test['RMSE_bpm']:.3f} BPM  "
        f"MAPE {best_test['MAPE_pct']:.2f}%  Pearson(HR) {best_test['Pearson']:.4f}  "
        f"signal_pearson {best_test['signal_Pearson']:.4f}")
    return best_test, best_epoch, history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    _state['started_at'] = datetime.now()
    _seed_everything(SEED)
    log(f"[*] Seed fixed to {SEED}")
    log(f"[*] BiPhysFormer (PhysFormer official + BiLevel Routing Attention)")
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
