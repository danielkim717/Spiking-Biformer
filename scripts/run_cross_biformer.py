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

출력:
  - 매 epoch 평가 결과를 results/cross_biformer/log.txt 누적
  - 각 실험 종료 시 best epoch + 최종 metric 종합 리포트
"""
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
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
            lines.append(f"- **{c['name']}**: best Pearson r = **{m['Pearson']:.4f}** "
                         f"(epoch {c['best_epoch']}, MAE {m['MAE']:.4f}, RMSE {m['RMSE']:.4f})")
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
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)
    log(f"  model params: {sum(p.numel() for p in model.parameters())}")
    log(f"  loss: α={ALPHA} (NegPearson) + β·(CE+LD), β = {BETA0}·{ETA}^((e-1)/{EPOCHS})")

    best = {'Pearson': -1.0, 'MAE': 0.0, 'RMSE': 0.0}
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
            optimizer.step()
            functional.reset_net(model)
            epoch_loss += float(loss.item()); nb += 1
            if i % 10 == 0:
                _save_status({
                    'experiment': label, 'epoch': epoch + 1, 'total_epochs': EPOCHS,
                    'step': i, 'total_steps': len(train_loader),
                    'loss': float(loss.item()), 'phase': 'Training',
                })
        avg_loss = epoch_loss / max(1, nb)

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
        mae = float(np.mean(np.abs(preds - gts)))
        rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
        pooled = float(np.corrcoef(preds.flatten(), gts.flatten())[0, 1])
        per_clip = per_clip_pearson(preds, gts)
        history.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'MAE': mae, 'RMSE': rmse,
                        'Pearson': pooled, 'per_clip_Pearson': per_clip})
        log(f"\nEpoch {epoch+1}/{EPOCHS}: Train avg loss {avg_loss:.4f}")
        log(f"  Eval MAE {mae:.4f}  RMSE {rmse:.4f}  pooled-Pearson {pooled:.4f}  "
            f"per-clip-Pearson {per_clip:.4f}")
        if pooled > best['Pearson']:
            best = {'Pearson': pooled, 'MAE': mae, 'RMSE': rmse}
            best_epoch = epoch + 1

    log(f"\n→ Best for {label}: epoch {best_epoch}, Pearson r = {best['Pearson']:.4f}, "
        f"MAE {best['MAE']:.4f}, RMSE {best['RMSE']:.4f}")
    return best, best_epoch, history


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    open(LOG, 'w', encoding='utf-8').close()
    _state['started_at'] = datetime.now()
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
            log(f"  {r['name']}:")
            log(f"    Best @ epoch {r['best_epoch']}: "
                f"Pearson r = {r['best']['Pearson']:.4f}, "
                f"MAE = {r['best']['MAE']:.4f}, RMSE = {r['best']['RMSE']:.4f}")

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
