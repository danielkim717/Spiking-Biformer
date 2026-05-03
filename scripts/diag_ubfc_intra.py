"""
UBFC-rPPG intra-dataset 진단.

목적: PURE 데이터 정합성 문제를 배제하고 모델 자체의 학습 능력 검증.
- UBFC 만 사용 (BVP 와 frame 이 1:1 비율로 깨끗)
- subject 1~37 → train, subject 38~49 → test (30:12 분할)
- face_crop=True (frame 가운데 정사각형 crop)
- 5 epoch 학습

판정 기준:
- Pearson r > +0.05 → 모델은 정상. cross-dataset 도메인 갭이 주요 원인.
- Pearson r < +0.05 → 모델 또는 다른 미발견 결함이 남아 있음.
"""
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional

from src.models.phys_biformer import PhysBiformer
from src.data.rppg_dataset import get_dataloader
from src.train import NegPearsonLoss, FrequencyLoss
from src.utils.metrics import calculate_metrics, update_experiment_summary


# 30 train / 12 test
TRAIN_SUBJECTS = ['subject1', 'subject3', 'subject4', 'subject5', 'subject8', 'subject9',
                  'subject10', 'subject11', 'subject12', 'subject13', 'subject14',
                  'subject15', 'subject16', 'subject17', 'subject18', 'subject20',
                  'subject22', 'subject23', 'subject24', 'subject25', 'subject26',
                  'subject27', 'subject30', 'subject31', 'subject32', 'subject33',
                  'subject34', 'subject35', 'subject36', 'subject37']
TEST_SUBJECTS  = ['subject38', 'subject39', 'subject40', 'subject41', 'subject42',
                  'subject43', 'subject44', 'subject45', 'subject46', 'subject47',
                  'subject48', 'subject49']

EPOCHS = 5
BATCH_SIZE = 2
LR = 3e-3
WD = 5e-5
V_THRESHOLD = 1.0
DATA_ROOT = 'D:\\UBFC-rPPG'

LIVE_FILE = os.path.join('results', 'live_progress.md')
STATUS_FILE = os.path.join('results', 'current_status.json')


_state = {'started_at': None, 'stop': False}


def _read_status():
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _save_status(d):
    os.makedirs('results', exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f)


def _format_eta(seconds):
    if seconds is None or seconds <= 0:
        return '계산 중...'
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if td.days > 0:
        return f"{td.days}일 {h}시간 {m}분"
    if h > 0:
        return f"{h}시간 {m}분"
    return f"{m}분 {s}초"


def _make_progress_bar(pct, width=20):
    filled = int(round(pct / 100.0 * width))
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def _write_live_progress():
    now = datetime.now()
    status = _read_status() or {}
    epoch = status.get('epoch', 0)
    total_epochs = status.get('total_epochs', EPOCHS)
    step = status.get('step', 0)
    total_steps = max(1, status.get('total_steps', 1))
    phase = status.get('phase', '대기중')
    loss = status.get('loss', 0.0)
    firing_rates = status.get('firing_rates', [])
    last_metric = status.get('last_metric', {})

    cur_pct = ((max(0, epoch - 1) + step / total_steps) / max(1, total_epochs)) * 100.0
    cur_pct = min(100.0, cur_pct)

    elapsed = (now - _state['started_at']).total_seconds() if _state['started_at'] else 0
    eta_s = None
    if cur_pct > 0.5:
        total_estimate = elapsed / (cur_pct / 100.0)
        eta_s = total_estimate - elapsed

    lines = []
    lines.append('# 📊 UBFC Intra-Dataset 진단 (face_crop ON)')
    lines.append('')
    lines.append(f"**마지막 업데이트**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')
    lines.append('## 🎯 현재 상태')
    lines.append(f"- **실험**: UBFC subjects 1~37 → train, 38~49 → test (intra-dataset)")
    lines.append(f"- **단계**: **{phase}** ⏳")
    lines.append(f"- **진행률**: `{_make_progress_bar(cur_pct)}` {cur_pct:.1f}%")
    lines.append(f"- **Epoch**: {epoch} / {total_epochs}")
    lines.append(f"- **Step**: {step} / {total_steps}")
    lines.append(f"- **ETA**: 약 {_format_eta(eta_s)}")
    lines.append(f"- **Loss**: `{loss:.6f}`")
    lines.append('')
    lines.append('## 🔬 SNN 발화율')
    if firing_rates:
        rates_str = ', '.join(f"{r*100:.2f}%" for r in firing_rates)
        avg = sum(firing_rates) / len(firing_rates)
        lines.append(f"> `[{rates_str}]`  (평균 {avg*100:.2f}%)")
    else:
        lines.append('> 계산 중...')
    lines.append('')
    if last_metric:
        lines.append('## 📌 직전 epoch 평가 결과')
        lines.append(f"- Epoch {last_metric.get('epoch', '?')}")
        lines.append(f"- MAE = {last_metric.get('MAE', 0):.4f}")
        lines.append(f"- RMSE = {last_metric.get('RMSE', 0):.4f}")
        lines.append(f"- **Pearson r = {last_metric.get('Pearson', 0):.4f}**")
        lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📌 변경 사항 (이번 진단 run)')
    lines.append('- UBFC intra-dataset (subject 분할) — PURE alignment 문제 배제')
    lines.append('- `face_crop=True` (frame 가운데 정사각형 crop) — (C) preprocessing 결함 보완')
    lines.append('- 그 외 모델/손실/하이퍼파라미터 일체 동일')
    lines.append('')

    os.makedirs('results', exist_ok=True)
    with open(LIVE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _progress_loop(interval=300):
    while not _state['stop']:
        try:
            _write_live_progress()
        except Exception as e:
            print(f"[progress logger] 갱신 실패: {e}", flush=True)
        for _ in range(interval):
            if _state['stop']:
                break
            time.sleep(1)


def main():
    _state['started_at'] = datetime.now()
    logger = threading.Thread(target=_progress_loop, args=(300,), daemon=True)
    logger.start()
    _write_live_progress()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] UBFC intra-dataset diagnostic on {device}", flush=True)
    print(f"    Train subjects: {len(TRAIN_SUBJECTS)} | Test subjects: {len(TEST_SUBJECTS)}", flush=True)
    print(f"    face_crop=True", flush=True)

    train_loader = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=TRAIN_SUBJECTS, shuffle=True)
    test_loader  = get_dataloader('UBFC-rPPG', DATA_ROOT, BATCH_SIZE, clip_len=160,
                                  face_crop=True, subjects_filter=TEST_SUBJECTS, shuffle=False)
    print(f"    train clips: {len(train_loader.dataset)}  test clips: {len(test_loader.dataset)}", flush=True)

    model = PhysBiformer(frame=160, patches=(4, 4, 4), v_threshold=V_THRESHOLD).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    last_metric = {}

    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = max(1, len(train_loader))

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss_pearson = pearson_criterion(outputs, labels)
                loss_ce, loss_ld = freq_criterion(outputs, labels)
                loss = loss_pearson + (loss_ce + loss_ld)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                functional.reset_net(model)

                if i % 10 == 0:
                    print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.6f}", flush=True)
                    _save_status({
                        'experiment': 'UBFC-intra',
                        'epoch': epoch + 1,
                        'total_epochs': EPOCHS,
                        'step': i,
                        'total_steps': len(train_loader),
                        'loss': loss.item(),
                        'phase': 'Training',
                        'firing_rates': getattr(model, 'last_firing_rates', []),
                        'last_metric': last_metric,
                    })

            print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/n_batches:.6f}", flush=True)

            # Eval
            print(f"[*] Starting Evaluation for Epoch {epoch+1}...", flush=True)
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(test_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    functional.reset_net(model)
                    all_preds.append(outputs.cpu())
                    all_gts.append(labels.cpu())
                    if j % 10 == 0:
                        _save_status({
                            'experiment': 'UBFC-intra',
                            'epoch': epoch + 1,
                            'total_epochs': EPOCHS,
                            'step': j,
                            'total_steps': len(test_loader),
                            'loss': 0.0,
                            'phase': 'Evaluation',
                            'last_metric': last_metric,
                        })
            all_preds = torch.cat(all_preds)
            all_gts = torch.cat(all_gts)
            metrics = calculate_metrics(all_preds, all_gts)
            metrics_native = {k: float(v) for k, v in metrics.items()}
            metrics_native['epoch'] = epoch + 1
            print(f"[*] Epoch {epoch+1} Results: {metrics}", flush=True)
            update_experiment_summary('UBFC-intra', metrics)
            last_metric = metrics_native
            _save_status({
                'experiment': 'UBFC-intra',
                'epoch': epoch + 1,
                'total_epochs': EPOCHS,
                'step': 0,
                'total_steps': 1,
                'loss': 0.0,
                'phase': f'Epoch {epoch+1} done',
                'last_metric': last_metric,
            })

        print("\n[*] UBFC intra-dataset 진단 완료.\n", flush=True)
    finally:
        _state['stop'] = True
        try:
            _write_live_progress()
        except Exception:
            pass


if __name__ == '__main__':
    main()
