"""
Prediction-head 를 PhysFormer 와 동일하게 수정한 뒤, Pearson 상승 추세를
빠르게 확인하기 위한 스크립트.

- PURE -> UBFC-rPPG, 5 epoch (전체 30 epoch 대신 짧게)
- 5분마다 results/live_progress.md 자동 갱신
"""
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import run_experiment

LIVE_FILE = os.path.join('results', 'live_progress.md')
STATUS_FILE = os.path.join('results', 'current_status.json')

EPOCHS = 5
TRAIN_DS = 'PURE'
TEST_DS = 'UBFC-rPPG'

_state = {
    'started_at': None,
    'stop': False,
}


def _read_status():
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


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

    cur_pct = ((max(0, epoch - 1) + step / total_steps) / max(1, total_epochs)) * 100.0
    cur_pct = min(100.0, cur_pct)

    elapsed = (now - _state['started_at']).total_seconds() if _state['started_at'] else 0
    eta_s = None
    if cur_pct > 0.5:
        total_estimate = elapsed / (cur_pct / 100.0)
        eta_s = total_estimate - elapsed

    lines = []
    lines.append('# 📊 Quick Verify — Prediction-Head 수정 후 Pearson 상승 추세 확인')
    lines.append('')
    lines.append(f"**마지막 업데이트**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')
    lines.append('## 🎯 현재 학습 상태')
    lines.append(f"- **실험**: `{TRAIN_DS} → {TEST_DS}` (Quick verify, {EPOCHS} epoch)")
    lines.append(f"- **단계**: **{phase}** ⏳")
    lines.append(f"- **진행률**: `{_make_progress_bar(cur_pct)}` {cur_pct:.1f}%")
    lines.append(f"- **Epoch**: {epoch} / {total_epochs}")
    lines.append(f"- **Step**: {step} / {total_steps}")
    lines.append(f"- **ETA**: 약 {_format_eta(eta_s)}")
    lines.append(f"- **Loss**: `{loss:.6f}`")
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 🔬 SNN 발화율')
    if firing_rates:
        rates_str = ', '.join(f"{r*100:.2f}%" for r in firing_rates)
        avg = sum(firing_rates) / len(firing_rates)
        lines.append(f"> `[{rates_str}]`  (평균 {avg*100:.2f}%)")
    else:
        lines.append('> 계산 중...')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📌 변경 사항 (이번 run)')
    lines.append('- Prediction head 를 PhysFormer 원본과 동일 구조로 교체')
    lines.append('  - `Upsample(2x temporal) + Conv3d + BN + ELU` × 2 + 공간 GAP + `Conv1d`')
    lines.append('- 그 외 골격/하이퍼파라미터 일체 변경 없음')
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

    try:
        print(f"\n{'='*60}", flush=True)
        print(f"[*] Quick verify: {TRAIN_DS} -> {TEST_DS}, {EPOCHS} epoch", flush=True)
        print(f"{'='*60}\n", flush=True)
        run_experiment(TRAIN_DS, TEST_DS, epochs=EPOCHS)
        print("\n[*] Quick verify 완료.\n", flush=True)
    finally:
        _state['stop'] = True
        try:
            _write_live_progress()
        except Exception:
            pass


if __name__ == '__main__':
    main()
