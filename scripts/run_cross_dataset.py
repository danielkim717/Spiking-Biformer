"""
Cross-dataset 학습 러너.

- PURE -> UBFC-rPPG, UBFC-rPPG -> PURE 순차 실행
- 5분마다 results/live_progress.md 자동 갱신 (백그라운드 스레드)
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

EXPERIMENTS = [
    ('PURE', 'UBFC-rPPG'),
    ('UBFC-rPPG', 'PURE'),
]

EPOCHS = 30


_state = {
    'started_at': None,
    'experiments': EXPERIMENTS,
    'current_idx': 0,
    'completed': [],   # list of dicts: {pair, metrics}
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
    cur_idx = _state['current_idx']
    total_exps = len(_state['experiments'])

    # 진행 상황 추정
    epoch = status.get('epoch', 0)
    total_epochs = status.get('total_epochs', EPOCHS)
    step = status.get('step', 0)
    total_steps = max(1, status.get('total_steps', 1))
    phase = status.get('phase', '대기중')
    loss = status.get('loss', 0.0)
    firing_rates = status.get('firing_rates', [])

    # 현재 실험 내 진행률 (epoch + step 기반)
    cur_exp_pct = 0.0
    if total_epochs > 0:
        cur_exp_pct = ((max(0, epoch - 1) + step / total_steps) / total_epochs) * 100.0
    cur_exp_pct = min(100.0, cur_exp_pct)

    # 전체 진행률
    overall_pct = (cur_idx / total_exps) * 100.0 + (cur_exp_pct / total_exps)
    overall_pct = min(100.0, overall_pct)

    # ETA
    elapsed = (now - _state['started_at']).total_seconds() if _state['started_at'] else 0
    eta_s = None
    if overall_pct > 0.5:
        total_estimate = elapsed / (overall_pct / 100.0)
        eta_s = total_estimate - elapsed

    lines = []
    lines.append('# 📊 실시간 학습 진행 보고서 및 성능 비교')
    lines.append('')
    lines.append(f"**마지막 업데이트**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('')
    lines.append('## 🎯 현재 학습 상태')

    if cur_idx < total_exps:
        train_ds, test_ds = _state['experiments'][cur_idx]
        lines.append(f"- **현재 실험**: `{train_ds} → {test_ds}` ({cur_idx + 1}/{total_exps})")
        lines.append(f"- **진행 단계**: **{phase}** ⏳")
        lines.append(f"- **현재 실험 진행률**: `{_make_progress_bar(cur_exp_pct)}` {cur_exp_pct:.1f}%")
        lines.append(f"- **전체 진행률**: `{_make_progress_bar(overall_pct)}` {overall_pct:.1f}%")
        lines.append(f"- **Epoch**: {epoch} / {total_epochs}")
        lines.append(f"- **Step**: {step} / {total_steps}")
        lines.append(f"- **예상 남은 시간(ETA)**: 약 {_format_eta(eta_s)}")
        lines.append(f"- **현재 Loss**: `{loss:.6f}`")
    else:
        lines.append('- **현재 실험**: `모두 완료 ✅`')
        lines.append('- **전체 진행률**: `[####################]` 100.0%')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 🔬 SNN 스파이크 모니터링 (Spike Firing Rate)')
    lines.append('본 지표는 각 LIF 층의 발화율입니다 (0%면 소실, 5~25%가 건강한 영역).')
    if firing_rates:
        rates_str = ', '.join(f"{r*100:.2f}%" for r in firing_rates)
        avg = sum(firing_rates) / len(firing_rates)
        lines.append(f"> **현재 발화율**: `[{rates_str}]`  (평균 {avg*100:.2f}%)")
    else:
        lines.append('> **현재 발화율**: `계산 중...`')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 🏆 rPPG 모델 간 성능 비교 (Cross-Dataset)')
    lines.append('')
    lines.append('| 모델 (Model)             | Train/Test | MAE ↓ | RMSE ↓ | Pearson r ↑ | 비고                  |')
    lines.append('| :---                     | :---       | :---: | :---: | :---:       | :---                  |')
    lines.append('| **DeepPhys (CNN)**       | UBFC/PURE  | 3.45  | 4.56  | 0.54        | Baseline 2018         |')
    lines.append('| **Physformer (ViT)**     | UBFC/PURE  | 2.37  | 3.12  | 0.82        | High Power            |')
    lines.append('| **Spiking Physformer**   | UBFC/PURE  | 2.21  | 2.98  | **0.85**    | SNN SOTA              |')

    # 완료된 실험의 결과 행
    completed_map = {c['pair']: c['metrics'] for c in _state['completed']}
    for (tr, te) in _state['experiments']:
        key = f"{tr}->{te}"
        if key in completed_map:
            m = completed_map[key]
            lines.append(f"| **Spiking Bi-Physformer**| {tr}/{te}  | {m['MAE']:.2f}  | {m['RMSE']:.2f}  | **{m['Pearson']:.2f}**    | **Proposed (SDLA+MS)**|")
        else:
            lines.append(f"| **Spiking Bi-Physformer**| {tr}/{te}  | (대기 중) | (대기 중) | (대기 중) | **Proposed (SDLA+MS)**|")
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('## 🛠️ 최근 작업 타임라인')
    lines.append(f"- `[{_state['started_at'].strftime('%H:%M') if _state['started_at'] else '--:--'}]` Cross-dataset 실험 시작 ({total_exps}개)")
    for c in _state['completed']:
        lines.append(f"- `[완료]` **{c['pair']}** — Pearson r = **{c['metrics']['Pearson']:.4f}**, MAE = {c['metrics']['MAE']:.4f}, RMSE = {c['metrics']['RMSE']:.4f}")
    if cur_idx < total_exps:
        train_ds, test_ds = _state['experiments'][cur_idx]
        lines.append(f"- `[진행]` **{train_ds} → {test_ds}** {phase} (Epoch {epoch}/{total_epochs}, Step {step}/{total_steps})")
    lines.append('')
    lines.append('---')
    lines.append('*본 보고서는 5분마다 자동 갱신됩니다.*')
    lines.append('')

    os.makedirs('results', exist_ok=True)
    with open(LIVE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _progress_loop(interval=300):
    """5분(300초)마다 live_progress.md 갱신."""
    while not _state['stop']:
        try:
            _write_live_progress()
        except Exception as e:
            print(f"[progress logger] 갱신 실패: {e}")
        # 1초 단위로 sleep하여 종료 신호에 빠르게 응답
        for _ in range(interval):
            if _state['stop']:
                break
            time.sleep(1)


def main():
    _state['started_at'] = datetime.now()

    logger = threading.Thread(target=_progress_loop, args=(300,), daemon=True)
    logger.start()

    # 초기 상태 1회 기록
    _write_live_progress()

    try:
        for idx, (train_ds, test_ds) in enumerate(EXPERIMENTS):
            _state['current_idx'] = idx
            print(f"\n{'='*60}")
            print(f"[*] 실험 {idx+1}/{len(EXPERIMENTS)}: {train_ds} -> {test_ds}")
            print(f"{'='*60}\n")
            _write_live_progress()

            # 학습 + 평가 (run_experiment 내부에서 매 epoch 평가 후 results/experiment_summary.md 갱신)
            metrics = run_experiment(train_ds, test_ds, epochs=EPOCHS)

            # run_experiment는 명시적 반환값이 없으므로 마지막 평가 결과를 summary 파일에서 읽어와야 한다.
            # 대신 status 파일과 summary md를 파싱한다.
            last = _read_summary_last_for_pair(f"{train_ds}->{test_ds}")
            if last is not None:
                _state['completed'].append({'pair': f"{train_ds}->{test_ds}", 'metrics': last})
            _write_live_progress()

        _state['current_idx'] = len(EXPERIMENTS)
        _write_live_progress()
        print("\n[*] 모든 cross-dataset 실험 완료.\n")
    finally:
        _state['stop'] = True
        # 마지막 1회 갱신
        try:
            _write_live_progress()
        except Exception:
            pass


def _read_summary_last_for_pair(pair):
    """results/experiment_summary.md에서 해당 pair의 마지막 줄을 파싱."""
    path = os.path.join('results', 'experiment_summary.md')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        last = None
        for line in lines:
            if pair in line and '|' in line:
                parts = [p.strip() for p in line.strip().strip('|').split('|')]
                # | dataset | 상태 | MAE | RMSE | Pearson | 비고 |
                if len(parts) >= 5:
                    try:
                        mae = float(parts[2]); rmse = float(parts[3]); pr = float(parts[4])
                        last = {'MAE': mae, 'RMSE': rmse, 'Pearson': pr}
                    except ValueError:
                        continue
        return last
    except Exception:
        return None


if __name__ == '__main__':
    main()
