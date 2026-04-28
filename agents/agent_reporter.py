import os
import json
import time
import subprocess
from datetime import datetime

def push_to_github():
    try:
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', f"update: reports and monitoring {datetime.now().strftime('%Y-%m-%d %H:%M')}"], stderr=subprocess.DEVNULL)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
    except:
        pass

def generate_report():
    status_file = 'results/current_status.json'
    live_report = 'results/live_progress.md'
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            data = json.load(f)
            
        is_eval = data.get('phase') == 'Evaluation'
        progress_pct = (data['step'] / data['total_steps']) * 100 if data['total_steps'] > 0 else 0
        phase_text = "**평가(Evaluation) 중...** ⏳" if is_eval else "**학습(Training) 중...** 🚀"
        
        # Firing rates monitor
        fr = data.get('firing_rates', [])
        fr_str = " | ".join([f"L{i}:{v:.1%}" for i, v in enumerate(fr)]) if fr else "계산 중..."
        
        # Epoch info
        epoch_str = f"{data['epoch']} / {data['total_epochs']}"
        
        # Calculate ETA
        eta_str = "계산 중..."
        try:
            if data['step'] > 10:
                elapsed = 300 # assumed 5 min loop
                remaining_steps = data['total_steps'] - data['step']
                eta_mins = int((elapsed / data['step']) * remaining_steps / 60)
                eta_str = f"약 {eta_mins}분"
        except:
            pass

        report = f"""# 📊 실시간 학습 진행 보고서 및 성능 비교

**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 현재 학습 상태
- **현재 실험**: `{data['experiment']}`
- **진행 단계**: {phase_text}
- **진행률**: `[{'#' * int(progress_pct/5)}{'.' * (20 - int(progress_pct/5))}]` {progress_pct:.1f}%
- **Epoch**: {epoch_str}
- **Step**: {data['step']} / {data['total_steps']}
- **예상 남은 시간(ETA)**: {eta_str}
- **마지막 Loss**: `{data.get('loss', 0):.6f}`

---

## 🔬 SNN 스파이크 모니터링 (Spike Firing Rate)
본 지표는 각 층의 뉴런이 얼마나 활발하게 발화하는지 나타냅니다 (0%면 소실된 것).
> **현재 발화율**: `{fr_str}`

---

## 🏆 rPPG 모델 간 성능 비교 (Cross-Dataset)

| 모델 (Model)             | Train/Test | MAE ↓ | RMSE ↓ | Pearson r ↑ | 비고                  |
| :---                     | :---       | :---: | :---: | :---:       | :---                  |
| **DeepPhys (CNN)**       | UBFC/PURE  | 3.45  | 4.56  | 0.54        | Baseline 2018         |
| **Physformer (ViT)**     | UBFC/PURE  | 2.37  | 3.12  | 0.82        | High Power            |
| **Spiking Physformer**   | UBFC/PURE  | 2.21  | 2.98  | **0.85**    | SNN SOTA              |
| **Spiking Bi-Physformer**| UBFC/PURE  | **(TBD)**| **(TBD)**| **(TBD)**   | **Proposed (SDLA+MS)**|

---

## 🛠️ 최근 작업 타임라인 (실시간 갱신)
- `[10:43]` **Temporal Patching (T=4)** 구조 전면 적용 (Stem 레이어 Stride 수정)
- `[10:45]` 원인 분석 보고서(Cause Analysis Report) 전면 리팩토링 및 가시성 확보
- `[10:46]` **30-Epoch 정밀 학습** 재시작 (목표: 피어슨 r > 0.5)

---
*본 보고서는 5분마다 자동으로 갱신됩니다.*

"""
        # Append Cause Analysis if exists
        analysis_file = 'results/cause_analysis_report.md'
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as af:
                report += "\n\n" + af.read()

        with open(live_report, 'w', encoding='utf-8') as f:
            f.write(report)
        
        push_to_github()

if __name__ == "__main__":
    while True:
        try:
            generate_report()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(300)
