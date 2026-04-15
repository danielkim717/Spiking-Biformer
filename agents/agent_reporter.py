"""
Periodic Reporter Agent: 30분마다 학습 진행 상황을 집계하여 보고서를 작성합니다.
"""
import os
import json
import time
import subprocess
from datetime import datetime

def push_to_github():
    try:
        subprocess.run(['git', 'add', 'results/'], check=True)
        subprocess.run(['git', 'commit', '-m', f"report: periodic progress update {datetime.now().strftime('%H:%M')}"], stderr=subprocess.DEVNULL)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
    except:
        pass

def generate_report():
    status_file = 'results/current_status.json'
    live_report = 'results/live_progress.md'
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            data = json.load(f)
        
        progress_pct = (data['epoch'] - 1) / data['total_epochs'] * 100 + (data['step'] / data['total_steps'] / data['total_epochs'] * 100)
        
        report = f"""# 실시간 학습 진행 보고서
**마지막 업데이트**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 현재 진행 상태
- **현재 실험**: `{data['experiment']}`
- **진행률**: `[{'#' * int(progress_pct/5)}{'.' * (20 - int(progress_pct/5))}]` {progress_pct:.1f}%
- **Epoch**: {data['epoch']} / {data['total_epochs']}
- **Step**: {data['step']} / {data['total_steps']}
- **마지막 Loss**: `{data['loss']:.6f}`

## 🚀 하드웨어 상태 (RTX 4060)
- **메모리 최적화**: Zip-Direct Loading 활성화 (디스크 절약 중)
"""
        with open(live_report, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 주기적 로그에도 추가
        with open('results/periodic_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M')}] Experiment: {data['experiment']}, Epoch: {data['epoch']}, Loss: {data['loss']:.4f}\n")

def main():
    print("[Reporter Agent] 30분 간격 정기 보고 시스템 가동...")
    while True:
        generate_report()
        push_to_github()
        # 30분(1800초) 대기
        time.sleep(1800)

if __name__ == '__main__':
    main()
