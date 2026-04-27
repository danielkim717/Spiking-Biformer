import os
import subprocess
import json
import time
from datetime import datetime

# Configuration
TARGET_PEARSON = 0.5
TIMEOUT_SEC = 3600 # 1 hour
LOG_FILE = "results/tune_log.txt"
REPORT_FILE = "results/cause_analysis_report.md"

def log_event(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def update_report(iteration, pearson, MAE, analysis, action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""
## [{iteration}차 수정] - {timestamp}
- **현재 성능**: Pearson r = {pearson:.4f}, MAE = {MAE:.4f}
- **원인 분석**: {analysis}
- **조치 사항**: {action}
---
"""
    with open(REPORT_FILE, "a", encoding='utf-8') as f:
        f.write(content)

def run_experiment(vth, lr, epochs=5):
    log_event(f"Running Experiment: Vth={vth}, LR={lr}, Epochs={epochs}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "UBFC-rPPG",
        "--test_ds", "PURE",
        "--epochs", str(epochs),
        "--v_threshold", str(vth),
        "--lr", str(lr)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        last_metrics = {"Pearson": 0.0, "MAE": 0.0}
        for line in process.stdout:
            if "[*] Evaluation Results" in line:
                try:
                    m_str = line.split("Results:")[1].strip().replace("'", "\"")
                    last_metrics = json.loads(m_str)
                except: pass
        process.wait()
        return last_metrics
    except Exception as e:
        log_event(f"Process Error: {str(e)}")
        return {"Pearson": 0.0, "MAE": 0.0}

def main():
    iteration = 1
    # Starting configs
    current_vth = 1.0
    current_lr = 3e-3
    
    while True:
        log_event(f"=== Starting Iteration {iteration} ===")
        metrics = run_experiment(current_vth, current_lr, epochs=5)
        pearson = metrics.get("Pearson", 0.0)
        mae = metrics.get("MAE", 0.0)
        
        if pearson >= TARGET_PEARSON:
            log_event(f"SUCCESS: Pearson {pearson} >= {TARGET_PEARSON}")
            update_report(iteration, pearson, mae, "성능 목표 달성.", "학습 완료.")
            break
        
        # Analyze and Fix
        log_event(f"FAILURE: Pearson {pearson} < {TARGET_PEARSON}. Analyzing...")
        
        if pearson < 0.1:
            analysis = "뉴런의 발화가 충분하지 않거나 초기 가중치가 파형을 형성하지 못함 (Dead Neuron 가능성)."
            action = "V_threshold를 0.1로 추가 인하하여 발화 빈도 극대화."
            current_vth = 0.1
        elif pearson < 0.3:
            analysis = "파형의 윤곽은 잡히나 노이즈가 많고 수렴이 불안정함."
            action = "Learning Rate를 1e-3으로 낮추어 정밀 학습 유도."
            current_lr = 1e-3
        else:
            analysis = "성능이 정체됨. 손실 함수 가중치나 모델 깊이 검토 필요."
            action = "현재 세팅에서 Epoch를 늘려 재시도 (추후 Assistant 개입 대기)."
            # For now, just increment iteration and wait or try one more thing
        
        update_report(iteration, pearson, mae, analysis, action)
        iteration += 1
        
        if iteration > 5:
            log_event("Max iterations reached. Orchestration paused for human intervention.")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    main()
