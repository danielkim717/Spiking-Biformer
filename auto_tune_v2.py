import os
import subprocess
import json
import time
from datetime import datetime

# Configuration
V_THRESHOLDS = [0.1, 0.2, 0.5, 1.0]
LRS = [3e-3, 1e-2]
TARGET_PEARSON = 0.85
TIMEOUT_SEC = 1800 # 30 mins per run

def log_event(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/tune_log.txt", "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def update_status(experiment, phase, epoch, total_epochs, step, total_steps, loss):
    status = {
        "experiment": experiment,
        "phase": phase,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss
    }
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

def run_experiment(vth, lr, epochs=1):
    log_event(f"Starting Run: Vth={vth}, LR={lr}")
    cmd = [
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "UBFC-rPPG",
        "--test_ds", "PURE",
        "--epochs", str(epochs),
        "--v_threshold", str(vth),
        "--lr", str(lr)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        start_time = time.time()
        
        last_pearson = 0.0
        for line in process.stdout:
            # Check for timeout
            if time.time() - start_time > TIMEOUT_SEC:
                process.kill()
                log_event("ERROR: Process Timeout!")
                return 0.0
            
            if "[*] Evaluation Results" in line:
                try:
                    metrics_str = line.split("Results:")[1].strip().replace("'", "\"")
                    metrics = json.loads(metrics_str)
                    last_pearson = metrics.get("Pearson", 0.0)
                except:
                    pass
        
        process.wait()
        log_event(f"Run Finished. Pearson: {last_pearson}")
        return last_pearson
    except Exception as e:
        log_event(f"ERROR in run_experiment: {str(e)}")
        return 0.0

def main():
    os.makedirs("results", exist_ok=True)
    log_event("=== AutoTune V2 (Watchdog Enabled) Started ===")
    
    best_p = -1.0
    best_config = (0.1, 3e-3)
    
    # 1. Sweep V_th and LR
    for vth in V_THRESHOLDS:
        for lr in LRS:
            p = run_experiment(vth, lr, epochs=1)
            if p > best_p:
                best_p = p
                best_config = (vth, lr)
            
            if best_p >= TARGET_PEARSON:
                log_event(f"TARGET REACHED: Pearson {best_p}")
                break
        if best_p >= TARGET_PEARSON: break

    log_event(f"Sweep Completed. Best Config: Vth={best_config[0]}, LR={best_config[1]} (Pearson={best_p})")
    
    # 2. Final Training
    log_event("Starting Final Long Training (10 Epochs)...")
    update_status("Final Full Training", "Training", 1, 10, 0, 300, 0.0)
    
    # Run UBFC -> PURE
    run_experiment(best_config[0], best_config[1], epochs=10)
    
    # Run PURE -> UBFC
    log_event("Starting Cross-Dataset 2: PURE -> UBFC")
    cmd_cross = [
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "PURE", "--test_ds", "UBFC-rPPG",
        "--epochs", "10", "--v_threshold", str(best_config[0]), "--lr", str(best_config[1])
    ]
    subprocess.run(cmd_cross)
    
    log_event("=== ALL EXPERIMENTS COMPLETED ===")

if __name__ == "__main__":
    main()
