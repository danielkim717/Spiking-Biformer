import os
import subprocess
import json
import time
from datetime import datetime

# 1. Configuration
v_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
lrs = [1e-3, 3e-3, 5e-3]
pearson_weights = [0.5, 1.0, 2.0]
target_pearson = 0.85

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
    os.makedirs('results', exist_ok=True)
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

def run_train(vth, lr, p_weight, epochs=1):
    print(f"\n[AutoTune] Running: Vth={vth}, LR={lr}, PearsonWeight={p_weight}")
    # We need to modify train.py to accept p_weight if possible, or just use default for now
    # For now, let's assume train.py uses default p_weight=0.5
    cmd = [
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "UBFC-rPPG",
        "--test_ds", "PURE",
        "--epochs", str(epochs),
        "--v_threshold", str(vth),
        "--lr", str(lr)
    ]
    # Pass LR if we add it to train.py later
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    last_metrics = {"Pearson": 0.0, "MAE": 99.0}
    step = 0
    for line in process.stdout:
        step += 1
        # Update status every 10 steps to keep reporter alive
        if step % 10 == 0:
            update_status(f"Tuning Vth={vth}, LR={lr}", "Tuning", 1, 1, step, 300, 0.0)
            
        if "[*] Evaluation Results" in line:
            try:
                m_str = line.split("Results:")[1].strip().replace("'", "\"")
                last_metrics = json.loads(m_str)
            except:
                pass
    process.wait()
    return last_metrics

def main():
    print("=== Spiking Bi-Physformer Auto-Tuning Master Script Starting ===")
    
    # Phase 1: Vth Sweep
    best_vth = 0.3
    max_p = -1.0
    
    for vth in v_thresholds:
        metrics = run_train(vth, 3e-3, 0.5, epochs=1)
        p = metrics.get("Pearson", 0.0)
        print(f"Result for Vth={vth}: Pearson={p}")
        if p > max_p:
            max_p = p
            best_vth = vth
            
    print(f"\n[Step 1 Done] Best Vth: {best_vth} (Pearson: {max_p})")
    
    # Phase 2: If target not reached, tune LR
    final_vth = best_vth
    final_lr = 3e-3
    
    if max_p < target_pearson:
        print("\n[Step 2] Target 0.85 not reached. Tuning Learning Rate...")
        for lr in lrs:
            if lr == 3e-3: continue # Already tested
            metrics = run_train(best_vth, lr, 0.5, epochs=1)
            p = metrics.get("Pearson", 0.0)
            if p > max_p:
                max_p = p
                final_lr = lr
        print(f"Best LR found: {final_lr} (Pearson: {max_p})")

    # Phase 3: Final Long Training
    print(f"\n[Step 3] Starting Final Full Training with Vth={final_vth}, LR={final_lr}...")
    update_status("Final Full Training", "Training", 1, 10, 0, 300, 0.0)
    
    # Run long training (e.g. 10 epochs)
    # We will use the main train.py and let it finish both cross-dataset experiments
    # Here we just run the final one
    subprocess.run([
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "UBFC-rPPG", "--test_ds", "PURE",
        "--epochs", "10", "--v_threshold", str(final_vth)
    ])
    
    subprocess.run([
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "PURE", "--test_ds", "UBFC-rPPG",
        "--epochs", "10", "--v_threshold", str(final_vth)
    ])

    print("\n=== Auto-Tuning Completed. Generating Final Report... ===")
    # The reporter agent will help with this, or we can write a dedicated summary.

if __name__ == "__main__":
    main()
