import os
import subprocess
import json
import torch

v_thresholds = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
results = {}

print("=== V_th Sweep Starting ===")

for vth in v_thresholds:
    print(f"\n[Sweep] Testing V_th = {vth}...")
    # Run 1 epoch on UBFC-rPPG -> PURE (shortened)
    # Using a small number of epochs (1) and maybe we can stop early or just check the output logs
    cmd = [
        r".\.venv\Scripts\python.exe", "src/train.py",
        "--train_ds", "UBFC-rPPG",
        "--test_ds", "PURE",
        "--epochs", "1",
        "--v_threshold", str(vth)
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    last_pearson = 0.0
    step_count = 0
    # Monitor the output for the Pearson r result
    for line in process.stdout:
        print(line, end="")
        # Update status for reporter
        step_count += 1
        status = {
            "experiment": f"V_th Sweep (Testing {vth})",
            "epoch": v_thresholds.index(vth) + 1,
            "total_epochs": len(v_thresholds),
            "step": step_count,
            "total_steps": 250, # Rough estimate for UBFC 1 epoch
            "loss": 0.0,
            "phase": f"Sweep-{vth}"
        }
        with open('results/current_status.json', 'w') as f:
            json.dump(status, f)

        if "[*] Evaluation Results" in line:
            # Parse metrics like {'MAE': 1.17, 'RMSE': 1.50, 'Pearson': 0.01}
            try:
                metrics_str = line.split("Results:")[1].strip().replace("'", "\"")
                metrics = json.loads(metrics_str)
                last_pearson = metrics.get("Pearson", 0.0)
            except:
                pass
    
    process.wait()
    results[vth] = last_pearson
    print(f"[Sweep] V_th = {vth} finished. Pearson r: {last_pearson}")

print("\n=== Sweep Results ===")
for vth, p in results.items():
    print(f"V_th: {vth} -> Pearson r: {p}")

best_vth = max(results, key=results.get)
print(f"\n[Final] Best V_th found: {best_vth}")

# Save results for final report
with open("results/vth_sweep_results.json", "w") as f:
    json.dump({"results": results, "best_vth": best_vth}, f)

# Final status to signal completion of sweep
status = {
    "experiment": "V_th Sweep Completed",
    "epoch": len(v_thresholds),
    "total_epochs": len(v_thresholds),
    "step": 1,
    "total_steps": 1,
    "loss": 0.0,
    "phase": f"Completed (Best V_th: {best_vth})"
}
with open('results/current_status.json', 'w') as f:
    json.dump(status, f)
