"""
현재 metrics.py 의 pooled-Pearson 계산이 학습 신호를 묻어버리는지 검증.

- gt: 클립별 z-norm 된 정현파 (실제 BVP 모사)
- pred: gt 와 클립별 위상이 잘 맞지만 클립별 baseline 만 다른 신호
       => per-clip Pearson 은 1 에 가깝지만 pooled Pearson 은 ~0
"""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.metrics import calculate_metrics


def per_clip_pearson(pred, gt):
    pearsons = []
    for i in range(pred.shape[0]):
        p = pred[i] - pred[i].mean()
        g = gt[i] - gt[i].mean()
        denom = (np.sqrt((p*p).sum()) * np.sqrt((g*g).sum())) + 1e-9
        pearsons.append((p*g).sum() / denom)
    return float(np.mean(pearsons))


def make_synthetic(n_clips=10, T=160, fps=30, hr_min=50, hr_max=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    gt = np.zeros((n_clips, T), dtype=np.float32)
    pred = np.zeros((n_clips, T), dtype=np.float32)
    pred_const = np.zeros((n_clips, T), dtype=np.float32)
    for i in range(n_clips):
        hr = rng.uniform(hr_min, hr_max)
        phase = rng.uniform(0, 2*np.pi)
        wave = np.sin(2*np.pi*(hr/60)*t + phase)
        # gt = z-norm
        gt[i] = (wave - wave.mean()) / (wave.std() + 1e-9)
        # pred1: same waveform shape but small additive offset per clip
        pred[i] = gt[i] + rng.normal(0, 0.1, T)   # high per-clip Pearson
        # pred2: near-constant zero output (collapse mode)
        pred_const[i] = rng.normal(0, 0.01, T)
    return torch.from_numpy(pred), torch.from_numpy(pred_const), torch.from_numpy(gt)


def run():
    pred_good, pred_collapse, gt = make_synthetic()

    print("=== Case A: pred matches per-clip waveform (should give ~1.0 Pearson) ===")
    m = calculate_metrics(pred_good, gt)
    print(f"  metrics.py (pooled flatten):  Pearson = {m['Pearson']:.6f}")
    pc = per_clip_pearson(pred_good.numpy(), gt.numpy())
    print(f"  per-clip mean (correct ref):  Pearson = {pc:.6f}")

    print()
    print("=== Case B: pred = constant ~0 (collapse mode, should give ~0.0 Pearson) ===")
    m = calculate_metrics(pred_collapse, gt)
    print(f"  metrics.py (pooled flatten):  Pearson = {m['Pearson']:.6f}")
    pc = per_clip_pearson(pred_collapse.numpy(), gt.numpy())
    print(f"  per-clip mean (correct ref):  Pearson = {pc:.6f}")


if __name__ == '__main__':
    run()
