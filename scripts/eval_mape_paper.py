"""Eval MAPE for all paper-reportable checkpoints.

per-subject MAPE = mean(|pred_hr - gt_hr| / gt_hr) * 100
"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.evaluation import evaluate_per_subject, get_subject_signals


def eval_ckpt(ckpt, dataset, path, split_range, label, pure_mode='subject_exclusive'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    common = dict(face_crop=True, dynamic_detection_freq=0,
                  data_type='diff_normalized', num_workers=4, pin_memory=True)
    if dataset == 'PURE':
        common['pure_split_mode'] = pure_mode
    loader = get_dataloader(dataset, path, batch_size=4, clip_len=160,
                            shuffle=False, chunk_step=80,
                            split_range=split_range, **common)
    model = ViT_BiPulseFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    model.eval()
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=2.0)
            rPPG = (rPPG - rPPG.mean(-1, keepdim=True)) / (rPPG.std(-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu().numpy()); all_g.append(labels.numpy())
    preds = np.concatenate(all_p); gts = np.concatenate(all_g)
    m = evaluate_per_subject(preds, gts, loader.dataset.samples,
                             fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    # Get per-subject HRs to compute test HR distribution
    sigs = get_subject_signals(preds, gts, loader.dataset.samples,
                               fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)
    pred_hrs = np.array([s['hr_pred'] for s in sigs.values()])
    gt_hrs = np.array([s['hr_gt'] for s in sigs.values()])
    print(f"\n=== {label} ===")
    print(f"  ckpt: {ckpt}")
    print(f"  Test n_subjects: {m['n_subjects']}")
    print(f"  Test GT HR: mean={gt_hrs.mean():.2f}, std={gt_hrs.std():.2f}, "
          f"range=[{gt_hrs.min():.1f}, {gt_hrs.max():.1f}]")
    print(f"  --- Paper-reportable metrics (per-subject) ---")
    print(f"  MAE         = {m['MAE_bpm']:.4f} BPM")
    print(f"  RMSE        = {m['RMSE_bpm']:.4f} BPM")
    print(f"  MAPE        = {m['MAPE_pct']:.4f} %")
    print(f"  Pearson(HR) = {m['Pearson']:.4f}")
    print(f"  signal_Pearson = {m['signal_Pearson_mean']:.4f}")
    return m


def main():
    print("=" * 75)
    print("BiPulseFormer — Per-subject (paper-comparable) MAPE 계산")
    print("=" * 75)

    targets = [
        # (label, ckpt, dataset, path, split_range, pure_mode)
        ("UBFC intra 60/40 RhythmFormer protocol (valid=test, 10 ep, StepLR) E8",
         "results/intra_ubfc_bipulseformer/checkpoints/UBFC-rPPG_to_UBFC-rPPG_epoch8.pt",
         "UBFC-rPPG", "D:\\UBFC-rPPG", (0.6, 1.0), 'subject_exclusive'),
        ("UBFC intra 7/1/2 (separate valid, 20 ep, OneCycleLR) E10",
         "results/intra_ubfc_bipulseformer_712_oc20/checkpoints/UBFC-rPPG_epoch10.pt",
         "UBFC-rPPG", "D:\\UBFC-rPPG", (0.8, 1.0), 'subject_exclusive'),
        ("PURE intra 7/1/2 random (separate valid, 20 ep, OneCycleLR) E11",
         "results/intra_pure_bipulseformer_712_oc20/checkpoints/PURE_epoch11.pt",
         "PURE", "D:\\PURE", (0.8, 1.0), 'subject_exclusive_random'),
        ("PURE intra 80/20 (valid=test, 10 ep, StepLR) E9",
         "results/intra_pure_bipulseformer_80_20/checkpoints/PURE_to_PURE_epoch9.pt",
         "PURE", "D:\\PURE", (0.8, 1.0), 'subject_exclusive'),
    ]

    results = []
    for label, ckpt, ds, path, sr, pm in targets:
        if not os.path.exists(ckpt):
            print(f"\n[!] missing: {ckpt}"); continue
        try:
            m = eval_ckpt(ckpt, ds, path, sr, label, pure_mode=pm)
            results.append((label, m))
        except Exception as e:
            print(f"[!] {label} failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 90)
    print(" FINAL SUMMARY (per-subject, paper-comparable)")
    print("=" * 90)
    print(f"{'Setup':<70} | {'MAE':>6} | {'RMSE':>6} | {'MAPE%':>6} | {'Pearson':>7}")
    print("-" * 120)
    for lbl, m in results:
        print(f"{lbl:<70} | {m['MAE_bpm']:>5.3f}  | {m['RMSE_bpm']:>5.3f}  | "
              f"{m['MAPE_pct']:>5.3f}  | {m['Pearson']:>7.4f}")


if __name__ == '__main__':
    main()
