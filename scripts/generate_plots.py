"""BiPhysFormer best checkpoint 로 scatter plot + Bland-Altman 생성.

Usage:
    python scripts/generate_plots.py --direction PURE_to_UBFC-rPPG --epoch 4
    python scripts/generate_plots.py --direction UBFC-rPPG_to_PURE --epoch 7
"""
import os
import sys
import io
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from src.models.biphysformer import ViT_BiPhysFormer
from src.data.rppg_dataset import get_dataloader
from src.evaluation import get_subject_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--test_dataset', type=str, required=True,
                        choices=['PURE', 'UBFC-rPPG'])
    parser.add_argument('--test_path', type=str, default=None,
                        help='Override test path (default: D:\\PURE or D:\\UBFC-rPPG)')
    parser.add_argument('--split_range', type=str, default=None,
                        help='e.g. "0.6,1.0" for intra-dataset test set; omit for cross-dataset')
    parser.add_argument('--name', type=str, required=True,
                        help='Plot filename prefix (e.g. intra_PURE_epoch9)')
    parser.add_argument('--out_dir', type=str, default='results/plots/biphysformer')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading {args.ckpt}')
    model = ViT_BiPhysFormer(
        patches=(4, 4, 4), dim=96, ff_dim=144, num_heads=4, num_layers=12,
        dropout_rate=0.1, theta=0.7, image_size=(160, 128, 128),
        n_win=(2, 2, 2), topk=4,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [info] {len(missing)} missing keys')
    if unexpected:
        print(f'  [warning] {len(unexpected)} unexpected keys ignored')
    model.eval()

    # Test dataset
    test_path = args.test_path
    if test_path is None:
        test_path = 'D:\\PURE' if args.test_dataset == 'PURE' else 'D:\\UBFC-rPPG'
    split_range = None
    if args.split_range:
        a, b = args.split_range.split(',')
        split_range = (float(a), float(b))
    print(f'Test: {args.test_dataset} at {test_path}, split={split_range}')
    test_loader = get_dataloader(
        args.test_dataset, test_path, batch_size=4, clip_len=160,
        face_crop=True, shuffle=False, data_type='diff_normalized',
        chunk_step=80, dynamic_detection_freq=0, num_workers=4, pin_memory=True,
        split_range=split_range,
    )
    print(f'  Test clips: {len(test_loader.dataset)}')

    # Run inference
    all_p, all_g = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            rPPG, _, _, _ = model(inputs, gra_sharp=2.0)
            rPPG = (rPPG - rPPG.mean(-1, keepdim=True)) / (rPPG.std(-1, keepdim=True) + 1e-8)
            all_p.append(rPPG.cpu()); all_g.append(labels)
    preds = torch.cat(all_p).numpy()
    gts = torch.cat(all_g).numpy()
    print(f'  preds: {preds.shape}, gts: {gts.shape}')

    # Per-subject HR extraction
    sigs = get_subject_signals(
        preds, gts, test_loader.dataset.samples,
        fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5,
    )

    pred_hrs = np.array([s['hr_pred'] for s in sigs.values()])
    gt_hrs = np.array([s['hr_gt'] for s in sigs.values()])
    subjects = list(sigs.keys())
    print(f'  n subjects: {len(subjects)}')
    print(f'  HR range: GT [{gt_hrs.min():.1f}, {gt_hrs.max():.1f}] BPM, '
          f'Pred [{pred_hrs.min():.1f}, {pred_hrs.max():.1f}] BPM')

    # Metrics
    abs_err = np.abs(pred_hrs - gt_hrs)
    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    mape = np.mean(abs_err / np.maximum(gt_hrs, 1e-9)) * 100
    pearson = np.corrcoef(pred_hrs, gt_hrs)[0, 1]
    print(f'\nMetrics: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, Pearson={pearson:.4f}')

    # 1. Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gt_hrs, pred_hrs, s=40, alpha=0.65, edgecolors='k', linewidths=0.5)
    lim_min = min(gt_hrs.min(), pred_hrs.min()) - 5
    lim_max = max(gt_hrs.max(), pred_hrs.max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='y=x (perfect)')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Ground Truth HR (BPM)', fontsize=12)
    ax.set_ylabel('Predicted HR (BPM)', fontsize=12)
    ax.set_title(f'BiPhysFormer — {args.name}\n'
                 f'MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, ρ={pearson:.3f}',
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    scatter_path = os.path.join(args.out_dir, f'scatter_{args.name}.png')
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    print(f'  Saved scatter: {scatter_path}')

    # 2. Bland-Altman plot
    mean_hrs = (gt_hrs + pred_hrs) / 2
    diff_hrs = pred_hrs - gt_hrs
    mean_diff = diff_hrs.mean()
    std_diff = diff_hrs.std(ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_hrs, diff_hrs, s=40, alpha=0.65, edgecolors='k', linewidths=0.5)
    ax.axhline(mean_diff, color='blue', linestyle='-', linewidth=1.5,
               label=f'Mean bias = {mean_diff:.2f} BPM')
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.2,
               label=f'+1.96 SD = {loa_upper:.2f} BPM')
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.2,
               label=f'−1.96 SD = {loa_lower:.2f} BPM')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('(GT + Pred) / 2 — Mean HR (BPM)', fontsize=12)
    ax.set_ylabel('Pred − GT — Difference (BPM)', fontsize=12)
    ax.set_title(f'Bland-Altman — {args.name}\n'
                 f'Bias = {mean_diff:.2f} BPM, LoA = [{loa_lower:.2f}, {loa_upper:.2f}] BPM',
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ba_path = os.path.join(args.out_dir, f'bland_altman_{args.name}.png')
    fig.tight_layout()
    fig.savefig(ba_path, dpi=120)
    plt.close(fig)
    print(f'  Saved Bland-Altman: {ba_path}')

    # 3. Save per-subject CSV for transparency
    csv_path = os.path.join(args.out_dir, f'per_subject_{args.name}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('subject,gt_hr,pred_hr,error\n')
        for vid, p, g in zip(subjects, pred_hrs, gt_hrs):
            f.write(f'{vid},{g:.3f},{p:.3f},{p - g:.3f}\n')
    print(f'  Saved CSV: {csv_path}')


if __name__ == '__main__':
    main()
