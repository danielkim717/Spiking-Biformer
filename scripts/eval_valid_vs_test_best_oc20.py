"""For 7/1/2 OC20 results, eval both VALID-best and TEST-best epoch.

Metrics: per-subject MAE, MAPE, Pearson(HR) (paper-comparable)
"""
import os, sys, io, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from src.models.bipulseformer import ViT_BiPulseFormer
from src.data.rppg_dataset import get_dataloader
from src.evaluation import evaluate_per_subject


def eval_ckpt(ckpt, dataset, path, split_range, pure_mode='subject_exclusive'):
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
    return evaluate_per_subject(preds, gts, loader.dataset.samples,
                                fs=30, diff_flag=True, low_pass=0.75, high_pass=2.5)


def find_test_best(summary_path):
    with open(summary_path) as f:
        data = json.load(f)
    if isinstance(data, list): data = data[0]
    # Test-best by per-subj MAE
    test_maes = [(h['epoch'], h['test']['MAE_subj']) for h in data['history']]
    test_best_ep, _ = min(test_maes, key=lambda x: x[1])
    return data['best_epoch'], test_best_ep


def main():
    configs = [
        # (name, result_dir, dataset, path, split_range, pure_mode, ckpt_prefix)
        ('PURE 7/1/2 random OC20',
         'results/intra_pure_bipulseformer_712_oc20',
         'PURE', 'D:\\PURE', (0.8, 1.0),
         'subject_exclusive_random', 'PURE_epoch'),
        ('UBFC 7/1/2 OC20',
         'results/intra_ubfc_bipulseformer_712_oc20',
         'UBFC-rPPG', 'D:\\UBFC-rPPG', (0.8, 1.0),
         'subject_exclusive', 'UBFC-rPPG_epoch'),
    ]

    results = []
    for label, rdir, ds, path, sr, pm, prefix in configs:
        valid_best_ep, test_best_ep = find_test_best(os.path.join(rdir, 'summary.json'))
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"  VALID-best epoch: {valid_best_ep}")
        print(f"  TEST-best  epoch: {test_best_ep}")
        print('='*70)

        for tag, ep in [('VALID-best', valid_best_ep), ('TEST-best', test_best_ep)]:
            ckpt = f'{rdir}/checkpoints/{prefix}{ep}.pt'
            if not os.path.exists(ckpt):
                print(f"  [!] missing: {ckpt}"); continue
            m = eval_ckpt(ckpt, ds, path, sr, pure_mode=pm)
            print(f"\n  ----- {tag} (E{ep}) -----")
            print(f"  MAE         = {m['MAE_bpm']:.4f} BPM")
            print(f"  RMSE        = {m['RMSE_bpm']:.4f} BPM")
            print(f"  MAPE        = {m['MAPE_pct']:.4f} %")
            print(f"  Pearson(HR) = {m['Pearson']:.4f}")
            print(f"  signal_Pearson = {m['signal_Pearson_mean']:.4f}")
            results.append((label, tag, ep, m))

    print("\n" + "=" * 90)
    print(" FINAL SUMMARY — 7/1/2 OC20 (per-subject, paper-comparable)")
    print("=" * 90)
    print(f"{'Setup':<28} | {'sel':<11} | {'Ep':>3} | {'MAE':>6} | {'RMSE':>6} | {'MAPE%':>6} | {'Pearson':>7}")
    print("-" * 110)
    for lbl, tag, ep, m in results:
        print(f"{lbl:<28} | {tag:<11} | {ep:>3} | {m['MAE_bpm']:>5.3f}  | "
              f"{m['RMSE_bpm']:>5.3f}  | {m['MAPE_pct']:>5.3f}  | {m['Pearson']:>7.4f}")


if __name__ == '__main__':
    main()
