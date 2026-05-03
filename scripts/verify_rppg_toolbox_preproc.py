"""
rPPG-Toolbox 전처리 (HaarCascade face detection + 1.5x box + Standardized) 동작 확인.
- PURE 와 UBFC 각각 1 subject 만 로드해 face detection 결과 출력
- 검출된 box 영역을 frame_sample.png 로 저장해 시각 확인
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.rppg_dataset import RPPGDataset


OUT_DIR = 'results/preproc_check'
os.makedirs(OUT_DIR, exist_ok=True)


def check(dataset_name, root, subjects):
    print("=" * 60)
    print(f"[*] {dataset_name}  (subjects={subjects})")
    print("=" * 60)
    ds = RPPGDataset(dataset_name, root, clip_len=160, img_size=128,
                     face_crop=True, subjects_filter=subjects,
                     standardize_input=True)
    if len(ds) == 0:
        print("  no samples"); return
    print(f"  samples: {len(ds)}")
    print(f"  cached face boxes: {ds._face_box_cache}")

    frames, bvp = ds[0]
    print(f"  frames shape: {tuple(frames.shape)}  dtype: {frames.dtype}")
    print(f"  frames mean={float(frames.mean()):.4e}  std={float(frames.std()):.4e}  "
          f"min={float(frames.min()):.3f}  max={float(frames.max()):.3f}")
    print(f"  bvp shape: {tuple(bvp.shape)}  mean={float(bvp.mean()):.3e}  std={float(bvp.std()):.3e}")

    # save first/middle/last frame for visual inspection
    # frames: (C, T, H, W); standardize 된 상태이므로 보기 위해 다시 [0,1] 으로 rescale
    f = frames.permute(1, 2, 3, 0).numpy()        # (T, H, W, C)
    f = f - f.min(); f = f / (f.max() + 1e-7)
    snaps = [f[0], f[len(f) // 2], f[-1]]
    grid = np.concatenate(snaps, axis=1)          # H, 3W, C
    grid = (grid * 255).astype(np.uint8)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    out = os.path.join(OUT_DIR, f"{dataset_name}_sample.png")
    cv2.imwrite(out, grid)
    print(f"  saved snapshot: {out}")


if __name__ == '__main__':
    check('PURE', 'D:\\PURE', ['01-01'])
    check('UBFC-rPPG', 'D:\\UBFC-rPPG', ['subject1'])
    print("\n[*] 검증 완료.")
