"""PURE/UBFC 데이터셋의 BVP 와 video frame 시간축 정합성 검증."""
import os
import json
import glob


def check_pure(root='D:/PURE'):
    print("=" * 60)
    print("PURE alignment check")
    print("=" * 60)
    subjects = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])[:3]
    for subj in subjects:
        json_path = os.path.join(root, subj, f"{subj}.json")
        img_dir = os.path.join(root, subj, subj)
        if not (os.path.exists(json_path) and os.path.exists(img_dir)):
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        bvp_n = len(data.get('/FullPackage', []))
        img_n = len(glob.glob(os.path.join(img_dir, "*.png")))
        # Try timestamps if present
        if bvp_n > 0:
            t0 = data['/FullPackage'][0].get('Timestamp', None)
            t1 = data['/FullPackage'][-1].get('Timestamp', None)
            duration = (t1 - t0) / 1e9 if (t0 and t1) else None  # PURE timestamps in nanoseconds
        else:
            duration = None
        if img_n > 0:
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            # Image timestamps from filenames (PURE filenames are timestamps in ns)
            try:
                fname0 = os.path.basename(img_files[0]).replace('Image', '').replace('.png', '')
                fname_last = os.path.basename(img_files[-1]).replace('Image', '').replace('.png', '')
                ft0, ft1 = int(fname0), int(fname_last)
                img_duration = (ft1 - ft0) / 1e9
            except Exception:
                img_duration = None
        else:
            img_duration = None

        bvp_rate = bvp_n / duration if duration else None
        img_rate = img_n / img_duration if img_duration else None

        print(f"  {subj}:")
        print(f"    BVP samples: {bvp_n}   duration: {duration}s   rate: {bvp_rate:.2f} Hz" if bvp_rate else f"    BVP samples: {bvp_n}")
        print(f"    Image count: {img_n}   duration: {img_duration}s   rate: {img_rate:.2f} fps" if img_rate else f"    Image count: {img_n}")
        print(f"    BVP/Img ratio: {bvp_n/img_n:.3f}")


def check_ubfc(root='D:/UBFC-rPPG'):
    print()
    print("=" * 60)
    print("UBFC-rPPG alignment check")
    print("=" * 60)
    base = os.path.join(root, 'DATASET_2')
    if not os.path.exists(base):
        base = root
    subjects = sorted([d for d in os.listdir(base) if d.startswith('subject')])[:3]
    for subj in subjects:
        subj_dir = os.path.join(base, subj)
        gt_path = os.path.join(subj_dir, 'ground_truth.txt')
        frames_dir = os.path.join(subj_dir, 'frames_extracted')
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        bvp_n = len(lines[0].strip().split()) if lines else 0
        img_n = len(glob.glob(os.path.join(frames_dir, "*.png"))) if os.path.exists(frames_dir) else 0
        ratio = bvp_n / img_n if img_n else None
        print(f"  {subj}: BVP samples={bvp_n}, frames={img_n}, ratio={ratio:.3f}" if ratio else f"  {subj}: BVP={bvp_n}, frames={img_n}")
        # gt_path line[1] = HR, line[2] = timestamp
        if len(lines) >= 3:
            hr_n = len(lines[1].strip().split())
            ts_n = len(lines[2].strip().split())
            print(f"    line0(BVP)={bvp_n}  line1(HR)={hr_n}  line2(ts)={ts_n}")


if __name__ == '__main__':
    check_pure()
    check_ubfc()
