"""
PURE BVP timestamp <-> image timestamp 정합성 검증.

PURE 데이터셋 FullPackage 의 각 항목은 Timestamp(ns) + Value(waveform) 구조.
Image 파일명도 timestamp 기반 (예: Image2406577.png).

검증 목표:
1. BVP 첫 sample 의 timestamp 와 image 첫 frame 의 timestamp 가 같은지
2. ::2 다운샘플 후 i 번째 BVP timestamp 가 i 번째 image timestamp 와 매칭되는지
3. 매칭 오차가 얼마나 큰지
"""
import os
import json
import glob


def check_subject(root, subj):
    json_path = os.path.join(root, subj, f"{subj}.json")
    img_dir = os.path.join(root, subj, subj)
    if not (os.path.exists(json_path) and os.path.exists(img_dir)):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    pkg = data.get('/FullPackage', [])
    bvp_ts = [item.get('Timestamp') for item in pkg]   # ns
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    img_ts = []
    for f in img_files:
        base = os.path.basename(f).replace('Image', '').replace('.png', '')
        try:
            img_ts.append(int(base))
        except ValueError:
            pass

    if not bvp_ts or not img_ts:
        return None

    # 1. 시작 offset
    bvp_start = bvp_ts[0]
    img_start = img_ts[0]
    offset_ns = bvp_start - img_start
    offset_ms = offset_ns / 1e6

    # 2. 다운샘플 후 매칭
    bvp_ds = bvp_ts[::2]
    n = min(len(bvp_ds), len(img_ts))
    diffs = [(bvp_ds[i] - img_ts[i]) / 1e6 for i in range(n)]   # ms
    diffs_abs = [abs(d) for d in diffs]
    max_diff = max(diffs_abs)
    mean_diff = sum(diffs_abs) / len(diffs_abs)

    # 3. 다른 다운샘플 (1::2) 도 비교
    bvp_ds_odd = bvp_ts[1::2]
    n2 = min(len(bvp_ds_odd), len(img_ts))
    diffs_odd = [(bvp_ds_odd[i] - img_ts[i]) / 1e6 for i in range(n2)]
    diffs_odd_abs = [abs(d) for d in diffs_odd]
    max_diff_odd = max(diffs_odd_abs)
    mean_diff_odd = sum(diffs_odd_abs) / len(diffs_odd_abs)

    return {
        'subj': subj,
        'bvp_count': len(bvp_ts),
        'img_count': len(img_ts),
        'start_offset_ms': offset_ms,
        'even_ds_mean_diff_ms': mean_diff,
        'even_ds_max_diff_ms': max_diff,
        'odd_ds_mean_diff_ms': mean_diff_odd,
        'odd_ds_max_diff_ms': max_diff_odd,
    }


def main():
    root = 'D:/PURE'
    subjects = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])[:5]
    print(f"{'subj':<6}  {'BVP':>5}  {'IMG':>5}  {'offset(ms)':>10}  {'even mean':>10}  {'even max':>10}  {'odd mean':>10}  {'odd max':>10}")
    print('-' * 80)
    for subj in subjects:
        r = check_subject(root, subj)
        if r:
            print(f"{r['subj']:<6}  {r['bvp_count']:>5}  {r['img_count']:>5}  "
                  f"{r['start_offset_ms']:>10.2f}  "
                  f"{r['even_ds_mean_diff_ms']:>10.2f}  {r['even_ds_max_diff_ms']:>10.2f}  "
                  f"{r['odd_ds_mean_diff_ms']:>10.2f}  {r['odd_ds_max_diff_ms']:>10.2f}")


if __name__ == '__main__':
    main()
