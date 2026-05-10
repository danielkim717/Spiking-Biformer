"""
rPPG-Toolbox 호환 cross-dataset 평가 — paper level (MAE 1.44, ρ 0.98) 재현용.

차이 (per-clip 평가 → per-subject 평가):
  - rPPG-Toolbox: 모든 clip 을 subject 별로 concat (5sec×11 = ~60sec) → 한 번에 HR 추정
  - 짧은 5sec 클립 vs 긴 60sec 시그널 → frequency resolution 10배 차이
  - DiffNormalized 신호는 cumsum + detrend + Butterworth 후처리 후 periodogram 으로 HR 추출

References:
  - https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/metrics.py
  - https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/post_process.py
"""
import numpy as np
import scipy
import scipy.signal
from scipy.sparse import spdiags


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value=100):
    """rPPG-Toolbox _detrend (Tarvainen 2002 smoothness prior).
    Sparse linear solve 로 최적화 (원본 dense O(N^3) → sparse O(N))."""
    from scipy.sparse import spdiags as _spdiags, eye as _seye
    from scipy.sparse.linalg import spsolve as _spsolve
    N = input_signal.shape[0]
    ones = np.ones(N)
    diags_data = np.array([ones, -2 * ones, ones])
    D = _spdiags(diags_data, np.array([0, 1, 2]), N - 2, N).tocsc()
    A = (_seye(N) + (lambda_value ** 2) * (D.T @ D)).tocsc()
    smooth = _spsolve(A, input_signal)
    return input_signal - smooth


def _calculate_fft_hr_periodogram(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    """rPPG-Toolbox _calculate_fft_hr (paper-level, low_pass=0.75, high_pass=2.5).
    next_power_of_2 nfft 로 frequency resolution 향상."""
    ppg_signal = np.expand_dims(np.asarray(ppg_signal, dtype=np.float64), 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_freq = np.take(f_ppg, fmask)
    mask_pxx = np.take(pxx_ppg, fmask)
    return float(np.take(mask_freq, np.argmax(mask_pxx, 0))[0] * 60)


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True,
                               low_pass=0.75, high_pass=2.5):
    """rPPG-Toolbox calculate_metric_per_video 그대로.

    predictions, labels: 1D numpy array (subject 의 모든 clip concatenated)
    diff_flag: True 면 DiffNormalized → cumsum 후 detrend.
    Returns: (gt_hr, pred_hr, signal_pearson)
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if diff_flag:
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # rPPG-Toolbox paper-level: butter order 1, [0.75, 2.5] Hz (paper recommended)
    # 원본 코드 default 는 [0.6, 3.3] Hz 이지만 NeurIPS 2023 toolbox paper 결과 재현엔 0.75/2.5 권장
    b, a = scipy.signal.butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))

    hr_pred = _calculate_fft_hr_periodogram(predictions, fs=fs, low_pass=low_pass, high_pass=high_pass)
    hr_label = _calculate_fft_hr_periodogram(labels, fs=fs, low_pass=low_pass, high_pass=high_pass)

    # Signal-level Pearson on filtered signals
    p = predictions - predictions.mean()
    g = labels - labels.mean()
    denom = (np.sqrt((p * p).sum()) * np.sqrt((g * g).sum())) + 1e-9
    sig_pearson = float((p * g).sum() / denom)

    return hr_label, hr_pred, sig_pearson


def _aggregate_overlapping(arrays, starts, lengths, total_length, mode='mean'):
    """Sliding window 으로 추출된 prediction chunks 를 video-length 신호로 합치기.
    arrays: list of 1D arrays (각 chunk prediction)
    starts: list of int (각 chunk의 first_frame_idx)
    lengths: list of int (각 chunk의 length, 보통 모두 동일)
    total_length: int (target signal length)
    mode: 'mean' → 겹치는 frame을 평균, 'last' → 마지막 chunk 값 사용
    """
    buf = np.zeros(total_length, dtype=np.float64)
    cnt = np.zeros(total_length, dtype=np.float64)
    for arr, s, L in zip(arrays, starts, lengths):
        end = min(s + L, total_length)
        actual_L = end - s
        if actual_L <= 0:
            continue
        buf[s:end] += arr[:actual_L]
        cnt[s:end] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return buf / cnt


def get_subject_signals(preds_array, gts_array, samples, fs=30, diff_flag=True,
                        low_pass=0.75, high_pass=2.5):
    """Subject 별 concat + 후처리된 신호 dict 반환.
    Returns: dict[video_id] -> (filtered_pred, filtered_label, hr_pred, hr_label)
    파형 시각화용."""
    by_subj = {}
    for i, samp in enumerate(samples):
        vid = samp['video_id']
        first = samp['first_frame_idx']
        by_subj.setdefault(vid, []).append((first, i))

    out = {}
    for vid, items in by_subj.items():
        items.sort(key=lambda x: x[0])
        idxs = [it[1] for it in items]
        starts = [it[0] for it in items]
        T_clip = preds_array.shape[1]
        last_start = max(starts)
        total_length = last_start + T_clip
        # Sliding window aggregation: 겹치는 부분 평균
        if len(starts) > 1 and (starts[1] - starts[0]) < T_clip:
            pred_concat = _aggregate_overlapping(
                [preds_array[i] for i in idxs], starts, [T_clip] * len(idxs), total_length)
            gt_concat = _aggregate_overlapping(
                [gts_array[i] for i in idxs], starts, [T_clip] * len(idxs), total_length)
        else:
            pred_concat = np.concatenate([preds_array[i] for i in idxs])
            gt_concat = np.concatenate([gts_array[i] for i in idxs])
        if pred_concat.shape[0] < 32:
            continue

        # Apply same post-processing as calculate_metric_per_video
        if diff_flag:
            pred_proc = _detrend(np.cumsum(pred_concat), 100)
            gt_proc = _detrend(np.cumsum(gt_concat), 100)
        else:
            pred_proc = _detrend(pred_concat, 100)
            gt_proc = _detrend(gt_concat, 100)

        b, a = scipy.signal.butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        pred_filt = scipy.signal.filtfilt(b, a, np.double(pred_proc))
        gt_filt = scipy.signal.filtfilt(b, a, np.double(gt_proc))

        hr_pred = _calculate_fft_hr_periodogram(pred_filt, fs=fs, low_pass=low_pass, high_pass=high_pass)
        hr_gt = _calculate_fft_hr_periodogram(gt_filt, fs=fs, low_pass=low_pass, high_pass=high_pass)

        out[vid] = {
            'pred_raw': pred_concat,        # raw model output (DiffNormalized)
            'gt_raw': gt_concat,            # raw label (DiffNormalized)
            'pred_filt': pred_filt,         # cumsum + detrend + bandpass
            'gt_filt': gt_filt,
            'hr_pred': hr_pred,
            'hr_gt': hr_gt,
        }
    return out


def evaluate_per_subject(preds_array, gts_array, samples, fs=30, diff_flag=True,
                         low_pass=0.75, high_pass=2.5):
    """rPPG-Toolbox 호환 per-subject 평가.

    preds_array, gts_array: (N_clips, T) numpy arrays — DataLoader 출력 순서대로
    samples: dataset.samples (len N_clips). 각 sample 에 'video_id', 'first_frame_idx' 필요.

    Returns dict:
      - MAE_bpm, RMSE_bpm, MAPE_pct: subject 별 HR error 평균
      - Pearson: subject 별 HR 의 Pearson (predicted vs gt)
      - signal_Pearson_mean: subject 의 filtered signal-level Pearson 평균
      - n_subjects: 평가에 사용된 subject 수
    """
    # Group by video_id, sort by first_frame_idx
    by_subj = {}
    for i, samp in enumerate(samples):
        vid = samp['video_id']
        first = samp['first_frame_idx']
        by_subj.setdefault(vid, []).append((first, i))

    pred_hrs, gt_hrs, sig_pearsons = [], [], []
    for vid, items in by_subj.items():
        items.sort(key=lambda x: x[0])
        idxs = [it[1] for it in items]
        starts = [it[0] for it in items]
        T_clip = preds_array.shape[1]
        last_start = max(starts)
        total_length = last_start + T_clip
        if len(starts) > 1 and (starts[1] - starts[0]) < T_clip:
            pred_concat = _aggregate_overlapping(
                [preds_array[i] for i in idxs], starts, [T_clip] * len(idxs), total_length)
            gt_concat = _aggregate_overlapping(
                [gts_array[i] for i in idxs], starts, [T_clip] * len(idxs), total_length)
        else:
            pred_concat = np.concatenate([preds_array[i] for i in idxs])
            gt_concat = np.concatenate([gts_array[i] for i in idxs])
        if pred_concat.shape[0] < 32:
            continue  # 너무 짧으면 skip
        try:
            gt_hr, pred_hr, sig_p = calculate_metric_per_video(
                pred_concat, gt_concat, fs=fs, diff_flag=diff_flag,
                low_pass=low_pass, high_pass=high_pass,
            )
            pred_hrs.append(pred_hr)
            gt_hrs.append(gt_hr)
            sig_pearsons.append(sig_p)
        except Exception as e:
            print(f"[evaluate_per_subject] skip {vid}: {e}")
            continue

    pred_hrs = np.array(pred_hrs)
    gt_hrs = np.array(gt_hrs)
    abs_err = np.abs(pred_hrs - gt_hrs)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))
    nz = gt_hrs != 0
    mape = float(np.mean(abs_err[nz] / gt_hrs[nz]) * 100.0) if nz.any() else 0.0

    if len(pred_hrs) >= 2 and pred_hrs.std() > 1e-9 and gt_hrs.std() > 1e-9:
        hr_pearson = float(np.corrcoef(pred_hrs, gt_hrs)[0, 1])
    else:
        hr_pearson = 0.0

    sig_p_mean = float(np.mean(sig_pearsons)) if sig_pearsons else 0.0

    return {
        'MAE_bpm': mae,
        'RMSE_bpm': rmse,
        'MAPE_pct': mape,
        'Pearson': hr_pearson,
        'signal_Pearson_mean': sig_p_mean,
        'n_subjects': len(pred_hrs),
    }
