import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class RPPGDataset(Dataset):
    """rPPG-Toolbox 스타일 전처리 적용 dataset.

    face_crop=True 일 때 rPPG-Toolbox 의 BaseLoader.face_detection 동작을 재현:
      - HaarCascade (frontalface_default) + LARGER_BOX_COEF (1.5)
      - dynamic_detection_freq>0 일 때 매 freq frame 마다 재검출 (rPPG-Toolbox DYNAMIC_DETECTION_FREQUENCY)
      - dynamic_detection_freq==0 또는 None 이면 첫 프레임만 검출하고 video 전체에 동일 box 사용
      - 검출 실패 시 직전 성공 box 또는 center-square fallback

    data_type 옵션 (rPPG-Toolbox `DATA_TYPE`/`LABEL_TYPE`):
      - 'standardized': 입력 / label 모두 z-score (mean 0, std 1)
      - 'diff_normalized': PhysFormer/Spiking-PhysFormer 가 사용하는 형식
            data:  (frame_{t+1} - frame_t) / (frame_{t+1} + frame_t + 1e-7), 전체 std 로 나눔
            label: np.diff(label) / std(diff)
        clip_len 출력을 유지하기 위해 raw 입력은 clip_len+1 프레임 사용.
    """

    def __init__(self, dataset_name, root_dir, clip_len=160, img_size=128,
                 face_crop=False, subjects_filter=None,
                 face_detection_backend='HC', larger_box_coef=1.5,
                 dynamic_detection_freq=0, data_type='standardized',
                 split_range=None,
                 # legacy aliases for backwards compatibility
                 dynamic_detection=None, standardize_input=None):
        """split_range: (begin, end) ∈ [0, 1] tuple. subject 정렬 후 해당 비율 구간만 사용.
        rPPG-Toolbox `BEGIN`/`END` 와 동일 — 0.0-0.8 train, 0.8-1.0 valid 표준.
        """
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.img_size = img_size
        self.face_crop = face_crop
        self.face_detection_backend = face_detection_backend
        self.larger_box_coef = larger_box_coef
        self.split_range = split_range
        # Backwards compat: standardize_input=True/False maps to data_type
        if standardize_input is False:
            data_type = 'raw'
        self.data_type = data_type
        # Backwards compat: dynamic_detection=True (legacy bool) -> 30
        if dynamic_detection is True and dynamic_detection_freq == 0:
            dynamic_detection_freq = 30
        self.dynamic_detection_freq = dynamic_detection_freq
        self.subjects_filter = set(subjects_filter) if subjects_filter else None
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True)
        ])
        # video_id -> dict[box_idx -> (x0,y0,x1,y1)]   (dynamic, box_idx = abs_frame // freq)
        # video_id -> (x0,y0,x1,y1)                    (static)
        self._face_box_cache = {}
        # video_id -> sorted list of all frame paths (full video)
        self._video_img_list = {}
        if self.face_crop and self.face_detection_backend == 'HC':
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError(f"HaarCascade load 실패: {cascade_path}")
        else:
            self._haar = None
        self._prepare_data()

    def _detect_face_box_raw(self, img):
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        found = len(faces) >= 1
        if found:
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            cx, cy = x + w // 2, y + h // 2
            new_size = int(max(w, h) * self.larger_box_coef)
        else:
            s = min(H, W)
            cx, cy = W // 2, H // 2
            new_size = s
        half = new_size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(W, x0 + new_size)
        y1 = min(H, y0 + new_size)
        side = min(x1 - x0, y1 - y0)
        x1 = x0 + side
        y1 = y0 + side
        return (int(x0), int(y0), int(x1), int(y1)), found

    def _crop_center_square(self, img):
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return img[y0:y0 + s, x0:x0 + s]

    def _get_box_for_frame(self, video_id, abs_frame_idx):
        if self.dynamic_detection_freq > 0:
            boxes = self._face_box_cache.get(video_id, {})
            box_idx = abs_frame_idx // self.dynamic_detection_freq
            if box_idx in boxes:
                return boxes[box_idx]
            keys = [k for k in boxes if k <= box_idx]
            if keys:
                return boxes[max(keys)]
            return None
        return self._face_box_cache.get(video_id, None)

    def _crop_face(self, img, video_id, abs_frame_idx):
        if not self.face_crop:
            return img
        box = self._get_box_for_frame(video_id, abs_frame_idx)
        if box is None:
            return self._crop_center_square(img)
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]

    def _prepare_data(self):
        print(f"[Dataset] 준비 중: {self.dataset_name} at {self.root_dir} "
              f"(data_type={self.data_type}, dyn_freq={self.dynamic_detection_freq})")

        # diff_normalized 는 raw 입력 N+1 → 출력 N 이므로 한 프레임 더 읽어둠.
        need_count = self.clip_len + (1 if self.data_type == 'diff_normalized' else 0)

        if self.dataset_name == 'PURE':
            subjects = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
            if self.subjects_filter is not None:
                subjects = [s for s in subjects if s in self.subjects_filter]
            if self.split_range is not None:
                n = len(subjects)
                s_idx = int(self.split_range[0] * n)
                e_idx = int(self.split_range[1] * n)
                subjects = subjects[s_idx:e_idx]
                print(f"[Dataset] split_range={self.split_range} → {len(subjects)} subjects "
                      f"({subjects[0] if subjects else '-'} ... {subjects[-1] if subjects else '-'})")
            for subj in subjects:
                json_path = os.path.join(self.root_dir, subj, f"{subj}.json")
                img_dir = os.path.join(self.root_dir, subj, subj)
                if os.path.exists(json_path) and os.path.exists(img_dir):
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    bvp_data = [item['Value']['waveform'] for item in data['/FullPackage']]
                    bvp_data = bvp_data[::2]  # 60→30Hz
                    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

                    min_len = min(len(bvp_data), len(img_files))
                    bvp_data = bvp_data[:min_len]
                    img_files = img_files[:min_len]
                    self._video_img_list[subj] = img_files

                    for i in range(0, min_len - need_count + 1, self.clip_len):
                        self.samples.append({
                            'video_id': subj,
                            'first_frame_idx': i,
                            'img_paths': img_files[i:i + need_count],
                            'bvp': bvp_data[i:i + need_count]
                        })

        elif self.dataset_name == 'UBFC-rPPG':
            base_dir = os.path.join(self.root_dir, 'DATASET_2')
            if not os.path.exists(base_dir):
                base_dir = self.root_dir

            subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('subject')],
                              key=lambda s: int(s.replace('subject', '')))
            if self.subjects_filter is not None:
                subjects = [s for s in subjects if s in self.subjects_filter]
            if self.split_range is not None:
                n = len(subjects)
                s_idx = int(self.split_range[0] * n)
                e_idx = int(self.split_range[1] * n)
                subjects = subjects[s_idx:e_idx]
                print(f"[Dataset] split_range={self.split_range} → {len(subjects)} subjects "
                      f"({subjects[0] if subjects else '-'} ... {subjects[-1] if subjects else '-'})")
            for subj in subjects:
                subj_dir = os.path.join(base_dir, subj)
                vid_path = os.path.join(subj_dir, 'vid.avi')
                gt_path = os.path.join(subj_dir, 'ground_truth.txt')

                if os.path.exists(vid_path) and os.path.exists(gt_path):
                    frames_dir = os.path.join(subj_dir, 'frames_extracted')
                    if not os.path.exists(frames_dir):
                        print(f"[*] Extracting frames for {subj} ...")
                        os.makedirs(frames_dir, exist_ok=True)
                        cap = cv2.VideoCapture(vid_path)
                        idx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.png"), frame)
                            idx += 1
                        cap.release()

                    with open(gt_path, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 0:
                        bvp_data = [float(x) for x in lines[0].strip().split()]
                        all_imgs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
                        self._video_img_list[subj] = all_imgs
                        frame_count = len(all_imgs)

                        min_len = min(len(bvp_data), frame_count)
                        for i in range(0, min_len - need_count + 1, self.clip_len):
                            self.samples.append({
                                'video_id': subj,
                                'frames_dir': frames_dir,
                                'first_frame_idx': i,
                                'start_idx': i,
                                'bvp': bvp_data[i:i + need_count]
                            })

        print(f"[Dataset] {self.dataset_name} 샘플 생성 완료: 총 {len(self.samples)} 클립")

        if self.face_crop and self.face_detection_backend == 'HC':
            self._predetect_faces()

    def _predetect_faces(self):
        n_videos = len(self._video_img_list)
        if self.dynamic_detection_freq > 0:
            total_det, total_fb = 0, 0
            for vid, imgs in self._video_img_list.items():
                boxes = {}
                last_box = None
                for box_idx, frame_idx in enumerate(range(0, len(imgs), self.dynamic_detection_freq)):
                    img = cv2.imread(imgs[frame_idx])
                    if img is None:
                        if last_box is not None:
                            boxes[box_idx] = last_box
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    box, found = self._detect_face_box_raw(img)
                    if found:
                        boxes[box_idx] = box
                        last_box = box
                        total_det += 1
                    else:
                        total_fb += 1
                        if last_box is not None:
                            boxes[box_idx] = last_box
                        else:
                            boxes[box_idx] = box  # center fallback from _detect_face_box_raw
                            last_box = box
                self._face_box_cache[vid] = boxes
            print(f"[Dataset] face detection (HC, 1.5x, dynamic@{self.dynamic_detection_freq}): "
                  f"{total_det} detected, {total_fb} HC-fallback over {n_videos} videos")
        else:
            n_det, n_fb = 0, 0
            for vid, imgs in self._video_img_list.items():
                if not imgs:
                    n_fb += 1
                    continue
                first_img = cv2.imread(imgs[0])
                if first_img is None:
                    n_fb += 1
                    continue
                first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
                box, found = self._detect_face_box_raw(first_img)
                self._face_box_cache[vid] = box
                if found:
                    n_det += 1
                else:
                    n_fb += 1
            print(f"[Dataset] face detection (HC, 1.5x, static): "
                  f"{n_det} detected, {n_fb} fallback over {n_videos} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bvp = torch.tensor(sample['bvp'], dtype=torch.float32)
        video_id = sample.get('video_id', None)
        first_idx = sample.get('first_frame_idx', 0)

        need_count = self.clip_len + (1 if self.data_type == 'diff_normalized' else 0)

        frames = []
        if self.dataset_name == 'PURE':
            for j, img_path in enumerate(sample['img_paths']):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.face_crop:
                    img = self._crop_face(img, video_id, first_idx + j)
                frames.append(self.transform(img))
        elif self.dataset_name == 'UBFC-rPPG':
            start_idx = sample['start_idx']
            frames_dir = sample['frames_dir']
            for j in range(need_count):
                img_path = os.path.join(frames_dir, f"{start_idx + j:05d}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.face_crop:
                        img = self._crop_face(img, video_id, first_idx + j)
                    frames.append(self.transform(img))
                else:
                    break
            while len(frames) < need_count:
                frames.append(frames[-1] if len(frames) > 0 else torch.zeros((3, self.img_size, self.img_size)))

        frames = torch.stack(frames)               # (T_raw, C, H, W)
        frames = frames.permute(1, 0, 2, 3)        # (C, T_raw, H, W)

        if self.data_type == 'standardized':
            mean = frames.mean()
            std = frames.std() + 1e-7
            frames = (frames - mean) / std
            frames = torch.nan_to_num(frames, nan=0.0)
            bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-7)
        elif self.data_type == 'diff_normalized':
            # frames: (C, T_raw=clip_len+1, H, W) → diff along T → (C, clip_len, H, W)
            f_next = frames[:, 1:]
            f_prev = frames[:, :-1]
            diff = (f_next - f_prev) / (f_next + f_prev + 1e-7)
            diff_std = diff.std() + 1e-7
            frames = diff / diff_std
            frames = torch.nan_to_num(frames, nan=0.0)
            # bvp: (T_raw=clip_len+1,) → diff → (clip_len,)
            bvp_diff = bvp[1:] - bvp[:-1]
            bvp_std = bvp_diff.std() + 1e-7
            bvp = bvp_diff / bvp_std
            bvp = torch.nan_to_num(bvp, nan=0.0)
        elif self.data_type == 'raw':
            pass

        return frames, bvp


def get_dataloader(dataset_name, root_dir, batch_size=2, clip_len=30, img_size=128,
                   face_crop=False, subjects_filter=None, shuffle=True,
                   face_detection_backend='HC', larger_box_coef=1.5,
                   dynamic_detection_freq=0, data_type='standardized',
                   split_range=None,
                   dynamic_detection=None, standardize_input=None,
                   drop_last=None):
    dataset = RPPGDataset(dataset_name, root_dir, clip_len=clip_len, img_size=img_size,
                          face_crop=face_crop, subjects_filter=subjects_filter,
                          face_detection_backend=face_detection_backend,
                          larger_box_coef=larger_box_coef,
                          dynamic_detection_freq=dynamic_detection_freq,
                          data_type=data_type,
                          split_range=split_range,
                          dynamic_detection=dynamic_detection,
                          standardize_input=standardize_input)
    # 마지막 incomplete batch 가 SNN 내부 BN 통계 (track_running_stats=False) 를
    # 불안정하게 만들 수 있어 train 에서는 drop_last=True 권장.
    if drop_last is None:
        drop_last = bool(shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=drop_last)
