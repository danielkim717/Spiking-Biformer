import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class RPPGDataset(Dataset):
    def __init__(self, dataset_name, root_dir, clip_len=160, img_size=128):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.img_size = img_size
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True)
        ])
        self._prepare_data()

    def _prepare_data(self):
        print(f"[Dataset] 준비 중: {self.dataset_name} at {self.root_dir}")
        if self.dataset_name == 'PURE':
            # PURE 형식: 01-01 등의 폴더
            subjects = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            for subj in subjects:
                json_path = os.path.join(self.root_dir, subj, f"{subj}.json")
                img_dir = os.path.join(self.root_dir, subj, subj)
                if os.path.exists(json_path) and os.path.exists(img_dir):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    bvp_data = [item['Value']['waveform'] for item in data['/FullPackage']]
                    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                    
                    # 샘플 갯수 맞추기
                    min_len = min(len(bvp_data), len(img_files))
                    bvp_data = bvp_data[:min_len]
                    img_files = img_files[:min_len]
                    
                    for i in range(0, min_len - self.clip_len, self.clip_len):
                        self.samples.append({
                            'img_paths': img_files[i:i+self.clip_len],
                            'bvp': bvp_data[i:i+self.clip_len]
                        })
                        
        elif self.dataset_name == 'UBFC-rPPG':
            # UBFC-rPPG DATASET_2 형식
            base_dir = os.path.join(self.root_dir, 'DATASET_2')
            if not os.path.exists(base_dir):
                base_dir = self.root_dir # fallback
                
            subjects = [d for d in os.listdir(base_dir) if d.startswith('subject')]
            for subj in subjects:
                subj_dir = os.path.join(base_dir, subj)
                vid_path = os.path.join(subj_dir, 'vid.avi')
                gt_path = os.path.join(subj_dir, 'ground_truth.txt')
                
                if os.path.exists(vid_path) and os.path.exists(gt_path):
                    # 병목 해결: 1회 프레임 추출 (Pre-extraction)
                    frames_dir = os.path.join(subj_dir, 'frames_extracted')
                    if not os.path.exists(frames_dir):
                        print(f"[*] Extracting frames for {subj} to solve data loading bottleneck...")
                        os.makedirs(frames_dir, exist_ok=True)
                        cap = cv2.VideoCapture(vid_path)
                        idx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.png"), frame)
                            idx += 1
                        cap.release()
                        
                    with open(gt_path, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 0:
                        bvp_data = [float(x) for x in lines[0].strip().split()]
                        
                        # 프레임 개수는 폴더 내 png 개수로 산정
                        frame_count = len(glob.glob(os.path.join(frames_dir, "*.png")))
                        
                        min_len = min(len(bvp_data), frame_count)
                        for i in range(0, min_len - self.clip_len, self.clip_len):
                            self.samples.append({
                                'frames_dir': frames_dir,
                                'start_idx': i,
                                'bvp': bvp_data[i:i+self.clip_len]
                            })
                            
        print(f"[Dataset] {self.dataset_name} 샘플 생성 완료: 총 {len(self.samples)} 클립")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bvp = torch.tensor(sample['bvp'], dtype=torch.float32)
        
        frames = []
        if self.dataset_name == 'PURE':
            for img_path in sample['img_paths']:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(img))
        elif self.dataset_name == 'UBFC-rPPG':
            start_idx = sample['start_idx']
            frames_dir = sample['frames_dir']
            for i in range(self.clip_len):
                img_path = os.path.join(frames_dir, f"{start_idx + i:05d}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(self.transform(img))
                else:
                    break
            
            # 부족한 프레임 패딩 (드문 경우)
            while len(frames) < self.clip_len:
                frames.append(frames[-1] if len(frames) > 0 else torch.zeros((3, self.img_size, self.img_size)))
        
        frames = torch.stack(frames) # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3) # (C, T, H, W) 모델에 맞게
        
        # BVP Normalize
        bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-7)
        
        return frames, bvp

def get_dataloader(dataset_name, root_dir, batch_size=2, clip_len=30, img_size=128):
    dataset = RPPGDataset(dataset_name, root_dir, clip_len=clip_len, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
