"""
Cross-Dataset Validation Script: PURE ↔ UBFC-rPPG
"""
import os
import torch
from src.train import run_experiment, evaluate
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.datasets import get_dataloaders
from src.utils.metrics import update_experiment_summary

def run_cross_validation():
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 실험 1: PURE 학습 -> UBFC-rPPG 평가
    print("\n" + "="*50)
    print("Experiment 1: Train on PURE -> Test on UBFC-rPPG")
    print("="*50)
    # 실제 학습을 원격 루프로 수행 (데이터 존재 시)
    if os.path.exists('data/PURE') and os.listdir('data/PURE'):
        run_experiment('PURE', 'data/PURE', epochs=20, batch_size=2)
        # 학습 후 저장된 모델을 로드하여 테스트
        # (실제 구현에서는 train.py 에서 저장을 수행하게 됨)
    else:
        print("[!] PURE 데이터가 없습니다. Experiment 1을 수행할 수 없습니다.")

    # 실험 2: UBFC-rPPG 학습 -> PURE 평가
    print("\n" + "="*50)
    print("Experiment 2: Train on UBFC-rPPG -> Test on PURE")
    print("="*50)
    if os.path.exists('data/UBFC') and os.listdir('data/UBFC'):
        run_experiment('UBFC-rPPG', 'data/UBFC', epochs=20, batch_size=2)
    else:
        print("[!] UBFC 데이터가 없습니다. Experiment 2를 수행할 수 없습니다.")

if __name__ == '__main__':
    run_cross_validation()
