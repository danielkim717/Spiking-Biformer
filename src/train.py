"""
학습 로그 및 상태 상세 기록을 포함한 Train 스크립트.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from spikingjelly.activation_based import functional
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.zip_dataset import get_zip_dataloader
from src.utils.metrics import calculate_metrics, update_experiment_summary

def save_status(status):
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

def run_experiment(train_ds, test_ds, epochs=30, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Experiment: {train_ds} -> {test_ds} on {device}")
    
    os.makedirs('results', exist_ok=True)
    
    zip_paths = {
        'PURE': 'data/PURE/PURE.zip',
        'UBFC-rPPG': 'data/UBFC/UBFC-rPPG.zip'
    }
    
    # 데이터 로더 준비
    print(f"[*] Loading Train Dataset: {train_ds}...")
    train_loader = get_zip_dataloader(zip_paths[train_ds], train_ds, batch_size)
    print(f"[*] Loading Test Dataset: {test_ds}...")
    test_loader = get_zip_dataloader(zip_paths[test_ds], test_ds, batch_size)

    if len(train_loader) == 0:
        print(f"[!] {train_ds} 샘플을 찾지 못했습니다. 경로와 ZIP 구조를 확인하세요.")
        return

    model = SpikingBiPhysformer(img_size=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            functional.reset_net(model)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.6f}")
                save_status({
                    'experiment': f"{train_ds}->{test_ds}",
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'step': i,
                    'total_steps': len(train_loader),
                    'loss': loss.item()
                })
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ds', type=str, required=True)
    parser.add_argument('--test_ds', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    run_experiment(args.train_ds, args.test_ds, args.epochs)
