"""
학습 상태를 추적하고 파일로 기록하는 기능을 추가한 Train 스크립트.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.datasets import get_dataloaders
from src.data.zip_dataset import get_zip_dataloader
from src.utils.metrics import calculate_metrics, update_experiment_summary

def save_status(status):
    """현재 학습 상태를 JSON 파일로 저장 (리포터용)"""
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

def run_experiment(train_ds, test_ds, epochs=30, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)
    
    model = SpikingBiPhysformer(img_size=128, patch_size=8, embed_dim=128).to(device)
    
    zip_paths = {'PURE': 'data/PURE/PURE.zip', 'UBFC-rPPG': 'data/UBFC/UBFC-rPPG.zip'}
    train_loader = get_zip_dataloader(zip_paths[train_ds], train_ds, batch_size) if os.path.exists(zip_paths[train_ds]) else []
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        if len(train_loader) == 0: return
        
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
            
            # 10 스텝마다 상태 저장
            if i % 10 == 0:
                save_status({
                    'experiment': f"{train_ds}->{test_ds}",
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'step': i,
                    'total_steps': len(train_loader),
                    'loss': loss.item()
                })
        
        print(f"Epoch {epoch+1}/{epochs} Completed: Loss {epoch_loss/len(train_loader):.4f}")
