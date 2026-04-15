"""
Spiking Bi-Physformer 학습 및 평가 메인 스크립트.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.datasets import get_dataloaders
from src.utils.metrics import calculate_metrics, update_experiment_summary

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        # [B, T, C, H, W] -> Model -> [B, T]
        outputs = model(inputs)
        
        # SNN 상태 초기화 (필수)
        functional.reset_net(model)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            functional.reset_net(model)
            
            all_preds.append(outputs)
            all_gts.append(labels)
            
    metrics = calculate_metrics(torch.cat(all_preds), torch.cat(all_gts))
    return metrics

def run_experiment(dataset_name, root_dir, epochs=10, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Experiment: {dataset_name} on {device}")
    
    model = SpikingBiPhysformer(img_size=128, patch_size=8, embed_dim=128).to(device)
    train_loader = get_dataloaders(root_dir, dataset_name, batch_size=batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        if len(train_loader) == 0:
            print(f"[!] {dataset_name} 데이터가 비어 있습니다. Mock 모드로 전환하거나 건너뜁니다.")
            break
            
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
    # 최종 결과 기록 (데이터가 있을 때만)
    if len(train_loader) > 0:
        metrics = evaluate(model, train_loader, device) # 데모용으로 train_loader 사용
        update_experiment_summary(dataset_name, metrics)
        print(f"[+] {dataset_name} 리포트 업데이트 완료.")

if __name__ == '__main__':
    # 예시: PURE 데이터셋 실험 실행 (데이터 확보 후 활성화)
    # run_experiment('PURE', 'data/PURE')
    pass
