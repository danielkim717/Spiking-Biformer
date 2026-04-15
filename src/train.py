import os
import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
from src.model.spiking_physformer import SpikingBiPhysformer
from src.data.datasets import get_dataloaders

def train_model(data_dir, epochs=5, batch_size=4, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"[*] Training을 시작합니다. 장치: {device}")
    
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    model = SpikingBiPhysformer(num_classes=2, embed_dim=128, depth=2, num_heads=4, T=4)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # SNN 뉴런의 Membrane Potential 리셋 (가장 중요)
            functional.reset_net(model)
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}")
        
    print("[*] 학습 완료!")
    return model

if __name__ == '__main__':
    # 테스트 구동
    train_model(data_dir='./data')
