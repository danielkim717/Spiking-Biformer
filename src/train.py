"""
학습 로그 및 상태 상세 기록을 포함한 Train 스크립트.
에포크마다 평가를 수행하도록 개선되었습니다.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from src.models.phys_biformer import PhysBiformer
from src.data.rppg_dataset import get_dataloader
from src.utils.metrics import calculate_metrics, update_experiment_summary

def save_status(status):
    with open('results/current_status.json', 'w') as f:
        json.dump(status, f)

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, labels):
        preds_mean = torch.mean(preds, dim=1, keepdim=True)
        labels_mean = torch.mean(labels, dim=1, keepdim=True)
        
        preds_std = preds - preds_mean
        labels_std = labels - labels_mean
        
        cov = torch.sum(preds_std * labels_std, dim=1)
        var_preds = torch.sqrt(torch.sum(preds_std ** 2, dim=1) + 1e-8)
        var_labels = torch.sqrt(torch.sum(labels_std ** 2, dim=1) + 1e-8)
        
        pearson = cov / (var_preds * var_labels)
        return 1.0 - torch.mean(pearson)

class FrequencyLoss(nn.Module):
    def __init__(self, fps=30):
        super().__init__()
        self.fps = fps
        
    def forward(self, preds, labels):
        pred_fft = torch.fft.rfft(preds, dim=1)
        pred_psd = torch.abs(pred_fft) ** 2
        
        label_fft = torch.fft.rfft(labels, dim=1)
        label_psd = torch.abs(label_fft) ** 2
        
        freqs = torch.fft.rfftfreq(preds.shape[1], d=1.0/self.fps).to(preds.device)
        valid_idx = (freqs >= 0.66) & (freqs <= 3.0) # 40 ~ 180 BPM
        
        pred_psd = pred_psd[:, valid_idx]
        label_psd = label_psd[:, valid_idx]
        
        if pred_psd.shape[1] == 0:
            return torch.tensor(0.0).to(preds.device), torch.tensor(0.0).to(preds.device)
            
        pred_prob = F.softmax(pred_psd, dim=1)
        label_prob = F.softmax(label_psd, dim=1)
        
        target_class = torch.argmax(label_prob, dim=1)
        loss_ce = F.cross_entropy(pred_psd, target_class)
        loss_ld = F.kl_div(F.log_softmax(pred_psd, dim=1), label_prob, reduction='batchmean')
        
        return loss_ce, loss_ld

def run_experiment(train_ds, test_ds, epochs=30, batch_size=2, v_threshold=1.0, lr=3e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Experiment: {train_ds} -> {test_ds} on {device} (V_th={v_threshold}, LR={lr})")
    
    os.makedirs('results', exist_ok=True)
    
    dataset_paths = {
        'PURE': 'D:\\PURE',
        'UBFC-rPPG': 'D:\\UBFC-rPPG'
    }
    
    train_loader = get_dataloader(train_ds, dataset_paths[train_ds], batch_size, clip_len=160)
    test_loader = get_dataloader(test_ds, dataset_paths[test_ds], batch_size, clip_len=160)

    if len(train_loader) == 0 or len(test_loader) == 0:
        print(f"[!] {train_ds} 또는 {test_ds} 샘플을 찾지 못했습니다.")
        return

    model = PhysBiformer(frame=160, patches=(4, 4, 4), v_threshold=v_threshold).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    mse_criterion = nn.MSELoss()
    pearson_criterion = NegPearsonLoss()
    freq_criterion = FrequencyLoss(fps=30)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss_mse = mse_criterion(outputs, labels)
            loss_pearson = pearson_criterion(outputs, labels)
            loss_ce, loss_ld = freq_criterion(outputs, labels)
            
            # Weighted loss: MSE + Freq + Pearson (for better shape)
            loss = 0.5 * loss_mse + 0.5 * (loss_ce + loss_ld) + 0.1 * loss_pearson
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            functional.reset_net(model)
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.6f}")
                save_status({
                    'experiment': f"{train_ds}->{test_ds}",
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'step': i,
                    'total_steps': len(train_loader),
                    'loss': loss.item(),
                    'phase': 'Training',
                    'firing_rates': getattr(model, 'last_firing_rates', [])
                })
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.6f}")
        
        # Intermediate Evaluation every epoch
        print(f"[*] Starting Evaluation for Epoch {epoch+1}...")
        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                functional.reset_net(model)
                all_preds.append(outputs.cpu())
                all_gts.append(labels.cpu())
                
                if j % 10 == 0:
                    save_status({
                        'experiment': f"{train_ds}->{test_ds}",
                        'epoch': epoch + 1,
                        'total_epochs': epochs,
                        'step': j,
                        'total_steps': len(test_loader),
                        'loss': 0.0,
                        'phase': 'Evaluation'
                    })
                    
        all_preds = torch.cat(all_preds)
        all_gts = torch.cat(all_gts)
        metrics = calculate_metrics(all_preds, all_gts)
        print(f"[*] Epoch {epoch+1} Results: {metrics}")
        update_experiment_summary(f"{train_ds}->{test_ds}", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ds', type=str, required=True)
    parser.add_argument('--test_ds', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--v_threshold', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-3)
    args = parser.parse_args()
    run_experiment(args.train_ds, args.test_ds, args.epochs, v_threshold=args.v_threshold, lr=args.lr)
