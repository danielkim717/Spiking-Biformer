"""
Metrics 및 결과 기록 유틸리티.
"""
import os
import torch
import numpy as np

def calculate_metrics(pred, gt):
    """rPPG 성능지표 계산: MAE, RMSE, Pearson Correlation"""
    pred = pred.detach().cpu().numpy().flatten()
    gt = gt.detach().cpu().numpy().flatten()
    
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt)**2))
    pearson = np.corrcoef(pred, gt)[0, 1]
    
    return {'MAE': mae, 'RMSE': rmse, 'Pearson': pearson}

def update_experiment_summary(dataset_name, metrics, filename='results/experiment_summary.md'):
    """결과 레포트(Markdown 테이블) 갱신"""
    os.makedirs('results', exist_ok=True)
    
    header = "| 데이터셋 | 상태 | MAE | RMSE | Pearson r | 비고 |\n"
    separator = "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    row = f"| {dataset_name} | 완료 | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | {metrics['Pearson']:.4f} | RTX 4060 학습 |\n"
    
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Spiking Bi-Physformer 실험 결과 요약\n\n")
            f.write(header + separator + row)
    else:
        # 기존 테이블에 행 추가 또는 갱신 (단순 추가 방식)
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(row)
