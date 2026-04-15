"""
Metrics 및 결과 기록 유틸리티 (교차 검증 및 벤치마크 대응).
"""
import os
import torch
import numpy as np

def calculate_metrics(pred, gt):
    pred = pred.detach().cpu().numpy().flatten()
    gt = gt.detach().cpu().numpy().flatten()
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt)**2))
    pearson = np.corrcoef(pred, gt)[0, 1] if len(gt) > 1 else 0
    return {'MAE': mae, 'RMSE': rmse, 'Pearson': pearson}

def update_experiment_summary(dataset_pair, metrics):
    """결과 레포트(Markdown 테이블) 갱신"""
    os.makedirs('results', exist_ok=True)
    summary_file = 'results/experiment_summary.md'
    comp_file = 'results/cross_dataset_comparison.md'
    
    # 1. 일반 요약 업데이트
    row = f"| {dataset_pair} | 완료 | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | {metrics['Pearson']:.4f} | RTX 4060 학습 |\n"
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Spiking Bi-Physformer 실험 결과\n\n| 데이터셋 | 상태 | MAE | RMSE | Pearson r | 비고 |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(row)
    
    # 2. 벤치마크 비교표 업데이트 (기존 파일이 있으면 해당 행을 문자열 치환)
    if os.path.exists(comp_file):
        with open(comp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Train->Test 형태 분석
        train_ds, test_ds = dataset_pair.split('->')
        # 테이블 내 (대기 중) 자리 찾기
        target_str = f"| **{train_ds}** | **{test_ds}** | **Spiking Bi-Physformer** | (대기 중) | (대기 중) | (대기 중) |"
        new_row = f"| **{train_ds}** | **{test_ds}** | **Spiking Bi-Physformer** | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {metrics['Pearson']:.2f} |"
        
        if target_str in content:
            new_content = content.replace(target_str, new_row)
            with open(comp_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
