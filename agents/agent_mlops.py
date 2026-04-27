"""
MLOps Agent: 전체 프로젝트 파이프라인(데이터 확보, 학습, 배포)을 자율적으로 오케스트레이션합니다.
"""
import os
import subprocess
import time
from agents.agent_data import run_data_pipeline
from src.train import run_experiment

def git_commit_and_push(message):
    """결과물 자동 Git Push"""
    print(f"[*] Git Push 시도 중: {message}")
    try:
        subprocess.run(['git', 'add', '-A'], check=True)
        # 커밋할 내용이 없는 경우를 위해 에러 무시
        subprocess.run(['git', 'commit', '-m', message], stderr=subprocess.DEVNULL)
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        print("[+] Git Push 성공")
    except Exception as e:
        print(f"[-] Git Push 실패: {e}")

def run_mlops_cycle():
    print("=== Spiking Bi-Physformer MLOps Cycle 시작 ===")
    
    # 1. 데이터 확보 시도 (경로 확인)
    run_data_pipeline()
    
    # 2. Cross Dataset 실험 큐 (Queue) 정의
    experiments = [
        {'train': 'UBFC-rPPG', 'test': 'PURE'},
        {'train': 'PURE', 'test': 'UBFC-rPPG'}
    ]
    
    # 비교표 초기 생성
    subprocess.run(['python', 'scripts/compare_metrics.py'])

    for exp in experiments:
        print(f"\n[MLOps] Cross-dataset 학습 시작: {exp['train']} -> {exp['test']}")
        # 빠른 테스트를 위해 epochs를 작게 설정
        run_experiment(exp['train'], exp['test'], epochs=5, batch_size=2)
        
        # 실험 완료 후 비교표 갱신
        subprocess.run(['python', 'scripts/compare_metrics.py'])
        
        # 실험 완료 후 자동 푸시
        git_commit_and_push(f"chore: {exp['train']}->{exp['test']} cross dataset results updated")

    print("\n=== MLOps Cycle 종료. 주기적으로 재실행 대기. ===")

if __name__ == '__main__':
    run_mlops_cycle()
