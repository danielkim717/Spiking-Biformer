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
    
    # 1. 데이터 확보 시도
    run_data_pipeline()
    
    # 2. 실험 큐 (Queue) 정의
    experiments = [
        {'name': 'PURE', 'path': 'data/PURE'},
        {'name': 'UBFC-rPPG', 'path': 'data/UBFC'},
    ]
    
    for exp in experiments:
        if os.path.exists(exp['path']) and os.listdir(exp['path']):
            print(f"\n[MLOps] {exp['name']} 학습 데이터 확인됨. 실험 시작.")
            run_experiment(exp['name'], exp['path'])
            
            # 실험 완료 후 자동 푸시
            git_commit_and_push(f"chore: {exp['name']} experimental results and metrics updated")
        else:
            print(f"\n[MLOps] {exp['name']} 데이터가 아직 준비되지 않았습니다. 건너뜁니다.")

    print("\n=== MLOps Cycle 종료. 주기적으로 재실행 대기. ===")

if __name__ == '__main__':
    run_mlops_cycle()
