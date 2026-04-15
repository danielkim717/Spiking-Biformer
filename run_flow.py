import argparse
from agents.agent_data import run_data_pipeline
from agents.agent_architect import run_architect
from agents.agent_coder import run_model_build
from agents.agent_mlops import run_mlops_pipeline

def main():
    print("=== Spiking Bi-Physformer 멀티 에이전트 워크플로우 시작 ===\n")
    
    print("[1/4] Data Agent 실행 중...")
    run_data_pipeline()
    
    print("\n[2/4] Architect Agent 실행 중...")
    run_architect()
    
    print("\n[3/4] Coder Agent 실행 중...")
    run_model_build()
    
    print("\n[4/4] MLOps Agent 실행 중...")
    run_mlops_pipeline()
    
    print("\n=== 모든 워크플로우 임무 완료 ===")

if __name__ == '__main__':
    main()
