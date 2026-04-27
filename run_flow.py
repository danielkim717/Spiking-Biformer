import multiprocessing
import time
from agents.agent_data import run_data_pipeline
from agents.agent_architect import run_architect
from agents.agent_coder import run_model_build
from agents.agent_mlops import run_mlops_cycle
from agents.agent_reporter import main as run_reporter

def start_mlops():
    run_mlops_cycle()

def start_reporter():
    run_reporter()

def main():
    print("=== Spiking Bi-Physformer 멀티 에이전트 워크플로우 시작 ===\n")
    
    print("[1/3] Architect & Coder Agent 실행 중...")
    run_architect()
    run_model_build()
    
    print("\n[2/3] MLOps & Reporter Agent 동시 시작 (백그라운드)...")
    
    p_reporter = multiprocessing.Process(target=start_reporter)
    p_mlops = multiprocessing.Process(target=start_mlops)
    
    p_reporter.start()
    p_mlops.start()
    
    p_mlops.join()  # MLOps가 실험을 다 끝낼 때까지 대기
    print("\n[3/3] 모든 워크플로우 임무 완료! 30분 보고 시스템은 수동으로 종료 가능합니다.")
    
    # 리포터는 데몬처럼 돌거나 여기서 터미네이트
    p_reporter.terminate()
    print("=== Spiking-Biformer 오케스트레이션 완전 종료 ===")

if __name__ == '__main__':
    # 멀티프로세싱 호환을 위해 세팅
    multiprocessing.freeze_support()
    main()
