import sys
import os

# 모듈 인식을 위해 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.git_manager import auto_commit_and_push

def run_mlops_pipeline():
    print("[MLOps Agent] 워크플로우 최종 상태 및 결과 로그를 형상 관리 시스템에 추가합니다.")
    
    # 깃 커밋 수행
    auto_commit_and_push("SNN Multi-Agent 워크플로우 진행 상태 자동 기록")
