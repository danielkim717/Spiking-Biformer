import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.drive_loader import download_and_extract_from_drive

def run_data_pipeline():
    print("[Data Agent] 구글 드라이브 연동 상태 검사 및 PyTorch DataLoader 파이프라인을 준비합니다.")
    
    pure_id = '139wOE6vTKHETJaMbaAVAanhMC3c88ov9'
    ubfc_id = '1df-n-QmIuNJDDMb5UcZNmp_b-A-6s9OJ'
    
    # 두 데이터셋 다운로드
    print("\n[Data Agent] PURE 데이터셋 다운로드를 시도합니다...")
    download_and_extract_from_drive(pure_id, dest_folder='./data/PURE')
    
    print("\n[Data Agent] UBFC-rPPG 데이터셋 다운로드를 시도합니다...")
    download_and_extract_from_drive(ubfc_id, dest_folder='./data/UBFC')

if __name__ == "__main__":
    run_data_pipeline()
