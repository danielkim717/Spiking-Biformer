"""
Benchmark Runner: Zip 파일을 감지하고 교차 검증을 시작합니다.
"""
import os
import time
import subprocess

def check_data_ready():
    # 사용자가 업로드한 경로
    pure_zip = 'data/PURE/PURE.zip'
    ubfc_zip = 'data/UBFC/UBFC-rPPG.zip'
    return os.path.exists(pure_zip) and os.path.exists(ubfc_zip)

def main():
    print("=== Spiking Bi-Physformer Zip-Aware Runner ===")
    os.environ['PYTHONPATH'] = os.getcwd()

    while True:
        if check_data_ready():
            print("[!] ZIP 데이터 감지됨. 별도의 압축 해제 없이 즉시 학습 루프를 시작합니다.")
            
            # Phase 1: UBFC -> PURE
            subprocess.run(['py', '-3.11', 'src/train.py', '--train_ds', 'UBFC-rPPG', '--test_ds', 'PURE', '--epochs', '30'])
            
            # Phase 2: PURE -> UBFC
            subprocess.run(['py', '-3.11', 'src/train.py', '--train_ds', 'PURE', '--test_ds', 'UBFC-rPPG', '--epochs', '30'])
            
            break
        time.sleep(30)

if __name__ == '__main__':
    main()
