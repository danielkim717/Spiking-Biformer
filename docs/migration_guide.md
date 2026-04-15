# 리눅스 서버 마이그레이션 가이드 (Linux Migration Guide)

본 가이드는 현재 Windows에서 진행 중인 Spiking Bi-Physformer 프로젝트를 리눅스 서버로 이전하고, 리눅스 환경의 Antigravity 에이전트가 이어서 작업을 수행할 수 있도록 돕습니다.

## 1. 코드 및 상태 이전 (GitHub 기반)
모든 설계도와 진행 상태를 GitHub에 백업해 두었습니다. 리눅스 서버에서 다음 명령을 수행하세요:

```bash
# 1. 저장소 클론
git clone https://github.com/danielkim717/Spiking-Biformer.git
cd Spiking-Biformer

# 2. 브랜치 및 최신 상태 확인
git pull origin main
```

## 2. 데이터 이전 (수동)
81GB의 데이터셋은 보안 및 용량 문제로 GitHub에 포함되지 않았습니다. Windows의 `data/` 폴더에 있는 ZIP 파일들을 리눅스 서버의 동일한 위치로 복사해 주세요:
- `data/PURE/PURE.zip`
- `data/UBFC/UBFC-rPPG.zip`

## 3. 환경 구축 (Linux Setup)
리눅스 서버 GPU 드라이버 및 CUDA 환경이 갖춰져 있다면, 아래 명령으로 의존성을 설치합니다:

```bash
# 가상환경 추천
python3 -m venv venv
source venv/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
# OpenCV Headless (리눅스 서버용)
pip install opencv-python-headless
```

## 4. Antigravity 재개 방법
리눅스 서버에서 Antigravity를 실행한 후, 저에게 다음 문장을 입력해 주시면 즉시 맥락을 파악하고 학습을 재개합니다:

> **"Spiking-Biformer 프로젝트를 리눅스에서 이어서 진행해줘. GitHub 저장소와 docs/ 폴더의 계획서를 참고해서 현재 어디까지 진행되었는지 파악하고, benchmark_runner.py를 실행해."**

## 5. 지속적 보고
이전 후에도 `agent_reporter.py`가 30분마다 GitHub로 결과를 쏘아 올릴 것이므로, 사용자님은 어디서든 학습 상황을 확인하실 수 있습니다.
