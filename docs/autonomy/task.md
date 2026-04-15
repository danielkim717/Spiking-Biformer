# Spiking Bi-Physformer 프로젝트 작업 목록

## 1. 멀티 에이전트 및 환경 구축
- [x] 에이전트 구조 정의 및 디렉토리 생성
- [x] Python 3.11 및 필수 라이브러리(CUDA 지원) 설치
- [x] GitHub 저장소 (`Spiking-Biformer`) 연동 및 초기 코드 푸시

## 2. Spiking Bi-Physformer 모델 고도화
- [x] Spiking Biformer (LIF + Bi-level Attention) 구현
- [x] Physformer 기반 rPPG 신호 추출 아키텍처 확정
- [x] 모델 단위 테스트 통과 (Forward/Backward)

## 3. 데이터셋 확보 및 벤치마크 준비
- [/] `data/PURE.zip`, `data/UBFC.zip` 파일 배치 대기 중 (Job1 감시 중)
- [x] 기존 모델(Physformer) 교차 검증 Baseline 수치 조사 완료
- [ ] MMPD/UBFC-Phys 데이터셋 신청 가이드 보완

## 4. 교차 데이터셋 검증 및 비교 분석 (PURE ↔ UBFC)
- [x] `scripts/benchmark_runner.py` 고도화 및 백그라운드 가동
- [ ] Experiment A: UBFC 학습 -> PURE 테스트 수행 대기
- [ ] Experiment B: PURE 학습 -> UBFC 테스트 수행 대기
- [x] `results/cross_dataset_comparison.md` 비교 분석 리포트 틀 마련
- [x] RTX 4060 Mixed Precision(FP16) 및 SNN 최적화 로직 적용 완료

## 5. 결과 보고 및 GitHub 동기화
- [ ] `results/experiment_summary.md` 결과 표 작성 자동화
- [ ] 에피소드별 자동 Git Commit & Push (MLOps Agent)
- [ ] 최종 결과물 및 보고서(Artifact) 업데이트
