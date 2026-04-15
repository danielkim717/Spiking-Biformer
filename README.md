# Spiking Bi-Physformer

Multi-Agent 기반의 Spiking Biformer Physformer 구현체입니다.

## 아키텍처 개요
1. **Spiking Patch Embedding**: 이미지를 SNN 토큰으로 변환
2. **Spiking Bi-level Routing Attention (Biformer 기반)**: SpikingJelly의 LIF 뉴런을 이용하여 효율적인 글로벌-로컬 어텐션 연산
3. **Spiking MLP / Head**: Spiking 기반의 네트워크 출력 구조 

## 멀티 에이전트 구조
이 프로젝트는 다음과 같은 에이전트 워크플로우로 구성되어 있습니다.
- **Architect Agent** (`agents/agent_architect.py`)
- **Coder Agent** (`agents/agent_coder.py`)
- **Data Agent** (`agents/agent_data.py`)
- **MLOps Agent** (`agents/agent_mlops.py`)

## 실행 방법
```bash
python run_flow.py
```
