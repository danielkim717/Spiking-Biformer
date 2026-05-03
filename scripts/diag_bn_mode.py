"""
같은 입력에 대해 train mode vs eval mode 의 발화율 차이를 직접 측정.
BN running stats 미수렴이 spike 0% 의 원인인지 확정.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from spikingjelly.activation_based import functional
from src.models.phys_biformer import PhysBiformer


def get_firing(model):
    rates = list(getattr(model, 'last_firing_rates', []))
    return [float(r) for r in rates]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    model = PhysBiformer(frame=160, patches=(4, 4, 4), v_threshold=1.0).to(device)
    x = torch.randn(2, 3, 160, 128, 128, device=device)

    print("=" * 60)
    print("초기 모델 (가중치 학습 안 됨)")
    print("=" * 60)
    model.train()
    _ = model(x); functional.reset_net(model)
    fr_train = get_firing(model)
    print(f"  train mode FR: {[f'{r*100:.2f}%' for r in fr_train]}  avg {np.mean(fr_train)*100:.2f}%")

    model.eval()
    with torch.no_grad():
        _ = model(x); functional.reset_net(model)
    fr_eval = get_firing(model)
    print(f"  eval  mode FR: {[f'{r*100:.2f}%' for r in fr_eval]}  avg {np.mean(fr_eval)*100:.2f}%")

    print()
    print("=" * 60)
    print("훈련 흉내내기: 같은 batch 100회 forward (running stats 갱신)")
    print("=" * 60)
    model.train()
    for _ in range(100):
        _ = model(x)
        functional.reset_net(model)

    model.train()
    _ = model(x); functional.reset_net(model)
    fr_train = get_firing(model)
    print(f"  train mode FR: {[f'{r*100:.2f}%' for r in fr_train]}  avg {np.mean(fr_train)*100:.2f}%")

    model.eval()
    with torch.no_grad():
        _ = model(x); functional.reset_net(model)
    fr_eval = get_firing(model)
    print(f"  eval  mode FR: {[f'{r*100:.2f}%' for r in fr_eval]}  avg {np.mean(fr_eval)*100:.2f}%")

    print()
    # 더 많이
    print("=" * 60)
    print("더 흉내내기: 추가 200회 forward")
    print("=" * 60)
    model.train()
    for _ in range(200):
        _ = model(x)
        functional.reset_net(model)

    model.train()
    _ = model(x); functional.reset_net(model)
    fr_train = get_firing(model)
    print(f"  train mode FR: {[f'{r*100:.2f}%' for r in fr_train]}  avg {np.mean(fr_train)*100:.2f}%")

    model.eval()
    with torch.no_grad():
        _ = model(x); functional.reset_net(model)
    fr_eval = get_firing(model)
    print(f"  eval  mode FR: {[f'{r*100:.2f}%' for r in fr_eval]}  avg {np.mean(fr_eval)*100:.2f}%")


if __name__ == '__main__':
    main()
