"""Spiking Bi-Physformer 모델 단위 테스트"""
import sys
import os
import torch
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model.spiking_physformer import SpikingBiPhysformer
from spikingjelly.activation_based import functional


class TestSpikingBiPhysformer(unittest.TestCase):
    def test_forward_rppg(self):
        """rPPG 모드: [B, T, C, H, W] → [B, T] 출력 검증"""
        model = SpikingBiPhysformer(
            img_size=64, patch_size=8, in_chans=3,
            embed_dim=64, depth=2, num_heads=4, topk=4
        )
        # 비디오 클립: Batch 2, Time 16, Channel 3, H 64, W 64
        x = torch.randn(2, 16, 3, 64, 64)
        out = model(x)
        functional.reset_net(model)
        
        self.assertEqual(out.shape, (2, 16), f"출력 형태 불일치: {out.shape}")
        print(f"[PASS] Forward pass 성공. Output shape: {out.shape}")
        print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    def test_backward(self):
        """역전파(Backward) 검증"""
        model = SpikingBiPhysformer(
            img_size=64, patch_size=8, in_chans=3,
            embed_dim=64, depth=2, num_heads=4
        )
        x = torch.randn(1, 8, 3, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        functional.reset_net(model)
        
        # 그래디언트가 None이 아닌지 확인
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad, "역전파 실패: gradient가 없음")
        print(f"[PASS] Backward pass 성공. Loss: {loss.item():.4f}")


if __name__ == '__main__':
    unittest.main()
