import sys
import os
import torch
import unittest

# 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.spiking_physformer import SpikingBiPhysformer

class TestSpikingModel(unittest.TestCase):
    def test_forward_pass(self):
        # Spiking 모델 입력: Batch 2, Channel 3, Height 224, Width 224
        x = torch.randn(2, 3, 224, 224)
        
        # 모델 인스턴스화
        model = SpikingBiPhysformer(
            img_size=224, 
            patch_size=16, 
            in_chans=3, 
            num_classes=2, 
            embed_dim=128, 
            depth=2, 
            num_heads=4, 
            T=4
        )
        
        print("\n모델 파라미터 개수:", sum(p.numel() for p in model.parameters()))
        
        # Forward pass 실행
        out = model(x)
        
        # 출력 텐서 모양 검증. [Batch, Classes] -> [2, 2]
        self.assertEqual(out.shape, (2, 2))
        print("순전파(Forward Pass) 및 텐서 차원(Shape) 검증 성공. Output shape:", out.shape)

if __name__ == '__main__':
    unittest.main()
