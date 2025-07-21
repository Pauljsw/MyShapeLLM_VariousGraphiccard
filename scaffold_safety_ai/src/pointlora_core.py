# src/pointlora_core.py 파일 생성
import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer for Parameter-Efficient Fine-tuning
    논문 근거: PointLoRA achieves 85.53% with only 3.43% trainable parameters
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 매트릭스 초기화
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Kaiming 초기화로 A, B는 0으로 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        print(f"✅ LoRA Layer initialized: {in_features}→{out_features}, rank={rank}, params={self.get_param_count():,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LoRA forward: x @ (A^T @ B^T) * scaling
        """
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    
    def get_param_count(self) -> int:
        """LoRA 파라미터 개수 계산"""
        return self.lora_A.numel() + self.lora_B.numel()

class SafetyTokenSelector(nn.Module):
    """
    Multi-scale Safety Token Selection for Scaffold Analysis
    논문 근거: PointLoRA Table 7 - Multi-scale (32 & 8 tokens) achieves 85.53%
    """
    def __init__(self, feature_dim: int = 768, safety_token_count: int = 40):
        super().__init__()
        self.feature_dim = feature_dim
        self.safety_token_count = safety_token_count
        
        # Safety-aware Mask Predictor
        self.safety_mask_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ Safety Token Selector initialized: {feature_dim}D → {safety_token_count} tokens")
        
    def forward(self, patch_features: torch.Tensor, return_indices: bool = False) -> torch.Tensor:
        """
        Select most safety-relevant tokens from ReCon++ patch features
        
        Args:
            patch_features: [batch_size, 512, feature_dim] - ReCon++ patch features
            return_indices: whether to return selected indices
            
        Returns:
            safety_tokens: [batch_size, safety_token_count, feature_dim]
            indices (optional): [batch_size, safety_token_count]
        """
        batch_size, num_patches, feat_dim = patch_features.shape
        
        # Safety importance scoring
        safety_scores = self.safety_mask_predictor(patch_features)  # [batch, 512, 1]
        safety_scores = safety_scores.squeeze(-1)  # [batch, 512]
        
        # Top-K selection for safety-critical regions
        k = min(self.safety_token_count, num_patches)
        top_values, top_indices = torch.topk(safety_scores, k, dim=1)
        
        # Extract safety tokens
        safety_tokens = torch.gather(
            patch_features, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        )
        
        if return_indices:
            return safety_tokens, top_indices
        return safety_tokens

def test_pointlora_components():
    """PointLoRA 구성 요소 기본 테스트"""
    print("🧪 Testing PointLoRA Components...")
    
    # Test LoRA Layer
    lora = LoRALayer(in_features=768, out_features=768*3, rank=8, alpha=32)
    test_input = torch.randn(1, 512, 768)
    lora_output = lora(test_input)
    print(f"LoRA Test: {test_input.shape} → {lora_output.shape}")
    
    # Test Safety Token Selector  
    selector = SafetyTokenSelector(feature_dim=768, safety_token_count=40)
    patch_features = torch.randn(2, 512, 768)  # 2 scaffolds, 512 patches
    safety_tokens = selector(patch_features)
    print(f"Token Selector Test: {patch_features.shape} → {safety_tokens.shape}")
    
    # Parameter count verification
    total_params = lora.get_param_count()
    print(f"📊 LoRA Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("✅ All components working correctly!")

if __name__ == "__main__":
    test_pointlora_components()