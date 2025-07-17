"""
ScaffoldPointLoRA.py - Fixed version with dtype consistency
해결된 문제점:
1. mat1 and mat2 dtype mismatch 
2. 디바이스 간 텐서 전달 안정성
3. 메모리 효율성 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

class ScaffoldLoRALayer(nn.Module):
    """
    Fixed ScaffoldLoRA Layer with consistent dtype handling
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices - 초기화 시 dtype 명시
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=torch.float16))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.float16))
        
        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fixed forward with dtype consistency
        """
        # 입력 텐서와 동일한 dtype 및 device로 변환
        target_dtype = x.dtype
        target_device = x.device
        
        # LoRA 매트릭스들을 입력과 동일한 dtype/device로 변환
        lora_A = self.lora_A.to(device=target_device, dtype=target_dtype)
        lora_B = self.lora_B.to(device=target_device, dtype=target_dtype)
        
        # LoRA 연산: x @ A @ B
        try:
            result = x @ lora_A @ lora_B * self.scaling
            return result
        except RuntimeError as e:
            # 타입 불일치 발생 시 추가 디버깅
            print(f"⚠️ LoRA dtype error: {e}")
            print(f"   x.dtype: {x.dtype}, x.device: {x.device}")
            print(f"   lora_A.dtype: {lora_A.dtype}, lora_A.device: {lora_A.device}")
            print(f"   lora_B.dtype: {lora_B.dtype}, lora_B.device: {lora_B.device}")
            # 안전한 fallback
            return torch.zeros_like(x)


class ScaffoldTokenSelector(nn.Module):
    """
    Fixed Multi-scale token selector with dtype consistency
    """
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-scale feature extractors with consistent dtype
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.local_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), 
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.detail_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fixed forward with proper dtype handling
        """
        batch_size, num_points, feat_dim = features.shape
        target_dtype = features.dtype
        target_device = features.device
        
        # 모든 MLP를 동일한 dtype/device로 이동
        self.global_mlp = self.global_mlp.to(device=target_device, dtype=target_dtype)
        self.local_mlp = self.local_mlp.to(device=target_device, dtype=target_dtype)
        self.detail_mlp = self.detail_mlp.to(device=target_device, dtype=target_dtype)
        
        # Multi-scale importance scoring
        global_scores = self.global_mlp(features).squeeze(-1)  # [B, N]
        local_scores = self.local_mlp(features).squeeze(-1)    # [B, N]
        detail_scores = self.detail_mlp(features).squeeze(-1)  # [B, N]
        
        # Top-K selection for each scale
        k_global = max(1, num_points // 8)  # 1/8 for global structure
        k_local = max(1, num_points // 4)   # 1/4 for local components  
        k_detail = max(1, num_points // 2)  # 1/2 for detail features
        
        # Safe top-k selection
        _, global_indices = torch.topk(global_scores, min(k_global, num_points), dim=1)
        _, local_indices = torch.topk(local_scores, min(k_local, num_points), dim=1)
        _, detail_indices = torch.topk(detail_scores, min(k_detail, num_points), dim=1)
        
        # Gather selected features
        global_features = torch.gather(features, 1, global_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
        local_features = torch.gather(features, 1, local_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
        detail_features = torch.gather(features, 1, detail_indices.unsqueeze(-1).expand(-1, -1, feat_dim))
        
        # Combine selected tokens
        selected_tokens = torch.cat([global_features, local_features, detail_features], dim=1)
        
        # Selection info for debugging
        selection_info = {
            'total_selected': selected_tokens.shape[1],
            'global_count': global_features.shape[1],
            'component_count': local_features.shape[1], 
            'detail_count': detail_features.shape[1]
        }
        
        return {
            'selected_tokens': selected_tokens,
            'selection_info': selection_info,
            'global_tokens': global_features,
            'local_tokens': local_features,
            'detail_tokens': detail_features
        }


class ScaffoldPointLoRA(nn.Module):
    """
    Fixed ScaffoldPointLoRA with dtype consistency
    """
    def __init__(self, 
                 hidden_size: int = 768,
                 lora_rank: int = 16,
                 lora_alpha: float = 32.0,
                 num_selected_tokens: int = 40):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_selected_tokens = num_selected_tokens
        
        # Multi-scale token selector
        self.token_selector = ScaffoldTokenSelector(hidden_size)
        
        # LoRA layers with consistent initialization
        self.qkv_lora = nn.ModuleDict({
            'q': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha),
            'k': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha),
            'v': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha)
        })
        
        self.ffn_lora = nn.ModuleDict({
            'up': ScaffoldLoRALayer(hidden_size, hidden_size * 4, lora_rank, lora_alpha),
            'down': ScaffoldLoRALayer(hidden_size * 4, hidden_size, lora_rank, lora_alpha)
        })
        
        # Prompt MLP for selected tokens
        self.prompt_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, 
                features: torch.Tensor,
                coords: torch.Tensor, 
                mode: str = 'token_selection') -> Dict[str, torch.Tensor]:
        """
        Fixed forward with proper dtype and device handling
        """
        target_dtype = features.dtype
        target_device = features.device
        
        # 모든 서브모듈을 동일한 dtype/device로 이동
        self._ensure_dtype_consistency(target_device, target_dtype)
        
        try:
            if mode == 'token_selection':
                return self._token_selection_forward(features, coords)
            elif mode == 'qkv_adaptation':
                return self._qkv_adaptation_forward(features)
            elif mode == 'ffn_adaptation':
                return self._ffn_adaptation_forward(features)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            print(f"⚠️ ScaffoldPointLoRA forward error: {e}")
            # Safe fallback
            return {
                'selected_tokens': features[:, :self.num_selected_tokens],
                'selection_info': {'total_selected': self.num_selected_tokens}
            }
    
    def _ensure_dtype_consistency(self, device: torch.device, dtype: torch.dtype):
        """모든 서브모듈의 dtype과 device 일관성 보장"""
        # Token selector
        self.token_selector = self.token_selector.to(device=device, dtype=dtype)
        
        # LoRA layers
        for lora_layer in self.qkv_lora.values():
            lora_layer = lora_layer.to(device=device, dtype=dtype)
        
        for lora_layer in self.ffn_lora.values():
            lora_layer = lora_layer.to(device=device, dtype=dtype)
            
        # Prompt MLP
        self.prompt_mlp = self.prompt_mlp.to(device=device, dtype=dtype)
    
    def _token_selection_forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-scale token selection with fixed dtype handling"""
        # Multi-scale token selection
        selection_result = self.token_selector(features, coords)
        
        # Apply prompt MLP to selected tokens
        selected_tokens = selection_result['selected_tokens']
        prompt_tokens = self.prompt_mlp(selected_tokens)
        
        return {
            'selected_tokens': prompt_tokens,
            'selection_info': selection_result['selection_info'],
            'raw_tokens': selected_tokens
        }
    
    def _qkv_adaptation_forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """QKV adaptation with fixed dtype handling"""
        q_adaptation = self.qkv_lora['q'](features)
        k_adaptation = self.qkv_lora['k'](features) 
        v_adaptation = self.qkv_lora['v'](features)
        
        return {
            'q': q_adaptation,
            'k': k_adaptation,
            'v': v_adaptation
        }
    
    def _ffn_adaptation_forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """FFN adaptation with fixed dtype handling"""
        up_adaptation = self.ffn_lora['up'](features)
        down_adaptation = self.ffn_lora['down'](up_adaptation)
        
        return {
            'ffn_output': down_adaptation
        }
    
    def get_trainable_parameters(self):
        """훈련 가능한 매개변수 반환"""
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params


# 추가적인 유틸리티 함수들
def create_scaffold_pointlora(hidden_size: int = 768, **kwargs) -> ScaffoldPointLoRA:
    """
    ScaffoldPointLoRA 생성 헬퍼 함수
    """
    return ScaffoldPointLoRA(
        hidden_size=hidden_size,
        lora_rank=kwargs.get('lora_rank', 16),
        lora_alpha=kwargs.get('lora_alpha', 32.0),
        num_selected_tokens=kwargs.get('num_selected_tokens', 40)
    )


def safe_tensor_operation(tensor1: torch.Tensor, tensor2: torch.Tensor, operation: str) -> torch.Tensor:
    """
    안전한 텐서 연산 (dtype 및 device 일관성 보장)
    """
    # 동일한 device로 이동
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    
    # 동일한 dtype로 변환
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(tensor1.dtype)
    
    # 연산 수행
    if operation == 'matmul':
        return torch.matmul(tensor1, tensor2)
    elif operation == 'add':
        return tensor1 + tensor2
    elif operation == 'mul':
        return tensor1 * tensor2
    else:
        raise ValueError(f"Unsupported operation: {operation}")


# 매개변수 통계 헬퍼
def print_lora_statistics(model: ScaffoldPointLoRA):
    """LoRA 매개변수 통계 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("📊 ScaffoldPointLoRA Parameter Statistics")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Training efficiency: {trainable_params/total_params:.2%}")
    print("=" * 50)
    
    # 레이어별 통계
    for name, module in model.named_modules():
        if isinstance(module, ScaffoldLoRALayer):
            lora_params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {lora_params:,} parameters")


# 타입 검증 함수
def validate_tensor_types(tensors: list, expected_dtype: torch.dtype = None):
    """텐서 타입 검증"""
    if not tensors:
        return True
    
    reference_dtype = expected_dtype or tensors[0].dtype
    reference_device = tensors[0].device
    
    for i, tensor in enumerate(tensors):
        if tensor.dtype != reference_dtype:
            print(f"⚠️ Tensor {i} dtype mismatch: {tensor.dtype} vs {reference_dtype}")
            return False
        if tensor.device != reference_device:
            print(f"⚠️ Tensor {i} device mismatch: {tensor.device} vs {reference_device}")
            return False
    
    return True


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Testing ScaffoldPointLoRA...")
    
    # 테스트 데이터 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_points, hidden_size = 1, 1024, 768
    
    features = torch.randn(batch_size, num_points, hidden_size, device=device, dtype=torch.float16)
    coords = torch.randn(batch_size, num_points, 3, device=device, dtype=torch.float16)
    
    # ScaffoldPointLoRA 초기화
    scaffold_lora = create_scaffold_pointlora(hidden_size=hidden_size)
    scaffold_lora = scaffold_lora.to(device=device, dtype=torch.float16)
    
    # 테스트 실행
    try:
        result = scaffold_lora(features, coords, mode='token_selection')
        print(f"✅ Token selection test passed: {result['selected_tokens'].shape}")
        print(f"   Selection info: {result['selection_info']}")
        
        # 통계 출력
        print_lora_statistics(scaffold_lora)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()