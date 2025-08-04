# LayerNorm 차원 오류 수정
# scaffold_safety_ai/src/integrate_shapellm_fixed.py 의 SafetyTokenSelector 부분 수정

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# PointLoRA 핵심 모듈 import (기존 파일 사용)
from .pointlora_core import LoRALayer, SafetyTokenSelector

class SafeShapeLLMIntegration:
    """
    안전한 ShapeLLM 통합 방식
    기존 코드의 import 오류와 구조적 문제 해결
    """
    
    def __init__(self, shapellm_path: str = None):
        self.shapellm_path = Path(shapellm_path) if shapellm_path else Path.cwd()
        self.setup_environment()
        
    def setup_environment(self):
        """ShapeLLM 환경 안전하게 설정"""
        try:
            # ShapeLLM 경로를 Python path에 추가
            if str(self.shapellm_path) not in sys.path:
                sys.path.insert(0, str(self.shapellm_path))
            
            # 실제 ShapeLLM 디렉토리 찾기
            shapellm_dirs = [
                self.shapellm_path.parent,  # 상위 디렉토리 확인
                self.shapellm_path.parent.parent,  # 더 상위 확인
                Path("/home/aimgroup/ChoSW/mcp-server-demo/ShapeLLM")  # 실제 경로
            ]
            
            for potential_dir in shapellm_dirs:
                if (potential_dir / "llava").exists():
                    self.shapellm_path = potential_dir
                    if str(potential_dir) not in sys.path:
                        sys.path.insert(0, str(potential_dir))
                    print(f"✅ Found ShapeLLM at: {potential_dir}")
                    break
            else:
                print(f"⚠️ ShapeLLM llava directory not found, using mock mode")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False

class FixedSafetyTokenSelector(nn.Module):
    """
    수정된 Safety Token Selector - LayerNorm 오류 해결
    """
    
    def __init__(self, feature_dim: int = 768, safety_token_count: int = 40):
        super().__init__()
        self.feature_dim = feature_dim
        self.safety_token_count = safety_token_count
        
        # 수정된 importance predictor - LayerNorm 위치 변경
        self.importance_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),  # 입력 차원에 맞게 수정
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ FixedSafetyTokenSelector initialized: {feature_dim}D → {safety_token_count} tokens")
        
    def forward(self, features: torch.Tensor):
        """
        Safety token selection with fixed dimensions
        
        Args:
            features: [batch_size, seq_len, feature_dim]
            
        Returns:
            safety_tokens: [batch_size, safety_token_count, feature_dim]
            selected_indices: [batch_size, safety_token_count]
        """
        batch_size, seq_len, feat_dim = features.shape
        
        # Importance scoring with correct dimensions
        scores = self.importance_network(features).squeeze(-1)  # [B, S]
        
        # Top-K selection
        k = min(self.safety_token_count, seq_len)
        _, indices = torch.topk(scores, k, dim=1)
        
        # Extract tokens
        safety_tokens = torch.gather(
            features, 1, 
            indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        )
        
        return safety_tokens, indices

class ScaffoldSafetyWrapper(nn.Module):
    """
    수정된 Scaffold Safety 래퍼 - LayerNorm 오류 해결
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        # 기본 설정
        self.config = config or {
            'lora_rank': 16,
            'lora_alpha': 32,
            'safety_token_count': 40,
            'feature_dim': 768
        }
        
        # 수정된 PointLoRA 구성 요소들
        self.safety_selector = FixedSafetyTokenSelector(  # 수정된 버전 사용
            feature_dim=self.config['feature_dim'],
            safety_token_count=self.config['safety_token_count']
        )
        
        # 수정된 Safety classifier - LayerNorm 위치 조정
        self.safety_classifier = nn.Sequential(
            nn.Linear(self.config['feature_dim'], self.config['feature_dim'] // 2),
            nn.LayerNorm(self.config['feature_dim'] // 2),  # 차원 맞춤
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config['feature_dim'] // 2, 3)  # safe, warning, danger
        )
        
        # LoRA layers storage
        self.lora_layers = nn.ModuleDict()
        
        # 수정된 Mock feature extractor
        self.mock_feature_extractor = self._create_mock_feature_extractor()
        
        print(f"✅ ScaffoldSafetyWrapper initialized")
        self._print_parameter_stats()
    
    def _create_mock_feature_extractor(self):
        """수정된 Mock feature extractor"""
        return nn.Sequential(
            nn.Linear(6, 256),  # xyz+rgb input
            nn.LayerNorm(256),  # 올바른 차원
            nn.ReLU(),
            nn.Linear(256, self.config['feature_dim']),
            nn.LayerNorm(self.config['feature_dim'])  # 올바른 차원
        )
    
    # src/integrate_shapellm.py 의 add_lora_layer 함수 수정
    def add_lora_layer(self, layer_name: str, in_features: int, out_features: int):
        lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=self.config['lora_rank'],
            alpha=self.config['lora_alpha']
        )
        # 핵심: nn.ModuleDict에 등록해야 parameters()에 포함됨
        self.lora_layers[layer_name] = lora_layer
        
        # 추가: 명시적으로 parameter 등록 확인
        print(f"✅ LoRA layer added: {layer_name}")
        print(f"   Parameters registered: {any(p.requires_grad for p in lora_layer.parameters())}")
    
    def forward_safety_analysis(self, point_cloud: torch.Tensor):
        """수정된 Safety analysis forward pass"""
        batch_size, num_points, point_dim = point_cloud.shape
        
        # 1. 안전한 포인트 처리
        if num_points > 512:
            # Simple downsampling for mock
            indices = torch.randperm(num_points)[:512]
            sampled_points = point_cloud[:, indices, :]
        else:
            sampled_points = point_cloud
            # Padding if necessary
            if sampled_points.shape[1] < 512:
                padding_size = 512 - sampled_points.shape[1] 
                padding = torch.zeros(batch_size, padding_size, point_dim)
                sampled_points = torch.cat([sampled_points, padding], dim=1)
        
        # 2. Mock feature extraction with correct dimensions
        features = self.mock_feature_extractor(sampled_points)  # [batch, 512, 768]
        
        # 3. Safety token selection
        safety_tokens, safety_indices = self.safety_selector(features)
        
        # 4. Safety classification - 차원 확인
        avg_safety_features = safety_tokens.mean(dim=1)  # [batch, 768]
        
        # 차원 디버깅
        print(f"🔍 Debug: avg_safety_features shape: {avg_safety_features.shape}")
        
        safety_logits = self.safety_classifier(avg_safety_features)
        safety_probs = torch.softmax(safety_logits, dim=-1)
        
        return {
            'safety_tokens': safety_tokens,      # [batch, 40, 768] - A안→B안 연결용!
            'safety_indices': safety_indices,   # [batch, 40]
            'safety_logits': safety_logits,     # [batch, 3]
            'safety_probs': safety_probs,       # [batch, 3]
            'features': features,               # [batch, 512, 768]
            'predicted_class': torch.argmax(safety_probs, dim=-1),
            'confidence': torch.max(safety_probs, dim=-1)[0]
        }
    
    # src/integrate_shapellm.py 의 set_training_mode 함수 수정
    def set_training_mode(self, scaffold_training: bool = True):
        # 전체 모델 freeze
        for param in self.parameters():
            param.requires_grad = False
        
        if scaffold_training:
            # 명시적으로 LoRA 파라미터만 활성화
            for lora_layer in self.lora_layers.values():
                for param in lora_layer.parameters():
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"🎯 Training mode set:")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
        # src/integrate_shapellm.py 의 _print_parameter_stats 함수 수정
    def _print_parameter_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        
        # 수정: LoRA 파라미터 정확히 계산
        lora_params = 0
        for lora_layer in self.lora_layers.values():
            lora_params += sum(p.numel() for p in lora_layer.parameters())
        
        safety_params = sum(p.numel() for name, p in self.named_parameters() 
                        if 'safety' in name)
        
        print(f"📊 Parameter Statistics:")
        print(f"   Total: {total_params:,}")
        print(f"   LoRA: {lora_params:,}")
        print(f"   Safety: {safety_params:,}")
        print(f"   Trainable: {lora_params + safety_params:,}")

def create_scaffold_model(config: dict = None):
    """Scaffold Safety 모델 생성 함수"""
    return ScaffoldSafetyWrapper(config)

# ============================================================================
# 수정된 테스트 함수
# ============================================================================

def test_fixed_integration():
    """수정된 통합 코드 테스트"""
    print("🧪 Testing Fixed Integration...")
    
    # 1. 환경 설정 테스트
    integration = SafeShapeLLMIntegration()
    
    # 2. 모델 생성 테스트
    config = {
        'lora_rank': 16,
        'lora_alpha': 32,
        'safety_token_count': 40,
        'feature_dim': 768
    }
    
    model = create_scaffold_model(config)
    
    # 3. LoRA 레이어 추가 테스트
    model.add_lora_layer('attention_qkv', 768, 768*3)
    model.add_lora_layer('mlp_fc1', 768, 3072)
    
    # 4. 훈련 모드 설정
    model.set_training_mode(scaffold_training=True)
    
    # 5. Forward pass 테스트 - 차원 안전성 확보
    print("🚀 Testing forward pass with dimension safety...")
    test_point_cloud = torch.randn(2, 8192, 6)  # batch=2, points=8192, xyz+rgb
    
    try:
        with torch.no_grad():
            results = model.forward_safety_analysis(test_point_cloud)
        
        print(f"✅ Test Results:")
        print(f"   Safety tokens: {results['safety_tokens'].shape}")
        print(f"   Predicted classes: {results['predicted_class']}")
        print(f"   Confidence: {results['confidence']}")
        
        return model, results
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # 수정된 통합 테스트 실행
    model, results = test_fixed_integration()
    if model is not None:
        print("✅ Fixed integration test completed!")
    else:
        print("❌ Integration test failed!")