# 파일 관리 및 수정 계획
# scaffold_safety_ai 프로젝트의 기존 파일 처리 방법

# ============================================================================
# 1. 기존 파일 백업 및 수정
# ============================================================================

# 현재 위치로 이동
cd scaffold_safety_ai

# 기존 파일 백업
echo "📦 Backing up existing files..."
cp src/integrate_shapellm.py src/integrate_shapellm_backup.py
echo "✅ Backup created: src/integrate_shapellm_backup.py"

# ============================================================================
# 2. 수정된 integrate_shapellm.py 생성
# ============================================================================

cat > src/integrate_shapellm_fixed.py << 'EOF'
# src/integrate_shapellm_fixed.py
# 수정된 ShapeLLM 통합 - 기존 오류 해결

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
            
            # 필요한 디렉토리 확인
            required_dirs = ['llava', 'llava/model', 'llava/serve']
            for dir_name in required_dirs:
                dir_path = self.shapellm_path / dir_name
                if not dir_path.exists():
                    print(f"⚠️ Warning: {dir_name} not found at {self.shapellm_path}")
            
            print(f"✅ ShapeLLM environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False

class ScaffoldSafetyWrapper(nn.Module):
    """
    기존 코드 문제점을 해결한 Scaffold Safety 래퍼
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
        
        # PointLoRA 구성 요소들 (기존 pointlora_core.py 사용)
        self.safety_selector = SafetyTokenSelector(
            feature_dim=self.config['feature_dim'],
            safety_token_count=self.config['safety_token_count']
        )
        
        # Safety classifier 추가
        self.safety_classifier = nn.Sequential(
            nn.LayerNorm(self.config['feature_dim']),
            nn.Linear(self.config['feature_dim'], self.config['feature_dim'] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config['feature_dim'] // 2, 3)  # safe, warning, danger
        )
        
        # LoRA layers storage
        self.lora_layers = nn.ModuleDict()
        
        # Mock feature extractor (실제로는 ShapeLLM 사용)
        self.mock_feature_extractor = self._create_mock_feature_extractor()
        
        print(f"✅ ScaffoldSafetyWrapper initialized")
        self._print_parameter_stats()
    
    def _create_mock_feature_extractor(self):
        """Mock feature extractor for testing"""
        return nn.Sequential(
            nn.Linear(6, 256),  # xyz+rgb input
            nn.ReLU(),
            nn.Linear(256, self.config['feature_dim']),
            nn.LayerNorm(self.config['feature_dim'])
        )
    
    def add_lora_layer(self, layer_name: str, in_features: int, out_features: int):
        """동적으로 LoRA 레이어 추가"""
        lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=self.config['lora_rank'],
            alpha=self.config['lora_alpha']
        )
        self.lora_layers[layer_name] = lora_layer
        print(f"✅ LoRA layer added: {layer_name} ({lora_layer.get_param_count():,} params)")
    
    def forward_safety_analysis(self, point_cloud: torch.Tensor):
        """Safety analysis forward pass"""
        batch_size, num_points, point_dim = point_cloud.shape
        
        # 1. Feature extraction (Mock)
        # 실제로는 ShapeLLM의 ReCon++ encoder 사용
        if num_points > 512:
            # Simple downsampling for mock
            indices = torch.randperm(num_points)[:512]
            sampled_points = point_cloud[:, indices, :]
        else:
            sampled_points = point_cloud
        
        # Mock feature extraction
        features = self.mock_feature_extractor(sampled_points)  # [batch, 512, 768]
        
        # 2. Safety token selection
        safety_tokens, safety_indices = self.safety_selector(features)
        
        # 3. Safety classification
        avg_safety_features = safety_tokens.mean(dim=1)
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
    
    def set_training_mode(self, scaffold_training: bool = True):
        """LoRA와 Safety 모듈만 훈련 모드로 설정"""
        # 전체 모델 freeze
        for param in self.parameters():
            param.requires_grad = False
        
        if scaffold_training:
            # LoRA와 Safety 모듈만 학습 가능
            for name, param in self.named_parameters():
                if any(keyword in name for keyword in ['lora', 'safety']):
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"🎯 Training mode set:")
        print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def _print_parameter_stats(self):
        """파라미터 통계 출력"""
        total_params = sum(p.numel() for p in self.parameters())
        lora_params = sum(p.numel() for name, p in self.named_parameters() if 'lora' in name)
        safety_params = sum(p.numel() for name, p in self.named_parameters() if 'safety' in name)
        
        print(f"📊 Parameter Statistics:")
        print(f"   Total: {total_params:,}")
        print(f"   LoRA: {lora_params:,}")
        print(f"   Safety: {safety_params:,}")
        print(f"   Trainable: {lora_params + safety_params:,}")

def create_scaffold_model(config: dict = None):
    """Scaffold Safety 모델 생성 함수"""
    return ScaffoldSafetyWrapper(config)

# ============================================================================
# 테스트 함수들
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
    
    # 5. Forward pass 테스트
    test_point_cloud = torch.randn(2, 8192, 6)  # batch=2, points=8192, xyz+rgb
    
    with torch.no_grad():
        results = model.forward_safety_analysis(test_point_cloud)
    
    print(f"✅ Test Results:")
    print(f"   Safety tokens: {results['safety_tokens'].shape}")
    print(f"   Predicted classes: {results['predicted_class']}")
    print(f"   Confidence: {results['confidence']}")
    
    return model, results

if __name__ == "__main__":
    # 수정된 통합 테스트 실행
    model, results = test_fixed_integration()
    print("✅ Fixed integration test completed!")
EOF

# ============================================================================
# 3. 새 파일들 생성
# ============================================================================

# 테스트 디렉토리 생성
mkdir -p tests

# 테스트 파일 생성
cat > tests/test_integration.py << 'EOF'
# tests/test_integration.py
# 통합 테스트

import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integrate_shapellm_fixed import test_fixed_integration, create_scaffold_model
from src.pointlora_core import test_pointlora_components

def test_all_components():
    """모든 구성 요소 테스트"""
    print("🧪 Running Complete Integration Tests")
    print("=" * 50)
    
    # 1. 기본 PointLoRA 구성 요소 테스트
    print("\n📋 Step 1: Testing PointLoRA components...")
    try:
        test_pointlora_components()
        print("✅ PointLoRA components test passed")
    except Exception as e:
        print(f"❌ PointLoRA test failed: {e}")
        return False
    
    # 2. 수정된 통합 테스트
    print("\n📋 Step 2: Testing fixed integration...")
    try:
        model, results = test_fixed_integration()
        print("✅ Fixed integration test passed")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    # 3. A안→B안 연결 테스트
    print("\n📋 Step 3: Testing Stage A→B connection...")
    try:
        safety_tokens = results['safety_tokens']
        print(f"   Safety tokens ready for Stage B: {safety_tokens.shape}")
        print("✅ Stage A→B connection ready")
    except Exception as e:
        print(f"❌ Stage connection test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Ready for training.")
    return True

if __name__ == "__main__":
    success = test_all_components()
    if success:
        print("\n🚀 Ready to proceed with Stage A training!")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
EOF

# 스크립트 디렉토리 생성
mkdir -p scripts

# 간단한 실행 스크립트 생성
cat > scripts/run_tests.py << 'EOF'
# scripts/run_tests.py
# 모든 테스트 실행

import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_integration import test_all_components

if __name__ == "__main__":
    print("🚀 Running all scaffold safety tests...")
    success = test_all_components()
    
    if success:
        print("\n✅ All systems ready!")
        print("Next steps:")
        print("1. Run Stage A training: python scripts/train_stage_a.py")
        print("2. Prepare real scaffold data")
        print("3. Integrate with actual ShapeLLM")
    else:
        print("\n❌ Tests failed. Please fix issues first.")
EOF

# ============================================================================
# 4. 실행 가이드
# ============================================================================

echo ""
echo "📁 File Management Complete!"
echo "============================="
echo ""
echo "📋 Current file structure:"
echo "scaffold_safety_ai/"
echo "├── src/"
echo "│   ├── pointlora_core.py           # ✅ 유지 (기존)"
echo "│   ├── integrate_shapellm.py       # ⚠️ 원본 (문제 있음)"
echo "│   ├── integrate_shapellm_backup.py # 📦 백업"
echo "│   └── integrate_shapellm_fixed.py  # 🆕 수정된 버전"
echo "├── tests/"
echo "│   └── test_integration.py         # 🆕 통합 테스트"
echo "└── scripts/"
echo "    └── run_tests.py                # 🆕 테스트 실행기"
echo ""
echo "🎯 Next actions:"
echo "1. Run tests: python scripts/run_tests.py"
echo "2. If tests pass, replace original file:"
echo "   mv src/integrate_shapellm_fixed.py src/integrate_shapellm.py"
echo "3. Continue with Stage A training preparation"
echo ""
echo "Ready to test the fixed integration!"