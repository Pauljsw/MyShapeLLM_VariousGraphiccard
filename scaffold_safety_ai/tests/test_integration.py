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
