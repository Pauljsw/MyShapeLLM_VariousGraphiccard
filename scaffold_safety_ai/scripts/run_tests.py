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
