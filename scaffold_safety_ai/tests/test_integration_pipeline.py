# scaffold_safety_ai/tests/test_integration_pipeline.py

import torch
import sys
from pathlib import Path

# 현재 파일의 경로를 기준으로 'scaffold_safety_ai' 디렉토리를 찾아 sys.path에 추가
project_root_for_src = Path(__file__).resolve().parent.parent
if str(project_root_for_src) not in sys.path:
    sys.path.insert(0, str(project_root_for_src))
    print(f"✅ 'scaffold_safety_ai' added to sys.path: {project_root_for_src}")

# ShapeLLM의 ReConV2 모듈을 찾기 위한 상위 경로도 추가
shapellm_root = Path(__file__).resolve().parents[3]
if str(shapellm_root) not in sys.path:
    sys.path.insert(0, str(shapellm_root))
    print(f"✅ 'ShapeLLM' root added to sys.path: {shapellm_root}")

from src.integrate_shapellm_gemini import PointLoRAReconEncoder

def main():
    """
    PointLoRA와 ReCon 모델의 통합 파이프라인을 테스트하는 함수
    """
    print("🚀 PointLoRA-Recon 통합 테스트를 시작합니다.")

    # ReCon 모델에 전달할 가상의 설정(config) 객체
    class ReconConfig:
        def __init__(self):
            # ShapeLLM 논문의 "large" 모델 파라미터와 유사하게 설정
            self.embed_dim = 1024
            self.num_group = 512
            self.group_size = 32
            self.with_color = True
            self.mask_type = 'causal'
            self.mask_ratio = 0.5
            self.stop_grad = False
            
            # --- 수정된 부분: global_query 차원 불일치 해결 ---
            # large.pth 모델은 global_query를 img 15개 + text 1개로 사용하는 것으로 추정
            self.img_queries = 15
            self.text_queries = 1
            
            self.depth = 24
            self.decoder_depth = 8
            self.num_heads = 16
            self.drop_path_rate = 0.1
            self.pretrained_model_name = ""
            
            # PointLoRA와 안전 토큰 선택을 위한 파라미터
            self.lora_rank = 16
            self.lora_alpha = 32
            self.safety_token_count = 40
            
            # --- 수정된 부분: contrast_type과 large_embedding 추가 ---
            self.contrast_type = "byol"
            self.large_embedding = False # ReCon.py의 PatchEmbedding에서 사용

    # 1. 모델 초기화
    try:
        model_config = ReconConfig()
        model = PointLoRAReconEncoder(model_config)
        print("✅ PointLoRAReconEncoder 모델이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"❌ 모델 초기화 중 오류 발생: {e}")
        return

    # 2. 사전 훈련된 가중치 로드
    checkpoint_path = "/home/aimgroup/ChoSW/mcp-server-demo/ShapeLLM/checkpoints/recon/large.pth"
    try:
        model.load_pretrained_weights(checkpoint_path)
        print("✅ 사전 훈련된 가중치 로드 성공.")
    except Exception as e:
        print(f"❌ 사전 훈련된 가중치 로드 중 오류 발생: {e}")
        return

    # 3. 훈련 모드 설정 (LoRA와 Safety Head만 활성화)
    try:
        # GPU 사용 가능 여부 확인 및 모델 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"✅ 모델이 {device}로 이동되었습니다.")

        model.set_training_mode(scaffold_training=True)
        print("✅ 훈련 모드 설정 완료. LoRA 및 Safety Head만 활성화되었습니다.")
    except Exception as e:
        print(f"❌ 훈련 모드 설정 중 오류 발생: {e}")
        return

    # 4. 더미 데이터 생성 및 순전파 테스트
    test_point_cloud = torch.randn(2, 8192, 6, device=device)  # (batch_size, num_points, xyz+rgb)
    print(f"✅ 테스트 데이터 생성 완료: {test_point_cloud.shape}")

    try:
        with torch.no_grad():
            results = model.forward_safety_analysis(test_point_cloud)
            
        # 결과 텐서의 형태(shape) 검증
        assert results['safety_tokens'].shape == (2, 40, model_config.embed_dim)
        assert results['predicted_class'].shape == (2,)
        assert results['confidence'].shape == (2,)

        print("--- 테스트 결과 ---")
        print(f"✅ 순전파(forward) 성공! 모든 모듈이 올바르게 연결되었습니다.")
        print(f"  - Safety Tokens Shape: {results['safety_tokens'].shape}")
        print(f"  - Predicted Classes: {results['predicted_class']}")
        print(f"  - Confidence Scores: {results['confidence']}")
        print("-------------------")
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")

    except AssertionError as e:
        print(f"❌ 결과 텐서 형태가 예상과 다릅니다: {e}")
    except Exception as e:
        print(f"❌ 순전파 실행 중 오류 발생: {e}")
        
if __name__ == "__main__":
    main()
