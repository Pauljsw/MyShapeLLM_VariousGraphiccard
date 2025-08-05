# scaffold_safety_ai/tests/test_training_pipeline.py
# Mock 데이터로 전체 훈련 파이프라인 검증

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import time

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.integrate_shapellm_gemini import PointLoRAReconEncoder

class MockScaffoldDataset(Dataset):
    """
    Mock 비계 안전 데이터셋
    실제 데이터 구조와 동일하지만 랜덤하게 생성
    """
    
    def __init__(self, num_samples: int = 100, num_points: int = 8192):
        self.num_samples = num_samples
        self.num_points = num_points
        
        # Mock 안전 평가 카테고리
        self.safety_categories = [
            "매우 안전", "안전", "주의 필요", "위험", "매우 위험"
        ]
        
        # Mock QA 템플릿
        self.qa_templates = [
            {
                "question": "이 비계의 안전성을 평가해주세요.",
                "answer_templates": [
                    "이 비계는 {safety_level}입니다. 난간 높이 {height}cm, 기준 대비 {status}.",
                    "안전 등급: {safety_level}. 주요 문제: {issues}.",
                    "종합 평가: {safety_level}. 개선 필요 사항: {improvements}."
                ]
            },
            {
                "question": "비계에서 발견되는 주요 안전 문제는 무엇인가요?",
                "answer_templates": [
                    "주요 문제: {issues}. 위험도: {risk_level}.",
                    "안전 문제 {count}개 발견: {issues_list}.",
                    "긴급 조치 필요: {urgent_issues}."
                ]
            },
            {
                "question": "이 비계가 안전 규정을 준수하는지 확인해주세요.",
                "answer_templates": [
                    "규정 준수 여부: {compliance}. 근거: {regulation}.",
                    "법적 요구사항 {compliance_rate}% 충족. 미준수 항목: {violations}.",
                    "안전기준 적합성: {compliance}. 개선 권고사항: {recommendations}."
                ]
            }
        ]
        
        # Mock 데이터 생성
        self.data = self._generate_mock_data()
        
    def _generate_mock_data(self) -> List[Dict]:
        """Mock 훈련 데이터 생성"""
        data = []
        
        for i in range(self.num_samples):
            # Mock point cloud (비계 형태를 시뮬레이션)
            point_cloud = self._generate_scaffold_point_cloud()
            
            # Mock safety assessment
            safety_level = np.random.choice(self.safety_categories)
            safety_score = np.random.randint(1, 6)  # 1-5 점수
            
            # Mock QA pair 생성
            qa_template = np.random.choice(self.qa_templates)
            question = qa_template["question"]
            answer = self._generate_mock_answer(qa_template, safety_level, safety_score)
            
            data.append({
                'point_cloud': point_cloud,
                'question': question,
                'answer': answer,
                'safety_score': safety_score,
                'sample_id': f"mock_scaffold_{i:04d}"
            })
            
        return data
    
    def _generate_scaffold_point_cloud(self) -> torch.Tensor:
        """비계 형태를 모방한 mock point cloud 생성"""
        points = []
        
        # 수직 기둥들 생성 (4개 모서리)
        for x in [-2, 2]:
            for z in [-2, 2]:
                # 각 기둥마다 수직으로 점들 생성
                y_coords = np.linspace(0, 10, 200)  # 10m 높이
                pillar_points = np.column_stack([
                    np.full(200, x) + np.random.normal(0, 0.1, 200),
                    y_coords + np.random.normal(0, 0.1, 200),
                    np.full(200, z) + np.random.normal(0, 0.1, 200)
                ])
                points.append(pillar_points)
        
        # 수평 난간들 생성
        heights = [1.0, 5.0, 9.0]  # 다양한 높이의 난간
        for height in heights:
            # 4면의 난간
            for side in range(4):
                if side == 0:  # front
                    x_coords = np.linspace(-2, 2, 100)
                    side_points = np.column_stack([
                        x_coords + np.random.normal(0, 0.05, 100),
                        np.full(100, height) + np.random.normal(0, 0.05, 100),
                        np.full(100, -2) + np.random.normal(0, 0.05, 100)
                    ])
                elif side == 1:  # back
                    x_coords = np.linspace(-2, 2, 100)
                    side_points = np.column_stack([
                        x_coords + np.random.normal(0, 0.05, 100),
                        np.full(100, height) + np.random.normal(0, 0.05, 100),
                        np.full(100, 2) + np.random.normal(0, 0.05, 100)
                    ])
                elif side == 2:  # left
                    z_coords = np.linspace(-2, 2, 100)
                    side_points = np.column_stack([
                        np.full(100, -2) + np.random.normal(0, 0.05, 100),
                        np.full(100, height) + np.random.normal(0, 0.05, 100),
                        z_coords + np.random.normal(0, 0.05, 100)
                    ])
                else:  # right
                    z_coords = np.linspace(-2, 2, 100)
                    side_points = np.column_stack([
                        np.full(100, 2) + np.random.normal(0, 0.05, 100),
                        np.full(100, height) + np.random.normal(0, 0.05, 100),
                        z_coords + np.random.normal(0, 0.05, 100)
                    ])
                points.append(side_points)
        
        # 모든 점들 결합
        all_points = np.vstack(points)
        
        # RGB 컬러 추가 (회색톤 비계)
        colors = np.random.uniform(0.4, 0.7, (all_points.shape[0], 3))
        
        # XYZ + RGB
        point_cloud_with_color = np.hstack([all_points, colors])
        
        # 지정된 개수로 샘플링
        if point_cloud_with_color.shape[0] > self.num_points:
            indices = np.random.choice(point_cloud_with_color.shape[0], self.num_points, replace=False)
            point_cloud_with_color = point_cloud_with_color[indices]
        elif point_cloud_with_color.shape[0] < self.num_points:
            # 부족한 경우 패딩
            needed = self.num_points - point_cloud_with_color.shape[0]
            padding = np.tile(point_cloud_with_color[-1], (needed, 1))
            point_cloud_with_color = np.vstack([point_cloud_with_color, padding])
        
        return torch.tensor(point_cloud_with_color, dtype=torch.float32)
    
    def _generate_mock_answer(self, qa_template: Dict, safety_level: str, safety_score: int) -> str:
        """Mock 답변 생성"""
        answer_template = np.random.choice(qa_template["answer_templates"])
        
        # Mock 변수들
        mock_values = {
            'safety_level': safety_level,
            'height': np.random.randint(80, 120),
            'status': '적합' if safety_score >= 3 else '부적합',
            'risk_level': ['낮음', '보통', '높음'][min(safety_score//2, 2)],
            'issues': '난간 높이 부족, 연결부 느슨함',
            'improvements': '난간 높이 조정, 연결부 점검',
            'compliance': '준수' if safety_score >= 3 else '미준수',
            'regulation': '산업안전보건기준 제15조',
            'compliance_rate': np.random.randint(60, 100),
            'violations': '난간 높이 기준',
            'recommendations': '즉시 보수 작업 필요',
            'count': np.random.randint(1, 5),
            'issues_list': '난간 높이, 연결부 상태',
            'urgent_issues': '구조 안정성 점검'
        }
        
        # 템플릿에 값 대입
        try:
            return answer_template.format(**mock_values)
        except KeyError:
            return answer_template
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'point_cloud': sample['point_cloud'],
            'question': sample['question'],
            'answer': sample['answer'],
            'safety_score': torch.tensor(sample['safety_score'], dtype=torch.long),
            'sample_id': sample['sample_id']
        }


class MockTrainingPipeline:
    """Mock 데이터 기반 훈련 파이프라인 테스트"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # 모델 설정
        self.config = self._create_mock_config()
        
        # 모델 초기화
        self.model = self._initialize_model()
        
        # 옵티마이저 설정
        self.optimizer = self._setup_optimizer()
        
    def _create_mock_config(self):
        """Mock 설정 생성"""
        class MockConfig:
            def __init__(self):
                self.embed_dim = 1024
                self.num_group = 512
                self.group_size = 32
                self.with_color = True
                self.mask_type = 'causal'
                self.mask_ratio = 0.5
                self.stop_grad = False
                self.img_queries = 15
                self.text_queries = 1
                self.depth = 24
                self.decoder_depth = 8
                self.num_heads = 16
                self.drop_path_rate = 0.1
                self.pretrained_model_name = ""
                self.lora_rank = 16
                self.lora_alpha = 32
                self.safety_token_count = 40
                self.contrast_type = "byol"
                self.large_embedding = False
        
        return MockConfig()
    
    def _initialize_model(self):
        """모델 초기화"""
        print("🔧 Initializing model...")
        model = PointLoRAReconEncoder(self.config)
        model.to(self.device)
        
        # 훈련 모드 설정 (LoRA만 활성화)
        model.set_training_mode(scaffold_training=True)
        
        return model
    
    def _setup_optimizer(self):
        """옵티마이저 설정"""
        # LoRA 파라미터만 수집
        lora_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lora_params.append(param)
        
        print(f"🔧 Trainable parameters: {sum(p.numel() for p in lora_params):,}")
        
        return optim.AdamW(lora_params, lr=5e-4, weight_decay=0.01)
    
    def test_forward_pass(self, dataloader):
        """Forward pass 테스트"""
        print("\n🧪 Testing forward pass...")
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 3:  # 처음 3개 배치만 테스트
                    break
                
                point_clouds = batch['point_cloud'].to(self.device)
                safety_scores = batch['safety_score'].to(self.device)
                
                print(f"  Batch {i+1}: point_clouds.shape = {point_clouds.shape}")
                
                try:
                    results = self.model.forward_safety_analysis(point_clouds)
                    
                    print(f"    ✅ Forward pass successful!")
                    print(f"    - Safety tokens: {results['safety_tokens'].shape}")
                    print(f"    - Predicted classes: {results['predicted_class']}")
                    print(f"    - Confidence: {results['confidence']}")
                    print(f"    - Ground truth: {safety_scores}")
                    
                except Exception as e:
                    print(f"    ❌ Forward pass failed: {e}")
                    return False
        
        return True
    
    def test_backward_pass(self, dataloader):
        """Backward pass 테스트"""
        print("\n🧪 Testing backward pass...")
        
        self.model.train()
        
        for i, batch in enumerate(dataloader):
            if i >= 3:  # 처음 3개 배치만 테스트
                break
            
            point_clouds = batch['point_cloud'].to(self.device)
            safety_scores = batch['safety_score'].to(self.device)
            
            try:
                self.optimizer.zero_grad()
                
                results = self.model.forward_safety_analysis(point_clouds)
                
                # 간단한 분류 손실 계산
                loss = nn.CrossEntropyLoss()(results['safety_logits'], safety_scores - 1)  # 0-4 범위로 조정
                
                loss.backward()
                self.optimizer.step()
                
                print(f"  Batch {i+1}: Loss = {loss.item():.4f}")
                print(f"    ✅ Backward pass successful!")
                
            except Exception as e:
                print(f"  Batch {i+1}: ❌ Backward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
    
    def test_mini_training(self, dataloader, num_steps=10):
        """미니 훈련 테스트"""
        print(f"\n🏃‍♂️ Testing mini training ({num_steps} steps)...")
        
        self.model.train()
        
        losses = []
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            point_clouds = batch['point_cloud'].to(self.device)
            safety_scores = batch['safety_score'].to(self.device)
            
            try:
                self.optimizer.zero_grad()
                
                results = self.model.forward_safety_analysis(point_clouds)
                loss = nn.CrossEntropyLoss()(results['safety_logits'], safety_scores - 1)
                
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                
                if step % 2 == 0:
                    print(f"  Step {step+1}/{num_steps}: Loss = {loss.item():.4f}")
                
            except Exception as e:
                print(f"  Step {step+1}: ❌ Training step failed: {e}")
                return False
        
        end_time = time.time()
        avg_loss = sum(losses) / len(losses)
        
        print(f"  ✅ Mini training completed!")
        print(f"  - Average loss: {avg_loss:.4f}")
        print(f"  - Training time: {end_time - start_time:.2f} seconds")
        print(f"  - Time per step: {(end_time - start_time) / num_steps:.2f} sec/step")
        
        return True
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 Starting mock training pipeline test...")
        print("=" * 60)
        
        # Mock 데이터셋 생성
        print("📦 Creating mock dataset...")
        train_dataset = MockScaffoldDataset(num_samples=50, num_points=8192)
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        
        print(f"✅ Mock dataset created: {len(train_dataset)} samples")
        
        # 테스트 단계들
        tests = [
            ("Forward Pass", lambda: self.test_forward_pass(train_dataloader)),
            ("Backward Pass", lambda: self.test_backward_pass(train_dataloader)),
            ("Mini Training", lambda: self.test_mini_training(train_dataloader, num_steps=10))
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED with exception: {e}")
                results[test_name] = False
        
        # 최종 결과
        print("\n" + "="*60)
        print("🎯 FINAL TEST RESULTS")
        print("="*60)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:<20}: {status}")
            if not passed:
                all_passed = False
        
        print("="*60)
        if all_passed:
            print("🎉 ALL TESTS PASSED! Training pipeline is ready for real data.")
        else:
            print("⚠️ Some tests failed. Please check the issues above.")
        
        return all_passed


def main():
    """메인 실행 함수"""
    pipeline = MockTrainingPipeline()
    success = pipeline.run_full_test()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Collect real scaffold point cloud data")
        print("2. Prepare scaffold safety QA dataset")
        print("3. Run full training with real data")
        print("4. Evaluate on validation set")
    else:
        print("\n⚠️ Fix the issues and run the test again.")


if __name__ == "__main__":
    main()