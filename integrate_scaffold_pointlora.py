"""
ShapeLLM과 ScaffoldPointLoRA 통합
기존 ShapeLLM의 ReCon++ 인코더와 MLP 프로젝터에 PointLoRA 적용
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# 현재 프로젝트의 ShapeLLM 모듈들 import
# (실제 경로는 프로젝트 구조에 맞게 조정 필요)
try:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    from ReConV2.models.transformer import PatchEmbedding
    from ScaffoldPointLoRA import ScaffoldPointLoRA  # 위에서 구현한 모듈
except ImportError as e:
    print(f"Import 오류: {e}")
    print("프로젝트 PYTHONPATH 설정을 확인해주세요.")


class ScaffoldEnhancedCLIPVisionTower(CLIPVisionTower):
    """
    ScaffoldPointLoRA가 통합된 ShapeLLM의 Vision Tower
    기존 ReCon++ 인코더에 PointLoRA를 적용하여 비계 안전 도메인에 특화
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)
        
        # ScaffoldPointLoRA 초기화
        self.scaffold_lora = None
        self.use_scaffold_lora = getattr(args, 'use_scaffold_lora', True)
        self.scaffold_lora_rank = getattr(args, 'scaffold_lora_rank', 16)
        self.scaffold_lora_alpha = getattr(args, 'scaffold_lora_alpha', 32.0)
        
        # 학습 모드 설정
        self.training_stage = getattr(args, 'training_stage', 'full')  # 'full', 'lora_only'
        
        logging.info(f"ScaffoldEnhanced CLIPVisionTower 초기화")
        logging.info(f"use_scaffold_lora: {self.use_scaffold_lora}")
        logging.info(f"training_stage: {self.training_stage}")
    
    def load_model(self, device_map=None):
        """모델 로드 후 ScaffoldPointLoRA 적용"""
        # 기본 ShapeLLM 모델 로드
        super().load_model(device_map)
        
        if self.use_scaffold_lora:
            # ScaffoldPointLoRA 초기화 및 적용
            self._setup_scaffold_lora()
            self._apply_scaffold_lora()
            self._freeze_non_lora_parameters()
            
            logging.info("ScaffoldPointLoRA 적용 완료")
            self._log_parameter_statistics()
    
    def _setup_scaffold_lora(self):
        """ScaffoldPointLoRA 설정"""
        # ReCon++의 hidden_size 추출 (일반적으로 768 or 1024)
        if hasattr(self.vision_tower.model, 'embed'):
            hidden_size = self.vision_tower.model.embed.embed_dim
        else:
            hidden_size = 768  # 기본값
            
        self.scaffold_lora = ScaffoldPointLoRA(
            hidden_size=hidden_size,
            lora_rank=self.scaffold_lora_rank,
            lora_alpha=self.scaffold_lora_alpha,
            num_selected_tokens=40
        )
        
        # GPU로 이동
        if hasattr(self.vision_tower, 'device'):
            self.scaffold_lora = self.scaffold_lora.to(self.vision_tower.device)
        else:
            self.scaffold_lora = self.scaffold_lora.to("cuda:0")
    
    def _apply_scaffold_lora(self):
        """기존 ShapeLLM 모듈에 PointLoRA 통합"""
        
        # 1. ReCon++ Transformer 블록들에 LoRA 적용
        self._integrate_transformer_lora()
        
        # 2. MLP Projection 레이어들에 LoRA 적용  
        self._integrate_projection_lora()
        
        logging.info("모든 레이어에 ScaffoldPointLoRA 통합 완료")
    
    def _integrate_transformer_lora(self):
        """ReCon++ Transformer 블록들에 LoRA 적용"""
        
        # ReCon++ 모델의 transformer 블록들 찾기
        if hasattr(self.vision_tower.model, 'blocks'):
            transformer_blocks = self.vision_tower.model.blocks
        elif hasattr(self.vision_tower.model, 'layers'):
            transformer_blocks = self.vision_tower.model.layers
        else:
            logging.warning("Transformer 블록을 찾을 수 없습니다.")
            return
        
        for i, block in enumerate(transformer_blocks):
            # Self-Attention QKV에 LoRA 적용
            if hasattr(block, 'attn'):
                self._wrap_attention_with_lora(block.attn, f"block_{i}")
            
            # FFN에 LoRA 적용
            if hasattr(block, 'mlp'):
                self._wrap_ffn_with_lora(block.mlp, f"block_{i}")
    
    def _wrap_attention_with_lora(self, attention_module, block_name):
        """Attention 모듈을 LoRA로 감싸기"""
        
        # 기존 QKV projection 저장
        original_qkv = {}
        if hasattr(attention_module, 'qkv'):
            original_qkv['qkv'] = attention_module.qkv
        else:
            if hasattr(attention_module, 'q_proj'):
                original_qkv['q'] = attention_module.q_proj
            if hasattr(attention_module, 'k_proj'): 
                original_qkv['k'] = attention_module.k_proj
            if hasattr(attention_module, 'v_proj'):
                original_qkv['v'] = attention_module.v_proj
        
        # LoRA wrapper 함수 생성
        def lora_attention_forward(original_forward):
            def wrapped_forward(x, *args, **kwargs):
                # 원본 forward 실행
                original_output = original_forward(x, *args, **kwargs)
                
                # ScaffoldPointLoRA 적용 (좌표 정보가 필요한 경우)
                if hasattr(self, '_current_point_coords') and self.scaffold_lora is not None:
                    lora_adaptations = self.scaffold_lora(
                        x, self._current_point_coords, mode='qkv_adaptation'
                    )
                    
                    # LoRA adaptation을 원본 출력에 추가
                    if isinstance(original_output, tuple):
                        adapted_output = original_output[0] + lora_adaptations['q'] + lora_adaptations['k'] + lora_adaptations['v']
                        return (adapted_output,) + original_output[1:]
                    else:
                        return original_output + lora_adaptations['q'] + lora_adaptations['k'] + lora_adaptations['v']
                
                return original_output
            return wrapped_forward
        
        # Forward 함수 래핑
        attention_module.forward = lora_attention_forward(attention_module.forward)
        
        logging.debug(f"Attention LoRA 적용 완료: {block_name}")
    
    def _wrap_ffn_with_lora(self, ffn_module, block_name):
        """FFN 모듈을 LoRA로 감싸기"""
        
        def lora_ffn_forward(original_forward):
            def wrapped_forward(x, *args, **kwargs):
                # 원본 forward 실행
                original_output = original_forward(x, *args, **kwargs)
                
                # ScaffoldPointLoRA FFN adaptation 적용
                if hasattr(self, '_current_point_coords') and self.scaffold_lora is not None:
                    lora_adaptation = self.scaffold_lora(
                        x, self._current_point_coords, mode='ffn_adaptation'
                    )
                    
                    # LoRA adaptation을 원본 출력에 추가
                    return original_output + lora_adaptation['ffn_output']
                
                return original_output
            return wrapped_forward
        
        # Forward 함수 래핑
        ffn_module.forward = lora_ffn_forward(ffn_module.forward)
        
        logging.debug(f"FFN LoRA 적용 완료: {block_name}")
    
    def _integrate_projection_lora(self):
        """MLP Projection 레이어들에 LoRA 적용"""
        
        # ShapeLLM의 3개 프로젝터 찾기: local, global, APE
        projection_modules = {}
        
        # 가능한 projection 모듈들 탐색
        for name, module in self.named_modules():
            if 'proj' in name.lower() or 'projection' in name.lower():
                if 'local' in name.lower():
                    projection_modules['local'] = module
                elif 'global' in name.lower():
                    projection_modules['global'] = module
                elif 'ape' in name.lower():
                    projection_modules['ape'] = module
        
        # 각 projection에 LoRA 적용
        for proj_name, proj_module in projection_modules.items():
            self._wrap_projection_with_lora(proj_module, proj_name)
    
    def _wrap_projection_with_lora(self, projection_module, proj_name):
        """Projection 모듈을 LoRA로 감싸기"""
        
        def lora_projection_forward(original_forward):
            def wrapped_forward(x, *args, **kwargs):
                # 원본 forward 실행
                original_output = original_forward(x, *args, **kwargs)
                
                # ScaffoldPointLoRA projection adaptation 적용
                if hasattr(self, '_current_point_coords') and self.scaffold_lora is not None:
                    lora_adaptations = self.scaffold_lora(
                        x, self._current_point_coords, mode='projection_adaptation'
                    )
                    
                    # 해당 projection의 LoRA adaptation 추가
                    if proj_name in lora_adaptations:
                        return original_output + lora_adaptations[proj_name]
                    elif f"{proj_name}_projection" in lora_adaptations:
                        return original_output + lora_adaptations[f"{proj_name}_projection"]
                
                return original_output
            return wrapped_forward
        
        # Forward 함수 래핑
        projection_module.forward = lora_projection_forward(projection_module.forward)
        
        logging.debug(f"Projection LoRA 적용 완료: {proj_name}")
    
    def _freeze_non_lora_parameters(self):
        """LoRA가 아닌 매개변수들 고정"""
        
        if self.training_stage == 'lora_only':
            # 모든 기존 매개변수 고정
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            
            # LoRA 매개변수만 학습 가능하게 설정
            if self.scaffold_lora is not None:
                for param in self.scaffold_lora.get_trainable_parameters():
                    param.requires_grad = True
            
            logging.info("LoRA-only 모드: 기존 매개변수 고정, LoRA 매개변수만 학습")
        
        elif self.training_stage == 'full':
            # 전체 fine-tuning (기본 ShapeLLM 방식)
            logging.info("Full fine-tuning 모드: 모든 매개변수 학습")
    
    def _log_parameter_statistics(self):
        """매개변수 통계 로깅"""
        if self.scaffold_lora is not None:
            param_info = self.scaffold_lora.get_parameter_count()
            
            # 전체 모델 매개변수 계산
            total_model_params = sum(p.numel() for p in self.parameters())
            trainable_model_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            logging.info("=== 매개변수 통계 ===")
            logging.info(f"전체 모델 매개변수: {total_model_params:,}")
            logging.info(f"훈련 가능 매개변수: {trainable_model_params:,}")
            logging.info(f"LoRA 매개변수: {param_info['trainable_parameters']:,}")
            logging.info(f"훈련 효율성: {trainable_model_params/total_model_params:.2%}")
            logging.info(f"메모리 절약: {1-(trainable_model_params/total_model_params):.2%}")
    
    @torch.no_grad()
    def forward(self, pts):
        """
        ScaffoldPointLoRA가 적용된 forward pass
        """
        print("🏗️ [DEBUG] ScaffoldEnhanced CLIPVisionTower.forward() called")
        
        # 포인트 좌표 정보 저장 (LoRA에서 사용)
        if isinstance(pts, list):
            # 첫 번째 포인트 클라우드의 좌표 사용 (xyz)
            if len(pts) > 0 and pts[0].shape[-1] >= 3:
                self._current_point_coords = pts[0][:, :, :3].unsqueeze(0)  # [1, N, 3]
        elif pts.shape[-1] >= 3:
            self._current_point_coords = pts[:, :, :3]  # [B, N, 3]
        
        # Multi-scale token selection 수행
        if self.scaffold_lora is not None and hasattr(self, '_current_point_coords'):
            
            # 기본 features 추출 (부모 클래스 forward 호출 전에 미리 준비)
            if isinstance(pts, list):
                point_features_list = []
                for pt in pts:
                    # 간단한 embedding으로 초기 features 생성
                    feat = torch.randn(1, pt.shape[0], self.scaffold_lora.hidden_size).to(pt.device)
                    point_features_list.append(feat)
                point_features = point_features_list[0]  # 첫 번째 사용
            else:
                point_features = torch.randn(pts.shape[0], pts.shape[1], self.scaffold_lora.hidden_size).to(pts.device)
            
            # Multi-scale token selection
            selection_result = self.scaffold_lora(
                point_features, self._current_point_coords, mode='token_selection'
            )
            
            print(f"🎯 선택된 토큰 수: {selection_result['selected_tokens'].shape[1]}")
            print(f"🎯 선택 정보: {selection_result['selection_info']}")
        
        # 기본 ShapeLLM forward 실행 (LoRA가 자동으로 적용됨)
        return super().forward(pts)


def create_scaffold_enhanced_shapellm(model_path: str, **kwargs):
    """
    ScaffoldPointLoRA가 적용된 ShapeLLM 생성
    
    Args:
        model_path: ShapeLLM 모델 경로
        **kwargs: 추가 설정 (lora_rank, lora_alpha, training_stage 등)
    """
    
    # 설정 클래스 생성
    class ScaffoldConfig:
        def __init__(self, **kwargs):
            self.use_scaffold_lora = kwargs.get('use_scaffold_lora', True)
            self.scaffold_lora_rank = kwargs.get('scaffold_lora_rank', 16)
            self.scaffold_lora_alpha = kwargs.get('scaffold_lora_alpha', 32.0)
            self.training_stage = kwargs.get('training_stage', 'lora_only')  # 'lora_only' or 'full'
    
    config = ScaffoldConfig(**kwargs)
    
    try:
        # 기존 ShapeLLM 모델 로드
        from transformers import AutoModel, AutoTokenizer
        
        print(f"🏗️ ScaffoldEnhanced ShapeLLM 로딩 중: {model_path}")
        
        # 원본 vision tower 생성
        vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14', config)
        
        # ScaffoldPointLoRA로 향상된 vision tower로 교체
        enhanced_vision_tower = ScaffoldEnhancedCLIPVisionTower('openai/clip-vit-large-patch14', config)
        enhanced_vision_tower.load_model()
        
        print("✅ ScaffoldEnhanced ShapeLLM 로딩 완료")
        return enhanced_vision_tower
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None


def save_scaffold_lora_weights(model, save_path: str):
    """
    ScaffoldPointLoRA 가중치만 저장 (효율적 저장)
    
    Args:
        model: ScaffoldEnhancedCLIPVisionTower
        save_path: 저장 경로
    """
    if hasattr(model, 'scaffold_lora') and model.scaffold_lora is not None:
        lora_state_dict = {}
        
        # LoRA 매개변수만 추출
        for name, param in model.scaffold_lora.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.data.cpu()
        
        # 설정 정보도 함께 저장
        save_dict = {
            'lora_state_dict': lora_state_dict,
            'config': {
                'scaffold_lora_rank': model.scaffold_lora_rank,
                'scaffold_lora_alpha': model.scaffold_lora_alpha,
                'training_stage': model.training_stage,
                'hidden_size': model.scaffold_lora.hidden_size
            }
        }
        
        torch.save(save_dict, save_path)
        print(f"✅ LoRA 가중치 저장 완료: {save_path}")
        print(f"📊 저장된 매개변수 수: {len(lora_state_dict):,}")
    else:
        print("❌ ScaffoldPointLoRA가 초기화되지 않음")


def load_scaffold_lora_weights(model, load_path: str):
    """
    저장된 ScaffoldPointLoRA 가중치 로드
    
    Args:
        model: ScaffoldEnhancedCLIPVisionTower
        load_path: 로드 경로
    """
    if not Path(load_path).exists():
        print(f"❌ 파일이 존재하지 않음: {load_path}")
        return False
    
    try:
        checkpoint = torch.load(load_path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        config = checkpoint.get('config', {})
        
        if hasattr(model, 'scaffold_lora') and model.scaffold_lora is not None:
            # LoRA 가중치 로드
            missing_keys, unexpected_keys = model.scaffold_lora.load_state_dict(lora_state_dict, strict=False)
            
            print(f"✅ LoRA 가중치 로드 완료: {load_path}")
            print(f"📊 로드된 매개변수 수: {len(lora_state_dict):,}")
            if missing_keys:
                print(f"⚠️ 누락된 키: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 예상치 못한 키: {unexpected_keys}")
            
            return True
        else:
            print("❌ ScaffoldPointLoRA가 초기화되지 않음")
            return False
            
    except Exception as e:
        print(f"❌ LoRA 가중치 로드 실패: {e}")
        return False


# 비계 데이터셋 처리를 위한 유틸리티
class ScaffoldDataProcessor:
    """
    비계 포인트 클라우드 데이터 전처리 및 증강
    """
    
    def __init__(self, target_points: int = 8192):
        self.target_points = target_points
        
        # 비계 안전 검사 항목들
        self.safety_checkpoints = {
            'structural_integrity': ['joints', 'connections', 'support_posts'],
            'working_platform': ['platform_surface', 'guardrails', 'access_points'],
            'height_safety': ['fall_protection', 'ladder_safety', 'vertical_spacing'],
            'material_condition': ['corrosion', 'damage', 'wear_patterns']
        }
    
    def process_scaffold_pointcloud(self, pts_file: str) -> Dict[str, Any]:
        """
        비계 포인트 클라우드 처리
        
        Args:
            pts_file: .npy 포인트 클라우드 파일 경로
            
        Returns:
            dict: 처리된 데이터와 메타정보
        """
        try:
            # 포인트 클라우드 로드
            points = np.load(pts_file)
            print(f"📂 원본 포인트 클라우드 shape: {points.shape}")
            
            # 좌표 정규화
            coords = points[:, :3]  # x, y, z
            colors = points[:, 3:6] if points.shape[1] >= 6 else None
            
            # 포인트 수 조정
            if len(coords) > self.target_points:
                # 다운샘플링 (FPS 사용)
                indices = self._farthest_point_sample(coords, self.target_points)
                coords = coords[indices]
                if colors is not None:
                    colors = colors[indices]
            elif len(coords) < self.target_points:
                # 업샘플링 (복제 + 노이즈)
                coords = self._upsample_points(coords, self.target_points)
                if colors is not None:
                    colors = self._upsample_points(colors, self.target_points)
            
            # 좌표 정규화 (-1 ~ 1)
            coords_normalized = self._normalize_coordinates(coords)
            
            # 최종 포인트 클라우드 구성
            if colors is not None:
                final_points = np.concatenate([coords_normalized, colors], axis=1)
            else:
                # 색상 정보가 없으면 dummy 생성
                dummy_colors = np.ones((len(coords_normalized), 3)) * 0.5
                final_points = np.concatenate([coords_normalized, dummy_colors], axis=1)
            
            # 메타 정보 추출
            metadata = self._extract_scaffold_metadata(coords_normalized)
            
            return {
                'points': final_points,
                'coordinates': coords_normalized,
                'colors': colors,
                'metadata': metadata,
                'original_shape': points.shape,
                'processed_shape': final_points.shape
            }
            
        except Exception as e:
            print(f"❌ 포인트 클라우드 처리 실패: {e}")
            return None
    
    def _farthest_point_sample(self, points: np.ndarray, num_samples: int) -> np.ndarray:
        """Farthest Point Sampling"""
        n_points = len(points)
        indices = np.zeros(num_samples, dtype=int)
        distances = np.full(n_points, float('inf'))
        
        # 첫 번째 점은 랜덤 선택
        indices[0] = np.random.randint(0, n_points)
        
        for i in range(1, num_samples):
            # 마지막 선택된 점에서 모든 점까지의 거리 계산
            last_point = points[indices[i-1]]
            dists = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dists)
            
            # 가장 먼 점 선택
            indices[i] = np.argmax(distances)
            distances[indices[i]] = 0
            
        return indices
    
    def _upsample_points(self, points: np.ndarray, target_size: int) -> np.ndarray:
        """포인트 업샘플링 (복제 + 노이즈)"""
        current_size = len(points)
        repeat_times = target_size // current_size + 1
        
        # 포인트 복제
        repeated = np.tile(points, (repeat_times, 1))[:target_size]
        
        # 작은 노이즈 추가
        noise = np.random.normal(0, 0.01, repeated.shape)
        upsampled = repeated + noise
        
        return upsampled
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """좌표 정규화 (-1 ~ 1)"""
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        ranges = max_vals - min_vals
        
        # 0으로 나누기 방지
        ranges[ranges == 0] = 1.0
        
        normalized = 2 * (coords - min_vals) / ranges - 1
        return normalized
    
    def _extract_scaffold_metadata(self, coords: np.ndarray) -> Dict[str, Any]:
        """비계 구조 메타데이터 추출"""
        metadata = {}
        
        # 기본 통계
        metadata['height_range'] = [coords[:, 2].min(), coords[:, 2].max()]
        metadata['width_range'] = [coords[:, 0].min(), coords[:, 0].max()]
        metadata['depth_range'] = [coords[:, 1].min(), coords[:, 1].max()]
        
        # 높이별 밀도 분석 (층별 구조 파악)
        height_bins = np.linspace(coords[:, 2].min(), coords[:, 2].max(), 10)
        height_density, _ = np.histogram(coords[:, 2], bins=height_bins)
        metadata['height_density'] = height_density.tolist()
        
        # 수직 구조 감지 (기둥, 지지대)
        vertical_variance = np.var(coords[:, :2], axis=0)  # x, y 축 분산
        metadata['vertical_structure_strength'] = float(np.mean(vertical_variance))
        
        # 수평 구조 감지 (플랫폼, 작업면)
        z_variance = np.var(coords[:, 2])
        metadata['horizontal_structure_strength'] = float(z_variance)
        
        return metadata


# 테스트 및 실행 함수
def test_scaffold_integration():
    """전체 통합 테스트"""
    print("🏗️ ScaffoldPointLoRA + ShapeLLM 통합 테스트 시작")
    
    # 1. 데이터 처리 테스트
    processor = ScaffoldDataProcessor()
    
    # 실제 비계 데이터가 있다면 사용, 없으면 더미 데이터 생성
    scaffold_file = "assets/SW_Scaffold_8192.npy"
    if Path(scaffold_file).exists():
        print(f"📂 실제 비계 데이터 로드: {scaffold_file}")
        processed_data = processor.process_scaffold_pointcloud(scaffold_file)
    else:
        print("📂 더미 비계 데이터 생성")
        # 더미 비계 포인트 클라우드 생성
        dummy_points = np.random.randn(8192, 6)
        np.save("dummy_scaffold.npy", dummy_points)
        processed_data = processor.process_scaffold_pointcloud("dummy_scaffold.npy")
    
    if processed_data:
        print(f"✅ 데이터 처리 완료:")
        print(f"   - 원본 shape: {processed_data['original_shape']}")
        print(f"   - 처리 후 shape: {processed_data['processed_shape']}")
        print(f"   - 메타데이터: {list(processed_data['metadata'].keys())}")
    
    # 2. ScaffoldEnhanced ShapeLLM 생성
    enhanced_model = create_scaffold_enhanced_shapellm(
        model_path="qizekun/ShapeLLM_13B_general_v1.0",
        use_scaffold_lora=True,
        scaffold_lora_rank=16,
        scaffold_lora_alpha=32.0,
        training_stage='lora_only'
    )
    
    if enhanced_model:
        print("✅ ScaffoldEnhanced ShapeLLM 생성 완료")
        
        # 3. 테스트 forward pass
        if processed_data:
            test_points = torch.FloatTensor(processed_data['points']).unsqueeze(0)  # [1, N, 6]
            
            try:
                with torch.no_grad():
                    output = enhanced_model(test_points)
                print("✅ Forward pass 테스트 완료")
                print(f"   - 출력 타입: {type(output)}")
                if hasattr(output, 'shape'):
                    print(f"   - 출력 shape: {output.shape}")
            except Exception as e:
                print(f"❌ Forward pass 실패: {e}")
    
    print("🏁 통합 테스트 완료")


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 통합 테스트 실행
    test_scaffold_integration()
