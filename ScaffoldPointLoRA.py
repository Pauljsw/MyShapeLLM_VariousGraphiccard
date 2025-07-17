"""
ScaffoldPointLoRA: 비계 안전 검증 특화 PointLoRA 구현
ShapeLLM의 ReCon++ 3D encoder에 적용하여 비계 구조의 안전 관련 특징을 효율적으로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class ScaffoldLoRALayer(nn.Module):
    """
    비계 안전 특화 LoRA 레이어
    - 기존 Linear layer에 low-rank adaptation 적용
    - 비계 안전 관련 특징에 가중치 부여
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 매트릭스 (작은 rank로 효율성 확보)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 비계 안전 특화 가중치
        self.safety_weights = nn.Parameter(torch.ones(out_features) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA forward pass with safety weighting"""
        # Standard LoRA: x @ A^T @ B^T
        lora_out = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        # 비계 안전 특징에 가중치 적용
        safety_weighted = lora_out * self.safety_weights.unsqueeze(0)
        
        return safety_weighted


class ScaffoldTokenSelector(nn.Module):
    """
    비계 구조 특화 Multi-Scale Token Selection
    - 3단계 스케일로 비계의 계층적 구조 분석
    - 안전 위험도 기반 토큰 우선순위 설정
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 비계 구조의 3단계 스케일 정의
        self.scales = {
            'global': 256,    # 전체 비계 구조 (프레임워크)
            'component': 128, # 구성요소 (파이프, 조인트, 플랫폼)
            'detail': 64      # 세부사항 (연결부, 안전장치)
        }
        
        # 각 스케일별 토큰 선택기
        self.global_selector = self._build_selector(self.scales['global'], 16)
        self.component_selector = self._build_selector(self.scales['component'], 16) 
        self.detail_selector = self._build_selector(self.scales['detail'], 8)
        
        # 비계 안전 우선순위 정의
        self.safety_priorities = {
            'structural_joints': 0.95,    # 구조적 연결부 (최고 위험)
            'working_platforms': 0.90,    # 작업 플랫폼
            'guardrails': 0.85,          # 안전난간
            'access_points': 0.80,       # 접근/출입구
            'support_posts': 0.75,       # 지지대
            'bracing_elements': 0.70     # 가새 요소
        }
        
        # 공유 프롬프트 MLP (PointLoRA 논문의 Shared Prompt MLP)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def _build_selector(self, num_centers: int, num_selected: int) -> nn.Module:
        """토큰 선택기 구축"""
        return nn.Sequential(
            nn.Linear(self.embed_dim, num_centers),
            nn.ReLU(),
            nn.Linear(num_centers, num_selected),
            nn.Sigmoid()  # 선택 확률
        )
    
    def forward(self, point_features: torch.Tensor, point_coords: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Multi-scale token selection for scaffold safety analysis
        
        Args:
            point_features: [B, N, D] point cloud features
            point_coords: [B, N, 3] point coordinates
            
        Returns:
            selected_tokens: [B, 40, D] selected tokens (16+16+8)
            selection_info: dict with selection details
        """
        batch_size, num_points, feat_dim = point_features.shape
        
        # 1. 각 스케일에서 토큰 선택
        global_tokens = self._select_tokens_by_scale(
            point_features, point_coords, 'global', self.global_selector
        )
        
        component_tokens = self._select_tokens_by_scale(
            point_features, point_coords, 'component', self.component_selector
        )
        
        detail_tokens = self._select_tokens_by_scale(
            point_features, point_coords, 'detail', self.detail_selector
        )
        
        # 2. 선택된 토큰들을 연결
        selected_tokens = torch.cat([global_tokens, component_tokens, detail_tokens], dim=1)
        
        # 3. 공유 프롬프트 MLP 적용 (PointLoRA의 핵심)
        enhanced_tokens = self.prompt_mlp(selected_tokens)
        
        selection_info = {
            'global_count': global_tokens.shape[1],
            'component_count': component_tokens.shape[1], 
            'detail_count': detail_tokens.shape[1],
            'total_selected': enhanced_tokens.shape[1]
        }
        
        return enhanced_tokens, selection_info
    
    def _select_tokens_by_scale(self, features: torch.Tensor, coords: torch.Tensor, 
                               scale: str, selector: nn.Module) -> torch.Tensor:
        """특정 스케일에서 토큰 선택"""
        
        # 1. FPS로 중심점 샘플링
        num_centers = self.scales[scale]
        center_indices = self._farthest_point_sample(coords, num_centers)
        center_features = torch.gather(
            features, 1, 
            center_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        )
        
        # 2. k-NN으로 local neighborhood 구성
        local_features = self._get_local_neighborhoods(features, coords, center_indices)
        
        # 3. 안전 우선순위 기반 토큰 선택
        safety_scores = self._compute_safety_scores(local_features, coords, scale)
        selection_probs = selector(center_features)
        
        # 안전 점수와 선택 확률 결합
        final_scores = selection_probs * safety_scores.unsqueeze(-1)
        
        # 상위 토큰들 선택
        num_selected = selector[-2].out_features  # 마지막 Linear layer의 output size
        _, top_indices = torch.topk(final_scores.squeeze(-1), num_selected, dim=1)
        
        selected_tokens = torch.gather(
            center_features, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, center_features.shape[-1])
        )
        
        return selected_tokens
    
    def _compute_safety_scores(self, features: torch.Tensor, coords: torch.Tensor, scale: str) -> torch.Tensor:
        """비계 안전 우선순위 기반 점수 계산"""
        batch_size, num_points, _ = coords.shape
        safety_scores = torch.ones(batch_size, num_points, device=coords.device)
        
        if scale == 'global':
            # 전체 구조에서는 주요 지지점과 연결부에 높은 점수
            # 높이 기반 위험도 (높을수록 위험)
            height_risk = torch.sigmoid((coords[:, :, 2] - coords[:, :, 2].mean(dim=1, keepdim=True)) / 5.0)
            safety_scores *= (1.0 + height_risk * self.safety_priorities['structural_joints'])
            
        elif scale == 'component':
            # 구성요소 레벨에서는 작업 플랫폼과 난간에 집중
            # Z축 변화량으로 플랫폼 영역 감지
            z_variance = torch.var(coords[:, :, 2:3], dim=1, keepdim=True)
            platform_likelihood = torch.exp(-z_variance / 0.1)  # 평평한 영역일 가능성
            safety_scores *= (1.0 + platform_likelihood.squeeze(-1) * self.safety_priorities['working_platforms'])
            
        elif scale == 'detail':
            # 세부 레벨에서는 연결부와 안전장치에 집중
            # 점들 간의 거리 변화로 연결부 감지
            distances = torch.cdist(coords[:, :, :], coords[:, :, :])
            connection_density = torch.sum(distances < 0.5, dim=-1).float()  # 0.5m 내 점들 개수
            safety_scores *= (1.0 + (connection_density / num_points) * self.safety_priorities['structural_joints'])
            
        return safety_scores
    
    def _farthest_point_sample(self, coords: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Farthest Point Sampling (FPS) 구현"""
        batch_size, num_points, _ = coords.shape
        device = coords.device
        
        # 초기화
        sampled_indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
        distances = torch.full((batch_size, num_points), float('inf'), device=device)
        
        # 첫 번째 점은 랜덤 선택
        sampled_indices[:, 0] = torch.randint(0, num_points, (batch_size,), device=device)
        
        for i in range(1, num_samples):
            # 마지막 선택된 점에서 모든 점까지의 거리 계산
            last_selected = sampled_indices[:, i-1]
            last_points = coords[torch.arange(batch_size), last_selected]
            
            dists = torch.norm(coords - last_points.unsqueeze(1), dim=2)
            distances = torch.minimum(distances, dists)
            
            # 가장 먼 점 선택
            sampled_indices[:, i] = torch.argmax(distances, dim=1)
            
            # 선택된 점의 거리를 0으로 설정
            distances[torch.arange(batch_size), sampled_indices[:, i]] = 0
            
        return sampled_indices
    
    def _get_local_neighborhoods(self, features: torch.Tensor, coords: torch.Tensor, 
                               center_indices: torch.Tensor, k: int = 32) -> torch.Tensor:
        """k-NN으로 local neighborhood 특징 추출"""
        batch_size = features.shape[0]
        num_centers = center_indices.shape[1]
        
        # 중심점 좌표 추출
        center_coords = torch.gather(
            coords, 1, 
            center_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        
        local_features = []
        for i in range(num_centers):
            center = center_coords[:, i:i+1, :]  # [B, 1, 3]
            
            # 중심점에서 모든 점까지의 거리 계산
            distances = torch.norm(coords - center, dim=2)  # [B, N]
            
            # k개의 가장 가까운 점들 선택
            _, knn_indices = torch.topk(distances, k, dim=1, largest=False)
            
            # 해당 점들의 특징 추출
            knn_features = torch.gather(
                features, 1,
                knn_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            )
            
            # Max pooling으로 local feature 집계
            local_feat, _ = torch.max(knn_features, dim=1)  # [B, D]
            local_features.append(local_feat)
            
        return torch.stack(local_features, dim=1)  # [B, num_centers, D]


class ScaffoldPointLoRA(nn.Module):
    """
    비계 안전 검증 특화 PointLoRA 메인 모듈
    ShapeLLM의 ReCon++ 인코더에 통합하여 사용
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 lora_rank: int = 16, 
                 lora_alpha: float = 32.0,
                 num_selected_tokens: int = 40):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        self.num_selected_tokens = num_selected_tokens
        
        # Multi-Scale Token Selector
        self.token_selector = ScaffoldTokenSelector(hidden_size)
        
        # LoRA layers for QKV projections (ShapeLLM의 transformer attention에 적용)
        self.qkv_lora = nn.ModuleDict({
            'q': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha),
            'k': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha), 
            'v': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha)
        })
        
        # LoRA layers for FFN (Feed-Forward Network에 적용)
        self.ffn_lora = nn.ModuleDict({
            'up': ScaffoldLoRALayer(hidden_size, hidden_size * 4, lora_rank, lora_alpha),
            'down': ScaffoldLoRALayer(hidden_size * 4, hidden_size, lora_rank, lora_alpha)
        })
        
        # MLP Projection LoRA (ShapeLLM의 3개 프로젝터에 적용)
        self.projection_lora = nn.ModuleDict({
            'local': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha),
            'global': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha),
            'ape': ScaffoldLoRALayer(hidden_size, hidden_size, lora_rank, lora_alpha)
        })
        
    def forward(self, 
                point_features: torch.Tensor, 
                point_coords: torch.Tensor,
                mode: str = 'token_selection') -> dict:
        """
        ScaffoldPointLoRA forward pass
        
        Args:
            point_features: [B, N, D] ReCon++에서 추출된 point features
            point_coords: [B, N, 3] point coordinates  
            mode: 'token_selection' | 'qkv_adaptation' | 'ffn_adaptation' | 'projection_adaptation'
            
        Returns:
            dict with adapted features and selection info
        """
        
        if mode == 'token_selection':
            # Multi-scale token selection 수행
            selected_tokens, selection_info = self.token_selector(point_features, point_coords)
            
            return {
                'selected_tokens': selected_tokens,
                'selection_info': selection_info,
                'original_features': point_features
            }
            
        elif mode == 'qkv_adaptation':
            # Attention QKV 레이어에 LoRA 적용
            q_adapted = self.qkv_lora['q'](point_features)
            k_adapted = self.qkv_lora['k'](point_features) 
            v_adapted = self.qkv_lora['v'](point_features)
            
            return {
                'q': q_adapted,
                'k': k_adapted, 
                'v': v_adapted
            }
            
        elif mode == 'ffn_adaptation':
            # FFN 레이어에 LoRA 적용
            up_adapted = self.ffn_lora['up'](point_features)
            # 일반적으로 FFN에서 activation 함수 적용
            up_activated = F.gelu(up_adapted)
            down_adapted = self.ffn_lora['down'](up_activated)
            
            return {
                'ffn_output': down_adapted
            }
            
        elif mode == 'projection_adaptation':
            # MLP Projection 레이어에 LoRA 적용
            local_adapted = self.projection_lora['local'](point_features)
            global_adapted = self.projection_lora['global'](point_features)
            ape_adapted = self.projection_lora['ape'](point_features)
            
            return {
                'local_projection': local_adapted,
                'global_projection': global_adapted,
                'ape_projection': ape_adapted
            }
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """훈련 가능한 매개변수만 반환 (LoRA 매개변수들)"""
        trainable_params = []
        
        # Token selector parameters
        trainable_params.extend(list(self.token_selector.parameters()))
        
        # LoRA parameters
        for module_dict in [self.qkv_lora, self.ffn_lora, self.projection_lora]:
            for module in module_dict.values():
                trainable_params.extend([module.lora_A, module.lora_B, module.safety_weights])
                
        return trainable_params
    
    def get_parameter_count(self) -> dict:
        """매개변수 개수 분석"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'memory_saved_ratio': 1 - (trainable_params / total_params) if total_params > 0 else 0
        }


# 사용 예시 및 테스트 코드
def test_scaffold_pointlora():
    """ScaffoldPointLoRA 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트 데이터 생성 (비계 포인트 클라우드 시뮬레이션)
    batch_size = 2
    num_points = 8192  # ShapeLLM에서 사용하는 점 개수
    hidden_size = 768
    
    # 가상의 비계 포인트 클라우드 (실제로는 ShapeLLM ReCon++에서 나오는 features)
    point_features = torch.randn(batch_size, num_points, hidden_size).to(device)
    point_coords = torch.randn(batch_size, num_points, 3).to(device)
    
    # ScaffoldPointLoRA 초기화
    scaffold_lora = ScaffoldPointLoRA(
        hidden_size=hidden_size,
        lora_rank=16,
        lora_alpha=32.0,
        num_selected_tokens=40
    ).to(device)
    
    print("=== ScaffoldPointLoRA 테스트 ===")
    
    # 1. Token Selection 테스트
    print("\n1. Multi-Scale Token Selection 테스트:")
    selection_result = scaffold_lora(point_features, point_coords, mode='token_selection')
    print(f"선택된 토큰 shape: {selection_result['selected_tokens'].shape}")
    print(f"선택 정보: {selection_result['selection_info']}")
    
    # 2. QKV Adaptation 테스트  
    print("\n2. QKV Adaptation 테스트:")
    qkv_result = scaffold_lora(point_features, point_coords, mode='qkv_adaptation')
    print(f"Q shape: {qkv_result['q'].shape}")
    print(f"K shape: {qkv_result['k'].shape}")
    print(f"V shape: {qkv_result['v'].shape}")
    
    # 3. 매개변수 효율성 분석
    print("\n3. 매개변수 효율성 분석:")
    param_info = scaffold_lora.get_parameter_count()
    print(f"전체 매개변수: {param_info['total_parameters']:,}")
    print(f"훈련 가능 매개변수: {param_info['trainable_parameters']:,}")
    print(f"훈련 가능 비율: {param_info['trainable_ratio']:.2%}")
    print(f"메모리 절약 비율: {param_info['memory_saved_ratio']:.2%}")
    
    return scaffold_lora

if __name__ == "__main__":
    test_scaffold_pointlora()
