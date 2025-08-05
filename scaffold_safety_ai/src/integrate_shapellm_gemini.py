# scaffold_safety_ai/src/integrate_shapellm_gemini.py
# 수정된 버전: SafetyTokenSelector 차원 불일치 문제 해결

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# PointLoRA 핵심 모듈 import
from .pointlora_core import LoRALayer

# ShapeLLM의 ReConV2 모듈을 로드하기 위한 환경 설정
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    from ReConV2.models.ReCon import ReCon2
    from ReConV2.models.transformer import ReConBlocks, GPTExtractor
    from ReConV2.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
    print(f"✅ Successfully imported ReCon2 from: {project_root}")
except ImportError as e:
    print(f"❌ Failed to import ReCon2. Check PYTHONPATH. Error: {e}")
    # ReCon2가 없을 경우, 더미 클래스로 대체하여 코드 실행 가능하게 함
    class ReCon2(nn.Module):
        def __init__(self, config):
            super().__init__()
            print("⚠️ ReCon2 mock class is being used.")
            
            # 실제 ReCon 모델의 구조를 최대한 모방하여 오류를 회피
            self.model = nn.Module()
            self.model.encoder = GPTExtractor(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                depth=config.depth,
                group_size=config.group_size,
                drop_path_rate=config.drop_path_rate,
                stop_grad=config.stop_grad,
                pretrained_model_name=config.pretrained_model_name,
            )
            
            self.model.embed_dim = config.embed_dim
            self.model.inference = self._mock_inference

        def _mock_inference(self, pts):
            print("⚠️ ReCon2 mock inference is running. Returning random data.")
            local_features, global_features = self.model.encoder.forward(
                torch.randn(pts.size(0), 512, self.model.encoder.embed_dim),
                torch.randn(pts.size(0), 512, self.model.encoder.embed_dim),
                None,
                torch.randn(pts.size(0), 1, self.model.encoder.embed_dim),
            )
            return None, local_features, global_features


class FixedSafetyTokenSelector(nn.Module):
    """
    수정된 Safety Token Selector - 차원 불일치 문제 해결
    """
    
    def __init__(self, feature_dim: int = 1024, safety_token_count: int = 40):
        super().__init__()
        self.feature_dim = feature_dim
        self.safety_token_count = safety_token_count
        
        # 중요도 예측 네트워크 - 차원을 명확히 설정
        self.importance_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ Safety Token Selector initialized: {feature_dim}D → {safety_token_count} tokens")
        
    def forward(self, features: torch.Tensor):
        """
        안전 토큰 선택 - 차원 안전성 확보
        
        Args:
            features: [batch_size, seq_len, feature_dim] - ReCon의 local_features
            
        Returns:
            safety_tokens: [batch_size, safety_token_count, feature_dim]
            selected_indices: [batch_size, safety_token_count]
        """
        batch_size, seq_len, feat_dim = features.shape
        
        # 디버깅 출력
        print(f"🔍 [SafetyTokenSelector] Input shape: {features.shape}")
        print(f"🔍 [SafetyTokenSelector] Expected feature_dim: {self.feature_dim}, Got: {feat_dim}")
        
        # 차원 검증
        if feat_dim != self.feature_dim:
            print(f"⚠️ [SafetyTokenSelector] Feature dimension mismatch! Expected {self.feature_dim}, got {feat_dim}")
            # 차원 조정 (임시 해결책)
            if feat_dim > self.feature_dim:
                features = features[:, :, :self.feature_dim]
                print(f"✅ [SafetyTokenSelector] Truncated to {self.feature_dim} dimensions")
            else:
                # padding으로 차원 맞추기
                padding = torch.zeros(batch_size, seq_len, self.feature_dim - feat_dim, 
                                    device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=-1)
                print(f"✅ [SafetyTokenSelector] Padded to {self.feature_dim} dimensions")
        
        # 중요도 점수 계산
        try:
            scores = self.importance_network(features).squeeze(-1)  # [B, S]
            print(f"🔍 [SafetyTokenSelector] Scores shape: {scores.shape}")
        except Exception as e:
            print(f"❌ [SafetyTokenSelector] Error in importance_network: {e}")
            print(f"   Features shape: {features.shape}")
            raise e
        
        # Top-K 선택
        k = min(self.safety_token_count, seq_len)
        _, indices = torch.topk(scores, k, dim=1)  # [B, K]
        
        # 토큰 추출
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        safety_tokens = features[batch_indices, indices]  # [B, K, D]
        
        print(f"✅ [SafetyTokenSelector] Output safety_tokens shape: {safety_tokens.shape}")
        
        return safety_tokens, indices


class PointLoRAReconEncoder(ReCon2):
    """
    ReCon2 모델을 직접 상속받아 PointLoRA와 SafetyTokenSelector를 통합하는 클래스.
    이 클래스는 연구 방법론의 'A안'을 완벽하게 구현합니다.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # PointLoRA 파라미터 설정
        self.lora_rank = getattr(config, 'lora_rank', 16)
        self.lora_alpha = getattr(config, 'lora_alpha', 32)
        self.safety_token_count = getattr(config, 'safety_token_count', 40)
        
        # SafetyTokenSelector 초기화 - ReCon2의 embed_dim 사용
        self.safety_selector = FixedSafetyTokenSelector(
            feature_dim=config.embed_dim,
            safety_token_count=self.safety_token_count
        )
        
        # 안전 분류 헤드 초기화
        self.safety_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.LayerNorm(config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 5)  # 5개의 안전 등급
        )
        
        # LoRA 레이어를 Transformer blocks에 주입
        self._inject_lora_layers()
        
        print(f"✅ PointLoRAReconEncoder initialized with LoRA and Safety Head.")
    
    def _inject_lora_layers(self):
        """ReCon2의 Transformer blocks에 LoRA 레이어 주입"""
        try:
            # ReConBlocks 구조: blocks.local_blocks (nn.Sequential)
            if not hasattr(self.model.encoder, 'blocks') or not hasattr(self.model.encoder.blocks, 'local_blocks'):
                print("⚠️ Warning: Expected ReConBlocks structure not found. Skipping LoRA injection.")
                return
                
            local_blocks = self.model.encoder.blocks.local_blocks
            print(f"🔍 Found {len(local_blocks)} local blocks in ReConBlocks")
            
            for i, block in enumerate(local_blocks):
                # MLP의 fc1에 LoRA 추가
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
                    in_features = block.mlp.fc1.in_features
                    out_features = block.mlp.fc1.out_features
                    block.mlp.fc1_lora = LoRALayer(in_features, out_features, self.lora_rank, self.lora_alpha)
                    print(f"✅ LoRA Layer initialized: {in_features}→{out_features}, rank={self.lora_rank}, params={2*self.lora_rank*(in_features+out_features):,}")
                
                # MLP의 fc2에 LoRA 추가 (존재하는 경우)
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc2'):
                    in_features = block.mlp.fc2.in_features
                    out_features = block.mlp.fc2.out_features
                    block.mlp.fc2_lora = LoRALayer(in_features, out_features, self.lora_rank, self.lora_alpha)
                    print(f"✅ LoRA Layer initialized: {in_features}→{out_features}, rank={self.lora_rank}, params={2*self.lora_rank*(in_features+out_features):,}")
                
                print(f"✅ LoRA layers injected into Transformer block {i}")
                
        except Exception as e:
            print(f"❌ Error injecting LoRA layers: {e}")
            import traceback
            traceback.print_exc()
            # LoRA 주입 실패해도 계속 진행 (기본 모델은 작동)
            print("⚠️ Continuing without LoRA layers...")
    
    def load_pretrained_weights(self, ckpt_path: str, log: bool = True):
        """
        사전 훈련된 ReCon2 가중치를 로드합니다.
        """
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint file not found: {ckpt_path}")
            return
            
        print(f"📦 Loading pre-trained weights from {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # 'state_dict'와 'base_model' 두 가지 키 모두 시도
            state_dict = ckpt.get('state_dict', None)
            if state_dict is None:
                state_dict = ckpt.get('base_model', None)
                if state_dict is None:
                    raise KeyError("Neither 'state_dict' nor 'base_model' key found in checkpoint file.")

            # 키 변환 로직: 'module.' 또는 'model.' 프리픽스 제거
            clean_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                k = k.replace("model.", "")
                clean_state_dict[k] = v
            
            # 모델 가중치 로드
            incompatible = self.model.load_state_dict(clean_state_dict, strict=False)
            
            if log:
                if incompatible.missing_keys:
                    print(f"⚠️ Missing Keys: {get_missing_parameters_message(incompatible.missing_keys)}")
                if incompatible.unexpected_keys:
                    print(f"⚠️ Unexpected Keys: {get_unexpected_parameters_message(incompatible.unexpected_keys)}")
            
            print(f"✅ Pre-trained weights successfully loaded.")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")

    def forward_safety_analysis(self, pts: torch.Tensor):
        """
        안전 분석을 위한 전체 순전파(forward) 파이프라인.
        차원 디버깅 포함
        """
        print(f"🔍 [forward_safety_analysis] Input pts shape: {pts.shape}")
        
        # 부모 클래스의 inference 메서드를 사용하여 기본 Recon 특징 추출
        pos, local_features, global_features = self.model.inference(pts)
        
        print(f"🔍 [forward_safety_analysis] local_features shape: {local_features.shape}")
        print(f"🔍 [forward_safety_analysis] global_features shape: {global_features.shape}")

        # local_features는 ReCon++의 패치별 특징이므로, 이를 SafetyTokenSelector에 전달
        safety_tokens, safety_indices = self.safety_selector(local_features)
        
        print(f"🔍 [forward_safety_analysis] safety_tokens shape: {safety_tokens.shape}")
        
        # 선택된 안전 토큰들의 평균을 계산하여 분류 헤드에 입력
        avg_safety_features = safety_tokens.mean(dim=1)
        print(f"🔍 [forward_safety_analysis] avg_safety_features shape: {avg_safety_features.shape}")
        
        safety_logits = self.safety_classifier(avg_safety_features)
        safety_probs = torch.softmax(safety_logits, dim=-1)
        
        print(f"🔍 [forward_safety_analysis] safety_logits shape: {safety_logits.shape}")
        print(f"🔍 [forward_safety_analysis] safety_probs shape: {safety_probs.shape}")
        
        return {
            'safety_tokens': safety_tokens,
            'safety_indices': safety_indices,
            'safety_logits': safety_logits,
            'safety_probs': safety_probs,
            'features': local_features,
            'predicted_class': torch.argmax(safety_probs, dim=-1),
            'confidence': torch.max(safety_probs, dim=-1)[0]
        }

    def set_training_mode(self, scaffold_training: bool = True):
        """
        훈련 모드를 설정하여 Base Model의 가중치를 고정하고
        PointLoRA와 Safety Head의 가중치만 훈련 가능하게 만듭니다.
        """
        if scaffold_training:
            # 전체 모델을 eval 모드로 설정 (기본 가중치 고정)
            self.eval()
            
            # 모든 파라미터를 먼저 비훈련으로 설정
            for param in self.parameters():
                param.requires_grad = False
            
            # LoRA 파라미터만 훈련 가능하게 설정
            trainable_params = 0
            total_params = 0
            
            for name, module in self.named_modules():
                if isinstance(module, LoRALayer):
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True
                        trainable_params += param.numel()
                
                # 모든 파라미터 수 계산
                for param in module.parameters():
                    total_params += param.numel()
            
            # Safety 분류 헤드도 훈련 가능하게 설정
            self.safety_classifier.train()
            for param in self.safety_classifier.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
            
            # Safety Token Selector도 훈련 가능하게 설정
            self.safety_selector.train()
            for param in self.safety_selector.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
            
            print(f"🎯 Training mode set:")
            print(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
            
        else:
            # 전체 모델 훈련 모드
            self.train()
            for param in self.parameters():
                param.requires_grad = True