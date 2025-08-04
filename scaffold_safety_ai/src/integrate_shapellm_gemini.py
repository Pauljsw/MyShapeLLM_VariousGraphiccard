# LayerNorm 차원 오류 수정 및 Recon 모델 직접 상속
# scaffold_safety_ai/src/integrate_shapellm_gemini.py

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# PointLoRA 핵심 모듈 import
from .pointlora_core import LoRALayer, SafetyTokenSelector

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


class PointLoRAReconEncoder(ReCon2):
    """
    ReCon2 모델을 직접 상속받아 PointLoRA와 SafetyTokenSelector를 통합하는 클래스.
    이 클래스는 연구 방법론의 'A안'을 완벽하게 구현합니다.
    """
    def __init__(self, config: dict):
        # 부모 클래스(ReCon2)의 생성자를 호출하여 기본 Recon 모델 구조를 로드
        super().__init__(config)
        self.config = config
        
        # LoRA 레이어를 Vision Transformer(ReCon++)의 블록에 직접 주입
        self._add_lora_layers(
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha
        )
        
        # 안전 토큰 선택 모듈 초기화
        self.safety_selector = SafetyTokenSelector(
            feature_dim=config.embed_dim,
            safety_token_count=config.safety_token_count
        )
        
        # 안전성 등급 분류를 위한 헤드(Head)
        self.safety_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.LayerNorm(config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, 3) # 3: safe, warning, danger
        )
        
        print("✅ PointLoRAReconEncoder initialized with LoRA and Safety Head.")

    def _add_lora_layers(self, lora_rank, lora_alpha):
        """
        ReCon2의 MaskTransformer 인코더 블록에 LoRA 레이어를 추가하는 내부 메서드.
        """
        if not hasattr(self.model.encoder, 'blocks'):
            print("⚠️ Warning: Mock model encoder has no 'blocks' attribute. Skipping LoRA injection.")
            return

        # ReConBlocks 내부의 local_blocks(nn.Sequential)을 순회해야 함
        for i, block in enumerate(self.model.encoder.blocks.local_blocks):
            # Attention 블록의 QKV 프로젝션에 LoRA 적용
            qkv_dim = block.attn.qkv.in_features
            block.attn.qkv_lora = LoRALayer(qkv_dim, qkv_dim * 3, lora_rank, lora_alpha)

            # FFN(Feed-Forward Network)의 첫 번째 Linear 레이어에 LoRA 적용
            ffn_dim = block.mlp.fc1.in_features
            block.mlp.fc1_lora = LoRALayer(ffn_dim, block.mlp.fc1.out_features, lora_rank, lora_alpha)
            
            print(f"✅ LoRA layers injected into Transformer block {i}")
            
    def load_pretrained_weights(self, ckpt_path: str, log: bool = True):
        """
        사전 훈련된 ReCon 모델의 가중치를 로드하는 메서드.
        """
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint file not found: {ckpt_path}")
            return
            
        print(f"📦 Loading pre-trained weights from {ckpt_path}...")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            # --- 수정된 부분: 'state_dict'와 'base_model' 두 가지 키 모두 시도 ---
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
        """
        # 부모 클래스의 inference 메서드를 사용하여 기본 Recon 특징 추출
        pos, local_features, global_features = self.model.inference(pts)

        # local_features는 ReCon++의 패치별 특징이므로, 이를 SafetyTokenSelector에 전달
        safety_tokens, safety_indices = self.safety_selector(local_features)
        
        # 선택된 안전 토큰들의 평균을 계산하여 분류 헤드에 입력
        avg_safety_features = safety_tokens.mean(dim=1)
        safety_logits = self.safety_classifier(avg_safety_features)
        safety_probs = torch.softmax(safety_logits, dim=-1)
        
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
        # 모든 파라미터 고정
        for param in self.parameters():
            param.requires_grad = False
        
        if scaffold_training:
            # LoRA와 Safety Head의 파라미터만 활성화
            for name, param in self.named_parameters():
                if 'lora' in name or 'safety_selector' in name or 'safety_classifier' in name:
                    param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"🎯 Training mode set:")
        print(f"   Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    
    def _print_parameter_stats(self):
        """
        전체 파라미터 및 훈련 가능 파라미터 통계 출력
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        lora_params = sum(p.numel() for name, p in self.named_parameters() if 'lora' in name)
        safety_selector_params = sum(p.numel() for name, p in self.named_parameters() if 'safety_selector' in name)
        safety_classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'safety_classifier' in name)
        
        print(f"📊 Parameter Statistics:")
        print(f"   Total model params: {total_params:,}")
        print(f"   LoRA params: {lora_params:,}")
        print(f"   Safety Selector params: {safety_selector_params:,}")
        print(f"   Safety Classifier params: {safety_classifier_params:,}")
        print(f"   Total trainable params: {lora_params + safety_selector_params + safety_classifier_params:,}")

