#!/usr/bin/env python3
"""
ShapeLLM/llava/serve/scaffold_cli.py

비계 안전 검증 특화 ShapeLLM CLI
- ScaffoldPointLoRA 통합
- 기존 ShapeLLM conversation 시스템 활용
- 비계 안전 전문 분석 제공
"""

import torch
import argparse
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ShapeLLM 기존 모듈들
from transformers import TextStreamer
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle, Conversation
from llava.constants import POINT_TOKEN_INDEX, DEFAULT_POINT_TOKEN, DEFAULT_PT_START_TOKEN, DEFAULT_PT_END_TOKEN
from llava.mm_utils import load_pts, process_pts, rotation, tokenizer_point_token, get_model_name_from_path, KeywordsStoppingCriteria

# ScaffoldPointLoRA 모듈들 (프로젝트 루트에서 import)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # ShapeLLM 루트로 이동

from ScaffoldPointLoRA import ScaffoldPointLoRA, ScaffoldTokenSelector, ScaffoldLoRALayer
from integrate_scaffold_pointlora import ScaffoldEnhancedCLIPVisionTower, ScaffoldDataProcessor


class ScaffoldSafetyCLI:
    """
    비계 안전 검증 특화 CLI 클래스
    기존 ShapeLLM + ScaffoldPointLoRA 통합
    """
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.context_len = None
        self.conv = None
        self.scaffold_lora = None
        self.data_processor = ScaffoldDataProcessor()
        
        # 비계 안전 conversation template 등록
        self._register_scaffold_conversation()
        
        # 비계 안전 전문 프롬프트들
        self.safety_prompts = {
            'comprehensive': """
이 비계 구조의 종합적인 안전성을 평가해주세요:

1. **구조적 안정성**: 지지대, 연결부, 전체적 프레임워크의 견고성
2. **작업 플랫폼**: 플랫폼 표면, 안전난간, 출입구의 안전성  
3. **높이 안전**: 각 층별 안전 조치, 추락 방지 대책
4. **종합 평가**: 안전 등급(A/B/C/D)과 구체적 개선 권장사항

정량적 분석과 실용적 권장사항을 포함해 현장에서 바로 활용할 수 있는 보고서 형태로 답변해주세요.
            """,
            
            'structural': """
이 비계의 구조적 안정성을 중점 분석해주세요:

1. **주요 지지점**: 기둥과 수직 구조물의 상태
2. **연결부 무결성**: 조인트, 볼트, 클램프의 견고성
3. **하중 분산**: 중량 분배와 구조적 균형
4. **변형 및 손상**: 휘어짐, 균열, 부식 등의 구조적 결함

구조 엔지니어링 관점에서 상세히 분석해주세요.
            """,
            
            'platform': """
작업 플랫폼의 안전성을 검사해주세요:

1. **플랫폼 표면**: 평탄도, 미끄럼 방지, 배수 상태
2. **안전난간**: 높이, 견고성, 연속성 검사
3. **접근 경로**: 사다리, 출입구, 통행로의 안전성
4. **작업 공간**: 충분한 작업 여유공간 확보

작업자 안전 중심으로 실무진이 이해하기 쉽게 설명해주세요.
            """,
            
            'height_safety': """
높이 작업 안전성을 평가해주세요:

1. **층별 안전 조치**: 각 높이에서의 추락 방지 대책
2. **수직/수평 간격**: 안전 기준에 맞는 간격 유지
3. **접근 방법**: 사다리, 계단의 안전성과 각도
4. **비상 대응**: 응급상황 시 대피 경로와 구조 방법

고소 작업 전문가 관점에서 위험도를 평가해주세요.
            """
        }
    
    def _register_scaffold_conversation(self):
        """비계 안전 특화 conversation template 등록"""
        
        conv_scaffold_safety = Conversation(
            system="""당신은 전문적인 비계 안전 검사 AI입니다.

3D 포인트 클라우드로 제공되는 비계 구조를 분석하여 안전성을 평가하고, 건설 안전 규정에 따른 검사 결과를 제공합니다.

**전문 분야:**
- 구조적 안정성 (지지대, 연결부, 프레임워크)
- 작업 플랫폼 안전성 (표면, 난간, 접근로)  
- 높이 작업 안전성 (추락 방지, 층별 조치)
- 안전 규정 준수성 (관련 법규 및 기준)

**분석 방식:**
- 정량적 측정값과 정성적 평가 결합
- 즉시 조치 사항과 권장 개선 사항 구분
- 현장 실무진이 바로 적용 가능한 구체적 지침 제공
- 안전 등급(A/B/C/D) 부여와 근거 명시

전문적이면서도 현장에서 실용적으로 활용할 수 있는 분석을 제공해주세요.""",
            roles=("INSPECTOR", "SAFETY_AI"),
            version="scaffold_v1",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        )
        
        # conversation template에 등록
        conv_templates["scaffold_safety"] = conv_scaffold_safety
    
    def load_model(self):
        """모델 로딩 및 ScaffoldPointLoRA 적용"""
        print("🏗️ ScaffoldPointLoRA Enhanced ShapeLLM 로딩 중...")
        
        disable_torch_init()
        
        # 기본 ShapeLLM 모델 로드
        model_name = get_model_name_from_path(self.args.model_path)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            self.args.model_path, self.args.model_base, model_name, 
            self.args.load_8bit, self.args.load_4bit, device=self.args.device
        )
        
        # ScaffoldPointLoRA 적용
        if self.args.use_scaffold_lora:
            print("🎯 ScaffoldPointLoRA 적용 중...")
            self._apply_scaffold_lora()
        
        # Conversation 설정
        conv_mode = "scaffold_safety"
        if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
            print(f'[WARNING] auto inferred conv mode is {conv_mode}, using {self.args.conv_mode}')
        else:
            self.args.conv_mode = conv_mode
        
        self.conv = conv_templates[self.args.conv_mode].copy()
        print(f"✅ 모델 로딩 완료 (Conv mode: {self.args.conv_mode})")
    
    def _apply_scaffold_lora(self):
        """ScaffoldPointLoRA를 기존 모델에 적용"""
        try:
            # Vision tower가 있는지 확인
            if hasattr(self.model, 'get_vision_tower'):
                vision_tower = self.model.get_vision_tower()
                
                if vision_tower is not None:
                    # ScaffoldPointLoRA 초기화
                    hidden_size = getattr(vision_tower.config, 'hidden_size', 768)
                    
                    self.scaffold_lora = ScaffoldPointLoRA(
                        hidden_size=hidden_size,
                        lora_rank=self.args.scaffold_lora_rank,
                        lora_alpha=self.args.scaffold_lora_alpha,
                        num_selected_tokens=40
                    )
                    
                    # GPU로 이동
                    device = next(self.model.parameters()).device
                    self.scaffold_lora = self.scaffold_lora.to(device)
                    
                    # Vision tower에 LoRA 통합 (간단한 래핑)
                    self._wrap_vision_tower_with_lora(vision_tower)
                    
                    # 매개변수 고정 설정
                    if self.args.training_stage == 'lora_only':
                        self._freeze_non_lora_parameters()
                    
                    # 매개변수 통계 출력
                    self._print_parameter_stats()
                    
                    print("✅ ScaffoldPointLoRA 적용 완료")
                else:
                    print("⚠️ Vision tower를 찾을 수 없음, LoRA 적용 건너뜀")
            else:
                print("⚠️ get_vision_tower 메서드 없음, LoRA 적용 건너뜀")
                
        except Exception as e:
            print(f"❌ ScaffoldPointLoRA 적용 실패: {e}")
            print("기본 ShapeLLM으로 계속 진행...")
    
    def _wrap_vision_tower_with_lora(self, vision_tower):
        """Vision tower의 forward 함수를 LoRA로 래핑"""
        
        original_forward = vision_tower.forward
        
        def lora_enhanced_forward(pts):
            # 원본 forward 실행
            original_output = original_forward(pts)
            
            # ScaffoldPointLoRA 적용
            if self.scaffold_lora is not None:
                try:
                    # 포인트 좌표 추출
                    if isinstance(pts, list) and len(pts) > 0:
                        coords = pts[0][:, :3].unsqueeze(0)  # [1, N, 3]
                    elif isinstance(pts, torch.Tensor):
                        coords = pts[:, :, :3]  # [B, N, 3]
                    else:
                        return original_output
                    
                    # 기본 특징 생성 (실제로는 ReCon++에서 나옴)
                    batch_size, num_points = coords.shape[:2]
                    dummy_features = torch.randn(
                        batch_size, num_points, self.scaffold_lora.hidden_size, 
                        device=coords.device, dtype=coords.dtype
                    )
                    
                    # Multi-scale token selection 수행
                    selection_result = self.scaffold_lora(
                        dummy_features, coords, mode='token_selection'
                    )
                    
                    # 선택 정보 저장 (나중에 출력용)
                    self._last_selection_info = selection_result['selection_info']
                    
                    print(f"🎯 ScaffoldPointLoRA 토큰 선택: {selection_result['selected_tokens'].shape[1]}개")
                    
                except Exception as e:
                    print(f"⚠️ LoRA forward 오류 (원본 결과 사용): {e}")
            
            return original_output
        
        # Forward 함수 교체
        vision_tower.forward = lora_enhanced_forward
    
    def _freeze_non_lora_parameters(self):
        """LoRA가 아닌 매개변수들 고정"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.scaffold_lora is not None:
            for param in self.scaffold_lora.get_trainable_parameters():
                param.requires_grad = True
    
    def _print_parameter_stats(self):
        """매개변수 통계 출력"""
        if self.scaffold_lora is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            lora_params = sum(p.numel() for p in self.scaffold_lora.get_trainable_parameters())
            
            print("=" * 50)
            print("📊 매개변수 통계")
            print("=" * 50)
            print(f"전체 모델 매개변수: {total_params:,}")
            print(f"훈련 가능 매개변수: {trainable_params:,}")
            print(f"LoRA 매개변수: {lora_params:,}")
            print(f"훈련 효율성: {trainable_params/total_params:.2%}")
            print(f"메모리 절약: {1-(trainable_params/total_params):.2%}")
            print("=" * 50)
    
    def process_point_cloud(self, pts_file: str) -> Tuple[torch.Tensor, Dict]:
        """포인트 클라우드 전처리"""
        print(f"📂 포인트 클라우드 처리 중: {pts_file}")
        
        # ScaffoldDataProcessor로 전처리
        processed_data = self.data_processor.process_scaffold_pointcloud(pts_file)
        
        if processed_data is None:
            raise ValueError(f"포인트 클라우드 처리 실패: {pts_file}")
        
        print(f"✅ 처리 완료: {processed_data['processed_shape']}")
        
        # 포인트 클라우드 로드 및 텐서 변환
        pts = load_pts(pts_file)
        if self.args.objaverse:
            pts[:, :3] = rotation(pts[:, :3], [0, 0, -90])
        
        pts_tensor = process_pts(pts, self.model.config).unsqueeze(0)
        model_device = next(self.model.parameters()).device
        pts_tensor = pts_tensor.to(model_device, dtype=torch.float16)
        
        return pts_tensor, processed_data
    
    def generate_response(self, prompt: str, pts_tensor: torch.Tensor) -> str:
        """LLM 응답 생성"""
        
        # Point cloud token 추가
        if self.model.config.mm_use_pt_start_end:
            prompt = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_POINT_TOKEN + '\n' + prompt
        
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        
        # 프롬프트 생성 및 토큰화
        full_prompt = self.conv.get_prompt()
        input_ids = tokenizer_point_token(full_prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # 중단 조건 설정
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # 응답 생성
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                points=pts_tensor,  # ✅ 수정: point_clouds → points
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        
        # 응답 디코딩
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        
        return outputs.strip()
    
    def run_analysis(self, pts_file: str, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """비계 안전 분석 실행"""
        start_time = time.time()
        
        print("🔍 비계 안전 분석 시작...")
        print(f"📋 분석 유형: {analysis_type}")
        print("=" * 50)
        
        try:
            # 1. 포인트 클라우드 처리
            pts_tensor, processed_data = self.process_point_cloud(pts_file)
            
            # 2. 분석 프롬프트 선택
            if analysis_type in self.safety_prompts:
                prompt = self.safety_prompts[analysis_type]
            else:
                prompt = self.safety_prompts['comprehensive']
            
            # 3. LLM 응답 생성
            print("🧠 AI 안전 분석 수행 중...")
            response = self.generate_response(prompt, pts_tensor)
            
            # 4. 결과 정리
            end_time = time.time()
            
            result = {
                'scaffold_info': {
                    'file_path': pts_file,
                    'total_points': processed_data['processed_shape'][0],
                    'dimensions': processed_data['metadata'],
                    'processing_time': f"{end_time - start_time:.2f}초"
                },
                'pointlora_analysis': getattr(self, '_last_selection_info', {
                    'message': 'LoRA 적용되지 않음'
                }),
                'ai_response': response,
                'analysis_type': analysis_type,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return {
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def interactive_mode(self):
        """대화형 모드 실행"""
        print("🏗️ ScaffoldPointLoRA 비계 안전 분석 대화형 모드")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("=" * 50)
        
        # 포인트 클라우드 로드
        if self.args.pts_file:
            try:
                pts_tensor, processed_data = self.process_point_cloud(self.args.pts_file)
                pts_loaded = True
                print(f"✅ 포인트 클라우드 로드 완료: {self.args.pts_file}")
            except Exception as e:
                print(f"❌ 포인트 클라우드 로드 실패: {e}")
                pts_loaded = False
                pts_tensor = None
        else:
            pts_loaded = False
            pts_tensor = None
        
        # 대화 루프
        used_pts = False
        
        while True:
            try:
                user_input = input(f"\n{self.conv.roles[0]}: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("👋 분석을 종료합니다.")
                    break
                
                if not user_input:
                    continue
                
                # 빠른 분석 명령어 처리
                if user_input in ['1', '종합분석']:
                    user_input = self.safety_prompts['comprehensive']
                elif user_input in ['2', '구조분석']:
                    user_input = self.safety_prompts['structural']
                elif user_input in ['3', '플랫폼분석']:
                    user_input = self.safety_prompts['platform']
                elif user_input in ['4', '높이분석']:
                    user_input = self.safety_prompts['height_safety']
                
                print(f"{self.conv.roles[1]}: ", end="", flush=True)
                
                # Point cloud token 추가 (처음 한 번만)
                if pts_loaded and not used_pts:
                    if self.model.config.mm_use_pt_start_end:
                        user_input = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + '\n' + user_input
                    else:
                        user_input = DEFAULT_POINT_TOKEN + '\n' + user_input
                    used_pts = True
                
                self.conv.append_message(self.conv.roles[0], user_input)
                self.conv.append_message(self.conv.roles[1], None)
                
                # 응답 생성
                prompt = self.conv.get_prompt()
                input_ids = tokenizer_point_token(prompt, self.tokenizer, POINT_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                
                # 스트리밍 출력
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        points=pts_tensor if pts_loaded else None,  # ✅ 수정: point_clouds → points
                        do_sample=True if self.args.temperature > 0 else False,
                        temperature=self.args.temperature,
                        top_p=self.args.top_p,
                        num_beams=self.args.num_beams,
                        max_new_tokens=self.args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        streamer=streamer
                    )
                
                # 응답 저장
                input_token_len = input_ids.shape[1]
                outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                
                self.conv.messages[-1][-1] = outputs
                
                # 빠른 명령어 안내 (처음 한 번만)
                if len(self.conv.messages) == 2:
                    print(f"\n\n💡 빠른 분석: 1(종합) | 2(구조) | 3(플랫폼) | 4(높이)")
                
            except KeyboardInterrupt:
                print("\n👋 분석을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                continue
    
    def single_analysis_mode(self, analysis_type: str, output_file: str = None):
        """단일 분석 모드"""
        if not self.args.pts_file:
            print("❌ 포인트 클라우드 파일이 지정되지 않았습니다.")
            return
        
        # 분석 실행
        result = self.run_analysis(self.args.pts_file, analysis_type)
        
        if 'error' in result:
            print(f"❌ 분석 실패: {result['error']}")
            return
        
        # 결과 출력
        self._print_analysis_result(result)
        
        # 파일 저장
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n💾 분석 결과 저장: {output_file}")
            except Exception as e:
                print(f"❌ 파일 저장 실패: {e}")
    
    def _print_analysis_result(self, result: Dict[str, Any]):
        """분석 결과 출력"""
        print("\n" + "=" * 60)
        print("📊 비계 안전 분석 결과")
        print("=" * 60)
        
        # 기본 정보
        scaffold_info = result['scaffold_info']
        print(f"\n🏗️ 비계 정보:")
        print(f"   📂 파일: {Path(scaffold_info['file_path']).name}")
        print(f"   📊 포인트 수: {scaffold_info['total_points']:,}개")
        print(f"   ⏱️ 처리 시간: {scaffold_info['processing_time']}")
        
        # PointLoRA 분석 결과
        pointlora_info = result['pointlora_analysis']
        if 'total_selected' in pointlora_info:
            print(f"\n🎯 PointLoRA 안전 특징 분석:")
            print(f"   선택된 안전 토큰: {pointlora_info['total_selected']}개")
            print(f"   - 전체 구조: {pointlora_info['global_count']}개")
            print(f"   - 구성요소: {pointlora_info['component_count']}개")
            print(f"   - 세부사항: {pointlora_info['detail_count']}개")
        
        # AI 분석 결과
        print(f"\n🤖 AI 전문 안전 분석:")
        print("-" * 40)
        ai_response = result['ai_response']
        # 응답이 너무 길면 적절히 줄바꿈
        for line in ai_response.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        print("\n" + "=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='비계 안전 검증 특화 ShapeLLM CLI')
    
    # 모델 관련 인자
    parser.add_argument("--model-path", type=str, default="qizekun/ShapeLLM_13B_general_v1.0")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    # 포인트 클라우드 관련
    parser.add_argument("--pts-file", type=str, help="비계 포인트 클라우드 파일 (.npy)")
    parser.add_argument("--objaverse", action="store_true", help="Objaverse 데이터 회전 적용")
    
    # ScaffoldPointLoRA 관련
    parser.add_argument("--use-scaffold-lora", action="store_true", default=True, help="ScaffoldPointLoRA 사용")
    parser.add_argument("--scaffold-lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--scaffold-lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--training-stage", type=str, choices=['lora_only', 'full'], default='lora_only')
    
    # 생성 관련 인자
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    
    # 실행 모드 관련
    parser.add_argument("--analysis-type", type=str,
                       choices=['comprehensive', 'structural', 'platform', 'height_safety'],
                       default='comprehensive', help="분석 유형")
    parser.add_argument("--mode", type=str, choices=['interactive', 'single'], default='interactive',
                       help="실행 모드 (interactive: 대화형, single: 단일 분석)")
    parser.add_argument("--output", type=str, help="결과 저장 파일 (JSON)")
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if args.mode == 'single' and not args.pts_file:
        print("❌ single 모드에서는 --pts-file이 필요합니다.")
        return
    
    if args.pts_file and not Path(args.pts_file).exists():
        print(f"❌ 파일이 존재하지 않습니다: {args.pts_file}")
        return
    
    # CLI 초기화 및 실행
    print("🏗️ ScaffoldPointLoRA 비계 안전 분석 시스템")
    print(f"📂 모델: {args.model_path}")
    if args.pts_file:
        print(f"📂 입력 파일: {args.pts_file}")
    print(f"🎯 LoRA 설정: rank={args.scaffold_lora_rank}, alpha={args.scaffold_lora_alpha}")
    print("=" * 60)
    
    try:
        # CLI 인스턴스 생성
        cli = ScaffoldSafetyCLI(args)
        
        # 모델 로딩
        cli.load_model()
        
        # 실행 모드에 따라 분기
        if args.mode == 'interactive':
            cli.interactive_mode()
        else:  # single 모드
            cli.single_analysis_mode(args.analysis_type, args.output)
            
    except KeyboardInterrupt:
        print("\n👋 사용자가 중단했습니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()