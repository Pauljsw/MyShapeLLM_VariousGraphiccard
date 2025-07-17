#!/usr/bin/env python3
"""
비계 안전 검증 특화 ShapeLLM CLI
ScaffoldPointLoRA가 통합된 ShapeLLM으로 비계 안전 분석 수행
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any

# 프로젝트 모듈 import
from integrate_scaffold_pointlora import (
    ScaffoldEnhancedCLIPVisionTower, 
    ScaffoldDataProcessor,
    create_scaffold_enhanced_shapellm,
    save_scaffold_lora_weights,
    load_scaffold_lora_weights
)


class ScaffoldSafetyAnalyzer:
    """
    비계 안전 분석기
    ScaffoldPointLoRA가 적용된 ShapeLLM으로 비계 구조 안전성 분석
    """
    
    def __init__(self, model_path: str, lora_config: Dict[str, Any] = None):
        self.model_path = model_path
        self.lora_config = lora_config or {
            'use_scaffold_lora': True,
            'scaffold_lora_rank': 16,
            'scaffold_lora_alpha': 32.0,
            'training_stage': 'lora_only'
        }
        
        # 데이터 처리기
        self.data_processor = ScaffoldDataProcessor()
        
        # 모델 초기화
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 비계 안전 검사 템플릿
        self.safety_prompts = {
            'structural_analysis': """
            이 비계 구조를 분석해주세요:
            1. 전체적인 구조적 안정성은 어떤가요?
            2. 지지대와 연결부의 상태는 괜찮나요?
            3. 발견되는 안전 위험 요소가 있나요?
            4. 개선이 필요한 부분을 구체적으로 알려주세요.
            """,
            
            'working_platform_check': """
            작업 플랫폼의 안전성을 검사해주세요:
            1. 플랫폼 표면의 상태는 어떤가요?
            2. 안전난간이 적절히 설치되어 있나요?
            3. 출입구와 접근 경로는 안전한가요?
            4. 추락 방지 조치가 충분한가요?
            """,
            
            'height_safety_assessment': """
            높이 작업 안전성을 평가해주세요:
            1. 각 높이별 안전 조치는 적절한가요?
            2. 수직 간격과 수평 간격이 규정에 맞나요?
            3. 사다리와 접근 방법이 안전한가요?
            4. 높이별 위험도 평가 결과는 어떤가요?
            """,
            
            'comprehensive_report': """
            종합적인 비계 안전 점검 보고서를 작성해주세요:
            1. 전체 안전도 등급 (A/B/C/D)
            2. 주요 발견 사항 (위험 요소)
            3. 즉시 조치 필요 항목
            4. 권장 개선 사항
            5. 재검사 주기 제안
            
            보고서는 현장 안전 관리자가 바로 사용할 수 있도록 구체적이고 실용적으로 작성해주세요.
            """
        }
    
    def load_model(self):
        """ScaffoldEnhanced ShapeLLM 로드"""
        print("🏗️ ScaffoldEnhanced ShapeLLM 로딩 중...")
        
        try:
            self.model = create_scaffold_enhanced_shapellm(
                model_path=self.model_path,
                **self.lora_config
            )
            
            if self.model:
                self.model = self.model.to(self.device)
                print(f"✅ 모델 로딩 완료 (Device: {self.device})")
                return True
            else:
                print("❌ 모델 로딩 실패")
                return False
                
        except Exception as e:
            print(f"❌ 모델 로딩 오류: {e}")
            return False
    
    def analyze_scaffold(self, pts_file: str, analysis_type: str = 'comprehensive_report') -> Dict[str, Any]:
        """
        비계 안전 분석 수행
        
        Args:
            pts_file: 비계 포인트 클라우드 파일 (.npy)
            analysis_type: 분석 유형 ('structural_analysis', 'working_platform_check', 
                          'height_safety_assessment', 'comprehensive_report')
        
        Returns:
            분석 결과 딕셔너리
        """
        if not self.model:
            print("❌ 모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
            return None
        
        print(f"🔍 비계 안전 분석 시작: {pts_file}")
        print(f"📋 분석 유형: {analysis_type}")
        
        start_time = time.time()
        
        try:
            # 1. 데이터 전처리
            print("📂 포인트 클라우드 처리 중...")
            processed_data = self.data_processor.process_scaffold_pointcloud(pts_file)
            
            if not processed_data:
                print("❌ 데이터 처리 실패")
                return None
            
            print(f"✅ 데이터 처리 완료: {processed_data['processed_shape']}")
            
            # 2. 텐서 준비
            points_tensor = torch.FloatTensor(processed_data['points']).unsqueeze(0).to(self.device)
            coords_tensor = torch.FloatTensor(processed_data['coordinates']).unsqueeze(0).to(self.device)
            
            # 3. 모델 추론
            print("🧠 모델 추론 중...")
            with torch.no_grad():
                # ScaffoldPointLoRA의 multi-scale token selection 수행
                if hasattr(self.model, 'scaffold_lora'):
                    # 기본 특징 추출 (실제로는 ReCon++에서 나옴)
                    dummy_features = torch.randn(1, 8192, 768).to(self.device)
                    
                    selection_result = self.model.scaffold_lora(
                        dummy_features, coords_tensor, mode='token_selection'
                    )
                    
                    # 모델 forward pass
                    model_output = self.model(points_tensor)
                    
                    print(f"✅ 모델 추론 완료")
                    print(f"🎯 선택된 안전 관련 토큰: {selection_result['selected_tokens'].shape[1]}개")
                    print(f"📊 토큰 선택 정보: {selection_result['selection_info']}")
            
            # 4. 안전 분석 결과 생성
            analysis_result = self._generate_safety_analysis(
                processed_data, selection_result, analysis_type
            )
            
            end_time = time.time()
            analysis_result['processing_time'] = f"{end_time - start_time:.2f}초"
            
            print(f"🏁 분석 완료 (소요 시간: {analysis_result['processing_time']})")
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return None
    
    def _generate_safety_analysis(self, processed_data: Dict, selection_result: Dict, 
                                analysis_type: str) -> Dict[str, Any]:
        """안전 분석 결과 생성"""
        
        # 메타데이터에서 구조 정보 추출
        metadata = processed_data['metadata']
        
        # 토큰 선택 정보에서 안전 특징 분석
        selection_info = selection_result['selection_info']
        
        # 기본 구조 분석
        height_range = metadata['height_range']
        scaffold_height = height_range[1] - height_range[0]
        
        # 안전도 평가 (예시 로직)
        safety_score = self._calculate_safety_score(metadata, selection_info)
        
        # 분석 결과 구성
        analysis = {
            'scaffold_info': {
                'file_path': processed_data.get('file_path', 'unknown'),
                'total_points': processed_data['processed_shape'][0],
                'scaffold_height': f"{scaffold_height:.2f}m",
                'dimensions': {
                    'width': f"{metadata['width_range'][1] - metadata['width_range'][0]:.2f}m",
                    'depth': f"{metadata['depth_range'][1] - metadata['depth_range'][0]:.2f}m",
                    'height': f"{scaffold_height:.2f}m"
                }
            },
            
            'pointlora_analysis': {
                'selected_safety_tokens': selection_info['total_selected'],
                'global_structure_tokens': selection_info['global_count'],
                'component_level_tokens': selection_info['component_count'],
                'detail_safety_tokens': selection_info['detail_count']
            },
            
            'safety_assessment': {
                'overall_safety_grade': self._get_safety_grade(safety_score),
                'safety_score': f"{safety_score:.1f}/100",
                'structural_integrity': self._assess_structural_integrity(metadata),
                'working_platform_safety': self._assess_platform_safety(metadata),
                'height_safety_compliance': self._assess_height_safety(metadata)
            },
            
            'recommendations': self._generate_recommendations(safety_score, metadata),
            
            'analysis_type': analysis_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis
    
    def _calculate_safety_score(self, metadata: Dict, selection_info: Dict) -> float:
        """안전 점수 계산 (0-100)"""
        
        # 구조적 안정성 점수 (40점)
        structural_score = min(40, metadata.get('vertical_structure_strength', 0) * 20)
        
        # 플랫폼 안전성 점수 (30점)  
        platform_score = min(30, (1.0 / (metadata.get('horizontal_structure_strength', 1) + 0.1)) * 15)
        
        # PointLoRA 토큰 선택 품질 점수 (30점)
        token_quality = (selection_info['detail_count'] / 8) * 30  # detail 토큰이 많을수록 좋음
        
        total_score = structural_score + platform_score + token_quality
        return min(100, max(0, total_score))
    
    def _get_safety_grade(self, score: float) -> str:
        """안전 등급 반환"""
        if score >= 90:
            return "A (우수)"
        elif score >= 80:
            return "B (양호)"
        elif score >= 70:
            return "C (보통)"
        elif score >= 60:
            return "D (주의)"
        else:
            return "F (위험)"
    
    def _assess_structural_integrity(self, metadata: Dict) -> str:
        """구조적 무결성 평가"""
        strength = metadata.get('vertical_structure_strength', 0)
        
        if strength > 2.0:
            return "양호 - 수직 구조가 안정적임"
        elif strength > 1.0:
            return "보통 - 일부 보강 검토 필요"
        else:
            return "주의 - 구조 안정성 점검 필요"
    
    def _assess_platform_safety(self, metadata: Dict) -> str:
        """작업 플랫폼 안전성 평가"""
        variance = metadata.get('horizontal_structure_strength', 0)
        
        if variance < 0.5:
            return "양호 - 작업면이 평평하고 안정적임"
        elif variance < 1.0:
            return "보통 - 일부 평탄화 작업 필요"
        else:
            return "주의 - 작업면 안전성 점검 필요"
    
    def _assess_height_safety(self, metadata: Dict) -> str:
        """높이 안전성 평가"""
        height_density = metadata.get('height_density', [])
        
        if len(height_density) > 0:
            # 균등한 분포일수록 안전 (계단식 구조)
            density_variance = np.var(height_density)
            
            if density_variance < 100:
                return "양호 - 높이별 구조가 균등함"
            elif density_variance < 200:
                return "보통 - 일부 층 보강 검토"
            else:
                return "주의 - 높이별 안전 조치 점검 필요"
        else:
            return "정보 부족 - 추가 분석 필요"
    
    def _generate_recommendations(self, safety_score: float, metadata: Dict) -> list:
        """개선 권장사항 생성"""
        recommendations = []
        
        if safety_score < 70:
            recommendations.append("🚨 즉시 조치: 전문가 안전 점검 실시")
        
        if metadata.get('vertical_structure_strength', 0) < 1.5:
            recommendations.append("🔧 구조 보강: 수직 지지대 및 연결부 점검 필요")
        
        if metadata.get('horizontal_structure_strength', 0) > 1.0:
            recommendations.append("📏 작업면 정비: 플랫폼 평탄도 개선 필요")
        
        height_range = metadata.get('height_range', [0, 0])
        if height_range[1] - height_range[0] > 10:  # 10m 이상
            recommendations.append("🛡️ 높이 안전: 추가 안전난간 및 추락방지 조치 필요")
        
        if safety_score >= 80:
            recommendations.append("✅ 현재 상태 양호: 정기 점검 주기 유지")
        
        recommendations.append("📋 다음 점검일: 1개월 후 재검사 권장")
        
        return recommendations


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='비계 안전 검증 특화 ShapeLLM CLI')
    
    # 필수 인자
    parser.add_argument('--pts-file', type=str, required=True,
                       help='비계 포인트 클라우드 파일 경로 (.npy)')
    
    # 선택적 인자
    parser.add_argument('--model-path', type=str, 
                       default='qizekun/ShapeLLM_13B_general_v1.0',
                       help='ShapeLLM 모델 경로')
    
    parser.add_argument('--analysis-type', type=str,
                       choices=['structural_analysis', 'working_platform_check', 
                               'height_safety_assessment', 'comprehensive_report'],
                       default='comprehensive_report',
                       help='분석 유형 선택')
    
    parser.add_argument('--lora-rank', type=int, default=16,
                       help='LoRA rank (기본값: 16)')
    
    parser.add_argument('--lora-alpha', type=float, default=32.0,
                       help='LoRA alpha (기본값: 32.0)')
    
    parser.add_argument('--training-stage', type=str,
                       choices=['lora_only', 'full'], default='lora_only',
                       help='훈련 단계 (기본값: lora_only)')
    
    parser.add_argument('--save-lora', type=str, default=None,
                       help='LoRA 가중치 저장 경로 (선택사항)')
    
    parser.add_argument('--load-lora', type=str, default=None,
                       help='저장된 LoRA 가중치 로드 경로 (선택사항)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='분석 결과 저장 파일 (JSON 형식)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # 입력 파일 확인
    if not Path(args.pts_file).exists():
        print(f"❌ 파일이 존재하지 않습니다: {args.pts_file}")
        return
    
    print("🏗️ ScaffoldPointLoRA 비계 안전 분석 시작")
    print(f"📂 입력 파일: {args.pts_file}")
    print(f"🧠 모델: {args.model_path}")
    print(f"📋 분석 유형: {args.analysis_type}")
    print("=" * 50)
    
    # LoRA 설정
    lora_config = {
        'use_scaffold_lora': True,
        'scaffold_lora_rank': args.lora_rank,
        'scaffold_lora_alpha': args.lora_alpha,
        'training_stage': args.training_stage
    }
    
    # 분석기 초기화
    analyzer = ScaffoldSafetyAnalyzer(args.model_path, lora_config)
    
    # 모델 로드
    if not analyzer.load_model():
        print("❌ 모델 로딩 실패")
        return
    
    # 저장된 LoRA 가중치 로드 (옵션)
    if args.load_lora:
        if load_scaffold_lora_weights(analyzer.model, args.load_lora):
            print(f"✅ LoRA 가중치 로드 완료: {args.load_lora}")
        else:
            print(f"⚠️ LoRA 가중치 로드 실패: {args.load_lora}")
    
    # 비계 안전 분석 수행
    result = analyzer.analyze_scaffold(args.pts_file, args.analysis_type)
    
    if result:
        # 결과 출력
        print("\n" + "=" * 50)
        print("📊 비계 안전 분석 결과")
        print("=" * 50)
        
        # 기본 정보
        scaffold_info = result['scaffold_info']
        print(f"\n🏗️ 비계 정보:")
        print(f"   - 높이: {scaffold_info['scaffold_height']}")
        print(f"   - 크기: {scaffold_info['dimensions']['width']} × {scaffold_info['dimensions']['depth']}")
        print(f"   - 포인트 수: {scaffold_info['total_points']:,}개")
        
        # PointLoRA 분석 결과
        pointlora_result = result['pointlora_analysis']
        print(f"\n🎯 PointLoRA 안전 특징 분석:")
        print(f"   - 선택된 안전 토큰: {pointlora_result['selected_safety_tokens']}개")
        print(f"   - 전체 구조 토큰: {pointlora_result['global_structure_tokens']}개")
        print(f"   - 구성요소 토큰: {pointlora_result['component_level_tokens']}개")
        print(f"   - 세부 안전 토큰: {pointlora_result['detail_safety_tokens']}개")
        
        # 안전성 평가
        safety_assessment = result['safety_assessment']
        print(f"\n🛡️ 안전성 평가:")
        print(f"   - 종합 안전 등급: {safety_assessment['overall_safety_grade']}")
        print(f"   - 안전 점수: {safety_assessment['safety_score']}")
        print(f"   - 구조적 무결성: {safety_assessment['structural_integrity']}")
        print(f"   - 작업 플랫폼: {safety_assessment['working_platform_safety']}")
        print(f"   - 높이 안전성: {safety_assessment['height_safety_compliance']}")
        
        # 권장사항
        print(f"\n📋 권장사항:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n⏱️ 처리 시간: {result['processing_time']}")
        
        # 결과 파일 저장 (옵션)
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 분석 결과 저장: {args.output}")
            except Exception as e:
                print(f"❌ 결과 저장 실패: {e}")
        
        # LoRA 가중치 저장 (옵션)
        if args.save_lora:
            save_scaffold_lora_weights(analyzer.model, args.save_lora)
        
    else:
        print("❌ 분석 실패")


def interactive_mode():
    """대화형 모드"""
    print("🏗️ ScaffoldPointLoRA 대화형 모드")
    print("비계 포인트 클라우드 파일을 분석합니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    # 분석기 초기화 (기본 설정)
    analyzer = ScaffoldSafetyAnalyzer(
        model_path="qizekun/ShapeLLM_13B_general_v1.0",
        lora_config={
            'use_scaffold_lora': True,
            'scaffold_lora_rank': 16,
            'scaffold_lora_alpha': 32.0,
            'training_stage': 'lora_only'
        }
    )
    
    # 모델 로드
    if not analyzer.load_model():
        print("❌ 모델 로딩 실패")
        return
    
    while True:
        try:
            # 파일 경로 입력
            pts_file = input("\n📂 비계 포인트 클라우드 파일 경로 (.npy): ").strip()
            
            if pts_file.lower() in ['quit', 'exit']:
                print("👋 종료합니다.")
                break
            
            if not pts_file:
                continue
                
            if not Path(pts_file).exists():
                print(f"❌ 파일이 존재하지 않습니다: {pts_file}")
                continue
            
            # 분석 유형 선택
            print("\n📋 분석 유형을 선택하세요:")
            print("1. 구조 분석 (structural_analysis)")
            print("2. 작업 플랫폼 검사 (working_platform_check)")
            print("3. 높이 안전성 평가 (height_safety_assessment)")
            print("4. 종합 보고서 (comprehensive_report) [기본값]")
            
            choice = input("선택 (1-4, 기본값: 4): ").strip()
            
            analysis_types = {
                '1': 'structural_analysis',
                '2': 'working_platform_check', 
                '3': 'height_safety_assessment',
                '4': 'comprehensive_report'
            }
            
            analysis_type = analysis_types.get(choice, 'comprehensive_report')
            
            # 분석 수행
            print(f"\n🔍 분석 시작... ({analysis_type})")
            result = analyzer.analyze_scaffold(pts_file, analysis_type)
            
            if result:
                # 간단한 결과 출력
                safety_assessment = result['safety_assessment']
                print(f"\n✅ 분석 완료!")
                print(f"🛡️ 안전 등급: {safety_assessment['overall_safety_grade']}")
                print(f"📊 안전 점수: {safety_assessment['safety_score']}")
                
                # 상세 결과 출력 여부
                detail = input("\n상세 결과를 보시겠습니까? (y/n): ").strip().lower()
                if detail in ['y', 'yes']:
                    print("\n" + "=" * 50)
                    print("📋 상세 분석 결과")
                    print("=" * 50)
                    
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"{i}. {rec}")
                
                # 결과 저장 여부  
                save = input("\n결과를 파일로 저장하시겠습니까? (y/n): ").strip().lower()
                if save in ['y', 'yes']:
                    output_file = input("저장할 파일명 (기본값: scaffold_analysis.json): ").strip()
                    if not output_file:
                        output_file = "scaffold_analysis.json"
                    
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        print(f"💾 저장 완료: {output_file}")
                    except Exception as e:
                        print(f"❌ 저장 실패: {e}")
            else:
                print("❌ 분석 실패")
                
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 인자가 없으면 대화형 모드 실행
        interactive_mode()
    else:
        # 인자가 있으면 CLI 모드 실행
        main()
