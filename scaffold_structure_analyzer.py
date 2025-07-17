"""
scaffold_structure_analyzer.py

포인트 클라우드 데이터 기반 비계 구조 분석 시스템
실제 3D 좌표를 분석하여 구체적인 안전 문제점 식별
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json

class ScaffoldStructureAnalyzer:
    """
    포인트 클라우드 데이터 기반 비계 구조 분석기
    실제 3D 좌표를 분석하여 구체적인 안전 문제점 식별
    """
    
    def __init__(self, safety_standards: Dict[str, float] = None):
        self.safety_standards = safety_standards or {
            'max_platform_spacing': 2.0,  # 최대 플랫폼 간격 (m)
            'max_vertical_spacing': 2.0,   # 최대 수직 간격 (m)
            'min_support_density': 0.5,    # 최소 지지대 밀도 (supports/m²)
            'max_cantilever_length': 1.2,  # 최대 돌출 길이 (m)
            'min_railing_height': 1.0,     # 최소 난간 높이 (m)
            'max_deflection_ratio': 1/300, # 최대 처짐 비율
            'min_cross_bracing_angle': 30, # 최소 교차 브레이싱 각도 (도)
        }
        
        # 분석 결과 저장
        self.analysis_results = {}
        
    def analyze_scaffold_structure(self, points: np.ndarray) -> Dict[str, Any]:
        """
        포인트 클라우드 기반 비계 구조 종합 분석
        
        Args:
            points: numpy array of shape (N, 3) - 3D 좌표 [x, y, z]
            
        Returns:
            Dict containing detailed structural analysis
        """
        print(f"🔍 Analyzing scaffold structure with {len(points)} points...")
        
        # 1. 구조 요소 분리
        structural_components = self._identify_structural_components(points)
        
        # 2. 각 구조 요소별 분석
        platform_analysis = self._analyze_platforms(structural_components.get('platforms', []))
        support_analysis = self._analyze_supports(structural_components.get('supports', []))
        bracing_analysis = self._analyze_bracing(structural_components.get('bracing', []))
        railing_analysis = self._analyze_railings(structural_components.get('railings', []))
        
        # 3. 종합 안전성 평가
        safety_assessment = self._comprehensive_safety_assessment(
            platform_analysis, support_analysis, bracing_analysis, railing_analysis
        )
        
        # 4. 구체적인 문제점 식별
        specific_issues = self._identify_specific_issues(points, structural_components)
        
        return {
            'structural_components': structural_components,
            'platform_analysis': platform_analysis,
            'support_analysis': support_analysis,
            'bracing_analysis': bracing_analysis,
            'railing_analysis': railing_analysis,
            'safety_assessment': safety_assessment,
            'specific_issues': specific_issues,
            'recommendations': self._generate_recommendations(specific_issues)
        }
    
    def _identify_structural_components(self, points: np.ndarray) -> Dict[str, List[Dict]]:
        """
        포인트 클라우드에서 구조 요소 분리
        """
        components = {
            'platforms': [],
            'supports': [],
            'bracing': [],
            'railings': []
        }
        
        # Height-based clustering for initial separation
        height_clusters = self._cluster_by_height(points)
        
        for height_level, cluster_points in height_clusters.items():
            # 각 높이에서 구조 요소 식별
            level_components = self._identify_components_at_level(cluster_points, height_level)
            
            for comp_type, comp_list in level_components.items():
                components[comp_type].extend(comp_list)
        
        return components
    
    def _cluster_by_height(self, points: np.ndarray) -> Dict[float, np.ndarray]:
        """높이 기반 클러스터링"""
        height_clusters = {}
        
        # Z 좌표 기준으로 클러스터링
        z_coords = points[:, 2]
        
        # DBSCAN으로 높이별 클러스터링
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(z_coords.reshape(-1, 1))
        
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # 노이즈 제외
                cluster_mask = clustering.labels_ == cluster_id
                cluster_points = points[cluster_mask]
                avg_height = np.mean(cluster_points[:, 2])
                height_clusters[avg_height] = cluster_points
        
        return height_clusters
    
    def _identify_components_at_level(self, points: np.ndarray, height: float) -> Dict[str, List[Dict]]:
        """특정 높이에서 구조 요소 식별"""
        components = {
            'platforms': [],
            'supports': [],
            'bracing': [],
            'railings': []
        }
        
        # 수평 구조 (플랫폼) 식별
        horizontal_points = self._find_horizontal_structures(points)
        if len(horizontal_points) > 100:  # 충분한 포인트가 있는 경우
            platform_info = {
                'points': horizontal_points,
                'height': height,
                'bbox': self._get_bounding_box(horizontal_points),
                'center': np.mean(horizontal_points, axis=0),
                'area': self._calculate_platform_area(horizontal_points)
            }
            components['platforms'].append(platform_info)
        
        # 수직 구조 (지지대) 식별
        vertical_points = self._find_vertical_structures(points)
        if len(vertical_points) > 50:
            support_info = {
                'points': vertical_points,
                'height': height,
                'bbox': self._get_bounding_box(vertical_points),
                'center': np.mean(vertical_points, axis=0),
                'length': self._calculate_support_length(vertical_points)
            }
            components['supports'].append(support_info)
        
        # 경사 구조 (브레이싱) 식별
        diagonal_points = self._find_diagonal_structures(points)
        if len(diagonal_points) > 20:
            bracing_info = {
                'points': diagonal_points,
                'height': height,
                'bbox': self._get_bounding_box(diagonal_points),
                'angle': self._calculate_bracing_angle(diagonal_points)
            }
            components['bracing'].append(bracing_info)
        
        return components
    
    def _find_horizontal_structures(self, points: np.ndarray) -> np.ndarray:
        """수평 구조 식별 (플랫폼)"""
        # Z 좌표의 분산이 작은 점들을 수평 구조로 판단
        z_std = np.std(points[:, 2])
        if z_std < 0.1:  # 10cm 이하의 높이 차이
            return points
        
        # 국소적으로 수평한 영역 찾기
        horizontal_mask = np.abs(points[:, 2] - np.median(points[:, 2])) < 0.1
        return points[horizontal_mask]
    
    def _find_vertical_structures(self, points: np.ndarray) -> np.ndarray:
        """수직 구조 식별 (지지대)"""
        # X, Y 좌표의 분산이 작고 Z 좌표의 분산이 큰 점들
        xy_std = np.std(points[:, :2], axis=0)
        z_std = np.std(points[:, 2])
        
        # 수직 구조의 조건: xy 분산 < 임계값, z 분산 > 임계값
        if np.all(xy_std < 0.2) and z_std > 0.5:
            return points
        
        return np.array([])
    
    def _find_diagonal_structures(self, points: np.ndarray) -> np.ndarray:
        """대각선 구조 식별 (브레이싱)"""
        # 점들의 주성분 분석으로 대각선 구조 식별
        if len(points) < 10:
            return np.array([])
        
        # PCA로 주 방향 분석
        centered_points = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 가장 큰 고유벡터의 Z 성분이 중간 정도인 경우 대각선 구조
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        z_component = abs(main_direction[2])
        
        if 0.3 < z_component < 0.7:  # 대각선 구조의 조건
            return points
        
        return np.array([])
    
    def _analyze_platforms(self, platforms: List[Dict]) -> Dict[str, Any]:
        """플랫폼 분석"""
        analysis = {
            'total_platforms': len(platforms),
            'platform_details': [],
            'spacing_violations': [],
            'area_issues': [],
            'elevation_data': []
        }
        
        for i, platform in enumerate(platforms):
            platform_detail = {
                'id': i,
                'height': platform['height'],
                'area': platform['area'],
                'center': platform['center'].tolist(),
                'bbox': platform['bbox'],
                'safety_status': 'SAFE'
            }
            
            # 플랫폼 간격 검사
            if i > 0:
                prev_platform = platforms[i-1]
                spacing = abs(platform['height'] - prev_platform['height'])
                if spacing > self.safety_standards['max_platform_spacing']:
                    violation = {
                        'type': 'EXCESSIVE_SPACING',
                        'platforms': [i-1, i],
                        'measured_spacing': spacing,
                        'max_allowed': self.safety_standards['max_platform_spacing'],
                        'coordinates': {
                            'platform1': prev_platform['center'].tolist(),
                            'platform2': platform['center'].tolist()
                        }
                    }
                    analysis['spacing_violations'].append(violation)
                    platform_detail['safety_status'] = 'VIOLATION'
            
            # 플랫폼 면적 검사
            if platform['area'] < 1.0:  # 최소 1m² 필요
                area_issue = {
                    'platform_id': i,
                    'measured_area': platform['area'],
                    'min_required': 1.0,
                    'coordinates': platform['center'].tolist()
                }
                analysis['area_issues'].append(area_issue)
                platform_detail['safety_status'] = 'WARNING'
            
            analysis['platform_details'].append(platform_detail)
            analysis['elevation_data'].append({
                'height': platform['height'],
                'area': platform['area']
            })
        
        return analysis
    
    def _analyze_supports(self, supports: List[Dict]) -> Dict[str, Any]:
        """지지대 분석"""
        analysis = {
            'total_supports': len(supports),
            'support_details': [],
            'density_issues': [],
            'stability_concerns': []
        }
        
        if not supports:
            return analysis
        
        # 지지대 밀도 계산
        support_positions = np.array([s['center'][:2] for s in supports])  # X, Y 좌표만
        
        # 전체 영역 계산
        if len(support_positions) > 3:
            hull = ConvexHull(support_positions)
            total_area = hull.volume  # 2D에서는 volume이 면적
            density = len(supports) / total_area
            
            if density < self.safety_standards['min_support_density']:
                density_issue = {
                    'measured_density': density,
                    'min_required': self.safety_standards['min_support_density'],
                    'total_area': total_area,
                    'support_count': len(supports)
                }
                analysis['density_issues'].append(density_issue)
        
        # 각 지지대 분석
        for i, support in enumerate(supports):
            support_detail = {
                'id': i,
                'center': support['center'].tolist(),
                'height': support['height'],
                'length': support['length'],
                'bbox': support['bbox'],
                'safety_status': 'SAFE'
            }
            
            # 지지대 길이 검사
            if support['length'] > 6.0:  # 6m 이상은 중간 지지 필요
                stability_concern = {
                    'support_id': i,
                    'length': support['length'],
                    'max_unsupported': 6.0,
                    'coordinates': support['center'].tolist(),
                    'recommendation': 'Add intermediate support'
                }
                analysis['stability_concerns'].append(stability_concern)
                support_detail['safety_status'] = 'WARNING'
            
            analysis['support_details'].append(support_detail)
        
        return analysis
    
    def _analyze_bracing(self, bracing: List[Dict]) -> Dict[str, Any]:
        """브레이싱 분석"""
        analysis = {
            'total_bracing': len(bracing),
            'bracing_details': [],
            'angle_violations': [],
            'missing_bracing': []
        }
        
        for i, brace in enumerate(bracing):
            brace_detail = {
                'id': i,
                'angle': brace['angle'],
                'height': brace['height'],
                'bbox': brace['bbox'],
                'safety_status': 'SAFE'
            }
            
            # 브레이싱 각도 검사
            if brace['angle'] < self.safety_standards['min_cross_bracing_angle']:
                angle_violation = {
                    'bracing_id': i,
                    'measured_angle': brace['angle'],
                    'min_required': self.safety_standards['min_cross_bracing_angle'],
                    'coordinates': brace['bbox']
                }
                analysis['angle_violations'].append(angle_violation)
                brace_detail['safety_status'] = 'VIOLATION'
            
            analysis['bracing_details'].append(brace_detail)
        
        return analysis
    
    def _identify_specific_issues(self, points: np.ndarray, components: Dict) -> List[Dict]:
        """구체적인 문제점 식별"""
        issues = []
        
        # 1. 구조적 불안정성
        structural_issues = self._check_structural_stability(points, components)
        issues.extend(structural_issues)
        
        # 2. 규정 위반 사항
        code_violations = self._check_code_compliance(components)
        issues.extend(code_violations)
        
        # 3. 안전 위험 요소
        safety_hazards = self._identify_safety_hazards(points, components)
        issues.extend(safety_hazards)
        
        return issues
    
    def _check_structural_stability(self, points: np.ndarray, components: Dict) -> List[Dict]:
        """구조적 안정성 검사"""
        issues = []
        
        # 처짐 검사
        deflection_issues = self._check_deflection(points)
        issues.extend(deflection_issues)
        
        # 하중 분산 검사
        load_distribution_issues = self._check_load_distribution(components)
        issues.extend(load_distribution_issues)
        
        return issues
    
    def _check_deflection(self, points: np.ndarray) -> List[Dict]:
        """처짐 검사"""
        issues = []
        
        # 수평 요소들의 처짐 검사
        horizontal_points = self._find_horizontal_structures(points)
        if len(horizontal_points) > 10:
            # 중앙부와 끝부분의 높이 차이 측정
            x_coords = horizontal_points[:, 0]
            z_coords = horizontal_points[:, 2]
            
            # 스팬 계산
            span = np.max(x_coords) - np.min(x_coords)
            
            if span > 2.0:  # 2m 이상의 스팬
                # 중앙부 처짐 측정
                center_x = (np.max(x_coords) + np.min(x_coords)) / 2
                center_mask = np.abs(x_coords - center_x) < 0.2
                
                if np.any(center_mask):
                    center_z = np.mean(z_coords[center_mask])
                    edge_z = np.mean([np.mean(z_coords[x_coords < center_x - 0.5]), 
                                     np.mean(z_coords[x_coords > center_x + 0.5])])
                    
                    deflection = edge_z - center_z
                    max_allowed_deflection = span * self.safety_standards['max_deflection_ratio']
                    
                    if deflection > max_allowed_deflection:
                        issue = {
                            'type': 'EXCESSIVE_DEFLECTION',
                            'severity': 'HIGH',
                            'location': [center_x, np.mean(horizontal_points[:, 1]), center_z],
                            'measured_deflection': deflection,
                            'max_allowed': max_allowed_deflection,
                            'span': span,
                            'description': f"Platform deflection of {deflection:.3f}m exceeds limit at ({center_x:.2f}, {center_z:.2f})"
                        }
                        issues.append(issue)
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[Dict]:
        """구체적인 개선 권장사항 생성"""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'EXCESSIVE_DEFLECTION':
                rec = {
                    'priority': 'HIGH',
                    'action': 'Add intermediate support',
                    'location': issue['location'],
                    'description': f"Install vertical support at coordinates ({issue['location'][0]:.2f}, {issue['location'][1]:.2f}, {issue['location'][2]:.2f})",
                    'estimated_cost': 'Medium',
                    'timeline': '1-2 days'
                }
                recommendations.append(rec)
            
            elif issue['type'] == 'EXCESSIVE_SPACING':
                rec = {
                    'priority': 'MEDIUM',
                    'action': 'Reduce platform spacing',
                    'location': issue.get('coordinates', {}).get('platform1', []),
                    'description': f"Add intermediate platform between levels {issue['measured_spacing']:.1f}m apart",
                    'estimated_cost': 'High',
                    'timeline': '2-3 days'
                }
                recommendations.append(rec)
        
        return recommendations
    
    # 유틸리티 메서드들
    def _get_bounding_box(self, points: np.ndarray) -> List[float]:
        """바운딩 박스 계산"""
        if len(points) == 0:
            return [0, 0, 0, 0, 0, 0]
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        return [min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2]]
    
    def _calculate_platform_area(self, points: np.ndarray) -> float:
        """플랫폼 면적 계산"""
        if len(points) < 3:
            return 0.0
        
        # 2D 프로젝션으로 면적 계산
        xy_points = points[:, :2]
        try:
            hull = ConvexHull(xy_points)
            return hull.volume  # 2D에서는 volume이 면적
        except:
            return 0.0
    
    def _calculate_support_length(self, points: np.ndarray) -> float:
        """지지대 길이 계산"""
        if len(points) < 2:
            return 0.0
        
        z_coords = points[:, 2]
        return np.max(z_coords) - np.min(z_coords)
    
    def _calculate_bracing_angle(self, points: np.ndarray) -> float:
        """브레이싱 각도 계산"""
        if len(points) < 2:
            return 0.0
        
        # 점들의 주 방향 벡터 계산
        if len(points) >= 3:
            centered_points = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # 수평면과의 각도 계산
            horizontal_projection = np.array([main_direction[0], main_direction[1], 0])
            if np.linalg.norm(horizontal_projection) > 0:
                horizontal_projection = horizontal_projection / np.linalg.norm(horizontal_projection)
                angle = np.arccos(np.dot(main_direction, horizontal_projection)) * 180 / np.pi
                return min(angle, 180 - angle)
        
        return 0.0
    
    def _comprehensive_safety_assessment(self, platform_analysis: Dict, support_analysis: Dict, 
                                       bracing_analysis: Dict, railing_analysis: Dict) -> Dict:
        """종합 안전성 평가"""
        total_violations = (
            len(platform_analysis.get('spacing_violations', [])) +
            len(support_analysis.get('density_issues', [])) +
            len(bracing_analysis.get('angle_violations', []))
        )
        
        if total_violations == 0:
            grade = 'A'
            status = 'SAFE'
        elif total_violations <= 2:
            grade = 'B'
            status = 'CAUTION'
        elif total_violations <= 4:
            grade = 'C'
            status = 'WARNING'
        else:
            grade = 'D'
            status = 'DANGEROUS'
        
        return {
            'overall_grade': grade,
            'safety_status': status,
            'total_violations': total_violations,
            'compliance_rate': max(0, 100 - total_violations * 10)
        }
    
    def _check_code_compliance(self, components: Dict) -> List[Dict]:
        """규정 준수 검사"""
        issues = []
        # 구현 예정
        return issues
    
    def _identify_safety_hazards(self, points: np.ndarray, components: Dict) -> List[Dict]:
        """안전 위험 요소 식별"""
        issues = []
        # 구현 예정
        return issues
    
    def _check_load_distribution(self, components: Dict) -> List[Dict]:
        """하중 분산 검사"""
        issues = []
        # 구현 예정
        return issues


def format_analysis_for_llm(analysis_result: Dict) -> str:
    """
    분석 결과를 LLM이 이해할 수 있는 형태로 포맷팅
    """
    formatted_text = f"""
SCAFFOLD STRUCTURE ANALYSIS REPORT

STRUCTURAL COMPONENTS IDENTIFIED:
- Platforms: {analysis_result['structural_components']['platforms'].__len__()} detected
- Supports: {analysis_result['structural_components']['supports'].__len__()} detected
- Bracing: {analysis_result['structural_components']['bracing'].__len__()} detected

SAFETY ASSESSMENT:
- Overall Grade: {analysis_result['safety_assessment']['overall_grade']}
- Safety Status: {analysis_result['safety_assessment']['safety_status']}
- Compliance Rate: {analysis_result['safety_assessment']['compliance_rate']:.1f}%

SPECIFIC ISSUES IDENTIFIED:
"""
    
    for i, issue in enumerate(analysis_result['specific_issues']):
        formatted_text += f"""
Issue {i+1}: {issue['type']}
- Location: ({issue['location'][0]:.2f}, {issue['location'][1]:.2f}, {issue['location'][2]:.2f})
- Severity: {issue['severity']}
- Description: {issue['description']}
"""
    
    formatted_text += "\nRECOMMENDATIONS:\n"
    for i, rec in enumerate(analysis_result['recommendations']):
        formatted_text += f"""
{i+1}. {rec['action']} (Priority: {rec['priority']})
   - Location: ({rec['location'][0]:.2f}, {rec['location'][1]:.2f}, {rec['location'][2]:.2f})
   - Description: {rec['description']}
   - Timeline: {rec['timeline']}
"""
    
    return formatted_text


# 테스트 및 사용 예시
if __name__ == "__main__":
    # 테스트 데이터 생성
    test_points = np.random.rand(1000, 3) * 10
    
    # 분석기 초기화
    analyzer = ScaffoldStructureAnalyzer()
    
    # 구조 분석 수행
    result = analyzer.analyze_scaffold_structure(test_points)
    
    # 결과 출력
    formatted_result = format_analysis_for_llm(result)
    print(formatted_result)