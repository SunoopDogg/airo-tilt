import numpy as np
import cv2
from typing import List


class Geometry:
    @staticmethod
    def _prepare_mask(mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return np.zeros((0, 0), dtype=np.uint8)
        if mask.dtype != np.uint8:
            if mask.dtype == bool:
                return mask.astype(np.uint8)
            return (mask > 0).astype(np.uint8)
        return mask

    @staticmethod
    def extract_contour(mask: np.ndarray) -> np.ndarray:
        """마스크에서 최대 면적 외곽 윤곽선 추출"""
        prepared = Geometry._prepare_mask(mask)
        if prepared.size == 0 or prepared.max() == 0:
            return np.array([])
        contours, _ = cv2.findContours(
            prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.reshape(-1, 2)
        return np.array([])

    @staticmethod
    def extract_rectangle_contour(mask: np.ndarray) -> np.ndarray:
        """마스크 윤곽선을 4각형으로 근사화하여 정규화된 꼭짓점 추출"""
        prepared = Geometry._prepare_mask(mask)
        if prepared.size == 0 or prepared.max() == 0:
            return np.array([])
        contours, _ = cv2.findContours(
            prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.array([])
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            vertices = approx.reshape(-1, 2)
            return Geometry.normalize_rectangle_vertices(vertices)
        return np.array([])

    @staticmethod
    def normalize_rectangle_vertices(vertices: np.ndarray) -> np.ndarray:
        """사각형 꼭짓점을 시계방향으로 정렬하여 일반화

        Args:
            vertices: 4개의 꼭짓점 좌표 배열 (4, 2)

        Returns:
            시계방향으로 정렬된 꼭짓점 배열 (왼쪽 상단 -> 오른쪽 상단 -> 오른쪽 하단 -> 왼쪽 하단)
        """
        if len(vertices) != 4:
            return vertices

        # 중복 점 제거 및 고유한 점 확인
        unique_vertices, indices = np.unique(
            vertices, axis=0, return_inverse=True)

        # 고유한 점이 4개 미만인 경우 (중복이 있는 경우)
        if len(unique_vertices) < 4:
            # 중복이 있을 때는 원본 vertices에서 처리
            # 하지만 여전히 정규화 시도
            pass
        else:
            # 고유한 점들로 대체
            vertices = unique_vertices

        # 무게중심(centroid) 계산
        centroid = np.mean(vertices, axis=0)

        # 각 점에서 무게중심까지의 각도 계산 (시계방향 정렬을 위해)
        angles = np.arctan2(
            vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

        # 각도 기준으로 정렬 (시계방향: -π부터 π까지)
        sorted_indices = np.argsort(angles)
        sorted_vertices = vertices[sorted_indices]

        # 시작점을 왼쪽 상단으로 조정
        # x + y 합이 가장 작은 점을 찾아서 배열의 첫 번째로 이동
        sum_coords = sorted_vertices[:, 0] + sorted_vertices[:, 1]
        start_idx = np.argmin(sum_coords)

        # 배열을 회전시켜 왼쪽 상단부터 시작하도록 조정
        normalized_vertices = np.roll(sorted_vertices, -start_idx, axis=0)

        return normalized_vertices

    @staticmethod
    def calculate_side_angles(vertices: np.ndarray) -> np.ndarray:
        """사각형의 각 변의 각도를 계산

        Args:
            vertices: 정규화된 4개의 꼭짓점 좌표 배열 (4, 2)

        Returns:
            각 변의 각도 배열 (4,) - 도 단위
        """
        if len(vertices) != 4:
            return np.array([])

        angles = []
        for i in range(4):
            # 다음 점의 인덱스 (마지막 점은 첫 번째 점으로)
            next_idx = (i + 1) % 4

            # 벡터 계산
            vector = vertices[next_idx] - vertices[i]

            # 각도 계산 (라디안 -> 도)
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

            # 각도를 0~360도 범위로 정규화
            if angle < 0:
                angle += 360

            angles.append(angle)

        return np.array(angles)

    @staticmethod
    def calculate_tilt_differences(base_contour: np.ndarray, contours: List[np.ndarray]) -> List[np.ndarray]:
        """base 컨투어와 다른 컨투어들의 변 각도 차이를 계산

        Args:
            base_contour: 기준이 되는 정규화된 컨투어 (4, 2)
            contours: 비교할 정규화된 컨투어들의 리스트

        Returns:
            각 컨투어의 base 컨투어와의 변 각도 차이들의 리스트
            각 요소는 4개 변의 각도 차이를 담은 배열
        """
        if len(base_contour) != 4:
            print("Base contour must have exactly 4 vertices")
            return []

        # base 컨투어의 변 각도들 계산
        base_angles = Geometry.calculate_side_angles(base_contour)

        tilt_differences = []

        for i, contour in enumerate(contours):
            if len(contour) != 4:
                print(f"Contour {i} must have exactly 4 vertices, skipping...")
                continue

            # 현재 컨투어의 변 각도들 계산
            current_angles = Geometry.calculate_side_angles(contour)

            # 각도 차이 계산
            angle_diffs = current_angles - base_angles

            # 각도 차이를 -180~180도 범위로 정규화
            for j in range(len(angle_diffs)):
                if angle_diffs[j] > 180:
                    angle_diffs[j] -= 360
                elif angle_diffs[j] < -180:
                    angle_diffs[j] += 360

            tilt_differences.append(angle_diffs)

        return tilt_differences

    @staticmethod
    def analyze_tilt_from_contours(contours: List[np.ndarray], base_index: int = 0) -> dict:
        """컨투어들 중 하나를 base로 하여 기울기 차이를 분석

        Args:
            contours: 정규화된 컨투어들의 리스트
            base_index: base로 사용할 컨투어의 인덱스 (기본값: 0)

        Returns:
            분석 결과를 담은 딕셔너리
        """
        if not contours or len(contours) <= base_index:
            return {"error": "Invalid contours or base_index"}

        base_contour = contours[base_index]
        other_contours = [contour for i, contour in enumerate(
            contours) if i != base_index]

        if not other_contours:
            return {
                "base_index": base_index,
                "base_angles": Geometry.calculate_side_angles(base_contour).tolist(),
                "tilt_differences": [],
                "message": "No other contours to compare"
            }

        # 기울기 차이 계산
        tilt_differences = Geometry.calculate_tilt_differences(
            base_contour, other_contours)

        # 결과 정리
        result = {
            "base_index": base_index,
            "base_angles": Geometry.calculate_side_angles(base_contour).tolist(),
            "tilt_differences": [diff.tolist() for diff in tilt_differences],
            "summary": {
                "total_contours": len(contours),
                "compared_contours": len(other_contours),
                "average_tilt_per_side": []
            }
        }

        # 각 변별 평균 기울기 차이 계산
        if tilt_differences:
            for side in range(4):
                side_diffs = [diff[side]
                              for diff in tilt_differences if len(diff) > side]
                if side_diffs:
                    avg_tilt = np.mean(side_diffs)
                    result["summary"]["average_tilt_per_side"].append(avg_tilt)

        return result

    @staticmethod
    def analyze_tilt_from_base_contour(base_contour: np.ndarray, comparison_contours: List[np.ndarray]) -> dict:
        """주어진 base 컨투어를 기준으로 다른 컨투어들의 기울기 차이를 분석
        
        Args:
            base_contour: 기준이 되는 정규화된 컨투어 (예: ArUco 마커)
            comparison_contours: 비교할 정규화된 컨투어들의 리스트
            
        Returns:
            분석 결과를 담은 딕셔너리
        """
        if base_contour is None or not comparison_contours:
            return {"error": "Invalid base_contour or comparison_contours"}
        
        # 기울기 차이 계산
        tilt_differences = Geometry.calculate_tilt_differences(
            base_contour, comparison_contours)
        
        # 결과 정리
        result = {
            "base_angles": Geometry.calculate_side_angles(base_contour).tolist(),
            "tilt_differences": [diff.tolist() for diff in tilt_differences],
            "summary": {
                "total_contours": len(comparison_contours),
                "average_tilt_per_side": []
            }
        }
        
        # 각 변별 평균 기울기 차이 계산
        if tilt_differences:
            for side in range(4):
                side_diffs = [diff[side]
                             for diff in tilt_differences if len(diff) > side]
                if side_diffs:
                    avg_tilt = np.mean(side_diffs)
                    result["summary"]["average_tilt_per_side"].append(avg_tilt)
        
        return result
