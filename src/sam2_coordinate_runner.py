import matplotlib
import numpy as np

from typing import List, Tuple, Optional

from ui import Visualizer, CoordinateCollector
from core import SAM2Predictor
from utils import Geometry, ArUcoMarkerDetector, ImageLoader

# matplotlib 백엔드 설정
matplotlib.use('TkAgg')


def detect_aruco_marker_contour(image_name: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """ArUco 마커를 검출하고 정규화된 컨투어와 포즈 데이터를 반환

    Args:
        image_name: 이미지 파일 이름

    Returns:
        튜플: (정규화된 ArUco 마커 컨투어, ArUco 포즈 데이터 딕셔너리)
        - 컨투어: 4x2 numpy array 또는 None
        - 포즈 데이터: corners, ids, r_vectors, t_vectors 포함 딕셔너리 또는 None
    """
    # ArUco detector 초기화 (기본 4x4_50 딕셔너리, 36mm 마커 크기 사용)
    detector = ArUcoMarkerDetector(
        aruco_dict_type="DICT_4X4_50",
        marker_size_mm=36.0
    )

    # 이미지 경로 설정
    image_path = f"images/{image_name}"

    # ArUco 마커 검출 및 분석
    try:
        # 이미지 로드
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 마커 검출
        corners, ids, rejected = detector.detect_markers(image)
        
        if ids is None or len(ids) == 0:
            print("No ArUco markers detected in the image.")
            return None, None

        # 포즈 추정
        r_vectors, t_vectors = detector.estimate_pose(corners)
        
        # 첫 번째 마커의 corners 추출 및 정규화
        first_marker_corners = np.array(corners[0][0], dtype=np.float32)
        normalized_contour = Geometry.normalize_rectangle_vertices(first_marker_corners)
        
        # ArUco 데이터 딕셔너리 생성
        aruco_data = {
            'corners': corners,
            'ids': ids,
            'r_vectors': r_vectors,
            't_vectors': t_vectors,
            'detector': detector,  # 나중에 draw_markers 메서드 사용을 위해
            'marker_id': int(ids[0][0])
        }
        
        print(f"ArUco marker detected - ID: {aruco_data['marker_id']}")
        if r_vectors and t_vectors:
            # 첫 번째 마커의 오리엔테이션 정보 출력
            roll, pitch, yaw = detector.get_marker_orientation(r_vectors[0])
            print(f"  Rotation - Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
            
        return normalized_contour, aruco_data

    except Exception as e:
        print(f"Error detecting ArUco marker: {e}")
        return None, None


def segment_objects_from_points(image_name: str, coordinates: List[tuple]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """포인트 좌표 기반으로 객체 세그먼테이션 수행 및 결과 시각화"""
    # SAM2 예측기 초기화
    predictor = SAM2Predictor()

    # 마스크 예측
    image, best_masks = predictor.predict_masks_from_points(
        image_name, coordinates)

    # 결과 시각화
    Visualizer.show_masks_on_image(image, best_masks)

    # 마스크의 컨투어 추출
    contours = [Geometry.extract_rectangle_contour(
        mask) for mask in best_masks]
    # 컨투어 시각화
    Visualizer.show_contours_on_image(image, contours)

    return best_masks, contours


def segment_objects_from_boxes(image_name: str, boxes: List[tuple]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """바운딩 박스 기반으로 객체 세그먼테이션 수행 및 결과 시각화"""
    # SAM2 예측기 초기화
    predictor = SAM2Predictor()

    # 마스크 예측
    image, masks = predictor.predict_masks_from_boxes(image_name, boxes)

    # 결과 시각화
    Visualizer.show_masks_on_image(image, masks)

    # 마스크의 컨투어 추출
    contours = [Geometry.extract_rectangle_contour(mask) for mask in masks]
    print("Contours:", contours)

    # 컨투어 시각화
    Visualizer.show_contours_on_image(image, contours)

    return masks, contours


def visualize_multiple_contours_overlay(image_name: str, contours_list: List[List[np.ndarray]]) -> None:
    """여러 세트의 정규화된 컨투어들을 하나의 이미지에 오버레이하여 시각화

    Args:
        image_name: 이미지 파일 이름
        contours_list: 여러 세트의 컨투어 리스트들
    """
    # 이미지 로드
    image = ImageLoader.load_image(image_name)

    # 오버레이 시각화
    Visualizer.show_all_normalized_contours_overlay(image, contours_list)

def visualize_contours_with_aruco_pose(image_name: str, contours_list: List[np.ndarray], 
                                       aruco_data: Optional[dict] = None) -> None:
    """컨투어와 ArUco 마커 포즈를 함께 시각화하는 향상된 함수
    
    Args:
        image_name: 이미지 파일 이름
        contours_list: 정규화된 컨투어들의 리스트
        aruco_data: ArUco 마커 데이터 (detect_aruco_marker_contour에서 반환된 딕셔너리)
    """
    # 이미지 로드
    image = ImageLoader.load_image(image_name)
    
    # 향상된 시각화 호출
    Visualizer.show_contours_and_aruco_pose(image, contours_list, aruco_data)


def analyze_contour_tilt_differences(base_contour: np.ndarray, contours: List[np.ndarray], visualize: bool = True) -> dict:
    """ArUco 마커를 기준으로 컨투어들의 기울기 차이를 분석하는 함수

    Args:
        base_contour: 기준이 되는 ArUco 마커의 정규화된 컨투어
        contours: 비교할 객체들의 정규화된 컨투어 리스트
        visualize: 결과를 시각화할지 여부 (기본값: True)

    Returns:
        Tilt difference analysis results
    """
    if base_contour is None or not contours:
        return {"error": "Invalid base_contour or contours"}

    # Geometry 클래스의 메서드를 직접 호출하여 분석
    result = Geometry.analyze_tilt_from_base_contour(base_contour, contours)

    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
        return result

    print(f"\n=== ArUco-based Contour Tilt Difference Analysis Results ===")
    print(f"Base: ArUco Marker")
    print(f"Base contour side angles: {result['base_angles']}")
    print(f"Total comparison contours: {result['summary']['total_contours']}")

    if result['tilt_differences']:
        print(f"\nTilt differences from ArUco marker (degrees):")
        for i, diff in enumerate(result['tilt_differences']):
            print(f"  Object {i+1}: {diff}")

        if result['summary']['average_tilt_per_side']:
            print(f"\nAverage tilt difference by side:")
            sides = ['Top', 'Right', 'Bottom', 'Left']
            for i, avg_tilt in enumerate(result['summary']['average_tilt_per_side']):
                print(f"  {sides[i]} side: {avg_tilt:.2f}°")

    # Visualization
    if visualize and result['tilt_differences']:
        print("\n=== Angle Difference Visualization (ArUco as base) ===")
        Visualizer.visualize_angle_differences(
            base_contour, contours, result)

    return result


if __name__ == "__main__":
    FILE_NAME = 'aruco_test_05.png'

    try:
        # 먼저 ArUco 마커 검출 (컨투어와 포즈 데이터 모두 가져오기)
        aruco_contour, aruco_data = detect_aruco_marker_contour(FILE_NAME)

        # 사용자 입력 수집
        collector = CoordinateCollector(FILE_NAME, mode='box')
        coordinates, boxes = collector.collect_data()

        # Point-based processing
        if coordinates:
            print(f"Collected coordinates: {coordinates}")
            print("Processing points...")
            masks, contours = segment_objects_from_points(
                FILE_NAME, coordinates)

            # 향상된 시각화: 컨투어 + ArUco 포즈
            visualize_contours_with_aruco_pose(FILE_NAME, contours, aruco_data)

            # Tilt difference analysis with ArUco as base
            if aruco_contour is not None and contours:
                print("\n=== Point-based Tilt Analysis (ArUco as base) ===")
                analyze_contour_tilt_differences(aruco_contour, contours)
            elif contours and len(contours) > 1:
                # Fallback: 첫 번째 객체를 base로 사용
                print(
                    "\n=== Point-based Contour Tilt Analysis (No ArUco, using first object as base) ===")
                result = Geometry.analyze_tilt_from_contours(
                    contours, base_index=0)

        # Box-based processing
        if boxes:
            print(f"Collected boxes: {boxes}")
            print("Processing boxes...")
            masks, contours = segment_objects_from_boxes(FILE_NAME, boxes)

            # 향상된 시각화: 컨투어 + ArUco 포즈
            visualize_contours_with_aruco_pose(FILE_NAME, contours, aruco_data)

            # Tilt difference analysis with ArUco as base
            if aruco_contour is not None and contours:
                print("\n=== Box-based Tilt Analysis (ArUco as base) ===")
                analyze_contour_tilt_differences(aruco_contour, contours)
            elif contours and len(contours) > 1:
                # Fallback: 첫 번째 객체를 base로 사용
                print(
                    "\n=== Box-based Contour Tilt Analysis (No ArUco, using first object as base) ===")
                result = Geometry.analyze_tilt_from_contours(
                    contours, base_index=0)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists in the 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
