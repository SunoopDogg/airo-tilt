import matplotlib
import numpy as np

from typing import List, Tuple, Optional

from ui import Visualizer, CoordinateCollector
from core import SAM2Predictor
from utils import Geometry, ArUcoMarkerDetector, ImageLoader

# matplotlib 백엔드 설정
matplotlib.use('TkAgg')


def detect_aruco_markers(image_name: str) -> Tuple[List[np.ndarray], Optional[dict]]:
    """ArUco 마커들을 검출하고 정규화된 컨투어들과 포즈 데이터를 반환

    Args:
        image_name: 이미지 파일 이름

    Returns:
        튜플: (정규화된 ArUco 마커 컨투어들의 리스트, ArUco 포즈 데이터 딕셔너리)
        - 컨투어들: ArUco 마커 컨투어들의 리스트 (각각 4x2 numpy array)
        - 포즈 데이터: corners, ids, r_vectors, t_vectors, centroids 포함 딕셔너리 또는 None
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
            return [], None

        # 포즈 추정
        r_vectors, t_vectors = detector.estimate_pose(corners)

        # 모든 마커의 corners 추출 및 정규화
        normalized_contours = []
        centroids = []

        for i in range(len(ids)):
            marker_corners = np.array(corners[i][0], dtype=np.float32)
            normalized_contour = Geometry.normalize_rectangle_vertices(
                marker_corners)
            normalized_contours.append(normalized_contour)

            # 중심점 계산
            centroid = np.mean(marker_corners, axis=0)
            centroids.append(centroid)

        # ArUco 데이터 딕셔너리 생성
        aruco_data = {
            'corners': corners,
            'ids': ids,
            'r_vectors': r_vectors,
            't_vectors': t_vectors,
            'detector': detector,
            'marker_ids': [int(id[0]) for id in ids],
            'centroids': centroids,
            'count': len(ids)
        }

        print(
            f"ArUco markers detected - Count: {aruco_data['count']}, IDs: {aruco_data['marker_ids']}")
        if r_vectors and t_vectors:
            for i, marker_id in enumerate(aruco_data['marker_ids']):
                roll, pitch, yaw = detector.get_marker_orientation(
                    r_vectors[i])
                print(
                    f"  Marker {marker_id} - Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")

        return normalized_contours, aruco_data

    except Exception as e:
        print(f"Error detecting ArUco markers: {e}")
        return [], None


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


def analyze_vertical_tilt_differences(aruco_contours: List[np.ndarray], aruco_data: Optional[dict],
                                      contours: List[np.ndarray], visualize: bool = True, image_name: str = None) -> dict:
    """ArUco 마커 기준으로 컨투어들의 수직선 기울기 차이를 분석하는 함수

    Args:
        aruco_contours: ArUco 마커들의 정규화된 컨투어 리스트
        aruco_data: ArUco 마커 데이터
        contours: 비교할 객체들의 정규화된 컨투어 리스트
        visualize: 결과를 시각화할지 여부 (기본값: True)

    Returns:
        Vertical tilt difference analysis results
    """
    if not contours:
        return {"error": "No contours to analyze"}

    # ArUco 마커가 있는 경우 기준 각도 계산
    if aruco_contours and aruco_data:
        reference_angle = Geometry.calculate_reference_angle_from_markers(
            aruco_contours, aruco_data)
    else:
        # ArUco 마커가 없는 경우 수직선(0도)을 기준으로 사용
        reference_angle = 0.0
        print("No ArUco markers detected, using vertical line (0°) as reference")

    # 수직선 기울기 차이 분석
    result = Geometry.analyze_vertical_tilt_differences(
        reference_angle, contours)

    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
        return result

    print(f"=== Vertical Tilt Difference Analysis Results ===")
    print(f"Reference angle: {result['reference_angle']:.2f}°")
    print(f"Total comparison contours: {result['summary']['total_contours']}")

    if result['tilt_differences']:
        print(f"Vertical line tilt differences from reference (degrees):")
        for i, diff in enumerate(result['tilt_differences']):
            print(f"  Object {i+1}:")
            print(f"    Left vertical line: {diff['left_diff']:.2f}°")
            print(f"    Right vertical line: {diff['right_diff']:.2f}°")
            print(f"    Average: {diff['average_diff']:.2f}°")

        print(f"Summary:")
        print(
            f"  Average left tilt: {result['summary']['average_left_tilt']:.2f}°")
        print(
            f"  Average right tilt: {result['summary']['average_right_tilt']:.2f}°")
        print(
            f"  Average overall tilt: {result['summary']['average_overall_tilt']:.2f}°")

    # Visualization
    if visualize and result['tilt_differences']:
        print("=== Vertical Line Visualization ===")
        # Load image for visualization background if image_name is provided
        image = ImageLoader.load_image(image_name) if image_name else None
        Visualizer.visualize_vertical_angle_differences(reference_angle, contours, result, image)

    return result


if __name__ == "__main__":
    FILE_NAME = 'aruco_test_05.png'

    try:
        # 먼저 ArUco 마커 검출 (컨투어와 포즈 데이터 모두 가져오기)
        aruco_contours, aruco_data = detect_aruco_markers(FILE_NAME)

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
            # 새로운 수직선 기울기 분석
            print("=== Point-based Vertical Tilt Analysis ===")
            analyze_vertical_tilt_differences(
                aruco_contours, aruco_data, contours, visualize=True, image_name=FILE_NAME)

        # Box-based processing
        if boxes:
            print(f"Collected boxes: {boxes}")
            print("Processing boxes...")
            masks, contours = segment_objects_from_boxes(FILE_NAME, boxes)

            # 향상된 시각화: 컨투어 + ArUco 포즈
            visualize_contours_with_aruco_pose(FILE_NAME, contours, aruco_data)

            # Tilt difference analysis with ArUco as base
            # 새로운 수직선 기울기 분석
            print("=== Box-based Vertical Tilt Analysis ===")
            analyze_vertical_tilt_differences(
                aruco_contours, aruco_data, contours, visualize=True, image_name=FILE_NAME)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists in the 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
