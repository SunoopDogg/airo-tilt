import matplotlib
import numpy as np

from typing import List, Tuple

from ui import Visualizer, CoordinateCollector
from core import SAM2Predictor
from utils import Geometry

# matplotlib 백엔드 설정
matplotlib.use('TkAgg')


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
    from utils import ImageLoader

    # 이미지 로드
    image = ImageLoader.load_image(image_name)

    # 오버레이 시각화
    Visualizer.show_all_normalized_contours_overlay(image, contours_list)


def analyze_contour_tilt_differences(contours: List[np.ndarray], base_index: int = 0, visualize: bool = True) -> dict:
    """정규화된 컨투어들의 기울기 차이를 분석하는 함수

    Args:
        contours: 정규화된 컨투어들의 리스트
        base_index: base로 사용할 컨투어의 인덱스 (기본값: 0)
        visualize: 결과를 시각화할지 여부 (기본값: True)

    Returns:
        Tilt difference analysis results
    """
    result = Geometry.analyze_tilt_from_contours(contours, base_index)

    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
        return result

    print(f"\n=== Contour Tilt Difference Analysis Results ===")
    print(f"Base contour index: {result['base_index']}")
    print(f"Base contour side angles: {result['base_angles']}")
    print(f"Total contours: {result['summary']['total_contours']}")
    print(f"Compared contours: {result['summary']['compared_contours']}")

    if result['tilt_differences']:
        print(f"\nTilt differences by contour (degrees):")
        for i, diff in enumerate(result['tilt_differences']):
            print(f"  Contour {i+1}: {diff}")

        if result['summary']['average_tilt_per_side']:
            print(f"\nAverage tilt difference by side:")
            sides = ['Top', 'Right', 'Bottom', 'Left']
            for i, avg_tilt in enumerate(result['summary']['average_tilt_per_side']):
                print(f"  {sides[i]} side: {avg_tilt:.2f}°")

    # Visualization
    if visualize and result['tilt_differences']:
        base_contour = contours[base_index]
        other_contours = [contour for i, contour in enumerate(
            contours) if i != base_index]

        print("\n=== Angle Difference Visualization ===")
        Visualizer.visualize_angle_differences(
            base_contour, other_contours, result)

    return result


if __name__ == "__main__":
    FILE_NAME = 'erica10.jpg'

    try:
        collector = CoordinateCollector(FILE_NAME, mode='box')
        coordinates, boxes = collector.collect_data()

        # Point-based processing
        if coordinates:
            print(f"Collected coordinates: {coordinates}")
            print("Processing points...")
            masks, contours = segment_objects_from_points(
                FILE_NAME, coordinates)

            visualize_multiple_contours_overlay(FILE_NAME, contours)

            # Tilt difference analysis
            if contours and len(contours) > 1:
                print("\n=== Point-based Contour Tilt Analysis ===")
                analyze_contour_tilt_differences(contours, base_index=0)

        # Box-based processing
        if boxes:
            print(f"Collected boxes: {boxes}")
            print("Processing boxes...")
            masks, contours = segment_objects_from_boxes(FILE_NAME, boxes)

            visualize_multiple_contours_overlay(FILE_NAME, contours)

            # Tilt difference analysis
            if contours and len(contours) > 1:
                print("\n=== Box-based Contour Tilt Analysis ===")
                analyze_contour_tilt_differences(contours, base_index=0)

        if not coordinates and not boxes:
            print("No coordinates or boxes collected.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists in the 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
