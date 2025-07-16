import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from ui import Visualizer, CoordinateCollector
from core import SAM2Predictor

# matplotlib 백엔드 설정
matplotlib.use('TkAgg')


def apply_sam2_to_coordinates(image_name: str, coordinates: List[tuple]) -> List[np.ndarray]:
    """좌표를 기반으로 SAM2 예측을 수행하고 결과를 시각화합니다."""
    # SAM2 예측기 초기화
    predictor = SAM2Predictor()

    # 마스크 예측
    image, best_masks = predictor.predict_masks_from_points(
        image_name, coordinates)

    # 결과 시각화
    Visualizer.show_masks_on_image(image, best_masks)

    # 마스크의 컨투어 추출
    contours = [Visualizer.extract_rectangle_contour(
        mask) for mask in best_masks]
    # 컨투어 시각화
    Visualizer.show_contours_on_image(image, contours)

    return best_masks


def apply_sam2_to_boxes(image_name: str, boxes: List[tuple]) -> List[np.ndarray]:
    """박스를 기반으로 SAM2 예측을 수행하고 결과를 시각화합니다."""
    # SAM2 예측기 초기화
    predictor = SAM2Predictor()

    # 마스크 예측
    image, masks = predictor.predict_masks_from_boxes(image_name, boxes)

    # 결과 시각화
    Visualizer.show_masks_on_image(image, masks)

    # 마스크의 컨투어 추출
    contours = [Visualizer.extract_rectangle_contour(mask) for mask in masks]
    # 컨투어 시각화
    Visualizer.show_contours_on_image(image, contours)

    return masks


if __name__ == "__main__":
    FILE_NAME = 'IMG_0940.jpg'

    try:
        collector = CoordinateCollector(FILE_NAME, mode='point')
        coordinates, boxes = collector.collect_data()

        # 포인트 기반 처리
        if coordinates:
            print(f"Collected coordinates: {coordinates}")
            print("Processing points...")
            apply_sam2_to_coordinates(FILE_NAME, coordinates)

        # 박스 기반 처리
        if boxes:
            print(f"Collected boxes: {boxes}")
            print("Processing boxes...")
            apply_sam2_to_boxes(FILE_NAME, boxes)

        if not coordinates and not boxes:
            print("No coordinates or boxes collected.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists in the 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
