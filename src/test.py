import matplotlib
import numpy as np

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
    image, best_masks = predictor.predict_masks(image_name, coordinates)

    # 결과 시각화
    Visualizer.show_masks_on_image(image, best_masks)

    return best_masks


if __name__ == "__main__":
    FILE_NAME = 'IMG_0940.jpg'

    try:
        collector = CoordinateCollector(FILE_NAME)
        coordinates = collector.collect_coordinates()

        if coordinates:
            print(f"Collected coordinates: {coordinates}")
            apply_sam2_to_coordinates(FILE_NAME, coordinates)
        else:
            print("No coordinates collected.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists in the 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
