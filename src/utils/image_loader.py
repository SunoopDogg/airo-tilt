import os
import numpy as np

from PIL import Image


class ImageLoader:
    """공통 이미지 로딩 클래스"""

    @staticmethod
    def get_image_path(image_name: str) -> str:
        """이미지 파일의 전체 경로를 반환합니다."""
        return os.path.join(os.getcwd(), 'images', image_name)

    @staticmethod
    def load_image(image_name: str) -> np.ndarray:
        """이미지를 로드하고 RGB 배열로 변환합니다."""
        image_path = ImageLoader.get_image_path(image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        return np.array(image.convert("RGB"))
