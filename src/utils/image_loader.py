import os
import numpy as np

from PIL import Image


class ImageLoader:
    """공통 이미지 로딩 클래스"""

    @staticmethod
    def get_image_path(image_name: str) -> str:
        """이미지 파일명을 절대 경로로 변환"""
        return os.path.join(os.getcwd(), 'images', image_name)

    @staticmethod
    def load_image(image_name: str) -> np.ndarray:
        """이미지 파일 로드 및 RGB 배열로 변환"""
        image_path = ImageLoader.get_image_path(image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        # 이미지 90도 회전
        # image = image.rotate(-90, expand=True)
        return np.array(image.convert("RGB"))
