import os
import numpy as np

from PIL import Image


class ImageLoader:
    """
    공통 이미지 로딩 클래스
    """

    @staticmethod
    def get_image_path(image_name: str) -> str:
        """
        이미지 파일명을 절대 경로로 변환
        
        Args:
            image_name: 이미지 파일명
            
        Returns:
            이미지의 절대 경로
        """
        return os.path.join(os.getcwd(), 'images', image_name)

    @staticmethod
    def load_image(image_name: str) -> np.ndarray:
        """
        이미지 파일 로드 및 RGB 배열로 변환
        
        Args:
            image_name: 이미지 파일명
            
        Returns:
            RGB format의 이미지 numpy array
            
        Raises:
            FileNotFoundError: 이미지 파일이 존재하지 않을 때
        """
        image_path = ImageLoader.get_image_path(image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        return np.array(image.convert("RGB"))
