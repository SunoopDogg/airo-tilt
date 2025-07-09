import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from matplotlib.axes import Axes


class Visualizer:
    @staticmethod
    def show_masks_on_image(image: np.ndarray, masks: List[np.ndarray]) -> None:
        """이미지에 마스크들을 시각화합니다."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for i, mask in enumerate(masks):
            Visualizer.show_mask(mask, plt.gca(), random_color=True)
        plt.axis('off')
        plt.show()

    @staticmethod
    def show_mask(mask: np.ndarray, ax: Axes, random_color: bool = False, borders: bool = True) -> None:
        """단일 마스크를 시각화합니다."""
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(
                contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(
                mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

    @staticmethod
    def extract_contour(mask: np.ndarray) -> np.ndarray:
        """마스크에서 컨투어를 추출합니다."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 컨투어를 선택
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.reshape(-1, 2)
        return np.array([])

    @staticmethod
    def extract_rectangle_contour(mask: np.ndarray) -> np.ndarray:
        """마스크에서 사각형 형태의 컨투어를 추출합니다."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 가장 큰 컨투어를 선택
            largest_contour = max(contours, key=cv2.contourArea)
            # 사각형으로 근사화
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:  # 사각형이면
                return approx.reshape(-1, 2)
        return np.array([])

    @staticmethod
    def show_contours_on_image(image: np.ndarray, contours: List[np.ndarray]) -> None:
        """이미지에 컨투어들을 시각화합니다."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()
        for contour in contours:
            if contour.size > 0:
                # 컨투어를 닫힌 형태로 그리기 위해 마지막 점을 첫 점으로 연결
                contour = np.concatenate([contour, contour[:1]])
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)

            # 각 변의 각도 계산
            angles = []
            for i in range(len(contour) - 1):
                dx = contour[i + 1, 0] - contour[i, 0]
                dy = contour[i + 1, 1] - contour[i, 1]
                angle = np.arctan2(dy, dx) * (180 / np.pi)
                angles.append(angle)

            # 각도 출력
            for i, angle in enumerate(angles):
                ax.text(contour[i, 0], contour[i, 1], f"{angle:.2f}°",
                        color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        plt.axis('off')
        plt.show()
