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
