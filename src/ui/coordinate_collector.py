import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils.image_loader import ImageLoader


class CoordinateCollector:
    def __init__(self, image_name: str) -> None:
        self.image_name = image_name
        self.coordinates: List[Tuple[int, int]] = []
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.image: Optional[np.ndarray] = None

    def _on_click(self, event) -> None:
        """마우스 클릭 이벤트 처리"""
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # 왼쪽 클릭
            x, y = int(event.xdata), int(event.ydata)
            self.coordinates.append((x, y))
            print(f"Coordinates: {x}, {y}")

            # 클릭한 지점에 마커 표시
            self.ax.plot(x, y, 'go', markersize=8,
                         markeredgecolor='white', markeredgewidth=2)
            self.fig.canvas.draw()

    def _on_key_press(self, event) -> None:
        """키보드 이벤트 처리"""
        if event.key == 'c':
            self._clear_coordinates()
        elif event.key == 's':
            self._save_coordinates()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)

    def _load_and_display_image(self) -> None:
        """이미지를 로드하고 matplotlib으로 표시합니다."""
        self.image = ImageLoader.load_image(self.image_name)

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"Coordinate Collection: {self.image_name}\nClick: Add | 'c': Clear | 's': Save | 'q': Quit")
        self.ax.axis('on')  # 좌표 표시를 위해 축 켜기

        # 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _clear_coordinates(self) -> None:
        """좌표를 지우고 이미지를 다시 그립니다."""
        print("Clearing coordinates...")
        self.coordinates.clear()

        # 이미지 다시 그리기
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"Coordinate Collection: {self.image_name}\nClick: Add | 'c': Clear | 's': Save | 'q': Quit")
        self.ax.axis('on')
        self.fig.canvas.draw()

    def _save_coordinates(self) -> None:
        """현재 좌표를 파일에 저장합니다."""
        if self.coordinates:
            with open('coordinates.txt', 'w') as f:
                for coord in self.coordinates:
                    f.write(f"{coord[0]} {coord[1]}\n")
            print("Coordinates saved to coordinates.txt")
        else:
            print("No coordinates to save.")

    def collect_coordinates(self) -> List[Tuple[int, int]]:
        """사용자로부터 좌표를 수집합니다."""
        self._load_and_display_image()

        plt.show()

        return self.coordinates
