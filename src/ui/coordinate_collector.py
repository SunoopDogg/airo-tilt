import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import List, Tuple, Optional

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils.image_loader import ImageLoader


class CoordinateCollector:
    def __init__(self, image_name: str, mode: str = 'point') -> None:
        self.image_name = image_name
        self.mode = mode  # 'point' or 'box'
        self.coordinates: List[Tuple[int, int]] = []
        self.boxes: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None
        self.image: Optional[np.ndarray] = None

        # 박스 드래그 관련 변수
        self.dragging = False
        self.start_pos = None
        self.current_rect = None

    def _on_click(self, event) -> None:
        """마우스 클릭 이벤트 처리"""
        if event.inaxes != self.ax:
            return

        if self.mode == 'point':
            self._handle_point_click(event)
        elif self.mode == 'box':
            self._handle_box_click(event)

    def _handle_point_click(self, event) -> None:
        """포인트 모드에서 마우스 클릭 처리"""
        if event.button == 1:  # 왼쪽 클릭
            x, y = int(event.xdata), int(event.ydata)
            self.coordinates.append((x, y))
            print(f"Point: {x}, {y}")

            # 클릭한 지점에 마커 표시
            self.ax.plot(x, y, 'go', markersize=8,
                         markeredgecolor='white', markeredgewidth=2)
            self.fig.canvas.draw()

    def _handle_box_click(self, event) -> None:
        """박스 모드에서 마우스 클릭 처리"""
        if event.button == 1:  # 왼쪽 클릭 - 드래그 시작
            if not self.dragging:
                self.dragging = True
                self.start_pos = (int(event.xdata), int(event.ydata))
                print(f"Box drag started at: {self.start_pos}")

    def _on_mouse_move(self, event) -> None:
        """마우스 이동 이벤트 처리 (드래그 중일 때)"""
        if self.mode == 'box' and self.dragging and event.inaxes == self.ax:
            if self.current_rect:
                self.current_rect.remove()

            x1, y1 = self.start_pos
            x2, y2 = int(event.xdata), int(event.ydata)

            # 박스 그리기
            width = x2 - x1
            height = y2 - y1

            self.current_rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
            )
            self.ax.add_patch(self.current_rect)
            self.fig.canvas.draw()

    def _on_mouse_release(self, event) -> None:
        """마우스 릴리스 이벤트 처리 (드래그 종료)"""
        if self.mode == 'box' and self.dragging and event.inaxes == self.ax:
            self.dragging = False
            x1, y1 = self.start_pos
            x2, y2 = int(event.xdata), int(event.ydata)

            # 좌표 정렬 (x1 < x2, y1 < y2)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # 최소 크기 체크
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                box = (x1, y1, x2, y2)
                self.boxes.append(box)
                print(f"Box added: {box}")

                # 최종 박스 그리기
                if self.current_rect:
                    self.current_rect.remove()

                final_rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='blue', facecolor='none', alpha=0.8
                )
                self.ax.add_patch(final_rect)
                self.fig.canvas.draw()
            else:
                print("Box too small, ignored")
                if self.current_rect:
                    self.current_rect.remove()
                    self.fig.canvas.draw()

            self.current_rect = None
            self.start_pos = None

    def _on_key_press(self, event) -> None:
        """키보드 이벤트 처리"""
        if event.key == 'c':
            self._clear_all()
        elif event.key == 's':
            self._save_data()
        elif event.key == 'm':
            self._toggle_mode()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)

    def _toggle_mode(self) -> None:
        """포인트 모드와 박스 모드 전환"""
        self.mode = 'box' if self.mode == 'point' else 'point'
        print(f"Mode switched to: {self.mode}")
        self._update_title()

    def _load_and_display_image(self) -> None:
        """이미지를 로드하고 matplotlib으로 표시합니다."""
        self.image = ImageLoader.load_image(self.image_name)

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(self.image)
        self._update_title()
        self.ax.axis('on')  # 좌표 표시를 위해 축 켜기

        # 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect(
            'button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _update_title(self) -> None:
        """제목 업데이트"""
        mode_text = "POINT" if self.mode == 'point' else "BOX"
        if self.mode == 'point':
            instruction = "Click: Add Point"
        else:
            instruction = "Drag: Draw Box"

        self.ax.set_title(
            f"Mode: {mode_text} | {self.image_name}\n"
            f"{instruction} | 'c': Clear | 's': Save | 'm': Toggle Mode | 'q': Quit"
        )

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

    def _clear_all(self) -> None:
        """모든 데이터를 지우고 이미지를 다시 그립니다."""
        print("Clearing all data...")
        self.coordinates.clear()
        self.boxes.clear()
        self.dragging = False
        self.current_rect = None
        self.start_pos = None

        # 이미지 다시 그리기
        self.ax.clear()
        self.ax.imshow(self.image)
        self._update_title()
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

    def _save_data(self) -> None:
        """현재 좌표와 박스 데이터를 파일에 저장합니다."""
        saved_something = False

        if self.coordinates:
            with open('coordinates.txt', 'w') as f:
                for coord in self.coordinates:
                    f.write(f"{coord[0]} {coord[1]}\n")
            print(
                f"Coordinates saved to coordinates.txt ({len(self.coordinates)} points)")
            saved_something = True

        if self.boxes:
            with open('boxes.txt', 'w') as f:
                for box in self.boxes:
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")
            print(f"Boxes saved to boxes.txt ({len(self.boxes)} boxes)")
            saved_something = True

        if not saved_something:
            print("No data to save.")

    def collect_coordinates(self) -> List[Tuple[int, int]]:
        """사용자로부터 좌표를 수집합니다."""
        self._load_and_display_image()
        plt.show()
        return self.coordinates

    def collect_boxes(self) -> List[Tuple[int, int, int, int]]:
        """사용자로부터 박스를 수집합니다."""
        self.mode = 'box'
        self._load_and_display_image()
        plt.show()
        return self.boxes

    def collect_data(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
        """사용자로부터 좌표와 박스 모두를 수집합니다."""
        self._load_and_display_image()
        plt.show()
        return self.coordinates, self.boxes
