import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional

from matplotlib.axes import Axes


class Visualizer:
    @staticmethod
    def show_masks_on_image(image: np.ndarray, masks: List[np.ndarray]) -> None:
        """이미지 위에 마스크 오버레이 표시"""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            Visualizer.show_mask(mask, plt.gca(), random_color=True)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_mask(mask: np.ndarray, ax: Axes, random_color: bool = False, borders: bool = True) -> None:
        """단일 마스크 렌더링 및 경계선 표시"""
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
    def show_contours_on_image(image: np.ndarray, contours: List[np.ndarray]) -> None:
        """이미지 위에 윤곽선 및 각도 정보 표시"""
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
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_all_normalized_contours_overlay(image: np.ndarray, contours_list: List[np.ndarray]) -> None:
        """모든 정규화된 사각형 컨투어를 하나의 이미지에 오버레이하여 시각화

        Args:
            image: 배경 이미지
            contours_list: 정규화된 컨투어들의 리스트
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # 색상 팔레트 정의 (다양한 객체를 구분하기 위해)
        object_colors = ['yellow', 'cyan', 'magenta',
                         'orange', 'lime', 'pink', 'purple', 'brown']
        vertex_colors = ['red', 'green', 'blue', 'white']
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        vertex_labels = ['TL', 'TR', 'BR', 'BL']

        valid_contours = []
        for i, contour in enumerate(contours_list):
            if contour.size > 0 and len(contour) == 4:
                valid_contours.append((i, contour))

        if not valid_contours:
            plt.title('No Valid Normalized Contours Found',
                      fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.show()
            return

        # 모든 유효한 컨투어를 오버레이
        for obj_idx, contour in valid_contours:
            object_color = object_colors[obj_idx % len(object_colors)]

            # 사각형 그리기 (닫힌 형태)
            rectangle = np.vstack([contour, contour[0:1]])  # 마지막 점을 첫 점으로 연결
            ax.plot(rectangle[:, 0], rectangle[:, 1],
                    color=object_color, linewidth=3, alpha=0.8,
                    label=f'Object {obj_idx + 1}')

            # 각 꼭짓점을 색상별로 표시
            for vertex_idx, vertex in enumerate(contour):
                vertex_color = vertex_colors[vertex_idx]

                # 꼭짓점 원으로 표시
                if vertex_color == 'white':
                    ax.plot(vertex[0], vertex[1], 'o', color=vertex_color, markersize=-1,
                            markeredgecolor='black', markeredgewidth=2)
                else:
                    ax.plot(vertex[0], vertex[1], 'o',
                            color=vertex_color, markersize=-1)

                # 꼭짓점 라벨 표시 (객체 번호 + 꼭짓점 타입)
                ax.text(vertex[0] + 15, vertex[1] - 15, f'{obj_idx + 1}-{vertex_labels[vertex_idx]}',
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=2))

        plt.title(f'All Normalized Rectangle Contours Overlay ({len(valid_contours)} objects)',
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_contours_and_aruco_pose(image: np.ndarray, contours_list: List[np.ndarray],
                                     aruco_data: Optional[dict] = None) -> None:
        """모든 정규화된 컨투어와 ArUco 마커 포즈를 함께 시각화

        Args:
            image: 배경 이미지
            contours_list: 정규화된 컨투어들의 리스트
            aruco_data: ArUco 마커 데이터 (corners, ids, r_vectors, t_vectors, detector 포함)
        """
        import cv2

        # 이미지 복사본 생성
        img_with_aruco = image.copy()

        # ArUco 마커와 포즈 축 그리기 (있는 경우)
        if aruco_data is not None and 'detector' in aruco_data:
            detector = aruco_data['detector']
            corners = aruco_data.get('corners', [])
            ids = aruco_data.get('ids', None)
            r_vectors = aruco_data.get('r_vectors', None)
            t_vectors = aruco_data.get('t_vectors', None)

            if corners and ids is not None:
                # ArUco 마커 테두리와 포즈 축 그리기
                img_with_aruco = detector.draw_markers(
                    img_with_aruco, corners, ids, r_vectors, t_vectors
                )

        # matplotlib 설정
        plt.figure(figsize=(12, 10))
        plt.imshow(img_with_aruco)
        ax = plt.gca()

        # 색상 팔레트 정의
        object_colors = ['yellow', 'cyan', 'magenta',
                         'orange', 'lime', 'pink', 'purple', 'brown']
        vertex_colors = ['red', 'green', 'blue', 'white']
        vertex_labels = ['TL', 'TR', 'BR', 'BL']

        # 유효한 컨투어 필터링
        valid_contours = []
        for i, contour in enumerate(contours_list):
            if contour.size > 0 and len(contour) == 4:
                valid_contours.append((i, contour))

        # 컨투어 오버레이
        for obj_idx, contour in valid_contours:
            object_color = object_colors[obj_idx % len(object_colors)]

            # 사각형 그리기 (닫힌 형태)
            rectangle = np.vstack([contour, contour[0:1]])
            ax.plot(rectangle[:, 0], rectangle[:, 1],
                    color=object_color, linewidth=2.5, alpha=0.8,
                    label=f'Object {obj_idx + 1}')

            # 각 꼭짓점 표시
            for vertex_idx, vertex in enumerate(contour):
                vertex_color = vertex_colors[vertex_idx]

                # 꼭짓점 원으로 표시
                if vertex_color == 'white':
                    ax.plot(vertex[0], vertex[1], 'o', color=vertex_color, markersize=6,
                            markeredgecolor='black', markeredgewidth=1.5)
                else:
                    ax.plot(vertex[0], vertex[1], 'o',
                            color=vertex_color, markersize=6)

                # 꼭짓점 라벨 표시
                ax.text(vertex[0] + 10, vertex[1] - 10,
                        f'{obj_idx + 1}-{vertex_labels[vertex_idx]}',
                        fontsize=8, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=1))

        # ArUco 정보 텍스트 추가
        if aruco_data is not None:
            marker_id = aruco_data.get('marker_id', 'Unknown')
            info_text = f"ArUco Marker ID: {marker_id} (with 3D pose axes)"
            ax.text(10, 30, info_text, fontsize=12, fontweight='bold',
                    color='white', bbox=dict(facecolor='blue', alpha=0.7, pad=5))

        # 제목 설정
        title = f'Enhanced Visualization: {len(valid_contours)} Objects'
        if aruco_data is not None:
            title += ' + ArUco Pose Reference'
        plt.title(title, fontsize=16, fontweight='bold')

        # 범례 표시
        if valid_contours:
            plt.legend(loc='upper right', fontsize=10)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_angle_differences(base_contour: np.ndarray, contours: List[np.ndarray],
                                    tilt_analysis_result: dict) -> None:
        """각 변의 각도 차이를 시각적으로 비교하는 함수

        Args:
            base_contour: 기준 컨투어
            contours: 비교할 컨투어들
            tilt_analysis_result: analyze_tilt_from_contours 결과
        """
        if len(contours) == 0:
            print("No contours to compare.")
            return

        base_angles = np.array(tilt_analysis_result['base_angles'])
        tilt_differences = tilt_analysis_result['tilt_differences']

        # Subplot setup (2 rows: top for angle comparison, bottom for contour visualization)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Contour Angle Difference Analysis',
                     fontsize=16, fontweight='bold')

        # Color palette
        colors = plt.cm.Set1(np.linspace(0, 1, len(contours) + 1))
        side_names = ['Top', 'Right', 'Bottom', 'Left']

        # 1. Angle comparison by side (bar chart)
        ax1 = axes[0, 0]
        x_pos = np.arange(4)
        width = 0.8 / (len(contours) + 1)

        # Base contour angles
        bars_base = ax1.bar(x_pos - width * len(contours) / 2, base_angles,
                            width, label='Base', color='black', alpha=0.7)

        # Comparison contours angles
        for i, contour in enumerate(contours):
            if i < len(tilt_differences):
                contour_angles = base_angles + np.array(tilt_differences[i])
                bars = ax1.bar(x_pos - width * len(contours) / 2 + width * (i + 1),
                               contour_angles, width,
                               label=f'Contour {i+1}', color=colors[i+1], alpha=0.7)

        ax1.set_xlabel('Side')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Angle Comparison by Side')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(side_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Angle difference heatmap
        ax2 = axes[0, 1]
        if tilt_differences:
            diff_matrix = np.array(tilt_differences)
            im = ax2.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                            vmin=-max(abs(diff_matrix.min()),
                                      abs(diff_matrix.max())),
                            vmax=max(abs(diff_matrix.min()), abs(diff_matrix.max())))

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Angle Difference (degrees)')

            # Add text labels
            for i in range(diff_matrix.shape[0]):
                for j in range(diff_matrix.shape[1]):
                    text = ax2.text(j, i, f'{diff_matrix[i, j]:.1f}°',
                                    ha="center", va="center", color="black", fontweight='bold')

            ax2.set_xlabel('Side')
            ax2.set_ylabel('Contour')
            ax2.set_title('Angle Difference Heatmap')
            ax2.set_xticks(range(4))
            ax2.set_xticklabels(side_names)
            ax2.set_yticks(range(len(contours)))
            ax2.set_yticklabels(
                [f'Contour {i+1}' for i in range(len(contours))])

        # 3. Base contour shape comparison
        ax3 = axes[1, 0]
        Visualizer._draw_single_contour(
            ax3, base_contour, 'black', 'Base', base_angles)
        ax3.set_title('Base Contour')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # 4. All contours overlay
        ax4 = axes[1, 1]
        # Base contour
        Visualizer._draw_single_contour(
            ax4, base_contour, 'black', 'Base', base_angles, alpha=0.7)

        # Comparison contours
        for i, contour in enumerate(contours):
            if i < len(tilt_differences):
                contour_angles = base_angles + np.array(tilt_differences[i])
                Visualizer._draw_single_contour(ax4, contour, colors[i+1], f'Contour {i+1}',
                                                contour_angles, alpha=0.7)

        ax4.set_title('All Contours Overlay')
        ax4.legend()
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 5. Detailed side-by-side comparison
        Visualizer._show_detailed_side_comparison(
            base_angles, tilt_differences, side_names)

    @staticmethod
    def _draw_single_contour(ax, contour: np.ndarray, color: str, label: str,
                             angles: np.ndarray, alpha: float = 1.0) -> None:
        """Helper function to draw a single contour"""
        if len(contour) != 4:
            return

        # Draw contour as closed shape
        closed_contour = np.vstack([contour, contour[0:1]])
        ax.plot(closed_contour[:, 0], closed_contour[:, 1],
                color=color, linewidth=2, label=label, alpha=alpha)

        # Mark vertices
        ax.scatter(contour[:, 0], contour[:, 1],
                   color=color, s=50, alpha=alpha, zorder=5)

        # Display angle information
        for i in range(4):
            mid_point = (contour[i] + contour[(i + 1) % 4]) / 2
            ax.annotate(f'{angles[i]:.1f}°', mid_point,
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    @staticmethod
    def _show_detailed_side_comparison(base_angles: np.ndarray, tilt_differences: List,
                                       side_names: List[str]) -> None:
        """Function to show detailed comparison for each side"""
        if not tilt_differences:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detailed Angle Comparison by Side',
                     fontsize=14, fontweight='bold')

        axes = axes.flatten()
        colors = plt.cm.Set1(np.linspace(0, 1, len(tilt_differences) + 1))

        for side in range(4):
            ax = axes[side]

            # Base angle
            ax.bar(0, base_angles[side], color='black',
                   alpha=0.7, label='Base')

            # Comparison contours angles
            for i, diff in enumerate(tilt_differences):
                if len(diff) > side:
                    contour_angle = base_angles[side] + diff[side]
                    ax.bar(i + 1, contour_angle, color=colors[i+1], alpha=0.7,
                           label=f'Contour {i+1}')

                    # Display difference value
                    ax.annotate(f'{diff[side]:+.1f}°',
                                (i + 1, contour_angle),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))

            ax.set_title(f'{side_names[side]} Side')
            ax.set_ylabel('Angle (degrees)')
            ax.set_xlabel('Contour')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set x-axis labels
            x_labels = ['Base'] + \
                [f'C{i+1}' for i in range(len(tilt_differences))]
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)

        plt.tight_layout()
        plt.show()
