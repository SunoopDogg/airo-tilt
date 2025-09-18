import os
import cv2
import time

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any, Union
from utils.aruco_marker import ArUcoMarkerDetector


# Global configuration variables
VIDEO_SOURCE = 'http://192.168.0.53:4747/video'
MODE = 'realtime'  # 'realtime' or 'image'
MARKER_SIZE_MM = 36.0
DICT_TYPE = 'DICT_4X4_50'
DISPLAY_INFO = True
RECORD_OUTPUT = False
IMAGE_PATH = 'images/aruco_test_04.png'  # Used in 'image' mode


class RealtimeArUcoDetector:
    """Real-time ArUco marker detection from IP webcam or local camera"""

    # Connection settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BUFFER_SIZE = 1  # for IP cameras to reduce latency

    # Display settings
    PANEL_HEIGHT = 80
    PANEL_OPACITY = 0.3
    MAX_MARKERS_DISPLAY = 3

    # Text rendering settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.7
    FONT_SCALE_MEDIUM = 0.6
    FONT_SCALE_SMALL = 0.5
    FONT_SCALE_TINY = 0.4
    FONT_THICKNESS_THICK = 2
    FONT_THICKNESS_NORMAL = 1

    # Colors (BGR format)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (200, 200, 200)
    COLOR_BLACK = (0, 0, 0)

    # Video recording settings
    VIDEO_FPS = 20.0
    VIDEO_CODEC = 'mp4v'

    def __init__(self,
                 video_source: Union[str, int] = None,
                 marker_size_mm: float = None,
                 aruco_dict_type: str = None,
                 display_info: bool = None,
                 record_output: bool = None):
        """
        Initialize real-time ArUco detector

        Args:
            video_source: Video source URL or camera index
            marker_size_mm: Physical size of the marker in millimeters
            aruco_dict_type: Type of ArUco dictionary to use
            display_info: Whether to display marker information on screen
            record_output: Whether to record the output video
        """
        # Use global variables if parameters not provided
        self.video_source = video_source if video_source is not None else VIDEO_SOURCE
        self.display_info = display_info if display_info is not None else DISPLAY_INFO
        self.record_output = record_output if record_output is not None else RECORD_OUTPUT

        marker_size_mm = marker_size_mm if marker_size_mm is not None else MARKER_SIZE_MM
        aruco_dict_type = aruco_dict_type if aruco_dict_type is not None else DICT_TYPE

        # Initialize ArUco detector
        self.detector = ArUcoMarkerDetector(
            aruco_dict_type=aruco_dict_type,
            marker_size_mm=marker_size_mm
        )

        # Video capture object
        self.cap = None
        self.video_writer = None

        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Detection statistics
        self.marker_history = {}  # Track marker appearances

    def connect_to_stream(self) -> bool:
        """
        Connect to video stream with retry mechanism

        Returns:
            True if connection successful, False otherwise
        """
        retry_count = 0

        while retry_count < self.MAX_RETRIES:
            try:
                print(f"Attempting to connect to: {self.video_source}")

                # Try to open video stream
                if isinstance(self.video_source, str) and self.video_source.startswith('http'):
                    # IP webcam stream
                    self.cap = cv2.VideoCapture(self.video_source)
                else:
                    # Local camera (convert to int if string number)
                    if isinstance(self.video_source, str) and self.video_source.isdigit():
                        self.cap = cv2.VideoCapture(int(self.video_source))
                    else:
                        self.cap = cv2.VideoCapture(self.video_source)

                # Set buffer size to reduce latency for IP cameras
                if isinstance(self.video_source, str) and self.video_source.startswith('http'):
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.BUFFER_SIZE)

                # Check if stream opened successfully
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Successfully connected to stream")
                        print(
                            f"Stream resolution: {frame.shape[1]}x{frame.shape[0]}")

                        # Initialize video writer if recording
                        if self.record_output:
                            self.init_video_writer(
                                frame.shape[1], frame.shape[0])

                        return True

            except Exception as e:
                print(f"Connection attempt {retry_count + 1} failed: {e}")

            retry_count += 1
            if retry_count < self.MAX_RETRIES:
                print(f"Retrying in {self.RETRY_DELAY} seconds...")
                time.sleep(self.RETRY_DELAY)

        print("Failed to connect to video stream")
        return False

    def init_video_writer(self, width: int, height: int):
        """
        Initialize video writer for recording output

        Args:
            width: Frame width
            height: Frame height
        """
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/aruco_detection_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*self.VIDEO_CODEC)
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.VIDEO_FPS, (width, height))
        print(f"Recording output to: {output_path}")

    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def draw_info_panel(self, frame: np.ndarray, markers_info: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw information panel on the frame

        Args:
            frame: Input frame
            markers_info: List of detected marker information

        Returns:
            Frame with information panel
        """
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], self.PANEL_HEIGHT), self.COLOR_BLACK, -1)
        frame = cv2.addWeighted(
            overlay, self.PANEL_OPACITY, frame, 1 - self.PANEL_OPACITY, 0)

        # Draw FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                    self.FONT, self.FONT_SCALE_LARGE, self.COLOR_GREEN, self.FONT_THICKNESS_THICK)

        # Draw marker information
        if markers_info:
            markers_text = f"Markers Found: {len(markers_info)}"
            cv2.putText(frame, markers_text, (10, 60),
                        self.FONT, self.FONT_SCALE_MEDIUM, self.COLOR_YELLOW, self.FONT_THICKNESS_THICK)

            # Draw individual marker info
            x_offset = 250
            for i, marker in enumerate(markers_info[:self.MAX_MARKERS_DISPLAY]):
                marker_text = f"ID:{marker['id']} Dist:{marker['distance']:.2f}m"
                cv2.putText(frame, marker_text, (x_offset + i*200, 30),
                            self.FONT, self.FONT_SCALE_SMALL, self.COLOR_YELLOW, self.FONT_THICKNESS_NORMAL)

                angles_text = f"R:{marker['roll']:.1f} P:{marker['pitch']:.1f} Y:{marker['yaw']:.1f}"
                cv2.putText(frame, angles_text, (x_offset + i*200, 50),
                            self.FONT, self.FONT_SCALE_TINY, self.COLOR_YELLOW, self.FONT_THICKNESS_NORMAL)

        # Draw controls
        controls_text = "Controls: [Q] Quit | [S] Screenshot | [R] Reset Stats"
        cv2.putText(frame, controls_text, (frame.shape[1] - 400, frame.shape[0] - 20),
                    self.FONT, self.FONT_SCALE_SMALL, self.COLOR_GRAY, self.FONT_THICKNESS_NORMAL)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame for ArUco detection

        Args:
            frame: Input frame

        Returns:
            Processed frame and marker information list
        """
        # Detect markers
        corners, ids, rejected = self.detector.detect_markers(frame)

        markers_info = []

        if corners and ids is not None:
            # Estimate poses
            r_vectors, t_vectors = self.detector.estimate_pose(corners)

            # Draw markers with axes
            frame = self.detector.draw_markers(
                frame, corners, ids, r_vectors, t_vectors)

            # Collect marker information
            for i, marker_id in enumerate(ids):
                marker_id_val = int(marker_id[0])

                # Update marker history
                if marker_id_val not in self.marker_history:
                    self.marker_history[marker_id_val] = 0
                self.marker_history[marker_id_val] += 1

                # Get marker details
                if i < len(r_vectors) and i < len(t_vectors):
                    distance = self.detector.calculate_marker_distance(
                        t_vectors[i])
                    roll, pitch, yaw = self.detector.get_marker_orientation(
                        r_vectors[i])

                    marker_info = {
                        'id': marker_id_val,
                        'distance': distance,
                        'roll': roll,
                        'pitch': pitch,
                        'yaw': yaw,
                        'corners': corners[i][0]
                    }
                    markers_info.append(marker_info)

                    # Draw distance on marker
                    center = np.mean(corners[i][0], axis=0).astype(int)
                    distance_text = f"{distance:.2f}m"
                    cv2.putText(frame, distance_text,
                                (center[0] - 20, center[1] + 40),
                                self.FONT, self.FONT_SCALE_SMALL, self.COLOR_GREEN, self.FONT_THICKNESS_THICK)

        return frame, markers_info

    def save_screenshot(self, frame: np.ndarray):
        """Save a screenshot of the current frame"""
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/aruco_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def reset_statistics(self):
        """Reset detection statistics"""
        self.marker_history.clear()
        print("Statistics reset")

    def run(self):
        """Main loop for real-time detection"""
        # Connect to stream
        if not self.connect_to_stream():
            return

        print("\n=== Real-time ArUco Detection Started ===")
        print("Press 'Q' to quit")
        print("Press 'S' to save screenshot")
        print("Press 'R' to reset statistics")
        print("=" * 40)

        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Failed to read frame, attempting to reconnect...")
                    if not self.connect_to_stream():
                        break
                    continue

                # Process frame
                processed_frame, markers_info = self.process_frame(frame)

                # Calculate FPS
                self.calculate_fps()

                # Draw info panel if enabled
                if self.display_info:
                    processed_frame = self.draw_info_panel(
                        processed_frame, markers_info)

                # Record frame if enabled
                if self.record_output and self.video_writer:
                    self.video_writer.write(processed_frame)

                # Display frame
                cv2.imshow('ArUco Real-time Detection', processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_screenshot(processed_frame)
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        except Exception as e:
            print(f"Error during processing: {e}")

        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")

        # Print final statistics
        print("\n=== Final Statistics ===")
        print(f"Unique markers detected: {len(self.marker_history)}")
        if self.marker_history:
            print("Marker frequency:")
            for marker_id, count in sorted(self.marker_history.items()):
                print(f"  Marker {marker_id}: {count} detections")

        # Release resources
        if self.cap:
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()
            print("Video recording saved")

        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main function for ArUco marker detection"""

    if MODE == 'image' and IMAGE_PATH:
        # Process single image
        detector = ArUcoMarkerDetector(
            aruco_dict_type=DICT_TYPE,
            marker_size_mm=MARKER_SIZE_MM
        )

        print(f"Processing image: {IMAGE_PATH}")
        try:
            results = detector.detect_and_analyze(IMAGE_PATH)

            print(f"Detected {results['markers_detected']} markers")
            for marker in results['markers']:
                print(f"\nMarker ID: {marker['id']}")
                print(f"  Bounding Box: {marker['bounding_box']}")
                if 'distance_m' in marker:
                    print(f"  Distance: {marker['distance_m']:.3f} m")
                    print(f"  Rotation: Roll={marker['rotation']['roll']:.1f}°, "
                          f"Pitch={marker['rotation']['pitch']:.1f}°, "
                          f"Yaw={marker['rotation']['yaw']:.1f}°")

            # Save visualization
            if 'visualization' in results:
                output_path = "output/aruco_detection_result.jpg"
                os.makedirs("output", exist_ok=True)
                cv2.imwrite(output_path, results['visualization'])
                print(f"\nVisualization saved to {output_path}")

                # Display result using matplotlib
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(
                    results['visualization'], cv2.COLOR_BGR2RGB)

                ax.imshow(img_rgb)
                ax.set_title(
                    f"ArUco Detection Result - {results['markers_detected']} markers detected", fontsize=14)
                ax.axis('off')

                # Add text information for each marker
                info_text = []
                for marker in results['markers']:
                    text = f"ID {marker['id']}"
                    if 'distance_m' in marker:
                        text += f": {marker['distance_m']:.3f}m"
                        text += f" (R:{marker['rotation']['roll']:.0f}° P:{marker['rotation']['pitch']:.0f}° Y:{marker['rotation']['yaw']:.0f}°)"
                    info_text.append(text)

                if info_text:
                    # Add marker information as text below the image
                    fig.text(0.5, 0.02, " | ".join(info_text),
                             ha='center', fontsize=10, wrap=True)

                plt.tight_layout()
                plt.show()
                print("Close the matplotlib window to continue...")

        except Exception as e:
            print(f"Error processing image: {e}")

    else:
        # Real-time detection mode (default)
        print(f"Starting real-time ArUco detection...")
        print(f"Source: {VIDEO_SOURCE}")
        print(f"Marker size: {MARKER_SIZE_MM}mm")
        print(f"Dictionary: {DICT_TYPE}")

        # Create and run real-time detector
        detector = RealtimeArUcoDetector(
            video_source=VIDEO_SOURCE,
            marker_size_mm=MARKER_SIZE_MM,
            aruco_dict_type=DICT_TYPE,
            display_info=DISPLAY_INFO,
            record_output=RECORD_OUTPUT
        )

        detector.run()


if __name__ == "__main__":
    main()
