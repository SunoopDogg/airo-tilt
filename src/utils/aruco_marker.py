import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class ArUcoMarkerDetector:
    """ArUco marker detection and pose estimation class"""

    # Default values
    DEFAULT_MARKER_SIZE_MM = 36.0
    DEFAULT_DICT_TYPE = "DICT_4X4_50"
    DEFAULT_MARKER_IMAGE_SIZE = 200

    # Camera calibration defaults
    DEFAULT_FOCAL_LENGTH = 800
    DEFAULT_IMAGE_WIDTH = 640
    DEFAULT_IMAGE_HEIGHT = 480

    # Visualization parameters
    AXIS_LENGTH_RATIO = 0.5  # Axis length as ratio of marker size

    # ArUco dictionary types
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    def __init__(self,
                 aruco_dict_type: str = None,
                 marker_size_mm: float = None,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize ArUco marker detector

        Args:
            aruco_dict_type: Type of ArUco dictionary to use (defaults to DICT_4X4_50)
            marker_size_mm: Physical size of the marker in millimeters (defaults to 36mm)
            camera_matrix: Camera intrinsic matrix for pose estimation
            dist_coeffs: Camera distortion coefficients
        """
        self.aruco_dict_type = aruco_dict_type or self.DEFAULT_DICT_TYPE
        self.marker_size_mm = marker_size_mm or self.DEFAULT_MARKER_SIZE_MM
        # Convert to meters for pose estimation
        self.marker_size_m = marker_size_mm / 1000.0

        # Initialize ArUco dictionary and detector
        if aruco_dict_type in self.ARUCO_DICT:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(
                self.ARUCO_DICT[aruco_dict_type])
        else:
            raise ValueError(
                f"Unknown ArUco dictionary type: {aruco_dict_type}")

        # Initialize detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.detector_params)

        # Camera calibration parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        # Default camera matrix if not provided (can be updated with calibration)
        if self.camera_matrix is None:
            # Default intrinsic parameters (should be calibrated for accurate pose estimation)
            cx = self.DEFAULT_IMAGE_WIDTH / 2
            cy = self.DEFAULT_IMAGE_HEIGHT / 2
            self.camera_matrix = np.array([[self.DEFAULT_FOCAL_LENGTH, 0, cx],
                                          [0, self.DEFAULT_FOCAL_LENGTH, cy],
                                          [0, 0, 1]], dtype=float)

        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1))

    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """
        Detect ArUco markers in an image

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            corners: List of detected marker corners
            ids: List of marker IDs
            rejected: List of rejected candidate markers
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        return corners, ids, rejected

    def estimate_pose(self, corners: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Estimate 3D pose of detected markers

        Args:
            corners: List of marker corners from detect_markers

        Returns:
            r_vectors: Rotation vectors for each marker
            t_vectors: Translation vectors for each marker
        """
        if not corners:
            return [], []

        r_vectors = []
        t_vectors = []

        # Define 3D object points for a square marker
        obj_points = np.array([
            [-self.marker_size_m/2, self.marker_size_m/2, 0],
            [self.marker_size_m/2, self.marker_size_m/2, 0],
            [self.marker_size_m/2, -self.marker_size_m/2, 0],
            [-self.marker_size_m/2, -self.marker_size_m/2, 0]
        ], dtype=np.float32)

        for corner in corners:
            # Estimate pose for each marker
            success, r_vector, t_vector = cv2.solvePnP(
                obj_points,
                corner[0],
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                r_vectors.append(r_vector)
                t_vectors.append(t_vector)

        return r_vectors, t_vectors

    def draw_markers(self,
                     image: np.ndarray,
                     corners: List[np.ndarray],
                     ids: Optional[np.ndarray],
                     r_vectors: Optional[List[np.ndarray]] = None,
                     t_vectors: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Draw detected markers with IDs and optional pose axes

        Args:
            image: Input image
            corners: Detected marker corners
            ids: Marker IDs
            r_vectors: Rotation vectors (optional, for drawing axes)
            t_vectors: Translation vectors (optional, for drawing axes)

        Returns:
            Image with drawn markers
        """
        img_copy = image.copy()

        if corners and ids is not None:
            # Draw marker borders
            cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)

            # Draw pose axes if available
            if r_vectors is not None and t_vectors is not None:
                axis_length = self.marker_size_m * self.AXIS_LENGTH_RATIO
                for r_vector, t_vector in zip(r_vectors, t_vectors):
                    cv2.drawFrameAxes(
                        img_copy,
                        self.camera_matrix,
                        self.dist_coeffs,
                        r_vector,
                        t_vector,
                        axis_length
                    )

        return img_copy

    def get_marker_corners_as_boxes(self, corners: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Convert marker corners to bounding boxes for SAM2 integration

        Args:
            corners: List of marker corners

        Returns:
            List of bounding boxes as (x_min, y_min, x_max, y_max)
        """
        boxes = []
        for corner in corners:
            points = corner[0]
            x_min = int(np.min(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            x_max = int(np.max(points[:, 0]))
            y_max = int(np.max(points[:, 1]))
            boxes.append((x_min, y_min, x_max, y_max))

        return boxes

    def calculate_marker_distance(self, t_vector: np.ndarray) -> float:
        """
        Calculate distance from camera to marker

        Args:
            t_vector: Translation vector from pose estimation

        Returns:
            Distance in meters
        """
        return np.linalg.norm(t_vector)

    def get_marker_orientation(self, r_vector: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract rotation angles from rotation vector

        Args:
            vector: Rotation vector from pose estimation

        Returns:
            Euler angles (roll, pitch, yaw) in degrees
        """
        # Convert rotation vector to rotation matrix
        r_matrix, _ = cv2.Rodrigues(r_vector)

        # Extract Euler angles
        sy = np.sqrt(r_matrix[0, 0]**2 + r_matrix[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(r_matrix[2, 1], r_matrix[2, 2])
            y = np.arctan2(-r_matrix[2, 0], sy)
            z = np.arctan2(r_matrix[1, 0], r_matrix[0, 0])
        else:
            x = np.arctan2(-r_matrix[1, 2], r_matrix[1, 1])
            y = np.arctan2(-r_matrix[2, 0], sy)
            z = 0

        # Convert to degrees
        roll = np.degrees(x)
        pitch = np.degrees(y)
        yaw = np.degrees(z)

        return roll, pitch, yaw

    def calibrate_camera(self,
                         calibration_images: List[str],
                         board_corners: Tuple[int, int],
                         square_size_mm: float = 36.0,
                         marker_size_mm: float = 27.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate camera using ArUco board images

        Args:
            calibration_images: List of paths to calibration images
            board_corners: Number of corners in the board (cols, rows)
            square_size_mm: Size of board squares in mm
            marker_size_mm: Size of markers in mm

        Returns:
            camera_matrix: Calibrated camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            reprojection_error: Average reprojection error
        """
        # Create ArUco board
        board = cv2.aruco.GridBoard(
            board_corners,
            square_size_mm / 1000.0,
            marker_size_mm / 1000.0,
            self.aruco_dict
        )

        all_corners = []
        all_ids = []
        image_size = None

        # Detect markers in all calibration images
        for img_path in calibration_images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detect_markers(gray)

            if ids is not None and len(ids) > 0:
                all_corners.append(corners)
                all_ids.append(ids)

                if image_size is None:
                    image_size = gray.shape[::-1]

        if not all_corners:
            raise ValueError("No markers detected in calibration images")

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, r_vectors, t_vectors = cv2.aruco.calibrateCameraAruco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None
        )

        # Update internal parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        return camera_matrix, dist_coeffs, ret

    def save_calibration(self, filepath: str):
        """
        Save camera calibration parameters to file

        Args:
            filepath: Path to save calibration data
        """
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'marker_size_mm': self.marker_size_mm,
            'aruco_dict_type': self.aruco_dict_type
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

    def load_calibration(self, filepath: str):
        """
        Load camera calibration parameters from file

        Args:
            filepath: Path to calibration data file
        """
        import json
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)

        self.camera_matrix = np.array(calibration_data['camera_matrix'])
        self.dist_coeffs = np.array(calibration_data['dist_coeffs'])
        self.marker_size_mm = calibration_data.get('marker_size_mm', 36.0)
        self.marker_size_m = self.marker_size_mm / 1000.0

    def detect_and_analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive detection and analysis of ArUco markers in an image

        Args:
            image_path: Path to the input image

        Returns:
            Dictionary containing detection results and analysis

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be read or decoded
        """
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(
                f"Could not read or decode image from {image_path}. "
                f"Please ensure it's a valid image file."
            )

        # Detect markers
        corners, ids, rejected = self.detect_markers(image)

        results = {
            'image_path': image_path,
            'image_shape': image.shape,
            'markers_detected': len(ids) if ids is not None else 0,
            'markers': []
        }

        if corners and ids is not None:
            # Estimate poses
            r_vectors, t_vectors = self.estimate_pose(corners)

            # Get bounding boxes for SAM2 integration
            boxes = self.get_marker_corners_as_boxes(corners)

            # Analyze each marker
            for i, marker_id in enumerate(ids):
                marker_info = {
                    'id': int(marker_id[0]),
                    'corners': corners[i][0].tolist(),
                    'bounding_box': boxes[i],
                }

                if i < len(r_vectors):
                    distance = self.calculate_marker_distance(t_vectors[i])
                    roll, pitch, yaw = self.get_marker_orientation(
                        r_vectors[i])

                    marker_info.update({
                        'distance_m': float(distance),
                        'rotation': {
                            'roll': float(roll),
                            'pitch': float(pitch),
                            'yaw': float(yaw)
                        },
                        'position': t_vectors[i].flatten().tolist(),
                        'rotation_vector': r_vectors[i].flatten().tolist()
                    })

                results['markers'].append(marker_info)

            # Create visualization
            result_image = self.draw_markers(
                image, corners, ids, r_vectors, t_vectors)
            results['visualization'] = result_image

        return results
