import numpy as np

from typing import List, Tuple

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.device_manager import DeviceManager
from utils.image_loader import ImageLoader


class SAM2Predictor:
    def __init__(self, checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
                 config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml"):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = DeviceManager.get_device()
        DeviceManager.configure_device(self.device)

        self.sam2_model = build_sam2(
            self.config_path, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def _load_and_preprocess_image(self, image_name: str) -> np.ndarray:
        """이미지를 로드하고 전처리합니다."""
        return ImageLoader.load_image(image_name)

    def _prepare_input_data(self, coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """좌표 데이터를 모델 입력 형식으로 변환합니다."""
        input_points = np.array([
            [[coord[0], coord[1]]] for coord in coordinates
        ])
        input_labels = np.array([[1] for _ in range(len(coordinates))])

        return input_points, input_labels

    def _perform_prediction(self, input_points: np.ndarray, input_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SAM2 모델을 사용해 예측을 수행합니다."""
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        # (batch_size) x (num_predicted_masks_per_input) x H x W
        print(f"Masks shape: {masks.shape}")

        return masks, scores, logits

    def _select_best_masks(self, masks: np.ndarray, scores: np.ndarray, coordinates: List[Tuple[int, int]]) -> List[np.ndarray]:
        """가장 높은 점수의 마스크들을 선택합니다."""
        best_mask_indices = np.argmax(scores, axis=1)
        best_masks = [masks[i][best_mask_indices[i]]
                      for i in range(len(coordinates))]
        return best_masks

    def predict_masks(self, image_name: str, coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """이미지와 좌표를 기반으로 마스크를 예측합니다."""
        # 이미지 로드 및 전처리
        image = self._load_and_preprocess_image(image_name)
        self.predictor.set_image(image)

        # 좌표 데이터 준비
        input_points, input_labels = self._prepare_input_data(coordinates)

        # 예측 수행
        masks, scores, logits = self._perform_prediction(
            input_points, input_labels)

        # 최고 점수 마스크 선택
        best_masks = self._select_best_masks(masks, scores, coordinates)

        return image, best_masks
