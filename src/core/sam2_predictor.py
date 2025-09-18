import numpy as np

from typing import List, Tuple

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.device_manager import DeviceManager
from utils.image_loader import ImageLoader


class SAM2Predictor:
    """
    SAM2 모델을 사용한 이미지 세그멘테이션 예측 클래스
    
    포인트 및 바운딩 박스 기반 객체 마스크 예측 기능 제공
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/sam2.1_hiera_large.pt",
                 config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml"):
        """
        SAM2 예측기 초기화
        
        Args:
            checkpoint_path: SAM2 모델 checkpoint 파일 경로
            config_path: SAM2 모델 config 파일 경로
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = DeviceManager.get_device()
        DeviceManager.configure_device(self.device)

        self.sam2_model = build_sam2(
            self.config_path, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def _load_and_preprocess_image(self, image_name: str) -> np.ndarray:
        """
        이미지 로드 및 전처리 수행
        
        Args:
            image_name: 이미지 파일명
            
        Returns:
            전처리된 이미지 numpy array (RGB format)
        """
        return ImageLoader.load_image(image_name)



    def _prepare_points_data(self, coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        포인트 좌표 데이터를 SAM2 모델 입력 형식으로 변환
        
        Args:
            coordinates: (x, y) 좌표 tuple의 list
            
        Returns:
            input_points: SAM2 모델용 포인트 배열
            input_labels: SAM2 모델용 레이블 배열
        """
        input_points = np.array([
            [[coord[0], coord[1]]] for coord in coordinates
        ])
        input_labels = np.array([[1] for _ in range(len(coordinates))])

        return input_points, input_labels

    def _prepare_boxes_data(self, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        바운딩 박스 데이터를 SAM2 모델 입력 형식으로 변환
        
        Args:
            boxes: (x1, y1, x2, y2) 형식의 bounding box tuple list
            
        Returns:
            SAM2 모델용 bounding box 배열
        """
        input_boxes = np.array([
            [box[0], box[1], box[2], box[3]] for box in boxes
        ])
        return input_boxes



    def _perform_points_prediction(self, input_points: np.ndarray, input_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        포인트 기반 SAM2 모델 예측 수행
        
        Args:
            input_points: 모델 입력용 포인트 좌표 배열
            input_labels: 모델 입력용 레이블 배열
            
        Returns:
            masks: 예측된 마스크 배열
            scores: 각 마스크의 신뢰도 점수
            logits: 마스크 예측 logit 값
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        print(f"Masks shape: {masks.shape}")
        print(f"Scores shape: {scores.shape}")

        return masks, scores, logits

    def _perform_boxes_prediction(self, input_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        바운딩 박스 기반 SAM2 모델 예측 수행
        
        Args:
            input_boxes: 모델 입력용 bounding box 배열
            
        Returns:
            masks: 예측된 마스크 배열
            scores: 각 마스크의 신뢰도 점수
            logits: 마스크 예측 logit 값
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        print(f"Masks shape: {masks.shape}")
        print(f"Scores shape: {scores.shape}")

        return masks, scores, logits

    def _select_best_masks(self, masks: np.ndarray, scores: np.ndarray, coordinates: List[Tuple[int, int]]) -> List[np.ndarray]:
        """
        신뢰도 점수 기반으로 최적 마스크 선택
        
        Args:
            masks: 예측된 마스크 배열
            scores: 각 마스크의 신뢰도 점수
            coordinates: 원본 입력 좌표 list
            
        Returns:
            각 좌표에 대한 최적 마스크 list
        """
        if len(coordinates) == 1:
            best_mask_index = np.argmax(scores)
            best_masks = [masks[best_mask_index]]
        else:
            best_mask_indices = np.argmax(scores, axis=1)
            best_masks = [masks[i][best_mask_indices[i]]
                          for i in range(len(coordinates))]

        return best_masks

    def predict_masks_from_points(self, image_name: str, coordinates: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        포인트 좌표 기반으로 객체 마스크 예측
        
        Args:
            image_name: 입력 이미지 파일명
            coordinates: 클릭한 포인트 좌표 list
            
        Returns:
            image: 원본 이미지 array
            best_masks: 예측된 최적 마스크 list
        """
        # 이미지 로드 및 전처리
        image = self._load_and_preprocess_image(image_name)
        self.predictor.set_image(image)

        # 포인트 데이터 준비
        input_points, input_labels = self._prepare_points_data(coordinates)

        # 예측 수행
        masks, scores, logits = self._perform_points_prediction(
            input_points, input_labels)

        # 최고 점수 마스크 선택
        best_masks = self._select_best_masks(masks, scores, coordinates)

        return image, best_masks

    def predict_masks_from_boxes(self, image_name: str, boxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        바운딩 박스 기반으로 객체 마스크 예측
        
        Args:
            image_name: 입력 이미지 파일명
            boxes: bounding box 좌표 list (x1, y1, x2, y2)
            
        Returns:
            image: 원본 이미지 array
            best_masks: 예측된 최적 마스크 list
        """
        # 이미지 로드 및 전처리
        image = self._load_and_preprocess_image(image_name)
        self.predictor.set_image(image)

        # 박스 데이터 준비
        input_boxes = self._prepare_boxes_data(boxes)

        # 예측 수행
        masks, scores, logits = self._perform_boxes_prediction(input_boxes)

        # 최고 점수 마스크 선택
        if len(boxes) == 1:
            best_masks = [masks.squeeze(0)]
        else:
            best_masks = [masks[i].squeeze(0) for i in range(len(boxes))]

        return image, best_masks
