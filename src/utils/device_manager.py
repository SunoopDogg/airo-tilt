import torch


class DeviceManager:
    """
    PyTorch 디바이스 관리 클래스
    """
    
    @staticmethod
    def get_device() -> torch.device:
        """
        최적 연산 디바이스 자동 선택
        
        Returns:
            사용 가능한 최적의 torch.device (CUDA, MPS, 또는 CPU)
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"using device: {device}")
        return device

    @staticmethod
    def configure_device(device: torch.device) -> None:
        """
        디바이스별 최적화 옵션 설정 적용
        
        Args:
            device: 설정할 torch.device
        """
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
