"""
디바이스 유틸리티 모듈
CUDA GPU 가속을 지원하는 유틸리티 함수들을 제공합니다.
"""

import logging

logger = logging.getLogger(__name__)

def get_device():
    """
    사용 가능한 최적의 디바이스를 반환합니다.
    우선순위: CUDA GPU > CPU
    
    Returns:
        torch.device: 사용할 디바이스 객체
    """
    try:
        import torch  # type: ignore
        has_cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
        if has_cuda:
            device = torch.device("cuda")
            try:
                logger.info(f"CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
            except Exception:
                logger.info("CUDA GPU 사용")
        else:
            device = torch.device("cpu")
            logger.info("CPU 사용")
        return device
    except Exception:
        class _CPUDevice:
            def __str__(self):
                return "cpu"
        logger.info("torch 미설치 또는 초기화 실패로 CPU 사용")
        return _CPUDevice()


def get_device_info():
    """
    현재 시스템의 디바이스 정보를 반환합니다.
    
    Returns:
        dict: 디바이스 정보를 담은 딕셔너리
    """
    try:
        import torch  # type: ignore
        info = {
            "cuda_available": hasattr(torch, 'cuda') and torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if hasattr(torch, 'cuda') and torch.cuda.is_available() else "N/A",
            "cuda_device_count": torch.cuda.device_count() if hasattr(torch, 'cuda') else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if hasattr(torch, 'cuda') and torch.cuda.device_count() > 0 else "N/A",
            "torch_version": getattr(torch, "__version__", "unknown"),
            "recommended_device": str(get_device())
        }
    except Exception:
        info = {
            "cuda_available": False,
            "cuda_version": "N/A",
            "cuda_device_count": 0,
            "cuda_device_name": "N/A",
            "torch_version": "not_installed",
            "recommended_device": "cpu"
        }
    return info


def to_device(model, device=None):
    """
    모델을 지정된 디바이스로 이동합니다.
    
    Args:
        model: PyTorch 모델
        device: 이동할 디바이스 (None인 경우 get_device() 사용)
        
    Returns:
        device로 이동된 모델
    """
    if device is None:
        device = get_device()
    try:
        import torch  # type: ignore
        if hasattr(model, 'to'):
            return model.to(device)
        return model
    except Exception:
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info = get_device_info()
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    device = get_device()
    logger.info(f"권장 디바이스: {device}")
    
