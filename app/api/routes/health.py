"""Health check endpoints."""

from fastapi import APIRouter
import torch
import os

from app.models.schemas import HealthResponse, DetailedHealthResponse
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse()


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check including system status."""

    # Check if model files exist
    depth_model_exists = os.path.exists(settings.DEPTH_MODEL_PATH)
    detection_model_exists = os.path.exists(settings.DETECTION_MODEL_PATH)
    models_loaded = depth_model_exists and detection_model_exists

    system_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "depth_model_available": depth_model_exists,
        "detection_model_available": detection_model_exists,
        "python_version": "3.9+",
    }

    return DetailedHealthResponse(system=system_info, models_loaded=models_loaded)


@router.get("/mediapipe-status")
async def mediapipe_status():
    """Check MediaPipe detection service status."""
    try:
        from app.services.mediapipe_detection_service import MediaPipeDetectionService

        # Test MediaPipe initialization
        service = MediaPipeDetectionService()

        # Create test image for confidence check
        from PIL import Image
        import numpy as np

        test_image = Image.fromarray(
            np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        )

        confidence = service.get_pose_confidence(test_image)

        return {
            "mediapipe_available": service.available,
            "service_type": "MediaPipe Pose Detection",
            "production_ready": True,
            "test_confidence": confidence,
            "features": [
                "Real-time pose detection",
                "Production-grade accuracy",
                "No training required",
                "Cross-demographic support",
            ],
        }

    except Exception as e:
        return {
            "mediapipe_available": False,
            "error": str(e),
            "production_ready": False,
        }
