"""Service factory with MediaPipe as primary detection service."""

from app.services.mediapipe_detection_service import MediaPipeDetectionService
from app.services.measurement_service import MeasurementService
from app.services.analysis_service import AnalysisService
from app.utils.logger import get_logger
from app.core.development_mode import check_model_availability

logger = get_logger(__name__)


def create_services():
    """Create service instances with MediaPipe as production detection."""
    
    # Use MediaPipe for production-grade detection
    detection_service = MediaPipeDetectionService()
    
    # Create depth service with fallback
    try:
        model_status = check_model_availability()
        if model_status['depth_model_available']:
            from app.services.depth_service import DepthService
            depth_service = DepthService()
            logger.info("✅ Using real depth estimation model")
        else:
            from app.services.depth_service_dev import DepthServiceDev
            depth_service = DepthServiceDev()
            logger.info("⚠️ Using mock depth service")
    except Exception as e:
        logger.warning(f"Depth service creation failed: {e}, using mock")
        from app.services.depth_service_dev import DepthServiceDev
        depth_service = DepthServiceDev()
    
    measurement_service = MeasurementService(max_distance_threshold=0.01)
    
    logger.info("✅ Services created with MediaPipe detection (production-ready)")
    return depth_service, detection_service, measurement_service


def create_analysis_service():
    """Create analysis service with MediaPipe detection."""
    depth_service, detection_service, measurement_service = create_services()
    
    return AnalysisService(
        depth_service=depth_service,
        detection_service=detection_service,
        measurement_service=measurement_service
    )
