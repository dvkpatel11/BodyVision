"""Production service factory for MediaPipe-based body analysis."""

from app.services.mediapipe_detection_service import MediaPipeDetectionService
from app.services.measurement_service import MeasurementService
from app.services.analysis_service import AnalysisService
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_production_services():
    """Create production-ready service instances."""
    
    logger.info("🏭 Creating production services for BodyVision")
    
    # MediaPipe detection service (production-grade)
    detection_service = MediaPipeDetectionService()
    
    # MediaPipe-native measurement service
    measurement_service = MeasurementService()
    
    logger.info("✅ Production services created successfully")
    return detection_service, measurement_service


def create_analysis_service():
    """Create the main analysis service for production."""
    detection_service, measurement_service = create_production_services()
    
    return AnalysisService(
        detection_service=detection_service,
        measurement_service=measurement_service
    )
