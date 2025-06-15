"""Development mode utilities for BodyVision."""

import os
from app.utils.logger import get_logger

logger = get_logger(__name__)

def is_development_mode():
    """Check if running in development mode (models may not be available)."""
    return os.getenv('BODYVISION_DEV_MODE', 'false').lower() == 'true'

def check_model_availability():
    """Check which models are available."""
    depth_model = "app/models/weights/best_depth_Ours_Bilinear_inc_3_net_G.pth"
    detection_model = "app/models/weights/csv_retinanet_25.pt"
    
    depth_available = os.path.exists(depth_model)
    detection_available = os.path.exists(detection_model)
    
    status = {
        'depth_model_available': depth_available,
        'detection_model_available': detection_available,
        'all_models_available': depth_available and detection_available
    }
    
    if not status['all_models_available']:
        logger.warning("Some models are missing - running in development mode")
        if not depth_available:
            logger.warning(f"Missing depth model: {depth_model}")
        if not detection_available:
            logger.warning(f"Missing detection model: {detection_model}")
    
    return status
