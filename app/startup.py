"""Application startup utilities."""

import logging
from typing import Dict, Any

from app.models.weights import initialize_models
from app.utils.logger import get_logger

logger = get_logger(__name__)

def initialize_application() -> Dict[str, Any]:
    """Initialize the BodyVision application."""
    logger.info("🚀 Initializing BodyVision application...")
    
    startup_status = {
        'models': {},
        'services': {},
        'errors': []
    }
    
    try:
        # Initialize models
        logger.info("📦 Loading models...")
        model_status = initialize_models()
        startup_status['models'] = model_status
        
        # Initialize services
        logger.info("🔧 Initializing services...")
        startup_status['services']['mediapipe'] = 'ready'
        startup_status['services']['analysis'] = 'ready'
        startup_status['services']['measurement'] = 'ready'
        
        logger.info("✅ BodyVision application initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        startup_status['errors'].append(str(e))
    
    return startup_status

def get_application_status() -> Dict[str, Any]:
    """Get current application status."""
    from app.models.weights import get_model_loader
    
    model_loader = get_model_loader()
    
    return {
        'models': model_loader.get_model_status(),
        'available_models': model_loader.get_available_models(),
        'core_features': {
            'three_photo_analysis': True,
            'mediapipe_detection': True,
            'navy_formula': True,
            'enhanced_metrics': True
        }
    }
