"""Model loading utilities with graceful degradation for missing files."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading of ML models with graceful degradation."""
    
    def __init__(self):
        self.models_dir = Path("app/models/weights")
        self.loaded_models = {}
        self.model_status = {}
        
    def load_depth_model(self) -> Optional[Any]:
        """Load depth estimation model with graceful failure."""
        model_path = self.models_dir / "best_depth_Ours_Bilinear_inc_3_net_G.pth"
        
        try:
            if model_path.exists():
                logger.info(f"Loading depth model from {model_path}")
                model = torch.load(model_path, map_location='cpu')
                self.loaded_models['depth'] = model
                self.model_status['depth'] = 'loaded'
                logger.info("✅ Depth model loaded successfully")
                return model
            else:
                logger.warning(f"⚠️ Depth model not found at {model_path}")
                self.model_status['depth'] = 'missing'
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load depth model: {e}")
            self.model_status['depth'] = 'error'
            return None
    
    def load_detection_model(self) -> Optional[Any]:
        """Load detection model with graceful failure."""
        model_path = self.models_dir / "csv_retinanet_25.pt"
        
        try:
            if model_path.exists():
                logger.info(f"Loading detection model from {model_path}")
                model = torch.load(model_path, map_location='cpu')
                self.loaded_models['detection'] = model
                self.model_status['detection'] = 'loaded'
                logger.info("✅ Detection model loaded successfully")
                return model
            else:
                logger.warning(f"⚠️ Detection model not found at {model_path}")
                self.model_status['detection'] = 'missing'
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load detection model: {e}")
            self.model_status['detection'] = 'error'
            return None
    
    def get_model_status(self) -> Dict[str, str]:
        """Get status of all models."""
        return self.model_status.copy()
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return self.model_status.get(model_name) == 'loaded'
    
    def get_available_models(self) -> list:
        """Get list of successfully loaded models."""
        return [name for name, status in self.model_status.items() if status == 'loaded']
    
    def initialize_all_models(self) -> Dict[str, str]:
        """Initialize all models and return status report."""
        logger.info("🚀 Initializing BodyVision models...")
        
        # MediaPipe is handled separately (auto-downloads)
        self.model_status['mediapipe'] = 'auto_managed'
        
        # Try to load optional models
        self.load_depth_model()
        self.load_detection_model()
        
        # Report status
        status_report = self.get_model_status()
        
        loaded_count = sum(1 for s in status_report.values() if s in ['loaded', 'auto_managed'])
        total_count = len(status_report)
        
        logger.info(f"📊 Model loading complete: {loaded_count}/{total_count} available")
        
        for model_name, status in status_report.items():
            if status == 'loaded':
                logger.info(f"✅ {model_name}: Ready")
            elif status == 'auto_managed':
                logger.info(f"🤖 {model_name}: Auto-managed")
            elif status == 'missing':
                logger.warning(f"⚠️ {model_name}: Model file not found (will use fallback)")
            elif status == 'error':
                logger.error(f"❌ {model_name}: Failed to load")
        
        return status_report

# Global model loader instance
model_loader = ModelLoader()

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return model_loader

def initialize_models() -> Dict[str, str]:
    """Initialize all models (convenience function)."""
    return model_loader.initialize_all_models()
