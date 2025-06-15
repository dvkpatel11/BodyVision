"""Development-friendly detection service with graceful fallbacks."""

import os
from typing import Dict, Optional
from PIL import Image
import numpy as np

from app.utils.logger import get_logger
from app.core.development_mode import check_model_availability

logger = get_logger(__name__)

class DetectionServiceDev:
    """Detection service with development mode support."""
    
    def __init__(self, model_path: Optional[str] = None, classes_path: Optional[str] = None):
        self.model_path = model_path or 'app/models/weights/csv_retinanet_25.pt'
        self.classes_path = classes_path or 'config/classes.csv'
        self.model_status = check_model_availability()
        
        if self.model_status['detection_model_available']:
            # Import and use real detection service
            try:
                from app.services.detection_service import DetectionService
                self.real_service = DetectionService(model_path, classes_path)
                self.use_real_model = True
                logger.info("Using real RetinaNet detection model")
            except Exception as e:
                logger.warning(f"Failed to load real model, using mock: {e}")
                self.use_real_model = False
        else:
            self.use_real_model = False
            logger.warning("Detection model not available - using mock detections")
    
    async def detect_body_parts(self, image: Image.Image) -> Dict[str, Dict[str, int]]:
        """Detect body parts with fallback to mock data."""
        
        if self.use_real_model:
            try:
                return await self.real_service.detect_body_parts(image)
            except Exception as e:
                logger.error(f"Real model failed, falling back to mock: {e}")
        
        # Mock detection based on image size
        width, height = image.size
        
        # Generate reasonable mock bounding boxes
        neck_box = {
            'x1': int(width * 0.4),
            'y1': int(height * 0.15),
            'x2': int(width * 0.6),
            'y2': int(height * 0.25)
        }
        
        stomach_box = {
            'x1': int(width * 0.35),
            'y1': int(height * 0.5),
            'x2': int(width * 0.65),
            'y2': int(height * 0.7)
        }
        
        logger.info("Using mock body part detections")
        return {
            'neck': neck_box,
            'stomach': stomach_box
        }
    
    def visualize_detections(self, image: Image.Image, bbox: Dict[str, Dict[str, int]]) -> np.ndarray:
        """Visualize detections (same as real service)."""
        import cv2
        
        # Convert PIL to numpy
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        if 'neck' in bbox:
            neck = bbox['neck']
            cv2.rectangle(img_array, (neck['x1'], neck['y1']), (neck['x2'], neck['y2']), 
                         color=(0, 0, 255), thickness=2)
            cv2.putText(img_array, 'Neck (Mock)', (neck['x1'], neck['y1']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if 'stomach' in bbox:
            stomach = bbox['stomach']
            cv2.rectangle(img_array, (stomach['x1'], stomach['y1']), (stomach['x2'], stomach['y2']), 
                         color=(0, 255, 0), thickness=2)
            cv2.putText(img_array, 'Stomach (Mock)', (stomach['x1'], stomach['y1']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_array
