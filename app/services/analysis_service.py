"""Main analysis service orchestrating the complete body analysis pipeline."""

import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

from app.services.depth_service import DepthService
from app.services.detection_service import DetectionService  
from app.services.measurement_service import MeasurementService
from app.utils.math_utils import navy_body_fat_formula
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisService:
    """Main service for complete body analysis."""
    
    def __init__(
        self,
        depth_service: DepthService,
        detection_service: DetectionService,
        measurement_service: MeasurementService
    ):
        self.depth_service = depth_service
        self.detection_service = detection_service
        self.measurement_service = measurement_service
    
    async def analyze_body(
        self, 
        image: Image.Image,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete body analysis pipeline.
        
        Args:
            image: Input PIL image
            user_metadata: User information (height, weight, age, sex)
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting complete body analysis")
            
            # Step 1: Get depth estimation
            depth_map = await self.depth_service.estimate_depth(image)
            
            # Step 2: Detect body parts
            detections = await self.detection_service.detect_body_parts(image)
            
            # Step 3: Calculate measurements
            measurements = await self.measurement_service.calculate_body_measurements(
                depth_map, detections
            )
            
            # Step 4: Calculate body composition metrics
            results = await self._calculate_body_composition(measurements, user_metadata)
            
            # Add metadata
            results.update({
                'measurements': measurements,
                'detections': detections,
                'analysis_timestamp': np.datetime64('now').astype(str)
            })
            
            logger.info("Complete body analysis finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Body analysis failed: {str(e)}")
            raise
    
    async def _calculate_body_composition(
        self, 
        measurements: Dict[str, float],
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate body composition metrics."""
        
        results = {}
        
        # Extract user metadata with defaults
        if user_metadata:
            height_cm = user_metadata.get('height', 182)  # Default 182cm
            sex = user_metadata.get('sex', 'male').lower()
        else:
            height_cm = 182
            sex = 'male'
        
        # Calculate body fat percentage using Navy formula
        if 'neck_circumference' in measurements and 'waist_circumference' in measurements:
            neck_cm = measurements['neck_circumference'] * 100  # Convert to cm
            waist_cm = measurements['waist_circumference'] * 100  # Convert to cm
            
            try:
                body_fat_percentage = navy_body_fat_formula(neck_cm, waist_cm, height_cm, sex)
                results['body_fat_percentage'] = round(body_fat_percentage, 2)
                results['neck_cm'] = round(neck_cm, 2)
                results['waist_cm'] = round(waist_cm, 2)
                
                # Add health categorization
                results['body_fat_category'] = self._categorize_body_fat(body_fat_percentage, sex)
                
            except Exception as e:
                logger.error(f"Body fat calculation failed: {str(e)}")
                results['body_fat_percentage'] = None
                results['error'] = str(e)
        
        return results
    
    def _categorize_body_fat(self, body_fat_percentage: float, sex: str) -> str:
        """Categorize body fat percentage into health ranges."""
        
        if sex == 'male':
            if body_fat_percentage < 6:
                return 'Essential Fat'
            elif body_fat_percentage < 14:
                return 'Athletes'
            elif body_fat_percentage < 18:
                return 'Fitness'
            elif body_fat_percentage < 25:
                return 'Average'
            else:
                return 'Obese'
        else:  # female
            if body_fat_percentage < 16:
                return 'Essential Fat'
            elif body_fat_percentage < 21:
                return 'Athletes'
            elif body_fat_percentage < 25:
                return 'Fitness'
            elif body_fat_percentage < 32:
                return 'Average'
            else:
                return 'Obese'
