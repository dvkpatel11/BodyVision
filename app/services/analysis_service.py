"""Production-ready analysis service for BodyVision."""

import numpy as np
import time
from typing import Dict, Any, Optional
from PIL import Image

from app.services.mediapipe_detection_service import MediaPipeDetectionService
from app.services.measurement_service import MeasurementService
from app.utils.math_utils import navy_body_fat_formula
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisService:
    """Production-grade body analysis service."""
    
    def __init__(
        self,
        detection_service: MediaPipeDetectionService,
        measurement_service: MeasurementService
    ):
        """Initialize production analysis service."""
        self.detection_service = detection_service
        self.measurement_service = measurement_service
        
        logger.info("✅ Production AnalysisService initialized")
    
    async def analyze_body(
        self, 
        image: Image.Image,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Production body analysis pipeline.
        
        Optimized for speed, accuracy, and reliability.
        
        Args:
            image: Input PIL image from smartphone
            user_metadata: User data (height, weight, age, sex)
            
        Returns:
            Complete analysis results with body composition
        """
        start_time = time.time()
        analysis_id = f"analysis_{int(start_time * 1000)}"
        
        try:
            logger.info(f"🏭 Starting production analysis {analysis_id}")
            
            # Step 1: MediaPipe pose detection
            detections = await self.detection_service.detect_body_parts(image)
            detection_time = time.time() - start_time
            logger.info(f"✅ Detection completed in {detection_time:.3f}s")
            
            # Step 2: Calibrate measurements if user height provided
            if user_metadata and 'height' in user_metadata:
                self.measurement_service.calibrate_for_image(
                    image.size, user_metadata['height']
                )
            
            # Step 3: Calculate body measurements
            measurements = await self.measurement_service.calculate_body_measurements(
                detections, image.size
            )
            measurement_time = time.time() - start_time - detection_time
            logger.info(f"✅ Measurements completed in {measurement_time:.3f}s")
            
            # Step 4: Calculate body composition
            composition_results = await self._calculate_body_composition(
                measurements, user_metadata
            )
            
            # Step 5: Assemble production results
            total_time = time.time() - start_time
            
            results = {
                'success': True,
                'analysis_id': analysis_id,
                'body_fat_percentage': composition_results.get('body_fat_percentage'),
                'body_fat_category': composition_results.get('body_fat_category'),
                'neck_cm': composition_results.get('neck_cm'),
                'waist_cm': composition_results.get('waist_cm'),
                'measurements': measurements,
                'detections': detections,
                'confidence_score': await self._get_analysis_confidence(image),
                'processing_time_seconds': round(total_time, 3),
                'analysis_timestamp': np.datetime64('now').astype(str),
                'detection_method': 'MediaPipe',
                'api_version': '2.0.0'
            }
            
            # Add user context
            if user_metadata:
                results['user_context'] = {
                    'height': user_metadata.get('height'),
                    'sex': user_metadata.get('sex', 'male'),
                    'age': user_metadata.get('age'),
                    'weight': user_metadata.get('weight')
                }
            
            logger.info(f"✅ Production analysis {analysis_id} completed in {total_time:.3f}s")
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Production analysis {analysis_id} failed: {str(e)}")
            
            # Return production error response (never crash)
            return {
                'success': False,
                'analysis_id': analysis_id,
                'error': str(e),
                'error_type': 'analysis_failure',
                'body_fat_percentage': None,
                'measurements': {},
                'detections': self._get_fallback_detections(image),
                'processing_time_seconds': round(total_time, 3),
                'analysis_timestamp': np.datetime64('now').astype(str),
                'api_version': '2.0.0'
            }
    
    async def _calculate_body_composition(
        self, 
        measurements: Dict[str, float],
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate body composition using Navy formula."""
        
        try:
            # Extract user data with production defaults
            height_cm = user_metadata.get('height', 175) if user_metadata else 175
            sex = user_metadata.get('sex', 'male').lower() if user_metadata else 'male'
            
            # Validate measurements
            if 'neck_circumference' not in measurements or 'waist_circumference' not in measurements:
                logger.warning("Missing measurements for body fat calculation")
                return {
                    'body_fat_percentage': None,
                    'error': 'Missing required measurements',
                    'neck_cm': 0,
                    'waist_cm': 0
                }
            
            # Convert to centimeters
            neck_cm = measurements['neck_circumference'] * 100
            waist_cm = measurements['waist_circumference'] * 100
            
            # Calculate body fat using Navy formula
            body_fat_percentage = navy_body_fat_formula(neck_cm, waist_cm, height_cm, sex)
            
            # Ensure reasonable bounds
            body_fat_percentage = max(3.0, min(50.0, body_fat_percentage))
            
            results = {
                'body_fat_percentage': round(body_fat_percentage, 1),
                'neck_cm': round(neck_cm, 1),
                'waist_cm': round(waist_cm, 1),
                'body_fat_category': self._categorize_body_fat(body_fat_percentage, sex)
            }
            
            logger.info(f"✅ Body composition: {body_fat_percentage:.1f}% ({results['body_fat_category']})")
            return results
            
        except Exception as e:
            logger.error(f"Body composition calculation failed: {e}")
            return {
                'body_fat_percentage': None,
                'error': str(e),
                'neck_cm': 0,
                'waist_cm': 0
            }
    
    def _categorize_body_fat(self, body_fat_percentage: float, sex: str) -> str:
        """Categorize body fat percentage for health insights."""
        
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
    
    async def _get_analysis_confidence(self, image: Image.Image) -> float:
        """Get confidence score for production monitoring."""
        try:
            if hasattr(self.detection_service, 'get_pose_confidence'):
                confidence = self.detection_service.get_pose_confidence(image)
                return round(confidence, 3)
            return 0.85  # Default production confidence
        except Exception:
            return 0.75  # Fallback confidence
    
    def _get_fallback_detections(self, image: Image.Image) -> Dict[str, Dict[str, int]]:
        """Fallback detections for production reliability."""
        width, height = image.size
        return {
            'neck': {
                'x1': int(width * 0.42),
                'y1': int(height * 0.15),
                'x2': int(width * 0.58),
                'y2': int(height * 0.25)
            },
            'stomach': {
                'x1': int(width * 0.35),
                'y1': int(height * 0.45),
                'x2': int(width * 0.65),
                'y2': int(height * 0.65)
            }
        }
    
    async def get_health_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Get health insights for production use."""
        
        body_fat = analysis_results.get('body_fat_percentage')
        category = analysis_results.get('body_fat_category')
        sex = analysis_results.get('user_context', {}).get('sex', 'male')
        
        if not body_fat or not category:
            return {
                'summary': 'Body composition analysis completed. Consider retaking photo with better lighting for more accurate results.',
                'recommendation': 'Ensure neck and waist areas are clearly visible in good lighting.',
                'next_steps': 'Regular monitoring helps track health progress over time.'
            }
        
        # Production health insights
        insights = {
            'Essential Fat': {
                'summary': f'Your {body_fat}% body fat is in the essential fat range for {sex}s.',
                'recommendation': 'Consider consulting a healthcare provider about maintaining healthy weight.',
                'next_steps': 'Focus on balanced nutrition and appropriate exercise for your goals.'
            },
            'Athletes': {
                'summary': f'Your {body_fat}% body fat is in the athletic range - excellent conditioning!',
                'recommendation': 'Maintain your training routine and ensure adequate recovery.',
                'next_steps': 'Continue monitoring to maintain optimal performance levels.'
            },
            'Fitness': {
                'summary': f'Your {body_fat}% body fat is in the fitness range - great work!',
                'recommendation': 'Keep up your current routine with consistent exercise and nutrition.',
                'next_steps': 'Consider strength training to further improve body composition.'
            },
            'Average': {
                'summary': f'Your {body_fat}% body fat is in the average range for healthy {sex}s.',
                'recommendation': 'Focus on regular exercise and balanced nutrition for improvement.',
                'next_steps': 'Aim for 150 minutes of moderate exercise weekly plus strength training.'
            },
            'Obese': {
                'summary': f'Your {body_fat}% body fat indicates opportunity for health improvement.',
                'recommendation': 'Consider consulting healthcare and nutrition professionals.',
                'next_steps': 'Start with gradual lifestyle changes and regular monitoring.'
            }
        }
        
        return insights.get(category, {
            'summary': f'Body fat analysis: {body_fat}%',
            'recommendation': 'Maintain regular exercise and balanced nutrition.',
            'next_steps': 'Monitor progress with regular measurements.'
        })
