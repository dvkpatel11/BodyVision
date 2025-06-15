"""Core body analyzer that orchestrates the entire analysis pipeline."""

from typing import Dict, Any, Optional
from PIL import Image

from app.services.analysis_service import AnalysisService
from app.services import create_analysis_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BodyAnalyzer:
    """Main class for body analysis operations."""
    
    def __init__(self, analysis_service: Optional[AnalysisService] = None):
        self.analysis_service = analysis_service or create_analysis_service()
        logger.info("BodyAnalyzer initialized")
    
    async def analyze(
        self, 
        image: Image.Image,
        height: float = 182,
        weight: Optional[float] = None,
        age: Optional[int] = None,
        sex: str = 'male'
    ) -> Dict[str, Any]:
        """
        Perform complete body analysis.
        
        Args:
            image: Input image
            height: Height in cm
            weight: Weight in kg (optional)
            age: Age in years (optional)
            sex: 'male' or 'female'
            
        Returns:
            Complete analysis results
        """
        user_metadata = {
            'height': height,
            'weight': weight,
            'age': age,
            'sex': sex
        }
        
        return await self.analysis_service.analyze_body(image, user_metadata)
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results for display."""
        
        if 'error' in results:
            return f"Analysis Error: {results['error']}"
        
        output_lines = []
        
        if 'neck_cm' in results and 'waist_cm' in results:
            output_lines.append(f"Neck Circumference: {results['neck_cm']:.2f} cm")
            output_lines.append(f"Waist Circumference: {results['waist_cm']:.2f} cm")
        
        if 'body_fat_percentage' in results and results['body_fat_percentage'] is not None:
            output_lines.append(f"Body Fat Percentage: {results['body_fat_percentage']:.2f}%")
            
            if 'body_fat_category' in results:
                output_lines.append(f"Category: {results['body_fat_category']}")
        
        return '\n'.join(output_lines) if output_lines else "No measurements available"
