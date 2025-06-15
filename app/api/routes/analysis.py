"""Body analysis endpoints."""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import uuid
from datetime import datetime
from typing import Optional

from app.core.body_analyzer import BodyAnalyzer
from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, ErrorResponse, 
    MeasurementResult, BodyCompositionResult, DetectionResult, BoundingBox,
    HealthCategoriesResponse
)
from app.utils.logger import get_logger
from app.core.config import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# Dependency to get body analyzer
async def get_body_analyzer():
    """Dependency to provide body analyzer instance."""
    return BodyAnalyzer()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_body(
    image: UploadFile = File(..., description="Body image for analysis"),
    height: float = Form(..., description="Height in centimeters"),
    weight: Optional[float] = Form(None, description="Weight in kilograms"),
    age: Optional[int] = Form(None, description="Age in years"),
    sex: str = Form("male", description="Gender (male/female)"),
    analyzer: BodyAnalyzer = Depends(get_body_analyzer)
):
    """
    Analyze body composition from uploaded image.
    
    Returns detailed body measurements and composition analysis.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting analysis {analysis_id}")
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG)"
            )
        
        # Check file size
        image_data = await image.read()
        if len(image_data) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Convert to PIL Image
        try:
            pil_image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )
        
        # Validate request data
        try:
            request_data = AnalysisRequest(
                height=height,
                weight=weight,
                age=age,
                sex=sex
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input data: {str(e)}"
            )
        
        # Run analysis
        results = await analyzer.analyze(
            image=pil_image,
            height=request_data.height,
            weight=request_data.weight,
            age=request_data.age,
            sex=request_data.sex
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response = AnalysisResponse(
            analysis_id=analysis_id,
            measurements=MeasurementResult(
                neck_circumference=results['measurements']['neck_circumference'],
                waist_circumference=results['measurements']['waist_circumference']
            ),
            body_composition=BodyCompositionResult(
                body_fat_percentage=results.get('body_fat_percentage'),
                body_fat_category=results.get('body_fat_category'),
                neck_cm=results.get('neck_cm', 0),
                waist_cm=results.get('waist_cm', 0)
            ),
            detections=DetectionResult(
                neck=BoundingBox(**results['detections']['neck']),
                stomach=BoundingBox(**results['detections']['stomach'])
            ),
            analysis_timestamp=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 3),
            metadata={
                "input_height": height,
                "input_weight": weight,
                "input_age": age,
                "input_sex": sex,
                "image_size": pil_image.size,
                "image_mode": pil_image.mode
            }
        )
        
        logger.info(f"Analysis {analysis_id} completed in {processing_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/history/{analysis_id}")
async def get_analysis_history(analysis_id: str):
    """Get historical analysis results (placeholder for future database integration)."""
    # TODO: Implement database lookup
    raise HTTPException(
        status_code=501, 
        detail="History feature not yet implemented. Coming in future release."
    )

@router.get("/health-categories", response_model=HealthCategoriesResponse)
async def get_health_categories():
    """Get body fat percentage health categories by gender."""
    return HealthCategoriesResponse(
        male={
            "essential_fat": "< 6%",
            "athletes": "6-14%", 
            "fitness": "14-18%",
            "average": "18-25%",
            "obese": "> 25%"
        },
        female={
            "essential_fat": "< 16%",
            "athletes": "16-21%",
            "fitness": "21-25%", 
            "average": "25-32%",
            "obese": "> 32%"
        }
    )

@router.get("/limits")
async def get_upload_limits():
    """Get file upload limits and supported formats."""
    return {
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
        "allowed_extensions": list(settings.ALLOWED_EXTENSIONS),
        "supported_formats": ["JPEG", "PNG"],
        "recommended_resolution": "Minimum 400x300 pixels"
    }
