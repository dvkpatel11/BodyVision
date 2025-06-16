"""Body analysis endpoints - Updated for 3-photo support."""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from PIL import Image
import io
import time
import uuid
from datetime import datetime
from typing import Optional, Dict

from app.core.body_analyzer import BodyAnalyzer
from app.models.schemas import (
    AnalysisRequest, AnalysisResponse, MeasurementResult, BodyCompositionResult, 
    DetectionResult, BoundingBox, HealthCategoriesResponse
)
from app.utils.logger import get_logger
from app.core.config import get_settings

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

async def get_body_analyzer():
    """Dependency to provide body analyzer instance."""
    return BodyAnalyzer()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_body_three_photos(
    # 3-Photo Input (Required)
    front_image: UploadFile = File(..., description="Front view photo (0° - facing camera)"),
    side_image: UploadFile = File(..., description="Side view photo (90° - profile view)"),
    back_image: UploadFile = File(..., description="Back view photo (180° - rear view)"),
    
    # User Data
    height: float = Form(..., description="Height in centimeters"),
    weight: Optional[float] = Form(None, description="Weight in kilograms"),
    age: Optional[int] = Form(None, description="Age in years"),
    sex: str = Form("male", description="Gender (male/female)"),
    
    analyzer: BodyAnalyzer = Depends(get_body_analyzer)
):
    """
    Analyze body composition from 3 uploaded photos.
    
    **Required Photos:**
    - Front view: Face camera directly, arms slightly away from sides
    - Side view: Turn 90° right, arms relaxed at sides  
    - Back view: Turn around completely, arms slightly away from sides
    
    Returns comprehensive body measurements and composition analysis.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting 3-photo analysis {analysis_id}")
        
        # Validate all 3 images
        images = {
            'front': front_image,
            'side': side_image, 
            'back': back_image
        }
        
        processed_images = {}
        
        for view_name, image_file in images.items():
            # Validate image file
            if not image_file.content_type or not image_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail=f"{view_name.title()} image must be a valid image file (JPEG, PNG)"
                )
            
            # Check file size
            image_data = await image_file.read()
            max_size = getattr(settings, 'MAX_FILE_SIZE', 15 * 1024 * 1024)  # 15MB default
            if len(image_data) > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"{view_name.title()} image too large. Maximum size is {max_size // (1024*1024)}MB"
                )
            
            # Convert to PIL Image
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                processed_images[view_name] = pil_image
                logger.debug(f"✅ {view_name.title()} image processed: {pil_image.size}")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {view_name} image file: {str(e)}"
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
        
        # Run 3-photo analysis
        results = await analyzer.analyze_three_photos(
            front_image=processed_images['front'],
            side_image=processed_images['side'],
            back_image=processed_images['back'],
            height=request_data.height,
            weight=request_data.weight,
            age=request_data.age,
            sex=request_data.sex
        )
        
        processing_time = time.time() - start_time
        
        # Format response with enhanced metrics
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
                neck=BoundingBox(**results['detections']['front']['neck']),
                stomach=BoundingBox(**results['detections']['front']['stomach'])
            ),
            analysis_timestamp=datetime.now().isoformat(),
            processing_time_seconds=round(processing_time, 3),
            metadata={
                "input_height": height,
                "input_weight": weight,
                "input_age": age,
                "input_sex": sex,
                "front_image_size": processed_images['front'].size,
                "side_image_size": processed_images['side'].size,
                "back_image_size": processed_images['back'].size,
                "analysis_mode": "comprehensive_3_photo",
                "photos_processed": 3,
                **results.get('enhanced_metrics', {})
            }
        )
        
        logger.info(f"3-photo analysis {analysis_id} completed in {processing_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"3-photo analysis {analysis_id} failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"3-photo analysis failed: {str(e)}"
        )

# Legacy single-photo endpoint for backward compatibility
@router.post("/analyze-single")
async def analyze_body_single_photo(
    image: UploadFile = File(..., description="Body image for analysis"),
    height: float = Form(..., description="Height in centimeters"),
    weight: Optional[float] = Form(None, description="Weight in kilograms"),
    age: Optional[int] = Form(None, description="Age in years"),
    sex: str = Form("male", description="Gender (male/female)"),
    analyzer: BodyAnalyzer = Depends(get_body_analyzer)
):
    """
    Legacy single-photo analysis endpoint.
    
    Note: For best results, use the 3-photo /analyze endpoint.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Starting single-photo analysis {analysis_id}")
        
        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG)"
            )
        
        # Check file size
        image_data = await image.read()
        max_size = getattr(settings, 'MAX_FILE_SIZE', 15 * 1024 * 1024)
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
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
        
        # Run single-photo analysis
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
                "image_mode": pil_image.mode,
                "analysis_mode": "legacy_single_photo"
            }
        )
        
        logger.info(f"Single-photo analysis {analysis_id} completed in {processing_time:.3f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single-photo analysis {analysis_id} failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Single-photo analysis failed: {str(e)}"
        )

@router.get("/health-categories", response_model=HealthCategoriesResponse)
async def get_health_categories():
    """Get body fat percentage health categories by gender."""
    return HealthCategoriesResponse(
        male={
            "essential_fat": "2-5%",
            "athletes": "6-13%", 
            "fitness": "14-17%",
            "average": "18-24%",
            "obese": "25%+"
        },
        female={
            "essential_fat": "10-13%",
            "athletes": "14-20%",
            "fitness": "21-24%", 
            "average": "25-31%",
            "obese": "32%+"
        }
    )

@router.get("/limits")
async def get_upload_limits():
    """Get file upload limits and supported formats."""
    max_size = getattr(settings, 'MAX_FILE_SIZE', 15 * 1024 * 1024)
    allowed_ext = getattr(settings, 'ALLOWED_EXTENSIONS', {'.jpg', '.jpeg', '.png'})
    
    return {
        "max_file_size_mb": max_size // (1024 * 1024),
        "max_files_per_request": 3,
        "allowed_extensions": list(allowed_ext),
        "supported_formats": ["JPEG", "PNG", "WebP"],
        "recommended_resolution": "Minimum 720p, optimal 1080p+",
        "analysis_mode": "3_photo_comprehensive"
    }
