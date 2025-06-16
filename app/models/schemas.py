"""Pydantic models for 3-photo analysis request/response validation."""

from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime

class GenderEnum(str, Enum):
    male = "male"
    female = "female"

class AnalysisRequest(BaseModel):
    height: float = Field(..., ge=100, le=250, description="Height in centimeters")
    weight: Optional[float] = Field(None, ge=30, le=300, description="Weight in kilograms")
    age: Optional[int] = Field(None, ge=1, le=120, description="Age in years")
    sex: GenderEnum = Field(GenderEnum.male, description="Gender")

class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left x coordinate")
    y1: int = Field(..., description="Top y coordinate") 
    x2: int = Field(..., description="Right x coordinate")
    y2: int = Field(..., description="Bottom y coordinate")

class MultiViewDetectionResult(BaseModel):
    """Detection results from multiple photo angles."""
    front: Dict[str, BoundingBox] = Field(default_factory=dict, description="Front view detections")
    side: Dict[str, BoundingBox] = Field(default_factory=dict, description="Side view detections")
    back: Dict[str, BoundingBox] = Field(default_factory=dict, description="Back view detections")
    confidence_score: Optional[float] = Field(None, description="Overall detection confidence")

class DetectionResult(BaseModel):
    """Legacy single-photo detection result."""
    neck: BoundingBox
    stomach: BoundingBox
    confidence_score: Optional[float] = Field(None, description="Detection confidence")

class EnhancedMeasurementResult(BaseModel):
    """Comprehensive measurements from 3-photo analysis."""
    # Core measurements (required)
    neck_circumference: float = Field(..., description="Neck circumference in meters")
    waist_circumference: float = Field(..., description="Waist circumference in meters")
    
    # Enhanced measurements (optional)
    chest_circumference: Optional[float] = Field(None, description="Chest circumference in meters")
    hip_circumference: Optional[float] = Field(None, description="Hip circumference in meters")
    shoulder_width: Optional[float] = Field(None, description="Shoulder width in meters")
    
    # Metadata
    measurements_unit: str = Field("meters", description="Unit of measurements")
    measurement_confidence: Optional[float] = Field(None, description="Measurement quality score")

class MeasurementResult(BaseModel):
    """Legacy single-photo measurement result."""
    neck_circumference: float = Field(..., description="Neck circumference in meters")
    waist_circumference: float = Field(..., description="Waist circumference in meters")
    measurements_unit: str = Field("meters", description="Unit of measurements")

class ComprehensiveBodyComposition(BaseModel):
    """Complete body composition analysis with 9 health metrics."""
    # Core metrics
    body_fat_percentage: Optional[float] = Field(None, description="Body fat percentage")
    body_fat_category: Optional[str] = Field(None, description="Health category")
    
    # Enhanced metrics
    lean_muscle_mass_kg: Optional[float] = Field(None, description="Lean muscle mass in kg")
    bmi: Optional[float] = Field(None, description="Body Mass Index")
    bmi_category: Optional[str] = Field(None, description="BMI category")
    
    # Ratios and proportions
    waist_to_hip_ratio: Optional[float] = Field(None, description="Waist-to-hip ratio")
    chest_to_waist_ratio: Optional[float] = Field(None, description="Chest-to-waist ratio")
    
    # Body measurements (cm)
    neck_cm: float = Field(..., description="Neck circumference in cm")
    waist_cm: float = Field(..., description="Waist circumference in cm")
    chest_cm: Optional[float] = Field(None, description="Chest circumference in cm")
    hip_cm: Optional[float] = Field(None, description="Hip circumference in cm")
    shoulder_width_cm: Optional[float] = Field(None, description="Shoulder width in cm")
    
    # Additional metrics
    body_surface_area_m2: Optional[float] = Field(None, description="Body surface area in m²")

class BodyCompositionResult(BaseModel):
    """Legacy single-photo body composition result."""
    body_fat_percentage: Optional[float] = Field(None, description="Body fat percentage")
    body_fat_category: Optional[str] = Field(None, description="Health category")
    neck_cm: float = Field(..., description="Neck circumference in cm")
    waist_cm: float = Field(..., description="Waist circumference in cm")

class HealthAssessment(BaseModel):
    """Health risk assessment based on measurements."""
    cardiovascular_risk: Optional[str] = Field(None, description="Cardiovascular risk level")
    sleep_apnea_risk: Optional[str] = Field(None, description="Sleep apnea risk level")
    overall_health_score: Optional[float] = Field(None, description="Overall health score 0-100")
    recommendations: List[str] = Field(default_factory=list, description="Health recommendations")

class QualityMetrics(BaseModel):
    """Analysis quality and confidence metrics."""
    overall_confidence: float = Field(..., description="Overall analysis confidence")
    detection_quality: float = Field(..., description="Pose detection quality")
    measurement_consistency: float = Field(..., description="Cross-angle measurement consistency")
    photo_quality_scores: Dict[str, float] = Field(default_factory=dict, description="Individual photo quality")

class ComprehensiveAnalysisResponse(BaseModel):
    """Complete 3-photo analysis response with all metrics."""
    success: bool = Field(True, description="Analysis success status")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    analysis_mode: str = Field("comprehensive_3_photo", description="Analysis mode")
    photos_processed: int = Field(3, description="Number of photos processed")
    
    # Core results
    measurements: EnhancedMeasurementResult
    body_composition: ComprehensiveBodyComposition
    health_assessment: HealthAssessment
    
    # Detection data
    detections: MultiViewDetectionResult
    quality_metrics: QualityMetrics
    
    # Metadata
    analysis_timestamp: str = Field(..., description="ISO timestamp of analysis")
    processing_time_seconds: float = Field(..., description="Processing time")
    api_version: str = Field("2.0.0", description="API version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AnalysisResponse(BaseModel):
    """Legacy single-photo analysis response."""
    success: bool = Field(True, description="Analysis success status")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    measurements: MeasurementResult
    body_composition: BodyCompositionResult
    detections: DetectionResult
    analysis_timestamp: str = Field(..., description="ISO timestamp of analysis")
    processing_time_seconds: float = Field(..., description="Processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for client handling")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    analysis_id: Optional[str] = Field(None, description="Analysis ID if applicable")
    photos_processed: int = Field(0, description="Number of photos successfully processed")

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    service: str = Field("BodyVision API", description="Service name")
    version: str = Field("2.0.0", description="API version")

class DetailedHealthResponse(HealthResponse):
    system: Dict[str, Any] = Field(default_factory=dict, description="System information")
    models_loaded: bool = Field(False, description="Whether ML models are loaded")
    mediapipe_available: bool = Field(False, description="MediaPipe availability")

class HealthCategoriesResponse(BaseModel):
    male: Dict[str, str]
    female: Dict[str, str]

# Photo upload validation models
class PhotoUpload(BaseModel):
    """Individual photo upload validation."""
    view_type: str = Field(..., description="Photo view type: front, side, back")
    quality_score: Optional[float] = Field(None, description="Photo quality score")
    size_bytes: int = Field(..., description="File size in bytes")
    dimensions: tuple = Field(..., description="Image dimensions (width, height)")
    format: str = Field(..., description="Image format")

class ThreePhotoUpload(BaseModel):
    """Three-photo upload validation."""
    front_photo: PhotoUpload
    side_photo: PhotoUpload  
    back_photo: PhotoUpload
    total_size_bytes: int = Field(..., description="Total upload size")
    upload_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Analysis configuration models
class AnalysisConfig(BaseModel):
    """Analysis configuration options."""
    enable_enhanced_metrics: bool = Field(True, description="Calculate all 9 metrics")
    enable_quality_assessment: bool = Field(True, description="Assess photo/analysis quality")
    enable_health_recommendations: bool = Field(True, description="Generate health insights")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")
    measurement_units: str = Field("metric", description="Measurement units: metric or imperial")
