"""Pydantic models for request/response validation."""

from pydantic import BaseModel, validator, Field
from typing import Optional, Dict, Any, List
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

class DetectionResult(BaseModel):
    neck: BoundingBox
    stomach: BoundingBox
    confidence_score: Optional[float] = Field(None, description="Detection confidence")

class MeasurementResult(BaseModel):
    neck_circumference: float = Field(..., description="Neck circumference in meters")
    waist_circumference: float = Field(..., description="Waist circumference in meters")
    measurements_unit: str = Field("meters", description="Unit of measurements")

class BodyCompositionResult(BaseModel):
    body_fat_percentage: Optional[float] = Field(None, description="Body fat percentage")
    body_fat_category: Optional[str] = Field(None, description="Health category")
    neck_cm: float = Field(..., description="Neck circumference in cm")
    waist_cm: float = Field(..., description="Waist circumference in cm")

class AnalysisResponse(BaseModel):
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

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    service: str = Field("BodyVision API", description="Service name")
    version: str = Field("2.0.0", description="API version")

class DetailedHealthResponse(HealthResponse):
    system: Dict[str, Any] = Field(default_factory=dict, description="System information")
    models_loaded: bool = Field(False, description="Whether ML models are loaded")

class HealthCategoriesResponse(BaseModel):
    male: Dict[str, str]
    female: Dict[str, str]
