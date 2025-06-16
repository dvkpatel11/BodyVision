"""Data models for BodyVision analysis."""

from .body_composition import (
    BodyComposition, 
    ThreePhotoMeasurements,
    BodyFatCategory,
    BMICategory,
    RiskLevel
)

__all__ = [
    'BodyComposition',
    'ThreePhotoMeasurements', 
    'BodyFatCategory',
    'BMICategory',
    'RiskLevel'
]
