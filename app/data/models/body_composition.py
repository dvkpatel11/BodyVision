"""Body composition data models and calculations."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class BodyFatCategory(Enum):
    ESSENTIAL_FAT = "Essential Fat"
    ATHLETES = "Athletes"
    FITNESS = "Fitness"
    AVERAGE = "Average"
    OBESE = "Obese"
    UNKNOWN = "Unknown"

class BMICategory(Enum):
    UNDERWEIGHT = "Underweight"
    NORMAL = "Normal"
    OVERWEIGHT = "Overweight"
    OBESE = "Obese"
    UNKNOWN = "Unknown"

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    ELEVATED = "Elevated"
    UNKNOWN = "Unknown"

@dataclass
class BodyComposition:
    """Complete body composition data structure."""
    # Core measurements
    height_cm: float
    weight_kg: Optional[float]
    sex: str
    age: Optional[int]
    
    # Circumference measurements
    neck_cm: float
    waist_cm: float
    chest_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    shoulder_width_cm: Optional[float] = None
    
    # Calculated metrics
    body_fat_percentage: Optional[float] = None
    body_fat_category: BodyFatCategory = BodyFatCategory.UNKNOWN
    lean_muscle_mass_kg: Optional[float] = None
    bmi: Optional[float] = None
    bmi_category: BMICategory = BMICategory.UNKNOWN
    
    # Ratios
    waist_to_hip_ratio: Optional[float] = None
    chest_to_waist_ratio: Optional[float] = None
    
    # Additional metrics
    body_surface_area_m2: Optional[float] = None
    
    # Risk assessments
    cardiovascular_risk: RiskLevel = RiskLevel.UNKNOWN
    sleep_apnea_risk: RiskLevel = RiskLevel.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'sex': self.sex,
            'age': self.age,
            'neck_cm': self.neck_cm,
            'waist_cm': self.waist_cm,
            'chest_cm': self.chest_cm,
            'hip_cm': self.hip_cm,
            'shoulder_width_cm': self.shoulder_width_cm,
            'body_fat_percentage': self.body_fat_percentage,
            'body_fat_category': self.body_fat_category.value,
            'lean_muscle_mass_kg': self.lean_muscle_mass_kg,
            'bmi': self.bmi,
            'bmi_category': self.bmi_category.value,
            'waist_to_hip_ratio': self.waist_to_hip_ratio,
            'chest_to_waist_ratio': self.chest_to_waist_ratio,
            'body_surface_area_m2': self.body_surface_area_m2,
            'cardiovascular_risk': self.cardiovascular_risk.value,
            'sleep_apnea_risk': self.sleep_apnea_risk.value
        }
    
    @classmethod
    def from_measurements(
        cls,
        height_cm: float,
        weight_kg: Optional[float],
        sex: str,
        neck_cm: float,
        waist_cm: float,
        age: Optional[int] = None,
        chest_cm: Optional[float] = None,
        hip_cm: Optional[float] = None,
        shoulder_width_cm: Optional[float] = None
    ):
        """Create BodyComposition from basic measurements."""
        return cls(
            height_cm=height_cm,
            weight_kg=weight_kg,
            sex=sex,
            age=age,
            neck_cm=neck_cm,
            waist_cm=waist_cm,
            chest_cm=chest_cm,
            hip_cm=hip_cm,
            shoulder_width_cm=shoulder_width_cm
        )

@dataclass 
class ThreePhotoMeasurements:
    """Measurements extracted from 3-photo analysis."""
    front_view_measurements: Dict[str, float]
    side_view_measurements: Dict[str, float]
    back_view_measurements: Dict[str, float]
    
    # Combined/averaged measurements
    neck_circumference: float
    waist_circumference: float
    chest_circumference: Optional[float] = None
    hip_circumference: Optional[float] = None
    shoulder_width: Optional[float] = None
    
    # Quality metrics
    measurement_consistency: float = 0.0
    confidence_score: float = 0.0
    
    def to_body_composition(
        self, 
        height_cm: float, 
        weight_kg: Optional[float], 
        sex: str, 
        age: Optional[int] = None
    ) -> BodyComposition:
        """Convert measurements to BodyComposition object."""
        return BodyComposition.from_measurements(
            height_cm=height_cm,
            weight_kg=weight_kg,
            sex=sex,
            age=age,
            neck_cm=self.neck_circumference * 100,  # Convert to cm
            waist_cm=self.waist_circumference * 100,
            chest_cm=self.chest_circumference * 100 if self.chest_circumference else None,
            hip_cm=self.hip_circumference * 100 if self.hip_circumference else None,
            shoulder_width_cm=self.shoulder_width * 100 if self.shoulder_width else None
        )
