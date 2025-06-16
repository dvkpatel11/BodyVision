"""Mathematical utilities for body measurement calculations."""

import numpy as np
from typing import Optional


def point_cloud(depth: np.ndarray) -> np.ndarray:
    """
    Convert depth map to 3D point cloud using camera intrinsics.
    
    Args:
        depth: Depth map as numpy array
        
    Returns:
        3D point cloud coordinates (x, y, z)
    """
    f_mm = 3.519
    width_mm = 4.61
    height_mm = 3.46
    tan_horFov = width_mm / (2 * f_mm)
    tan_verFov = height_mm / (2 * f_mm)

    width = depth.shape[1]
    height = depth.shape[0]

    cx, cy = width / 2, height / 2
    fx = width / (2 * tan_horFov)
    fy = height / (2 * tan_verFov)
    
    xx, yy = np.tile(range(width), height).reshape(height, width), \
             np.repeat(range(height), width).reshape(height, width)
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy

    xyz = np.dstack((xx * depth, yy * depth, depth))
    return xyz


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)
        
    Returns:
        Distance between points
    """
    return np.sqrt(np.sum((p2 - p1) ** 2))


def navy_body_fat_formula(
    neck_cm: float, 
    waist_cm: float, 
    height_cm: float, 
    sex: str,
    hip_cm: Optional[float] = None
) -> float:
    """
    Calculate body fat percentage using US Navy formula.
    
    Args:
        neck_cm: Neck circumference in centimeters
        waist_cm: Waist circumference in centimeters  
        height_cm: Height in centimeters
        sex: 'male' or 'female'
        hip_cm: Hip circumference in centimeters (required for females)
        
    Returns:
        Body fat percentage
        
    Raises:
        ValueError: If invalid sex or missing hip measurement for females
    """
    sex = sex.lower()
    
    if sex == 'male':
        # Male formula: BF% = 495 / (1.0324 - 0.19077 * log10(waist - neck) + 0.15456 * log10(height)) - 450
        body_fat = (495 / (1.0324 - 0.19077 * np.log10(waist_cm - neck_cm) + 
                          0.15456 * np.log10(height_cm))) - 450
                          
    elif sex == 'female':
        # Female formula requires hip measurement
        if hip_cm is None or hip_cm <= 0:
            raise ValueError("Hip circumference is required for female body fat calculation")
            
        # Female formula: BF% = 163.205 * log10(waist + hip - neck) - 97.684 * log10(height) - 78.387
        body_fat = (163.205 * np.log10(waist_cm + hip_cm - neck_cm) - 
                   97.684 * np.log10(height_cm) - 78.387)
    else:
        raise ValueError(f"Unsupported sex: {sex}. Must be 'male' or 'female'")
    
    # Ensure reasonable bounds (essential fat minimums)
    if sex == 'male':
        body_fat = max(2.0, body_fat)  # Male essential fat minimum ~2%
    else:
        body_fat = max(10.0, body_fat)  # Female essential fat minimum ~10%
        
    # Cap at reasonable maximum
    body_fat = min(50.0, body_fat)
    
    return body_fat


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate Body Mass Index."""
    if weight_kg is None or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def categorize_bmi(bmi: float) -> str:
    """Categorize BMI value."""
    if bmi == 0:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def calculate_lean_muscle_mass(weight_kg: float, body_fat_percentage: float) -> float:
    """Calculate lean muscle mass from weight and body fat percentage."""
    if weight_kg is None or body_fat_percentage is None or weight_kg <= 0:
        return 0.0
    
    fat_mass_kg = weight_kg * (body_fat_percentage / 100)
    lean_mass_kg = weight_kg - fat_mass_kg
    return max(0.0, lean_mass_kg)


def calculate_waist_to_hip_ratio(waist_cm: float, hip_cm: float) -> float:
    """Calculate waist-to-hip ratio."""
    if hip_cm == 0 or hip_cm is None:
        return 0.0
    return waist_cm / hip_cm


def calculate_chest_to_waist_ratio(chest_cm: float, waist_cm: float) -> float:
    """Calculate chest-to-waist ratio (V-taper indicator)."""
    if waist_cm == 0 or waist_cm is None:
        return 0.0
    return chest_cm / waist_cm


def calculate_body_surface_area(weight_kg: float, height_cm: float) -> float:
    """
    Calculate body surface area using DuBois formula.
    BSA = 0.20247 * (height_m^0.725) * (weight_kg^0.425)
    """
    if weight_kg is None or weight_kg <= 0:
        return 0.0
        
    height_m = height_cm / 100
    bsa = 0.20247 * (height_m ** 0.725) * (weight_kg ** 0.425)
    return bsa


def calculate_shoulder_symmetry_score(left_shoulder_width: float, right_shoulder_width: float) -> float:
    """
    Calculate shoulder symmetry score (0-100).
    100 = perfect symmetry, lower scores indicate asymmetry.
    """
    if left_shoulder_width <= 0 or right_shoulder_width <= 0:
        return 0.0
    
    avg_width = (left_shoulder_width + right_shoulder_width) / 2
    difference = abs(left_shoulder_width - right_shoulder_width)
    asymmetry_ratio = difference / avg_width
    
    # Convert to 0-100 score (100 = perfect symmetry)
    symmetry_score = max(0, 100 * (1 - asymmetry_ratio * 2))
    return min(100, symmetry_score)


def assess_cardiovascular_risk(waist_to_hip_ratio: float, sex: str) -> str:
    """
    Assess cardiovascular risk based on waist-to-hip ratio.
    
    Risk thresholds:
    - Male: >0.9 = high risk
    - Female: >0.85 = high risk
    """
    if waist_to_hip_ratio == 0:
        return "Unknown"
    
    sex = sex.lower()
    
    if sex == 'male':
        if waist_to_hip_ratio > 0.9:
            return "High"
        elif waist_to_hip_ratio > 0.85:
            return "Moderate"
        else:
            return "Low"
    else:  # female
        if waist_to_hip_ratio > 0.85:
            return "High"
        elif waist_to_hip_ratio > 0.80:
            return "Moderate"
        else:
            return "Low"


def assess_sleep_apnea_risk(neck_cm: float, sex: str) -> str:
    """
    Assess sleep apnea risk based on neck circumference.
    
    Risk thresholds:
    - Male: >43.2cm (17 inches) = increased risk
    - Female: >40.6cm (16 inches) = increased risk
    """
    if neck_cm == 0:
        return "Unknown"
    
    sex = sex.lower()
    
    if sex == 'male':
        if neck_cm > 43.2:
            return "Elevated"
        else:
            return "Low"
    else:  # female
        if neck_cm > 40.6:
            return "Elevated"
        else:
            return "Low"


def calculate_comprehensive_health_metrics(
    height_cm: float,
    weight_kg: Optional[float],
    neck_cm: float,
    waist_cm: float,
    chest_cm: float,
    hip_cm: Optional[float],
    shoulder_width_cm: float,
    sex: str
) -> dict:
    """
    Calculate all 9 comprehensive health metrics.
    
    Returns:
        Dictionary with all health metrics
    """
    
    metrics = {}
    
    # 1. Body Fat Percentage (Navy Formula)
    try:
        body_fat = navy_body_fat_formula(neck_cm, waist_cm, height_cm, sex, hip_cm)
        metrics['body_fat_percentage'] = round(body_fat, 1)
        metrics['body_fat_category'] = categorize_body_fat(body_fat, sex)
    except (ValueError, ZeroDivisionError) as e:
        metrics['body_fat_percentage'] = None
        metrics['body_fat_category'] = "Unknown"
    
    # 2. Lean Muscle Mass
    if weight_kg and metrics.get('body_fat_percentage'):
        lean_mass = calculate_lean_muscle_mass(weight_kg, metrics['body_fat_percentage'])
        metrics['lean_muscle_mass_kg'] = round(lean_mass, 1)
    else:
        metrics['lean_muscle_mass_kg'] = None
    
    # 3. BMI & Analysis
    if weight_kg:
        bmi = calculate_bmi(weight_kg, height_cm)
        metrics['bmi'] = round(bmi, 1)
        metrics['bmi_category'] = categorize_bmi(bmi)
    else:
        metrics['bmi'] = None
        metrics['bmi_category'] = "Unknown"
    
    # 4. Waist-to-Hip Ratio
    if hip_cm:
        whr = calculate_waist_to_hip_ratio(waist_cm, hip_cm)
        metrics['waist_to_hip_ratio'] = round(whr, 3)
        metrics['cardiovascular_risk'] = assess_cardiovascular_risk(whr, sex)
    else:
        metrics['waist_to_hip_ratio'] = None
        metrics['cardiovascular_risk'] = "Unknown"
    
    # 5. Neck Circumference & Sleep Apnea Risk
    metrics['neck_cm'] = round(neck_cm, 1)
    metrics['sleep_apnea_risk'] = assess_sleep_apnea_risk(neck_cm, sex)
    
    # 6. Shoulder Width & Symmetry (simplified - assumes symmetric for now)
    metrics['shoulder_width_cm'] = round(shoulder_width_cm, 1)
    metrics['shoulder_symmetry_score'] = 95.0  # Placeholder - would need left/right measurements
    
    # 7. Chest-to-Waist Ratio
    if chest_cm > 0:
        cwr = calculate_chest_to_waist_ratio(chest_cm, waist_cm)
        metrics['chest_to_waist_ratio'] = round(cwr, 3)
    else:
        metrics['chest_to_waist_ratio'] = None
    
    # 8. Body Surface Area
    if weight_kg:
        bsa = calculate_body_surface_area(weight_kg, height_cm)
        metrics['body_surface_area_m2'] = round(bsa, 3)
    else:
        metrics['body_surface_area_m2'] = None
    
    # 9. Additional measurements
    metrics['waist_cm'] = round(waist_cm, 1)
    metrics['chest_cm'] = round(chest_cm, 1) if chest_cm > 0 else None
    metrics['hip_cm'] = round(hip_cm, 1) if hip_cm else None
    
    return metrics


def categorize_body_fat(body_fat_percentage: float, sex: str) -> str:
    """Categorize body fat percentage for health insights."""
    
    if body_fat_percentage is None:
        return "Unknown"
    
    sex = sex.lower()
    
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
