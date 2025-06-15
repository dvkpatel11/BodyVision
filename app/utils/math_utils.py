"""Mathematical utilities for body measurement calculations."""

import numpy as np
from typing import Tuple


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


def navy_body_fat_formula(neck_cm: float, waist_cm: float, height_cm: float, sex: str) -> float:
    """
    Calculate body fat percentage using US Navy formula.
    
    Args:
        neck_cm: Neck circumference in centimeters
        waist_cm: Waist circumference in centimeters  
        height_cm: Height in centimeters
        sex: 'male' or 'female'
        
    Returns:
        Body fat percentage
    """
    if sex.lower() == 'male':
        body_fat = (495 / (1.0324 - 0.19077 * np.log10(waist_cm - neck_cm) + 
                          0.15456 * np.log10(height_cm))) - 450
    elif sex.lower() == 'female':
        # TODO: Implement female formula
        # body_fat = (495 / (1.29579 - 0.35004 * log10(waist + hip - neck) + 0.22100 * log10(height))) - 450
        raise NotImplementedError("Female body fat calculation not yet implemented")
    else:
        raise ValueError(f"Unsupported sex: {sex}")
    
    return max(0.0, body_fat)  # Ensure non-negative result
