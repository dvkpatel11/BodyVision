"""Service for body measurement calculations."""

import numpy as np
from typing import Dict, Tuple, Any

from app.utils.math_utils import point_cloud, calculate_distance
from app.utils.image_utils import calculate_gradient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MeasurementService:
    """Service for calculating body measurements from depth and detection data."""
    
    def __init__(self, max_distance_threshold: float = 0.01):
        self.max_distance_threshold = max_distance_threshold
    
    def get_body_part_boundaries(self, grad_x: np.ndarray, bbox: Dict[str, int]) -> Tuple[int, int, int]:
        """
        Find the boundaries of a body part using gradient analysis.
        
        Args:
            grad_x: Gradient image
            bbox: Bounding box with keys 'x1', 'y1', 'x2', 'y2'
            
        Returns:
            Tuple of (row, left_boundary, right_boundary)
        """
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        row = cy
        col_start = x1
        col_end = x2

        # Find initial gradient extrema
        min_grad = 0
        max_grad = 0
        centre_left_x = col_start
        centre_right_x = col_end
        
        for i in range(col_start, col_end):
            if grad_x[row, i] < min_grad:
                min_grad = grad_x[row, i]
                centre_left_x = i
            if grad_x[row, i] >= max_grad:
                max_grad = grad_x[row, i]
                centre_right_x = i

        # Refine boundaries
        for i in range(centre_left_x + 1, cx):
            if grad_x[row, i] > grad_x[row, centre_left_x] and grad_x[row, i] < 0:
                centre_left_x = i
        centre_left_x += 1

        for i in range(centre_right_x - 1, cx, -1):
            if grad_x[row, i] < grad_x[row, centre_right_x] and grad_x[row, i] > 0:
                centre_right_x = i
        centre_right_x -= 1

        return row, centre_left_x, centre_right_x
    
    def calculate_circumference(self, xyz: np.ndarray, row: int, left: int, right: int) -> float:
        """
        Calculate circumference from 3D point cloud data.
        
        Args:
            xyz: 3D point cloud
            row: Row index
            left: Left boundary
            right: Right boundary
            
        Returns:
            Half circumference (multiply by 2 for full circumference)
        """
        if right <= left + 1:
            logger.warning(f"Invalid boundaries: left={left}, right={right}")
            return 0.0
        
        p1 = xyz[row, left, :]
        p2 = xyz[row, left + 1, :]
        total_distance = 0.0
        
        for i in range(left + 2, right):
            distance = calculate_distance(p1, p2)
            if distance <= self.max_distance_threshold:
                total_distance += distance
            p1 = p2
            p2 = xyz[row, i, :]

        return total_distance
    
    async def calculate_body_measurements(
        self, 
        depth: np.ndarray, 
        detections: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        Calculate body measurements from depth map and detections.
        
        Args:
            depth: Depth map
            detections: Dictionary of detected body parts with bounding boxes
            
        Returns:
            Dictionary containing body measurements
        """
        try:
            logger.info("Starting body measurement calculation")
            
            # Convert depth to 3D point cloud
            xyz = point_cloud(depth)
            
            # Calculate gradient for boundary detection
            grad_x = calculate_gradient(depth, filter_type='scharr', smoothen=False)
            
            measurements = {}
            
            # Calculate neck circumference
            if 'neck' in detections:
                row, left, right = self.get_body_part_boundaries(grad_x, detections['neck'])
                neck_half = self.calculate_circumference(xyz, row, left, right)
                measurements['neck_circumference'] = neck_half * 2
            
            # Calculate waist circumference  
            if 'stomach' in detections:
                row, left, right = self.get_body_part_boundaries(grad_x, detections['stomach'])
                waist_half = self.calculate_circumference(xyz, row, left, right)
                measurements['waist_circumference'] = waist_half * 2
            
            logger.info("Body measurement calculation completed")
            return measurements
            
        except Exception as e:
            logger.error(f"Body measurement calculation failed: {str(e)}")
            raise
