"""Development-friendly depth service with mock data."""

import numpy as np
from PIL import Image
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

class DepthServiceDev:
    """Mock depth service for development when model is unavailable."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        logger.warning("Using mock depth service - install model for real depth estimation")
    
    async def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Generate mock depth map based on image size."""
        
        width, height = image.size
        
        # Create a simple mock depth map
        # Simulate depth that decreases from center outward
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create depth map with person-like shape in center
        distance_from_center = np.sqrt(X**2 + Y**2)
        depth_map = 2.0 - distance_from_center  # Closer in center
        depth_map = np.clip(depth_map, 0.5, 2.0)  # Reasonable depth range
        
        logger.info("Generated mock depth map")
        return depth_map.astype(np.float32)
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Create visualization of mock depth map."""
        import cv2
        
        # Normalize depth map for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth_colored
