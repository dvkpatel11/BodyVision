"""Image processing utilities for body analysis."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal


def calculate_gradient(
    image: np.ndarray, 
    filter_type: Literal['scharr', 'sobel'] = 'scharr',
    smoothen: bool = False
) -> np.ndarray:
    """
    Calculate gradient of image using specified filter.
    
    Args:
        image: Input image
        filter_type: Type of gradient filter ('scharr' or 'sobel')
        smoothen: Whether to apply Gaussian blur before gradient calculation
        
    Returns:
        Gradient image
    """
    ddepth = cv2.CV_16S
    
    if smoothen:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    if filter_type == 'scharr':
        grad_x = cv2.Scharr(image, ddepth, 1, 0)
    elif filter_type == 'sobel':
        grad_x = cv2.Sobel(image, ddepth, 1, 0)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return grad_x


def visualize_depth_profile(depth: np.ndarray, row: int = 250, col_range: tuple = (310, 370)):
    """
    Visualize depth profile at specified row and column range.
    
    Args:
        depth: Depth map
        row: Row to visualize
        col_range: (start_col, end_col) range to visualize
    """
    col1, col2 = col_range
    x = np.arange(col2 - col1)
    y = depth[row, col1:col2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid()
    plt.title(f'Depth Profile at Row {row}')
    plt.xlabel('Column')
    plt.ylabel('Depth')
    plt.show()


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for body analysis.
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Add any necessary preprocessing steps
    # For now, just ensure proper format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image.astype(np.float32)
