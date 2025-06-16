"""Image processing utilities for 3-photo analysis."""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ThreePhotoImageProcessor:
    """Processes and validates images for 3-photo analysis."""
    
    def __init__(self):
        self.max_dimension = 2048
        self.min_dimension = 400
        self.target_quality = 90
        
    def validate_photo_set(
        self, 
        photos: Dict[str, Image.Image]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a set of 3 photos for analysis.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        required_views = ['front', 'side', 'back']
        
        # Check all required photos are present
        for view in required_views:
            if view not in photos:
                errors.append(f"Missing {view} view photo")
                continue
                
            photo = photos[view]
            
            # Validate individual photo
            is_valid, photo_errors = self.validate_single_photo(photo, view)
            if not is_valid:
                errors.extend([f"{view}: {error}" for error in photo_errors])
        
        # Check photo consistency
        if len(photos) == 3:
            consistency_errors = self._check_photo_consistency(photos)
            errors.extend(consistency_errors)
        
        return len(errors) == 0, errors
    
    def validate_single_photo(
        self, 
        photo: Image.Image, 
        view_type: str
    ) -> Tuple[bool, List[str]]:
        """Validate a single photo for analysis."""
        errors = []
        
        try:
            # Check dimensions
            width, height = photo.size
            
            if min(width, height) < self.min_dimension:
                errors.append(f"Image too small: {width}x{height} (minimum {self.min_dimension})")
            
            if max(width, height) > self.max_dimension:
                errors.append(f"Image too large: {width}x{height} (maximum {self.max_dimension})")
            
            # Check aspect ratio (should be roughly portrait)
            aspect_ratio = width / height
            if aspect_ratio > 1.5:  # Too wide
                errors.append(f"Image too wide (aspect ratio {aspect_ratio:.2f})")
            elif aspect_ratio < 0.4:  # Too narrow
                errors.append(f"Image too narrow (aspect ratio {aspect_ratio:.2f})")
            
            # Check format
            if photo.mode not in ['RGB', 'RGBA']:
                errors.append(f"Unsupported image mode: {photo.mode}")
            
            # Check if image is mostly black/white (poor quality)
            if self._is_poor_quality(photo):
                errors.append("Poor image quality detected")
                
        except Exception as e:
            errors.append(f"Error validating image: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _check_photo_consistency(self, photos: Dict[str, Image.Image]) -> List[str]:
        """Check consistency across the 3 photos."""
        errors = []
        
        sizes = {view: photo.size for view, photo in photos.items()}
        
        # Check if sizes are reasonably similar (within 20%)
        widths = [size[0] for size in sizes.values()]
        heights = [size[1] for size in sizes.values()]
        
        width_ratio = max(widths) / min(widths) if min(widths) > 0 else float('inf')
        height_ratio = max(heights) / min(heights) if min(heights) > 0 else float('inf')
        
        if width_ratio > 1.5:
            errors.append("Photo widths vary too much between views")
        
        if height_ratio > 1.5:
            errors.append("Photo heights vary too much between views")
        
        return errors
    
    def _is_poor_quality(self, photo: Image.Image) -> bool:
        """Check if image appears to be poor quality."""
        try:
            # Convert to grayscale for analysis
            gray = photo.convert('L')
            pixels = np.array(gray)
            
            # Check if image is mostly very dark or very bright
            mean_brightness = np.mean(pixels)
            if mean_brightness < 30 or mean_brightness > 225:
                return True
            
            # Check contrast (standard deviation)
            contrast = np.std(pixels)
            if contrast < 20:  # Very low contrast
                return True
                
            return False
            
        except Exception:
            return False  # If we can't analyze, assume it's okay
    
    def preprocess_photos(
        self, 
        photos: Dict[str, Image.Image]
    ) -> Dict[str, Image.Image]:
        """Preprocess photos for analysis."""
        processed = {}
        
        for view, photo in photos.items():
            try:
                processed_photo = self._preprocess_single_photo(photo, view)
                processed[view] = processed_photo
                logger.debug(f"Preprocessed {view} photo: {processed_photo.size}")
            except Exception as e:
                logger.error(f"Error preprocessing {view} photo: {e}")
                # Use original photo if preprocessing fails
                processed[view] = photo
        
        return processed
    
    def _preprocess_single_photo(self, photo: Image.Image, view_type: str) -> Image.Image:
        """Preprocess a single photo."""
        # Convert to RGB if needed
        if photo.mode != 'RGB':
            photo = photo.convert('RGB')
        
        # Resize if too large
        width, height = photo.size
        if max(width, height) > self.max_dimension:
            ratio = self.max_dimension / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            photo = photo.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return photo
    
    def get_photo_quality_scores(
        self, 
        photos: Dict[str, Image.Image]
    ) -> Dict[str, float]:
        """Calculate quality scores for each photo."""
        scores = {}
        
        for view, photo in photos.items():
            try:
                score = self._calculate_quality_score(photo)
                scores[view] = score
            except Exception as e:
                logger.error(f"Error calculating quality for {view}: {e}")
                scores[view] = 0.5  # Default moderate quality
        
        return scores
    
    def _calculate_quality_score(self, photo: Image.Image) -> float:
        """Calculate a quality score (0-1) for a photo."""
        try:
            gray = photo.convert('L')
            pixels = np.array(gray)
            
            # Brightness score (prefer moderate brightness)
            brightness = np.mean(pixels)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Contrast score
            contrast = np.std(pixels)
            contrast_score = min(contrast / 50, 1.0)  # Normalize to 0-1
            
            # Size score (prefer larger images)
            width, height = photo.size
            size_score = min((width * height) / (800 * 600), 1.0)
            
            # Combined score
            quality_score = (brightness_score * 0.3 + 
                           contrast_score * 0.5 + 
                           size_score * 0.2)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default moderate quality if analysis fails
