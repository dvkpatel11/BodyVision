"""Photo validation utilities for 3-photo uploads."""

from typing import Dict, List, Tuple, Optional
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class PhotoValidator:
    """Validates photo uploads for 3-photo analysis."""
    
    def __init__(self):
        self.max_file_size = 15 * 1024 * 1024  # 15MB total
        self.max_single_file_size = 8 * 1024 * 1024  # 8MB per photo
        self.min_dimensions = (400, 300)
        self.max_dimensions = (2048, 2048)
        self.allowed_formats = {'.jpg', '.jpeg', '.png', '.webp'}
        
    def validate_photo_uploads(
        self,
        front_data: bytes,
        side_data: bytes, 
        back_data: bytes
    ) -> Tuple[bool, List[str], Optional[Dict[str, Image.Image]]]:
        """
        Validate 3-photo upload data.
        
        Returns:
            (is_valid, error_messages, processed_images)
        """
        errors = []
        images = {}
        
        photo_data = {
            'front': front_data,
            'side': side_data,
            'back': back_data
        }
        
        # Check total size
        total_size = sum(len(data) for data in photo_data.values())
        if total_size > self.max_file_size:
            errors.append(f"Total upload size ({total_size // (1024*1024)}MB) exceeds limit ({self.max_file_size // (1024*1024)}MB)")
        
        # Validate each photo
        for view, data in photo_data.items():
            try:
                is_valid, photo_errors, image = self._validate_single_photo(data, view)
                
                if is_valid and image:
                    images[view] = image
                else:
                    errors.extend(photo_errors)
                    
            except Exception as e:
                errors.append(f"{view} photo: {str(e)}")
        
        # Additional validation if all photos loaded successfully
        if len(images) == 3:
            consistency_errors = self._validate_photo_consistency(images)
            errors.extend(consistency_errors)
        
        return len(errors) == 0, errors, images if len(errors) == 0 else None
    
    def _validate_single_photo(
        self, 
        data: bytes, 
        view: str
    ) -> Tuple[bool, List[str], Optional[Image.Image]]:
        """Validate a single photo upload."""
        errors = []
        
        try:
            # Check file size
            if len(data) > self.max_single_file_size:
                errors.append(f"File too large ({len(data) // (1024*1024)}MB, max {self.max_single_file_size // (1024*1024)}MB)")
            
            if len(data) < 1024:  # Less than 1KB
                errors.append("File too small (likely corrupted)")
            
            # Try to load as image
            try:
                image = Image.open(io.BytesIO(data))
                
                # Validate image properties
                img_errors = self._validate_image_properties(image, view)
                errors.extend(img_errors)
                
                if len(errors) == 0:
                    return True, [], image
                else:
                    return False, errors, None
                    
            except Exception as e:
                errors.append(f"Invalid image file: {str(e)}")
                return False, errors, None
                
        except Exception as e:
            errors.append(f"Error processing file: {str(e)}")
            return False, errors, None
    
    def _validate_image_properties(self, image: Image.Image, view: str) -> List[str]:
        """Validate image properties."""
        errors = []
        
        # Check dimensions
        width, height = image.size
        
        if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
            errors.append(f"Image too small: {width}x{height} (minimum {self.min_dimensions[0]}x{self.min_dimensions[1]})")
        
        if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
            errors.append(f"Image too large: {width}x{height} (maximum {self.max_dimensions[0]}x{self.max_dimensions[1]})")
        
        # Check format
        if image.format and image.format.lower() not in ['jpeg', 'jpg', 'png', 'webp']:
            errors.append(f"Unsupported format: {image.format}")
        
        # Check mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            errors.append(f"Unsupported color mode: {image.mode}")
        
        # Check aspect ratio
        if width > 0 and height > 0:
            aspect_ratio = width / height
            if aspect_ratio > 2.0:
                errors.append(f"Image too wide (aspect ratio {aspect_ratio:.2f})")
            elif aspect_ratio < 0.3:
                errors.append(f"Image too tall (aspect ratio {aspect_ratio:.2f})")
        
        return errors
    
    def _validate_photo_consistency(self, images: Dict[str, Image.Image]) -> List[str]:
        """Validate consistency across the 3 photos."""
        errors = []
        
        try:
            # Check if all images have reasonable similar dimensions
            sizes = {view: img.size for view, img in images.items()}
            
            widths = [size[0] for size in sizes.values()]
            heights = [size[1] for size in sizes.values()]
            
            # Calculate size variation
            width_variation = (max(widths) - min(widths)) / max(widths) if max(widths) > 0 else 0
            height_variation = (max(heights) - min(heights)) / max(heights) if max(heights) > 0 else 0
            
            if width_variation > 0.5:  # More than 50% variation
                errors.append("Photo widths vary significantly between views")
            
            if height_variation > 0.5:  # More than 50% variation  
                errors.append("Photo heights vary significantly between views")
            
            # Check formats are consistent
            formats = {view: img.format for view, img in images.items()}
            unique_formats = set(formats.values())
            
            if len(unique_formats) > 2:  # Allow some format variation
                errors.append("Too many different image formats used")
            
        except Exception as e:
            logger.error(f"Error validating photo consistency: {e}")
            # Don't add error - consistency check is optional
        
        return errors
    
    def get_validation_summary(
        self, 
        images: Dict[str, Image.Image]
    ) -> Dict[str, Any]:
        """Get a summary of photo validation results."""
        summary = {
            'total_photos': len(images),
            'photos': {},
            'total_pixels': 0,
            'average_size': (0, 0),
            'formats_used': set()
        }
        
        total_width = 0
        total_height = 0
        
        for view, image in images.items():
            width, height = image.size
            total_width += width
            total_height += height
            summary['total_pixels'] += width * height
            summary['formats_used'].add(image.format or 'unknown')
            
            summary['photos'][view] = {
                'size': (width, height),
                'format': image.format,
                'mode': image.mode,
                'pixels': width * height
            }
        
        if len(images) > 0:
            summary['average_size'] = (
                total_width // len(images),
                total_height // len(images)
            )
        
        summary['formats_used'] = list(summary['formats_used'])
        
        return summary

# Global validator instance
photo_validator = PhotoValidator()

def validate_three_photos(
    front_data: bytes,
    side_data: bytes,
    back_data: bytes
) -> Tuple[bool, List[str], Optional[Dict[str, Image.Image]]]:
    """Convenience function for 3-photo validation."""
    return photo_validator.validate_photo_uploads(front_data, side_data, back_data)
