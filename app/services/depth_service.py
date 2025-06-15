"""Depth estimation service using Pix2Pix model."""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional
import tempfile

# Add legacy modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'legacy'))

from loaders import aligned_data_loader
from models import pix2pix_model

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DepthService:
    """Service for depth estimation using Pix2Pix model."""
    
    def __init__(self, model_path: Optional[str] = None, opt=None):
        self.model_path = model_path or 'app/models/weights/best_depth_Ours_Bilinear_inc_3_net_G.pth'
        self.opt = opt
        self.model = None
        self.batch_size = 1
        
        # Configure PyTorch for compatibility
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        
        logger.info("DepthService initialized")
    
    def _load_model(self):
        """Load the Pix2Pix depth estimation model."""
        if self.model is None:
            try:
                # Create minimal options if not provided
                if self.opt is None:
                    self.opt = self._create_minimal_options()
                
                self.model = pix2pix_model.Pix2PixModel(self.opt)
                self.model.switch_to_eval()
                
                logger.info("Pix2Pix depth model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load depth model: {str(e)}")
                raise
    
    def _create_minimal_options(self):
        """Create minimal options object for the model."""
        class MinimalOptions:
            def __init__(self):
                # Add required attributes based on your original options
                self.checkpoints_dir = 'app/models/weights'
                self.name = 'test_local'
                self.model = 'pix2pix'
                self.input_nc = 3
                self.output_nc = 1
                self.ngf = 64
                self.norm = 'batch'
                self.no_dropout = True
                self.init_type = 'normal'
                self.init_gain = 0.02
                self.gpu_ids = [0] if torch.cuda.is_available() else []
                self.isTrain = False
                self.netG = 'resnet_9blocks'
                self.which_epoch = 'best_depth_Ours_Bilinear_inc_3'
                self.suffix = ''
                
        return MinimalOptions()
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image to temporary file."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        image.save(temp_path, 'JPEG')
        return temp_path
    
    def _create_temp_file_list(self, image_path: str) -> str:
        """Create temporary file list for the dataloader."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
        with os.fdopen(temp_fd, 'w') as f:
            f.write(image_path)
        return temp_path
    
    async def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth map from input image using Pix2Pix model.
        
        Args:
            image: Input PIL image
            
        Returns:
            Depth map as numpy array
        """
        try:
            logger.info("Starting depth estimation with Pix2Pix model")
            
            # Load model if not already loaded
            self._load_model()
            
            # Save image to temporary file
            temp_image_path = self._save_temp_image(image)
            temp_file_list = self._create_temp_file_list(temp_image_path)
            
            try:
                # Get image dimensions
                img_array = np.array(image)
                img_shape = [img_array.shape[0], img_array.shape[1]]
                
                # Create data loader
                video_data_loader = aligned_data_loader.DAVISDataLoader(
                    temp_file_list, self.batch_size
                )
                video_dataset = video_data_loader.load_data()
                
                # Process through model
                depth_map = None
                for i, data in enumerate(video_dataset):
                    stacked_img = data[0]
                    targets = data[1]
                    
                    # Run depth prediction
                    depth_map = self.model.run_and_save_DAVIS(
                        stacked_img, targets, 'data/outputs/', img_shape
                    )
                    break  # Only process first (and only) image
                
                if depth_map is None:
                    raise RuntimeError("Depth estimation failed - no output generated")
                
                logger.info("Depth estimation completed successfully")
                return depth_map
                
            finally:
                # Cleanup temporary files
                try:
                    os.unlink(temp_image_path)
                    os.unlink(temp_file_list)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            raise
    
    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create visualization of depth map.
        
        Args:
            depth_map: Depth map array
            
        Returns:
            Normalized depth map for visualization
        """
        # Normalize depth map for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        return depth_colored
