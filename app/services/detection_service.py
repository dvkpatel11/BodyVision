"""Body part detection service using RetinaNet."""

import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, Optional
import tempfile

# Add legacy modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'legacy'))

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, UnNormalizer, Normalizer
import model as legacy_model

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionService:
    """Service for detecting body parts using RetinaNet."""
    
    def __init__(self, model_path: Optional[str] = None, classes_path: Optional[str] = None):
        self.model_path = model_path or 'app/models/weights/csv_retinanet_25.pt'
        self.classes_path = classes_path or 'config/classes.csv'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.unnormalize = UnNormalizer()
        
        logger.info(f"DetectionService initialized with device: {self.device}")
    
    def _load_model(self):
        """Load the RetinaNet model."""
        if self.model is None:
            try:
                # Create a temporary dataset to get num_classes
                temp_csv = self._create_temp_csv("dummy.jpg")
                temp_dataset = CSVDataset(
                    train_file=temp_csv, 
                    class_list=self.classes_path, 
                    transform=transforms.Compose([Normalizer(), Resizer()])
                )
                
                # Load model
                self.model = legacy_model.resnet50(
                    num_classes=temp_dataset.num_classes(), 
                    pretrained=False
                )
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Cleanup temp file
                os.unlink(temp_csv)
                
                logger.info("RetinaNet model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load detection model: {str(e)}")
                raise
    
    def _create_temp_csv(self, image_path: str) -> str:
        """Create temporary CSV file for the image."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
        with os.fdopen(temp_fd, 'w') as f:
            # CSV format: image_path,x1,y1,x2,y2,class_name
            f.write(f"{image_path},,,,\n")
        return temp_path
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save PIL image to temporary file."""
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)  # Close file descriptor, we'll use the path
        image.save(temp_path, 'JPEG')
        return temp_path
    
    async def detect_body_parts(self, image: Image.Image) -> Dict[str, Dict[str, int]]:
        """
        Detect body parts in the input image using RetinaNet.
        
        Args:
            image: Input PIL image
            
        Returns:
            Dictionary of detected body parts with bounding boxes
        """
        try:
            logger.info("Starting body part detection with RetinaNet")
            
            # Load model if not already loaded
            self._load_model()
            
            # Save image to temporary file (needed for CSVDataset)
            temp_image_path = self._save_temp_image(image)
            temp_csv_path = self._create_temp_csv(temp_image_path)
            
            try:
                # Create dataset and dataloader
                dataset = CSVDataset(
                    train_file=temp_csv_path,
                    class_list=self.classes_path,
                    transform=transforms.Compose([Normalizer(), Resizer()])
                )
                
                sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
                dataloader = DataLoader(dataset, num_workers=0, collate_fn=collater, batch_sampler=sampler)
                
                # Run inference
                for idx, data in enumerate(dataloader):
                    with torch.no_grad():
                        scores, classification, transformed_anchors = self.model(
                            data['img'].to(self.device).float()
                        )
                        
                        scores = scores.cpu().numpy()
                        classification = classification.cpu().numpy()
                        transformed_anchors = transformed_anchors.cpu().numpy()
                        
                        # Extract bounding boxes
                        bbox = self._extract_bounding_boxes(classification, transformed_anchors)
                        
                        logger.info("Body part detection completed successfully")
                        return bbox
                        
            finally:
                # Cleanup temporary files
                try:
                    os.unlink(temp_image_path)
                    os.unlink(temp_csv_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Body part detection failed: {str(e)}")
            raise
    
    def _extract_bounding_boxes(self, classification: np.ndarray, transformed_anchors: np.ndarray) -> Dict[str, Dict[str, int]]:
        """Extract bounding boxes for detected body parts."""
        
        def get_bbox(classification, transformed_anchors, label):
            """Get bounding box for specific label."""
            try:
                indices = np.where(classification == label)[0]
                if len(indices) == 0:
                    logger.warning(f"No detection found for label {label}")
                    return {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}  # Default bbox
                
                idx = indices[0]  # Take first detection
                coords = transformed_anchors[idx, :]
                return {
                    'x1': int(coords[0]),
                    'y1': int(coords[1]),
                    'x2': int(coords[2]),
                    'y2': int(coords[3])
                }
            except Exception as e:
                logger.warning(f"Failed to extract bbox for label {label}: {str(e)}")
                return {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}
        
        bbox = {}
        bbox['neck'] = get_bbox(classification, transformed_anchors, label=0)
        bbox['stomach'] = get_bbox(classification, transformed_anchors, label=1)
        
        logger.debug(f"Extracted bounding boxes: {bbox}")
        return bbox
    
    def visualize_detections(self, image: Image.Image, bbox: Dict[str, Dict[str, int]]) -> np.ndarray:
        """
        Visualize detections on image for debugging.
        
        Args:
            image: Original PIL image
            bbox: Bounding box dictionary
            
        Returns:
            Image with bounding boxes drawn
        """
        # Convert PIL to numpy
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        if 'neck' in bbox:
            neck = bbox['neck']
            cv2.rectangle(img_array, (neck['x1'], neck['y1']), (neck['x2'], neck['y2']), 
                         color=(0, 0, 255), thickness=2)
            cv2.putText(img_array, 'Neck', (neck['x1'], neck['y1']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if 'stomach' in bbox:
            stomach = bbox['stomach']
            cv2.rectangle(img_array, (stomach['x1'], stomach['y1']), (stomach['x2'], stomach['y2']), 
                         color=(0, 255, 0), thickness=2)
            cv2.putText(img_array, 'Stomach', (stomach['x1'], stomach['y1']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_array
