"""Gradio application integration with FastAPI backend."""

import gradio as gr
import requests
import io
from PIL import Image
from typing import Tuple

from app.utils.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class GradioFastAPIInterface:
    """Gradio interface that uses FastAPI backend."""
    
    def __init__(self, api_base_url: str = None):
        self.api_base_url = api_base_url or f"http://localhost:{settings.PORT}"
        logger.info(f"GradioFastAPIInterface initialized with API: {self.api_base_url}")
    
    def analyze_image(
        self, 
        image: Image.Image,
        height: float,
        weight: float,
        age: int,
        sex: str
    ) -> Tuple[str, str]:
        """
        Process image using FastAPI backend.
        
        Returns:
            Tuple of (formatted_summary, json_results)
        """
        try:
            logger.info("Processing image via FastAPI backend")
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Prepare request
            files = {"image": ("image.jpg", img_byte_arr, "image/jpeg")}
            data = {
                "height": height,
                "weight": weight if weight > 0 else None,
                "age": age if age > 0 else None,
                "sex": sex.lower()
            }
            
            # Call FastAPI endpoint
            response = requests.post(
                f"{self.api_base_url}/api/v1/analysis/analyze",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format summary
                summary = self._format_results(result)
                
                # Return JSON string
                import json
                json_results = json.dumps(result, indent=2)
                
                logger.info("Analysis completed successfully via API")
                return summary, json_results
                
            else:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                logger.error(error_msg)
                return error_msg, f'{{"error": "{error_msg}"}}'
                
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, f'{{"error": "{str(e)}"}}'
    
    def _format_results(self, result: dict) -> str:
        """Format API results for display."""
        if not result.get('success', False):
            return f"Analysis Error: {result.get('error', 'Unknown error')}"
        
        body_comp = result.get('body_composition', {})
        measurements = result.get('measurements', {})
        
        lines = []
        
        if 'neck_cm' in body_comp and 'waist_cm' in body_comp:
            lines.append(f"Neck Circumference: {body_comp['neck_cm']:.2f} cm")
            lines.append(f"Waist Circumference: {body_comp['waist_cm']:.2f} cm")
        
        if body_comp.get('body_fat_percentage') is not None:
            lines.append(f"Body Fat Percentage: {body_comp['body_fat_percentage']:.2f}%")
            
            if body_comp.get('body_fat_category'):
                lines.append(f"Category: {body_comp['body_fat_category']}")
        
        processing_time = result.get('processing_time_seconds', 0)
        lines.append(f"\nProcessing Time: {processing_time:.3f} seconds")
        
        return '\n'.join(lines) if lines else "No measurements available"
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""
        
        interface = gr.Interface(
            fn=self.analyze_image,
            inputs=[
                gr.Image(type="pil", label="📸 Upload your photo"),
                gr.Number(label="📏 Height (cm)", value=175, minimum=100, maximum=250),
                gr.Number(label="⚖️ Weight (kg)", value=70, minimum=0, maximum=300),
                gr.Number(label="🎂 Age (years)", value=30, minimum=0, maximum=120),
                gr.Radio(["Male", "Female"], label="👤 Gender", value="Male")
            ],
            outputs=[
                gr.Textbox(label="📊 Analysis Summary", lines=6),
                gr.Code(label="🔍 Detailed Results (JSON)", language="json")
            ],
            title="🏃‍♂️ BodyVision - AI Body Analysis (FastAPI Backend)",
            description="""
            Upload a photo to get instant body composition analysis using AI.
            
            **📋 Instructions:**
            - Stand at least 1 meter from camera
            - Ensure neck and waist are clearly visible
            - Good lighting and minimal background preferred
            - Wear fitted clothing for better accuracy
            
            **🔧 Powered by FastAPI backend**
            """,
            examples=[
                ["assets/samples/204.jpg", 175, 75, 28, "Male"]
            ] if os.path.exists("assets/samples/204.jpg") else None,
            theme="soft",
            allow_flagging="never"
        )
        
        return interface

def create_gradio_app():
    """Create Gradio app instance."""
    interface = GradioFastAPIInterface()
    return interface.create_interface()
