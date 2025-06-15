"""Gradio app integration with FastAPI backend."""

import gradio as gr
import requests
import io
import json
from PIL import Image
from typing import Tuple, Optional

from app.utils.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class GradioFastAPIInterface:
    """Gradio interface that calls FastAPI backend."""
    
    def __init__(self, api_base_url: Optional[str] = None):
        self.api_base_url = api_base_url or f"http://localhost:{settings.PORT}"
        logger.info(f"GradioFastAPIInterface initialized with API: {self.api_base_url}")
    
    def analyze_via_api(
        self, 
        image: Optional[Image.Image],
        height: float,
        weight: float,
        age: int,
        sex: str
    ) -> Tuple[str, str, str]:
        """
        Analyze image using FastAPI backend.
        
        Returns:
            Tuple of (summary, json_results, insights)
        """
        try:
            if image is None:
                return "❌ Please upload an image", "{}", "Upload a clear photo for analysis."
            
            logger.info("🌐 Analyzing via FastAPI backend")
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=90)
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
                
                # Format for Gradio display
                summary = self._format_api_summary(result)
                json_results = json.dumps(result, indent=2, default=str)
                insights = self._format_api_insights(result)
                
                logger.info("✅ Analysis via API completed successfully")
                return summary, json_results, insights
                
            else:
                error_msg = f"❌ API Error ({response.status_code}): {response.text}"
                logger.error(error_msg)
                return error_msg, f'{{"error": "{error_msg}"}}', "Please try again or check server status."
                
        except Exception as e:
            error_msg = f"❌ Request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, f'{{"error": "{str(e)}"}}', "Check that FastAPI server is running."
    
    def _format_api_summary(self, result: dict) -> str:
        """Format FastAPI response for summary display."""
        
        if not result.get('success', False):
            return f"❌ Analysis failed: {result.get('error', 'Unknown error')}"
        
        body_comp = result.get('body_composition', {}) or result
        
        lines = []
        lines.append("🌐 BodyVision API Analysis Results")
        lines.append("=" * 38)
        
        if body_comp.get('body_fat_percentage') is not None:
            lines.append(f"📊 Body Fat: {body_comp['body_fat_percentage']}%")
            lines.append(f"🏷️  Category: {body_comp.get('body_fat_category', 'Unknown')}")
            lines.append("")
            lines.append("📏 Measurements:")
            lines.append(f"   • Neck: {body_comp.get('neck_cm', 0):.1f} cm")
            lines.append(f"   • Waist: {body_comp.get('waist_cm', 0):.1f} cm")
        
        lines.append("")
        lines.append("⚡ Performance:")
        lines.append(f"   • Processing: {result.get('processing_time_seconds', 0):.3f}s")
        lines.append(f"   • Analysis ID: {result.get('analysis_id', 'N/A')}")
        
        return "\n".join(lines)
    
    def _format_api_insights(self, result: dict) -> str:
        """Format API insights for display."""
        
        if not result.get('success', False):
            return "Upload a clear photo for personalized insights via our API."
        
        # Use body_composition data or fallback to root level
        body_comp = result.get('body_composition', {}) or result
        body_fat = body_comp.get('body_fat_percentage')
        category = body_comp.get('body_fat_category', 'Unknown')
        
        if body_fat is None:
            return (
                "💡 API Analysis Tips:\n"
                "• Ensure good image quality\n"
                "• Check that FastAPI server is running\n"
                "• Verify MediaPipe detection is working\n"
                "• Try with different lighting conditions"
            )
        
        return (
            f"🌐 API Health Insights:\n\n"
            f"📊 Your {body_fat}% body fat places you in the '{category}' category.\n\n"
            f"🎯 This analysis was processed through our production FastAPI backend "
            f"using MediaPipe detection and the Navy body fat formula.\n\n"
            f"📈 Regular monitoring can help track your health progress over time!"
        )
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface for FastAPI backend."""
        
        return gr.Interface(
            fn=self.analyze_via_api,
            inputs=[
                gr.Image(type="pil", label="📸 Upload Photo"),
                gr.Number(label="📏 Height (cm)", value=175, minimum=100, maximum=250),
                gr.Number(label="⚖️ Weight (kg)", value=0, minimum=0, maximum=300),
                gr.Number(label="🎂 Age", value=0, minimum=0, maximum=120),
                gr.Radio(["Male", "Female"], label="👤 Gender", value="Male")
            ],
            outputs=[
                gr.Textbox(label="📊 API Analysis Summary", lines=10),
                gr.Code(label="🔍 Full API Response", language="json"),
                gr.Textbox(label="💡 Health Insights", lines=8)
            ],
            title="🌐 BodyVision - FastAPI Backend Analysis",
            description="""
            **Analyze body composition via FastAPI backend**
            
            This interface calls the production FastAPI server for analysis.
            Ensure the FastAPI server is running at http://localhost:8000
            """,
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )


def create_fastapi_gradio_app(api_url: Optional[str] = None):
    """Create Gradio app that uses FastAPI backend."""
    interface = GradioFastAPIInterface(api_url)
    return interface.create_interface()
