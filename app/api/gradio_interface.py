"""Production-ready Gradio interface for BodyVision."""

import gradio as gr
import asyncio
import time
import json
from PIL import Image
from typing import Tuple, Optional
import numpy as np

from app.services import create_analysis_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProductionGradioInterface:
    """Production-grade Gradio interface for BodyVision."""
    
    def __init__(self):
        """Initialize production Gradio interface."""
        try:
            self.analysis_service = create_analysis_service()
            logger.info("✅ Production Gradio interface initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gradio interface: {e}")
            raise
    
    def analyze_image_sync(
        self, 
        image: Optional[Image.Image],
        height: float,
        weight: float,
        age: int,
        sex: str
    ) -> Tuple[str, str, str]:
        """
        Synchronous wrapper for async body analysis.
        
        Returns:
            Tuple of (summary, detailed_json, insights)
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if image is None:
                return "❌ Please upload an image", "{}", "Upload a clear photo showing your neck and waist areas."
            
            if not (100 <= height <= 250):
                return "❌ Height must be between 100-250 cm", "{}", "Please enter a valid height."
            
            logger.info("🎨 Processing image via Gradio interface")
            
            # Prepare user metadata
            user_metadata = {
                'height': height,
                'sex': sex.lower(),
            }
            
            if weight > 0:
                user_metadata['weight'] = weight
            if age > 0:
                user_metadata['age'] = age
            
            # Run analysis in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    self.analysis_service.analyze_body(image, user_metadata)
                )
            finally:
                loop.close()
            
            # Format results for Gradio display
            summary = self._format_summary(results)
            detailed_json = json.dumps(results, indent=2, default=str)
            insights = self._format_insights(results)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Gradio analysis completed in {total_time:.3f}s")
            
            return summary, detailed_json, insights
            
        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"❌ Analysis failed: {str(e)}"
            logger.error(f"Gradio analysis failed in {error_time:.3f}s: {e}")
            
            return error_msg, f'{{"error": "{str(e)}"}}', "Please try again with a clearer image."
    
    def _format_summary(self, results: dict) -> str:
        """Format analysis results for summary display."""
        
        if not results.get('success', False):
            return f"❌ Analysis failed: {results.get('error', 'Unknown error')}"
        
        # Extract key metrics
        body_fat = results.get('body_fat_percentage')
        category = results.get('body_fat_category', 'Unknown')
        neck_cm = results.get('neck_cm', 0)
        waist_cm = results.get('waist_cm', 0)
        confidence = results.get('confidence_score', 0)
        processing_time = results.get('processing_time_seconds', 0)
        
        # Build summary
        lines = []
        lines.append("🎯 BodyVision Analysis Results")
        lines.append("=" * 35)
        
        if body_fat is not None:
            lines.append(f"📊 Body Fat Percentage: {body_fat}%")
            lines.append(f"🏷️  Health Category: {category}")
            lines.append("")
            lines.append("📏 Measurements:")
            lines.append(f"   • Neck: {neck_cm:.1f} cm")
            lines.append(f"   • Waist: {waist_cm:.1f} cm")
        else:
            lines.append("⚠️  Could not calculate body fat percentage")
            lines.append("   Please try with a clearer image")
        
        lines.append("")
        lines.append("🔍 Analysis Quality:")
        lines.append(f"   • Confidence: {confidence:.1%}")
        lines.append(f"   • Processing Time: {processing_time:.2f}s")
        
        return "\n".join(lines)
    
    def _format_insights(self, results: dict) -> str:
        """Format health insights for display."""
        
        if not results.get('success', False):
            return "Upload a clear photo with good lighting for personalized health insights."
        
        body_fat = results.get('body_fat_percentage')
        category = results.get('body_fat_category', 'Unknown')
        sex = results.get('user_context', {}).get('sex', 'male')
        
        if body_fat is None:
            return (
                "💡 Tips for Better Results:\n"
                "• Ensure neck and waist are clearly visible\n"
                "• Use good lighting (natural light preferred)\n"
                "• Stand 1-2 meters from camera\n"
                "• Wear fitted clothing\n"
                "• Keep camera at chest level"
            )
        
        # Generate insights based on category
        insights_map = {
            'Essential Fat': {
                'summary': f"Your {body_fat}% body fat is in the essential fat range for {sex}s.",
                'advice': "This is very low body fat. Consider consulting a healthcare provider about maintaining healthy weight and nutrition.",
                'action': "Focus on balanced nutrition and appropriate exercise for your health goals."
            },
            'Athletes': {
                'summary': f"Excellent! Your {body_fat}% body fat is in the athletic range.",
                'advice': "You're in outstanding physical condition with low body fat and good muscle definition.",
                'action': "Maintain your training routine and ensure adequate recovery between workouts."
            },
            'Fitness': {
                'summary': f"Great work! Your {body_fat}% body fat is in the fitness range.",
                'advice': "You're in good shape with healthy body composition and visible muscle definition.",
                'action': "Continue your current routine. Consider adding variety to prevent plateaus."
            },
            'Average': {
                'summary': f"Your {body_fat}% body fat is in the average healthy range for {sex}s.",
                'advice': "You're within normal ranges. There's room for improvement if you have fitness goals.",
                'action': "Consider strength training 2-3x per week and balanced nutrition for better composition."
            },
            'Obese': {
                'summary': f"Your {body_fat}% body fat indicates opportunity for health improvement.",
                'advice': "Higher body fat levels may increase health risks. Consider lifestyle changes.",
                'action': "Consult healthcare and nutrition professionals for a personalized improvement plan."
            }
        }
        
        insight = insights_map.get(category, {
            'summary': f"Body fat analysis: {body_fat}%",
            'advice': "Continue monitoring your health with regular measurements.",
            'action': "Maintain balanced nutrition and regular exercise."
        })
        
        return (
            f"💡 Health Insights:\n\n"
            f"📋 Summary: {insight['summary']}\n\n"
            f"🎯 Guidance: {insight['advice']}\n\n"
            f"🚀 Next Steps: {insight['action']}\n\n"
            f"📈 Remember: Body composition is just one health metric. "
            f"Regular exercise, balanced nutrition, and overall wellness are key!"
        )
    
    def create_interface(self) -> gr.Interface:
        """Create the production Gradio interface."""
        
        # Custom CSS for better appearance
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gr-button-primary {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border: none;
        }
        .gr-form {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }
        """
        
        interface = gr.Interface(
            fn=self.analyze_image_sync,
            inputs=[
                gr.Image(
                    type="pil",
                    label="📸 Upload Your Photo",
                    info="Stand 1-2 meters from camera, ensure neck and waist are visible"
                ),
                gr.Number(
                    label="📏 Height (cm)",
                    value=175,
                    minimum=100,
                    maximum=250,
                    info="Your height in centimeters"
                ),
                gr.Number(
                    label="⚖️ Weight (kg) - Optional",
                    value=0,
                    minimum=0,
                    maximum=300,
                    info="Leave as 0 if you prefer not to share"
                ),
                gr.Number(
                    label="🎂 Age (years) - Optional",
                    value=0,
                    minimum=0,
                    maximum=120,
                    info="Leave as 0 if you prefer not to share"
                ),
                gr.Radio(
                    ["Male", "Female"],
                    label="👤 Gender",
                    value="Male",
                    info="Required for accurate body fat calculation"
                )
            ],
            outputs=[
                gr.Textbox(
                    label="📊 Analysis Summary",
                    lines=12,
                    max_lines=15,
                    info="Your body composition analysis results"
                ),
                gr.Code(
                    label="🔍 Detailed Results (JSON)",
                    language="json",
                    info="Complete analysis data for developers"
                ),
                gr.Textbox(
                    label="💡 Personalized Health Insights",
                    lines=10,
                    max_lines=12,
                    info="Health guidance based on your results"
                )
            ],
            title="🏃‍♂️ BodyVision - AI Body Composition Analysis",
            description="""
            **Get instant body fat percentage analysis from just a photo!**
            
            BodyVision uses advanced AI (MediaPipe + Navy body fat formula) to analyze your body composition.
            Simply upload a clear photo and get accurate results in seconds.
            
            📋 **Instructions:**
            • Stand 1-2 meters from the camera
            • Ensure your neck and waist areas are clearly visible  
            • Use good lighting (natural light works best)
            • Wear fitted clothing for better accuracy
            • Keep the camera at chest level
            """,
            examples=[
                ["assets/samples/204.jpg", 175, 70, 25, "Male"] if os.path.exists("assets/samples/204.jpg") else None
            ],
            theme=gr.themes.Soft(),
            css=custom_css,
            allow_flagging="never",
            analytics_enabled=False
        )
        
        return interface


def create_app():
    """Factory function to create Gradio app."""
    try:
        interface = ProductionGradioInterface()
        return interface.create_interface()
    except Exception as e:
        logger.error(f"Failed to create Gradio app: {e}")
        raise
