"""Enhanced Gradio interface for BodyVision with visualization options."""

import gradio as gr
import asyncio
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import cv2

from app.core.body_analyzer import BodyAnalyzer
from app.services import create_analysis_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GradioInterface:
    """Enhanced Gradio web interface for BodyVision."""
    
    def __init__(self):
        self.analyzer = BodyAnalyzer()
        self.analysis_service = create_analysis_service()
        logger.info("Enhanced GradioInterface initialized")
    
    def analyze_image(
        self, 
        image: Image.Image,
        height: float,
        weight: float,
        age: int,
        sex: str,
        show_detections: bool = True,
        show_depth: bool = True
    ) -> Tuple[str, str, Optional[Image.Image], Optional[Image.Image]]:
        """
        Process image and return analysis results with visualizations.
        
        Returns:
            Tuple of (formatted_summary, json_results, detection_viz, depth_viz)
        """
        try:
            logger.info("Processing new image analysis request with visualizations")
            
            # Run async analysis in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                self.analyzer.analyze(
                    image=image,
                    height=height,
                    weight=weight if weight > 0 else None,
                    age=age if age > 0 else None,
                    sex=sex.lower()
                )
            )
            
            # Generate visualizations if requested
            detection_viz = None
            depth_viz = None
            
            if show_detections and 'detections' in results:
                try:
                    detection_array = self.analysis_service.detection_service.visualize_detections(
                        image, results['detections']
                    )
                    detection_viz = Image.fromarray(cv2.cvtColor(detection_array, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    logger.warning(f"Failed to generate detection visualization: {str(e)}")
            
            if show_depth:
                try:
                    # Get depth map
                    depth_map = loop.run_until_complete(
                        self.analysis_service.depth_service.estimate_depth(image)
                    )
                    depth_colored = self.analysis_service.depth_service.visualize_depth(depth_map)
                    depth_viz = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    logger.warning(f"Failed to generate depth visualization: {str(e)}")
            
            loop.close()
            
            # Format results for display
            summary = self.analyzer.format_results(results)
            
            # Convert results to JSON string for detailed view
            import json
            json_results = json.dumps(results, indent=2, default=str)
            
            logger.info("Analysis completed successfully with visualizations")
            return summary, json_results, detection_viz, depth_viz
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, f'{{"error": "{str(e)}"}}', None, None
    
    def create_interface(self) -> gr.Interface:
        """Create and configure the enhanced Gradio interface."""
        
        with gr.Blocks(title="BodyVision - AI Body Analysis", theme="soft") as interface:
            gr.Markdown("""
            # 🏃‍♂️ BodyVision - AI Body Analysis
            
            Upload a photo to get instant body composition analysis using AI.
            
            **📋 Instructions:**
            - Stand at least 1 meter from camera
            - Ensure neck and waist are clearly visible  
            - Good lighting and minimal background preferred
            - Wear fitted clothing for better accuracy
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    image_input = gr.Image(type="pil", label="📸 Upload your photo")
                    height_input = gr.Number(label="📏 Height (cm)", value=175, minimum=100, maximum=250)
                    weight_input = gr.Number(label="⚖️ Weight (kg)", value=70, minimum=0, maximum=300)
                    age_input = gr.Number(label="🎂 Age (years)", value=30, minimum=0, maximum=120)
                    sex_input = gr.Radio(["Male", "Female"], label="👤 Gender", value="Male")
                    
                    # Visualization options
                    gr.Markdown("### 🎨 Visualization Options")
                    show_detections = gr.Checkbox(label="Show body part detections", value=True)
                    show_depth = gr.Checkbox(label="Show depth map", value=True)
                    
                    # Analyze button
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    # Output displays
                    with gr.Row():
                        summary_output = gr.Textbox(label="📊 Analysis Summary", lines=6)
                        json_output = gr.Code(label="🔍 Detailed Results (JSON)", language="json")
                    
                    with gr.Row():
                        detection_output = gr.Image(label="🎯 Body Part Detections", type="pil")
                        depth_output = gr.Image(label="🌊 Depth Map", type="pil")
            
            # Connect the analyze button
            analyze_btn.click(
                fn=self.analyze_image,
                inputs=[
                    image_input, height_input, weight_input, age_input, sex_input,
                    show_detections, show_depth
                ],
                outputs=[summary_output, json_output, detection_output, depth_output]
            )
            
            # Example section
            gr.Markdown("### 📚 Example")
            if os.path.exists("assets/samples/204.jpg"):
                gr.Examples(
                    examples=[["assets/samples/204.jpg", 175, 75, 28, "Male", True, True]],
                    inputs=[image_input, height_input, weight_input, age_input, sex_input, show_detections, show_depth]
                )
        
        return interface


def create_app():
    """Create and return the enhanced Gradio app."""
    interface = GradioInterface()
    return interface.create_interface()
