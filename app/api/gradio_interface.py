"""Production-ready Gradio interface for BodyVision with 3-photo support."""

import gradio as gr
import requests
import time
import json
import io
from PIL import Image
from typing import Tuple, Optional
import numpy as np

from app.utils.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ProductionGradioInterface:
    """Production-grade Gradio interface for BodyVision 3-photo analysis."""

    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize production Gradio interface with remote backend."""
        try:
            self.api_base_url = api_base_url or f"http://localhost:{settings.PORT}"
            # Test backend connectivity
            self._test_backend_connection()
            logger.info(
                f"✅ Production Gradio interface initialized with backend: {self.api_base_url}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gradio interface: {e}")
            raise

    def _test_backend_connection(self):
        """Test if FastAPI backend is accessible."""
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/health/", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Backend returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to FastAPI backend at {self.api_base_url}: {e}"
            )

    def analyze_three_photos_sync(
        self,
        front_image: Optional[Image.Image],
        side_image: Optional[Image.Image],
        back_image: Optional[Image.Image],
        height: float,
        weight: float,
        age: int,
        sex: str,
    ) -> Tuple[str, str, str]:
        """
        Analyze 3 images using remote FastAPI backend.

        Returns:
            Tuple of (summary, detailed_json, insights)
        """
        start_time = time.time()

        try:
            # Validate all 3 images are provided
            if not all([front_image, side_image, back_image]):
                missing = []
                if not front_image:
                    missing.append("Front view")
                if not side_image:
                    missing.append("Side view")
                if not back_image:
                    missing.append("Back view")

                return (
                    f"❌ Missing required photos: {', '.join(missing)}",
                    "{}",
                    "Please upload all 3 required photos:\n• Front view (facing camera)\n• Side view (90° profile)\n• Back view (rear view)",
                )

            if not (100 <= height <= 250):
                return (
                    "❌ Height must be between 100-250 cm",
                    "{}",
                    "Please enter a valid height.",
                )

            logger.info("🎨 Processing 3 photos via remote FastAPI backend")

            # Prepare all 3 images for upload
            files = {}
            for name, image in [
                ("front_image", front_image),
                ("side_image", side_image),
                ("back_image", back_image),
            ]:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG", quality=90)
                img_byte_arr.seek(0)
                files[name] = (f"{name}.jpg", img_byte_arr, "image/jpeg")

            # Prepare request data
            data = {
                "height": height,
                "sex": sex.lower(),
            }

            # Add optional fields only if provided
            if weight > 0:
                data["weight"] = weight
            if age > 0:
                data["age"] = age

            # Call FastAPI backend with 3 photos
            response = requests.post(
                f"{self.api_base_url}/api/v1/analysis/analyze",
                files=files,
                data=data,
                timeout=60,  # Longer timeout for 3-photo processing
            )

            if response.status_code == 200:
                results = response.json()

                # Format results for Gradio display
                summary = self._format_three_photo_summary(results)
                detailed_json = json.dumps(results, indent=2, default=str)
                insights = self._format_three_photo_insights(results)

                total_time = time.time() - start_time
                logger.info(f"✅ 3-photo analysis completed in {total_time:.3f}s")

                return summary, detailed_json, insights
            else:
                error_msg = (
                    f"❌ Backend error ({response.status_code}): {response.text}"
                )
                logger.error(error_msg)
                return (
                    error_msg,
                    f'{{"error": "{error_msg}"}}',
                    "Backend service encountered an error. Please try again.",
                )

        except requests.exceptions.Timeout:
            error_msg = "❌ Request timed out - 3-photo analysis takes longer"
            logger.error(error_msg)
            return (
                error_msg,
                f'{{"error": "timeout"}}',
                "3-photo analysis is taking longer than expected. Please try again.",
            )

        except requests.exceptions.ConnectionError:
            error_msg = "❌ Cannot connect to backend service"
            logger.error(error_msg)
            return (
                error_msg,
                f'{{"error": "connection_failed"}}',
                "Backend service is unavailable. Please check if the server is running.",
            )

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"❌ 3-photo analysis failed: {str(e)}"
            logger.error(f"3-photo analysis failed in {error_time:.3f}s: {e}")

            return (
                error_msg,
                f'{{"error": "{str(e)}"}}',
                "Please try again with clear, well-lit photos.",
            )

    def _format_three_photo_summary(self, results: dict) -> str:
        """Format 3-photo analysis results for summary display."""

        if not results.get("success", False):
            return f"❌ Analysis failed: {results.get('error', 'Unknown error')}"

        # Extract metrics from results
        body_fat = results.get("body_fat_percentage")
        category = results.get("body_fat_category", "Unknown")
        lean_mass = results.get("lean_muscle_mass_kg")
        bmi = results.get("bmi")
        bmi_category = results.get("bmi_category")
        whr = results.get("waist_to_hip_ratio")
        neck_cm = results.get("neck_cm", 0)
        waist_cm = results.get("waist_cm", 0)
        chest_cm = results.get("chest_cm", 0)
        hip_cm = results.get("hip_cm", 0)
        shoulder_width = results.get("shoulder_width_cm", 0)
        body_surface_area = results.get("body_surface_area_m2", 0)

        # Quality metrics
        confidence = results.get("confidence_score", 0)
        processing_time = results.get("processing_time_seconds", 0)
        analysis_id = results.get("analysis_id", "N/A")
        photos_processed = results.get("photos_processed", 3)

        # Build comprehensive summary
        lines = []
        lines.append("🎯 BodyVision 3-Photo Analysis Results")
        lines.append("=" * 45)

        if body_fat is not None:
            lines.append(f"🔥 Body Fat Percentage: {body_fat}%")
            lines.append(f"🏷️  Health Category: {category}")
            lines.append("")

            # Enhanced metrics
            if lean_mass:
                lines.append(f"💪 Lean Muscle Mass: {lean_mass} kg")
            if bmi:
                lines.append(f"📊 BMI: {bmi} ({bmi_category})")
            if whr:
                lines.append(f"❤️  Waist-to-Hip Ratio: {whr:.3f}")
            if body_surface_area:
                lines.append(f"🧬 Body Surface Area: {body_surface_area:.2f} m²")

            lines.append("")
            lines.append("📏 Detailed Measurements:")
            lines.append(f"   • Neck: {neck_cm:.1f} cm")
            lines.append(f"   • Waist: {waist_cm:.1f} cm")
            if chest_cm:
                lines.append(f"   • Chest: {chest_cm:.1f} cm")
            if hip_cm:
                lines.append(f"   • Hip: {hip_cm:.1f} cm")
            if shoulder_width:
                lines.append(f"   • Shoulder Width: {shoulder_width:.1f} cm")
        else:
            lines.append("⚠️  Could not calculate body fat percentage")
            lines.append("   Please ensure all 3 photos are clear and well-lit")

        lines.append("")
        lines.append("🔍 Analysis Quality:")
        lines.append(f"   • Overall Confidence: {confidence:.1%}")
        lines.append(f"   • Photos Processed: {photos_processed}/3")
        lines.append(f"   • Processing Time: {processing_time:.2f}s")
        lines.append(f"   • Analysis ID: {analysis_id}")
        lines.append("   • Mode: 3-Photo Comprehensive")

        return "\n".join(lines)

    def _format_three_photo_insights(self, results: dict) -> str:
        """Format health insights for 3-photo analysis."""

        if not results.get("success", False):
            return "Upload 3 clear, well-lit photos for comprehensive health insights."

        body_fat = results.get("body_fat_percentage")
        category = results.get("body_fat_category", "Unknown")
        cv_risk = results.get("cardiovascular_risk", "Unknown")
        sleep_risk = results.get("sleep_apnea_risk", "Unknown")

        # Get user context
        user_context = results.get("user_context", {})
        sex = user_context.get("sex", "male")

        if body_fat is None:
            return (
                "💡 Tips for Better 3-Photo Results:\n"
                "• Ensure all 3 views are clearly visible\n"
                "• Use consistent, good lighting for all photos\n"
                "• Stand 1.5-2 meters from camera\n"
                "• Wear fitted clothing\n"
                "• Keep camera at chest level\n"
                "• Take photos in this order: Front → Side → Back\n"
                "• Check that backend service is running"
            )

        # Generate comprehensive insights
        insights_map = {
            "Essential Fat": {
                "summary": f"Your {body_fat}% body fat is in the essential fat range for {sex}s.",
                "advice": "This is very low body fat. Consider consulting a healthcare provider about maintaining healthy weight and nutrition.",
                "action": "Focus on balanced nutrition and appropriate exercise for your health goals.",
            },
            "Athletes": {
                "summary": f"Excellent! Your {body_fat}% body fat is in the athletic range.",
                "advice": "You're in outstanding physical condition with low body fat and excellent muscle definition.",
                "action": "Maintain your training routine and ensure adequate recovery between workouts.",
            },
            "Fitness": {
                "summary": f"Great work! Your {body_fat}% body fat is in the fitness range.",
                "advice": "You're in good shape with healthy body composition and visible muscle definition.",
                "action": "Continue your current routine. Consider adding variety to prevent plateaus.",
            },
            "Average": {
                "summary": f"Your {body_fat}% body fat is in the average healthy range for {sex}s.",
                "advice": "You're within normal ranges. There's room for improvement if you have fitness goals.",
                "action": "Consider strength training 2-3x per week and balanced nutrition for better composition.",
            },
            "Obese": {
                "summary": f"Your {body_fat}% body fat indicates opportunity for health improvement.",
                "advice": "Higher body fat levels may increase health risks. Consider lifestyle changes.",
                "action": "Consult healthcare and nutrition professionals for a personalized improvement plan.",
            },
        }

        insight = insights_map.get(
            category,
            {
                "summary": f"Body fat analysis: {body_fat}%",
                "advice": "Continue monitoring your health with regular measurements.",
                "action": "Maintain balanced nutrition and regular exercise.",
            },
        )

        insight_text = (
            f"💡 Comprehensive Health Insights:\n\n"
            f"📋 Summary: {insight['summary']}\n\n"
            f"🎯 Guidance: {insight['advice']}\n\n"
            f"🚀 Next Steps: {insight['action']}\n\n"
        )

        # Add risk assessments
        if cv_risk != "Unknown":
            insight_text += f"❤️  Cardiovascular Risk: {cv_risk}\n"
        if sleep_risk != "Unknown":
            insight_text += f"😴 Sleep Apnea Risk: {sleep_risk}\n"

        insight_text += (
            f"\n📈 3-Photo Advantage: This comprehensive analysis provides "
            f"enhanced accuracy through multi-angle assessment including posture, "
            f"symmetry, and complete body composition.\n\n"
            f"🌐 Analysis powered by MediaPipe 3-photo detection with FastAPI backend."
        )

        return insight_text

    def create_interface(self) -> gr.Interface:
        """Create the production Gradio interface for 3-photo analysis."""

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
        .photo-upload {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        """

        interface = gr.Interface(
            fn=self.analyze_three_photos_sync,
            inputs=[
                gr.Image(
                    type="pil",
                    label="📸 Front View Photo (Required)",
                    elem_classes=["photo-upload"],
                ),
                gr.Image(
                    type="pil",
                    label="📸 Side View Photo (Required)",
                    elem_classes=["photo-upload"],
                ),
                gr.Image(
                    type="pil",
                    label="📸 Back View Photo (Required)",
                    elem_classes=["photo-upload"],
                ),
                gr.Number(
                    label="📏 Height (cm)",
                    value=175,
                    minimum=100,
                    maximum=250,
                    info="Your height in centimeters",
                ),
                gr.Number(
                    label="⚖️ Weight (kg) - Optional",
                    value=0,
                    minimum=0,
                    maximum=300,
                    info="Leave as 0 if you prefer not to share",
                ),
                gr.Number(
                    label="🎂 Age (years) - Optional",
                    value=0,
                    minimum=0,
                    maximum=120,
                    info="Leave as 0 if you prefer not to share",
                ),
                gr.Radio(
                    ["Male", "Female"],
                    label="👤 Gender",
                    value="Male",
                    info="Required for accurate body fat calculation",
                ),
            ],
            outputs=[
                gr.Textbox(
                    label="📊 Comprehensive Analysis Summary",
                    lines=20,
                    max_lines=25,
                    info="Your complete 9-metric body composition analysis",
                ),
                gr.Code(
                    label="🔍 Detailed Results (JSON)",
                    language="json",
                    # info="Complete analysis data including all metrics",
                ),
                gr.Textbox(
                    label="💡 Personalized Health Insights",
                    lines=15,
                    max_lines=20,
                    info="Health guidance and risk assessment based on 3-photo analysis",
                ),
            ],
            title="🏃‍♂️ BodyVision - 3-Photo AI Body Analysis",
            description=f"""
            **Comprehensive body composition analysis from 3 photos!**
            
            BodyVision analyzes your complete body composition using 3 photos for maximum accuracy and delivers 9 key health metrics.
            
            📋 **Required Photos (All 3 Needed):**
            1. **Front View**: Face the camera directly, arms slightly away from sides
            2. **Side View**: Turn 90° to your right, arms relaxed at sides  
            3. **Back View**: Turn around completely, arms slightly away from sides
            
            📊 **You'll Get 9 Health Metrics:**
            • Body Fat Percentage • Lean Muscle Mass • BMI Analysis
            • Waist-to-Hip Ratio • Neck Circumference • Chest-to-Waist Ratio
            • Shoulder Width & Symmetry • Body Surface Area • Risk Assessment
            
            🎯 **Photo Guidelines:**
            • Stand 4-6 feet from camera • Use good, even lighting
            • Wear fitted athletic clothing • Plain background preferred
            • Keep camera at chest level • Take all photos in same session
            
            🌐 **Backend:** Remote FastAPI service at {self.api_base_url}
            """,
            examples=[],
            theme=gr.themes.Soft(),
            css=custom_css,
            allow_flagging="never",
            analytics_enabled=False,
        )

        return interface


def create_app(api_url: Optional[str] = None):
    """Factory function to create Gradio app with 3-photo remote backend."""
    try:
        interface = ProductionGradioInterface(api_url)
        return interface.create_interface()
    except Exception as e:
        logger.error(f"Failed to create 3-photo Gradio app: {e}")
        raise
