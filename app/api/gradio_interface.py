"""Enhanced production-ready Gradio interface for BodyVision with improved UX."""

import gradio as gr
import requests
import time
import json
import io
from PIL import Image
from typing import Tuple, Optional
from app.api.result_parser import BodyVisionResultParser
from app.utils.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ProductionGradioInterface:
    """Enhanced production-grade Gradio interface with improved UX and error handling."""

    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize enhanced Gradio interface with remote backend."""
        try:
            self.api_base_url = api_base_url or f"http://localhost:{settings.PORT}"
            self.session = requests.Session()
            self.session.timeout = 60
            # Test backend connectivity
            self._test_backend_connection()
            logger.info(
                f"✅ Enhanced Gradio interface initialized with backend: {self.api_base_url}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gradio interface: {e}")
            raise

    def _test_backend_connection(self):
        """Test if FastAPI backend is accessible with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.api_base_url}/api/v1/health/", timeout=5
                )
                if response.status_code == 200:
                    return
                else:
                    raise ConnectionError(
                        f"Backend returned status {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(
                        f"Cannot connect to FastAPI backend at {self.api_base_url} after {max_retries} attempts: {e}"
                    )
                time.sleep(1)  # Brief delay between retries

    def _validate_images(self, front_image, side_image, back_image) -> Tuple[bool, str]:
        """Enhanced image validation with quality checks."""
        missing_images = []
        if not front_image:
            missing_images.append("Front view")
        if not side_image:
            missing_images.append("Side view")
        if not back_image:
            missing_images.append("Back view")

        if missing_images:
            return False, f"Missing required photos: {', '.join(missing_images)}"

        # Image quality validation
        for name, image in [
            ("Front", front_image),
            ("Side", side_image),
            ("Back", back_image),
        ]:
            if image.size[0] < 200 or image.size[1] < 200:
                return False, f"{name} image is too small (minimum 200x200 pixels)"

            # Check if image is too large (may cause processing issues)
            if image.size[0] > 4000 or image.size[1] > 4000:
                return False, f"{name} image is too large (maximum 4000x4000 pixels)"

        return True, ""

    def _validate_inputs(
        self, height: float, weight: float, age: int
    ) -> Tuple[bool, str]:
        """Enhanced input validation."""
        if not (100 <= height <= 250):
            return False, "Height must be between 100-250 cm"

        if weight < 0 or weight > 500:
            return False, "Weight must be between 0-500 kg (0 to skip)"

        if age < 0 or age > 120:
            return False, "Age must be between 0-120 years (0 to skip)"

        return True, ""

    def _prepare_image_for_upload(
        self, image: Image.Image, name: str
    ) -> Tuple[str, io.BytesIO, str]:
        """Prepare image for upload with optimization."""
        # Optimize image size while maintaining quality
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(img_byte_arr, format="JPEG", quality=85, optimize=True)
        img_byte_arr.seek(0)
        return (f"{name}.jpg", img_byte_arr, "image/jpeg")

    def analyze_three_photos_sync(
        self,
        front_image: Optional[Image.Image],
        side_image: Optional[Image.Image],
        back_image: Optional[Image.Image],
        height: float,
        weight: float,
        age: int,
        sex: str,
        progress=gr.Progress(),
    ) -> Tuple[str, str, str]:
        """
        Enhanced 3-photo analysis with progress tracking and better error handling.
        """
        start_time = time.time()

        try:
            # Progress: Input validation
            progress(0.1, desc="Validating inputs...")

            # Validate images
            valid_images, error_msg = self._validate_images(
                front_image, side_image, back_image
            )
            if not valid_images:
                return (f"❌ {error_msg}", "{}", self._get_photo_tips())

            # Validate other inputs
            valid_inputs, error_msg = self._validate_inputs(height, weight, age)
            if not valid_inputs:
                return (
                    f"❌ {error_msg}",
                    "{}",
                    "Please check your input values and try again.",
                )

            progress(0.2, desc="Preparing images...")
            logger.info("🎨 Processing 3 photos via remote FastAPI backend")

            # Prepare images for upload with optimization
            files = {}
            for name, image in [
                ("front_image", front_image),
                ("side_image", side_image),
                ("back_image", back_image),
            ]:
                files[name] = self._prepare_image_for_upload(image, name)

            progress(0.3, desc="Preparing request data...")

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

            progress(0.4, desc="Sending to backend for analysis...")

            # Call FastAPI backend with enhanced error handling
            try:
                response = self.session.post(
                    f"{self.api_base_url}/api/v1/analysis/analyze",
                    files=files,
                    data=data,
                    timeout=90,  # Increased timeout for complex processing
                )
            except requests.exceptions.Timeout:
                return self._handle_timeout_error()
            except requests.exceptions.ConnectionError:
                return self._handle_connection_error()

            progress(0.8, desc="Processing results...")

            if response.status_code == 200:
                raw_results = response.json()

                progress(0.85, desc="Validating and enhancing results...")

                parser = BodyVisionResultParser()
                enhanced_results = parser.parse_and_validate_results(raw_results)

                progress(0.9, desc="Formatting results...")

                # Format results using enhanced methods
                summary = parser.format_enhanced_summary(enhanced_results)
                detailed_json = json.dumps(enhanced_results, indent=2, default=str)
                insights = self._format_enhanced_insights(enhanced_results)

                total_time = time.time() - start_time
                logger.info(f"✅ 3-photo analysis completed in {total_time:.3f}s")

                progress(1.0, desc="Complete!")
                return summary, detailed_json, insights
            else:
                return self._handle_api_error(response)

        except Exception as e:
            return self._handle_unexpected_error(e, start_time)

    def _handle_timeout_error(self) -> Tuple[str, str, str]:
        """Handle timeout errors with helpful messaging."""
        error_msg = (
            "⏱️ Request timed out - 3-photo analysis is taking longer than expected"
        )
        logger.error(error_msg)
        return (
            error_msg,
            '{"error": "timeout"}',
            "The analysis is taking longer than usual. This can happen with:\n"
            "• Large image files\n"
            "• High server load\n"
            "• Complex pose detection\n\n"
            "💡 Try again with:\n"
            "• Smaller image files (< 2MB each)\n"
            "• Better lighting and clearer poses\n"
            "• Wait a moment and retry",
        )

    def _handle_connection_error(self) -> Tuple[str, str, str]:
        """Handle connection errors with helpful messaging."""
        error_msg = "🔌 Cannot connect to backend service"
        logger.error(error_msg)
        return (
            error_msg,
            '{"error": "connection_failed"}',
            "Backend service is unavailable. Please:\n"
            "• Check if the server is running\n"
            "• Verify the API URL is correct\n"
            "• Check your internet connection\n"
            "• Contact support if the issue persists",
        )

    def _handle_api_error(self, response) -> Tuple[str, str, str]:
        """Handle API errors with detailed messaging."""
        try:
            error_detail = response.json().get("detail", response.text)
        except:
            error_detail = response.text

        error_msg = f"🚨 Backend error ({response.status_code}): {error_detail}"
        logger.error(error_msg)

        return (
            error_msg,
            f'{{"error": "{error_detail}", "status_code": {response.status_code}}}',
            "The backend encountered an error. Please:\n"
            "• Check that all 3 photos are clear and well-lit\n"
            "• Ensure you're wearing appropriate clothing\n"
            "• Verify your input values are reasonable\n"
            "• Try again in a moment",
        )

    def _handle_unexpected_error(
        self, error: Exception, start_time: float
    ) -> Tuple[str, str, str]:
        """Handle unexpected errors with logging."""
        error_time = time.time() - start_time
        error_msg = f"⚠️ Analysis failed: {str(error)}"
        logger.error(
            f"3-photo analysis failed in {error_time:.3f}s: {error}", exc_info=True
        )

        return (
            error_msg,
            f'{{"error": "{str(error)}", "processing_time": {error_time:.3f}}}',
            "An unexpected error occurred. Please:\n"
            "• Try again with different photos\n"
            "• Check that images are clear and well-lit\n"
            "• Ensure stable internet connection\n"
            "• Contact support if issues persist",
        )

    def _get_photo_tips(self) -> str:
        """Get helpful photo tips for users."""
        return (
            "📸 Photo Requirements & Tips:\n\n"
            "✅ **Required Photos (All 3 Needed):**\n"
            "• Front view: Face camera directly\n"
            "• Side view: Turn 90° to your right\n"
            "• Back view: Turn around completely\n\n"
            "💡 **For Best Results:**\n"
            "• Good, even lighting (avoid shadows)\n"
            "• Plain background preferred\n"
            "• Stand 4-6 feet from camera\n"
            "• Camera at chest level\n"
            "• Wear fitted clothing\n"
            "• Arms slightly away from body\n"
            "• Take all photos in same session\n"
            "• Minimum 200x200 pixels per photo\n"
            "• Keep file sizes under 5MB each"
        )

    def _format_enhanced_insights(self, results: dict) -> str:
        """Enhanced insights with personalized recommendations."""
        if not results.get("success", False):
            return self._get_photo_tips()

        body_fat = results.get("body_fat_percentage")
        category = results.get("body_fat_category", "Unknown")

        # Get user context
        user_context = results.get("user_context", {})
        sex = user_context.get("sex", "male")
        age = user_context.get("age", 0)
        weight = user_context.get("weight", 0)

        if body_fat is None:
            return self._get_photo_tips()

        # Enhanced insights with age and gender considerations
        insights = self._get_personalized_insights(body_fat, category, sex, age, weight)

        # Risk assessments
        risk_info = []
        cv_risk = results.get("cardiovascular_risk")
        sleep_risk = results.get("sleep_apnea_risk")

        if cv_risk and cv_risk != "Unknown":
            risk_info.append(f"❤️  **Cardiovascular Risk:** {cv_risk}")
        if sleep_risk and sleep_risk != "Unknown":
            risk_info.append(f"😴 **Sleep Apnea Risk:** {sleep_risk}")

        final_insights = insights
        if risk_info:
            final_insights += f"\n\n🏥 **Health Risk Assessment:**\n" + "\n".join(
                risk_info
            )

        final_insights += (
            f"\n\n📈 **3-Photo Analysis Advantage:**\n"
            f"This comprehensive multi-angle assessment provides enhanced accuracy "
            f"through complete body composition analysis including posture evaluation, "
            f"symmetry assessment, and detailed circumference measurements.\n\n"
            f"⚡ Analysis powered by MediaPipe pose detection with FastAPI backend.\n"
            f"🔬 Based on validated anthropometric equations and health guidelines."
        )

        return final_insights

    def _get_personalized_insights(
        self, body_fat: float, category: str, sex: str, age: int, weight: float
    ) -> str:
        """Generate personalized insights based on user profile."""
        # Age-specific considerations
        age_context = ""
        if age > 0:
            if age < 25:
                age_context = " Your age group typically has higher metabolic rates."
            elif age < 40:
                age_context = " This is a crucial age for establishing healthy habits."
            elif age < 60:
                age_context = (
                    " Focus on maintaining muscle mass becomes increasingly important."
                )
            else:
                age_context = (
                    " Gentle, consistent exercise and proper nutrition are key."
                )

        # Enhanced category-specific insights
        insights_map = {
            "Essential Fat": {
                "summary": f"Your {body_fat:.1f}% body fat is in the essential fat range for {sex}s.",
                "advice": f"This is very low body fat. Essential fat is necessary for basic physiological functions.{age_context}",
                "action": "⚠️ Consider consulting a healthcare provider about maintaining healthy weight. Focus on balanced nutrition with adequate healthy fats.",
            },
            "Athletes": {
                "summary": f"Outstanding! Your {body_fat:.1f}% body fat is in the athletic range.",
                "advice": f"You're in exceptional physical condition with optimal body composition.{age_context}",
                "action": "🏆 Maintain your training routine. Ensure adequate recovery, hydration, and nutrition to support your active lifestyle.",
            },
            "Fitness": {
                "summary": f"Excellent work! Your {body_fat:.1f}% body fat is in the fitness range.",
                "advice": f"You're in great shape with healthy body composition and visible muscle definition.{age_context}",
                "action": "💪 Continue your current routine. Consider adding variety to prevent plateaus and maintain motivation.",
            },
            "Average": {
                "summary": f"Your {body_fat:.1f}% body fat is in the healthy average range for {sex}s.",
                "advice": f"You're within normal healthy ranges with room for improvement if desired.{age_context}",
                "action": "🎯 Consider strength training 2-3x per week combined with balanced nutrition for improved body composition.",
            },
            "Obese": {
                "summary": f"Your {body_fat:.1f}% body fat indicates significant opportunity for health improvement.",
                "advice": f"Higher body fat levels may increase various health risks.{age_context}",
                "action": "🏥 Strongly consider consulting healthcare and nutrition professionals for a personalized, sustainable improvement plan.",
            },
        }

        insight = insights_map.get(
            category,
            {
                "summary": f"Body fat analysis: {body_fat:.1f}%",
                "advice": f"Continue monitoring your health with regular measurements.{age_context}",
                "action": "Maintain balanced nutrition and regular physical activity.",
            },
        )

        return (
            f"💡 **Personalized Health Insights:**\n\n"
            f"📋 **Assessment:** {insight['summary']}\n\n"
            f"🎯 **Context:** {insight['advice']}\n\n"
            f"🚀 **Recommendations:** {insight['action']}"
        )

    def create_interface(self) -> gr.Interface:
        """Create the enhanced production Gradio interface."""

        # Enhanced custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
        }
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .gr-form {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .photo-upload {
            border: 3px dashed #667eea;
            border-radius: 12px;
            padding: 15px;
            background: rgba(102, 126, 234, 0.05);
            transition: all 0.3s ease;
        }
        .photo-upload:hover {
            border-color: #764ba2;
            background: rgba(118, 75, 162, 0.1);
        }
        .gr-textbox textarea {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        """

        interface = gr.Interface(
            fn=self.analyze_three_photos_sync,
            inputs=[
                gr.Image(
                    type="pil",
                    label="📸 Front View Photo *Required*",
                    elem_classes=["photo-upload"],
                    height=300,
                ),
                gr.Image(
                    type="pil",
                    label="📸 Side View Photo *Required*",
                    elem_classes=["photo-upload"],
                    height=300,
                ),
                gr.Image(
                    type="pil",
                    label="📸 Back View Photo *Required*",
                    elem_classes=["photo-upload"],
                    height=300,
                ),
                gr.Number(
                    label="📏 Height (cm) *Required*",
                    value=175,
                    minimum=100,
                    maximum=250,
                    step=0.5,
                    info="Your height in centimeters (e.g., 175.5)",
                ),
                gr.Number(
                    label="⚖️ Weight (kg) - Optional",
                    value=0,
                    minimum=0,
                    maximum=500,
                    step=0.1,
                    info="Leave as 0 if you prefer not to share",
                ),
                gr.Number(
                    label="🎂 Age (years) - Optional",
                    value=0,
                    minimum=0,
                    maximum=120,
                    step=1,
                    info="Leave as 0 if you prefer not to share",
                ),
                gr.Radio(
                    ["Male", "Female"],
                    label="👤 Gender *Required*",
                    value="Male",
                    info="Required for accurate body fat calculation algorithms",
                ),
            ],
            outputs=[
                gr.Textbox(
                    label="📊 Comprehensive Body Analysis Report",
                    lines=25,
                    max_lines=30,
                    info="Your complete 9-metric body composition analysis with health insights",
                    show_copy_button=True,
                ),
                gr.Code(
                    label="🔍 Detailed Technical Results (JSON)",
                    language="json",
                    # info="Complete raw analysis data - useful for developers and detailed review",
                    # show_copy_button=True,
                ),
                gr.Textbox(
                    label="💡 Personalized Health Insights & Recommendations",
                    lines=18,
                    max_lines=25,
                    info="Tailored health guidance and risk assessment based on your 3-photo analysis",
                    show_copy_button=True,
                ),
            ],
            title="🏃‍♂️ BodyVision Pro - AI-Powered 3-Photo Body Analysis",
            description=f"""
            **Professional body composition analysis using advanced AI and computer vision!**
            
            🎯 **What You Get:** Complete body composition analysis with 9+ health metrics from just 3 photos
            
            📊 **Health Metrics Analyzed:**
            • **Body Fat Percentage** (primary metric) • **Lean Muscle Mass** • **BMI Analysis & Category**
            • **Waist-to-Hip Ratio** • **Body Surface Area** • **Cardiovascular Risk Assessment**
            • **Neck/Waist/Chest/Hip Circumferences** • **Shoulder Width & Symmetry** • **Sleep Apnea Risk**
            
            📋 **Photo Requirements (All 3 Required):**
            1. **Front View**: Face camera directly, arms slightly away from sides, good posture
            2. **Side View**: Turn exactly 90° to your right, relaxed natural stance
            3. **Back View**: Turn around completely, arms slightly away from sides
            
            💡 **Pro Tips for Accuracy:**
            • **Lighting**: Use bright, even lighting (avoid harsh shadows)
            • **Distance**: Stand 4-6 feet from camera • **Background**: Plain wall preferred
            • **Clothing**: Fitted athletic wear or swimwear • **Camera**: Keep at chest level
            • **Timing**: Take all 3 photos in the same session for consistency
            • **Quality**: Minimum 200x200 pixels, maximum 4000x4000 pixels per photo
            
            🔬 **Technology:** Powered by MediaPipe pose detection, validated anthropometric equations, and clinical health guidelines
            
            🌐 **Backend Service:** {self.api_base_url}
            """,
            examples=None,
            theme=gr.themes.Soft(),
            css=custom_css,
            allow_flagging="never",
            analytics_enabled=False,
            concurrency_limit=3,  # Limit concurrent requests
            show_progress="full",  # Show detailed progress
        )

        return interface


def create_app(api_url: Optional[str] = None):
    """Factory function to create enhanced Gradio app with 3-photo remote backend."""
    try:
        interface = ProductionGradioInterface(api_url)
        return interface.create_interface()
    except Exception as e:
        logger.error(f"Failed to create enhanced 3-photo Gradio app: {e}")
        raise
