#!/bin/bash

# =========================================================
# Gradio Interface Enhancement Script
# Adds photo positioning guides for optimal user input
# =========================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🎨 Gradio Interface Enhancement Script${NC}"
echo -e "${BLUE}====================================${NC}"
echo -e "${YELLOW}Adding photo positioning guides for optimal user input${NC}"
echo ""

# Create backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups/gradio_enhancement_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

backup_file() {
    local file=$1
    if [ -f "$file" ]; then
        local backup_path="$BACKUP_DIR/${file//\//_}"
        cp "$file" "$backup_path"
        echo -e "${GREEN}✅ Backed up: $file${NC}"
    fi
}

echo -e "${BLUE}📦 Updating Gradio Interface with Photo Guides${NC}"

backup_file "app/api/gradio_interface.py"

cat > "app/api/gradio_interface.py" << 'EOF'
"""Enhanced Gradio interface with photo alignment guides for optimal user input."""

import gradio as gr
import requests
import time
import json
import io
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Dict
import numpy as np

from app.utils.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class EnhancedGradioInterface:
    """Enhanced Gradio interface with photo guides for optimal 3-photo analysis."""

    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize enhanced Gradio interface with photo guides."""
        try:
            self.api_base_url = api_base_url or f"http://localhost:{settings.PORT}"
            self._test_backend_connection()
            logger.info(
                f"✅ Enhanced Gradio interface initialized with backend: {self.api_base_url}"
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

    def _create_photo_guide_overlay(self, width: int = 400, height: int = 600, guide_type: str = "front") -> str:
        """Create a base64-encoded overlay image showing proper photo positioning guide."""
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Colors
        guide_color = (34, 197, 94, 100)  # Semi-transparent green
        outline_color = (34, 197, 94, 255)  # Solid green outline
        text_color = (255, 255, 255, 255)  # White text
        
        if guide_type == "front":
            self._draw_front_guide(draw, width, height, guide_color, outline_color)
            
        elif guide_type == "side":
            self._draw_side_guide(draw, width, height, guide_color, outline_color)
            
        elif guide_type == "back":
            self._draw_back_guide(draw, width, height, guide_color, outline_color)
        
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        overlay.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"

    def _draw_front_guide(self, draw, width, height, fill_color, outline_color):
        """Draw front view positioning guide."""
        center_x = width // 2
        
        # Head circle
        head_radius = width // 10
        head_y = height // 8
        draw.ellipse([
            center_x - head_radius, head_y - head_radius,
            center_x + head_radius, head_y + head_radius
        ], outline=outline_color, width=2)
        
        # Neck rectangle
        neck_width = width // 15
        neck_y = head_y + head_radius
        neck_height = height // 12
        draw.rectangle([
            center_x - neck_width, neck_y,
            center_x + neck_width, neck_y + neck_height
        ], outline=outline_color, width=2)
        
        # Shoulders line
        shoulder_width = width // 3
        shoulder_y = neck_y + neck_height
        draw.line([
            center_x - shoulder_width, shoulder_y,
            center_x + shoulder_width, shoulder_y
        ], fill=outline_color, width=3)
        
        # Torso rectangle (chest to waist area)
        torso_width = width // 4
        torso_height = height // 2.5
        torso_y = shoulder_y + 10
        draw.rectangle([
            center_x - torso_width, torso_y,
            center_x + torso_width, torso_y + torso_height
        ], outline=outline_color, width=2, fill=fill_color)
        
        # Arms (slightly away from body)
        arm_offset = width // 8
        arm_length = height // 3
        arm_width = width // 20
        
        # Left arm
        draw.rectangle([
            center_x - shoulder_width - arm_offset, shoulder_y,
            center_x - shoulder_width - arm_offset + arm_width, shoulder_y + arm_length
        ], outline=outline_color, width=2)
        
        # Right arm  
        draw.rectangle([
            center_x + shoulder_width + arm_offset - arm_width, shoulder_y,
            center_x + shoulder_width + arm_offset, shoulder_y + arm_length
        ], outline=outline_color, width=2)
        
        # Legs
        leg_width = width // 8
        leg_start_y = torso_y + torso_height
        leg_height = height - leg_start_y - 20
        
        # Left leg
        draw.rectangle([
            center_x - leg_width, leg_start_y,
            center_x - 5, leg_start_y + leg_height
        ], outline=outline_color, width=2)
        
        # Right leg
        draw.rectangle([
            center_x + 5, leg_start_y,
            center_x + leg_width, leg_start_y + leg_height
        ], outline=outline_color, width=2)

    def _draw_side_guide(self, draw, width, height, fill_color, outline_color):
        """Draw side view positioning guide."""
        
        # Head profile (circle)
        head_radius = width // 12
        head_x = width // 2
        head_y = height // 8
        draw.ellipse([
            head_x - head_radius, head_y - head_radius,
            head_x + head_radius, head_y + head_radius
        ], outline=outline_color, width=2)
        
        # Neck line
        neck_y = head_y + head_radius
        neck_height = height // 12
        draw.line([
            head_x, neck_y,
            head_x, neck_y + neck_height
        ], fill=outline_color, width=4)
        
        # Torso profile (curved for natural body shape)
        torso_points = [
            (head_x - width//8, neck_y + neck_height),  # Back shoulder
            (head_x + width//12, neck_y + neck_height + 10),  # Front shoulder
            (head_x + width//10, neck_y + height//3),  # Chest
            (head_x + width//15, neck_y + height//2),  # Waist
            (head_x + width//12, neck_y + height//1.5),  # Hip
            (head_x - width//20, neck_y + height//1.2),  # Back hip
            (head_x - width//8, neck_y + neck_height)  # Close shape
        ]
        draw.polygon(torso_points, outline=outline_color, width=2, fill=fill_color)
        
        # Arm (relaxed at side)
        arm_x = head_x + width//12
        arm_y = neck_y + neck_height + 20
        arm_length = height // 3
        draw.line([
            arm_x, arm_y,
            arm_x + width//20, arm_y + arm_length
        ], fill=outline_color, width=3)
        
        # Legs
        leg_start_y = neck_y + height//1.5
        leg_height = height - leg_start_y - 20
        
        # Front leg
        draw.line([
            head_x + width//20, leg_start_y,
            head_x + width//20, leg_start_y + leg_height
        ], fill=outline_color, width=4)
        
        # Back leg (slightly behind)
        draw.line([
            head_x - width//30, leg_start_y,
            head_x - width//30, leg_start_y + leg_height
        ], fill=outline_color, width=3)

    def _draw_back_guide(self, draw, width, height, fill_color, outline_color):
        """Draw back view positioning guide."""
        center_x = width // 2
        
        # Head circle (back of head)
        head_radius = width // 10
        head_y = height // 8
        draw.ellipse([
            center_x - head_radius, head_y - head_radius,
            center_x + head_radius, head_y + head_radius
        ], outline=outline_color, width=2)
        
        # Neck
        neck_width = width // 15
        neck_y = head_y + head_radius
        neck_height = height // 12
        draw.rectangle([
            center_x - neck_width, neck_y,
            center_x + neck_width, neck_y + neck_height
        ], outline=outline_color, width=2)
        
        # Shoulders (wider for back view)
        shoulder_width = width // 2.8
        shoulder_y = neck_y + neck_height
        draw.line([
            center_x - shoulder_width, shoulder_y,
            center_x + shoulder_width, shoulder_y
        ], fill=outline_color, width=4)
        
        # Back torso (trapezoid shape)
        back_points = [
            (center_x - shoulder_width, shoulder_y),
            (center_x + shoulder_width, shoulder_y),
            (center_x + width//5, shoulder_y + height//2),
            (center_x - width//5, shoulder_y + height//2)
        ]
        draw.polygon(back_points, outline=outline_color, width=2, fill=fill_color)
        
        # Arms (slightly away from body)
        arm_offset = width // 8
        arm_length = height // 3
        arm_width = width // 20
        
        # Left arm
        draw.rectangle([
            center_x - shoulder_width - arm_offset, shoulder_y,
            center_x - shoulder_width - arm_offset + arm_width, shoulder_y + arm_length
        ], outline=outline_color, width=2)
        
        # Right arm
        draw.rectangle([
            center_x + shoulder_width + arm_offset - arm_width, shoulder_y,
            center_x + shoulder_width + arm_offset, shoulder_y + arm_length
        ], outline=outline_color, width=2)
        
        # Spine line
        spine_start_y = neck_y + neck_height
        spine_end_y = shoulder_y + height//2
        draw.line([
            center_x, spine_start_y,
            center_x, spine_end_y
        ], fill=outline_color, width=2)

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
                    "Please upload all 3 required photos following the positioning guides shown above each upload area.",
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
                    "Backend service encountered an error. Please try again with properly positioned photos.",
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
                "Please try again with clear, well-positioned photos following the guides.",
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
            lines.append("   Please ensure all 3 photos follow the positioning guides")

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
            return "Upload 3 properly positioned photos for comprehensive health insights."

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
                "• Follow the positioning guides shown above each upload\n"
                "• Ensure all 3 views show clear body outlines\n"
                "• Use consistent, good lighting for all photos\n"
                "• Wear fitted clothing that shows body contours\n"
                "• Stand 4-6 feet from camera for all shots\n"
                "• Keep camera at chest level for all views\n"
                "• Take photos in sequence: Front → Side → Back"
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
        """Create the enhanced Gradio interface with photo guides."""

        # Enhanced CSS with photo guide styling
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
        .photo-guide-container {
            position: relative;
            border: 3px solid #22c55e;
            border-radius: 12px;
            padding: 15px;
            background: linear-gradient(145deg, #f0fdf4, #dcfce7);
            margin: 10px 0;
        }
        .photo-guide-header {
            background: #22c55e;
            color: white;
            padding: 8px 15px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 10px;
            text-align: center;
        }
        .positioning-tips {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
            border-radius: 4px;
        }
        """

        # Create photo guide overlays
        front_guide = self._create_photo_guide_overlay(400, 600, "front")
        side_guide = self._create_photo_guide_overlay(400, 600, "side")
        back_guide = self._create_photo_guide_overlay(400, 600, "back")

        interface = gr.Interface(
            fn=self.analyze_three_photos_sync,
            inputs=[
                gr.HTML(
                    value=f"""
                    <div class="photo-guide-container">
                        <div class="photo-guide-header">📸 FRONT VIEW - Position Guide</div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="flex-shrink: 0;">
                                <img src="{front_guide}" width="150" height="225" style="border: 2px solid #22c55e; border-radius: 8px;"/>
                            </div>
                            <div class="positioning-tips">
                                <strong>✅ Correct Position:</strong><br>
                                • Face camera directly<br>
                                • Arms slightly away from sides<br>
                                • Stand straight, feet shoulder-width apart<br>
                                • Wear fitted clothing<br>
                                • 4-6 feet from camera<br>
                                <strong>📏 Key Areas:</strong> Neck, waist, shoulders visible
                            </div>
                        </div>
                    </div>
                    """,
                    visible=True
                ),
                gr.Image(
                    type="pil",
                    label="📸 Upload Front View Photo",
                    elem_classes=["photo-guide-container"],
                ),
                
                gr.HTML(
                    value=f"""
                    <div class="photo-guide-container">
                        <div class="photo-guide-header">📸 SIDE VIEW - Position Guide</div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="flex-shrink: 0;">
                                <img src="{side_guide}" width="150" height="225" style="border: 2px solid #22c55e; border-radius: 8px;"/>
                            </div>
                            <div class="positioning-tips">
                                <strong>✅ Correct Position:</strong><br>
                                • Turn 90° to your right<br>
                                • Arms relaxed at sides<br>
                                • Natural standing posture<br>
                                • Look straight ahead (not at camera)<br>
                                • Same distance as front photo<br>
                                <strong>📏 Key Areas:</strong> Profile, posture, body depth
                            </div>
                        </div>
                    </div>
                    """,
                    visible=True
                ),
                gr.Image(
                    type="pil",
                    label="📸 Upload Side View Photo",
                    elem_classes=["photo-guide-container"],
                ),
                
                gr.HTML(
                    value=f"""
                    <div class="photo-guide-container">
                        <div class="photo-guide-header">📸 BACK VIEW - Position Guide</div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div style="flex-shrink: 0;">
                                <img src="{back_guide}" width="150" height="225" style="border: 2px solid #22c55e; border-radius: 8px;"/>
                            </div>
                            <div class="positioning-tips">
                                <strong>✅ Correct Position:</strong><br>
                                • Turn completely around (180°)<br>
                                • Arms slightly away from sides<br>
                                • Same posture as front view<br>
                                • Don't look back at camera<br>
                                • Keep same distance<br>
                                <strong>📏 Key Areas:</strong> Shoulders, spine, symmetry
                            </div>
                        </div>
                    </div>
                    """,
                    visible=True
                ),
                gr.Image(
                    type="pil",
                    label="📸 Upload Back View Photo",
                    elem_classes=["photo-guide-container"],
                ),
                
                gr.HTML(
                    value="""
                    <div style="background: #eff6ff; border: 2px solid #3b82f6; border-radius: 8px; padding: 15px; margin: 15px 0;">
                        <h4 style="color: #1e40af; margin: 0 0 10px 0;">📋 User Information</h4>
                        <p style="margin: 5px 0; font-size: 14px;">Please provide your details for accurate body composition analysis:</p>
                    </div>
                    """,
                    visible=True
                ),
                
                gr.Number(
                    label="📏 Height (cm)",
                    value=175,
                    minimum=100,
                    maximum=250,
                    info="Your height in centimeters - Required for accurate measurements",
                ),
                gr.Number(
                    label="⚖️ Weight (kg) - Optional",
                    value=0,
                    minimum=0,
                    maximum=300,
                    info="Leave as 0 if you prefer not to share - enhances BMI and lean mass calculations",
                ),
                gr.Number(
                    label="🎂 Age (years) - Optional",
                    value=0,
                    minimum=0,
                    maximum=120,
                    info="Leave as 0 if you prefer not to share - helps with health risk assessment",
                ),
                gr.Radio(
                    ["Male", "Female"],
                    label="👤 Gender",
                    value="Male",
                    info="Required for accurate body fat calculation using Navy formula",
                ),
            ],
            outputs=[
                gr.Textbox(
                    label="📊 Comprehensive Analysis Summary",
                    lines=20,
                    max_lines=25,
                    info="Your complete 9-metric body composition analysis based on 3-photo positioning",
                ),
                gr.Code(
                    label="🔍 Detailed Results (JSON)",
                    language="json",
                    info="Complete technical analysis data",
                ),
                gr.Textbox(
                    label="💡 Personalized Health Insights",
                    lines=15,
                    max_lines=20,
                    info="Health guidance and risk assessment based on your 3-photo analysis",
                ),
            ],
            title="🏃‍♂️ BodyVision - Guided 3-Photo AI Body Analysis",
            description=f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 10px 0;">🎯 Professional Body Composition Analysis</h2>
                <p style="margin: 0; font-size: 16px;">Follow the positioning guides above each photo upload for maximum accuracy</p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 8px; padding: 15px;">
                    <h4 style="color: #15803d; margin: 0 0 10px 0;">✅ What You'll Get:</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        <li>Body Fat Percentage (Navy Formula)</li>
                        <li>Lean Muscle Mass Estimation</li>
                        <li>BMI with Composition Context</li>
                        <li>Waist-to-Hip Ratio Analysis</li>
                        <li>Cardiovascular Risk Assessment</li>
                        <li>Sleep Apnea Risk Indicators</li>
                        <li>Shoulder Symmetry Analysis</li>
                        <li>Posture Assessment</li>
                        <li>Body Surface Area Calculation</li>
                    </ul>
                </div>
                
                <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px;">
                    <h4 style="color: #92400e; margin: 0 0 10px 0;">📸 Photo Quality Tips:</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                        <li>Use natural lighting when possible</li>
                        <li>Plain background preferred</li>
                        <li>Wear fitted athletic clothing</li>
                        <li>Take all photos in same session</li>
                        <li>Keep camera steady (use timer if alone)</li>
                        <li>Maintain same distance for all 3 views</li>
                        <li>Follow the green positioning guides</li>
                    </ul>
                </div>
            </div>
            
            <div style="background: #e0f2fe; border: 2px solid #0284c7; border-radius: 8px; padding: 15px; margin: 20px 0;">
                <h4 style="color: #0369a1; margin: 0 0 10px 0;">🎯 How It Works:</h4>
                <ol style="margin: 0; padding-left: 20px; font-size: 14px;">
                    <li><strong>Upload 3 Photos:</strong> Follow the positioning guides shown above each upload area</li>
                    <li><strong>MediaPipe Detection:</strong> Advanced AI detects body landmarks in all 3 views</li>
                    <li><strong>Multi-Angle Analysis:</strong> Combines measurements from front, side, and back views</li>
                    <li><strong>Navy Formula Calculation:</strong> Applies proven mathematical models for body fat</li>
                    <li><strong>Enhanced Metrics:</strong> Calculates 9 comprehensive health indicators</li>
                    <li><strong>Personalized Insights:</strong> Provides health guidance and recommendations</li>
                </ol>
            </div>
            
            🌐 **Powered by:** MediaPipe Pose Detection + US Navy Body Fat Formula + FastAPI Backend at {self.api_base_url}
            """,
            examples=[],
            theme=gr.themes.Soft(),
            css=custom_css,
            allow_flagging="never",
            analytics_enabled=False,
        )

        return interface


def create_app(api_url: Optional[str] = None):
    """Factory function to create enhanced Gradio app with photo guides."""
    try:
        interface = EnhancedGradioInterface(api_url)
        return interface.create_interface()
    except Exception as e:
        logger.error(f"Failed to create enhanced Gradio app: {e}")
        raise

# Update the ProductionGradioInterface to use the enhanced version
ProductionGradioInterface = EnhancedGradioInterface
EOF

echo -e "${GREEN}✅ Enhanced Gradio interface created with photo positioning guides${NC}"

echo ""
echo -e "${GREEN}🎉 Gradio Interface Enhancement Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

echo -e "${BLUE}📋 ENHANCEMENTS ADDED:${NC}"
echo -e "✅ 1. Photo Positioning Guides - ${GREEN}Visual outlines for proper positioning${NC}"
echo -e "✅ 2. Interactive Guide Images - ${GREEN}Base64-encoded overlay guides${NC}"
echo -e "✅ 3. Detailed Instructions - ${GREEN}Step-by-step positioning tips${NC}"
echo -e "✅ 4. Enhanced UI Styling - ${GREEN}Professional appearance with guides${NC}"
echo -e "✅ 5. Better Error Messages - ${GREEN}References to positioning guides${NC}"

echo ""
echo -e "${BLUE}🎯 NEW FEATURES:${NC}"
echo -e "• ${GREEN}Visual Body Outlines${NC} - Green guides showing correct positioning"
echo -e "• ${GREEN}Position Instructions${NC} - Detailed tips for each photo type"
echo -e "• ${GREEN}Quality Guidelines${NC} - Best practices for photo capture"
echo -e "• ${GREEN}Enhanced Layout${NC} - Professional styling with color-coded sections"
echo -e "• ${GREEN}Improved Feedback${NC} - Better error messages referencing guides"

echo ""
echo -e "${BLUE}🚀 NEXT STEPS:${NC}"
echo -e "1. ${YELLOW}Restart your Gradio interface: python start_gradio.py${NC}"
echo -e "2. ${YELLOW}Visit: http://localhost:7860${NC}"
echo -e "3. ${YELLOW}Test the new photo guides and improved user experience${NC}"

echo ""
echo -e "${BLUE}💾 BACKUP:${NC}"
echo -e "Original file backed up to: ${YELLOW}${BACKUP_DIR}/app_api_gradio_interface.py${NC}"

echo ""
echo -e "${YELLOW}🎉 Users will now see clear positioning guides for optimal photo capture!${NC}"
echo -e "${YELLOW}📸 The green outlines will help ensure proper MediaPipe detection.${NC}"