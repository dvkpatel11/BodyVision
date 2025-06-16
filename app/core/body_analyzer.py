"""Core body analyzer that orchestrates the entire 3-photo analysis pipeline."""

from typing import Dict, Any, Optional
from PIL import Image

from app.services.analysis_service import AnalysisService
from app.services import create_analysis_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BodyAnalyzer:
    """Main class for comprehensive 3-photo body analysis operations."""

    def __init__(self, analysis_service: Optional[AnalysisService] = None):
        self.analysis_service = analysis_service or create_analysis_service()
        logger.info("BodyAnalyzer initialized for 3-photo analysis")

    async def analyze_three_photos(
        self,
        front_image: Image.Image,
        side_image: Image.Image,
        back_image: Image.Image,
        height: float = 175,
        weight: Optional[float] = None,
        age: Optional[int] = None,
        sex: str = "male",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive 3-photo body analysis.

        Args:
            front_image: Front view photo (0°)
            side_image: Side view photo (90°)
            back_image: Back view photo (180°)
            height: Height in cm
            weight: Weight in kg (optional)
            age: Age in years (optional)
            sex: 'male' or 'female'

        Returns:
            Complete analysis results with 9 health metrics
        """
        user_metadata = {"height": height, "weight": weight, "age": age, "sex": sex}

        images = {"front": front_image, "side": side_image, "back": back_image}

        logger.info(f"Starting 3-photo analysis for {sex}, {height}cm")

        return await self.analysis_service.analyze_three_photos(images, user_metadata)

    async def analyze(
        self,
        image: Image.Image,
        height: float = 182,
        weight: Optional[float] = None,
        age: Optional[int] = None,
        sex: str = "male",
    ) -> Dict[str, Any]:
        """
        Legacy single-photo analysis method.

        Args:
            image: Input image
            height: Height in cm
            weight: Weight in kg (optional)
            age: Age in years (optional)
            sex: 'male' or 'female'

        Returns:
            Basic analysis results (limited accuracy)
        """
        user_metadata = {"height": height, "weight": weight, "age": age, "sex": sex}

        logger.info(f"Starting legacy single-photo analysis for {sex}, {height}cm")

        return await self.analysis_service.analyze_body(image, user_metadata)

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results for display."""

        if "error" in results:
            return f"Analysis Error: {results['error']}"

        output_lines = []

        # Basic measurements
        if "neck_cm" in results and "waist_cm" in results:
            output_lines.append(f"Neck Circumference: {results['neck_cm']:.1f} cm")
            output_lines.append(f"Waist Circumference: {results['waist_cm']:.1f} cm")

        # Body composition
        if (
            "body_fat_percentage" in results
            and results["body_fat_percentage"] is not None
        ):
            output_lines.append(
                f"Body Fat Percentage: {results['body_fat_percentage']:.1f}%"
            )

            if "body_fat_category" in results:
                output_lines.append(f"Category: {results['body_fat_category']}")

        # Enhanced metrics
        if "lean_muscle_mass_kg" in results and results["lean_muscle_mass_kg"]:
            output_lines.append(
                f"Lean Muscle Mass: {results['lean_muscle_mass_kg']:.1f} kg"
            )

        if "bmi" in results and results["bmi"]:
            output_lines.append(
                f"BMI: {results['bmi']:.1f} ({results.get('bmi_category', 'Unknown')})"
            )

        if "waist_to_hip_ratio" in results and results["waist_to_hip_ratio"]:
            output_lines.append(
                f"Waist-to-Hip Ratio: {results['waist_to_hip_ratio']:.3f}"
            )

        # Analysis quality
        if "processing_time_seconds" in results:
            output_lines.append(
                f"Processing Time: {results['processing_time_seconds']:.2f}s"
            )

        return "\n".join(output_lines) if output_lines else "No measurements available"

    def get_health_insights(self, results: Dict[str, Any]) -> str:
        """Generate health insights from analysis results."""

        if not results.get("success", True) or "error" in results:
            return "Unable to generate health insights due to analysis error."

        insights = []

        # Body fat insights
        body_fat = results.get("body_fat_percentage")
        category = results.get("body_fat_category")

        if body_fat and category:
            insights.append(
                f"Your {body_fat}% body fat places you in the '{category}' category."
            )

        # Risk assessments
        cv_risk = results.get("cardiovascular_risk")
        if cv_risk:
            insights.append(f"Cardiovascular risk level: {cv_risk}")

        sleep_risk = results.get("sleep_apnea_risk")
        if sleep_risk:
            insights.append(f"Sleep apnea risk: {sleep_risk}")

        # BMI insights
        bmi = results.get("bmi")
        bmi_category = results.get("bmi_category")
        if bmi and bmi_category:
            insights.append(f"BMI of {bmi} indicates {bmi_category} weight status.")

        return "\n".join(insights) if insights else "Analysis completed successfully."
