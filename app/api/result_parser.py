"""Enhanced result parsing for BodyVision API responses with validation and error handling."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BodyVisionResultParser:
    """Enhanced parser for BodyVision API results with comprehensive validation."""

    # Expected ranges for validation
    BODY_FAT_RANGES = {
        "male": {
            "Essential Fat": (2, 5),
            "Athletes": (6, 13),
            "Fitness": (14, 17),
            "Average": (18, 24),
            "Obese": (25, 100),
        },
        "female": {
            "Essential Fat": (10, 13),
            "Athletes": (14, 20),
            "Fitness": (21, 24),
            "Average": (25, 31),
            "Obese": (32, 100),
        },
    }

    MEASUREMENT_RANGES = {
        "neck_cm": (25, 60),
        "waist_cm": (50, 150),
        "chest_cm": (70, 150),
        "hip_cm": (70, 150),
        "shoulder_width_cm": (30, 80),
    }

    def __init__(self):
        """Initialize the enhanced result parser."""
        pass

    def parse_and_validate_results(
        self, api_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse and validate API response with comprehensive error checking.

        Args:
            api_response: Raw API response dictionary

        Returns:
            Enhanced and validated results dictionary
        """
        try:
            # Start with the original response
            enhanced_results = api_response.copy()

            # Validate basic structure
            if not self._validate_response_structure(api_response):
                return self._create_error_response("Invalid response structure")

            # Extract and validate core data
            success = api_response.get("success", False)
            if not success:
                return self._create_error_response(
                    api_response.get("error", "Analysis failed")
                )

            # Parse metadata
            metadata = api_response.get("metadata", {})
            sex = metadata.get("input_sex", "male").lower()
            height = metadata.get("input_height", 0)
            weight = metadata.get("input_weight")
            age = metadata.get("input_age")

            # Parse body composition
            body_comp = api_response.get("body_composition", {})
            body_fat = body_comp.get("body_fat_percentage")
            category = body_comp.get("body_fat_category", "Unknown")

            # Parse measurements
            measurements = api_response.get("measurements", {})
            neck_cm = body_comp.get("neck_cm", 0)
            waist_cm = body_comp.get("waist_cm", 0)

            # Validate critical measurements
            validation_issues = self._validate_measurements(
                body_fat, neck_cm, waist_cm, sex
            )

            # Calculate missing metrics
            enhanced_results.update(
                self._calculate_additional_metrics(
                    height, weight, age, sex, neck_cm, waist_cm, body_fat
                )
            )

            # Add validation results
            enhanced_results["validation"] = {
                "has_issues": len(validation_issues) > 0,
                "issues": validation_issues,
                "data_quality_score": self._calculate_quality_score(api_response),
                "recommendations": self._get_validation_recommendations(
                    validation_issues
                ),
            }

            # Enhance with user context
            enhanced_results["user_context"] = {
                "sex": sex,
                "age": age,
                "weight": weight,
                "height": height,
                "analysis_timestamp": api_response.get("analysis_timestamp"),
                "analysis_id": api_response.get("analysis_id"),
            }

            # Add comprehensive health insights
            enhanced_results.update(
                self._generate_health_insights(
                    body_fat, category, sex, age, weight, neck_cm, waist_cm
                )
            )

            logger.info(
                f"Successfully parsed and enhanced results for analysis {api_response.get('analysis_id', 'unknown')}"
            )
            return enhanced_results

        except Exception as e:
            logger.error(f"Error parsing API response: {e}", exc_info=True)
            return self._create_error_response(f"Parsing error: {str(e)}")

    def _validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """Validate that response has required structure."""
        required_fields = ["success", "analysis_id"]

        for field in required_fields:
            if field not in response:
                logger.warning(f"Missing required field: {field}")
                return False

        return True

    def _validate_measurements(
        self, body_fat: Optional[float], neck_cm: float, waist_cm: float, sex: str
    ) -> list:
        """Validate measurements against expected ranges."""
        issues = []

        # Validate body fat percentage
        if body_fat is not None:
            if body_fat < 1 or body_fat > 60:
                issues.append(f"Body fat {body_fat}% is outside normal range (1-60%)")

            # Check against sex-specific ranges
            if sex in self.BODY_FAT_RANGES:
                all_ranges = []
                for category, (min_val, max_val) in self.BODY_FAT_RANGES[sex].items():
                    all_ranges.extend([min_val, max_val])
                overall_min, overall_max = min(all_ranges), max(all_ranges)

                if body_fat < overall_min or body_fat > overall_max:
                    issues.append(
                        f"Body fat {body_fat}% is unusual for {sex} (expected {overall_min}-{overall_max}%)"
                    )

        # Validate neck circumference
        if (
            neck_cm < self.MEASUREMENT_RANGES["neck_cm"][0]
            or neck_cm > self.MEASUREMENT_RANGES["neck_cm"][1]
        ):
            issues.append(
                f"Neck measurement {neck_cm}cm is outside normal range {self.MEASUREMENT_RANGES['neck_cm']}"
            )

        # Validate waist circumference
        if (
            waist_cm < self.MEASUREMENT_RANGES["waist_cm"][0]
            or waist_cm > self.MEASUREMENT_RANGES["waist_cm"][1]
        ):
            issues.append(
                f"Waist measurement {waist_cm}cm is outside normal range {self.MEASUREMENT_RANGES['waist_cm']}"
            )

        # Check logical relationships
        if neck_cm > 0 and waist_cm > 0:
            if neck_cm >= waist_cm:
                issues.append(
                    "Neck circumference should be smaller than waist circumference"
                )

        return issues

    def _calculate_additional_metrics(
        self,
        height: float,
        weight: Optional[float],
        age: Optional[int],
        sex: str,
        neck_cm: float,
        waist_cm: float,
        body_fat: Optional[float],
    ) -> Dict[str, Any]:
        """Calculate additional health metrics."""
        additional_metrics = {}

        try:
            # Calculate BMI if weight is available
            if weight and height > 0:
                height_m = height / 100
                bmi = weight / (height_m**2)
                additional_metrics["bmi"] = round(bmi, 1)
                additional_metrics["bmi_category"] = self._get_bmi_category(bmi)

            # Calculate lean muscle mass if body fat is available
            if body_fat is not None and weight:
                lean_mass = weight * (1 - body_fat / 100)
                additional_metrics["lean_muscle_mass_kg"] = round(lean_mass, 1)

            # Calculate body surface area (Mosteller formula)
            if weight and height > 0:
                bsa = ((weight * height) / 3600) ** 0.5
                additional_metrics["body_surface_area_m2"] = round(bsa, 2)

            # Estimate waist-to-hip ratio (assuming hip is ~1.1-1.2x waist for estimation)
            if waist_cm > 0:
                # This is an estimation - ideally we'd have actual hip measurement
                estimated_hip = waist_cm * 1.15  # Rough estimation
                whr = waist_cm / estimated_hip
                additional_metrics["waist_to_hip_ratio_estimated"] = round(whr, 3)
                additional_metrics["hip_cm_estimated"] = round(estimated_hip, 1)

            # Calculate neck-to-waist ratio
            if neck_cm > 0 and waist_cm > 0:
                nwr = neck_cm / waist_cm
                additional_metrics["neck_to_waist_ratio"] = round(nwr, 3)

        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")

        return additional_metrics

    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category based on WHO standards."""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    def _calculate_quality_score(self, response: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 0.0
        max_score = 0.0

        # Check metadata quality
        metadata = response.get("metadata", {})

        # Processing quality indicators
        if metadata.get("posture_score"):
            score += metadata["posture_score"] / 100
            max_score += 1.0

        if metadata.get("symmetry_score"):
            score += metadata["symmetry_score"] / 100
            max_score += 1.0

        if metadata.get("measurement_quality"):
            score += metadata["measurement_quality"]
            max_score += 1.0

        if metadata.get("multi_angle_validation"):
            score += metadata["multi_angle_validation"]
            max_score += 1.0

        # Image quality indicators
        if metadata.get("photos_processed", 0) == 3:
            score += 1.0
            max_score += 1.0

        # Confidence score if available
        detections = response.get("detections", {})
        confidence = detections.get("confidence_score")
        if confidence is not None:
            score += confidence
            max_score += 1.0

        return round(score / max_score if max_score > 0 else 0.5, 2)

    def _get_validation_recommendations(self, issues: list) -> list:
        """Get recommendations based on validation issues."""
        if not issues:
            return ["All measurements appear within normal ranges"]

        recommendations = []

        for issue in issues:
            if "body fat" in issue.lower():
                recommendations.append(
                    "Retake photos with better lighting and clearer pose"
                )
            elif "neck" in issue.lower():
                recommendations.append(
                    "Ensure neck is clearly visible in front view photo"
                )
            elif "waist" in issue.lower():
                recommendations.append(
                    "Check that waist area is clearly visible in all photos"
                )
            elif "circumference" in issue.lower():
                recommendations.append("Verify photos show clear body contours")

        if not recommendations:
            recommendations.append(
                "Review photo quality and retake if measurements seem incorrect"
            )

        return recommendations

    def _generate_health_insights(
        self,
        body_fat: Optional[float],
        category: str,
        sex: str,
        age: Optional[int],
        weight: Optional[float],
        neck_cm: float,
        waist_cm: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive health insights."""
        insights = {}

        # Cardiovascular risk assessment
        cv_risk = self._assess_cardiovascular_risk(body_fat, waist_cm, sex, age)
        insights["cardiovascular_risk"] = cv_risk

        # Sleep apnea risk (based on neck circumference)
        sleep_risk = self._assess_sleep_apnea_risk(neck_cm, waist_cm, sex)
        insights["sleep_apnea_risk"] = sleep_risk

        # Metabolic health indicators
        metabolic_health = self._assess_metabolic_health(body_fat, waist_cm, sex, age)
        insights["metabolic_health"] = metabolic_health

        return insights

    def _assess_cardiovascular_risk(
        self, body_fat: Optional[float], waist_cm: float, sex: str, age: Optional[int]
    ) -> str:
        """Assess cardiovascular risk based on measurements."""
        risk_factors = 0

        # Waist circumference risk (ATP III guidelines)
        if sex == "male" and waist_cm > 102:
            risk_factors += 2
        elif sex == "female" and waist_cm > 88:
            risk_factors += 2
        elif sex == "male" and waist_cm > 94:
            risk_factors += 1
        elif sex == "female" and waist_cm > 80:
            risk_factors += 1

        # Body fat risk
        if body_fat is not None:
            if sex == "male" and body_fat > 25:
                risk_factors += 1
            elif sex == "female" and body_fat > 32:
                risk_factors += 1

        # Age factor
        if age and ((sex == "male" and age > 45) or (sex == "female" and age > 55)):
            risk_factors += 1

        if risk_factors >= 3:
            return "Elevated"
        elif risk_factors >= 1:
            return "Moderate"
        else:
            return "Low"

    def _assess_sleep_apnea_risk(
        self, neck_cm: float, waist_cm: float, sex: str
    ) -> str:
        """Assess sleep apnea risk based on neck circumference."""
        risk_factors = 0

        # Neck circumference thresholds
        if sex == "male" and neck_cm > 43:
            risk_factors += 2
        elif sex == "female" and neck_cm > 41:
            risk_factors += 2
        elif sex == "male" and neck_cm > 40:
            risk_factors += 1
        elif sex == "female" and neck_cm > 38:
            risk_factors += 1

        # Waist as additional factor
        if waist_cm > 100:
            risk_factors += 1

        if risk_factors >= 2:
            return "High"
        elif risk_factors >= 1:
            return "Moderate"
        else:
            return "Low"

    def _assess_metabolic_health(
        self, body_fat: Optional[float], waist_cm: float, sex: str, age: Optional[int]
    ) -> str:
        """Assess overall metabolic health."""
        health_factors = 0
        total_factors = 0

        # Body fat assessment
        if body_fat is not None:
            total_factors += 1
            if sex == "male":
                if 6 <= body_fat <= 24:
                    health_factors += 1
            else:  # female
                if 14 <= body_fat <= 31:
                    health_factors += 1

        # Waist circumference
        total_factors += 1
        if sex == "male" and waist_cm <= 94:
            health_factors += 1
        elif sex == "female" and waist_cm <= 80:
            health_factors += 1

        if total_factors == 0:
            return "Insufficient data"

        health_ratio = health_factors / total_factors

        if health_ratio >= 0.8:
            return "Good"
        elif health_ratio >= 0.5:
            return "Fair"
        else:
            return "Needs attention"

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_message,
            "analysis_id": None,
            "validation": {
                "has_issues": True,
                "issues": [error_message],
                "data_quality_score": 0.0,
                "recommendations": ["Please try again with new photos"],
            },
        }

    def format_enhanced_summary(self, results: Dict[str, Any]) -> str:
        """Format the enhanced results into a comprehensive summary."""
        if not results.get("success", False):
            error_msg = results.get("error", "Unknown error occurred")
            return f"❌ Analysis failed: {error_msg}"

        # Extract all available metrics
        body_fat = results.get("body_fat_percentage")
        category = results.get("body_fat_category", "Unknown")

        # Body composition metrics
        lean_mass = results.get("lean_muscle_mass_kg")
        bmi = results.get("bmi")
        bmi_category = results.get("bmi_category", "Unknown")

        # Circumference measurements
        neck_cm = results.get("neck_cm", 0)
        waist_cm = results.get("waist_cm", 0)
        chest_cm = results.get("chest_cm", 0)
        hip_cm = results.get("hip_cm_estimated", 0)

        # Health ratios
        whr = results.get("waist_to_hip_ratio_estimated")
        nwr = results.get("neck_to_waist_ratio")
        bsa = results.get("body_surface_area_m2")

        # Quality and validation
        validation = results.get("validation", {})
        quality_score = validation.get("data_quality_score", 0)
        has_issues = validation.get("has_issues", False)
        issues = validation.get("issues", [])

        # Metadata
        metadata = results.get("metadata", {})
        processing_time = results.get("processing_time_seconds", 0)
        analysis_id = results.get("analysis_id", "N/A")
        photos_processed = metadata.get("photos_processed", 0)
        posture_score = metadata.get("posture_score", 0)
        symmetry_score = metadata.get("symmetry_score", 0)

        # Build comprehensive summary
        lines = ["🎯 BodyVision - Enhanced 3-Photo Analysis Results", "=" * 55, ""]

        # Validation warnings first
        if has_issues:
            lines.extend(
                [
                    "⚠️  **Data Quality Alert:**",
                    *[f"   • {issue}" for issue in issues[:3]],  # Show top 3 issues
                    "",
                ]
            )

        if body_fat is not None:
            lines.extend(
                [
                    f"🔥 **Body Fat Percentage:** {body_fat:.1f}%",
                    f"🏷️  **Health Category:** {category}",
                    "",
                ]
            )

            # Additional metrics
            if lean_mass:
                lines.append(f"💪 **Lean Muscle Mass:** {lean_mass:.1f} kg")
            if bmi:
                lines.append(f"📊 **BMI:** {bmi:.1f} ({bmi_category})")
            if bsa:
                lines.append(f"🧬 **Body Surface Area:** {bsa:.2f} m²")

            lines.append("")

            # Health ratios
            lines.append("📐 **Health Ratios:**")
            if whr:
                lines.append(f"   • Waist-to-Hip Ratio: {whr:.3f}")
            if nwr:
                lines.append(f"   • Neck-to-Waist Ratio: {nwr:.3f}")

            lines.append("")

            # Measurements
            lines.append("📏 **Body Measurements:**")
            if neck_cm > 0:
                lines.append(f"   • Neck: {neck_cm:.1f} cm")
            if waist_cm > 0:
                lines.append(f"   • Waist: {waist_cm:.1f} cm")
            if chest_cm > 0:
                lines.append(f"   • Chest: {chest_cm:.1f} cm")
            if hip_cm > 0:
                lines.append(f"   • Hip (estimated): {hip_cm:.1f} cm")
        else:
            lines.extend(
                [
                    "⚠️  **Could not calculate body fat percentage**",
                    "   Analysis may need improvement - see validation issues above",
                ]
            )

        # Quality assessment
        lines.extend(
            [
                "",
                "🔍 **Analysis Quality Assessment:**",
                f"   • Overall Data Quality: {quality_score:.0%}",
                f"   • Photos Successfully Processed: {photos_processed}/3",
            ]
        )

        if posture_score > 0:
            lines.append(f"   • Posture Assessment: {posture_score:.0f}/100")
        if symmetry_score > 0:
            lines.append(f"   • Body Symmetry Score: {symmetry_score:.0f}/100")

        lines.extend(
            [
                f"   • Processing Time: {processing_time:.2f}s",
                f"   • Analysis ID: {analysis_id}",
                f"   • Validation Status: {'⚠️ Issues Found' if has_issues else '✅ Clean'}",
            ]
        )

        # Health insights
        cv_risk = results.get("cardiovascular_risk")
        sleep_risk = results.get("sleep_apnea_risk")
        metabolic_health = results.get("metabolic_health")

        if any([cv_risk, sleep_risk, metabolic_health]):
            lines.extend(["", "🏥 **Health Risk Assessment:**"])
            if cv_risk:
                lines.append(f"   • Cardiovascular Risk: {cv_risk}")
            if sleep_risk:
                lines.append(f"   • Sleep Apnea Risk: {sleep_risk}")
            if metabolic_health:
                lines.append(f"   • Metabolic Health: {metabolic_health}")

        return "\n".join(lines)
