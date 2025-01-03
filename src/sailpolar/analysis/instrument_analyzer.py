"""
Instrument analysis and health monitoring functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from sailpolar.core.models import InstrumentHealth, InstrumentType, CalibrationStatus


@dataclass
class InstrumentAnalyzerConfig:
    """Configuration for instrument analysis."""

    min_samples: int = 1000
    analysis_window: pd.Timedelta = pd.Timedelta(minutes=30)
    max_gps_hdop: float = 5.0
    min_speed_for_analysis: float = 2.0
    max_speed_variation: float = 0.5  # knots
    max_heading_variation: float = 5.0  # degrees
    max_wind_variation: float = 2.0  # knots
    outlier_contamination: float = 0.1
    calibration_threshold: float = 0.9  # minimum acceptable calibration score


class InstrumentAnalyzer:
    """Analyzes and monitors instrument health and calibration."""

    def __init__(self, config: Optional[InstrumentAnalyzerConfig] = None):
        """Initialize with given configuration."""
        self.config = config or InstrumentAnalyzerConfig()
        self._analysis_history: Dict[str, List[InstrumentHealth]] = {}

    def analyze_all_instruments(self, df: pd.DataFrame) -> Dict[str, InstrumentHealth]:
        """
        Analyze health of all available instruments.

        Args:
            df: DataFrame with instrument data

        Returns:
            Dictionary mapping instrument types to their health status
        """
        results = {}

        # Analyze GPS
        if self._has_gps_data(df):
            results[InstrumentType.GPS] = self._analyze_gps(df)

        # Analyze Speed instruments
        if self._has_speed_data(df):
            results[InstrumentType.SPEED] = self._analyze_speed_instruments(df)

        # Analyze Wind instruments
        if self._has_wind_data(df):
            results[InstrumentType.WIND] = self._analyze_wind_instruments(df)

        # Analyze Compass
        if self._has_compass_data(df):
            results[InstrumentType.COMPASS] = self._analyze_compass(df)

        # Update analysis history
        for instrument_type, health in results.items():
            self._update_history(instrument_type.value, health)

        return results

    def _has_gps_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required GPS data."""
        required_columns = ["SOG", "COG", "latitude", "longitude"]
        return all(col in df.columns for col in required_columns)

    def _has_speed_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required speed data."""
        return "STW" in df.columns

    def _has_wind_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required wind data."""
        required_columns = ["AWS", "AWA", "TWS", "TWA"]
        return all(col in df.columns for col in required_columns)

    def _has_compass_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required compass data."""
        return "heading" in df.columns

    def _analyze_gps(self, df: pd.DataFrame) -> InstrumentHealth:
        """Analyze GPS health and accuracy."""
        # Check HDOP if available
        if "hdop" in df.columns:
            hdop_score = 1.0 - min(1.0, df["hdop"].mean() / self.config.max_gps_hdop)
        else:
            hdop_score = 0.8  # Default if HDOP not available

        # Check position consistency
        position_score = self._check_position_consistency(df)

        # Check speed/course consistency
        motion_score = self._check_motion_consistency(df)

        # Combine scores
        reliability_score = np.mean([hdop_score, position_score, motion_score])

        return InstrumentHealth(
            instrument_type=InstrumentType.GPS,
            reliability_score=float(reliability_score),
            calibration_status=CalibrationStatus.GOOD,  # GPS doesn't need calibration
            error_pattern="none" if reliability_score > 0.8 else "inconsistent_data",
            suggested_actions=self._get_gps_suggestions(reliability_score),
            confidence=0.9,  # High confidence in GPS analysis
        )

    def _analyze_speed_instruments(self, df: pd.DataFrame) -> InstrumentHealth:
        """Analyze speed instrument health and calibration."""
        # Compare STW with SOG in suitable conditions
        speed_comparison = self._analyze_speed_comparison(df)

        # Check for potential fouling
        fouling_score = self._check_paddle_wheel_fouling(df)

        # Check for noise and stuck readings
        signal_quality = self._analyze_signal_quality(df["STW"])

        # Calculate overall reliability
        reliability_score = np.mean(
            [speed_comparison["score"], fouling_score, signal_quality["score"]]
        )

        # Determine calibration status
        if speed_comparison["calibration_score"] > self.config.calibration_threshold:
            calibration_status = CalibrationStatus.GOOD
        else:
            calibration_status = CalibrationStatus.NEEDS_CALIBRATION

        return InstrumentHealth(
            instrument_type=InstrumentType.SPEED,
            reliability_score=float(reliability_score),
            calibration_status=calibration_status,
            error_pattern=speed_comparison["error_pattern"],
            suggested_actions=self._get_speed_suggestions(
                speed_comparison, fouling_score, signal_quality
            ),
            confidence=float(speed_comparison["confidence"]),
        )

    def _analyze_wind_instruments(self, df: pd.DataFrame) -> InstrumentHealth:
        """Analyze wind instrument health and calibration."""
        # Check wind data consistency
        consistency = self._check_wind_consistency(df)

        # Analyze wind sensor alignment
        alignment = self._check_wind_alignment(df)

        # Check for mechanical issues
        mechanical = self._check_wind_mechanical_health(df)

        # Calculate overall reliability
        reliability_score = np.mean(
            [consistency["score"], alignment["score"], mechanical["score"]]
        )

        # Determine calibration status
        if alignment["score"] > self.config.calibration_threshold:
            calibration_status = CalibrationStatus.GOOD
        else:
            calibration_status = CalibrationStatus.NEEDS_CALIBRATION

        return InstrumentHealth(
            instrument_type=InstrumentType.WIND,
            reliability_score=float(reliability_score),
            calibration_status=calibration_status,
            error_pattern=self._determine_wind_error_pattern(
                consistency, alignment, mechanical
            ),
            suggested_actions=self._get_wind_suggestions(
                consistency, alignment, mechanical
            ),
            confidence=float(
                np.mean(
                    [
                        consistency["confidence"],
                        alignment["confidence"],
                        mechanical["confidence"],
                    ]
                )
            ),
        )

    def _analyze_compass(self, df: pd.DataFrame) -> InstrumentHealth:
        """Analyze compass health and calibration."""
        # Compare with GPS course in suitable conditions
        course_comparison = self._analyze_course_comparison(df)

        # Check for magnetic interference
        interference = self._check_magnetic_interference(df)

        # Check for heading stability
        stability = self._check_heading_stability(df)

        # Calculate overall reliability
        reliability_score = np.mean(
            [course_comparison["score"], interference["score"], stability["score"]]
        )

        # Determine calibration status
        if (
            course_comparison["deviation"] < 5.0
        ):  # Less than 5 degrees average deviation
            calibration_status = CalibrationStatus.GOOD
        else:
            calibration_status = CalibrationStatus.NEEDS_CALIBRATION

        return InstrumentHealth(
            instrument_type=InstrumentType.COMPASS,
            reliability_score=float(reliability_score),
            calibration_status=calibration_status,
            error_pattern=self._determine_compass_error_pattern(
                course_comparison, interference, stability
            ),
            suggested_actions=self._get_compass_suggestions(
                course_comparison, interference, stability
            ),
            confidence=float(
                np.mean(
                    [
                        course_comparison["confidence"],
                        interference["confidence"],
                        stability["confidence"],
                    ]
                )
            ),
        )

    def _analyze_speed_comparison(self, df: pd.DataFrame) -> Dict:
        """Analyze speed instrument calibration by comparing with GPS."""
        # Only analyze when moving at sufficient speed in steady conditions
        valid_conditions = (df["SOG"] >= self.config.min_speed_for_analysis) & (
            df["SOG"].rolling(window="5min").std() <= self.config.max_speed_variation
        )

        valid_data = df[valid_conditions]

        if len(valid_data) < self.config.min_samples:
            return {
                "score": 0.5,
                "calibration_score": 0.5,
                "error_pattern": "insufficient_data",
                "confidence": 0.3,
            }

        # Calculate speed ratio
        speed_ratio = valid_data["STW"] / valid_data["SOG"]

        # Analyze ratio distribution
        ratio_mean = speed_ratio.mean()
        ratio_std = speed_ratio.std()

        # Calculate scores
        calibration_score = 1.0 - min(1.0, abs(ratio_mean - 1.0))
        consistency_score = 1.0 - min(1.0, ratio_std)

        return {
            "score": float(np.mean([calibration_score, consistency_score])),
            "calibration_score": float(calibration_score),
            "error_pattern": "needs_calibration" if calibration_score < 0.8 else "none",
            "confidence": float(1.0 - ratio_std),
        }

    def _check_paddle_wheel_fouling(self, df: pd.DataFrame) -> float:
        """Check for signs of paddle wheel fouling."""
        if len(df) < self.config.min_samples:
            return 0.5

        # Look for characteristic signs of fouling:
        # 1. Decreasing trend in STW/SOG ratio
        # 2. Increased noise in measurements
        # 3. Stuck readings

        # Calculate trend in speed ratio
        speed_ratio = df["STW"] / df["SOG"]
        trend = np.polyfit(range(len(speed_ratio)), speed_ratio, 1)[0]

        # Calculate noise level
        noise = speed_ratio.diff().std()

        # Check for stuck readings
        stuck_threshold = pd.Timedelta(seconds=10)
        stuck_mask = df["STW"].rolling(window=stuck_threshold).std() == 0
        stuck_percentage = stuck_mask.mean()

        # Combine indicators
        trend_score = 1.0 - min(1.0, abs(trend) * 1000)  # Scale trend appropriately
        noise_score = 1.0 - min(1.0, noise * 5)  # Scale noise appropriately
        stuck_score = 1.0 - stuck_percentage

        return float(np.mean([trend_score, noise_score, stuck_score]))

    def _check_wind_consistency(self, df: pd.DataFrame) -> Dict:
        """Check consistency of wind measurements."""
        if len(df) < self.config.min_samples:
            return {
                "score": 0.5,
                "confidence": 0.3,
                "details": {"error": "insufficient_data"},
            }

        # Calculate true wind from apparent wind and vice versa
        calculated_twa = self._calculate_true_wind_angle(
            df["AWA"], df["AWS"], df["SOG"], df["heading"]
        )

        # Compare calculated vs measured
        twa_diff = np.abs(calculated_twa - df["TWA"])

        # Calculate scores
        consistency_score = 1.0 - min(1.0, twa_diff.mean() / 30)  # Scale by 30 degrees
        confidence = 1.0 - min(1.0, twa_diff.std() / 15)  # Scale by 15 degrees

        return {
            "score": float(consistency_score),
            "confidence": float(confidence),
            "details": {
                "mean_difference": float(twa_diff.mean()),
                "max_difference": float(twa_diff.max()),
            },
        }

    def _calculate_true_wind_angle(
        self, awa: pd.Series, aws: pd.Series, sog: pd.Series, heading: pd.Series
    ) -> pd.Series:
        """Calculate true wind angle from apparent wind and boat motion."""
        # Convert apparent wind to vector components
        awa_rad = np.radians(awa)
        aws_x = aws * np.sin(awa_rad)
        aws_y = aws * np.cos(awa_rad)

        # Add boat motion
        heading_rad = np.radians(heading)
        boat_x = sog * np.sin(heading_rad)
        boat_y = sog * np.cos(heading_rad)

        # Calculate true wind components
        true_x = aws_x + boat_x
        true_y = aws_y + boat_y

        # Calculate true wind angle
        twa = np.degrees(np.arctan2(true_x, true_y))

        # Normalize to 0-360
        return twa % 360

    def _check_magnetic_interference(self, df: pd.DataFrame) -> Dict:
        """Check for signs of magnetic interference affecting compass."""
        if len(df) < self.config.min_samples:
            return {
                "score": 0.5,
                "confidence": 0.3,
                "details": {"error": "insufficient_data"},
            }

        # Compare heading with GPS COG in steady conditions
        steady_conditions = (df["SOG"] >= self.config.min_speed_for_analysis) & (
            df["SOG"].rolling(window="5min").std() <= self.config.max_speed_variation
        )

        if not steady_conditions.any():
            return {
                "score": 0.5,
                "confidence": 0.3,
                "details": {"error": "no_steady_conditions"},
            }

        # Calculate heading deviation
        deviation = np.abs(
            df.loc[steady_conditions, "heading"] - df.loc[steady_conditions, "COG"]
        )
        deviation = np.minimum(deviation, 360 - deviation)  # Handle wrapping

        # Look for patterns indicating interference
        heading_bins = pd.cut(df.loc[steady_conditions, "heading"], bins=8)
        deviation_by_heading = deviation.groupby(heading_bins).agg(["mean", "std"])

        # Score based on overall deviation and heading-dependence
        mean_score = 1.0 - min(1.0, deviation.mean() / 10)  # Scale by 10 degrees
        variation_score = 1.0 - min(1.0, deviation_by_heading["mean"].std() / 5)

        # Higher variation by heading indicates magnetic interference
        interference_detected = variation_score < 0.7

        return {
            "score": float(np.mean([mean_score, variation_score])),
            "confidence": float(0.8 if len(deviation) > 1000 else 0.5),
            "details": {
                "interference_detected": interference_detected,
                "mean_deviation": float(deviation.mean()),
                "max_deviation": float(deviation.max()),
                "heading_dependent": bool(variation_score < 0.7),
            },
        }

    def _check_heading_stability(self, df: pd.DataFrame) -> Dict:
        """Analyze heading stability and noise."""
        if len(df) < self.config.min_samples:
            return {
                "score": 0.5,
                "confidence": 0.3,
                "details": {"error": "insufficient_data"},
            }

        # Calculate heading rate of change
        heading_rate = df["heading"].diff().abs()

        # Remove normal turn rates (keep only potential instability)
        stable_mask = heading_rate <= self.config.max_heading_variation
        stability_score = stable_mask.mean()

        # Check for noise in stable periods
        if stable_mask.any():
            noise = df.loc[stable_mask, "heading"].diff().std()
            noise_score = 1.0 - min(1.0, noise)
        else:
            noise_score = 0.5

        return {
            "score": float(np.mean([stability_score, noise_score])),
            "confidence": 0.8,
            "details": {
                "stability_score": float(stability_score),
                "noise_score": float(noise_score),
                "max_rate": float(heading_rate.max()),
            },
        }

    def _analyze_signal_quality(self, series: pd.Series) -> Dict:
        """Analyze signal quality for any instrument."""
        if len(series) < self.config.min_samples:
            return {
                "score": 0.5,
                "confidence": 0.3,
                "details": {"error": "insufficient_data"},
            }

        # Check for noise
        noise_level = series.diff().std()
        noise_score = 1.0 - min(1.0, noise_level)

        # Check for stuck values
        stuck_threshold = pd.Timedelta(seconds=10)
        stuck_mask = series.rolling(window=stuck_threshold).std() == 0
        stuck_score = 1.0 - stuck_mask.mean()

        # Check for gaps
        timestamps = pd.Series(series.index)
        gaps = timestamps.diff()
        gap_score = 1.0 - min(1.0, gaps.max().total_seconds() / 60)

        return {
            "score": float(np.mean([noise_score, stuck_score, gap_score])),
            "confidence": 0.8,
            "details": {
                "noise_level": float(noise_level),
                "stuck_percentage": float(stuck_mask.mean()),
                "max_gap_seconds": float(gaps.max().total_seconds()),
            },
        }

    def _check_position_consistency(self, df: pd.DataFrame) -> float:
        """Check GPS position consistency with reported speed."""
        if len(df) < self.config.min_samples:
            return 0.5

        # Calculate distance between consecutive points
        lat_rad = np.radians(df["latitude"])
        lon_rad = np.radians(df["longitude"])

        dlat = lat_rad.diff()
        dlon = lon_rad.diff()

        # Haversine formula
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371 * c  # Earth radius in km

        # Convert to knots
        time_diff = df.index.to_series().diff().dt.total_seconds()
        calculated_speed = distances / time_diff * 3600 / 1.852

        # Compare with reported SOG
        speed_ratio = calculated_speed / df["SOG"].iloc[1:]

        # Score based on consistency
        consistency_score = 1.0 - min(1.0, abs(speed_ratio.mean() - 1.0))
        variation_score = 1.0 - min(1.0, speed_ratio.std())

        return float(np.mean([consistency_score, variation_score]))

    def _check_motion_consistency(self, df: pd.DataFrame) -> float:
        """Check consistency between GPS course and speed vectors."""
        if len(df) < self.config.min_samples:
            return 0.5

        # Convert SOG/COG to vectors
        cog_rad = np.radians(df["COG"])
        x_vel = df["SOG"] * np.sin(cog_rad)
        y_vel = df["SOG"] * np.cos(cog_rad)

        # Calculate acceleration
        x_acc = x_vel.diff()
        y_acc = y_vel.diff()

        # Check for physically reasonable accelerations
        max_reasonable_acc = 2.0  # knots per second
        reasonable_mask = np.sqrt(x_acc**2 + y_acc**2) <= max_reasonable_acc

        return float(reasonable_mask.mean())

    def _determine_wind_error_pattern(
        self, consistency: Dict, alignment: Dict, mechanical: Dict
    ) -> str:
        """Determine the most likely error pattern in wind data."""
        patterns = []

        if alignment["score"] < 0.8:
            patterns.append("alignment_error")

        if consistency["score"] < 0.8:
            patterns.append("calibration_error")

        if mechanical["score"] < 0.8:
            patterns.append("mechanical_issue")

        if not patterns:
            return "none"

        return ",".join(patterns)

    def _determine_compass_error_pattern(
        self, course_comparison: Dict, interference: Dict, stability: Dict
    ) -> str:
        """Determine the most likely error pattern in compass data."""
        if interference["details"].get("interference_detected", False):
            return "magnetic_interference"

        if course_comparison.get("deviation", 0) > 5.0:
            return "needs_calibration"

        if stability["score"] < 0.8:
            return "unstable_readings"

        return "none"

    def _get_gps_suggestions(self, reliability_score: float) -> List[str]:
        """Get suggestions for GPS issues."""
        suggestions = []

        if reliability_score < 0.8:
            suggestions.extend(
                [
                    "Check GPS antenna for clear sky view",
                    "Verify GPS antenna connection",
                    "Consider upgrading to dual GPS for redundancy",
                ]
            )

        return suggestions

    def _get_speed_suggestions(
        self, speed_comparison: Dict, fouling_score: float, signal_quality: Dict
    ) -> List[str]:
        """Get suggestions for speed instrument issues."""
        suggestions = []

        if speed_comparison["calibration_score"] < 0.8:
            suggestions.append("Calibrate speed sensor using GPS reference")

        if fouling_score < 0.8:
            suggestions.extend(
                [
                    "Clean paddle wheel sensor",
                    "Check for marine growth",
                    "Verify paddle wheel moves freely",
                ]
            )

        if signal_quality["score"] < 0.8:
            suggestions.extend(
                [
                    "Check sensor wiring",
                    "Verify sensor mounting",
                    "Consider sensor replacement if persistent",
                ]
            )

        return suggestions

    def _get_wind_suggestions(
        self, consistency: Dict, alignment: Dict, mechanical: Dict
    ) -> List[str]:
        """Get suggestions for wind instrument issues."""
        suggestions = []

        if alignment["score"] < 0.8:
            suggestions.extend(
                [
                    "Check wind vane alignment",
                    "Verify masthead unit installation",
                    "Perform wind sensor alignment calibration",
                ]
            )

        if consistency["score"] < 0.8:
            suggestions.extend(
                [
                    "Check wind speed calibration",
                    "Verify wind angle calibration",
                    "Compare with secondary wind source if available",
                ]
            )

        if mechanical["score"] < 0.8:
            suggestions.extend(
                [
                    "Check wind cups/vane for damage",
                    "Verify free movement of all parts",
                    "Inspect bearings and mounting",
                ]
            )

        return suggestions

    def _get_compass_suggestions(
        self, course_comparison: Dict, interference: Dict, stability: Dict
    ) -> List[str]:
        """Get suggestions for compass issues."""
        suggestions = []

        if interference["details"].get("interference_detected", False):
            suggestions.extend(
                [
                    "Check for nearby magnetic sources",
                    "Verify compass mounting location",
                    "Consider relocating electronic devices",
                ]
            )

        if course_comparison.get("deviation", 0) > 5.0:
            suggestions.extend(
                [
                    "Perform compass calibration",
                    "Create deviation table",
                    "Verify compass installation",
                ]
            )

        if stability["score"] < 0.8:
            suggestions.extend(
                [
                    "Check compass mounting",
                    "Verify power supply stability",
                    "Consider damping settings",
                ]
            )

        return suggestions

    def _update_history(self, instrument_type: str, health: InstrumentHealth):
        """Update instrument health history."""
        if instrument_type not in self._analysis_history:
            self._analysis_history[instrument_type] = []

        self._analysis_history[instrument_type].append(health)

        # Keep only last 24 hours of history
        cutoff_time = datetime.now() - pd.Timedelta(days=1)
        self._analysis_history[instrument_type] = [
            h
            for h in self._analysis_history[instrument_type]
            if h.last_updated > cutoff_time
        ]
