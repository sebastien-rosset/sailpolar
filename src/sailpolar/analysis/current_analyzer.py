"""
Current analysis and compensation functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy.optimize import minimize
from dataclasses import dataclass

from sailpolar.core.models import CurrentEstimate


@dataclass
class CurrentAnalyzerConfig:
    """Configuration for current analysis."""

    min_samples_for_estimation: int = 20
    max_current_speed: float = 8.0  # knots
    min_speed_for_estimation: float = 2.0  # minimum SOG for reliable estimates
    heading_change_threshold: float = 30  # degrees
    confidence_threshold: float = 0.7
    max_time_gap: pd.Timedelta = pd.Timedelta(minutes=5)
    vector_agreement_threshold: float = 0.8
    min_confidence: float = 0.3


class CurrentAnalyzer:
    """Analyzes and compensates for current effects."""

    def __init__(self, config: Optional[CurrentAnalyzerConfig] = None):
        """Initialize with given configuration."""
        self.config = config or CurrentAnalyzerConfig()

    def estimate_current(
        self, df: pd.DataFrame, tidal_predictions: Optional[Dict] = None
    ) -> Optional[CurrentEstimate]:
        """
        Estimate current using multiple methods and combine results.

        Args:
            df: DataFrame with sailing data
            tidal_predictions: Optional dictionary of tidal predictions

        Returns:
            CurrentEstimate object or None if estimation not possible
        """
        estimates = []

        # Method 1: Direct measurement (SOG vs STW vectors)
        if self._has_required_data(df, ["SOG", "COG", "STW", "heading"]):
            direct_estimate = self._estimate_from_vector_difference(df)
            if direct_estimate:
                estimates.append(direct_estimate)

        # Method 2: Use tidal predictions if available
        if tidal_predictions is not None:
            tidal_estimate = self._get_tidal_prediction(df.index[0], tidal_predictions)
            if tidal_estimate:
                estimates.append(tidal_estimate)

        # Method 3: Derive from boat behavior during turns
        turn_estimate = self._estimate_from_turns(df)
        if turn_estimate:
            estimates.append(turn_estimate)

        if not estimates:
            return None

        # Combine estimates with weighted average based on confidence
        return self._combine_estimates(estimates)

    def compensate_for_current(
        self, df: pd.DataFrame, current: CurrentEstimate
    ) -> pd.DataFrame:
        """
        Adjust measurements to remove current effects.

        Args:
            df: DataFrame with sailing data
            current: CurrentEstimate object with current information

        Returns:
            DataFrame with current-compensated values
        """
        compensated = df.copy()

        # Convert current to vector components
        current_x = current.speed * np.cos(np.radians(current.direction))
        current_y = current.speed * np.sin(np.radians(current.direction))

        # Adjust SOG and COG to get true boat motion relative to water
        sog_x = df["SOG"] * np.cos(np.radians(df["COG"]))
        sog_y = df["SOG"] * np.sin(np.radians(df["COG"]))

        # Remove current effect
        true_x = sog_x - current_x
        true_y = sog_y - current_y

        # Calculate corrected speed and course
        compensated["true_speed"] = np.sqrt(true_x**2 + true_y**2)
        compensated["true_course"] = np.degrees(np.arctan2(true_y, true_x))

        # Adjust apparent wind calculations
        self._compensate_wind_for_current(compensated, current)

        return compensated

    def _has_required_data(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check if DataFrame has required columns with valid data."""
        return (
            all(col in df.columns for col in columns)
            and not df[columns].isna().any().any()
        )

    def _estimate_from_vector_difference(
        self, df: pd.DataFrame
    ) -> Optional[CurrentEstimate]:
        """
        Estimate current from vector difference between SOG/COG and STW/heading.
        """
        # Ensure minimum speed for reliable estimation
        valid_data = df[df["SOG"] >= self.config.min_speed_for_estimation].copy()

        if len(valid_data) < self.config.min_samples_for_estimation:
            return None

        # Convert SOG/COG to vector
        sog_x = valid_data["SOG"] * np.cos(np.radians(valid_data["COG"]))
        sog_y = valid_data["SOG"] * np.sin(np.radians(valid_data["COG"]))

        # Convert STW/heading to vector
        stw_x = valid_data["STW"] * np.cos(np.radians(valid_data["heading"]))
        stw_y = valid_data["STW"] * np.sin(np.radians(valid_data["heading"]))

        # Current vector is the difference
        current_x = sog_x - stw_x
        current_y = sog_y - stw_y

        # Calculate mean current
        mean_current_x = current_x.mean()
        mean_current_y = current_y.mean()

        current_speed = np.sqrt(mean_current_x**2 + mean_current_y**2)
        current_direction = np.degrees(np.arctan2(mean_current_y, mean_current_x))

        # Sanity check on current speed
        if current_speed > self.config.max_current_speed:
            return None

        # Calculate confidence based on consistency of estimates
        variation = np.std(np.sqrt(current_x**2 + current_y**2))
        confidence = max(
            self.config.min_confidence, 1.0 - min(variation / current_speed, 0.8)
        )

        return CurrentEstimate(
            speed=float(current_speed),
            direction=float(current_direction % 360),
            confidence=confidence,
            source="measured",
            timestamp=df.index[0],
        )

    def _estimate_from_turns(self, df: pd.DataFrame) -> Optional[CurrentEstimate]:
        """
        Estimate current from boat behavior during turns.
        This method is particularly useful because turning through different headings
        relative to the current creates a distinctive pattern.
        """
        # Find significant heading changes
        heading_changes = df["heading"].diff().abs()
        turn_periods = heading_changes > self.config.heading_change_threshold

        if not turn_periods.any():
            return None

        # Extract turn segments
        turn_segments = self._extract_turn_segments(df[turn_periods])

        if len(turn_segments) < 2:
            return None

        # Optimize current vector to explain observed turn behavior
        def objective(current_vector):
            current_speed, current_direction = current_vector
            error = 0
            for segment in turn_segments:
                predicted_cog = self._predict_cog(
                    segment["STW"].mean(),
                    segment["heading"].mean(),
                    current_speed,
                    current_direction,
                )
                error += (predicted_cog - segment["COG"].mean()) ** 2
            return error

        result = minimize(
            objective,
            x0=[1.0, 0.0],  # Initial guess
            bounds=[(0, self.config.max_current_speed), (0, 360)],
            method="Nelder-Mead",
        )

        if not result.success:
            return None

        current_speed, current_direction = result.x
        confidence = self.config.min_confidence + (1.0 - self.config.min_confidence) / (
            1.0 + result.fun
        )

        return CurrentEstimate(
            speed=float(current_speed),
            direction=float(current_direction % 360),
            confidence=confidence,
            source="derived",
            timestamp=df.index[0],
        )

    def _extract_turn_segments(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract consistent segments during turns."""
        segments = []
        current_segment = []
        last_timestamp = None

        for idx, row in df.iterrows():
            if (
                last_timestamp is None
                or (idx - last_timestamp) <= self.config.max_time_gap
            ):
                current_segment.append(row)
            else:
                if len(current_segment) >= self.config.min_samples_for_estimation:
                    segments.append(pd.DataFrame(current_segment))
                current_segment = [row]
            last_timestamp = idx

        if len(current_segment) >= self.config.min_samples_for_estimation:
            segments.append(pd.DataFrame(current_segment))

        return segments

    def _get_tidal_prediction(
        self, timestamp: pd.Timestamp, predictions: Dict
    ) -> Optional[CurrentEstimate]:
        """Get current prediction from tidal model."""
        if not predictions:
            return None

        # Find nearest prediction time
        nearest_time = min(predictions.keys(), key=lambda x: abs(x - timestamp))
        prediction = predictions[nearest_time]

        # Adjust confidence based on time difference
        time_diff = abs((nearest_time - timestamp).total_seconds() / 3600)
        confidence = max(
            self.config.min_confidence,
            1.0 - (time_diff / 6),  # Reduce confidence with time difference
        )

        return CurrentEstimate(
            speed=prediction["speed"],
            direction=prediction["direction"],
            confidence=confidence,
            source="predicted",
            timestamp=timestamp,
        )

    def _combine_estimates(self, estimates: List[CurrentEstimate]) -> CurrentEstimate:
        """Combine multiple current estimates using weighted average."""
        if not estimates:
            return None

        # Weight by confidence
        total_weight = sum(est.confidence for est in estimates)

        # Convert to vector components for averaging
        weighted_x = sum(
            est.speed * np.cos(np.radians(est.direction)) * est.confidence
            for est in estimates
        )
        weighted_y = sum(
            est.speed * np.sin(np.radians(est.direction)) * est.confidence
            for est in estimates
        )

        # Calculate combined vector
        combined_speed = np.sqrt(weighted_x**2 + weighted_y**2) / total_weight
        combined_direction = np.degrees(np.arctan2(weighted_y, weighted_x))

        # Calculate vector agreement
        vector_agreement = self._calculate_vector_agreement(estimates)

        # Combined confidence based on individual confidences and agreement
        combined_confidence = (
            sum(est.confidence for est in estimates) / len(estimates)
        ) * vector_agreement

        return CurrentEstimate(
            speed=float(combined_speed),
            direction=float(combined_direction % 360),
            confidence=float(combined_confidence),
            source="combined",
            timestamp=estimates[0].timestamp,
        )

    def _calculate_vector_agreement(self, estimates: List[CurrentEstimate]) -> float:
        """Calculate how well different current estimates agree."""
        if len(estimates) < 2:
            return 1.0

        vectors = [
            (
                est.speed * np.cos(np.radians(est.direction)),
                est.speed * np.sin(np.radians(est.direction)),
            )
            for est in estimates
        ]

        # Calculate variance of vector components
        var_x = np.var([v[0] for v in vectors])
        var_y = np.var([v[1] for v in vectors])

        # Normalize by mean current speed
        mean_speed = np.mean([est.speed for est in estimates])
        if mean_speed == 0:
            return 1.0

        variance = (var_x + var_y) / (mean_speed**2)

        return max(self.config.min_confidence, 1.0 / (1.0 + variance))

    def _compensate_wind_for_current(
        self, df: pd.DataFrame, current: CurrentEstimate
    ) -> None:
        """Adjust apparent wind calculations for current effects."""
        # Convert wind to vector components
        awa_rad = np.radians(df["AWA"])
        aws_x = df["AWS"] * np.sin(awa_rad)
        aws_y = df["AWS"] * np.cos(awa_rad)

        # Add current components
        current_x = current.speed * np.cos(np.radians(current.direction))
        current_y = current.speed * np.sin(np.radians(current.direction))

        # Calculate true wind components
        tws_x = aws_x + current_x
        tws_y = aws_y + current_y

        # Update true wind speed and angle
        df["TWS"] = np.sqrt(tws_x**2 + tws_y**2)
        df["TWA"] = np.degrees(np.arctan2(tws_x, tws_y)) % 360
