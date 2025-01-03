"""
Core data models for the sailpolar package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum


class InstrumentType(Enum):
    """Types of instruments that can be analyzed."""

    WIND = "wind"
    SPEED = "speed"
    COMPASS = "compass"
    GPS = "gps"
    DEPTH = "depth"


class CalibrationStatus(Enum):
    """Possible calibration statuses for instruments."""

    GOOD = "good"
    NEEDS_CALIBRATION = "needs_calibration"
    POSSIBLY_FAULTY = "possibly_faulty"
    UNKNOWN = "unknown"


@dataclass
class InstrumentHealth:
    """Represents the health status of a single instrument."""

    instrument_type: InstrumentType
    reliability_score: float  # 0-1 score
    calibration_status: CalibrationStatus
    error_pattern: str
    suggested_actions: List[str]
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate the reliability score and confidence."""
        if not 0 <= self.reliability_score <= 1:
            raise ValueError("Reliability score must be between 0 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class CurrentEstimate:
    """Represents a current estimation with uncertainty."""

    speed: float  # knots
    direction: float  # degrees true
    confidence: float  # 0-1 score
    source: str  # 'measured', 'predicted', 'derived'
    timestamp: datetime
    uncertainty: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize uncertainty if not provided."""
        if not self.uncertainty:
            self.uncertainty = {
                "speed": self.speed * (1 - self.confidence),
                "direction": 10
                * (1 - self.confidence),  # 10 degrees at minimum confidence
            }


@dataclass
class WaveCondition:
    """Represents wave conditions."""

    significant_height: float  # meters
    direction: float  # degrees true
    period: float  # seconds
    source: str  # 'measured', 'forecast', 'derived'
    timestamp: datetime
    uncertainty: Optional[Dict[str, float]] = None


@dataclass
class SailConfiguration:
    """Represents the sail configuration."""

    main_sail: bool = True
    main_reefs: int = 0
    headsail: str = "genoa"  # 'genoa', 'jib', 'storm_jib', etc.
    headsail_furling: float = 0.0  # percentage
    other_sails: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PolarPoint:
    """Represents a single point on a polar diagram."""

    tws: float  # True wind speed (knots)
    twa: float  # True wind angle (degrees)
    speed: float  # Boat speed (knots)
    confidence: float  # 0-1 score
    source: str  # 'measured', 'predicted', 'manufacturer'
    sail_config: Optional[SailConfiguration] = None
    wave_condition: Optional[WaveCondition] = None
    current: Optional[CurrentEstimate] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Normalize angles and validate data."""
        self.twa = self.normalize_angle(self.twa)
        if not 0 <= self.tws:
            raise ValueError("TWS must be non-negative")
        if not 0 <= self.speed:
            raise ValueError("Speed must be non-negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to 0-360 range."""
        return angle % 360


@dataclass
class PolarData:
    """Represents a complete polar diagram."""

    points: List[PolarPoint]
    boat_type: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    measurement_period: Optional[Tuple[datetime, datetime]] = None
    conditions: Dict[str, any] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)

    def get_speed(self, tws: float, twa: float) -> Optional[float]:
        """Get interpolated speed for given TWS/TWA."""
        # Find nearest points
        twa = PolarPoint.normalize_angle(twa)
        relevant_points = [
            p for p in self.points if abs(p.tws - tws) < 2 and abs(p.twa - twa) < 10
        ]

        if not relevant_points:
            return None

        # Weight by distance and confidence
        total_weight = 0
        weighted_speed = 0

        for point in relevant_points:
            # Calculate weight based on TWS and TWA difference
            tws_diff = abs(point.tws - tws)
            twa_diff = min(abs(point.twa - twa), 360 - abs(point.twa - twa))

            # Use inverse distance weighting
            weight = 1 / (1 + tws_diff + 0.2 * twa_diff)
            weight *= point.confidence

            total_weight += weight
            weighted_speed += point.speed * weight

        if total_weight == 0:
            return None

        return weighted_speed / total_weight

    def get_optimal_twa(self, tws: float, beating: bool = False) -> Optional[float]:
        """Get optimal TWA for given TWS, optionally for beating angles."""
        if beating:
            twa_range = (30, 60)  # Typical beating angles
        else:
            twa_range = (0, 180)  # All angles

        angles = np.arange(twa_range[0], twa_range[1], 1)
        speeds = [self.get_speed(tws, twa) for twa in angles]

        # Filter out None values
        valid_indices = [i for i, s in enumerate(speeds) if s is not None]
        if not valid_indices:
            return None

        valid_angles = [angles[i] for i in valid_indices]
        valid_speeds = [speeds[i] for i in valid_indices]

        # Calculate VMG for each angle
        vmg = [
            speed * np.cos(np.radians(angle))
            for speed, angle in zip(valid_speeds, valid_angles)
        ]

        # Find angle with best VMG
        best_idx = np.argmax(vmg)
        return valid_angles[best_idx]


@dataclass
class PerformanceAnalysis:
    """Results of performance analysis."""

    polar_achievement: float  # Percentage of polar speeds achieved
    optimal_angles: Dict[str, float]  # Optimal angles for different conditions
    limiting_factors: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, any] = field(default_factory=dict)
