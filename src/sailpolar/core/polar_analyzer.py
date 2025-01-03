"""
Core polar analysis functionality.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from scipy.stats import gaussian_kde

from sailpolar.core.models import (
    PolarData, PolarPoint, CurrentEstimate, 
    InstrumentHealth, WaveCondition, SailConfiguration
)
from sailpolar.analysis.current_analyzer import CurrentAnalyzer
from sailpolar.analysis.instrument_analyzer import InstrumentAnalyzer
from sailpolar.parser.nmea0183 import NMEA0183Parser

@dataclass
class AnalysisConfig:
    """Configuration for polar analysis."""
    min_samples: int = 1000
    steady_state_duration: int = 300  # seconds
    speed_variation_threshold: float = 0.5  # knots
    heading_variation_threshold: float = 5.0  # degrees
    wind_variation_threshold: float = 2.0  # knots
    outlier_contamination: float = 0.1
    tws_bins: np.ndarray = field(default_factory=lambda: np.arange(0, 50, 2))
    twa_bins: np.ndarray = field(default_factory=lambda: np.arange(0, 360, 5))
    min_confidence_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_samples < 0:
            raise ValueError("min_samples must be positive")
        if self.steady_state_duration < 0:
            raise ValueError("steady_state_duration must be positive")
        if self.speed_variation_threshold < 0:
            raise ValueError("speed_variation_threshold must be positive")
        if self.heading_variation_threshold < 0:
            raise ValueError("heading_variation_threshold must be positive")
        if not 0 < self.outlier_contamination < 1:
            raise ValueError("outlier_contamination must be between 0 and 1")
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")

class PolarAnalyzer:
    """Main class for polar diagram analysis."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the analyzer with given configuration."""
        self.config = config or AnalysisConfig()
        self.current_analyzer = CurrentAnalyzer()
        self.instrument_analyzer = InstrumentAnalyzer()
        self.nmea_parser = NMEAParser()
        
    def analyze_file(self, filepath: str) -> PolarData:
        """Analyze NMEA log file and generate polar data."""
        # Parse NMEA data
        df = self.nmea_parser.parse_file(filepath)
        return self.analyze_data(df)
        
    def analyze_data(self, df: pd.DataFrame) -> PolarData:
        """Analyze parsed data and generate polar data."""
        # Check instrument health
        instrument_health = self.instrument_analyzer.analyze_all_instruments(df)
        
        # Estimate current
        current = self.current_analyzer.estimate_current(df)
        
        # Apply corrections based on instrument health
        df = self._apply_instrument_corrections(df, instrument_health)
        
        # Compensate for current
        if current:
            df = self.current_analyzer.compensate_for_current(df, current)
        
        # Identify steady state periods
        steady_mask = self._identify_steady_states(df)
        steady_data = df[steady_mask].copy()
        
        # Remove outliers
        clean_data = self._remove_outliers(steady_data)
        
        # Generate polar points
        points = self._generate_polar_points(clean_data, current)
        
        # Create polar data
        polar_data = PolarData(
            points=points,
            boat_type="unknown",  # Could be passed as parameter
            measurement_period=(df.index.min(), df.index.max()),
            conditions={
                "current": current,
                "instrument_health": instrument_health
            }
        )
        
        return polar_data

    def compare_with_manufacturer(self, measured: PolarData, manufacturer: PolarData) -> Dict:
        """Compare measured polar data with manufacturer's data."""
        comparison = {
            'overall_achievement': 0.0,
            'point_comparison': [],
            'analysis': {}
        }
        
        # Compare each measured point with manufacturer's data
        for point in measured.points:
            mfg_speed = manufacturer.get_speed(point.tws, point.twa)
            if mfg_speed is not None:
                achievement = point.speed / mfg_speed
                comparison['point_comparison'].append({
                    'tws': point.tws,
                    'twa': point.twa,
                    'measured_speed': point.speed,
                    'manufacturer_speed': mfg_speed,
                    'achievement': achievement,
                    'confidence': point.confidence
                })
        
        if comparison['point_comparison']:
            # Calculate overall achievement
            weighted_achievements = [
                p['achievement'] * p['confidence'] 
                for p in comparison['point_comparison']
            ]
            weights = [p['confidence'] for p in comparison['point_comparison']]
            comparison['overall_achievement'] = (
                sum(weighted_achievements) / sum(weights)
            )
            
            # Analyze patterns
            comparison['analysis'] = self._analyze_performance_patterns(
                comparison['point_comparison']
            )
            
        return comparison

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from the dataset."""
        # Select features for outlier detection
        features = ['STW', 'TWS', 'TWA']
        X = df[features].copy()
        
        # Normalize TWA to handle angle wrapping
        X['TWA_sin'] = np.sin(np.radians(X['TWA']))
        X['TWA_cos'] = np.cos(np.radians(X['TWA']))
        X.drop('TWA', axis=1, inplace=True)
        
        # Fit isolation forest
        iso_forest = IsolationForest(
            contamination=self.config.outlier_contamination,
            random_state=42
        )
        
        # Predict outliers
        is_inlier = iso_forest.fit_predict(X) == 1
        
        return df[is_inlier]

    def _correct_wind_data(
        self,
        df: pd.DataFrame,
        health: InstrumentHealth
    ) -> pd.DataFrame:
        """Apply corrections to wind data based on health assessment."""
        corrected = df.copy()
        
        if health.error_pattern == "alignment_error":
            # Correct systematic alignment error
            correction_angle = self._estimate_wind_alignment_correction(df)
            corrected['TWA'] = df['TWA'] + correction_angle
            
        elif health.error_pattern == "calibration_error":
            # Apply wind speed calibration factor
            correction_factor = self._estimate_wind_speed_correction(df)
            corrected['TWS'] = df['TWS'] * correction_factor
            
        return corrected

    def _correct_speed_data(
        self,
        df: pd.DataFrame,
        health: InstrumentHealth
    ) -> pd.DataFrame:
        """Apply corrections to speed data based on health assessment."""
        corrected = df.copy()
        
        if health.error_pattern == "paddle_wheel_fouling":
            # Estimate correction factor based on GPS SOG in suitable conditions
            correction_factor = self._estimate_speed_correction_factor(df)
            corrected['STW'] = df['STW'] * correction_factor
            
        return corrected

    def _estimate_wind_alignment_correction(self, df: pd.DataFrame) -> float:
        """Estimate wind sensor alignment correction."""
        # Compare tacking angles to detect systematic errors
        port_tacks = df[df['TWA'].between(30, 60)]
        stbd_tacks = df[df['TWA'].between(300, 330)]
        
        if len(port_tacks) < 100 or len(stbd_tacks) < 100:
            return 0.0
            
        # True tacking angle should be symmetric
        port_angle = port_tacks['TWA'].median()
        stbd_angle = 360 - stbd_tacks['TWA'].median()
        
        correction = (port_angle - stbd_angle) / 2
        return correction

    def _estimate_wind_speed_correction(self, df: pd.DataFrame) -> float:
        """Estimate wind speed correction factor."""
        # Could use various methods:
        # - Compare with forecast data
        # - Use statistical properties
        # - Compare with nearby weather stations
        return 1.0  # Default to no correction for now

    def _estimate_speed_correction_factor(self, df: pd.DataFrame) -> float:
        """Estimate speed correction factor using GPS data."""
        # Use periods with minimal current effect
        valid_conditions = (
            (df['STW'] > 2) &  # Minimum speed for reliable readings
            (abs(df['AWA']).between(60, 120))  # Beam reaching
        )
        
        if not valid_conditions.any():
            return 1.0
            
        speed_ratio = df.loc[valid_conditions, 'SOG'] / df.loc[valid_conditions, 'STW']
        
        # Use median ratio as correction factor
        return float(speed_ratio.median())

    def _analyze_performance_patterns(
        self,
        comparison_points: List[Dict]
    ) -> Dict:
        """Analyze patterns in performance comparison."""
        analysis = {
            'tws_patterns': {},
            'twa_patterns': {},
            'recommendations': []
        }
        
        # Group by wind speed ranges
        tws_ranges = pd.cut(
            [p['tws'] for p in comparison_points],
            bins=[0, 10, 15, 20, 25, 30, np.inf],
            labels=['0-10', '10-15', '15-20', '20-25', '25-30', '30+']
        )
        
        for tws_range in tws_ranges.unique():
            tws_points = [p for p, r in zip(comparison_points, tws_ranges) 
                         if r == tws_range]
            if tws_points:
                avg_achievement = np.mean([p['achievement'] for p in tws_points])
                analysis['tws_patterns'][str(tws_range)] = avg_achievement
        
        # Group by wind angle ranges
        twa_ranges = pd.cut(
            [p['twa'] for p in comparison_points],
            bins=[0, 45, 90, 135, 180],
            labels=['upwind', 'close reach', 'beam reach', 'broad reach']
        )
        
        for twa_range in twa_ranges.unique():
            twa_points = [p for p, r in zip(comparison_points, twa_ranges) 
                         if r == twa_range]
            if twa_points:
                avg_achievement = np.mean([p['achievement'] for p in twa_points])
                analysis['twa_patterns'][str(twa_range)] = avg_achievement
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_performance_recommendations(
            analysis['tws_patterns'],
            analysis['twa_patterns']
        )
        
        return analysis

    def _generate_performance_recommendations(
        self,
        tws_patterns: Dict[str, float],
        twa_patterns: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on performance patterns."""
        recommendations = []
        
        # Check for wind speed patterns
        if tws_patterns:
            worst_tws = min(tws_patterns.items(), key=lambda x: x[1])
            if worst_tws[1] < 0.9:  # More than 10% below target
                recommendations.append(
                    f"Performance in {worst_tws[0]} knots wind speed range is "
                    f"{worst_tws[1]:.1%} of target. Consider sail selection and trim."
                )
        
        # Check for wind angle patterns
        if twa_patterns:
            worst_twa = min(twa_patterns.items(), key=lambda x: x[1])
            if worst_twa[1] < 0.9:  # More than 10% below target
                recommendations.append(
                    f"Performance at {worst_twa[0]} is {worst_twa[1]:.1%} of target. "
                    "Review sail trim and helming technique."
                )
        
        return recommendations
