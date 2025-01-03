"""
SailPolar - ML-based polar analysis tool for sailboats
"""

from sailpolar.core.polar_analyzer import PolarAnalyzer
from sailpolar.core.models import PolarData, CurrentEstimate, InstrumentHealth

__version__ = "0.1.0"
__author__ = "Sebastien Rosset"

__all__ = [
    "PolarAnalyzer",
    "PolarData",
    "CurrentEstimate",
    "InstrumentHealth",
]