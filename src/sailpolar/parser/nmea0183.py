"""
NMEA 0183 parser module.

This module implements a parser for NMEA 0183 sentences. It handles the various
sentence types relevant for sailing data analysis, including:
- RMC (Recommended Minimum Navigation Information)
- GGA (Global Positioning System Fix Data)
- VHW (Water Speed and Heading)
- MWV (Wind Speed and Angle)
- VTG (Track Made Good and Ground Speed)
- HDG (Heading, Deviation & Variation)
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional, List
import re


@dataclass
class NMEA0183Error(Exception):
    """Base class for NMEA 0183 parsing errors."""

    message: str


@dataclass
class ChecksumError(NMEA0183Error):
    """Raised when checksum validation fails."""

    sentence: str
    computed: str
    received: str


@dataclass
class NMEASentence:
    """Base class for NMEA sentences."""

    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "NMEASentence":
        """Parse a raw NMEA sentence and return appropriate sentence object."""
        # Validate basic structure
        if not raw_sentence.startswith("$"):
            raise NMEA0183Error("Sentence must start with $")

        # Check for minimum length (at least $GP)
        if len(raw_sentence) < 3:
            raise NMEA0183Error("Invalid sentence format")

        # Check for complete identifier ($GPXXX)
        if len(raw_sentence) < 6:
            raise NMEA0183Error("Invalid sentence identifier")

        # Extract checksum if present
        if "*" in raw_sentence:
            sentence, checksum = raw_sentence.rsplit("*", 1)
            if not validate_checksum(sentence[1:], checksum):
                raise ChecksumError(
                    message="Checksum validation failed",
                    sentence=raw_sentence,
                    computed=calculate_checksum(sentence[1:]),
                    received=checksum,
                )
        else:
            sentence = raw_sentence

        # Split into fields
        fields = sentence.split(",")
        if len(fields) < 2:
            raise NMEA0183Error("Invalid sentence format")

        sentence_id = fields[0][1:]  # Remove $
        talker_id = sentence_id[:2]
        sentence_type = sentence_id[2:]

        # Create appropriate sentence object based on type
        sentence_class = SENTENCE_TYPES.get(sentence_type, NMEASentence)
        return sentence_class(
            talker_id=talker_id, sentence_type=sentence_type, raw=raw_sentence
        )


def validate_checksum(data: str, checksum: str) -> bool:
    """Validate NMEA checksum."""
    return calculate_checksum(data) == checksum.upper()


def calculate_checksum(data: str) -> str:
    """Calculate NMEA checksum."""
    checksum = 0
    for char in data:
        checksum ^= ord(char)
    return f"{checksum:02X}"


@dataclass
class RMCSentence(NMEASentence):
    """
    RMC - Recommended Minimum Navigation Information

    Format:
    $GPRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,x.x,a*hh
    1    = UTC of position fix
    2    = Status (A = active or V = void)
    3    = Latitude
    4    = N or S
    5    = Longitude
    6    = E or W
    7    = Speed over ground in knots
    8    = Track made good in degrees True
    9    = Date
    10   = Magnetic variation degrees
    11   = E or W
    12   = Checksum
    """

    status: str = None
    latitude: float = None
    longitude: float = None
    sog: float = None
    cog: float = None
    mag_var: float = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "RMCSentence":
        """Parse RMC sentence."""
        base = super().parse(raw_sentence)
        fields = raw_sentence.split(",")

        if len(fields) < 12:
            raise NMEA0183Error("Invalid RMC sentence")

        # Parse time
        time_str = fields[1]
        date_str = fields[9]
        if time_str and date_str:
            time_utc = time(
                hour=int(time_str[0:2]),
                minute=int(time_str[2:4]),
                second=int(time_str[4:6]),
            )
            date_utc = datetime.strptime(date_str, "%d%m%y").date()
            timestamp = datetime.combine(date_utc, time_utc)
        else:
            timestamp = None

        # Parse position
        if fields[3] and fields[4] and fields[5] and fields[6]:
            lat = float(fields[3][:2]) + float(fields[3][2:]) / 60
            if fields[4] == "S":
                lat = -lat

            lon = float(fields[5][:3]) + float(fields[5][3:]) / 60
            if fields[6] == "W":
                lon = -lon
        else:
            lat = lon = None

        return cls(
            talker_id=base.talker_id,
            sentence_type=base.sentence_type,
            raw=raw_sentence,
            timestamp=timestamp,
            status=fields[2],
            latitude=lat,
            longitude=lon,
            sog=float(fields[7]) if fields[7] else None,
            cog=float(fields[8]) if fields[8] else None,
            mag_var=float(fields[10]) if fields[10] else None,
        )


@dataclass
class VHWSentence(NMEASentence):
    """
    VHW - Water Speed and Heading

    Format:
    $--VHW,x.x,T,x.x,M,x.x,N,x.x,K*hh
    1    = Heading degrees true
    2    = T = True
    3    = Heading degrees magnetic
    4    = M = Magnetic
    5    = Speed, knots
    6    = N = Knots
    7    = Speed, Km/Hr
    8    = K = Kilometers
    9    = Checksum
    """

    heading_true: float = None
    heading_mag: float = None
    speed_knots: float = None
    speed_kmh: float = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VHWSentence":
        """Parse VHW sentence."""
        base = super().parse(raw_sentence)
        fields = raw_sentence.split(",")

        if len(fields) < 8:
            raise NMEA0183Error("Invalid VHW sentence")

        return cls(
            talker_id=base.talker_id,
            sentence_type=base.sentence_type,
            raw=raw_sentence,
            heading_true=float(fields[1]) if fields[1] else None,
            heading_mag=float(fields[3]) if fields[3] else None,
            speed_knots=float(fields[5]) if fields[5] else None,
            speed_kmh=float(fields[7]) if fields[7] else None,
        )


@dataclass
class MWVSentence(NMEASentence):
    """
    MWV - Wind Speed and Angle

    Format:
    $--MWV,x.x,a,x.x,a,A*hh
    1    = Wind Angle, 0 to 360 degrees
    2    = Reference, R = Relative, T = True
    3    = Wind Speed
    4    = Wind Speed Units, K/M/N
    5    = Status, A = Data Valid
    6    = Checksum
    """

    wind_angle: float = None
    reference: str = None
    wind_speed: float = None
    speed_units: str = None
    status: str = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "MWVSentence":
        """Parse MWV sentence."""
        base = super().parse(raw_sentence)
        fields = raw_sentence.split(",")

        if len(fields) < 6:
            raise NMEA0183Error("Invalid MWV sentence")

        # Convert wind speed to knots if necessary
        speed = float(fields[3]) if fields[3] else None
        if speed and fields[4]:
            if fields[4] == "K":
                speed = speed * 0.539957  # km/h to knots
            elif fields[4] == "M":
                speed = speed * 1.94384  # m/s to knots

        return cls(
            talker_id=base.talker_id,
            sentence_type=base.sentence_type,
            raw=raw_sentence,
            wind_angle=float(fields[1]) if fields[1] else None,
            reference=fields[2],
            wind_speed=speed,
            speed_units="N",  # Always convert to knots
            status=fields[5].split("*")[0],  # Remove checksum
        )


@dataclass
class VPWSentence(NMEASentence):
    """VPW - Speed Parallel to Wind

    Format: $--VPW,x.x,N,x.x,M*hh
    Example: $IIVPW,04.34,N,,M*7F

    Fields:
        speed_knots: Speed parallel to wind in knots
        speed_meters: Speed parallel to wind in meters per second
    """

    speed_knots: Optional[float] = None
    speed_meters: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VPWSentence":
        """Parse a raw VPW sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Parse specific VPW fields
        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            speed_knots=float(fields[0]) if fields[0] else None,
            speed_meters=float(fields[2]) if fields[2] else None,
        )


@dataclass
class VTGSentence(NMEASentence):
    """VTG - Track Made Good and Ground Speed

    Format: $--VTG,x.x,T,x.x,M,x.x,N,x.x,K*hh
    Example: $IIVTG,183.,T,,M,5.1,N,,K*67
    """

    track_true: Optional[float] = None
    track_magnetic: Optional[float] = None
    speed_knots: Optional[float] = None
    speed_kmh: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VTGSentence":
        """Parse a raw VTG sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Parse specific VTG fields
        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            track_true=float(fields[0]) if fields[0] else None,
            track_magnetic=float(fields[2]) if fields[2] else None,
            speed_knots=float(fields[4]) if fields[4] else None,
            speed_kmh=float(fields[6]) if fields[6] else None,
        )


class VWRSentence(NMEASentence):
    """VWR - Relative Wind Speed and Angle

    Format: $--VWR,x.x,a,x.x,N,x.x,M,x.x,K*hh
    Example: $IIVWR,022,R,23.2,N,11.9,M,,K*73
    """

    def __init__(self):
        super().__init__()
        self.wind_angle = None  # Wind angle in degrees from 0-180
        self.wind_direction = None  # L=Left, R=Right of bow
        self.wind_speed_knots = None  # Wind speed in knots
        self.wind_speed_ms = None  # Wind speed in meters/second
        self.wind_speed_kmh = None  # Wind speed in kilometers/hour

    @classmethod
    def parse(cls, nmea_str: str) -> "VWRSentence":
        sentence = cls()

        # Parse common fields
        sentence.parse_common_fields(nmea_str)

        fields = sentence.fields

        # Parse wind angle if present
        if fields[0]:
            sentence.wind_angle = float(fields[0])

        # Parse direction relative to bow
        if fields[1]:
            sentence.wind_direction = fields[1]  # 'L' or 'R'

        # Parse wind speed in knots
        if fields[2]:
            sentence.wind_speed_knots = float(fields[2])

        # Parse wind speed in meters/second
        if fields[4]:
            sentence.wind_speed_ms = float(fields[4])

        # Parse wind speed in km/h
        if fields[6]:
            sentence.wind_speed_kmh = float(fields[6])

        return sentence


class NMEA0183Parser:
    """Parser for NMEA 0183 sentences."""

    def __init__(self, fail_unknown=False):
        """Initialize the parser.

        Args:
            fail_unknown (bool): If True, raise NMEA0183Error when encountering
                               unknown sentence types. If False, skip them.
        """
        self.fail_unknown = fail_unknown
        # Map of sentence types to their parser classes
        self._sentence_parsers = {
            "RMC": RMCSentence,
            "VHW": VHWSentence,
            "MWV": MWVSentence,
            "VPW": VPWSentence,
            "VTG": VTGSentence,
            "VWR": VWRSentence,
        }

    @classmethod
    def parse(cls, raw_sentence: str) -> "NMEASentence":
        """Parse a raw NMEA sentence and return appropriate sentence object."""
        # Validate basic structure
        if not raw_sentence.startswith("$"):
            raise NMEA0183Error("Sentence must start with $")

        # Check minimum length for sentence identifier
        if len(raw_sentence) < 6:  # Need at least $XXXXX
            raise NMEA0183Error("Invalid sentence identifier")

        # Extract checksum if present
        if "*" in raw_sentence:
            sentence, checksum = raw_sentence.rsplit("*", 1)
            if not validate_checksum(sentence[1:], checksum):
                raise ChecksumError(
                    message="Checksum validation failed",
                    sentence=raw_sentence,
                    computed=calculate_checksum(sentence[1:]),
                    received=checksum,
                )
        else:
            sentence = raw_sentence

        # Split into fields
        fields = sentence.split(",")
        if len(fields) < 2:
            raise NMEA0183Error("Invalid sentence format")

    def parse_sentence(self, line):
        """Parse a single NMEA sentence.

        Args:
            line (str): The NMEA sentence to parse

        Returns:
            NMEASentence: The parsed sentence, or None if the sentence is invalid
            or unknown and fail_unknown is False

        Raises:
            NMEA0183Error: If the sentence is invalid or unknown and fail_unknown is True
        """
        if not line.startswith("$"):
            return None

        try:
            sentence_type = line[
                3:6
            ]  # Extract sentence type (e.g., 'RMC' from '$GPRMC')

            if sentence_type not in self._sentence_parsers:
                if self.fail_unknown:
                    raise NMEA0183Error(f"Unsupported sentence type: {sentence_type}")
                return None

            return self._sentence_parsers[sentence_type].parse(line)

        except NMEA0183Error:
            raise
        except Exception as e:
            raise NMEA0183Error(f"Failed to parse sentence: {str(e)}")

    def parse_file(self, filepath: str) -> List[NMEASentence]:
        """Parse a file containing NMEA sentences.

        Args:
            filepath (str): Path to the file containing NMEA sentences

        Returns:
            List[NMEASentence]: List of parsed sentences

        Raises:
            NMEA0183Error: If fail_unknown is True and an unknown sentence type is encountered
        """
        sentences = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    sentence = self.parse_sentence(line)
                    if sentence:
                        sentences.append(sentence)
                except (NMEA0183Error, ChecksumError) as e:
                    if self.fail_unknown and isinstance(e, NMEA0183Error):
                        raise  # Re-raise only NMEA0183Error when fail_unknown is True
                    continue  # Skip invalid sentences otherwise

        return sentences
