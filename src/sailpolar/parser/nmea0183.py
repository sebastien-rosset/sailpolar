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

from dataclasses import dataclass, field
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

        # Create base sentence object - no more SENTENCE_TYPES lookup
        return cls(talker_id=talker_id, sentence_type=sentence_type, raw=raw_sentence)


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
            raise NMEA0183Error("Invalid MWV sentence: insufficient fields")

        # Parse wind angle
        wind_angle = None
        if fields[1]:
            try:
                wind_angle = float(fields[1])
            except ValueError:
                raise NMEA0183Error(
                    f"Invalid MWV sentence: invalid wind angle value '{fields[1]}'"
                )

        # Convert wind speed to knots if necessary
        speed = None
        if fields[3]:
            try:
                # Clean up the speed value by taking only the first decimal number if multiple exist
                speed_parts = fields[3].split(".")
                speed_str = speed_parts[0]
                if len(speed_parts) > 1:
                    speed_str += "." + speed_parts[1]
                speed = float(speed_str)

                if fields[4]:
                    if fields[4] == "K":
                        speed = speed * 0.539957  # km/h to knots
                    elif fields[4] == "M":
                        speed = speed * 1.94384  # m/s to knots
            except (ValueError, IndexError):
                raise NMEA0183Error(
                    f"Invalid MWV sentence: invalid wind speed value '{fields[3]}'"
                )

        return cls(
            talker_id=base.talker_id,
            sentence_type=base.sentence_type,
            raw=raw_sentence,
            wind_angle=wind_angle,
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


@dataclass
class VWRSentence(NMEASentence):
    """VWR - Relative Wind Speed and Angle

    Format: $--VWR,x.x,a,x.x,N,x.x,M,x.x,K*hh
    Example: $IIVWR,022,R,23.2,N,11.9,M,,K*73

    Fields:
        wind_angle: Wind angle in degrees from 0-180
        wind_direction: L=port, R=starboard
        wind_speed_knots: Wind speed in knots
        wind_speed_ms: Wind speed in meters per second
        wind_speed_kmh: Wind speed in kilometers per hour
    """

    wind_angle: Optional[float] = None
    wind_direction: Optional[str] = None
    wind_speed_knots: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_speed_kmh: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VWRSentence":
        """Parse a raw VWR sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        def clean_float(value: str) -> Optional[float]:
            """Clean and convert string to float, handling malformed decimals."""
            if not value:
                return None
            try:
                # Clean up the value by taking only the first decimal number if multiple exist
                parts = value.split(".")
                clean_str = parts[0]
                if len(parts) > 1:
                    clean_str += "." + parts[1]
                return float(clean_str)
            except (ValueError, IndexError):
                raise NMEA0183Error(
                    f"Invalid VWR sentence: invalid numeric value '{value}'"
                )

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            wind_angle=clean_float(fields[0]),
            wind_direction=fields[1] if fields[1] else None,
            wind_speed_knots=clean_float(fields[2]),
            wind_speed_ms=clean_float(fields[4]),
            wind_speed_kmh=clean_float(fields[6]),
        )


@dataclass
class RMBSentence(NMEASentence):
    """RMB - Recommended Minimum Navigation Information

    Format: $--RMB,A,x.x,a,c--c,c--c,llll.ll,a,yyyyy.yy,a,x.x,x.x,x.x,A*hh
    Example: $GPRMB,A,0.34,R,,3XISC,5046.100,N,00118.430,W,001.0,197.1,004.8,V*05
    """

    data_status: Optional[str] = None  # A=valid, V=warning
    xte: Optional[float] = None  # Cross-track error in nm
    xte_direction: Optional[str] = None  # L/R = left/right of course
    origin_id: Optional[str] = None  # Origin waypoint ID
    dest_id: Optional[str] = None  # Destination waypoint ID
    dest_lat: Optional[float] = None  # Destination latitude
    dest_lat_dir: Optional[str] = None  # N/S
    dest_lon: Optional[float] = None  # Destination longitude
    dest_lon_dir: Optional[str] = None  # E/W
    range_to_dest: Optional[float] = None  # Range to destination in nm
    bearing_to_dest: Optional[float] = None  # Bearing to destination in degrees
    velocity_to_dest: Optional[float] = None  # Velocity towards destination in knots
    arrival_status: Optional[str] = None  # A=arrived, V=not arrived

    @classmethod
    def parse(cls, raw_sentence: str) -> "RMBSentence":
        """Parse a raw RMB sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Convert latitude/longitude to decimal degrees if present
        lat = None
        if fields[5]:  # Destination latitude
            lat = float(fields[5][:2]) + float(fields[5][2:]) / 60.0
            if fields[6] == "S":  # Southern hemisphere
                lat = -lat

        lon = None
        if fields[7]:  # Destination longitude
            lon = float(fields[7][:3]) + float(fields[7][3:]) / 60.0
            if fields[8] == "W":  # Western hemisphere
                lon = -lon

        # Parse all fields
        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            data_status=fields[0],
            xte=float(fields[1]) if fields[1] else None,
            xte_direction=fields[2],
            origin_id=fields[3],
            dest_id=fields[4],
            dest_lat=lat,
            dest_lat_dir=fields[6],
            dest_lon=lon,
            dest_lon_dir=fields[8],
            range_to_dest=float(fields[9]) if fields[9] else None,
            bearing_to_dest=float(fields[10]) if fields[10] else None,
            velocity_to_dest=float(fields[11]) if fields[11] else None,
            arrival_status=fields[12].split("*")[0] if len(fields) > 12 else None,
        )


@dataclass
class VWTSentence(NMEASentence):
    """VWT - True Wind Speed and Angle

    Format: $--VWT,x.x,a,x.x,N,x.x,M,x.x,K*hh
    Example: $IIVWT,031,R,19.2,N,9.8,M,,K*46

    Fields:
        wind_angle: True wind angle in degrees from 0-180
        wind_direction: L=port, R=starboard relative to bow
        wind_speed_knots: True wind speed in knots
        wind_speed_ms: True wind speed in meters per second
        wind_speed_kmh: True wind speed in kilometers per hour
    """

    wind_angle: Optional[float] = None
    wind_direction: Optional[str] = None
    wind_speed_knots: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_speed_kmh: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VWTSentence":
        """Parse a raw VWT sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Parse specific VWT fields
        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            wind_angle=float(fields[0]) if fields[0] else None,
            wind_direction=fields[1] if fields[1] else None,
            wind_speed_knots=float(fields[2]) if fields[2] else None,
            wind_speed_ms=float(fields[4]) if fields[4] else None,
            wind_speed_kmh=float(fields[6]) if fields[6] else None,
        )


@dataclass
class GGASentence(NMEASentence):
    """GGA - Global Positioning System Fix Data

    Format: $--GGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
    Example: $GPGGA,174000,5047.057,N,00117.967,W,1,11,0.8,1.9,M,48.5,M,,*50
    """

    timestamp: Optional[datetime] = None  # UTC time
    latitude: Optional[float] = None  # Decimal degrees
    longitude: Optional[float] = None  # Decimal degrees
    fix_quality: Optional[int] = None  # GPS quality indicator
    num_satellites: Optional[int] = None  # Number of satellites in use
    hdop: Optional[float] = None  # Horizontal dilution of precision
    altitude: Optional[float] = None  # Antenna altitude above mean sea level
    altitude_units: Optional[str] = None  # Units of antenna altitude
    geoid_sep: Optional[float] = None  # Geoidal separation
    geoid_units: Optional[str] = None  # Units of geoidal separation
    age_dgps: Optional[float] = None  # Age of differential correction
    ref_station_id: Optional[str] = None  # DGPS reference station ID

    @classmethod
    def parse(cls, raw_sentence: str) -> "GGASentence":
        """Parse a raw GGA sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Parse timestamp if present
        timestamp = None
        if fields[0]:
            time_str = fields[0]
            hours = int(time_str[0:2])
            minutes = int(time_str[2:4])
            seconds = int(time_str[4:6])
            timestamp = datetime.now().replace(
                hour=hours, minute=minutes, second=seconds, microsecond=0
            )

        # Parse latitude
        lat = None
        if fields[1]:
            lat = float(fields[1][:2]) + float(fields[1][2:]) / 60.0
            if fields[2] == "S":  # Southern hemisphere
                lat = -lat

        # Parse longitude
        lon = None
        if fields[3]:
            lon = float(fields[3][:3]) + float(fields[3][3:]) / 60.0
            if fields[4] == "W":  # Western hemisphere
                lon = -lon

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
            fix_quality=int(fields[5]) if fields[5] else None,
            num_satellites=int(fields[6]) if fields[6] else None,
            hdop=float(fields[7]) if fields[7] else None,
            altitude=float(fields[8]) if fields[8] else None,
            altitude_units=fields[9] if fields[9] else None,
            geoid_sep=float(fields[10]) if fields[10] else None,
            geoid_units=fields[11] if fields[11] else None,
            age_dgps=float(fields[12]) if fields[12] else None,
            ref_station_id=fields[13].split("*")[0] if fields[13] else None,
        )


@dataclass
class GSASentence(NMEASentence):
    """GSA - GPS DOP and Active Satellites

    Format: $--GSA,a,x,xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,x.x,x.x,x.x*hh
    Example: $GPGSA,A,3,07,08,10,11,,15,16,18,21,26,27,30,1.2,0.8,1.0*3D
    """

    mode: Optional[str] = None  # M=Manual, A=Automatic
    fix_type: Optional[int] = None  # 1=no fix, 2=2D fix, 3=3D fix
    sat_prns: List[str] = field(
        default_factory=list
    )  # List of PRNs of satellites used in solution
    pdop: Optional[float] = None  # Position Dilution of Precision
    hdop: Optional[float] = None  # Horizontal Dilution of Precision
    vdop: Optional[float] = None  # Vertical Dilution of Precision

    @classmethod
    def parse(cls, raw_sentence: str) -> "GSASentence":
        """Parse a raw GSA sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Extract satellite PRNs (fields 2-13)
        sat_prns = [prn for prn in fields[2:14] if prn]

        # Parse remaining fields after removing checksum from last field
        last_field = fields[14].split("*")[0] if "*" in fields[14] else fields[14]

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            mode=fields[0],
            fix_type=int(fields[1]) if fields[1] else None,
            sat_prns=sat_prns,
            pdop=float(fields[14]) if fields[14] else None,
            hdop=float(fields[15]) if fields[15] else None,
            vdop=float(fields[16].split("*")[0]) if fields[16] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable representation of the GSA sentence."""
        fix_types = {1: "no fix", 2: "2D fix", 3: "3D fix"}
        mode_desc = {"M": "Manual", "A": "Automatic"}

        return (
            f"GSA: {mode_desc.get(self.mode, self.mode)} mode, "
            f"{fix_types.get(self.fix_type, 'unknown')} using {len(self.sat_prns)} satellites. "
            f"DOP - Position: {self.pdop}, Horizontal: {self.hdop}, Vertical: {self.vdop}"
        )


@dataclass
class XTESentence(NMEASentence):
    """XTE - Cross-Track Error

    Format: $--XTE,A,A,x.x,a,N,a*hh
    Example: $GPXTE,A,A,0.67,L,N,A*12

    Fields:
        status_general: Status A=Valid, V=Invalid (Loran-C Blink or SNR warning)
        status_warning: Status A=Valid, V=Invalid
        cross_track_error: Cross-track error distance
        direction: Direction to steer, L=Left, R=Right
        units: Units of cross-track error (N=Nautical Miles)
        mode: Mode indicator (A=Autonomous, D=Differential, E=Estimated,
              M=Manual Input, S=Simulator, N=Data Not Valid)
    """

    status_general: Optional[str] = None
    status_warning: Optional[str] = None
    cross_track_error: Optional[float] = None
    direction: Optional[str] = None
    units: Optional[str] = None
    mode: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "XTESentence":
        """Parse a raw XTE sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Ensure minimum number of fields (5 required, 6th field is optional)
        if len(fields) < 5:
            raise NMEA0183Error("Invalid XTE sentence: insufficient fields")

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            status_general=fields[0],
            status_warning=fields[1],
            cross_track_error=float(fields[2]) if fields[2] else None,
            direction=fields[3],
            units=fields[4],
            mode=fields[5] if len(fields) > 5 else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        status = "Valid" if self.status_general == "A" else "Invalid"
        direction = "Left" if self.direction == "L" else "Right"
        return (
            f"XTE: {status}, {self.cross_track_error} NM "
            f"off track, steer {direction}, Mode: {self.mode}"
        )


@dataclass
class SatelliteInfo:
    """Information about a single satellite in view."""

    prn: int  # PRN - Satellite identification number
    elevation: Optional[int] = None  # Elevation in degrees (0-90)
    azimuth: Optional[int] = None  # Azimuth in degrees (0-359)
    snr: Optional[int] = None  # Signal-to-Noise ratio in dB (0-99)


@dataclass
class GSVSentence(NMEASentence):
    """GSV - GPS Satellites in View

    Format: $--GSV,x,x,xx,xx,xx,xxx,xx,...*hh
    Example: $GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        total_messages: Total number of GSV messages in this cycle (1-9)
        message_number: Message number in current cycle (1-9)
        satellites_in_view: Total number of satellites in view
        satellites: List of SatelliteInfo objects containing data for each satellite
    """

    # All fields from parent class must be redeclared in the same order
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None
    # GSV specific fields with default values
    total_messages: int = field(default=0)
    message_number: int = field(default=0)
    satellites_in_view: int = field(default=0)
    satellites: List[SatelliteInfo] = field(default_factory=list)

    @classmethod
    def parse(cls, raw_sentence: str) -> "GSVSentence":
        """Parse a raw GSV sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 3 fields (total_messages, message_number, sats_in_view)
        if len(fields) < 3:
            raise NMEA0183Error("Invalid GSV sentence: insufficient fields")

        # Parse the header fields
        total_messages = int(fields[0])
        message_number = int(fields[1])
        satellites_in_view = int(fields[2])

        # Process satellite data blocks (each block is 4 fields)
        satellites = []
        for i in range(3, len(fields), 4):
            # Check if we have a complete block
            if i + 3 >= len(fields):
                break

            # Only process if PRN is present (first field in block)
            if fields[i]:
                try:
                    sat = SatelliteInfo(
                        prn=int(fields[i]),
                        elevation=int(fields[i + 1]) if fields[i + 1] else None,
                        azimuth=int(fields[i + 2]) if fields[i + 2] else None,
                        snr=int(fields[i + 3]) if fields[i + 3] else None,
                    )
                    satellites.append(sat)
                except (ValueError, TypeError):
                    # Skip invalid satellite blocks
                    continue

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            total_messages=total_messages,
            message_number=message_number,
            satellites_in_view=satellites_in_view,
            satellites=satellites,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        sats_str = [
            f"PRN:{s.prn} El:{s.elevation}° Az:{s.azimuth}° SNR:{s.snr}dB"
            for s in self.satellites
        ]
        return (
            f"GSV: Message {self.message_number}/{self.total_messages}, "
            f"{self.satellites_in_view} satellites in view. "
            f"This message: {', '.join(sats_str)}"
        )


@dataclass
class RMESentence(NMEASentence):
    """RME - Estimated Error Information

    Format: $--RME,x.x,M,x.x,M,x.x,M*hh
    Example: $PGRME,3.1,M,4.1,M,5.1,M*2D

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        horizontal_error: Estimated horizontal position error in meters
        horizontal_units: Units for horizontal error (typically 'M' for meters)
        vertical_error: Estimated vertical error in meters
        vertical_units: Units for vertical error (typically 'M' for meters)
        overall_spherical_error: Estimated overall spherical equivalent error
        spherical_units: Units for spherical error (typically 'M' for meters)
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # RME specific fields with defaults
    horizontal_error: Optional[float] = None
    horizontal_units: Optional[str] = None
    vertical_error: Optional[float] = None
    vertical_units: Optional[str] = None
    overall_spherical_error: Optional[float] = None
    spherical_units: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "RMESentence":
        """Parse a raw RME sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 6 fields (3 pairs of values and units)
        if len(fields) < 6:
            raise NMEA0183Error("Invalid RME sentence: insufficient fields")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            horizontal_error=float(fields[0]) if fields[0] else None,
            horizontal_units=fields[1] if fields[1] else None,
            vertical_error=float(fields[2]) if fields[2] else None,
            vertical_units=fields[3] if fields[3] else None,
            overall_spherical_error=float(fields[4]) if fields[4] else None,
            spherical_units=fields[5] if fields[5] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"RME: Horizontal Error: {self.horizontal_error} {self.horizontal_units}, "
            f"Vertical Error: {self.vertical_error} {self.vertical_units}, "
            f"Spherical Error: {self.overall_spherical_error} {self.spherical_units}"
        )


@dataclass
class GLLSentence(NMEASentence):
    """GLL - Geographic Position - Latitude/Longitude

    Format: $--GLL,llll.ll,a,yyyyy.yy,a,hhmmss.ss,A,A*hh
    Example: $GPGLL,5047.056,N,00117.967,W,174001,A*31

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        status: Data status (A = Valid, V = Invalid)
        mode: Mode indicator (A = Autonomous, D = Differential, etc)
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # GLL specific fields with defaults
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: Optional[str] = None
    mode: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "GLLSentence":
        """Parse a raw GLL sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 6 fields for a valid GLL sentence
        if len(fields) < 6:
            raise NMEA0183Error("Invalid GLL sentence: insufficient fields")

        # Parse timestamp if present
        timestamp = None
        if fields[4]:
            try:
                time_str = fields[4]
                hours = int(time_str[0:2])
                minutes = int(time_str[2:4])
                seconds = int(time_str[4:6])
                timestamp = datetime.now().replace(
                    hour=hours, minute=minutes, second=seconds, microsecond=0
                )
            except (ValueError, IndexError):
                pass

        # Parse latitude and longitude if present
        lat = lon = None
        if fields[0] and fields[1]:
            try:
                lat = float(fields[0][:2]) + float(fields[0][2:]) / 60.0
                if fields[1] == "S":
                    lat = -lat
            except (ValueError, IndexError):
                pass

        if fields[2] and fields[3]:
            try:
                lon = float(fields[2][:3]) + float(fields[2][3:]) / 60.0
                if fields[3] == "W":
                    lon = -lon
            except (ValueError, IndexError):
                pass

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=timestamp,
            latitude=lat,
            longitude=lon,
            status=fields[5] if len(fields) > 5 else None,
            mode=fields[6] if len(fields) > 6 else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        pos = "Unknown"
        if self.latitude is not None and self.longitude is not None:
            pos = f"{abs(self.latitude):.4f}°{'S' if self.latitude < 0 else 'N'}, "
            pos += f"{abs(self.longitude):.4f}°{'W' if self.longitude < 0 else 'E'}"
        return f"GLL: Position: {pos}, Status: {self.status}"


@dataclass
class RMZSentence(NMEASentence):
    """RMZ - NMEA 0183 Altitude Information

    Format: $--RMZ,x.x,f,x.x,F,x.x,F*hh
    Example: $PGRMZ,1494,f,,,,*10

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        primary_altitude: Primary altitude value
        primary_units: Primary altitude units (typically 'f' for feet)
        middle_altitude: Middle altitude value (if available)
        middle_units: Middle altitude units (typically 'F' for feet)
        last_altitude: Last altitude value (if available)
        last_units: Last altitude units (typically 'F' for feet)
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # RMZ specific fields with defaults
    primary_altitude: Optional[float] = None
    primary_units: Optional[str] = None
    middle_altitude: Optional[float] = None
    middle_units: Optional[str] = None
    last_altitude: Optional[float] = None
    last_units: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "RMZSentence":
        """Parse a raw RMZ sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 2 fields (altitude and units)
        if len(fields) < 2:
            raise NMEA0183Error("Invalid RMZ sentence: insufficient fields")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            primary_altitude=float(fields[0]) if fields[0] else None,
            primary_units=fields[1] if fields[1] else None,
            middle_altitude=float(fields[2]) if len(fields) > 2 and fields[2] else None,
            middle_units=fields[3] if len(fields) > 3 and fields[3] else None,
            last_altitude=float(fields[4]) if len(fields) > 4 and fields[4] else None,
            last_units=fields[5] if len(fields) > 5 and fields[5] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        altitude_str = []
        if self.primary_altitude is not None:
            altitude_str.append(
                f"Primary: {self.primary_altitude} {self.primary_units}"
            )
        if self.middle_altitude is not None:
            altitude_str.append(f"Middle: {self.middle_altitude} {self.middle_units}")
        if self.last_altitude is not None:
            altitude_str.append(f"Last: {self.last_altitude} {self.last_units}")

        return f"RMZ: {', '.join(altitude_str) if altitude_str else 'No altitude data'}"

    @property
    def altitude_meters(self) -> Optional[float]:
        """Return the primary altitude converted to meters if available."""
        if self.primary_altitude is None:
            return None
        if self.primary_units and self.primary_units.lower() == "f":
            return self.primary_altitude * 0.3048  # Convert feet to meters
        return self.primary_altitude


@dataclass
class RMMSentence(NMEASentence):
    """RMM - MAP Datum Information

    Format: $--RMM,c--c*hh
    Example: $PGRMM,WGS84*00

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        map_datum: Map datum name (e.g., "WGS84", "NAD83")
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # RMM specific fields with defaults
    map_datum: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "RMMSentence":
        """Parse a raw RMM sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 1 field (map datum)
        if len(fields) < 1:
            raise NMEA0183Error("Invalid RMM sentence: insufficient fields")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            map_datum=fields[0] if fields[0] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"RMM: Map Datum: {self.map_datum if self.map_datum else 'Unknown'}"


@dataclass
class BODSentence(NMEASentence):
    """BOD - Bearing Origin to Destination

    Format: $--BOD,x.x,T,x.x,M,c--c,c--c*hh
    Example: $GPBOD,234.9,T,228.8,M,WAYP1,WAYP2*45

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        bearing_true: Bearing from origin to destination, true
        bearing_mag: Bearing from origin to destination, magnetic
        origin_id: Origin waypoint ID
        dest_id: Destination waypoint ID
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # BOD specific fields with defaults
    bearing_true: Optional[float] = None
    bearing_mag: Optional[float] = None
    origin_id: Optional[str] = None
    dest_id: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "BODSentence":
        """Parse a raw BOD sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 6 fields
        if len(fields) < 6:
            raise NMEA0183Error("Invalid BOD sentence: insufficient fields")

        # Validate bearing unit indicators
        if fields[1] != "T" or fields[3] != "M":
            raise NMEA0183Error("Invalid BOD sentence: invalid bearing unit indicators")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            bearing_true=float(fields[0]) if fields[0] else None,
            bearing_mag=float(fields[2]) if fields[2] else None,
            origin_id=fields[4] if fields[4] else None,
            dest_id=fields[5] if fields[5] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        bearings = []
        if self.bearing_true is not None:
            bearings.append(f"{self.bearing_true}°T")
        if self.bearing_mag is not None:
            bearings.append(f"{self.bearing_mag}°M")

        waypoints = []
        if self.origin_id:
            waypoints.append(f"from {self.origin_id}")
        if self.dest_id:
            waypoints.append(f"to {self.dest_id}")

        return (
            f"BOD: Bearing {' / '.join(bearings) if bearings else 'unknown'} "
            f"{' '.join(waypoints) if waypoints else '(no waypoints)'}"
        )


@dataclass
class RTESentence(NMEASentence):
    """RTE - Routes

    Format: $--RTE,x,x,a,c--c,c--c,....*hh
    Example: $GPRTE,2,1,c,0,WAYP1,WAYP2,WAYP3*XX

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        total_messages: Total number of messages for this route
        message_number: Message number for this sentence
        route_type: Type of route (c = complete route, w = working route)
        route_id: Route identifier (if provided)
        waypoints: List of waypoint IDs in the route
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # RTE specific fields with defaults
    total_messages: int = field(default=1)
    message_number: int = field(default=1)
    route_type: Optional[str] = None
    route_id: Optional[str] = None
    waypoints: List[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw_sentence: str) -> "RTESentence":
        """Parse a raw RTE sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 4 fields (total msgs, msg num, type, and at least one waypoint/route id)
        if len(fields) < 4:
            raise NMEA0183Error("Invalid RTE sentence: insufficient fields")

        # Validate route type
        route_type = fields[2].lower() if fields[2] else None
        if route_type not in ("c", "w"):
            raise NMEA0183Error("Invalid RTE sentence: invalid route type")

        # First waypoint field might be route ID or first waypoint
        route_id = fields[3] if len(fields) > 3 else None

        # Remaining fields are waypoints
        waypoints = fields[4:] if fields[4:] else []

        # Remove empty waypoints
        waypoints = [wp for wp in waypoints if wp]

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            total_messages=int(fields[0]) if fields[0] else 1,
            message_number=int(fields[1]) if fields[1] else 1,
            route_type=route_type,
            route_id=route_id,
            waypoints=waypoints,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        route_type_desc = {"c": "Complete", "w": "Working"}.get(
            self.route_type, "Unknown"
        )

        msg_count = f"Message {self.message_number}/{self.total_messages}"
        route_info = f"Route {self.route_id}" if self.route_id else "Unnamed route"
        waypoint_info = (
            f"{len(self.waypoints)} waypoints" if self.waypoints else "no waypoints"
        )

        return (
            f"RTE: {msg_count}, {route_type_desc} {route_info}, "
            f"{waypoint_info}: {', '.join(self.waypoints)}"
        )


@dataclass
class WPLSentence(NMEASentence):
    """WPL - Waypoint Location

    Format: $--WPL,llll.ll,a,yyyyy.yy,a,c--c*hh
    Example: $GPWPL,5047.057,N,00117.967,W,WAYPT1*52

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        waypoint_id: Waypoint identifier
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # WPL specific fields with defaults
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    waypoint_id: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "WPLSentence":
        """Parse a raw WPL sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 5 fields (lat, lat dir, lon, lon dir, waypoint id)
        if len(fields) < 5:
            raise NMEA0183Error("Invalid WPL sentence: insufficient fields")

        # Parse latitude and longitude
        lat = lon = None
        try:
            if fields[0] and fields[1]:
                lat = float(fields[0][:2]) + float(fields[0][2:]) / 60.0
                if fields[1] == "S":
                    lat = -lat

            if fields[2] and fields[3]:
                lon = float(fields[2][:3]) + float(fields[2][3:]) / 60.0
                if fields[3] == "W":
                    lon = -lon
        except (ValueError, IndexError) as e:
            raise NMEA0183Error(f"Invalid WPL coordinates: {str(e)}")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            latitude=lat,
            longitude=lon,
            waypoint_id=fields[4] if fields[4] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        pos = (
            f"{abs(self.latitude):.4f}°{'S' if self.latitude < 0 else 'N'}, "
            f"{abs(self.longitude):.4f}°{'W' if self.longitude < 0 else 'E'}"
            if self.latitude is not None and self.longitude is not None
            else "Unknown position"
        )

        return f"WPL: Waypoint {self.waypoint_id or 'Unknown'} at {pos}"


@dataclass
class DBTSentence(NMEASentence):
    """DBT - Depth Below Transducer

    Format: $--DBT,x.x,f,x.x,M,x.x,F*hh
    Example: $IIDBT,032.8,f,010.0,M,005.4,F*27

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        depth_feet: Depth in feet
        depth_meters: Depth in meters
        depth_fathoms: Depth in fathoms
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # DBT specific fields with defaults
    depth_feet: Optional[float] = None
    depth_meters: Optional[float] = None
    depth_fathoms: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "DBTSentence":
        """Parse a raw DBT sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 6 fields (3 depths with their units)
        if len(fields) < 6:
            raise NMEA0183Error("Invalid DBT sentence: insufficient fields")

        # Validate unit indicators
        if fields[1] != "f" or fields[3] != "M" or fields[5] != "F":
            raise NMEA0183Error("Invalid DBT sentence: invalid unit indicators")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            depth_feet=float(fields[0]) if fields[0] else None,
            depth_meters=float(fields[2]) if fields[2] else None,
            depth_fathoms=float(fields[4]) if fields[4] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        depths = []
        if self.depth_meters is not None:
            depths.append(f"{self.depth_meters:.1f}m")
        if self.depth_feet is not None:
            depths.append(f"{self.depth_feet:.1f}ft")
        if self.depth_fathoms is not None:
            depths.append(f"{self.depth_fathoms:.1f}F")

        return (
            f"DBT: Depth Below Transducer {' / '.join(depths) if depths else 'unknown'}"
        )


@dataclass
class HDMSentence(NMEASentence):
    """HDM - Heading Magnetic

    Format: $--HDM,x.x,M*hh
    Example: $IIHDM,245.1,M*25

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        heading: Magnetic heading in degrees
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # HDM specific fields with defaults
    heading: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "HDMSentence":
        """Parse a raw HDM sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need exactly 2 fields (heading and unit indicator)
        if len(fields) != 2:
            raise NMEA0183Error("Invalid HDM sentence: wrong number of fields")

        # Validate unit indicator is 'M' for Magnetic
        if fields[1] != "M":
            raise NMEA0183Error("Invalid HDM sentence: unit must be 'M' for Magnetic")

        # Parse heading
        heading = None
        if fields[0]:
            try:
                heading = float(fields[0])
                # Validate heading range (0-360 degrees)
                if not (0 <= heading <= 360):
                    raise NMEA0183Error(
                        "Invalid HDM sentence: heading must be between 0 and 360 degrees"
                    )
            except ValueError:
                raise NMEA0183Error("Invalid HDM sentence: invalid heading value")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            heading=heading,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self.heading is not None:
            return f"HDM: Magnetic Heading {self.heading:.1f}°"
        return "HDM: Heading unknown"


@dataclass
class MTWSentence(NMEASentence):
    """MTW - Mean Temperature of Water

    Format: $--MTW,x.x,C*hh
    Example: $IIMTW,18.5,C*11

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        temperature: Water temperature in degrees Celsius
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # MTW specific fields with defaults
    temperature: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "MTWSentence":
        """Parse a raw MTW sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need exactly 2 fields (temperature and unit indicator)
        if len(fields) != 2:
            raise NMEA0183Error("Invalid MTW sentence: wrong number of fields")

        # Validate unit indicator is 'C' for Celsius
        if fields[1] != "C":
            raise NMEA0183Error("Invalid MTW sentence: unit must be 'C' for Celsius")

        # Parse temperature
        temperature = None
        if fields[0]:
            try:
                temperature = float(fields[0])
            except ValueError:
                raise NMEA0183Error("Invalid MTW sentence: invalid temperature value")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            temperature=temperature,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self.temperature is not None:
            return f"MTW: Water Temperature {self.temperature:.1f}°C"
        return "MTW: Temperature unknown"

    @property
    def temperature_fahrenheit(self) -> Optional[float]:
        """Return temperature in Fahrenheit if available."""
        if self.temperature is None:
            return None
        return (self.temperature * 9 / 5) + 32


@dataclass
class MWDSentence(NMEASentence):
    """MWD - Wind Direction and Speed

    Format: $--MWD,x.x,T,x.x,M,x.x,N,x.x,M*hh
    Example: $IIMWD,180.0,T,185.0,M,12.8,N,6.6,M*52

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        direction_true: Wind direction in degrees True
        direction_magnetic: Wind direction in degrees Magnetic
        speed_knots: Wind speed in knots
        speed_mps: Wind speed in meters per second
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # MWD specific fields with defaults
    direction_true: Optional[float] = None
    direction_magnetic: Optional[float] = None
    speed_knots: Optional[float] = None
    speed_mps: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "MWDSentence":
        """Parse a raw MWD sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need exactly 8 fields (dir true, T, dir mag, M, speed knots, N, speed m/s, M)
        if len(fields) != 8:
            raise NMEA0183Error("Invalid MWD sentence: wrong number of fields")

        # Validate unit indicators
        if fields[1] != "T" or fields[3] != "M" or fields[5] != "N" or fields[7] != "M":
            raise NMEA0183Error("Invalid MWD sentence: invalid unit indicators")

        # Parse direction values and validate range (0-360)
        dir_true = dir_mag = None
        if fields[0]:
            try:
                dir_true = float(fields[0])
                if not (0 <= dir_true <= 360):
                    raise NMEA0183Error(
                        "Invalid MWD sentence: direction must be between 0 and 360 degrees"
                    )
            except ValueError:
                raise NMEA0183Error(
                    "Invalid MWD sentence: invalid true direction value"
                )

        if fields[2]:
            try:
                dir_mag = float(fields[2])
                if not (0 <= dir_mag <= 360):
                    raise NMEA0183Error(
                        "Invalid MWD sentence: direction must be between 0 and 360 degrees"
                    )
            except ValueError:
                raise NMEA0183Error(
                    "Invalid MWD sentence: invalid magnetic direction value"
                )

        # Parse speed values
        speed_knots = speed_mps = None
        if fields[4]:
            try:
                speed_knots = float(fields[4])
                if speed_knots < 0:
                    raise NMEA0183Error(
                        "Invalid MWD sentence: speed cannot be negative"
                    )
            except ValueError:
                raise NMEA0183Error("Invalid MWD sentence: invalid speed value")

        if fields[6]:
            try:
                speed_mps = float(fields[6])
                if speed_mps < 0:
                    raise NMEA0183Error(
                        "Invalid MWD sentence: speed cannot be negative"
                    )
            except ValueError:
                raise NMEA0183Error("Invalid MWD sentence: invalid speed value")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            direction_true=dir_true,
            direction_magnetic=dir_mag,
            speed_knots=speed_knots,
            speed_mps=speed_mps,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        parts = []

        # Add directions
        if self.direction_true is not None:
            parts.append(f"{self.direction_true:.1f}°T")
        if self.direction_magnetic is not None:
            parts.append(f"{self.direction_magnetic:.1f}°M")

        # Add speeds
        speeds = []
        if self.speed_knots is not None:
            speeds.append(f"{self.speed_knots:.1f}kts")
        if self.speed_mps is not None:
            speeds.append(f"{self.speed_mps:.1f}m/s")

        return (
            f"MWD: {' / '.join(parts)} @ {' / '.join(speeds)}"
            if parts and speeds
            else "MWD: No wind data"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        parts = []

        # Add directions
        if self.direction_true is not None:
            parts.append(f"{self.direction_true:.1f}°T")
        if self.direction_magnetic is not None:
            parts.append(f"{self.direction_magnetic:.1f}°M")

        # Add speeds
        speeds = []
        if self.speed_knots is not None:
            speeds.append(f"{self.speed_knots:.1f}kts")
        if self.speed_mps is not None:
            speeds.append(f"{self.speed_mps:.1f}m/s")

        return (
            f"MWD: {' / '.join(parts)} @ {' / '.join(speeds)}"
            if parts and speeds
            else "MWD: No wind data"
        )


@dataclass
class GBSSentence(NMEASentence):
    """GBS - GNSS Satellite Fault Detection

    Format: $--GBS,hhmmss.ss,x.x,x.x,x.x,x.x,x.x,x.x,x.x[,h[,h]]*hh
    Example: $GPGBS,235458.00,1.4,1.3,3.1,03,,-21.4,3.8,1,0*5B

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: UTC time of the position fix
        lat_error: Expected error in latitude (meters)
        lon_error: Expected error in longitude (meters)
        alt_error: Expected error in altitude (meters)
        failed_satellite: ID of the most likely failed satellite
        prob_missed_detection: Probability of missed detection
        bias_estimate: Estimate of bias on most likely failed satellite (meters)
        bias_std_dev: Standard deviation of bias estimate
        system_id: System ID (optional)
        signal_id: Signal ID (optional)
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # GBS specific fields with defaults
    lat_error: Optional[float] = None
    lon_error: Optional[float] = None
    alt_error: Optional[float] = None
    failed_satellite: Optional[int] = None
    prob_missed_detection: Optional[float] = None
    bias_estimate: Optional[float] = None
    bias_std_dev: Optional[float] = None
    system_id: Optional[int] = None
    signal_id: Optional[int] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "GBSSentence":
        """Parse a raw GBS sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 8 fields for basic GBS sentence
        if len(fields) < 8:
            raise NMEA0183Error("Invalid GBS sentence: insufficient fields")

        # Parse timestamp if present
        timestamp = None
        if fields[0]:
            try:
                time_str = fields[0]
                hours = int(time_str[0:2])
                minutes = int(time_str[2:4])
                seconds = int(time_str[4:6])
                timestamp = datetime.now().replace(
                    hour=hours, minute=minutes, second=seconds, microsecond=0
                )
            except (ValueError, IndexError):
                pass

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=timestamp,
            lat_error=float(fields[1]) if fields[1] else None,
            lon_error=float(fields[2]) if fields[2] else None,
            alt_error=float(fields[3]) if fields[3] else None,
            failed_satellite=int(fields[4]) if fields[4] else None,
            prob_missed_detection=float(fields[5]) if fields[5] else None,
            bias_estimate=float(fields[6]) if fields[6] else None,
            bias_std_dev=float(fields[7]) if fields[7] else None,
            system_id=int(fields[8]) if len(fields) > 8 and fields[8] else None,
            signal_id=int(fields[9]) if len(fields) > 9 and fields[9] else None,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        parts = []

        if self.timestamp is not None:
            parts.append(f"Time: {self.timestamp.strftime('%H:%M:%S')}")

        errors = []
        if self.lat_error is not None:
            errors.append(f"Lat: {self.lat_error:.1f}m")
        if self.lon_error is not None:
            errors.append(f"Lon: {self.lon_error:.1f}m")
        if self.alt_error is not None:
            errors.append(f"Alt: {self.alt_error:.1f}m")
        if errors:
            parts.append(f"Errors ({', '.join(errors)})")

        if self.failed_satellite is not None:
            parts.append(f"Failed Satellite: {self.failed_satellite}")
            if self.bias_estimate is not None:
                parts.append(f"Bias: {self.bias_estimate:.1f}m")
            if self.bias_std_dev is not None:
                parts.append(f"Std Dev: {self.bias_std_dev:.1f}m")

        if self.system_id is not None:
            parts.append(f"System ID: {self.system_id}")
        if self.signal_id is not None:
            parts.append(f"Signal ID: {self.signal_id}")

        return f"GBS: {' | '.join(parts)}"


@dataclass
class VLWSentence(NMEASentence):
    """VLW - Distance Traveled through Water

    Format: $--VLW,x.x,N,x.x,N*hh
    Example: $IIVLW,12.34,N,23.45,N*7D

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        total_distance: Total cumulative distance through water since installation (nm)
        trip_distance: Distance through water since reset (nm)
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # VLW specific fields with defaults
    total_distance: Optional[float] = None
    trip_distance: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "VLWSentence":
        """Parse a raw VLW sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need exactly 4 fields (total distance, N, trip distance, N)
        if len(fields) != 4:
            raise NMEA0183Error("Invalid VLW sentence: wrong number of fields")

        # Validate unit indicators are 'N' for nautical miles
        if fields[1] != "N" or fields[3] != "N":
            raise NMEA0183Error(
                "Invalid VLW sentence: distances must be in nautical miles (N)"
            )

        # Parse distance values
        total_distance = None
        if fields[0]:
            try:
                total_distance = float(fields[0])
                if total_distance < 0:
                    raise NMEA0183Error(
                        "Invalid VLW sentence: total distance cannot be negative"
                    )
            except ValueError:
                raise NMEA0183Error(
                    "Invalid VLW sentence: invalid total distance value"
                )

        trip_distance = None
        if fields[2]:
            try:
                trip_distance = float(fields[2])
                if trip_distance < 0:
                    raise NMEA0183Error(
                        "Invalid VLW sentence: trip distance cannot be negative"
                    )
            except ValueError:
                raise NMEA0183Error("Invalid VLW sentence: invalid trip distance value")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            total_distance=total_distance,
            trip_distance=trip_distance,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        parts = []

        if self.total_distance is not None:
            parts.append(f"Total: {self.total_distance:.2f}nm")
        if self.trip_distance is not None:
            parts.append(f"Trip: {self.trip_distance:.2f}nm")

        return f"VLW: {' | '.join(parts)}" if parts else "VLW: No distance data"


@dataclass
class MTASentence(NMEASentence):
    """MTA - Air Temperature

    Format: $--MTA,x.x,C*hh
    Example: $IIMTA,24.5,C*12

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        temperature: Air temperature in degrees Celsius
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # MTA specific fields with defaults
    temperature: Optional[float] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "MTASentence":
        """Parse a raw MTA sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need exactly 2 fields (temperature and unit indicator)
        if len(fields) != 2:
            raise NMEA0183Error("Invalid MTA sentence: wrong number of fields")

        # Validate unit indicator is 'C' for Celsius
        if fields[1] != "C":
            raise NMEA0183Error(
                "Invalid MTA sentence: temperature must be in Celsius (C)"
            )

        # Parse temperature
        temperature = None
        if fields[0]:
            try:
                temperature = float(fields[0])
            except ValueError:
                raise NMEA0183Error("Invalid MTA sentence: invalid temperature value")

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            temperature=temperature,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        if self.temperature is not None:
            return f"MTA: Air Temperature {self.temperature:.1f}°C"
        return "MTA: Temperature unknown"

    @property
    def temperature_fahrenheit(self) -> Optional[float]:
        """Return temperature in Fahrenheit if available."""
        if self.temperature is None:
            return None
        return (self.temperature * 9 / 5) + 32


@dataclass
class TXTSentence(NMEASentence):
    """TXT - Text Transmission

    Format: $--TXT,xx,xx,xx,c--c*hh
    Example: $GPTXT,01,01,02,ANTENNA OPEN*2B

    Fields:
        talker_id: Talker identifier
        sentence_type: Type of NMEA sentence
        raw: Raw NMEA sentence
        timestamp: Optional timestamp
        total_messages: Total number of messages in the sequence
        message_number: Message number in the sequence
        message_type: Type of message (00-99)
        text: The actual text message
    """

    # Required fields from parent
    talker_id: str
    sentence_type: str
    raw: str
    timestamp: Optional[datetime] = None

    # TXT specific fields with defaults
    total_messages: Optional[int] = None
    message_number: Optional[int] = None
    message_type: Optional[int] = None
    text: Optional[str] = None

    @classmethod
    def parse(cls, raw_sentence: str) -> "TXTSentence":
        """Parse a raw TXT sentence."""
        # Let parent class handle the basic parsing and validation
        sentence = super().parse(raw_sentence)

        # Split the data fields, skip the sentence identifier
        fields = raw_sentence.split(",")[1:]

        # Remove checksum from last field if present
        if "*" in fields[-1]:
            fields[-1] = fields[-1].split("*")[0]

        # Need at least 4 fields (total messages, message number, message type, text)
        if len(fields) < 4:
            raise NMEA0183Error("Invalid TXT sentence: insufficient fields")

        # Parse and validate numeric fields
        try:
            total_messages = int(fields[0]) if fields[0] else None
            message_number = int(fields[1]) if fields[1] else None
            message_type = int(fields[2]) if fields[2] else None

            # Validate ranges
            if total_messages is not None and (
                total_messages < 1 or total_messages > 99
            ):
                raise NMEA0183Error(
                    "Invalid TXT sentence: total messages must be between 1 and 99"
                )
            if message_number is not None and (
                message_number < 1 or message_number > 99
            ):
                raise NMEA0183Error(
                    "Invalid TXT sentence: message number must be between 1 and 99"
                )
            if message_type is not None and (message_type < 0 or message_type > 99):
                raise NMEA0183Error(
                    "Invalid TXT sentence: message type must be between 0 and 99"
                )

            # Validate message number doesn't exceed total messages
            if total_messages is not None and message_number is not None:
                if message_number > total_messages:
                    raise NMEA0183Error(
                        "Invalid TXT sentence: message number exceeds total messages"
                    )

        except ValueError:
            raise NMEA0183Error("Invalid TXT sentence: invalid numeric value")

        # Get the text message (might contain commas, so join remaining fields)
        text = ",".join(fields[3:]) if len(fields) > 3 else None

        return cls(
            talker_id=sentence.talker_id,
            sentence_type=sentence.sentence_type,
            raw=raw_sentence,
            timestamp=None,
            total_messages=total_messages,
            message_number=message_number,
            message_type=message_type,
            text=text,
        )

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        parts = []

        if self.message_number is not None and self.total_messages is not None:
            parts.append(f"Message {self.message_number}/{self.total_messages}")

        if self.message_type is not None:
            parts.append(f"Type {self.message_type:02d}")

        if self.text:
            parts.append(f'"{self.text}"')

        return f"TXT: {' | '.join(parts)}" if parts else "TXT: Empty message"


SENTENCE_TYPES = {
    "RMC": RMCSentence,
    "VHW": VHWSentence,
    "MWV": MWVSentence,
    "VPW": VPWSentence,
    "VTG": VTGSentence,
    "VWR": VWRSentence,
    "RMB": RMBSentence,
    "VWT": VWTSentence,
    "GGA": GGASentence,
    "GSA": GSASentence,
    "XTE": XTESentence,
    "GSV": GSVSentence,
    "RME": RMESentence,
    "GLL": GLLSentence,
    "RMZ": RMZSentence,
    "RMM": RMMSentence,
    "BOD": BODSentence,
    "RTE": RTESentence,
    "WPL": WPLSentence,
    "DBT": DBTSentence,
    "HDM": HDMSentence,
    "MTW": MTWSentence,
    "MWD": MWDSentence,
    "GBS": GBSSentence,
    "VLW": VLWSentence,
    "MTA": MTASentence,
    "TXT": TXTSentence,
}


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
        self._sentence_parsers = SENTENCE_TYPES

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
