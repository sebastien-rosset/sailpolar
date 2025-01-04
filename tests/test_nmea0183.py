"""
Tests for NMEA 0183 parser.
"""

import os
import pytest
from datetime import datetime
from sailpolar.parser.nmea0183 import (
    NMEA0183Parser,
    NMEA0183Error,
    ChecksumError,
    RMCSentence,
    VHWSentence,
    MWVSentence,
    VPWSentence,
    VTGSentence,
    GSVSentence,
    VWRSentence,
    RMBSentence,
    VWTSentence,
    GGASentence,
    GSASentence,
    XTESentence,
    RMESentence,
    GLLSentence,
    RMZSentence,
    RMMSentence,
    BODSentence,
    RTESentence,
    WPLSentence,
    DBTSentence,
    HDMSentence,
    MTWSentence,
    MWDSentence,
    GBSSentence,
    VLWSentence,
    MTASentence,
    TXTSentence,
    validate_checksum,
    calculate_checksum,
    NMEASentence,
)

# Get the absolute path to the test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_FILE = os.path.join(TEST_DATA_DIR, "Race-AIS-Sart-10m.txt")


def test_checksum_validation():
    """Test NMEA checksum calculation and validation."""
    # Test valid sentence
    sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
    data = sentence[1 : sentence.index("*")]
    checksum = sentence.split("*")[1]
    assert validate_checksum(data, checksum)

    # Test invalid checksum
    assert not validate_checksum(data, "00")

    # Test checksum calculation
    assert calculate_checksum(data) == checksum


def test_rmc_sentence_parsing():
    """Test parsing of RMC sentences."""
    sentence = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
    parser = NMEA0183Parser()
    parsed = parser.parse_sentence(sentence)

    assert isinstance(parsed, RMCSentence)
    assert parsed.talker_id == "GP"
    assert parsed.sentence_type == "RMC"
    assert parsed.status == "A"
    assert abs(parsed.latitude - 48.1173) < 0.0001
    assert abs(parsed.longitude - 11.5167) < 0.0001
    assert abs(parsed.sog - 22.4) < 0.0001
    assert abs(parsed.cog - 84.4) < 0.0001
    assert parsed.timestamp.year == 1994
    assert parsed.timestamp.month == 3
    assert parsed.timestamp.day == 23


def test_vhw_sentence_parsing():
    """Test parsing of VHW sentences."""
    sentence = "$IIVHW,245.1,T,245.1,M,004.01,N,007.4,K*63"  # Updated checksum
    parser = NMEA0183Parser()
    parsed = parser.parse_sentence(sentence)

    assert isinstance(parsed, VHWSentence)
    assert parsed.talker_id == "II"
    assert parsed.sentence_type == "VHW"
    assert abs(parsed.heading_true - 245.1) < 0.0001
    assert abs(parsed.heading_mag - 245.1) < 0.0001
    assert abs(parsed.speed_knots - 4.01) < 0.0001
    assert abs(parsed.speed_kmh - 7.4) < 0.0001


def test_mwv_sentence_parsing():
    """Test parsing of MWV sentences."""
    sentence = "$IIMWV,045.0,R,10.5,N,A*08"  # Updated checksum
    parser = NMEA0183Parser()
    parsed = parser.parse_sentence(sentence)

    assert isinstance(parsed, MWVSentence)
    assert parsed.talker_id == "II"
    assert parsed.sentence_type == "MWV"
    assert abs(parsed.wind_angle - 45.0) < 0.0001
    assert parsed.reference == "R"
    assert abs(parsed.wind_speed - 10.5) < 0.0001
    assert parsed.speed_units == "N"
    assert parsed.status == "A"


def test_invalid_sentences():
    """Test handling of invalid sentences."""
    parser = NMEA0183Parser()

    # Test empty sentence
    assert parser.parse_sentence("") is None

    # Test sentence without $
    assert parser.parse_sentence("GPRMC,123519,A") is None

    # Test sentence with invalid checksum
    with pytest.raises(ChecksumError):
        NMEASentence.parse("$GPRMC,123519,A*00")

    # Test incomplete sentence
    with pytest.raises(NMEA0183Error):
        NMEASentence.parse("$GPRMC")


def test_missing_fields():
    """Test handling of sentences with missing fields."""
    parser = NMEA0183Parser()

    # RMC with missing position
    sentence = "$GPRMC,123519,A,,,,,022.4,084.4,230394,003.1,W*XX"  # Checksum would need to be valid
    parsed = parser.parse_sentence(
        sentence.replace("XX", calculate_checksum(sentence[1 : sentence.index("*")]))
    )
    assert isinstance(parsed, RMCSentence)
    assert parsed.latitude is None
    assert parsed.longitude is None
    assert abs(parsed.sog - 22.4) < 0.0001

    # VHW with missing speed
    sentence = "$IIVHW,245.1,T,245.1,M,,,007.4,K*XX"
    parsed = parser.parse_sentence(
        sentence.replace("XX", calculate_checksum(sentence[1 : sentence.index("*")]))
    )
    assert isinstance(parsed, VHWSentence)
    assert abs(parsed.heading_true - 245.1) < 0.0001
    assert parsed.speed_knots is None


def test_nmea_error_messages():
    """Test error message clarity."""
    with pytest.raises(NMEA0183Error) as exc:
        NMEASentence.parse("$")
    assert "Invalid sentence format" in str(exc.value)

    with pytest.raises(NMEA0183Error) as exc:
        NMEASentence.parse("$GP")
    assert "Invalid sentence identifier" in str(exc.value)


def test_file_parsing_fail_unknown():
    """Test parsing of NMEA file with strict mode (fail on unknown sentences)."""
    parser = NMEA0183Parser(fail_unknown=True)
    sentences, frequency_stats = parser.parse_file(TEST_FILE)

    # Collect unique sentence types
    sentence_types = set(sentence.sentence_type for sentence in sentences)

    # Define the expected sentence types
    expected_sentence_types = {
        "BOD",
        "DBT",
        "GBS",
        "GGA",
        "GLL",
        "GSA",
        "GSV",
        "HDM",
        "MTA",
        "MTW",
        "MWD",
        "MWV",
        "RMB",
        "RMC",
        "RME",
        "RMM",
        "RMZ",
        "RTE",
        "TXT",
        "VHW",
        "VLW",
        "VPW",
        "VTG",
        "VWR",
        "VWT",
        "WPL",
        "XTE",
    }

    # Assert that the set of sentence types matches the expected set
    assert sentence_types == expected_sentence_types, "Unexpected sentence types found"

    # Print the set of unique sentence types
    print(f"Found {len(sentence_types)} Unique Sentence Types: {sentence_types}")

    # Print the frequency stats
    print("Frequency Stats:")
    for key, stats in frequency_stats.items():
        talker_id, sentence_type = key
        print(f"Talker ID: {talker_id}, Sentence Type: {sentence_type}")
        print(f"  Frequency: {stats['frequency']:.2f} Hz")
        print(f"  Min Delta: {stats['min_delta']:.4f} seconds")
        print(f"  Max Delta: {stats['max_delta']:.4f} seconds")
        print(f"  Delta Range: {stats['delta_range']:.4f} seconds")
        print(f"  Is Stable: {stats['is_stable']}")
        print()

    # Write assertions
    assert len(sentences) > 0, "No sentences were parsed from the file"
    assert len(frequency_stats) > 0, "No frequency stats were generated"

    # Example assertions based on the specific data
    assert (
        frequency_stats[("GP", "GGA")]["frequency"] == 1.0
    ), "Unexpected frequency for GP-GGA"
    assert frequency_stats[("GP", "RMC")][
        "is_stable"
    ], "GP-RMC frequency should be stable"


def test_file_parsing_skip_unknown():
    """Test parsing of NMEA file while skipping unknown sentences."""
    parser = NMEA0183Parser(fail_unknown=False)  # This is default behavior
    sentences, frequency_stats = parser.parse_file(TEST_FILE)

    assert len(sentences) > 0, "No sentences were parsed"

    # Verify all parsed sentences are of known types
    for sentence in sentences:
        assert isinstance(
            sentence,
            (
                RMCSentence,
                VHWSentence,
                MWVSentence,
                VPWSentence,
                VTGSentence,
                VWRSentence,
                RMBSentence,
                VWTSentence,
                GGASentence,
                GSASentence,
                XTESentence,
                GSVSentence,
                RMESentence,
                GLLSentence,
                RMZSentence,
                RMMSentence,
                BODSentence,
                RTESentence,
                WPLSentence,
                DBTSentence,
                HDMSentence,
                MTWSentence,
                MWDSentence,
                GBSSentence,
                VLWSentence,
                MTASentence,
                TXTSentence,
            ),
        ), f"Parser returned unsupported sentence type: {sentence.sentence_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
