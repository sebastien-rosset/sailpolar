# test_max_speed_polar.py
import os
from sailpolar.parser.nmea0183 import NMEA0183Parser
from sailpolar.polar.max_speed_polar import MaxSpeedPolar


def test_max_speed_polar():
    # Set up the file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nmea_file_path = os.path.join(current_dir, "data", "Race-AIS-Sart-10m.txt")

    # Parse the NMEA file
    parser = NMEA0183Parser()
    parsed_sentences = parser.parse_file(nmea_file_path)

    # Create an instance of MaxSpeedPolar
    max_speed_polar = MaxSpeedPolar()

    # Analyze the parsed sentences
    max_speed_polar.analyze(parsed_sentences)

    # Export the polar data to a CSV file
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "max_speed_polar.csv")
    max_speed_polar.export_csv(output_file_path)

    # Assert that the output file exists
    assert os.path.isfile(output_file_path)
