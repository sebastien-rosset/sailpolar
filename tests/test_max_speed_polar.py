# test_max_speed_polar.py
from datetime import timedelta
import os
from sailpolar.parser.nmea0183 import NMEA0183Parser
from sailpolar.polar.max_speed_polar import MaxSpeedPolar


def test_max_speed_polar():
    # Set up the file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nmea_file_path = os.path.join(current_dir, "data", "Race-AIS-Sart-10m.txt")

    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    pickle_file_path = os.path.join(output_dir, "nmea_data.pickle")
    time_window = timedelta(seconds=1.5)
    max_speed_polar = MaxSpeedPolar(time_window)
    parsed_sentences = None

    if os.path.isfile(pickle_file_path):
        # Load the parsed NMEA data from the pickle file
        parsed_sentences = NMEA0183Parser.load_nmea_data(pickle_file_path)
    else:
        # Parse the NMEA file
        parser = NMEA0183Parser()
        parsed_sentences, _ = parser.parse_file(nmea_file_path)
        # Save the parsed NMEA data to a pickle file
        NMEA0183Parser.save_nmea_data(parsed_sentences, pickle_file_path)

    # Analyze the parsed sentences
    print("Analyzing polar data...")
    max_speed_polar.analyze(parsed_sentences)

    # Export the polar data to a CSV file, using time_window in the file name
    output_file_path = os.path.join(
        output_dir, f"max_speed_polar_{time_window.total_seconds()}s.csv"
    )
    max_speed_polar.export_csv(output_file_path)

    # Assert that the output file exists
    assert os.path.isfile(output_file_path)
