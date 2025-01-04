import os
from abc import ABC, abstractmethod
import csv
import numpy as np


class BasePolar(ABC):
    def __init__(self):
        self.data = {}

    @abstractmethod
    def analyze(self, nmea_data):
        """Analyze NMEA data and extract polar information."""
        pass

    def _process_polar_data(self, round_speed=0.5, round_angle=5):
        """
        Process raw polar data into a structured format.

        Args:
            round_speed (float): Round wind speeds to nearest multiple
            round_angle (float): Round wind angles to nearest multiple

        Returns:
            tuple: Sorted wind speeds, sorted wind angles, processed data matrix
        """
        # Collect and round wind speeds and angles
        wind_speeds = set()
        wind_angles = set()
        processed_data = {}

        for (wind_speed, wind_angle, wind_ref), boat_speed in self.data.items():
            # Round wind speed and angle
            rounded_speed = round(wind_speed / round_speed) * round_speed
            rounded_angle = round(wind_angle / round_angle) * round_angle

            # Track unique wind speeds and angles
            wind_speeds.add(rounded_speed)
            wind_angles.add(rounded_angle)

            # Create key with rounded values
            key = (rounded_speed, rounded_angle, wind_ref)

            # Update max speed for this wind condition
            if key not in processed_data or boat_speed > processed_data[key]:
                processed_data[key] = boat_speed

        # Sort wind speeds and angles
        sorted_wind_speeds = sorted(list(wind_speeds))
        sorted_wind_angles = sorted(list(wind_angles))

        # Create a 2D matrix to store max boat speeds
        polar_matrix = np.zeros(
            (len(sorted_wind_angles) + 1, len(sorted_wind_speeds) + 1)
        )

        # Fill first row with wind speeds
        polar_matrix[0, 1:] = sorted_wind_speeds

        # Fill first column with wind angles
        polar_matrix[1:, 0] = sorted_wind_angles

        # Fill the matrix with max speeds
        for (wind_speed, wind_angle, wind_ref), boat_speed in processed_data.items():
            # Find indices
            speed_index = sorted_wind_speeds.index(wind_speed) + 1
            angle_index = sorted_wind_angles.index(wind_angle) + 1

            # Update matrix
            polar_matrix[angle_index, speed_index] = boat_speed

        return sorted_wind_speeds, sorted_wind_angles, polar_matrix

    def export_csv(self, file_path, round_speed=0.5, round_angle=5):
        """
        Export polar data to a CSV file.

        Args:
            file_path (str): Path to the output CSV file
            round_speed (float): Round wind speeds to nearest multiple
            round_angle (float): Round wind angles to nearest multiple
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Process polar data
        sorted_wind_speeds, sorted_wind_angles, polar_matrix = self._process_polar_data(
            round_speed, round_angle
        )

        # Prepare headers
        headers = ["twa/tws"] + [str(ws) for ws in sorted_wind_speeds]

        # Write to CSV
        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write headers
            writer.writerow(headers)

            # Write data rows
            for row in polar_matrix[1:]:  # Skip first row (wind speeds)
                writer.writerow([str(val) for val in row])
