import os
from abc import ABC, abstractmethod
import csv


class BasePolar(ABC):
    @abstractmethod
    def analyze(self, nmea_data):
        pass

    def export_csv(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Wind Speed", "Wind Angle", "Boat Speed"])
            for wind_speed, wind_angle, boat_speed in self._get_export_data():
                writer.writerow([wind_speed, wind_angle, boat_speed])

    @abstractmethod
    def _get_export_data(self):
        pass
