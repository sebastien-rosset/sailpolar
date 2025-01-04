from datetime import timedelta
import os
from sailpolar.polar.base_polar import BasePolar


class MaxSpeedPolar(BasePolar):
    def __init__(self, time_window=5):  # 5 seconds time window
        super().__init__()
        self.time_window = timedelta(seconds=time_window)

    def _extract_wind_data(self, sentence):
        """Extract wind speed and angle from different sentence types."""
        if sentence.sentence_type == "MWV":
            if sentence.wind_speed is None or sentence.wind_angle is None:
                return None
            return {
                "speed": sentence.wind_speed,
                "angle": sentence.wind_angle,
                "reference": sentence.reference,
                "units": sentence.speed_units,
            }
        elif sentence.sentence_type == "MWD":
            if sentence.speed_knots is None or sentence.direction_true is None:
                return None
            return {
                "speed": sentence.speed_knots,
                "angle": sentence.direction_true,
                "reference": "T",
                "units": "N",
            }
        elif sentence.sentence_type == "VWT":
            if sentence.wind_speed_knots is None or sentence.wind_angle is None:
                return None
            return {
                "speed": sentence.wind_speed_knots,
                "angle": sentence.wind_angle,
                "reference": "T",
                "units": "N",
            }
        elif sentence.sentence_type == "VWR":
            if sentence.wind_speed_knots is None or sentence.wind_angle is None:
                return None
            return {
                "speed": sentence.wind_speed_knots,
                "angle": sentence.wind_angle,
                "reference": "R",
                "units": "N",
            }
        return None

    def _extract_speed_data(self, sentence):
        """Extract boat speed from different sentence types."""
        if sentence.sentence_type == "VHW":
            return sentence.speed_knots
        elif sentence.sentence_type == "VTG":
            return sentence.speed_knots
        elif sentence.sentence_type == "VPW":
            return sentence.speed_knots
        return None

    def analyze(self, segments):
        for segment in segments:
            # Collect wind and speed measurements with timestamps
            wind_measurements = []
            speed_measurements = []

            # Collect relevant sentences
            for sentence in segment.sentences:
                if (
                    sentence.sentence_type in ["MWV", "MWD", "VWT", "VWR"]
                    and sentence.timestamp
                ):
                    wind_measurements.append(sentence)
                if (
                    sentence.sentence_type in ["VHW", "VTG", "VPW"]
                    and sentence.timestamp
                ):
                    speed_measurements.append(sentence)

            # Match wind and speed within time window
            for wind_sentence in wind_measurements:
                matching_speeds = [
                    speed
                    for speed in speed_measurements
                    if abs(speed.timestamp - wind_sentence.timestamp)
                    <= self.time_window
                ]

                # Process matching measurements
                for speed_sentence in matching_speeds:
                    wind_data = self._extract_wind_data(wind_sentence)
                    boat_speed = self._extract_speed_data(speed_sentence)

                    if wind_data and boat_speed is not None:
                        key = (
                            wind_data["speed"],
                            wind_data["angle"],
                            wind_data["reference"],
                        )

                        # Update max speed for this wind condition
                        if key not in self.data or boat_speed > self.data[key]:
                            self.data[key] = boat_speed
