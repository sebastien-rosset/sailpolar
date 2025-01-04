from sailpolar.polar.base_polar import BasePolar


class MaxSpeedPolar(BasePolar):
    def __init__(self):
        self.data = {}

    def analyze(self, nmea_data):
        for sentence in nmea_data:
            if sentence.sentence_type == "MWV":
                wind_angle = sentence.wind_angle
                wind_speed = sentence.wind_speed
            elif sentence.sentence_type == "VHW":
                boat_speed = sentence.water_speed
            else:
                continue

            key = (wind_speed, wind_angle)
            if key not in self.data or boat_speed > self.data[key]:
                self.data[key] = boat_speed

    def _get_export_data(self):
        for key, boat_speed in self.data.items():
            wind_speed, wind_angle = key
            yield wind_speed, wind_angle, boat_speed
