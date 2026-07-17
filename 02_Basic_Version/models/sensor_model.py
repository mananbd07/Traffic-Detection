import numpy as np

class SensorModel:
    def __init__(self):
        pass

    def get_sensor_data(self, lanes, waiting_time):
        avg_speed = np.maximum(5, 50 - lanes * 2)

        return {
            "avg_speed": avg_speed,
            "waiting_time": waiting_time.copy()
        }