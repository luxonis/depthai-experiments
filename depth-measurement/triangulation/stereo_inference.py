import math
import depthai as dai
import numpy as np


class StereoInference:
    def __init__(self, device: dai.Device, resolution: tuple[int, int]) -> None:
        calibData = device.readCalibration()
        baseline = calibData.getBaselineDistance(useSpecTranslation=True) * 10  # mm

        # Original mono frames shape
        self.width, self.heigth = resolution
        self.hfov = calibData.getFov(dai.CameraBoardSocket.CAM_C)

        focalLength = self.get_focal_length_pixels(self.width, self.hfov)
        self.dispScaleFactor = baseline * focalLength

    def get_focal_length_pixels(self, pixel_width, hfov):
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi / 180)

    def calculate_depth(self, disparity_pixels: float):
        try:
            return self.dispScaleFactor / disparity_pixels
        except ZeroDivisionError:
            return 0

    def calculate_distance(self, c1, c2):
        c1 = np.array(c1)
        c2 = np.array(c2)

        x_delta = c1[0] - c2[0]
        y_delta = c1[1] - c2[1]
        return math.sqrt(x_delta**2 + y_delta**2)

    def calc_angle(self, offset):
        return math.atan(math.tan(self.hfov / 2.0) * offset / (self.width / 2.0))

    def calc_spatials(self, coords, depth):
        x, y = coords
        bb_x_pos = x - self.width / 2
        bb_y_pos = y - self.heigth / 2

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = depth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)
        return [x, y, z]
