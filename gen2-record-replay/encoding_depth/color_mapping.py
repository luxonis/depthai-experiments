import numpy as np
from numba import jit


class ColorMapBase:
    def __init__(self, minValue, maxValue) -> None:
        if minValue >= maxValue:
            raise ValueError("minValue must be smaller than maxValue")
        self.minValue : float = minValue
        self.maxValue : float = maxValue

    def to_color(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def to_mono(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

HUE_LIMIT = 250
@jit(nopython=True)
def depth_to_rgb_hue(x) -> tuple:
    R = 0
    G = 0
    B = 0

    if 0 <= x <= 255 or 1275 < x <= 1529:
        R = 255
    elif 255 < x <= 510:
        R = 510 - x
    elif 510 < x <= 1020:
        R = 0
    elif 1020 < x <= 1275:
        R = x - 1020

    if 0 <= x <= 255:
        G = x
    elif 255 < x <= 765:
        G = 255
    elif 765 < x <= 1020:
        G = 1020 - x
    elif 1020 < x <= 1529:
        G = 0

    if 0 <= x <= 510:
        B = 0
    elif 510 < x <= 765:
        B = x - 510
    elif 765 < x <= 1275:
        B = 255
    elif 1275 < x <= 1529:
        B = 1529 - x
    return B, G, R

@jit(nopython=True)
def rgb_hue_to_depth(r, g, b) -> int:
    if r < HUE_LIMIT and g < HUE_LIMIT and b < HUE_LIMIT:
        return 0
    if r >= g and r >= b:
        if g >= b:
            return g - b
        else:
            return (g - b) + 1529
    elif g >= r and g >= b:
        return b - r + 510
    elif b >= g and b >= r:
        return r - g + 1020

class HueColorMap(ColorMapBase):
    def __init__(self, minValue, maxValue) -> None:
        super().__init__(minValue, maxValue)
        # Precompute the hue array
        self.hue = np.zeros((1530, 3), dtype=np.uint8)
        for i in range(1530):
            self.hue[i] = depth_to_rgb_hue(i)

    def to_color(self, frame: np.ndarray) -> np.ndarray:
        d_normal = ((frame - self.minValue) / (self.maxValue - self.minValue)) * 1529
        rgb_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                B, G, R = depth_to_rgb_hue(d_normal[x, y])
                rgb_image[x, y, 0] = B
                rgb_image[x, y, 1] = G
                rgb_image[x, y, 2] = R
        return rgb_image

    def to_mono(self, frame: np.ndarray) -> np.ndarray:
        # Expects BGR image
        depth = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                b, g, r = frame[x, y]
                depth[x, y] = rgb_hue_to_depth(r, g, b)
        depth = self.minValue + (self.maxValue - self.minValue) * (depth / 1529)
        return depth


class MonoColorMap(ColorMapBase):
    def __init__(self, minValue, maxValue) -> None:
        super().__init__(minValue, maxValue)

    def to_color(self, frame: np.ndarray) -> np.ndarray:
        d_normal = ((frame - self.minValue) / (self.maxValue - self.minValue)) * 255
        rgb_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                G = d_normal[x, y]
                rgb_image[x, y, 0] = G
                rgb_image[x, y, 1] = G
                rgb_image[x, y, 2] = G
        return rgb_image

    def to_mono(self, frame: np.ndarray) -> np.ndarray:
        # Expects BGR image
        depth = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        for x in range(frame.shape[0]):
            for y in range(frame.shape[1]):
                b, g, r = frame[x, y]
                depth[x, y] = int((float(b) + g + r) / 3)
        depth = self.minValue + (self.maxValue - self.minValue) * (depth / 255)
        return depth

if __name__ == "__main__":
    import cv2
    from comparison import print_all_metrics
    width = 1280
    height = 800
    frame = np.ones((height, width), dtype=np.uint16)
    for i in range(height):
        frame[i, :] = i
    # Create a colormap object
    cv2.imshow("depth", frame.astype(np.uint8))
    colormap = HueColorMap(0, height)
    # Convert the image to color
    color_image = colormap.to_color(frame)
    cv2.imshow("color", color_image)
    # Convert the image back to depth
    backconverted_frame = colormap.to_mono(color_image)
    cv2.imshow("depth_backconverted", backconverted_frame.astype(np.uint8))

    # Check that the frames are the same
    diff = np.abs(backconverted_frame - frame)
    print(f"Max difference: {np.max(diff)}")
    print_all_metrics(frame, backconverted_frame, height)
    cv2.waitKey(0)