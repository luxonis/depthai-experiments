import depthai as dai
from typing import List
import random
import colorsys


def filter_internal_cameras(devices: List[dai.DeviceInfo]) -> List[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def generate_vibrant_random_color() -> tuple[float, float, float, float]:
    hue = random.random()  # Hue (0.0 to 1.0, wraps around)
    saturation = random.uniform(
        0.7, 1.0
    )  # High saturation (0.7 to 1.0 for colorfulness)
    value = random.uniform(0.6, 1.0)  # Medium to high brightness (0.6 to 1.0)

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    return (r, g, b, 1)
