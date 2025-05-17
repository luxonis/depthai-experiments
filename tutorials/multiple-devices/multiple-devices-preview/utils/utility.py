import depthai as dai
from typing import List


def filter_internal_cameras(devices: List[dai.DeviceInfo]) -> List[dai.DeviceInfo]:
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices
