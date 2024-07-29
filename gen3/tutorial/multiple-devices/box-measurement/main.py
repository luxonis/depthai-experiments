import depthai as dai
from camera import Camera
from point_cloud_visualizer import PointCloudVisualizer


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


device_infos = filterInternalCameras(dai.Device.getAllAvailableDevices())
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

cameras: list[Camera] = []

for device_info in device_infos:
    cameras.append(Camera(device_info, len(cameras)+1, show_video=False, show_point_cloud=False))

PointCloudVisualizer(cameras)
