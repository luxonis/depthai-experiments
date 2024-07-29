import cv2
import depthai as dai
from birdseyeview import BirdsEyeView
from camera import Camera
import config


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def get_pipelines(cameras):
    pipelines = []
    for camera in cameras:
        pipelines.append(camera.pipeline)
    return pipelines


device_infos = filterInternalCameras(dai.Device.getAllAvailableDevices())
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

cameras: list[Camera] = []

friendly_id = 0
for device_info in device_infos:
    friendly_id += 1
    cameras.append(Camera(device_info, friendly_id, show_video=True))


birds_eye_view = BirdsEyeView(cameras, config.size[0], config.size[1], config.scale)

while all(pipeline.isRunning() for pipeline in get_pipelines(cameras)):
    key = cv2.waitKey(1)

    # QUIT - press `q` to quit
    if key == ord('q'):
        break

    # TOGGLE DEPTH VIEW - press `d` to toggle depth view
    if key == ord('d'):
        for camera in cameras:
            camera.show_detph = not camera.show_detph

    for camera in cameras:
        camera.update()

    birds_eye_view.render()
