import cv2
import depthai as dai
from birdseyeview import BirdsEyeView
from camera import Camera
from typing import List
import config

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

cameras: List[Camera] = []

friendly_id = 0
for device_info in device_infos:
    friendly_id += 1
    cameras.append(Camera(device_info, friendly_id, show_video=True))


birds_eye_view = BirdsEyeView(cameras, config.size[0], config.size[1], config.scale)

while True:
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
