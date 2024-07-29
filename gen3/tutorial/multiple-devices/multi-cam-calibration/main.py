import cv2
import depthai as dai
from camera import Camera


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


def select_camera(friendly_id: int):
    global selected_camera

    i = friendly_id - 1
    if i >= len(cameras) or i < 0: 
        return None 

    selected_camera = cameras[i]
    print(f"Selected camera {friendly_id}")

    return selected_camera


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
    cameras.append(Camera(device_info, friendly_id))

selected_camera = cameras[0]

select_camera(1)


while all(pipeline.isRunning() for pipeline in get_pipelines(cameras)):
    key = cv2.waitKey(1)

    # QUIT - press `q` to quit
    if key == ord('q'):
        break
    
    # CAMERA SELECTION - use the number keys to select a camera
    if key >= ord('1') and key <= ord('9'):
        select_camera(key - ord('1') + 1)

    # POSE ESTIMATION - press `p` to estimate the pose of the selected camera and save it to file
    if key == ord('p'):
        selected_camera.estimate_pose()

    for camera in cameras:
        camera.update()
