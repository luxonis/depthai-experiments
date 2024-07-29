import depthai as dai
import threading
import cv2


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


# Creates and returns pipeline with a specific device
def getPipeline(dev : dai.Device, stereo : bool, queues, mxid):
    pipeline = dai.Pipeline(dev)

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # For the demo, just set a larger RGB preview size for OAK-D
    if stereo:
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    queues["rgb-" + mxid] = cam_rgb.preview.createOutputQueue()

    return pipeline


# works with device and starts pipeline
def worker(dev_info,  queues : list, pipelines : list):
    device: dai.Device = dai.Device(dev_info)

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    pipeline = getPipeline(device, len(cameras)==3, queues, mxid)
    pipelines.append(pipeline)
    pipeline.start()
    

devices = filterInternalCameras(dai.Device.getAllAvailableDevices())
print(f'Found {len(devices)} internal devices')


queues = {}
threads : list[threading.Thread] = []
pipelines = []

for dev in devices:
    thread = threading.Thread(target=worker, args=(dev, queues, pipelines))
    thread.start()
    threads.append(thread)

for t in threads:
    t.join()

while any(pipeline.isRunning() for pipeline in pipelines):
    for name, queue in queues.items():
        if queue.has():
            cv2.imshow(name, queue.get().getCvFrame())
    if cv2.waitKey(1) == ord('q'):
        break

print('Devices closed')
