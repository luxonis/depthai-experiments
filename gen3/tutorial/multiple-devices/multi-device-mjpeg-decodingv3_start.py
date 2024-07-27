import depthai as dai
import threading
import contextlib
import cv2
import time
from queue import Queue


def filterInternalCameras(devices : list[dai.DeviceInfo]):
    filtered_devices = []
    for d in devices:
        if d.protocol != dai.XLinkProtocol.X_LINK_TCP_IP:
            filtered_devices.append(d)

    return filtered_devices


run = True
# This can be customized to pass multiple parameters
def getPipeline(dev, q : list):
    # Start defining a pipeline
    pipeline = dai.Pipeline(dev)

    # Define sources and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

    # Properties
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
    camRgb.video.link(videoEnc.input)

    q.append(videoEnc.bitstream.createOutputQueue(maxSize=1, blocking=False))

    return pipeline

def worker(dev_info, stack, queue):
    global run
    device: dai.Device = stack.enter_context(dai.Device(dev_info))

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    q = []
    pipeline = getPipeline(device, q)
    pipeline.start()

    while run:
        imgFrame = q[0].get()
        # decode
        frame = cv2.imdecode(imgFrame.getData(), cv2.IMREAD_COLOR)
        queue.put(frame)
    print('Closing thread')


device_infos = filterInternalCameras(dai.Device.getAllAvailableDevices())
print(f'Found {len(device_infos)} devices')

with contextlib.ExitStack() as stack:
    queues : dict[str, Queue] = {}
    threads : list[threading.Thread]= []
    for dev in device_infos:
        q = Queue(1)
        thread = threading.Thread(target=worker, args=(dev, stack, q))
        queues[dev.getMxId()] = q
        thread.start()
        threads.append(thread)

    while True:
        for name, q in queues.items():
            try:
                frame = q.get(block=False)
                frame = cv2.pyrDown(frame)
                cv2.imshow(name, frame)
            except:
                continue

        if cv2.waitKey(1) == ord('q'):
            run=False
            break

    for t in threads:
        t.join() # Wait for all threads to finish

print('Devices closed')
