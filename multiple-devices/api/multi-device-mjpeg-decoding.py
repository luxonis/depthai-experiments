import depthai as dai
import threading
import contextlib
import cv2
import time
from queue import Queue

run = True
# This can be customized to pass multiple parameters
def getPipeline(stereo):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define sources and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

    # Properties
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
    camRgb.video.link(videoEnc.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mjpeg")
    videoEnc.bitstream.link(xout.input)
    return pipeline

def worker(dev_info, stack, queue):
    global run
    openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    device: dai.Device = stack.enter_context(dai.Device(openvino_version, dev_info, False))

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    device.startPipeline(getPipeline(len(cameras)==3))
    q = device.getOutputQueue(name="mjpeg", maxSize=1, blocking=False)

    while run:
        imgFrame = q.get()
        # decode
        frame = cv2.imdecode(imgFrame.getData(), cv2.IMREAD_COLOR)
        queue.put(frame)
    print('Closing thread')


device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

with contextlib.ExitStack() as stack:
    queues = {}
    threads = []
    for dev in device_infos:
        time.sleep(1) # Currently required due to XLink race issues
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