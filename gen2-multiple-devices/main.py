#!/usr/bin/env python3

import cv2
import depthai as dai
import threading
import time

# This can be customized to pass multiple parameters
def getPipeline(device_type):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    # For the demo, just set a larger RGB preview size for OAK-D
    if device_type.startswith("OAK-D"):
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    # Create output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline

q_rgb_list = []
alive = True

def runDevice(index, device_info):
    # Note: the pipeline isn't set here, as we don't know yet what device it is.
    # The extra arguments passed are required by the existing overload variants
    openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    usb2_mode = False
    with dai.Device(openvino_version, device_info, usb2_mode) as device:
        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        print(index, "=== Connected to " + device_info.getMxId())
        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()
        print(index, "   >>> MXID:", mxid)
        print(index, "   >>> Cameras:", *[c.name for c in cameras])
        print(index, "   >>> USB speed:", usb_speed.name)

        device_type = "unknown"
        if   len(cameras) == 1: device_type = "OAK-1"
        elif len(cameras) == 3: device_type = "OAK-D"
        # If USB speed is UNKNOWN, assume it's a POE device
        if usb_speed == dai.UsbSpeed.UNKNOWN: device_type += "-POE"

        # Get a customized pipeline based on identified device type
        pipeline = getPipeline(device_type)
        print(index, "   >>> Loading pipeline for:", device_type)
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        stream_name = "rgb-" + mxid + "-" + device_type
        q_rgb_list.append((q_rgb, stream_name))
        print(index, "   >>> Pipeline started, created stream", stream_name)

        # TODO better
        while alive:
            time.sleep(0.1)

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

for index, device_info in enumerate(device_infos):
    print("Starting thread for device", index, device_info.getMxId())
    t = threading.Thread(target=runDevice, args=(index, device_info,))
    t.start()

while True:
    for q_rgb, stream_name in q_rgb_list:
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            cv2.imshow(stream_name, in_rgb.getCvFrame())

    if cv2.waitKey(1) == ord('q'):
        alive = False
        break
