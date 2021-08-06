#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(600, 600)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

q_rgb_list = []

# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    for device_info in dai.Device.getAllAvailableDevices():
        device = stack.enter_context(dai.Device(pipeline, device_info))
        print("Conected to " + device_info.getMxId())
        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_rgb_list.append(q_rgb)

    while True:
        for i, q_rgb in enumerate(q_rgb_list):
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                cv2.imshow("rgb-" + str(i + 1), in_rgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
