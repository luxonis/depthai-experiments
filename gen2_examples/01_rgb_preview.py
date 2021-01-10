#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the rgb frames from the output defined above
q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

while True:
    in_rgb = q_rgb.get()  # blocking call, will wait until a new data has arrived
    # data is originally represented as a flat 1D array, it needs to be converted into HxWxC form
    shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
    frame_rgb = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    # frame is transformed and ready to be shown
    cv2.imshow("rgb", frame_rgb)

    if cv2.waitKey(1) == ord('q'):
        break
