#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
left.out.link(depth.left)
right.out.link(depth.right)

# Create output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the disparity frames from the outputs defined above
q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

while True:
    in_rgb = q.get() # blocking call, will wait until a new data has arrived
    # data is originally represented as a flat 1D array, it needs to be converted into HxW form
    frame = in_rgb.getData().reshape((in_rgb.getHeight(), in_rgb.getWidth())).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    # frame is transformed, the color map will be applied to highlight the depth info
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    # frame is ready to be shown
    cv2.imshow("disparity", frame)

    if cv2.waitKey(1) == ord('q'):
        break
