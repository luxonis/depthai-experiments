#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.ColorCamera)
monoRight = pipeline.create(dai.node.ColorCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

monoRight.initialControl.setSharpness(0)
monoRight.initialControl.setLumaDenoise(0)
monoRight.initialControl.setChromaDenoise(4)

monoLeft.initialControl.setSharpness(0)
monoLeft.initialControl.setLumaDenoise(0)
monoLeft.initialControl.setChromaDenoise(4)

# monoLeft.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
# monoRight.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

# Linking
monoRight.video.link(xoutRight.input)
monoLeft.video.link(xoutLeft.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()

        if inLeft is not None:
            cv2.imshow("left", inLeft.getCvFrame())

        if inRight is not None:
            cv2.imshow("right", inRight.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
