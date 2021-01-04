#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setCamId(1)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setCamId(2)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
left.out.link(depth.left)
right.out.link(depth.right)

xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

device = dai.Device(pipeline)
device.startPipeline()

q = device.getOutputQueue(name="disparity", maxSize=4, overwrite=True)

while True:
    in_rgb = q.get()
    frame = in_rgb.getData().reshape((in_rgb.getHeight(), in_rgb.getWidth())).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    cv2.imshow("disparity", frame)

    if cv2.waitKey(1) == ord('q'):
        break
