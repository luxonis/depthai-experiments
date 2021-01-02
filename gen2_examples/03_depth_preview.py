#!/usr/bin/env python3

import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

left = pipeline.createMonoCamera()
left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setCamId(1)

right = pipeline.createMonoCamera()
right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setCamId(2)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
left.out.link(depth.left)
right.out.link(depth.right)

xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q = device.getOutputQueue("disparity")

while True:
    in_rgb = q.get()
    frame = in_rgb.getData().reshape((in_rgb.getHeight(), in_rgb.getWidth())).astype(np.uint8)
    frame = np.ascontiguousarray(frame)
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    cv2.imshow("disparity", frame)

    if cv2.waitKey(1) == ord('q'):
        break
