#!/usr/bin/env python3

import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

cam_right = pipeline.createMonoCamera()
cam_right.setCamId(2)
cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

xout_left = pipeline.createXLinkOut()
xout_left.setStreamName('left')
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName('right')

cam_left.out.link(xout_left.input)
cam_right.out.link(xout_right.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_left = device.getOutputQueue("left")
q_right = device.getOutputQueue("right")

frame_left = None
frame_right = None

while True:
    in_left = q_left.tryGet()
    in_right = q_right.tryGet()

    if in_left is not None:
        frame_left = in_left.getData().reshape((in_left.getHeight(), in_left.getWidth())).astype(np.uint8)
        frame_left = np.ascontiguousarray(frame_left)

    if in_right is not None:
        frame_right = in_right.getData().reshape((in_right.getHeight(), in_right.getWidth())).astype(np.uint8)
        frame_right = np.ascontiguousarray(frame_right)

    if frame_left is not None:
        cv2.imshow("left", frame_left)
    if frame_right is not None:
        cv2.imshow("right", frame_right)

    if cv2.waitKey(1) == ord('q'):
        break
