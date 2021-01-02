#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("left")
cam_left.out.link(xout_rgb.input)

found, device_info = depthai.XLinkConnection.getFirstDevice(depthai.XLinkDeviceState.X_LINK_UNBOOTED)
if not found:
    raise RuntimeError("Device not found")
device = depthai.Device(pipeline, device_info)
device.startPipeline()

q_left = device.getOutputQueue("left")

Path('07_data').mkdir(parents=True)

while True:
    in_left = q_left.get()
    shape = (in_left.getHeight(), in_left.getWidth())
    frame_left = in_left.getData().reshape(shape).astype(np.uint8)
    frame_left = np.ascontiguousarray(frame_left)
    cv2.imshow("left", frame_left)
    cv2.imwrite(f"07_data/{int(time.time() * 10000)}.png", frame_left)

    if cv2.waitKey(1) == ord('q'):
        break
