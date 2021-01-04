#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(3840, 2160)
cam_rgb.setCamId(0)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setInterleaved(False)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

device = dai.Device(pipeline)
device.startPipeline()

q_rgb = device.getOutputQueue(name="rgb")

Path('06_data').mkdir(parents=True)

while True:
    in_rgb = q_rgb.get()
    shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
    frame_rgb = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    cv2.imshow("rgb", frame_rgb)
    cv2.imwrite(f"06_data/{int(time.time() * 10000)}.png", frame_rgb)

    if cv2.waitKey(1) == ord('q'):
        break
