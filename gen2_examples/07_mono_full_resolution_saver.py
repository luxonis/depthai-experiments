#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - mono (grayscale) camera
cam_left = pipeline.createMonoCamera()
cam_left.setCamId(1)
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("left")
cam_left.out.link(xout_rgb.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the grayscale frames from the output defined above
q_left = device.getOutputQueue(name="left", maxSize=4, overwrite=True)

# Make sure the destination path is present before starting to store the examples
Path('07_data').mkdir(parents=True)

while True:
    in_left = q_left.get()  # blocking call, will wait until a new data has arrived
    # data is originally represented as a flat 1D array, it needs to be converted into HxW form
    shape = (in_left.getHeight(), in_left.getWidth())
    frame_left = in_left.getData().reshape(shape).astype(np.uint8)
    frame_left = np.ascontiguousarray(frame_left)
    # frame is transformed and ready to be shown
    cv2.imshow("left", frame_left)
    # after showing the frame, it's being stored inside a target directory as a PNG image
    cv2.imwrite(f"07_data/{int(time.time() * 10000)}.png", frame_left)

    if cv2.waitKey(1) == ord('q'):
        break
