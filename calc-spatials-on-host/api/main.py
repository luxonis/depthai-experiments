#!/usr/bin/env python3

import math

import depthai as dai
import numpy as np

from calc import HostSpatialsCalc
from utility import *

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")

    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y = 200
    x = 300
    step = 3
    delta = 5
    hostSpatials.setDeltaRoi(delta)

    print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")

    while True:
        depthData = depthQueue.get()
        # Calculate spatial coordiantes from depth frame
        spatials, centroid = hostSpatials.calc_spatials(depthData, (x, y))  # centroid == x/y in our case

        # Get disparity frame for nicer depth visualization
        disp = dispQ.get().getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        text.rectangle(disp, (x - delta, y - delta), (x + delta, y + delta))
        text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                     (x + 10, y + 20))
        text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                     (x + 10, y + 35))
        text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                     (x + 10, y + 50))

        # Show the frame
        cv2.imshow("depth", disp)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            y -= step
        elif key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('d'):
            x += step
        elif key == ord('r'):  # Increase Delta
            if delta < 50:
                delta += 1
                hostSpatials.setDeltaRoi(delta)
        elif key == ord('f'):  # Decrease Delta
            if 3 < delta:
                delta -= 1
                hostSpatials.setDeltaRoi(delta)
