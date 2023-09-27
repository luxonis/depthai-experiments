#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

cam_a = pipeline.createColorCamera()
# We assume the ToF camera sensor is on port CAM_A
cam_a.setBoardSocket(dai.CameraBoardSocket.CAM_A)

tof = pipeline.create(dai.node.ToF)
xout = pipeline.createXLinkOut()
xout.setStreamName("depth")
# Configure the ToF node
tofConfig = tof.initialConfig.get()
# tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MIN
tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MAX
tofConfig.depthParams.avgPhaseShuffle = False
tofConfig.depthParams.minimumAmplitude = 3.0
tof.initialConfig.set(tofConfig)
# Link the ToF sensor to the ToF node
cam_a.raw.link(tof.input)

tof.depth.link(xout.input)

def get_rgb_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = depth_map[y, x]*0.1
        print(f"RGB Value at ({x}, {y}): {pixel_value}cm")


cv2.namedWindow('Image')
# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras:', device.getConnectedCameraFeatures())
    q = device.getOutputQueue(name="depth")
    print(q)

    while True:
        imgFrame = q.get()  
        # blocking call, will wait until a new data has arrived
        depth_map = imgFrame.getFrame()
        # Colorize the depth frame to jet colormap
        depth_downscaled = depth_map[::4]
        non_zero_depth = depth_downscaled[depth_downscaled != 0] # Remove invalid depth values
        if len(non_zero_depth) == 0:
            min_depth, max_depth = 0, 0
        else:
            min_depth = np.percentile(non_zero_depth, 3)
            max_depth = np.percentile(non_zero_depth, 97)
        depth_colorized = np.interp(depth_map, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        cv2.setMouseCallback('Image', get_rgb_value)
        key = cv2.waitKey(1)

        depth_colorized = cv2.applyColorMap(depth_colorized, cv2.COLORMAP_JET)
        cv2.imshow('Image', depth_colorized)

        if key == ord('q'):
            break