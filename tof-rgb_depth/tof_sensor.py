#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

cam_a = pipeline.create(dai.node.Camera)
# We assume the ToF camera sensor is on port CAM_A
cam_a.setBoardSocket(dai.CameraBoardSocket.CAM_A)

tof = pipeline.create(dai.node.ToF)

# Configure the ToF node
tofConfig = tof.initialConfig.get()
# tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MIN
tofConfig.depthParams.freqModUsed = dai.RawToFConfig.DepthParams.TypeFMod.MAX
tofConfig.depthParams.avgPhaseShuffle = False
tofConfig.depthParams.minimumAmplitude = 3.0
tof.initialConfig.set(tofConfig)
# Link the ToF sensor to the ToF node
cam_a.raw.link(tof.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("depth")
tof.depth.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras:', device.getConnectedCameraFeatures())
    q = device.getOutputQueue(name="depth")

    while True:
        imgFrame = q.get()  # blocking call, will wait until a new data has arrived
        depth_map = imgFrame.getFrame()
        # Colorize the depth frame to jet colormap
        depth_downscaled = depth_map[::4]
        non_zero_depth = depth_downscaled[depth_downscaled != 0] # Remove invalid depth values
        cv2.imshow("ToF", non_zero_depth)
        if len(non_zero_depth) == 0:
            min_depth, max_depth = 0, 0
        else:
            min_depth = np.percentile(non_zero_depth, 3)
            max_depth = np.percentile(non_zero_depth, 97)
        depth_colorized = np.interp(depth_map, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        cv2.imshow("ToF", depth_colorized)
        depth_colorized = cv2.applyColorMap(depth_colorized, cv2.COLORMAP_JET)

        cv2.imshow("Colorized depth", depth_colorized)

        if cv2.waitKey(1) == ord('q'):
            break