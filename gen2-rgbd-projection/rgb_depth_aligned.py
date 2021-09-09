#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
left = pipeline.createMonoCamera()
right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

rgbOut = pipeline.createXLinkOut()
depthOut = pipeline.createXLinkOut()

rgbOut.setStreamName("rgb")
depthOut.setStreamName("depth")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.initialControl.setManualFocus(130)
camRgb.setIspScale(2, 3)

left.setResolution(monoResolution)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setResolution(monoResolution)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(230)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(depthOut.input) # NOTE: Change to stereo.depth ?

def getDepthFrame(packet):
    frameDepth = packet.getFrame()
    frameDepth = (frameDepth * 255. / maxDisparity).astype(np.uint8)
    frameDepth = cv2.applyColorMap(frameDepth, cv2.COLORMAP_HOT)
    return np.ascontiguousarray(frameDepth)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.getOutputQueue(name="rgb",   maxSize=4, blocking=False)
    device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    maxDisparity = stereo.getMaxDisparity()

    frames = {}
    frames["rgb"] = None
    frames["depth"] = None
    
    while True:
        queueEvents = device.getQueueEvents(("rgb", "depth"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                frames[queueName] = packets[-1].getCvFrame() if queueName == "rgb" else getDepthFrame(packets[-1])
                cv2.imshow(queueName, frames[queueName])

        # Blend when both received
        if frames["rgb"] is not None and frames["depth"] is not None:
            frameRgb, frameDepth = frames["rgb"], frames["depth"]

            # Need to have both frames in BGR format before blending
            if len(frameDepth.shape) < 3:
                frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, 0.6, frameDepth, 0.4 ,0)
            cv2.imshow("rgb-depth", blended)

            frames["rgb"] = frames["depth"] = None

        if cv2.waitKey(1) == ord('q'):
            break
