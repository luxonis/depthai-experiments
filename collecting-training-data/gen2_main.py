#!/usr/bin/env python3

import cv2
import depthai as dai

pipeline = dai.Pipeline()

rgb = pipeline.createColorCamera()
rgb.setPreviewSize(300, 300)
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setInterleaved(False)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(255)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
depth.setMedianFilter(median)
depth.setLeftRightCheck(False)
depth.setExtendedDisparity(False)
depth.setSubpixel(False)

left.out.link(depth.left)
right.out.link(depth.right)

# Create output
rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("rgb")
rgb.preview.link(rgbOut.input)
depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("disparity")
depth.disparity.link(depthOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the disparity frames from the outputs defined above
    qDepth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inDepth = qDepth.tryGet()
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            depthFrame = cv2.applyColorMap(depthFrame, cv2.COLORMAP_JET)

            # frame is ready to be shown
            cv2.imshow("disparity", depthFrame)

        inRgb = qRgb.tryGet()
        if inRgb is not None:
            cv2.imshow("bgr", inRgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break