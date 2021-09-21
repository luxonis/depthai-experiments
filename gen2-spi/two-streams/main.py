#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from time import sleep
import argparse

# Start defining a pipeline
pipeline = dai.Pipeline()

# Create color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath("mobilenet-ssd_openvino_2021.2_6shave.blob")
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
camRgb.preview.link(nn.input)

# Send mobilenet detections to the host (via XLink) and to MCU (via SPI)
nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

nnSpi = pipeline.createSPIOut()
nnSpi.setStreamName("nn")
nnSpi.setBusId(0)
nn.out.link(nnSpi.input)

# Create mono cameras for StereoDepth
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# StereoDepth
stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(245)

left.out.link(stereo.left)
right.out.link(stereo.right)

spatialLocationCalculator = pipeline.createSpatialLocationCalculator()
spatialLocationCalculator.setWaitForConfigInput(False)

# Link StereoDepth to spatialLocationCalculator
stereo.depth.link(spatialLocationCalculator.inputDepth)
# Set initial config for the spatialLocationCalculator
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
# Set ROI
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)

# Send depth frames to the host
xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)

# Send spatialLocationCalculator data to the host through the XLink
xoutSpatialData = pipeline.createXLinkOut()
xoutSpatialData.setStreamName("spatialData")
spatialLocationCalculator.out.link(xoutSpatialData.input)
# Send spatialLocationCalculator data through the SPI
spiOutSpatialData = pipeline.createSPIOut()
spiOutSpatialData.setStreamName("spatialData")
spiOutSpatialData.setBusId(0)
spatialLocationCalculator.out.link(spiOutSpatialData.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the depth frames from the outputs defined above
    nnQ = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    spatialQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    while True:
        nnData = nnQ.tryGet()
        if nnData is not None:
            # We could also decode the NN data (ROI, label, conf)
            # like we do on ESP32
            print(nnData.getData())

        inSpatialDatas = spatialQueue.get() # blocking call, will wait until a new data has arrived
        spatialDatas = inSpatialDatas.getSpatialLocations()

        for spatialData in spatialDatas:
            roi = spatialData.config.roi
            roi = roi.denormalize(width=left.getResolutionWidth(), height=left.getResolutionHeight())
            avgDepth = spatialData.depthAverage # same as Z coordinate
            avgDepthPixelCount = spatialData.depthAveragePixelCount
            depthConfidence = avgDepthPixelCount/roi.area()
            # print(f"Depth confidence: {int(depthConfidence*100)}%")
            x = spatialData.spatialCoordinates.x
            y = spatialData.spatialCoordinates.y
            z = spatialData.spatialCoordinates.z
            euclideanDistance = np.sqrt(x*x + y*y + z*z)
            print(f"Euclidean distance {int(euclideanDistance)} mm, X: {int(x)} mm, Y: {int(y)} mm, Z: {int(z)} mm")

        depthFrame = depthQueue.get().getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        color = (255, 255, 255)
        for spatialData in spatialDatas:
            roi = spatialData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            avgDepth = spatialData.depthAverage # same as Z coordinate
            avgDepthPixelCount = spatialData.depthAveragePixelCount
            depthConfidence = avgDepthPixelCount/roi.area()
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(spatialData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(spatialData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(spatialData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

            cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord('q'):
            break
