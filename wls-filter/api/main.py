#!/usr/bin/env python3

import math

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

lrcheck = False  # Better handling for occlusions

# Define a source - mono (grayscale) cameras
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(255)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
stereo.setLeftRightCheck(lrcheck)

left.out.link(stereo.left)
right.out.link(stereo.right)

# Create outputs
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDisparity.setStreamName("depth")
stereo.disparity.link(xoutDisparity.input)

xoutRectifiedRight = pipeline.create(dai.node.XLinkOut)
xoutRectifiedRight.setStreamName("rectifiedRight")
stereo.rectifiedRight.link(xoutRectifiedRight.input)


class Trackbar:
    def __init__(self, trackbarName, windowName, minValue, maxValue, defaultValue, handler):
        cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, handler)
        cv2.setTrackbarPos(trackbarName, windowName, defaultValue)


class WLSFilter:
    wlsStream = "wlsFilter"

    def on_trackbar_change_lambda(self, value):
        self._lambda = value * 100

    def on_trackbar_change_sigma(self, value):
        self._sigma = value / float(10)

    def __init__(self, _lambda, _sigma):
        self._lambda = _lambda
        self._sigma = _sigma
        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        cv2.namedWindow(self.wlsStream)
        self.lambdaTrackbar = Trackbar('Lambda', self.wlsStream, 0, 255, 80, self.on_trackbar_change_lambda)
        self.sigmaTrackbar = Trackbar('Sigma', self.wlsStream, 0, 100, 15, self.on_trackbar_change_sigma)

    def filter(self, disparity, right, depthScaleFactor):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
        self.wlsFilter.setLambda(self._lambda)
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
        self.wlsFilter.setSigmaColor(self._sigma)
        filteredDisp = self.wlsFilter.filter(disparity, right)

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            # raw depth values
            depthFrame = (depthScaleFactor / filteredDisp).astype(np.uint16)

        return filteredDisp, depthFrame


wlsFilter = WLSFilter(_lambda=8000, _sigma=1.5)

baseline = 75  # mm
disp_levels = 96
fov = 71.86

# Pipeline defined, now the device is connected to
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError(
            "Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams)
        )
    device.startPipeline(pipeline)
    # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
    qRight = device.getOutputQueue("rectifiedRight", maxSize=4, blocking=False)
    qDisparity = device.getOutputQueue("depth", maxSize=4, blocking=False)

    disp_multiplier = 255 / stereo.getMaxDisparity()

    while True:
        rightFrame = qRight.get().getFrame()
        disparityFrame = qDisparity.get().getFrame()

        cv2.imshow("rectified right", rightFrame)
        cv2.imshow("disparity", disparityFrame)

        focal = disparityFrame.shape[1] / (2. * math.tan(math.radians(fov / 2)))
        depthScaleFactor = baseline * focal

        filteredDisp, depthFrame = wlsFilter.filter(disparityFrame, rightFrame, depthScaleFactor)

        cv2.imshow("wls raw depth", depthFrame)

        filteredDisp = (filteredDisp * disp_multiplier).astype(np.uint8)
        cv2.imshow(wlsFilter.wlsStream, filteredDisp)

        coloredDisp = cv2.applyColorMap(filteredDisp, cv2.COLORMAP_HOT)
        cv2.imshow("wls colored disp", coloredDisp)

        if cv2.waitKey(1) == ord('q'):
            break
