#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np


# Start defining a pipeline
pipeline = dai.Pipeline()


lrcheck = False   # Better handling for occlusions

# Define a source - mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.createStereoDepth()
stereo.setOutputRectified(True) # The rectified streams are horizontally mirrored by default
stereo.setOutputDepth(False)
stereo.setConfidenceThreshold(255)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
stereo.setLeftRightCheck(lrcheck)


left.out.link(stereo.left)
right.out.link(stereo.right)

# Create outputs
xoutDisparity = pipeline.createXLinkOut()
xoutDisparity.setStreamName("depth")

stereo.disparity.link(xoutDisparity.input)

xoutRectifiedRight = pipeline.createXLinkOut()
xoutRectifiedRight.setStreamName("rectifiedRight")
stereo.rectifiedRight.link(xoutRectifiedRight.input)


class trackbar:
    def __init__(self, trackbarName, windowName, minValue, maxValue, defaultValue, handler):
        cv2.createTrackbar(trackbarName, windowName, minValue, maxValue, handler)
        cv2.setTrackbarPos(trackbarName, windowName, defaultValue)

class wlsFilter:
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
        self.lambdaTrackbar = trackbar('Lambda', self.wlsStream, 0, 255, 80, self.on_trackbar_change_lambda)
        self.sigmaTrackbar  = trackbar('Sigma',  self.wlsStream, 0, 100, 15, self.on_trackbar_change_sigma)

    def filter(self, disparity, right, left=None):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
        self.wlsFilter.setLambda(self._lambda)
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
        self.wlsFilter.setSigmaColor(self._sigma)
        filteredDisp = self.wlsFilter.filter(disparity, right)
        cv2.imshow(self.wlsStream, filteredDisp)

        cv2.normalize(filteredDisp, filteredDisp, 0, 255, cv2.NORM_MINMAX)
        coloredDisp = cv2.applyColorMap(filteredDisp, cv2.COLORMAP_JET)
        cv2.imshow(self.wlsStream + "_color", coloredDisp)


wlsFilter = wlsFilter(_lambda=8000, _sigma=1.5)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
    qRight = device.getOutputQueue("rectifiedRight", maxSize=4, blocking=False)
    qDisparity = device.getOutputQueue("depth", maxSize=4, blocking=False)

    rightFrame = None
    disparityFrame = None

    while True:
        inRight = qRight.get()
        inDisparity = qDisparity.get()

        rightFrame = inRight.getFrame()
        rightFrame = cv2.flip(rightFrame, flipCode=1)
        cv2.imshow("rectified right", rightFrame)


        disparityFrame = inDisparity.getFrame()
        cv2.imshow("disparity", disparityFrame)

        wlsFilter.filter(disparityFrame, rightFrame)
       

        if cv2.waitKey(1) == ord('q'):
            break
