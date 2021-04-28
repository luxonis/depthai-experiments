#!/usr/bin/env python3

import cv2
import depthai as dai
import cv2
import depthai as dai
import numpy as np
import math

# -------------------------
# Change this if needed:
# -------------------------


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

    def filter(self, disparity, right, depthScaleFactor):
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L92
        self.wlsFilter.setLambda(self._lambda)
        # https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/include/opencv2/ximgproc/disparity_filter.hpp#L99
        self.wlsFilter.setSigmaColor(self._sigma)
        filteredDisp = self.wlsFilter.filter(disparity, right)
        

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            # raw depth values
            depthFrame = (depthScaleFactor / filteredDisp).astype(np.uint16)


        return filteredDisp, depthFrame
       



FPS = 30
# Save every 30th frame:



# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createColorCamera()
stereo = pipeline.createStereoDepth()

xoutDisparity = pipeline.createXLinkOut()
xoutRectifiedRight = pipeline.createXLinkOut()

xoutDisparity.setStreamName("disparity")
xoutRectifiedRight.setStreamName("rectRight")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(FPS)
monoLeft.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG);

monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
monoRight.setBoardSocket(dai.CameraBoardSocket.RGB)
monoRight.setFps(FPS)
monoRight.setIspScale(1, 3)
monoRight.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG);
monoRight.initialControl.setManualFocus(135)


lrcheck = False
subpixel = False

# StereoDepth
stereo.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.isp.link(stereo.right)

stereo.disparity.link(xoutDisparity.input)
stereo.rectifiedRight.link(xoutRectifiedRight.input)

wlsFilter = wlsFilter(_lambda=8000, _sigma=1.5)

baseline = 3.25 #mm
disp_levels = 96
fov = 68.7

with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the grayscale / depth frames and nn data from the outputs defined above
    qRight = device.getOutputQueue("rectRight", maxSize=4, blocking=False)
    qDisparity = device.getOutputQueue("disparity", maxSize=4, blocking=False)

    rightFrame = None
    disparityFrame = None

    while True:
        inRight = qRight.get()
        inDisparity = qDisparity.get()

        # rightFrame = inRight.getFrame()
        data, w, h = inRight.getData(), inRight.getWidth(), inRight.getHeight()
        rightFrame = np.array(data).reshape((h, w)).astype(np.uint8)

        rightFrame = cv2.flip(rightFrame, flipCode=1)
        cv2.imshow("rectified right", rightFrame)


        disparityFrame = inDisparity.getFrame()
        cv2.imshow("disparity", disparityFrame)
        
        
        # coloredDisp = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)
        # cv2.imshow("colored raw disp", coloredDisp)


        focal = disparityFrame.shape[1] / (2. * math.tan(math.radians(fov / 2)))
        # print(focal)
        depthScaleFactor = baseline * focal

        filteredDisp, depthFrame = wlsFilter.filter(disparityFrame, rightFrame, depthScaleFactor)

        cv2.imshow("wls raw depth", depthFrame)
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        cv2.imshow("wls depth colorized", depthFrameColor)
        
        filteredDisp = (filteredDisp * (255/(disp_levels-1))).astype(np.uint8)
        cv2.imshow(wlsFilter.wlsStream, filteredDisp)

        coloredDisp = cv2.applyColorMap(filteredDisp, cv2.COLORMAP_JET)
        cv2.imshow("wls colored disp", coloredDisp)

        if cv2.waitKey(1) == ord('q'):
            break