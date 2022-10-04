#!/usr/bin/env python3

import cv2
import depthai as dai
from projector_3d import PointCloudVisualizer


COLOR = True

lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 50
config.postProcessing.thresholdFilter.maxRange = 2000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)
stereo.initialConfig.setConfidenceThreshold(50)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName('depth')
stereo.depth.link(xout_depth.input)

xout_confidence = pipeline.createXLinkOut()
xout_confidence.setStreamName('confidence')
stereo.confidenceMap.link(xout_confidence.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName('colorize')
if COLOR:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)


with dai.Device(pipeline) as device:

    device.setIrLaserDotProjectorBrightness(1200)
    depth_queue = device.getOutputQueue("depth", maxSize=1, blocking=False)
    rgb_queue = device.getOutputQueue("colorize", maxSize=1, blocking=False)
    confidence_queue = device.getOutputQueue("confidence", maxSize=1, blocking=False)


    calibData = device.readCalibration()
    if COLOR:
        w, h = camRgb.getIspSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
    else:
        w, h = monoRight.getResolutionSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
    pcl_converter = PointCloudVisualizer(intrinsics, None, w, h)

    depth_frame = None
    rgb_frame = None
    confidence_frame = None
    while True:
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        depth_in = depth_queue.tryGet()
        rgb_in = rgb_queue.tryGet()
        confidence_in = confidence_queue.tryGet()

        if depth_in is not None:
            depth_frame = depth_in.getFrame()

        if rgb_in is not None:
            rgb_frame = rgb_in.getCvFrame()

        if confidence_in is not None:
            confidence_frame = confidence_in.getFrame()

        if depth_frame is None and rgb_frame is None:
            continue

        cv2.imshow("color", rgb_frame)

        depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
        cv2.imshow("depth", depth_frame_color)

        confidence_frame_color = cv2.normalize(confidence_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        confidence_frame_color = cv2.equalizeHist(confidence_frame_color)
        confidence_frame_color = cv2.applyColorMap(confidence_frame_color, cv2.COLORMAP_HOT)
        cv2.imshow("confidence", confidence_frame_color)
        rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        pcl_converter.rgbd_to_projection(depth_frame, rgb)
        pcl_converter.visualize_pcd()

