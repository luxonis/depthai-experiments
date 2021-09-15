#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}")

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xrgbOut = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()
xoutRectifR = pipeline.createXLinkOut()

# XLinkOut
xrgbOut.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutRectifR.setStreamName("rectified_right")
STREAMS = ("depth", "rgb", "rectified_right")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setIspScale(1, 3)
camRgb.setInterleaved(False)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# StereoDepth
stereo.initialConfig.setConfidenceThreshold(230)
stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
stereo.setLeftRightCheck(True)
# stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# Linking
camRgb.video.link(xrgbOut.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.depth.link(xoutDepth.input)
stereo.rectifiedRight.link(xoutRectifR.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calibData = device.readCalibration()

    rgb_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 1280, 720))
    pcl_converter = PointCloudVisualizer(rgb_intrinsic, 1280, 720)

    q_list = tuple(device.getOutputQueue(name=s, maxSize=8, blocking=False) for s in STREAMS)
    pcl_frames = [None, None] # depth, rectified_right frames

    while True:
        for i, q in enumerate(q_list):
            name = q.getName()
            data = q.get()

            if name == "rgb":
                frame = cv2.cvtColor(data.getCvFrame(), cv2.COLOR_BGR2RGB)
            elif name == "depth":
                frame = data.getCvFrame().astype(np.uint16)
            else:
                frame = data.getFrame()
            cv2.imshow(name, frame)

            if i < 2:
                pcl_frames[i] = frame
        
        pcl_converter.rgbd_to_projection(pcl_frames[0], pcl_frames[1], True)
        pcl_converter.visualize_pcd()

        if cv2.waitKey(1) == ord('q'):
            break