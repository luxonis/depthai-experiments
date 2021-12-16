#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}")


# StereoDepth config options. TODO move to command line options
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
LRchecktresh = 5
confidenceThreshold = 200

def create_rgb_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    depthOut = pipeline.createXLinkOut()
    rgbOut = pipeline.createXLinkOut()

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setIspScale(1, 3)
    camRgb.setInterleaved(False)

    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.initialConfig.setConfidenceThreshold(confidenceThreshold)
    stereo.initialConfig.setLeftRightCheckThreshold(LRchecktresh);
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)  # NOTE: Subpixel cannot be enabled, since RGB stream is used
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    rgbOut.setStreamName("rgb")
    depthOut.setStreamName("depth")

    # Linking
    camRgb.video.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    streams = ["rgb", "depth"]
    maxDisparity = stereo.getMaxDisparity()

    return pipeline, streams, maxDisparity


def display_pointcloud(frameRgb, frameDepth):
    frameRgb = cv2.cvtColor(frameRgb, cv2.COLOR_BGR2RGB)
    pcl_converter.rgbd_to_projection(frameDepth, frameRgb, True)
    pcl_converter.visualize_pcd()


pipeline, streams, MAX_DISPARIRTY = create_rgb_pipeline()


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calibData = device.readCalibration()

    rgb_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 1280, 720))
    pcl_converter = PointCloudVisualizer(rgb_intrinsic, 1280, 720)
    pcl_frames = [None, None]  # rgb and depth frame
    queue_list = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]

    while True:
        for i, queue in enumerate(queue_list):
            name = queue.getName()
            image = queue.get()

            pcl_frames[i] = image.getCvFrame() if name == "rgb" else np.ascontiguousarray(image.getFrame())
            cv2.imshow(name, pcl_frames[i])

        display_pointcloud(*pcl_frames)

        if cv2.waitKey(1) == ord("q"):
            break
