#!/usr/bin/env python3

"""

Example of visualizing a pointcloud generated from a depth image (no color).
Depending on you min/max depth you may need to zoom in on the o3d viewer as it will initially be zoomed out.

"""

import cv2
import numpy as np
import depthai as dai

try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occured when importing PCL projector: {e}")


# StereoDepth config options.
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
LRcheckthresh = 5
confidenceThreshold = 200
stride = 4  # increase this if your computer can't handle deprojecting all the depth pixels into 3d
min_depth = 400  # mm
max_depth = 15000  # mm
speckle_range = 60

# Select camera resolution
res = {"height": 480, "width": 640,
       "THE_P": dai.MonoCameraProperties.SensorResolution.THE_480_P}
# res = {"height": 720, "width": 1080, "THE_P": dai.MonoCameraProperties.SensorResolution.THE_720_P}

def configureDepthPostProcessing(stereoDepthNode):
    """
    In-place post-processing configuration for a stereo depth node
    """
    stereoDepthNode.initialConfig.setConfidenceThreshold(confidenceThreshold)
    stereoDepthNode.initialConfig.setLeftRightCheckThreshold(LRcheckthresh)
    config = stereoDepthNode.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = speckle_range
    # config.postProcessing.temporalFilter.enable = True
    # config.postProcessing.spatialFilter.enable = True
    # config.postProcessing.spatialFilter.holeFillingRadius = 2
    # config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = min_depth  # mm
    config.postProcessing.thresholdFilter.maxRange = max_depth  # mm
    config.postProcessing.decimationFilter.decimationFactor = 1
    # config.censusTransform.enableMeanMode = True
    # config.costMatching.linearEquationParameters.alpha = 0
    # config.costMatching.linearEquationParameters.beta = 2
    stereoDepthNode.initialConfig.set(config)
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)
    stereoDepthNode.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

def create_depth_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth")

    # Properties
    mono_camera_resolution = res["THE_P"]
    left.setResolution(mono_camera_resolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(mono_camera_resolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    configureDepthPostProcessing(stereo)

    # Linking
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    streams = [depthOut.getStreamName()]
    max_disparity = stereo.initialConfig.getMaxDisparity()

    return pipeline, streams, max_disparity

def getDisparityFrame(frame, max_disparity):
    color_levels = 16
    disp = (frame.astype(float).copy() * (color_levels / max_disparity)).astype(np.uint8)
    disp = (255 * disp.astype(float) / color_levels).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp

if __name__ == "__main__":
    pipeline, streams, max_disparity = create_depth_pipeline()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        calib_data = device.readCalibration()
        depth_intrinsic = np.array(calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT, res["width"], res["height"]))

        pcl_converter = PointCloudVisualizer(
            depth_intrinsic, res["width"], res["height"])

        pcl_frames = [None]  # depth frame
        queue_list = [device.getOutputQueue(
            stream, maxSize=8, blocking=False) for stream in streams]
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                pcl_frames[i] = np.array(image.getFrame())
            if all([frame is not None for frame in pcl_frames]):
                depth_color_map = getDisparityFrame(pcl_frames[0], max_disparity)
                # visualize image
                cv2.imshow("Depth Image", depth_color_map)
                if cv2.waitKey(1) == "q":
                    break
                # visualize pointcloud
                pcl_converter.depth_to_projection(
                    pcl_frames[0], stride=stride, downsample=True)
            pcl_converter.visualize_pcd()
