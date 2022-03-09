#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import models.kornia_depth_to_3d
import torch
try:
    from projector_device import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occured when importing PCL projector: {e}")

############################################################################
# USER CONFIGURABLE PARAMETERS (also see configureDepthPostProcessing())

# Depth resolution
resolution = (640,400)
# Other options:
# resolution = (640,480) # OAK-D-Lite
# resolution = (1280, 720)
# resolution = (1280, 800)

# parameters to speed up visualization
downsample_pcl = True  # downsample the pointcloud before operating on it and visualizing

# StereoDepth config options.
# whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels

def configureDepthPostProcessing(stereoDepthNode):
    """
    In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps.
    """
    stereoDepthNode.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
    config = stereoDepthNode.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 60
    config.postProcessing.temporalFilter.enable = True

    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 700  # mm
    config.postProcessing.thresholdFilter.maxRange = 4000  # mm
    # config.postProcessing.decimationFilter.decimationFactor = 1
    config.censusTransform.enableMeanMode = True
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2
    stereoDepthNode.initialConfig.set(config)
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)
    stereoDepthNode.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

def get_resolution(width):
    if width==480: return dai.MonoCameraProperties.SensorResolution.THE_480_P
    elif width==720: return dai.MonoCameraProperties.SensorResolution.THE_720_P
    elif width==800: return dai.MonoCameraProperties.SensorResolution.THE_800_P
    else: return dai.MonoCameraProperties.SensorResolution.THE_400_P

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = pipeline.createColorCamera()
camRgb.setIspScale(1,3)

rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("rgb")
camRgb.isp.link(rgbOut.input)

# Configure Camera Properties
left = pipeline.createMonoCamera()
left.setResolution(get_resolution(resolution[1]))
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(get_resolution(resolution[1]))
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
configureDepthPostProcessing(stereo)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Depth -> PointCloud
nn = pipeline.createNeuralNetwork()
stereo.depth.link(nn.inputs["depth"])
# nn.inputs["depth"].setWaitForMessage(False)
# nn.inputs["depth"].setWaitForMessage(True)

calib_in = pipeline.createXLinkIn()
calib_in.setStreamName("calib_in")
calib_in.out.link(nn.inputs["matrix"])
# Only send calibration data once, and always reuse the message
nn.inputs["matrix"].setReusePreviousMessage(True)

pointsOut = pipeline.createXLinkOut()
pointsOut.setStreamName("pcl")
nn.out.link(pointsOut.input)


if __name__ == "__main__":
    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device() as device:
        # device.setIrLaserDotProjectorBrightness(400)
        device.setLogLevel(dai.LogLevel.ERR)

        blobPath = models.kornia_depth_to_3d.getPath(resolution)
        nn.setBlobPath(blobPath)
        device.startPipeline(pipeline)

        # baseline in mm
        calibData = device.readCalibration()
        M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
            dai.Size2f(resolution[0], resolution[1]),
        )

        print(torch.Tensor([M_right]))

        # self.camera_matrix = torch.Tensor([M_right])
        matrix = np.array([M_right], dtype=np.float16).flatten()
        print(matrix)
        buff = dai.NNData()
        buff.setLayer("matrix", matrix)
        device.getInputQueue("calib_in").send(buff)

        pcl_converter = PointCloudVisualizer()
        queue = device.getOutputQueue("pcl", maxSize=8, blocking=False)
        qRgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)

        # main stream loop
        while True:
            inRgb = qRgb.tryGet()
            if inRgb is not None:
                cv2.imshow("color", inRgb.getCvFrame())

            pcl_data = np.array(queue.get().getData()).view(np.float16).reshape(1, 3, resolution[1], resolution[0])
            # print(pcl_data)
            pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64)
            pcl_converter.visualize_pcl(pcl_data, downsample=downsample_pcl)

            if cv2.waitKey(1) == "q":
                pcl_converter.close_window()
                break
