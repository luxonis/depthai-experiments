#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import os

try:
    from projector_device import PointCloudVisualizer
except ImportError as e:
    raise ImportError(
        f"\033[1;5;31mError occured when importing PCL projector: {e}")

############################################################################
# USER CONFIGURABLE PARAMETERS (also see configureDepthPostProcessing())

COLOR = True # Stream & display color frames

# Depth resolution
resolution = (640,400) # 24 FPS (without visualization)

# parameters to speed up visualization
downsample_pcl = True  # downsample the pointcloud before operating on it and visualizing

# StereoDepth config options.
# whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels

def create_xyz(width, height, camera_matrix):
    xs = np.linspace(0, width - 1, width, dtype=np.float32)
    ys = np.linspace(0, height - 1, height, dtype=np.float32)

    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys)) # WxHx2
    points_2d = base_grid.transpose(1, 2, 0) # 1xHxWx2

    # unpack coordinates
    u_coord: np.array = points_2d[..., 0]
    v_coord: np.array = points_2d[..., 1]

    # unpack intrinsics
    fx: np.array = camera_matrix[0, 0]
    fy: np.array = camera_matrix[1, 1]
    cx: np.array = camera_matrix[0, 2]
    cy: np.array = camera_matrix[1, 2]

    # projective
    x_coord: np.array = (u_coord - cx) / fx
    y_coord: np.array = (v_coord - cy) / fy

    xyz = np.stack([x_coord, y_coord], axis=-1)
    return np.pad(xyz, ((0,0),(0,0),(0,1)), "constant", constant_values=1.0)

def getPath(resolution):
    (width, heigth) = resolution
    path = Path("models", "out")
    path.mkdir(parents=True, exist_ok=True)
    name = f"pointcloud_{width}x{heigth}"

    return_path = str(path / (name + '.blob'))
    if os.path.exists(return_path):
        return return_path

    # Model doesn't exist, create it
    import models.depth_to_3d
    return models.depth_to_3d.createBlob(resolution, path, name)

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

if COLOR:
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
nn.setBlobPath(getPath(resolution))
stereo.depth.link(nn.inputs["depth"])

xyz_in = pipeline.createXLinkIn()
xyz_in.setMaxDataSize(6144000)
xyz_in.setStreamName("xyz_in")
xyz_in.out.link(nn.inputs["xyz"])

# Only send xyz data once, and always reuse the message
nn.inputs["xyz"].setReusePreviousMessage(True)

pointsOut = pipeline.createXLinkOut()
pointsOut.setStreamName("pcl")
nn.out.link(pointsOut.input)


if __name__ == "__main__":
    print("Opening device")
    with dai.Device(pipeline) as device:
        # device.setLogLevel(dai.LogLevel.ERR)

        calibData = device.readCalibration()
        M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT,
            dai.Size2f(resolution[0], resolution[1]),
        )

        # Creater xyz data and send it to the device - to the pointcloud generation model (NeuralNetwork node)
        xyz = create_xyz(resolution[0], resolution[1], np.array(M_right).reshape(3,3))
        matrix = np.array([xyz], dtype=np.float16).view(np.int8)
        buff = dai.Buffer()
        buff.setData(matrix)
        device.getInputQueue("xyz_in").send(buff)

        pcl_converter = PointCloudVisualizer()
        queue = device.getOutputQueue("pcl", maxSize=8, blocking=False)
        if COLOR:
            qRgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)

        # main stream loop
        while True:
            if COLOR and qRgb.has():
                cv2.imshow("color", qRgb.get().getCvFrame())

            pcl_data = np.array(queue.get().getFirstLayerFp16()).reshape(1, 3, resolution[1], resolution[0])
            pcl_data = pcl_data.reshape(3, -1).T.astype(np.float64) / 1000.0
            pcl_converter.visualize_pcl(pcl_data, downsample=downsample_pcl)

            if cv2.waitKey(1) == "q":
                pcl_converter.close_window()
                break
