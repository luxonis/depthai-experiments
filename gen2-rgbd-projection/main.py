#!/usr/bin/env python3

import argparse
import cv2
import depthai as dai
import numpy as np
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--align-depth', dest='align_depth', action='store_true')
parser.add_argument('--no-pcl', dest='no_pcl', action='store_true')
parser.set_defaults(align_depth=False, no_pcl=False)

args = parser.parse_args()
align_depth = args.align_depth
no_pcl = args.no_pcl

if not no_pcl:
    try:
        from projector_3d import PointCloudVisualizer
    except ImportError as e:
        raise ImportError(
            f"\033[1;5;31mError occured when importing PCL projector: {e}")

############################################################################
# USER CONFIGURABLE PARAMETERS (also see configureDepthPostProcessing())

# Select color and depth image resolution you want (Not all cameras support all resolutions)
rgb_resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
depth_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

# parameters to speed up visualization
stride = 4  # skip points in the depth image when projecting to 3d pointcloud, only matters if align_depth == False
downsample_pcl = True  # downsample the pointcloud before operating on it and visualizing

# StereoDepth config options.
# whether or not to align the depth image on host (As opposed to on device), only matters if align_depth = True
align_on_host = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels
LRcheckthresh = 5
confidenceThreshold = 200
min_depth = 400  # mm
max_depth = 20000  # mm
speckle_range = 60
############################################################################

if (not lrcheck and not align_on_host) and align_depth:
    print("Warning: Cannot align on device without lr check!")
    align_on_host = True


def configureDepthPostProcessing(stereoDepthNode):
    """
    In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps.
    """
    stereoDepthNode.initialConfig.setConfidenceThreshold(confidenceThreshold)
    stereoDepthNode.initialConfig.setLeftRightCheckThreshold(LRcheckthresh)
    # stereoDepthNode.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)
    # stereoDepthNode.initialConfig.setBilateralFilterSigma(16)
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
    config.censusTransform.enableMeanMode = True
    config.costMatching.linearEquationParameters.alpha = 0
    config.costMatching.linearEquationParameters.beta = 2
    stereoDepthNode.initialConfig.set(config)
    stereoDepthNode.setLeftRightCheck(lrcheck)
    stereoDepthNode.setExtendedDisparity(extended)
    stereoDepthNode.setSubpixel(subpixel)
    stereoDepthNode.setRectifyEdgeFillColor(
        0)  # Black, to better see the cutout


def create_rgbd_pipeline():
    pipeline = dai.Pipeline()

    # Define sources
    camRgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    # Define outputs
    depthOut = pipeline.createXLinkOut()
    depthOut.setStreamName("depth")
    rgbOut = pipeline.createXLinkOut()
    rgbOut.setStreamName("rgb")

    # Configure Camera Properties
    camRgb.setResolution(rgb_resolution)
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    # Fixing focus to value used during calibration may improve alignment with depth.
    # I didnt find it essential but your mileage may vary.
    # you can use this function inside a device to get the lens position
    # lensPos = device.readCalibration().getLensPosition(dai.CameraBoardSocket.RGB)
    # camRgb.initialControl.setManualFocus(lensPos)

    mono_camera_resolution = depth_resolution
    left.setResolution(mono_camera_resolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(mono_camera_resolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # This block is all post-processing to depth image
    configureDepthPostProcessing(stereo)
    if not align_on_host and align_depth:
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking device side outputs to host side
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)
    camRgb.video.link(rgbOut.input)

    # Book-keeping
    streams = [depthOut.getStreamName(), rgbOut.getStreamName()]
    rgb_image_size = (camRgb.getResolutionWidth(), camRgb.getResolutionHeight())
    depth_image_size = (right.getResolutionWidth(), right.getResolutionHeight())

    return pipeline, streams, rgb_image_size, depth_image_size


if __name__ == "__main__":
    print("Initialize pipeline")
    print("Align Depth: {}".format(align_depth))
    print("Align on Host: {}".format(align_on_host))
    print("Visualize Pointcloud: {}".format(not no_pcl))
    pipeline, streams, rgb_image_size, depth_image_size = create_rgbd_pipeline()

    # Connect to device and start pipeline
    print("Opening device")
    with dai.Device(pipeline) as device:
        # get the camera calibration info
        calibData = device.readCalibration()
        # RIGHT CAMERA INFO
        right_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, *depth_image_size))
        right_distortion = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
        # This is needed because the depth image is made from the rectified right camera image, not the right camera image
        # (although in practice, I did not see a big difference)
        right_rotation = np.array(calibData.getStereoRightRectificationRotation())
        right_homography = np.matmul(np.matmul(right_intrinsic, right_rotation), np.linalg.inv(right_intrinsic))
        inverse_rectified_right_intrinsic = np.matmul(np.linalg.inv(right_intrinsic), np.linalg.inv(right_homography))
        rectified_right_intrinsic = np.linalg.inv(inverse_rectified_right_intrinsic)
        # COLOR CAMERA INFO
        _, rgb_default_width, rgb_default_height = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        rgb_orig_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB))
        # update camera intrinsics based on new image size
        rgb_intrinsic = rgb_orig_intrinsic.copy()
        width_scaling = depth_image_size[0] / rgb_default_width
        height_scaling = depth_image_size[1] / rgb_default_height
        rgb_intrinsic[0, :] *= width_scaling
        rgb_intrinsic[1, :] *= height_scaling
        rgb_distortion = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        # RIGHT --> RGB CAMERA INFO
        right_to_rgb_extrinsic = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.RGB))

        if align_depth:
            # setup cv window
            cv2.namedWindow("RGBD Alignment")
        else:
            # setup cv window
            cv2.namedWindow("Depth Image")

        if not no_pcl:
            # setup pcl converter
            if align_depth:
                pcl_converter = PointCloudVisualizer(rgb_intrinsic, *depth_image_size)
            else:
                pcl_converter = PointCloudVisualizer(rectified_right_intrinsic, *depth_image_size)

        # setup bookkeeping variables
        pcl_frames = [None, None]  # depth frame, color frame
        queue_list = [device.getOutputQueue(stream, maxSize=8, blocking=False) for stream in streams]
        pixel_coords = utils.pixel_coord_np(*depth_image_size).astype(float)

        # main stream loop
        print("Begin streaming at resolution: {} x {}".format(*depth_image_size))
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                if streams[i] == "rgb":
                    data, w, h = image.getData(), image.getWidth(), image.getHeight()
                    assert (w == rgb_image_size[0] and h == rgb_image_size[1]), \
                        "Rgb image does not match user-specified resolution! " \
                            "Please ensure the rgb_resolution you set is supported by this device."
                    yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
                    # pcl_frames[i] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                    pcl_frames[i] = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
                    pcl_frames[i] = cv2.resize(pcl_frames[i], depth_image_size, cv2.INTER_CUBIC)
                else:
                    depth_frame = np.array(image.getFrame())
                    h,w = depth_frame.shape
                    if (align_on_host and align_depth) or (not align_depth):
                        assert (w == depth_image_size[0] and h == depth_image_size[1]), \
                            "Depth image does not match user-specified resolution! " \
                                "Please ensure the depth_resolution you set is supported by this device."
                    if align_depth:
                        if not align_on_host:
                            pcl_frames[i] = cv2.resize(depth_frame, depth_image_size, cv2.INTER_NEAREST)
                        else:
                            # Approach 1: Use opencv's register depth method
                            depth_scale = 10.0  # extrinsic is in cm, but depth in mm so convert
                            depth_dilation = True
                            scaled_depth_frame = depth_frame.astype(float) / depth_scale
                            pcl_frames[i] = cv2.rgbd.registerDepth(
                                rectified_right_intrinsic, rgb_intrinsic, rgb_distortion, \
                                right_to_rgb_extrinsic, scaled_depth_frame, depth_image_size, \
                                depth_dilation)
                            pcl_frames[i] = np.array(pcl_frames[i] * depth_scale).astype(np.uint16)  # convert back from cm to mm
                            # Approach 2: Use slow but easy to interpert function
                            # pcl_frames[i] = utils.alignDepthToRGB(depth_frame, pixel_coords, inverse_rectified_right_intrinsic, rgb_intrinsic, right_to_rgb_extrinsic, *depth_image_size)
                    else:
                        pcl_frames[i] = depth_frame

            if any([frame is None for frame in pcl_frames]) and align_depth:
                print("Error: Need rgb AND depth frame to align!")
                continue
            elif pcl_frames[0] is None:
                print("Waiting on depth image to visualize")
                continue

            if align_depth:
                # Convert nonzero depth into red pixels in 3-channel image for visualization purposes
                blended_image = utils.overlayDepthFrame(rgb_frame=pcl_frames[1], depth_frame=pcl_frames[0], rgb_alpha=0.4)
                cv2.imshow("RGBD Alignment", blended_image)
                if not no_pcl:
                    pcl_converter.rgbd_to_projection(*pcl_frames, downsample=downsample_pcl)
                    pcl_converter.visualize_pcd()
            else:
                # depth_scale_factor was selected by some trial and error
                quantized_depth_image = utils.quantizeDepthFrame(pcl_frames[0], depth_scale_factor=5000)
                cv2.imshow("Depth Image", quantized_depth_image)
                if not no_pcl:
                    pcl_converter.depth_to_projection(pcl_frames[0], stride=stride, downsample=downsample_pcl)
                    pcl_converter.visualize_pcd()
            if cv2.waitKey(1) == "q":
                break
