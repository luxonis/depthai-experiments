#!/usr/bin/env python3

"""

Example of visualizing a pointcloud generated from an rgbd image (depth aligned to and combined with color image).
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


# visualization parameter
# instead of showing pointcloud output of alignment, show a cv window of alignment depth blended with rgb image
debug_alignment = True

# StereoDepth config options.
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # True  # Better accuracy for longer distance, fractional disparity 32-levels
LRcheckthresh = 5
confidenceThreshold = 200
min_depth = 400  # mm
max_depth = 1500  # mm
speckle_range = 60
align_on_host = True

if not lrcheck and not align_on_host:
    print("Warning: Cannot align on device without lr check!")
    align_on_host = True

# Select camera resolution (oak-d-lite only supports up to 480_P depth)
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
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    # Fixing focus to value used during calibration may improve alignment with depth.
    # I didnt find it essential but your mileage may vary.
    # you can use this function inside a device to get the lens position
    # lensPos = device.readCalibration().getLensPosition(dai.CameraBoardSocket.RGB)
    # camRgb.initialControl.setManualFocus(lensPos)

    mono_camera_resolution = res["THE_P"]
    left.setResolution(mono_camera_resolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(mono_camera_resolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # This block is all post-processing to depth image
    configureDepthPostProcessing(stereo)
    if not align_on_host:
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking device side outputs to host side
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)
    camRgb.video.link(rgbOut.input)

    # Book-keeping
    streams = [depthOut.getStreamName(), rgbOut.getStreamName()]
    max_disparity = stereo.initialConfig.getMaxDisparity()

    return pipeline, streams, max_disparity


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


# global variables are gross, but this is an experiment so...
pixel_coords = pixel_coord_np(res["width"], res["height"]).astype(float)


def alignDepthToRGB(depth_image, inverse_depth_intrinsic, rgb_intrinsic, depth_to_rgb_extrinsic, rgb_width, rgb_height):
    """
    Align the depth image with the color image. 
    """
    depth_scale = 10.0

    # depth to 3d coordinates [x, y, z]
    cam_coords = np.dot(inverse_depth_intrinsic, pixel_coords) * \
        depth_image.flatten().astype(float) / depth_scale

    # move the depth image 3d coordinates to the rgb camera  location
    cam_coords_homogeneous = np.vstack(
        (cam_coords, np.ones((1, cam_coords.shape[1]))))
    depth_points_homogeneous = np.dot(
        depth_to_rgb_extrinsic, cam_coords_homogeneous)

    # project the 3d depth points onto the rgb image plane
    rgb_frame_ref_cloud = depth_points_homogeneous[:3, :]
    rgb_frame_ref_cloud_normalized = rgb_frame_ref_cloud / \
        rgb_frame_ref_cloud[2, :]
    rgb_image_pts = np.matmul(rgb_intrinsic, rgb_frame_ref_cloud_normalized)
    rgb_image_pts = rgb_image_pts.astype(np.int16)
    u_v_z = np.vstack((rgb_image_pts, rgb_frame_ref_cloud[2, :]))
    lft = np.logical_and(0 <= u_v_z[0], u_v_z[0] < rgb_width)
    rgt = np.logical_and(0 <= u_v_z[1], u_v_z[1] < rgb_height)
    idx_bool = np.logical_and(lft, rgt)
    u_v_z_sampled = u_v_z[:, np.where(idx_bool)]
    y_idx = u_v_z_sampled[1].astype(int)
    x_idx = u_v_z_sampled[0].astype(int)

    # place the valid aligned points into a new depth image
    aligned_depth_image = np.full((rgb_height, rgb_width), 0, dtype=np.uint16)
    aligned_depth_image[y_idx, x_idx] = u_v_z_sampled[3]*depth_scale
    return aligned_depth_image


if __name__ == "__main__":
    pipeline, streams, max_disparity = create_rgbd_pipeline()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # get the camera calibration info
        calibData = device.readCalibration()
        left_intrinsic = np.array(calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.LEFT, res["width"], res["height"]))
        right_intrinsic = np.array(calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT, res["width"], res["height"]))
        right_distortion = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
        rgb_orig_intrinsic = np.array(calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, 1920, 1080))
        rgb_distortion = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
        rgb_intrinsic = cv2.getOptimalNewCameraMatrix(
            rgb_orig_intrinsic, rgb_distortion, (1920, 1080), 0, (res["width"], res["height"]))[0]
        right_to_rgb_extrinsic = np.array(calibData.getCameraExtrinsics(
            dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.RGB))

        inverse_right_intrinsic = np.linalg.inv(right_intrinsic)
        # This seems unneeded but was in the old examples
        # right_rotation = np.array(calibData.getStereoRightRectificationRotation())
        # right_homography = np.matmul(np.matmul(left_intrinsic, right_rotation), np.linalg.inv(right_intrinsic))
        # inverse_right_intrinsic = np.matmul(np.linalg.inv(right_intrinsic), np.linalg.inv(right_homography))

        if debug_alignment:
            # setup cv window
            cv2.namedWindow("RGBD Alignment")
        else:
            # setup pcl converter
            pcl_converter = PointCloudVisualizer(
                rgb_intrinsic, res["width"], res["height"])

        # setup bookkeeping variables
        pcl_frames = [None, None]  # depth frame, color frame
        queue_list = [device.getOutputQueue(
            stream, maxSize=8, blocking=False) for stream in streams]

        # main stream loop
        while True:
            for i, queue in enumerate(queue_list):
                name = queue.getName()
                image = queue.get()
                if streams[i] == "rgb":
                    data, w, h = image.getData(), image.getWidth(), image.getHeight()
                    yuv = np.array(data).reshape(
                        (h * 3 // 2, w)).astype(np.uint8)
                    # pcl_frames[i] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                    pcl_frames[i] = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
                    pcl_frames[i] = cv2.resize(
                        pcl_frames[i], (res["width"], res["height"]), cv2.INTER_CUBIC)
                else:
                    depth_frame = np.array(image.getFrame())
                    if not align_on_host:
                        pcl_frames[i] = cv2.resize(
                            depth_frame, (res["width"], res["height"]), cv2.INTER_NEAREST)
                    else:
                        # Approach 1: Use opencv's register depth method
                        depth_scale = 10.0  # extrinsic is in cm, but depth in mm so convert
                        depth_dilation = True
                        scaled_depth_frame = depth_frame.astype(float) / depth_scale
                        pcl_frames[i] = cv2.rgbd.registerDepth(
                            right_intrinsic, rgb_intrinsic, rgb_distortion, right_to_rgb_extrinsic, scaled_depth_frame, (res["width"], res["height"]), depth_dilation)
                        pcl_frames[i] = np.array(
                            pcl_frames[i] * depth_scale).astype(np.uint16)  # convert back from cm to mm
                        # Approach 2: Use slow but easy to interpert function
                        # pcl_frames[i] = alignDepthToRGB(depth_frame, inverse_right_intrinsic, rgb_intrinsic, right_to_rgb_extrinsic, res["width"], res["height"])

            if any([frame is None for frame in pcl_frames]):
                print("Error: Need rgb AND depth frame to align!")
                continue

            if debug_alignment:
                # Convert nonzero depth into red pixels in 3-channel image
                depth_three_channel = np.zeros_like(pcl_frames[1])
                depth_three_channel[:, :, 2] = (
                    255 * pcl_frames[0].astype(float) / max_disparity).astype(np.uint8)
                cond = depth_three_channel[:, :, 2] > 0
                depth_three_channel[cond, 2] = 255
                # Blend aligned depth + rgb image
                blended_image = 0.6 * depth_three_channel.astype(float) / 255 + \
                    0.4 * pcl_frames[1].astype(float) / 255
                blended_image = (255 * blended_image.astype(float) /
                                 blended_image.max()).astype(np.uint8)
                cv2.imshow("RGBD Alignment", blended_image)
                if cv2.waitKey(1) == "q":
                    break
            else:
                pcl_converter.rgbd_to_projection(*pcl_frames, downsample=True)
                pcl_converter.visualize_pcd()
