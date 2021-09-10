#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

try:
    from projector_3d import PointCloudVisualizer
except ImportError as e:
    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}")


# StereoDepth config options. TODO move to command line options
out_depth = False  # Disparity by default
out_rectified = True  # Output and display rectified streams
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
last_rectified_right = None

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF


def create_stereo_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    rectified_left = pipeline.createXLinkOut()
    rectified_right = pipeline.createXLinkOut()
    disparityOut = pipeline.createXLinkOut()

    # Properties
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(230)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setMedianFilter(median)  # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    rectified_left.setStreamName("rectified_left")
    rectified_right.setStreamName("rectified_right")
    disparityOut.setStreamName("disparity")

    # Linking
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.rectifiedLeft.link(rectified_left.input)
    stereo.rectifiedRight.link(rectified_right.input)
    stereo.disparity.link(disparityOut.input)

    streams = ["rectified_left", "rectified_right", "disparity"]
    maxDisparity = stereo.getMaxDisparity()

    return pipeline, streams, maxDisparity


def create_rgb_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    rectified_left = pipeline.createXLinkOut()
    rectified_right = pipeline.createXLinkOut()
    disparityOut = pipeline.createXLinkOut()
    rgbOut = pipeline.createXLinkOut()

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(2, 3)  # NOTE: This downscales to 1280x720 on the device?
    camRgb.setInterleaved(False)

    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(230)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setMedianFilter(median)  # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    rgbOut.setStreamName("rgb")
    rectified_left.setStreamName("rectified_left")
    rectified_right.setStreamName("rectified_right")
    disparityOut.setStreamName("disparity")

    # Linking
    camRgb.video.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.rectifiedLeft.link(rectified_left.input)
    stereo.rectifiedRight.link(rectified_right.input)
    stereo.disparity.link(disparityOut.input)

    streams = ["rgb", "rectified_left", "rectified_right", "disparity"]
    maxDisparity = stereo.getMaxDisparity()

    return pipeline, streams, maxDisparity


def convert_to_cv2_frame(name, image):
    # Workaround: needs to be global, because we dont get rectified_right and disparity simultaneously
    global last_rectified_right
    baseline = 75  # mm
    focal = right_intrinsic[0][0]
    disp_type = np.uint8 if not subpixel else np.uint16  # 5 bits fractional disparity
    disp_levels = 1 if not subpixel else 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()

    if name == "rgb":
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == "disparity":
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide="ignore"):  # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        frame = (disp * 255.0 / MAX_DISPARIRTY).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        frame = np.ascontiguousarray(frame)

        if pcl_converter is not None and last_rectified_right is not None:
            pcl_converter.rgbd_to_projection(depth, last_rectified_right, False)
            pcl_converter.visualize_pcd()
    else:  # rectified streams
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        # frame = cv2.flip(frame, 1)
        if name == "rectified_right":
            last_rectified_right = frame
    return frame




# Select pipeline
pipeline, streams, MAX_DISPARIRTY = create_stereo_pipeline()  # Works
# pipeline, streams, MAX_DISPARIRTY = create_rgb_pipeline()  # Unsupported image format error, TODO: Implement RGB blending


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.startPipeline()
    calibData = device.readCalibration()

    # NOTE: Should be equivalent?
    # right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
    right_intrinsic = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
    
    pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
    queue_list = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]

    while True:
        for i, queue in enumerate(queue_list):
            name = queue.getName()
            image = queue.get()
            frame = convert_to_cv2_frame(name, image)
            cv2.imshow(name, frame)

        if cv2.waitKey(1) == ord("q"):
            break
