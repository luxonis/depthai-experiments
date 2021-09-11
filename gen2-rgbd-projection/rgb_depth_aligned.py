#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

def create_pipeline():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.createColorCamera()
    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    rgbOut = pipeline.createXLinkOut()
    disparityOut = pipeline.createXLinkOut()

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(2, 3)  # NOTE: This downscales to 1280x720?
    camRgb.setInterleaved(False)

    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.initialConfig.setConfidenceThreshold(230)
    stereo.setLeftRightCheck(True)  # LR-check is required for depth alignment
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    rgbOut.setStreamName("rgb")
    disparityOut.setStreamName("disparity")

    # Linking
    camRgb.video.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.disparity.link(disparityOut.input)

    streams = ["rgb", "disparity"]
    maxDisparity = stereo.getMaxDisparity()

    return pipeline, streams, maxDisparity


def visualizeDisparityFrame(disparityFrame):
    disparityFrame = (disparityFrame * 255.0 / MAX_DISPARIRTY).astype(np.uint8)
    disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_HOT)
    return np.ascontiguousarray(disparityFrame)


def convert_to_cv2_frame(name, image):
    data, w, h = image.getData(), image.getWidth(), image.getHeight()

    if name == "rgb":
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == "disparity":
        depth = np.array(data).reshape((h, w))
        frame = visualizeDisparityFrame(depth)

    return frame


def blend_rgb_disparity(frameRgb, frameDisparity):
    # Need to have both frames in BGR format before blending
    if len(frameDisparity.shape) < 3:
        frameDisparity = cv2.cvtColor(frameDisparity, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frameRgb, 0.6, frameDisparity, 0.4, 0)


pipeline, streams, MAX_DISPARIRTY = create_pipeline()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    calibData = device.readCalibration()

    queue_list = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]
    blendFrames = [None, None]

    while True:
        for i, queue in enumerate(queue_list):
            name = queue.getName()
            image = queue.get()
            frame = convert_to_cv2_frame(name, image)
            cv2.imshow(name, frame)
            if i < 2:
                blendFrames[i] = frame

        blended = blend_rgb_disparity(*blendFrames)
        cv2.imshow("rgb-disparity", blended)

        if cv2.waitKey(1) == ord("q"):
            break
