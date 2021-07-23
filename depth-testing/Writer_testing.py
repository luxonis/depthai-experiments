import cv2
import argparse
import numpy as np
import depthai as dai
import os

# color codings
white  = [255, 255, 255]
orange = [143, 122, 4]
red    = [230, 9, 9]
green  = [4, 143, 7]
black  = [0, 0, 0]


test_type = 'oak_1'
font = cv2.FONT_HERSHEY_SIMPLEX

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def create_pipeline(enableLR, enableRgb):
    pipeline = dai.Pipeline()

    if enableLR:
        cam_left = pipeline.createMonoCamera()
        cam_right = pipeline.createMonoCamera()

        xout_left = pipeline.createXLinkOut()
        xout_right = pipeline.createXLinkOut()

        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                
        cam_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)

        cam_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P)
        cam_left.setFps(10)
        cam_right.setFps(10)

        xout_left.setStreamName("left")
        cam_left.out.link(xout_left.input)

        xout_right.setStreamName("right")
        cam_right.out.link(xout_right.input)

    if enableRgb:
        rgb_cam = pipeline.createColorCamera()
        rgb_cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K)
        rgb_cam.setInterleaved(False)
        rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb_cam.setFps(10)
        rgb_cam.setIspScale(1, 3)

        xout_rgb_isp = pipeline.createXLinkOut()
        xout_rgb_isp.setStreamName("rgb")
        rgb_cam.isp.link(xout_rgb_isp.input)        

    return pipeline


enabledLR = False
enabledRGB = False

focusThreshold = 20
if 'oak_1' in test_type:
    enabledRGB = True
else:
    enabledLR = True
    enabledRGB = True

# TODO(sachin): Add detection and acceptance of board in LR before continuing  focal check 

device = dai.Device() 
pipeline = create_pipeline(enabledLR, enabledRGB)
device.startPipeline(pipeline)

if enabledLR:
    left_camera_queue = device.getOutputQueue("left", 5, False)
    right_camera_queue = device.getOutputQueue("right", 5, False)
if enabledRGB:
    rgb_camera_queue = device.getOutputQueue("rgb", 5, False)

max_count = 90
rgb_count   = 0
left_count  = 0
right_count = 0

lens_position = 0

is_left_focused = False
is_right_focused = False
is_rgb_focused = False


calibration_handler = dai.CalibrationHandler()
calibration_handler.setLensPosition(dai.CameraBoardSocket.RGB, 135)
device.flashCalibration(calibration_handler)
print("Calibration Flashed")
