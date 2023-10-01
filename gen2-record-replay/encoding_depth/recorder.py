#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
import argparse
import pathlib
import json

parser = argparse.ArgumentParser()
parser.add_argument("-nf", "--num_frames", default=120, type=int, help="Number of frames to record")
parser.add_argument("-e", "--extended", action="store_true", help="Enable extended disparity")
parser.add_argument("-sub", "--subpixel", action="store_true", help="Enable subpixel disparity")
parser.add_argument("-lr", "--lr_check", action="store_true", help="Enable left-right check")
parser.add_argument("-path", "--path", default="./data", help="Path where to store the frames")
parser.add_argument("--subpixel_bits", default=3, type=int, help="Subpixel disparity bits")
args = parser.parse_args()

SKIP_FIRST_N_FRAMES = 15

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = args.extended
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = args.subpixel
# Better handling for occlusions:
lr_check = args.lr_check
num_frames_to_record = args.num_frames
subpixel_bits = args.subpixel_bits
num_subpixels = 2**subpixel_bits if subpixel else 1

# Create the directory with the dataset path
path = pathlib.Path(args.path)
if path.exists():
    print(f"Path {path} already exists - data will not be saved, please specify a different path to save the frames to.")
    exit(1)
else:
    path.mkdir(parents=True, exist_ok=False)
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
xoutDisp = pipeline.create(dai.node.XLinkOut)
xout = pipeline.create(dai.node.XLinkOut)


xout.setStreamName("depth")
xoutDisp.setStreamName("disparity")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setCamera("right")

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setSubpixelFractionalBits(subpixel_bits)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.depth.link(xout.input)
depth.disparity.link(xoutDisp.input)
# Create an array that will store the frames in WxHxNUM_FRAMES format
frames = np.zeros((num_frames_to_record, 800, 1280), dtype=np.uint16)


# Connect to device and start pipeline
with dai.Device() as device:
    device.setIrLaserDotProjectorBrightness(765)
    calibration = device.readCalibration()
    baselineDistance = calibration.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)[0][3]
    baselineDistance = abs(baselineDistance) * 10
    focalLength = calibration.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 800)[0][0]
    dispToDepthFactor = baselineDistance * focalLength

    device.startPipeline(pipeline)
    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qDisp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    for i in range(SKIP_FIRST_N_FRAMES):
        inDepth : dai.ImgFrame = q.get()
    for i in range(num_frames_to_record):
        inDepth : dai.ImgFrame = q.get()  # blocking call, will wait until a new data has arrived
        frame = inDepth.getFrame()
        frames[i, :, :] = frame

        frame = qDisp.get().getFrame()
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)
        cv2.imshow("disparity", frame)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break
np.save(str(path / "depth_frames.npy"), frames)
# Create the json file with relevant information (disp - to - depth, max disparity, baseline, focal length)
json_data = {
    "num_frames": num_frames_to_record,
    "disp_to_depth_factor": dispToDepthFactor,
    "max_disparity": depth.initialConfig.getMaxDisparity(),
    "num_subpixels": num_subpixels,
    "baseline": baselineDistance,
    "focal_length": focalLength
}

# Save the json file
with open(str(path / "depth_info.json"), "w") as f:
    json.dump(json_data, f, indent=4)

