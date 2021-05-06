#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import cmapy

parser = argparse.ArgumentParser()
parser.add_argument('-lrc', '--lrcheck',  default=False, action="store_true")
parser.add_argument('-r', '--rotate',  default=False, action="store_true")
parser.add_argument('-ext', '--extended', default=False, action="store_true")
parser.add_argument('-sub', '--subpixel', default=False, action="store_true")
parser.add_argument('-dct', '--disp-confidence-thr', default=230, type=int)
parser.add_argument('-fd',  '--force-depth-with-subpixel', default=False, action="store_true")
parser.add_argument('-tun', '--camera-tuning', type=str)
args = parser.parse_args()

print(dai.__version__)

# -------------------------
# Change this if needed:
# -------------------------
FPS = 30
# Save every 30th frame:
SAVE_EVERY = 30 # AKA 30FPS / 30 SAVE_EVERY => 1 log per second


stepSize = 0.05

if 1:  # OpenCV predefined maps
    colorMap = cv2.COLORMAP_JET  # COLORMAP_HOT, COLORMAP_TURBO, ...
else:  # matplotlib maps, see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colorMap = cmapy.cmap('inferno')

showDisparity = True
averageDepthCalc = True
if args.subpixel:
    if args.force_depth_with_subpixel:
        print("Subpixel enabled and `force_depth_with_subpixel` set")
        showDisparity = False
    else:
        print("Subpixel enabled and average depth calculator disabled.")
        print("To enable it (and view inverted depth instead of disparity), set `-fd`")
        averageDepthCalc = False

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createColorCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setFps(FPS)

monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
monoRight.setBoardSocket(dai.CameraBoardSocket.RGB)
monoRight.setFps(FPS)
monoRight.setIspScale(1, 3)
monoRight.initialControl.setManualFocus(135)
if args.camera_tuning is not None:
    monoRight.setCameraTuningBlobPath(args.camera_tuning)

if args.rotate:
    monoLeft.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    monoRight.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)


# StereoDepth
stereo.setConfidenceThreshold(args.disp_confidence_thr)
stereo.setRectifyEdgeFillColor(0)
stereo.setLeftRightCheck(args.lrcheck)
stereo.setExtendedDisparity(args.extended)
stereo.setSubpixel(args.subpixel)

monoLeft.out.link(stereo.left)
monoRight.isp.link(stereo.right)

if showDisparity:
    stereo.disparity.link(xoutDepth.input)
else:
    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)

if averageDepthCalc:
    stereo.depth.link(spatialLocationCalculator.inputDepth)


topLeft = dai.Point2f(0.5, 0.5)
bottomRight = dai.Point2f(0.6, 0.6)

spatialLocationCalculator.setWaitForConfigInput(False)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

sys_logger = pipeline.createSystemLogger()
sys_logger.setRate(FPS)  # 1 Hz

# Create output
linkOut = pipeline.createXLinkOut()
linkOut.setStreamName("sysinfo")
sys_logger.out.link(linkOut.input)

with dai.Device(pipeline) as device, open("log.txt", "a") as file:
    device.startPipeline()

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    q_sysinfo = device.getOutputQueue(name="sysinfo", maxSize=4, blocking=False)

    color = (255, 255, 255)

    print("Use WASD keys to move ROI!")
    counter = 0
    while True:
        out = ""
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.tryGet() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()
        if showDisparity:
            maxDisparity = 95
            if args.extended: maxDisparity *= 2
            if args.subpixel: maxDisparity *= 32
            depthFrame = (depthFrame * 255. / maxDisparity).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrame, colorMap)
        else:
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        if inDepthAvg is not None:
            spatialData = inDepthAvg.getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)
    
                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
                out += "------------------------------------------------------------------------------------\n"
                out += f"Timestamp: {str(inDepth.getTimestamp())}, Average depth: {depthData.spatialCoordinates.z} mm\n"

        info = q_sysinfo.tryGet()
        if info is not None:
            t = info.chipTemperature
            out += f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa0: {t.upa:.2f}, upa1: {t.dss:.2f}\n"

        # Log into file
        if counter > SAVE_EVERY:
            file.write(out)
            print(out)
            counter = 0
        else: counter+=1

        cv2.imshow("depth", depthFrameColor)

        newConfig = False
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)