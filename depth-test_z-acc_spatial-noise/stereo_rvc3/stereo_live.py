import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
import utils
import pipeline_creation


parser = argparse.ArgumentParser()
parser.add_argument('-ver', action="store_true", help='Use vertical stereo')
parser.add_argument('-hor', action="store_true", help='Use horizontal stereo')
parser.add_argument('-calib', type=str, default=None, help='Path to calibration file')

args = parser.parse_args()

if args.ver is False and args.hor is False:
    args.ver = True
    args.hor = True

# Create pipeline
device = dai.Device()
pipeline = dai.Pipeline()
if not args.calib:
    calibData = device.readCalibration()
else:
    calibData = dai.CalibrationHandler(args.calib)
    pipeline.setCalibrationData(calibData)

# Cameras
# Define sources and outputs
colorLeft = pipeline.create(dai.node.ColorCamera)
colorRight = pipeline.create(dai.node.ColorCamera)
colorVertical = pipeline.create(dai.node.ColorCamera)
colorVertical.setBoardSocket(dai.CameraBoardSocket.CAM_D)
colorVertical.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
colorLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
colorRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
colorRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)


outNames = []
if args.ver:
    #################################################  vertical ###############################################################
    stereoVertical, outNamesVertical = pipeline_creation.create_stereo(pipeline, "vertical", colorLeft.isp, colorVertical.isp, True, True, True)
    outNames += outNamesVertical
    meshLeft, meshVertical, scale = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorVertical.getBoardSocket(),
                                                                (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight()), vertical=True)
    stereoVertical.loadMeshData(meshLeft, meshVertical)
    stereoVertical.setVerticalStereo(True)
    ############################################################################################################################

if args.hor:
    #################################################  horizontal ##############################################################
    stereoHorizontal, outNamesHorizontal = pipeline_creation.create_stereo(pipeline, "horizontal", colorLeft.isp, colorRight.isp, True, True, True)
    outNames += outNamesHorizontal
    meshLeft, meshVertical, scale = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorRight.getBoardSocket(),
                                                                (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight()))
    stereoHorizontal.loadMeshData(meshLeft, meshVertical)
    stereoHorizontal.setLeftRightCheck(False)
    ############################################################################################################################

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)
    queues = {}
    print(outNames)
    for queueName in outNames:
        queues[queueName] = device.getOutputQueue(queueName, 4, False)
    while True:
        for queueName in queues.keys():
            inFrame = queues[queueName].tryGet()
            if inFrame is None:
                continue
            frame = inFrame.getCvFrame()
            if "disparity" in queueName:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # Colorize the disparity map
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            if "vertical" in queueName and ("rectified" in queueName or "disparity" in queueName):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow(queueName, frame)
        if cv2.waitKey(1) == ord('q'):
            break