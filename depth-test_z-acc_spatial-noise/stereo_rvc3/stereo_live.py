import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
import utils
import pipeline_creation
PATH_CALIB = "/home/matevz/Desktop/vermeerTestingThinkpad/camera_11/calibrations/calib5.json"
USE_VERTICAL=True


# Create pipeline
pipeline = dai.Pipeline()
calibData = dai.CalibrationHandler(PATH_CALIB)
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


if USE_VERTICAL:
    #################################################  vertical ###############################################################
    stereoVertical, outNames = pipeline_creation.create_stereo(pipeline, "vertical", colorLeft.isp, colorVertical.isp)

    meshLeft, meshVertical = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorVertical.getBoardSocket(),
                                                                (colorLeft.getResolutionWidth(), colorLeft.getResolutionHeight()), vertical=True)
    stereoVertical.loadMeshData(meshLeft, meshVertical)
    stereoVertical.setVerticalStereo(True)
    ############################################################################################################################

else:
    #################################################  horizontal ##############################################################
    colorLeft.setVideoSize(1280, 800)
    colorRight.setVideoSize(1280, 800)
    stereoHorizontal, outNames = pipeline_creation.create_stereo(pipeline, "horizontal", colorLeft.video, colorRight.video)

    meshLeft, meshVertical = pipeline_creation.create_mesh_on_host(calibData, colorLeft.getBoardSocket(), colorRight.getBoardSocket(), 
                                                                (colorLeft.getVideoWidth(), colorLeft.getVideoHeight()))
    stereoHorizontal.loadMeshData(meshLeft, meshVertical)
    ############################################################################################################################

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

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