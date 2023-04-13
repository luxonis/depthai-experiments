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

parser.add_argument('-videoDir', type=str, default=None, help='Path to video directory')

CONNECT_LEFT_RIGHT = False
CONNECT_RECTIFIED_LEFT_RIGHT = False

args = parser.parse_args()

if args.ver is False and args.hor is False:
    args.ver = True
    args.hor = True

videoInput = False
if args.videoDir is not None:
    videoInput = True

# Create pipeline
device = dai.Device()
pipeline = dai.Pipeline()
if not args.calib:
    calibData = device.readCalibration()
else:
    calibData = dai.CalibrationHandler(args.calib)
    pipeline.setCalibrationData(calibData)
width = 1920
height = 1200
if args.videoDir is None:
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



    stereoVerLeftIn = colorLeft.isp
    stereoVerRightIn = colorVertical.isp
    stereoHorLeftIn = colorLeft.isp
    stereoHorRightIn = colorRight.isp
else: # Video input
    xLinks = {}
    for xLinkName in ["left", "right", "vertical"]:
        xLink = pipeline.create(dai.node.XLinkIn)
        xLink.setStreamName(xLinkName)
        xLinks[xLinkName] = {}
        xLinks[xLinkName]["node"] = xLink
        videoPath = str(Path(args.videoDir) / (xLinkName + ".avi"))
        if not Path(videoPath).exists():
            print("ERROR: Video file not found: ", videoPath)
            exit()
        xLinks[xLinkName]["video"] = cv2.VideoCapture(videoPath)

    stereoVerLeftIn = xLinks["left"]["node"].out
    stereoVerRightIn = xLinks["vertical"]["node"].out
    stereoHorLeftIn = xLinks["left"]["node"].out
    stereoHorRightIn = xLinks["right"]["node"].out

stereoVerLeftInSocket = dai.CameraBoardSocket.LEFT
stereoVerRightInSocket = dai.CameraBoardSocket.CAM_D
stereoHorLeftInSocket = dai.CameraBoardSocket.LEFT
stereoHorRightInSocket = dai.CameraBoardSocket.RIGHT


outNames = []
if args.ver:
    #################################################  vertical ###############################################################
    stereoVertical, outNamesVertical = pipeline_creation.create_stereo(pipeline, "vertical", stereoVerLeftIn, stereoVerRightIn, CONNECT_LEFT_RIGHT, CONNECT_RECTIFIED_LEFT_RIGHT, CONNECT_RECTIFIED_LEFT_RIGHT)
    outNames += outNamesVertical
    meshLeft, meshVertical, scale = pipeline_creation.create_mesh_on_host(calibData, stereoVerLeftInSocket, stereoVerRightInSocket,
                                                                (width, height), vertical=True)
    stereoVertical.loadMeshData(meshLeft, meshVertical)
    stereoVertical.setVerticalStereo(True)
    ############################################################################################################################

if args.hor:
    #################################################  horizontal ##############################################################
    stereoHorizontal, outNamesHorizontal = pipeline_creation.create_stereo(pipeline, "horizontal", stereoHorLeftIn, stereoHorRightIn, CONNECT_LEFT_RIGHT, CONNECT_RECTIFIED_LEFT_RIGHT, CONNECT_RECTIFIED_LEFT_RIGHT)
    outNames += outNamesHorizontal
    meshLeft, meshVertical, scale = pipeline_creation.create_mesh_on_host(calibData, stereoHorLeftInSocket, stereoHorRightInSocket,
                                                                         (width, height))
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
    if args.videoDir is not None:
        for xLinkName in xLinks.keys():
            xLinks[xLinkName]["queue"] = device.getInputQueue(xLinkName)
    while True:
        if args.videoDir is not None:
            ts = dai.Clock.now()
            endOfVideo = False
            for xLinkName in xLinks.keys():
                xLink = xLinks[xLinkName]
                if not xLink["video"].isOpened():
                    endOfVideo = True
                    break
                read_correctly, frame = xLink["video"].read()
                if not read_correctly:
                    print("Video " + xLinkName + " finished")
                    xLink["video"].release()
                    endOfVideo = True
                    break
                frame = cv2.resize(frame, (width, height))
                # Convert to NV12
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = dai.ImgFrame()
                img.setData(frame.reshape(height*width))
                img.setTimestamp(ts)
                img.setWidth(width)
                img.setHeight(height)
                img.setType(dai.ImgFrame.Type.RAW8)
                xLink["queue"].send(img)
            if endOfVideo:
                break
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