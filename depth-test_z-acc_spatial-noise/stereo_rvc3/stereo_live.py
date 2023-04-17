import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
import utils
import pipeline_creation
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-ver', action="store_true", help='Use vertical stereo')
parser.add_argument('-hor', action="store_true", help='Use horizontal stereo')
parser.add_argument('-calib', type=str, default=None, help='Path to calibration file')
parser.add_argument('-saveFiles', action="store_true", help='Save files')
parser.add_argument('-videoDir', type=str, default=None, help='Path to video directory')
parser.add_argument('-imagesDir', type=str, default=None, help='Path to images directory')

CONNECT_LEFT_RIGHT = False
CONNECT_RECTIFIED_LEFT_RIGHT = True


USE_BLOCKING=False
args = parser.parse_args()

if args.ver is False and args.hor is False:
    args.ver = True
    args.hor = True

hostInput = False
if args.videoDir is not None or args.imagesDir is not None:
    hostInput = True
    USE_BLOCKING = True
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
if not hostInput:
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
        if args.videoDir is not None:
            videoPath = str(Path(args.videoDir) / (xLinkName + ".avi"))
            if not Path(videoPath).exists():
                print("ERROR: Video file not found: ", videoPath)
                exit()
            xLinks[xLinkName]["video"] = cv2.VideoCapture(videoPath)
        elif args.imagesDir is not None:
            imagePath = str(Path(args.imagesDir) / (xLinkName + ".png"))
            if not Path(imagePath).exists():
                print("ERROR: Images directory not found: ", imagePath)
                exit()
            xLinks[xLinkName]["image"] = cv2.imread(str(Path(imagePath)))
            print("Image path: ", imagePath)


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
    if hostInput:
        for xLinkName in xLinks.keys():
            xLinks[xLinkName]["queue"] = device.getInputQueue(xLinkName)
    while True:
        if hostInput:
            ts = dai.Clock.now()
            endOfVideo = False
            for xLinkName in xLinks.keys():
                xLink = xLinks[xLinkName]
                if args.videoDir:
                    if not xLink["video"].isOpened():
                        endOfVideo = True
                        break
                    read_correctly, frame = xLink["video"].read()
                    if not read_correctly:
                        print("Video " + xLinkName + " finished")
                        xLink["video"].release()
                        endOfVideo = True
                elif args.imagesDir:
                    print("Image " + xLinkName + " finished")
                    frame = xLink["image"]
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
            if USE_BLOCKING:
                inFrame = queues[queueName].get()
            else:
                inFrame = queues[queueName].tryGet()
            if inFrame is None:
                continue
            frame = inFrame.getCvFrame()
            if args.saveFiles:
                cv2.imwrite(queueName + ".png", frame)
            if "disparity" in queueName:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # Colorize the disparity map
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            if "vertical" in queueName and ("rectified" in queueName or "disparity" in queueName):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow(queueName, frame)
        if cv2.waitKey(1) == ord('q'):
            break
        if args.imagesDir:
            if cv2.waitKey(0) == ord('q'):
                break