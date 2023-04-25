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
parser.add_argument('-outDir', type=str, default="out", help="Output directory for depth maps and rectified files.")
parser.add_argument('-rect',action="store_true", default=False, help="Generate and display rectified streams.")
parser.add_argument('-fps', type=int, default=10, help="Set camera FPS.")
parser.add_argument('-useOpenCVDepth', action="store_true",
                    help='Use OpenCV to display frames')
parser.add_argument('-numLastFrames', type=int, default=None, help="Number of frames (last frames are used) for calculating the average depth.")

args = parser.parse_args()

CONNECT_LEFT_RIGHT = False


USE_BLOCKING=True
args = parser.parse_args()

if args.useOpenCVDepth:
    args.rect = True
    stereoOcv = cv2.StereoSGBM_create()
    # Set the parameters for StereoSGBM
    stereoOcv.setBlockSize(9)
    stereoOcv.setMinDisparity(0)
    stereoOcv.setNumDisparities(64)
    stereoOcv.setUniquenessRatio(10)
    stereoOcv.setSpeckleWindowSize(0)
    stereoOcv.setSpeckleRange(0)
    stereoOcv.setDisp12MaxDiff(0)


if args.ver is False and args.hor is False:
    args.ver = True
    args.hor = True

hostInput = False
if args.videoDir is not None or args.imagesDir is not None:
    hostInput = True
    USE_BLOCKING = True

if args.saveFiles:
    Path(args.outDir).mkdir(parents=True, exist_ok=True)
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
    colorLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    colorRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    for cam in [colorLeft, colorRight, colorVertical]:
        cam.setFps(args.fps)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)


    stereoVerLeftIn = colorLeft.isp
    stereoVerRightIn = colorVertical.isp
    stereoHorLeftIn = colorLeft.isp
    stereoHorRightIn = colorRight.isp
else: # Video input
    xLinks = {}
    altNames = {
        "left": "camb,c",
        "right": "camc,c",
        "vertical": "camd,c"
    }
    for xLinkName in ["left", "right", "vertical"]:
        xLink = pipeline.create(dai.node.XLinkIn)
        xLink.setStreamName(xLinkName)
        xLinks[xLinkName] = {}
        xLinks[xLinkName]["node"] = xLink
        if args.videoDir is not None:
            videoPath = str(Path(args.videoDir) / (xLinkName + ".avi"))
            if not Path(videoPath).exists():
                videoPath = str(Path(args.videoDir) /
                                (altNames[xLinkName] + ".avi"))
                if not Path(videoPath).exists():
                    print("ERROR: Video file not found: ", videoPath)
                    exit(1)
            xLinks[xLinkName]["video"] = cv2.VideoCapture(videoPath)
        elif args.imagesDir is not None:
            imagePath = str(Path(args.imagesDir) / (xLinkName + ".png"))
            if not Path(imagePath).exists():
                print("ERROR: Images directory not found: ", imagePath)
                exit(1)
            xLinks[xLinkName]["image"] = cv2.imread(str(Path(imagePath)))
            print("Image path: ", imagePath)
    if args.videoDir is not None:
        numFramesLeft = int(xLinks["left"]["video"].get(cv2.CAP_PROP_FRAME_COUNT))
        numFramesRight = int(xLinks["right"]["video"].get(cv2.CAP_PROP_FRAME_COUNT))
        numFramesVertical = int(xLinks["vertical"]["video"].get(cv2.CAP_PROP_FRAME_COUNT))
        minFrames = min(numFramesLeft, numFramesRight, numFramesVertical)
        if args.numLastFrames is not None:
            print(f"Using last {args.numLastFrames} frames for calculating the average depth.")
            for name in ["left", "right", "vertical"]:
                if args.numLastFrames > minFrames:
                    print(f"numLastFrames {args.numLastFrames} is greater than number of images in the left video {numFramesLeft}.")
                    break
                video = xLinks[name]["video"]
                video.set(cv2.CAP_PROP_POS_FRAMES, minFrames - args.numLastFrames)


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
    stereoVertical, outNamesVertical = pipeline_creation.create_stereo(
        pipeline, "vertical", stereoVerLeftIn, stereoVerRightIn, CONNECT_LEFT_RIGHT, args.rect, args.rect)
    outNames += outNamesVertical
    meshLeft, meshRight, verScale = pipeline_creation.create_mesh_on_host(calibData, stereoVerLeftInSocket, stereoVerRightInSocket,
                                                                (width, height), vertical=True)
    stereoVertical.loadMeshData(meshLeft, meshRight)
    stereoVertical.setVerticalStereo(True)
    ############################################################################################################################

if args.hor:
    #################################################  horizontal ##############################################################
    stereoHorizontal, outNamesHorizontal = pipeline_creation.create_stereo(
        pipeline, "horizontal", stereoHorLeftIn, stereoHorRightIn, CONNECT_LEFT_RIGHT, args.rect, args.rect)
    outNames += outNamesHorizontal
    meshLeft, meshRight, horScale = pipeline_creation.create_mesh_on_host(calibData, stereoHorLeftInSocket, stereoHorRightInSocket,
                                                                         (width, height))
    stereoHorizontal.loadMeshData(meshLeft, meshRight)
    stereoHorizontal.setLeftRightCheck(False)
    ############################################################################################################################

# Connect to device and start pipeline
verticalDepths = []
horizontalDepths = []
leftRectifiedVer = []
leftRectifiedHor = []

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
                        break
                elif args.imagesDir:
                    #  print("Image " + xLinkName + " finished")
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
        frames = {}
        for queueName in queues.keys():
            if USE_BLOCKING:
                inFrame = queues[queueName].get()
            else:
                inFrame = queues[queueName].tryGet()
            if inFrame is None:
                continue
            frames[queueName] = inFrame.getFrame()
            frame = inFrame.getCvFrame()
            if args.saveFiles:
                cv2.imwrite(str(Path(args.outDir) / queueName) + ".png", frame)

            if "disparity" in queueName:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # Colorize the disparity map
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            if "vertical" in queueName and ("rectified" in queueName or "disparity" in queueName):
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow(queueName, frame)
        # Handle saving and depth calculation
        for stereo in ["vertical", "horizontal"]:
            if not args.rect:
                continue
            rectLeft =  frames[stereo + "-rectified_left"]
            rectRight = frames[stereo + "-rectified_right"]
            disparity = frames[stereo + "-disparity"]
            scaleFac = verScale if stereo == "vertical" else horScale
            if args.useOpenCVDepth:
                disparity = stereoOcv.compute(rectLeft, rectRight)
                scaleFac = scaleFac / 2
                disparityShow = cv2.normalize(
                    disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disparityShow = cv2.applyColorMap(
                    disparityShow, cv2.COLORMAP_JET)
                cv2.imshow(stereo + "-disparity-opencv", disparityShow)
            with np.errstate(divide='ignore'):
                depthFrame = scaleFac / disparity
            depthFrame[depthFrame == np.inf] = 0
            if args.saveFiles:
                if stereo == "vertical":
                    verticalDepths.append(depthFrame)
                    leftRectifiedVer.append(rectLeft)
                else:
                    horizontalDepths.append(depthFrame)
                    leftRectifiedHor.append(rectLeft)
        if cv2.waitKey(1) == ord('q'):
            break
        if args.imagesDir:
            if cv2.waitKey(0) == ord('q'):
                break

if args.saveFiles:
    stackedHorizontalDepths = np.stack(horizontalDepths, axis=0)
    stackedVerticalDepths = np.stack(verticalDepths, axis=0)
    if args.rect:
        stackedRectifiedLeftHor = np.stack(leftRectifiedHor, axis=0)
        stackedRectifiedLeftVer = np.stack(leftRectifiedVer, axis=0)

    np.save(Path(args.outDir)/"horizontalDepth.npy", stackedHorizontalDepths)
    np.save(Path(args.outDir)/"verticalDepth.npy", stackedVerticalDepths)
    if args.rect:
        np.save(Path(args.outDir)/"leftRectifiedVertical.npy", stackedRectifiedLeftVer)
        np.save(Path(args.outDir)/"leftRectifiedHorizontal.npy", stackedRectifiedLeftHor)