#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
import yaml
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument('-si', action="store_true", help="Static input.")
parser.add_argument('-vid', action="store_true", help="Use video files")
parser.add_argument('-calib', type=str, default=None, help="Path to calibration file in json")
parser.add_argument('-left', type=str, default="left.png", help="left static input image")
parser.add_argument('-right', type=str, default="right.png", help="right static input image")
parser.add_argument('-bottom', type=str, default="bottom.png", help="bottom static input image")
parser.add_argument('-debug', action="store_true", default=False, help="Debug code.")
parser.add_argument('-rect', '--rectified', action="store_true", default=False, help="Generate and display rectified streams.")
parser.add_argument('-fps', type=int, default=15, help="Set camera FPS.")
parser.add_argument('-outVer', type=str, default="outDepthVerticalNumpy.npy", help="Output vertical depth numpy file.")
parser.add_argument('-outHor', type=str, default="outDepthHorizontalNumpy.npy", help="Output horizontal depth numpy file.")
parser.add_argument('-outLeftRectVer', type=str, default="outLeftRectVerticalNumpy.npy")
parser.add_argument('-outLeftRectHor', type=str, default="outLeftRectHorizontalNumpy.npy")
parser.add_argument('-saveFiles', action="store_true", default=False, help="Save output files.")


args = parser.parse_args()

staticInput = args.si

enableRectified = args.rectified
cameraFPS = args.fps
blockingOutputs = False

if staticInput and args.vid:
    print("Static input and video input cannot be used at the same time.")
    exit()

if staticInput:
    left = args.left
    right = args.right
    bottom = args.bottom

    leftImg = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
    rightImg = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
    bottomImg = cv2.imread(bottom, cv2.IMREAD_GRAYSCALE)

    width = leftImg.shape[1]
    height = leftImg.shape[0]

    if args.debug:
        cv2.imshow("leftImg", leftImg)
        cv2.imshow("rightImg", rightImg)
        cv2.imshow("bottomImg", bottomImg)
        cv2.waitKey(1)

elif args.vid:
    leftVideo = cv2.VideoCapture(args.left)
    rightVideo = cv2.VideoCapture(args.right)
    bottomVideo = cv2.VideoCapture(args.bottom)

forceFisheye = False

def rotationMatrixToEulerAngles(R) :
  
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
 
    R = (R_z @ ( R_y @ R_x ))
 
    return R

def stereoRectify(R, T):

    om = rotationMatrixToEulerAngles(R)
    om = om * -0.5
    r_r = eulerAnglesToRotationMatrix(om)
    t = r_r @ T

    idx = 0 if abs(t[0]) > abs(t[1]) else 1

    c = t[idx]
    nt = np.linalg.norm(t)
    uu = np.zeros(3)
    uu[idx] = 1 if c > 0 else -1
    
    ww = np.cross(t, uu)
    nw = np.linalg.norm(ww)
    
    if nw > 0:
        scale = math.acos(abs(c)/nt)/nw
        ww = ww * scale
        
    wR = eulerAnglesToRotationMatrix(ww)
    R1 = wR @ np.transpose(r_r)
    R2 = wR @ r_r

    return R1, R2

meshCellSize = 16

def downSampleMesh(mapXL, mapYL, mapXR, mapYR):
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight

def rotate_mesh_90_ccw(map_x, map_y):
    direction = 1
    map_x_rot = np.rot90(map_x, direction)
    map_y_rot = np.rot90(map_y, direction)
    return map_x_rot, map_y_rot

device = dai.Device()
# Create pipeline
pipeline = dai.Pipeline()

if args.calib is None:
    calibData = device.readCalibration()
else:
    calibData = dai.CalibrationHandler(args.calib)
    pipeline.setCalibrationData(calibData)

if staticInput or args.vid:
    monoLeft = pipeline.create(dai.node.XLinkIn)
    monoRight = pipeline.create(dai.node.XLinkIn)
    monoVertical = pipeline.create(dai.node.XLinkIn)
    monoLeft.setStreamName("inLeft")
    monoRight.setStreamName("inRight")
    monoVertical.setStreamName("inVertical")
else:
    monoLeft = pipeline.create(dai.node.ColorCamera)
    monoVertical = pipeline.create(dai.node.ColorCamera)
    monoRight = pipeline.create(dai.node.ColorCamera)
    monoLeft.setFps(cameraFPS)
    monoVertical.setFps(cameraFPS)
    monoRight.setFps(cameraFPS)

if enableRectified:
    xoutRectifiedVertical = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedRight = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedLeft = pipeline.create(dai.node.XLinkOut)
xoutDisparityVertical = pipeline.create(dai.node.XLinkOut)
xoutDisparityHorizontal = pipeline.create(dai.node.XLinkOut)
stereoVertical = pipeline.create(dai.node.StereoDepth)
stereoHorizontal = pipeline.create(dai.node.StereoDepth)
# syncNode = pipeline.create(dai.node.Sync)

if enableRectified:
    xoutRectifiedVertical.setStreamName("rectified_vertical")
    xoutRectifiedRight.setStreamName("rectified_right")
    xoutRectifiedLeft.setStreamName("rectified_left")
xoutDisparityVertical.setStreamName("disparity_vertical")
xoutDisparityHorizontal.setStreamName("disparity_horizontal")

# Define sources and outputs
if not staticInput and not args.vid:
    monoVertical.setBoardSocket(dai.CameraBoardSocket.VERTICAL)
    monoVertical.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)

    monoVertical.setVideoSize(1280, 800)
    monoLeft.setVideoSize(1280, 800)
    monoRight.setVideoSize(1280, 800)

# Linking
# monoRight.video.link(syncNode.input1)
# monoLeft.video.link(syncNode.input2)
# monoVertical.video.link(syncNode.input3)

# syncNode.output3.link(stereoVertical.left) # left input is bottom camera
# syncNode.output1.link(stereoVertical.right) # right input is right camera
if not staticInput and not args.vid:
    monoLeft.video.link(stereoVertical.left) # left input is bottom camera
    monoVertical.video.link(stereoVertical.right) # right input is right camera
else:
    monoLeft.out.link(stereoVertical.left) # left input is bottom camera
    monoVertical.out.link(stereoVertical.right) # right input is right camera

stereoVertical.disparity.link(xoutDisparityVertical.input)
if enableRectified:
    stereoVertical.rectifiedLeft.link(xoutRectifiedVertical.input)
    stereoVertical.rectifiedRight.link(xoutRectifiedRight.input)
stereoVertical.setVerticalStereo(True)

# syncNode.output2.link(stereoHorizontal.left)
# syncNode.output1.link(stereoHorizontal.right)
if not staticInput and not args.vid:
    monoLeft.video.link(stereoHorizontal.left)
    monoRight.video.link(stereoHorizontal.right)
else:
    monoLeft.out.link(stereoHorizontal.left)
    monoRight.out.link(stereoHorizontal.right)

stereoHorizontal.disparity.link(xoutDisparityHorizontal.input)
if enableRectified:
    stereoHorizontal.rectifiedLeft.link(xoutRectifiedLeft.input)
# stereoHorizontal.rectifiedRight.link(xoutRectifiedRight.input)
stereoHorizontal.setVerticalStereo(False)

stereoHorizontal.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
stereoVertical.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

if 1:
    width = 1280
    height = 800
    resolution = (width, height)

    # vertical

    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, width, height))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.VERTICAL, width, height))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.VERTICAL))

    T = np.array(calibData.getCameraTranslationVector(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.VERTICAL, False))
    R = np.array(calibData.getCameraRotationMatrix(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.VERTICAL))
    R1, R2 = stereoRectify(R, T)

    T2 = np.array(calibData.getCameraTranslationVector(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.VERTICAL, True))

    baselineVer = abs(T2[1])*10
    focalVer = M1[1][1]

    
    print(f"baseline {baselineVer}, focal {focalVer}")


    if not np.any(d1[4:]) and not np.any(d2[4:]):
        print("Fisheye model detected!")
        d1 = d1[:4]
        d2 = d2[:4]

    if forceFisheye or (len(d1) == 4 and len(d2) == 4):

        def calc_fov_D_H_V(f, w, h):
            return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))
        
        M1_focal = M1
        M1_focal = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M1, d1, resolution, R1)
        print(calc_fov_D_H_V(M1_focal[0][0], width, height))
        focalVer = M1_focal[1][1]

        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(M1, d1, R1, M1_focal, resolution, cv2.CV_32FC1)
        mapXV, mapYV = cv2.fisheye.initUndistortRectifyMap(M2, d2, R2, M1_focal, resolution, cv2.CV_32FC1)
    else:
        mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M1, resolution, cv2.CV_32FC1)
        mapXV, mapYV = cv2.initUndistortRectifyMap(M2, d2, R2, M1, resolution, cv2.CV_32FC1)

    verScaleFactor = baselineVer * focalVer * 32


    mapXL_rot, mapYL_rot = rotate_mesh_90_ccw(mapXL, mapYL)
    mapXV_rot, mapYV_rot = rotate_mesh_90_ccw(mapXV, mapYV)

    #clip for now due to HW limit
    mapXV_rot = mapXV_rot[:1024,:]
    mapYV_rot = mapYV_rot[:1024,:]
    mapXL_rot = mapXL_rot[:1024,:]
    mapYL_rot = mapYL_rot[:1024,:]

    leftMeshRot, verticalMeshRot = downSampleMesh(mapXL_rot, mapYL_rot, mapXV_rot, mapYV_rot)

    meshLeft = list(leftMeshRot.tobytes())
    meshVertical = list(verticalMeshRot.tobytes())
    stereoVertical.loadMeshData(meshLeft, meshVertical)

    # horizontal

    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, width, height))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, width, height))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))

    T = np.array(calibData.getCameraTranslationVector(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT, False))
    R = np.array(calibData.getCameraRotationMatrix(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    R1, R2 = stereoRectify(R, T)

    T2 = np.array(calibData.getCameraTranslationVector(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT, True))

    baselineHor = abs(T2[0])*10
    focalHor = M1[0][0]


    print(f"baseline {baselineHor}, focal {focalHor}")

    if not np.any(d1[4:]) and not np.any(d2[4:]):
        print("Fisheye model detected!")
        d1 = d1[:4]
        d2 = d2[:4]

    if forceFisheye or (len(d1) == 4 and len(d2) == 4): 
        def calc_fov_D_H_V(f, w, h):
            return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))
        
        print(M1)
        M1_focal = M1
        M1_focal = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M1, d1, resolution, R1)
        print(calc_fov_D_H_V(M1_focal[0][0], width, height))
        print(M1_focal)
        focalHor = M1_focal[0][0]


        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(M1, d1, R1, M1_focal, resolution, cv2.CV_32FC1)
        mapXR, mapYR = cv2.fisheye.initUndistortRectifyMap(M2, d2, R2, M1_focal, resolution, cv2.CV_32FC1)
    else:
        mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M1, resolution, cv2.CV_32FC1)
        mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M1, resolution, cv2.CV_32FC1)

    leftMesh, rightMesh = downSampleMesh(mapXL, mapYL, mapXR, mapYR)

    meshLeft = list(leftMesh.tobytes())
    meshRight = list(rightMesh.tobytes())
    stereoHorizontal.loadMeshData(meshLeft, meshRight)

    horScaleFactor = baselineHor * focalHor * 32


    if args.debug and staticInput:
        img_l_rectified = cv2.remap(leftImg, mapXL, mapYL, cv2.INTER_LINEAR)
        cv2.imshow("img_l_rectified", img_l_rectified)

        img_r_rectified = cv2.remap(rightImg, mapXR, mapYR, cv2.INTER_LINEAR)
        cv2.imshow("img_r_rectified", img_r_rectified)

        img_r_vertical = cv2.remap(rightImg, mapXR_rot, mapYR_rot, cv2.INTER_LINEAR)
        cv2.imshow("img_r_vertical", img_r_vertical)

        img_b_vertical = cv2.remap(bottomImg, mapXV_rot, mapYV_rot, cv2.INTER_LINEAR)
        cv2.imshow("img_b_vertical", img_b_vertical)


# Connect to device and start pipeline
verticalDepths = []
horizontalDepths = []
leftRectifiedVer = []
leftRectifiedHor = []

with device:
    device.startPipeline(pipeline)
    qDisparityHorizontal = device.getOutputQueue("disparity_horizontal", 4, blockingOutputs)
    qDisparityVertical = device.getOutputQueue("disparity_vertical", 4, blockingOutputs)
    if enableRectified:
        qRectifiedVertical = device.getOutputQueue("rectified_vertical", 4, blockingOutputs)
        qRectifiedRight = device.getOutputQueue("rectified_right", 4, blockingOutputs)
        qRectifiedLeft = device.getOutputQueue("rectified_left", 4, blockingOutputs)

    if staticInput or args.vid:
        qInLeft = device.getInputQueue("inLeft")
        qInRight = device.getInputQueue("inRight")
        qInVertical = device.getInputQueue("inVertical")

    while True:
        if staticInput:
            ts = dai.Clock.now()

            data = cv2.resize(leftImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(1)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            # print("left send")
            qInLeft.send(img)

            data = cv2.resize(rightImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(2)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            qInRight.send(img)
            # print("right send")

            data = cv2.resize(bottomImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(3)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            qInVertical.send(img)
            # print("vertical send")

        elif args.vid:
            if not leftVideo.isOpened() or not rightVideo.isOpened() or not bottomVideo.isOpened():
                print("End of video")
                break
            read_correctly, leftImg = leftVideo.read()
            if not read_correctly:
                print("End of left video")
                break

            read_correctly, rightImg = rightVideo.read()
            if not read_correctly:
                print("End of the right video")
                break

            read_correctly, bottomImg = bottomVideo.read()
            if not read_correctly:
                print("End of the bottom video")
                break

            leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
            rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)
            bottomImg = cv2.cvtColor(bottomImg, cv2.COLOR_BGR2GRAY)

            ts = dai.Clock.now()
            data = cv2.resize(leftImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(1)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            # print("left send")
            qInLeft.send(img)

            data = cv2.resize(rightImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(2)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            qInRight.send(img)
            # print("right send")

            data = cv2.resize(bottomImg, (width, height), interpolation = cv2.INTER_AREA)
            data = data.reshape(height*width)
            img = dai.ImgFrame()
            img.setData(data)
            img.setInstanceNum(3)
            img.setType(dai.ImgFrame.Type.RAW8)
            img.setWidth(width)
            img.setHeight(height)
            img.setTimestamp(ts)
            qInVertical.send(img)
            # print("vertical send")
        if enableRectified:
            inRectifiedVertical = qRectifiedVertical.get()
            frameRVertical = inRectifiedVertical.getCvFrame()
            leftRectifiedVer.append(inRectifiedVertical.getFrame())
            cv2.imshow("rectified_vertical", frameRVertical)

            inRectifiedRight = qRectifiedRight.get()
            frameRRight = inRectifiedRight.getCvFrame()
            cv2.imshow("rectified_right", frameRRight)

            inRectifiedLeft = qRectifiedLeft.get()
            frameRLeft = inRectifiedLeft.getCvFrame()
            leftRectifiedHor.append(inRectifiedLeft.getFrame())
            cv2.imshow("rectified_left", frameRLeft)

        inDisparityVertical = qDisparityVertical.get()
        frameDepth = inDisparityVertical.getFrame()
        # cv2.imshow("disparity", frameDepth)
        tsV = inDisparityVertical.getTimestampDevice()
        
        frameDepthFrame = inDisparityVertical.getFrame()
        with np.errstate(divide='ignore'):
            frameDepthFrame = verScaleFactor / frameDepth
        frameDepthFrame[frameDepthFrame == np.inf] = 0
        # frameDepthFrame[frameDepthFrame == np.-inf] = 0
        # frameDepthFrame[frameDepthFrame == nan] = 0

        # print(verScaleFactor)
        # np.save("frameDepthVertical.npy", frameDepthFrame)
        verticalDepths.append(frameDepthFrame)
        # print(tsV)

        # disp = (frameDepth / 32).astype(np.uint8)
        # cv2.imshow("disparity_vertical", disp)
        maxDisp = stereoVertical.initialConfig.getMaxDisparity()
        disp = (frameDepth * (255.0 / maxDisp)).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        cvColorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        cvColorMap[0] = [0, 0, 0]
        disp = cv2.applyColorMap(disp, cvColorMap)
        cv2.imshow("depth_vertical", cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE))
        if args.saveFiles:
            cv2.imwrite("frameDepthVertical.png", cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE))
        inDisparityHorizontal = qDisparityHorizontal.get()
        frameDepth = inDisparityHorizontal.getFrame()

        frameDepthFrame = inDisparityHorizontal.getFrame()
        with np.errstate(divide='ignore'):
            frameDepthFrame = horScaleFactor / frameDepth
        frameDepthFrame[frameDepthFrame == np.inf] = 0
        # frameDepthFrame[frameDepthFrame == np.-inf] = 0
        # frameDepthFrame[frameDepthFrame == nan] = 0

        # np.save("frameDepthHorizontal.npy", frameDepthFrame)
        horizontalDepths.append(frameDepthFrame)
        # cv2.imshow("disparity", frameDepth)
        tsH = inDisparityHorizontal.getTimestampDevice()
        # print(tsH)
        disparityTsDiff = tsV-tsH
        # if disparityTsDiff:
        #     print(disparityTsDiff)

        disp = (frameDepth / 32).astype(np.uint8)
        # cv2.imshow("disparity_horizontal", disp)
        # maxDisp = stereoVertical.initialConfig.getMaxDisparity()
        # disp = (frameDepth * (255.0 / maxDisp)).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cvColorMap)
        cv2.imshow("depth_horizontal", disp)
        if args.saveFiles:
            cv2.imwrite("frameDepthHorizontal.png", disp)


        if cv2.waitKey(1) == ord('q'):
            break

stackedHorizontalDepths = np.stack(horizontalDepths, axis=0)
stackedVerticalDepths = np.stack(verticalDepths, axis=0)
if enableRectified:
    stackedRectifiedLeftHor = np.stack(leftRectifiedHor, axis=0)
    stackedRectifiedLeftVer = np.stack(leftRectifiedVer, axis=0)
if args.saveFiles:
    np.save(args.outHor, stackedHorizontalDepths)
    np.save(args.outVer, stackedVerticalDepths)
    if enableRectified:
        np.save(args.outLeftRectVer, stackedRectifiedLeftVer)
        np.save(args.outLeftRectHor, stackedRectifiedLeftHor)
