#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
from utils import stereoRectify, rotate_mesh_90_ccw, downSampleMesh
import config
from config import args




device = dai.Device()
# Create pipeline
pipeline = dai.Pipeline()

if args.calib is None:
    calibData = device.readCalibration()
else:
    calibData = dai.CalibrationHandler(args.calib)
    pipeline.setCalibrationData(calibData)

if config.staticInput or args.vid:
    monoLeft = pipeline.create(dai.node.XLinkIn)
    monoLeft2 = pipeline.create(dai.node.XLinkIn)
    monoRight = pipeline.create(dai.node.XLinkIn)
    monoVertical = pipeline.create(dai.node.XLinkIn)
    monoLeft.setStreamName("inLeft")
    monoLeft2.setStreamName("inLeft2")
    monoRight.setStreamName("inRight")
    monoVertical.setStreamName("inVertical")
else:
    monoLeft = pipeline.create(dai.node.ColorCamera)
    monoVertical = pipeline.create(dai.node.ColorCamera)
    monoRight = pipeline.create(dai.node.ColorCamera)
    monoLeft.setFps(config.cameraFPS)
    monoVertical.setFps(config.cameraFPS)
    monoRight.setFps(config.cameraFPS)

if config.enableRectified:
    xoutRectifiedVertical = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedRight = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedLeft = pipeline.create(dai.node.XLinkOut)
    xoutRectifiedRightHor = pipeline.create(dai.node.XLinkOut)
xoutDisparityVertical = pipeline.create(dai.node.XLinkOut)
xoutDisparityHorizontal = pipeline.create(dai.node.XLinkOut)
stereoVertical = pipeline.create(dai.node.StereoDepth)
stereoHorizontal = pipeline.create(dai.node.StereoDepth)
# syncNode = pipeline.create(dai.node.Sync)

if config.enableRectified:
    xoutRectifiedVertical.setStreamName("rectified_vertical")
    xoutRectifiedRight.setStreamName("rectified_right")
    xoutRectifiedLeft.setStreamName("rectified_left")
    xoutRectifiedRightHor.setStreamName("rectified_right_hor")
xoutDisparityVertical.setStreamName("disparity_vertical")
xoutDisparityHorizontal.setStreamName("disparity_horizontal")

# Define sources and outputs
if not config.staticInput and not args.vid:
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
if not config.staticInput and not args.vid:
    monoLeft.video.link(stereoVertical.left) # left input is bottom camera
    monoVertical.video.link(stereoVertical.right) # right input is right camera
else:
    monoLeft2.out.link(stereoVertical.left) # left input is bottom camera
    monoVertical.out.link(stereoVertical.right) # right input is right camera

stereoVertical.disparity.link(xoutDisparityVertical.input)
if config.enableRectified:
    stereoVertical.rectifiedLeft.link(xoutRectifiedVertical.input)
    stereoVertical.rectifiedRight.link(xoutRectifiedRight.input)
if args.fullResolution:
    stereoVertical.setVerticalStereo(False)
    stereoVertical.setRectification(False)
else:
    stereoVertical.setVerticalStereo(True)

# syncNode.output2.link(stereoHorizontal.left)
# syncNode.output1.link(stereoHorizontal.right)
if not config.staticInput and not args.vid:
    monoLeft.video.link(stereoHorizontal.left)
    monoRight.video.link(stereoHorizontal.right)
else:
    monoLeft.out.link(stereoHorizontal.left)
    monoRight.out.link(stereoHorizontal.right)

stereoHorizontal.disparity.link(xoutDisparityHorizontal.input)
if config.enableRectified:
    stereoHorizontal.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereoHorizontal.rectifiedRight.link(xoutRectifiedRightHor.input)

stereoHorizontal.setVerticalStereo(False)
if args.fullResolution:
    stereoHorizontal.setRectification(False)


stereoHorizontal.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
stereoVertical.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)

if 1:
    if args.fullResolution:
        width = 1920
        height = 1200
    else:
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

    if config.forceFisheye or (len(d1) == 4 and len(d2) == 4):

        def calc_fov_D_H_V(f, w, h):
            return np.degrees(2*np.arctan(np.sqrt(w*w+h*h)/(2*f))), np.degrees(2*np.arctan(w/(2*f))), np.degrees(2*np.arctan(h/(2*f)))
        
        M1_focal = M1
        M1_focal = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(M1, d1, resolution, R1)
        print(calc_fov_D_H_V(M1_focal[0][0], width, height))
        focalVer = M1_focal[1][1]
        mapXL_v, mapYL_v = cv2.fisheye.initUndistortRectifyMap(M1, d1, R1, M1_focal, resolution, cv2.CV_32FC1)
        mapXL, mapYL = cv2.fisheye.initUndistortRectifyMap(M1, d1, R1, M1_focal, resolution, cv2.CV_32FC1)
        mapXV, mapYV = cv2.fisheye.initUndistortRectifyMap(M2, d2, R2, M1_focal, resolution, cv2.CV_32FC1)
    else:
        mapXL_v, mapYL_v = cv2.initUndistortRectifyMap(M1, d1, R1, M1, resolution, cv2.CV_32FC1)
        mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M1, resolution, cv2.CV_32FC1)
        mapXV, mapYV = cv2.initUndistortRectifyMap(M2, d2, R2, M1, resolution, cv2.CV_32FC1)

    verScaleFactor = baselineVer * focalVer * 32


    mapXL_rot, mapYL_rot = rotate_mesh_90_ccw(mapXL, mapYL)
    mapXV_rot, mapYV_rot = rotate_mesh_90_ccw(mapXV, mapYV)
    if not args.fullResolution:
        #clip for now due to HW limit
        mapXV_rot = mapXV_rot[:1024,:]
        mapYV_rot = mapYV_rot[:1024,:]
        mapXL_rot = mapXL_rot[:1024,:]
        mapYL_rot = mapYL_rot[:1024,:]

        leftMeshRot, verticalMeshRot = downSampleMesh(mapXL_rot, mapYL_rot, mapXV_rot, mapYV_rot)

        meshLeft = list(leftMeshRot.tobytes())
        meshVertical = list(verticalMeshRot.tobytes())
        stereoVertical.loadMeshData(meshLeft, meshVertical)
    else:
        stereoVertical.setRectification(False)

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
    if args.fullResolution:
        pt1 = dai.Point2f(config.cropXStart, config.cropYStart)
        pt2 = dai.Point2f(config.cropXEnd, config.cropYEnd)
        M1_ = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, width, height, pt1, pt2))
        print(f"orig {M1[0][0]}, {M1_[0][0]}")

    print(f"baseline {baselineHor}, focal {focalHor}")

    if not np.any(d1[4:]) and not np.any(d2[4:]):
        print("Fisheye model detected!")
        d1 = d1[:4]
        d2 = d2[:4]

    if config.forceFisheye or (len(d1) == 4 and len(d2) == 4): 
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

    if not args.fullResolution:
        meshLeft = list(leftMesh.tobytes())
        meshRight = list(rightMesh.tobytes())
        stereoHorizontal.loadMeshData(meshLeft, meshRight)

    horScaleFactor = baselineHor * focalHor * 32


    if args.debug and config.staticInput:
        img_l_rectified = cv2.remap(config.leftImg, mapXL, mapYL, cv2.INTER_LINEAR)
        cv2.imshow("img_l_rectified", img_l_rectified)

        img_r_rectified = cv2.remap(config.rightImg, mapXR, mapYR, cv2.INTER_LINEAR)
        cv2.imshow("img_r_rectified", img_r_rectified)

        img_r_vertical = cv2.remap(config.rightImg, mapXR_rot, mapYR_rot, cv2.INTER_LINEAR)
        cv2.imshow("img_r_vertical", img_r_vertical)

        img_b_vertical = cv2.remap(config.bottomImg, mapXV_rot, mapYV_rot, cv2.INTER_LINEAR)
        cv2.imshow("img_b_vertical", img_b_vertical)


# Connect to device and start pipeline
verticalDepths = []
horizontalDepths = []
leftRectifiedVer = []
leftRectifiedHor = []

def sendFrames(leftImg, rightImg, bottomImg, qInLeft, qInLeft2, qInRight, qInVertical, width, height, args):
    ts = dai.Clock.now()
    if not args.fullResolution:
        data = cv2.resize(leftImg, (width, height), interpolation = cv2.INTER_AREA)
        data = data.reshape(height*width)
        sendFrame(data, qInLeft, width, height, ts, 1)

        data = cv2.resize(leftImg, (width, height), interpolation = cv2.INTER_AREA)
        data = data.reshape(height*width)
        sendFrame(data, qInLeft2, width, height, ts, 1)

        data = cv2.resize(rightImg, (width, height), interpolation = cv2.INTER_AREA)
        data = data.reshape(height*width)
        sendFrame(data, qInRight, width, height, ts, 2)
        # print("right send")

        data = cv2.resize(bottomImg, (width, height), interpolation = cv2.INTER_AREA)
        data = data.reshape(height*width)
        sendFrame(data, qInVertical, width, height, ts, 3)
        # print("vertical send")
    else:
        # data = cv2.resize(leftImg, (width, height), interpolation = cv2.INTER_AREA)
        data = cv2.remap(leftImg, mapXL, mapYL, cv2.INTER_LINEAR)
        data = data[config.cropYStart:config.cropYEnd,config.cropXStart:config.cropXEnd]
        print(data.shape)
        data = data.flatten()
        sendFrame(data, qInLeft, config.cropWidth, config.cropHeight, ts, 1)



        # data = cv2.resize(rightImg, (width, height), interpolation = cv2.INTER_AREA)
        data = cv2.remap(rightImg, mapXR, mapYR, cv2.INTER_LINEAR)
        data = data[config.cropYStart:config.cropYEnd,config.cropXStart:config.cropXEnd]
        data = data.flatten()
        sendFrame(data, qInRight, config.cropWidth, config.cropHeight, ts, 2)


        data = cv2.remap(bottomImg, mapXV, mapYV, cv2.INTER_LINEAR)
        print(data.shape)

        data = data[config.cropYStart:config.cropYEnd,config.cropXStart:config.cropXEnd]
        print(data.shape)

        data = data[:,config.cropstart:config.cropstart+config.cropLength]
        data = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(data.shape)
        vertHeight = data.shape[0]
        vertWidth = data.shape[1]

        data = data.reshape(vertWidth*vertHeight)
        sendFrame(data, qInVertical, vertWidth, vertHeight, ts, 3)

        data = cv2.remap(leftImg, mapXL_v, mapYL_v, cv2.INTER_LINEAR)
        data = data[config.cropYStart:config.cropYEnd,config.cropXStart:config.cropXEnd]
        data = data[:,config.cropstart:config.cropstart+config.cropLength]
        data = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        vertHeight = data.shape[0]
        vertWidth = data.shape[1]

        data = data.reshape(vertWidth*vertHeight)
        sendFrame(data, qInLeft2, vertWidth, vertHeight, ts, 1)


def sendFrame(data, qIn, width, height, ts, instanceNum):
    img = dai.ImgFrame()
    img.setData(data)
    img.setInstanceNum(instanceNum)
    img.setType(dai.ImgFrame.Type.RAW8)
    img.setWidth(width)
    img.setHeight(height)
    img.setTimestamp(ts)
    qIn.send(img)



with device:
    device.startPipeline(pipeline)
    qDisparityHorizontal = device.getOutputQueue("disparity_horizontal", 4,  config.blockingOutputs)
    qDisparityVertical = device.getOutputQueue("disparity_vertical", 4,  config.blockingOutputs)
    if config.enableRectified:
        qRectifiedVertical = device.getOutputQueue("rectified_vertical", 4,  config.blockingOutputs)
        qRectifiedRight = device.getOutputQueue("rectified_right", 4,  config.blockingOutputs)
        qRectifiedLeft = device.getOutputQueue("rectified_left", 4,  config.blockingOutputs)
        qRectifiedRightHor = device.getOutputQueue("rectified_right_hor", 4,  config.blockingOutputs)

    if config.staticInput or args.vid:
        qInLeft = device.getInputQueue("inLeft")
        qInRight = device.getInputQueue("inRight")
        qInLeft2 = device.getInputQueue("inLeft2")
        qInVertical = device.getInputQueue("inVertical")

    while True:
        if config.staticInput:
            sendFrames(leftImg, rightImg, bottomImg, qInLeft, qInLeft2, qInRight, qInVertical, width, height, args)

        elif args.vid:
            if not  config. config.leftVideo.isOpened() or not  config.rightVideo.isOpened() or not  config.bottomVideo.isOpened():
                print("End of video")
                break
            read_correctly, leftImg =  config.leftVideo.read()
            if not read_correctly:
                print("End of left video")
                break

            read_correctly, rightImg =  config.rightVideo.read()
            if not read_correctly:
                print("End of the right video")
                break

            read_correctly, bottomImg =  config.bottomVideo.read()
            if not read_correctly:
                print("End of the bottom video")
                break

            leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
            rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)
            bottomImg = cv2.cvtColor(bottomImg, cv2.COLOR_BGR2GRAY)

            sendFrames(leftImg, rightImg, bottomImg, qInLeft, qInLeft2, qInRight, qInVertical, width, height, args)
        if config.enableRectified:
            inRectifiedVertical = qRectifiedVertical.get()
            frameRVertical = inRectifiedVertical.getCvFrame()
            leftRectifiedVer.append(inRectifiedVertical.getFrame())
            cv2.imshow("rectified_vertical", frameRVertical)
            if args.saveFiles:
                cv2.imwrite(str(Path(args.outDir) / "ver_rectified_vertical_camb.png"), frameRVertical)

            inRectifiedRight = qRectifiedRight.get()
            frameRRight = inRectifiedRight.getCvFrame()
            cv2.imshow("rectified_right", frameRRight)
            if args.saveFiles:
                cv2.imwrite(str(Path(args.outDir) / "ver_rectified_right_camd.png"), frameRRight)

            inRectifiedLeft = qRectifiedLeft.get()
            frameRLeft = inRectifiedLeft.getCvFrame()
            leftRectifiedHor.append(inRectifiedLeft.getFrame())
            cv2.imshow("rectified_left", frameRLeft)
            if args.saveFiles:
                cv2.imwrite(str(Path(args.outDir) / "hor_rectified_left_camb.png"), frameRLeft)

            inRectifiedRightHor = qRectifiedRightHor.get()
            frameRRightHor = inRectifiedRightHor.getCvFrame()
            cv2.imshow("rectified_right_hor", frameRRightHor)
            if args.saveFiles:
               cv2.imwrite(str(Path(args.outDir) / "hor_rectified_right_camc.png"), frameRRightHor)


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
            cv2.imwrite(str(Path(args.outDir)/"frameDepthVertical.png"), cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE))
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
            cv2.imwrite(str(Path(args.outDir)/"frameDepthHorizontal.png"), disp)


        if cv2.waitKey(1) == ord('q'):
            break

stackedHorizontalDepths = np.stack(horizontalDepths, axis=0)
stackedVerticalDepths = np.stack(verticalDepths, axis=0)
if config.enableRectified:
    stackedRectifiedLeftHor = np.stack(leftRectifiedHor, axis=0)
    stackedRectifiedLeftVer = np.stack(leftRectifiedVer, axis=0)
if args.saveFiles:
    np.save(Path(args.outDir)/"horizontalDepth.npy", stackedHorizontalDepths)
    np.save(Path(args.outDir)/"verticalDepth.npy", stackedVerticalDepths)
    if config.enableRectified:
        np.save(Path(args.outDir)/"leftRectifiedVertical.npy", stackedRectifiedLeftVer)
        np.save(Path(args.outDir)/"leftRectifiedHorizontal.npy", stackedRectifiedLeftHor)


