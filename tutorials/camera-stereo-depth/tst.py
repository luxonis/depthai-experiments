#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

# Linking
monoLeftOut = monoLeft.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
monoRightOut = monoRight.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

syncedLeftQueue = stereo.syncedLeft.createOutputQueue()
syncedRightQueue = stereo.syncedRight.createOutputQueue()
disparityQueue = stereo.disparity.createOutputQueue()

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

with pipeline:
    pipeline.start()
    maxDisparity = 1
    while pipeline.isRunning():
        leftSynced = syncedLeftQueue.get()
        rightSynced = syncedRightQueue.get()
        disparity = disparityQueue.get()
        assert isinstance(leftSynced, dai.ImgFrame)
        assert isinstance(rightSynced, dai.ImgFrame)
        assert isinstance(disparity, dai.ImgFrame)
        cv2.imshow("left", leftSynced.getCvFrame())
        cv2.imshow("right", rightSynced.getCvFrame())
        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))
        colorizedDisparity = cv2.applyColorMap(
            ((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap
        )
        cv2.imshow("disparity", colorizedDisparity)
        key = cv2.waitKey(1)
        if key == ord("q"):
            pipeline.stop()
            break
