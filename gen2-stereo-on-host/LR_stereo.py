from stereo_on_host import StereoSGBM
import cv2
import numpy as np
import glob
from pathlib import Path
import depthai as dai

pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
xoutLeft = pipeline.createXLinkOut()
xoutRight = pipeline.createXLinkOut()

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)

device = dai.Device()
calibObj = device.readCalibration()

M_left = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoLeftCameraId(), 1280, 720))
M_right = np.array(calibObj.getCameraIntrinsics(calibObj.getStereoRightCameraId(), 1280, 720))
R1 = np.array(calibObj.getStereoLeftRectificationRotation())
R2 = np.array(calibObj.getStereoRightRectificationRotation())

H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
H_right = np.matmul(np.matmul(M_right, R2), np.linalg.inv(M_right))

stereo_obj = StereoSGBM(7.5, H_right, H_left)

right = None
left = None

device.startPipeline(pipeline)
qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)

while True:
    inLeft = qLeft.tryGet()
    inRight = qRight.tryGet()
    
    if inLeft is not None:
        cv2.imshow("left", inLeft.getCvFrame())
        left = inLeft.getCvFrame()

    if inRight is not None:
        cv2.imshow("right", inRight.getCvFrame())
        right = inRight.getCvFrame()

    if cv2.waitKey(1) == ord('q'):
        break
    
    if right is not None and left is not None:
        stereo_obj.create_disparity_map(left, right)
                
