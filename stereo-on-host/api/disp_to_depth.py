"""
This example will calculate depth from disparity, and compare it to the depth calculated on the OAK camera.
"""

import depthai as dai
import numpy as np
from skimage.metrics import structural_similarity as ssim

pipeline = dai.Pipeline()

# Define sources and outputs
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

# Properties
monoLeft = pipeline.createMonoCamera()
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.out.link(stereo.left)

monoRight = pipeline.createMonoCamera()
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.out.link(stereo.right)

xDisp = pipeline.createXLinkOut()
xDisp.setStreamName("disparity")
stereo.disparity.link(xDisp.input)

xDepth = pipeline.createXLinkOut()
xDepth.setStreamName("depth")
stereo.depth.link(xDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qDisp = device.getOutputQueue(name="disparity", maxSize=1, blocking=True)
    qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=True)

    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, monoRight.getResolutionSize())
    focalLength = intrinsics[0][0]
    disp_levels = stereo.getMaxDisparity() / 95
    dispScaleFactor = baseline * focalLength * disp_levels

    while True:
        dispFrame = np.array(qDisp.get().getFrame())
        with np.errstate(divide='ignore'):
            calcedDepth = (dispScaleFactor / dispFrame).astype(np.uint16)

        depthFrame = np.array(qDepth.get().getFrame())

        # Note: SSIM calculation is quite slow.
        ssim_noise = ssim(depthFrame, calcedDepth)
        print(f'Similarity: {ssim_noise}')
