"""
This example will calculate depth from disparity, and compare it to the depth calculated on the OAK camera.
"""

import depthai as dai
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculateDispScaleFactor(device : dai.Device):
    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, monoRight.getResolutionSize())
    focalLength = intrinsics[0][0]
    disp_levels = stereo.initialConfig.getMaxDisparity() / 95
    dispScaleFactor = baseline * focalLength * disp_levels
    return dispScaleFactor


class DispToDepthControl(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def setDispScaleFactor(self, dispScaleFactor):
        self.dispScaleFactor = dispScaleFactor

    def build(self, disp : dai.Node.Output, depth : dai.Node.Output) -> "DispToDepthControl": 
        self.link_args(disp, depth)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, disp : dai.ImgFrame, depth : dai.ImgFrame):
        dispFrame = np.array(disp.getFrame())
        with np.errstate(divide='ignore'):
            calcedDepth = (self.dispScaleFactor / dispFrame).astype(np.uint16)

        depthFrame = np.array(depth.getFrame())

        # Note: SSIM calculation is quite slow.
        ssim_noise = ssim(depthFrame, calcedDepth, data_range=65535)
        print(f'Similarity: {ssim_noise}')


device = dai.Device()
with dai.Pipeline(device) as pipeline:
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    dispScaleFactor = calculateDispScaleFactor(device)

    host = pipeline.create(DispToDepthControl).build(
        stereo.disparity,
        stereo.depth
    )
    host.setDispScaleFactor(dispScaleFactor)
    
    print("pipeline created")
    pipeline.run()
    print("pipeline finished")


