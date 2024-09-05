"""
This example will calculate depth from disparity, and compare it to the depth calculated on the OAK camera.
"""

import depthai as dai
from host_nodes.host_disp_to_depth import DispToDepthControl

RESOLUTION_SIZE = (640, 480)

def calculateDispScaleFactor(device : dai.Device):
    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, RESOLUTION_SIZE)
    focalLength = intrinsics[0][0]
    disp_levels = stereo.initialConfig.getMaxDisparity() / 95
    dispScaleFactor = baseline * focalLength * disp_levels
    return dispScaleFactor


device = dai.Device()
with dai.Pipeline(device) as pipeline:
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = monoLeft.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.GRAY8)
    
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = monoRight.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.GRAY8)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    
    dispScaleFactor = calculateDispScaleFactor(device)

    host = pipeline.create(DispToDepthControl).build(
        disp=stereo.disparity,
        depth=stereo.depth
    )
    host.setDispScaleFactor(dispScaleFactor)
    
    print("pipeline created")
    pipeline.run()
    print("pipeline finished")