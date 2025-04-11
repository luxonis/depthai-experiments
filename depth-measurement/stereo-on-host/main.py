import depthai as dai

import depthai as dai
from util.arguments import initialize_argparser
from util.host_stereo_sgbm import StereoSGBM
from util.host_ssim import SSIM
from depthai_nodes.node import ApplyColormap
import cv2

RESOLUTION_SIZE = (640, 400)

_, args = initialize_argparser()

def calculateDispScaleFactor(device: dai.Device, stereo):
    calib = device.readCalibration()
    baseline = calib.getBaselineDistance(useSpecTranslation=True) * 10  # mm
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, RESOLUTION_SIZE)
    focalLength = intrinsics[0][0]
    disp_levels = stereo.initialConfig.getMaxDisparity() / 95
    dispScaleFactor = baseline * focalLength * disp_levels
    return dispScaleFactor

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()
with dai.Pipeline(device) as pipeline:
    mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left = mono_left.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.NV12)
    right = mono_right.requestOutput(size=RESOLUTION_SIZE, type=dai.ImgFrame.Type.NV12)

    cams = device.getConnectedCameras()
    depth_enabled = (
        dai.CameraBoardSocket.CAM_B in cams and dai.CameraBoardSocket.CAM_C in cams
    )
    if not depth_enabled:
        raise RuntimeError(
            "Unable to run this experiment on device without left & right cameras! (Available cameras: {})".format(
                cams
            )
        )

    calibObj = device.readCalibration()

    stereoSGBM = pipeline.create(StereoSGBM).build(
        monoLeftOut=left,
        monoRightOut=right,
        calibObj=calibObj,
        resolution=RESOLUTION_SIZE,
    )
    stereoSGBM.inputs["monoLeft"].setBlocking(False)
    stereoSGBM.inputs["monoLeft"].setMaxSize(2)
    stereoSGBM.inputs["monoRight"].setBlocking(False)
    stereoSGBM.inputs["monoRight"].setMaxSize(2)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left, right=right)
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)

    dispScaleFactor = calculateDispScaleFactor(device, stereo)

    ssim = pipeline.create(SSIM).build(
        disp=stereo.disparity, depth=stereo.depth
    )
    ssim.setDispScaleFactor(dispScaleFactor)

    depth_parser = pipeline.create(ApplyColormap).build(stereo.disparity)
    depth_parser.setMaxValue(int(stereo.initialConfig.getMaxDisparity()))
    depth_parser.setColormap(cv2.COLORMAP_JET)

    # visualizer.addTopic("SSIM Disparity", ssim.passthrough_disp, "ssim")
    # visualizer.addTopic("SSIM Stereo Depth", ssim.passthrough_depth, "ssim")
    visualizer.addTopic("SSIM score", ssim.output, "ssim")

    visualizer.addTopic("Depth test", depth_parser.out, "images")

    visualizer.addTopic("Left Cam", stereoSGBM.mono_left, "left")
    visualizer.addTopic("Right Cam", stereoSGBM.mono_right, "right")
    visualizer.addTopic("Disparity SGBM", stereoSGBM.disparity_out, "disparity")
    visualizer.addTopic("Rectified Left", stereoSGBM.rectified_left, "left")
    visualizer.addTopic("Rectified Right", stereoSGBM.rectified_right, "right")
    

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
