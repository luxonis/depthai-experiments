import argparse
import depthai as dai

from host_display_pointcloud import DisplayPointCloud

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--mono",
    default=False,
    action="store_true",
    help="use mono frame for pointcloud coloring instead of color frame",
)
args = parser.parse_args()

device = dai.Device()
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    calib_data = device.readCalibration()
    device.setIrLaserDotProjectorIntensity(1)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)

    """ In-place post-processing configuration for a stereo depth node
    The best combo of filters is application specific. Hard to say there is a one size fits all.
    They also are not free. Even though they happen on device, you pay a penalty in fps. """
    stereo.initialConfig.postProcessing.speckleFilter.enable = False
    stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 50
    stereo.initialConfig.postProcessing.temporalFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
    stereo.initialConfig.postProcessing.spatialFilter.numIterations = 1
    stereo.initialConfig.postProcessing.thresholdFilter.minRange = 400
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 200000
    stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 1

    if args.mono:
        width, height = right.getResolutionSize()
        intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_C, dai.Size2f(width, height)
        )
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(1, 3)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.initialControl.setManualFocus(130)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        width, height = cam.getIspSize()
        intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, dai.Size2f(width, height)
        )

    pointcloud = pipeline.create(dai.node.PointCloud)
    stereo.depth.link(pointcloud.inputDepth)

    display_pointcloud = pipeline.create(DisplayPointCloud).build(
        preview=stereo.rectifiedRight if args.mono else cam.isp,
        pointcloud=pointcloud.outputPointCloud,
    )

    print("Pipeline created.")
    pipeline.run()
