import depthai as dai
import numpy as np

from utils.host_box_measurement import BoxMeasurement
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

IMG_SHAPE = (
    640,
    400,
)  # higher resolution is possible, but it will slow down fps drastically

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    platform = device.getPlatform()
    calib_data = device.readCalibration()
    device.setIrLaserDotProjectorIntensity(1)

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        IMG_SHAPE,
        dai.ImgFrame.Type.RGB888i,
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_output = left.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)
    right_output = right.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(IMG_SHAPE[0], IMG_SHAPE[1])

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
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 15000
    stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 1

    width, height = IMG_SHAPE
    intrinsics = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, dai.Size2f(width, height)
    )
    dist_coeffs = np.array(
        calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
    )

    cam_out.link(stereo.inputAlignTo)
    rgbd = pipeline.create(dai.node.RGBD).build()
    stereo.depth.link(rgbd.inDepth)
    cam_out.link(rgbd.inColor)

    box_measurement = pipeline.create(BoxMeasurement).build(
        color=cam_out,
        pcl=rgbd.pcl,
        cam_intrinsics=intrinsics,
        dist_coeffs=dist_coeffs,
        max_dist=args.max_dist,
        min_box_size=args.min_box_size,
    )
    box_measurement.color_input.setBlocking(False)
    box_measurement.color_input.setMaxSize(4)
    box_measurement.pcl_input.setBlocking(False)
    box_measurement.pcl_input.setMaxSize(4)

    visualizer.addTopic("Main Stream", box_measurement.passthrough, "images")
    visualizer.addTopic("Box Detection", box_measurement.annotation_output, "images")
    visualizer.addTopic("Dimensions", box_measurement.measurements_output, "images")
    visualizer.addTopic("Point Cloud", rgbd.pcl, "pcl")

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
