import depthai as dai

from utils.host_box_measurement import BoxMeasurement
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

# Higher resolution for example THE_720_P makes better results but drastically lowers FPS
RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    calib_data = device.readCalibration()
    device.setIrLaserDotProjectorIntensity(1)

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(1, 3)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.initialControl.setManualFocus(130)

    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(RESOLUTION)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(RESOLUTION)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left.out, right=right.out)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

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

    width, height = cam.getIspSize()
    intrinsics = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, dai.Size2f(width, height)
    )

    box_measurement = pipeline.create(BoxMeasurement).build(
        color=cam.isp,
        depth=stereo.depth,
        cam_intrinsics=intrinsics,
        shape=(width, height),
        max_dist=args.max_dist,
        min_box_size=args.min_box_size,
    )
    box_measurement.inputs["color"].setBlocking(False)
    box_measurement.inputs["color"].setMaxSize(4)
    box_measurement.inputs["depth"].setBlocking(False)
    box_measurement.inputs["depth"].setMaxSize(4)
    
    # Configure visualizer to display box measurements
    visualizer.addTopic("Box Measurement", box_measurement.output, "images")
    visualizer.addTopic("Dimensions", box_measurement.dimensions_output, "text")

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
