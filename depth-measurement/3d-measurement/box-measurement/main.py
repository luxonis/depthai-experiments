from cv2 import STEREO_SGBM_MODE_HH
import depthai as dai

from utils.host_box_measurement import BoxMeasurement
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

# Higher resolution for example THE_720_P makes better results but drastically lowers FPS
RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P
# IMG_SHAPE = (1920, 1080)
IMG_SHAPE = (640, 400)
STEREO_SHAPE = (640, 400)

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
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_output = left.requestOutput(STEREO_SHAPE, type=dai.ImgFrame.Type.NV12)
    right_output = right.requestOutput(STEREO_SHAPE, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output,
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
    )
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(STEREO_SHAPE[0], STEREO_SHAPE[1])

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

    box_measurement = pipeline.create(BoxMeasurement).build(
        color=cam_out,
        depth=stereo.depth,
        cam_intrinsics=intrinsics,
        shape=STEREO_SHAPE,
        max_dist=args.max_dist,
        min_box_size=args.min_box_size,
    )
    box_measurement.inputs["color"].setBlocking(False)
    box_measurement.inputs["color"].setMaxSize(4)
    box_measurement.inputs["depth"].setBlocking(False)
    box_measurement.inputs["depth"].setMaxSize(4)

    
    # Configure visualizer to display box measurements
    visualizer.addTopic("Main Stream", box_measurement.passthrough, "images")
    visualizer.addTopic("Box Detection", box_measurement.annotation_output, "images")
    visualizer.addTopic("Dimensions", box_measurement.measurements_output, "images")

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
