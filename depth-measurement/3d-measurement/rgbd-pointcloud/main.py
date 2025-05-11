import depthai as dai
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

IMG_SHAPE = (640, 400)

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
if not device.setIrLaserDotProjectorIntensity(1):
    print(
        "Failed to set IR laser projector intensity. Maybe your device does not support this feature."
    )
with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    calib_data = device.readCalibration()

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = left.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)
    right_out = right.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_out,
        right=right_out,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)
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
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 200000
    stereo.initialConfig.postProcessing.decimationFilter.decimationFactor = 1

    rgbd = pipeline.create(dai.node.RGBD).build()
    stereo.depth.link(rgbd.inDepth)

    width, height = IMG_SHAPE
    if args.mono:
        mono_out_from_right = right.requestOutput(
            IMG_SHAPE, type=dai.ImgFrame.Type.RGB888i
        )
        mono_out_from_right.link(rgbd.inColor)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)

        intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_C, dai.Size2f(width, height)
        )
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            IMG_SHAPE,
            dai.ImgFrame.Type.RGB888i,
        )
        cam_out.link(rgbd.inColor)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, dai.Size2f(width, height)
        )

    visualizer.addTopic("preview", stereo.rectifiedRight if args.mono else cam_out)
    visualizer.addTopic("pointcloud", rgbd.pcl)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
