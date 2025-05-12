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
    platform = pipeline.getDefaultDevice().getPlatform()

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    left_out = left.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)
    right_out = right.requestOutput(IMG_SHAPE, type=dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_out,
        right=right_out,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)

    rgbd = pipeline.create(dai.node.RGBD).build()
    stereo.depth.link(rgbd.inDepth)

    width, height = IMG_SHAPE
    main_out = None
    if args.mono:
        mono_out_from_right = right.requestOutput(
            IMG_SHAPE, type=dai.ImgFrame.Type.RGB888i
        )
        mono_out_from_right.link(rgbd.inColor)
        if platform == dai.Platform.RVC4:
            stereo.depth.link(align.input)
            stereo.rectifiedRight.link(align.inputAlignTo)
        else:
            right_out.link(stereo.inputAlignTo)
        main_out = mono_out_from_right

    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            IMG_SHAPE,
            dai.ImgFrame.Type.RGB888i,
        )
        cam_out.link(rgbd.inColor)
        if platform == dai.Platform.RVC4:
            stereo.depth.link(align.input)
            cam_out.link(align.inputAlignTo)
            main_out = align.outputAligned
        else:
            cam_out.link(stereo.inputAlignTo)
            main_out = cam_out

    visualizer.addTopic("preview", main_out)
    visualizer.addTopic("pointcloud", rgbd.pcl)

    print("Pipeline created.")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
