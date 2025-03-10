import cv2
import depthai as dai
from utils.arguments import initialize_argparser
from utils.depth_color_transform import DepthColorTransform

color_resolution = (1280, 720)

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    color = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    color_output = color.requestOutput(color_resolution, dai.ImgFrame.Type.NV12)
    left_output = left.requestFullResolutionOutput(dai.ImgFrame.Type.NV12)
    right_output = right.requestFullResolutionOutput(dai.ImgFrame.Type.NV12)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_output,
        right=right_output,
        presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*color_resolution)

    depth_parser = pipeline.create(DepthColorTransform).build(stereo.disparity)
    depth_parser.setMaxDisparity(stereo.initialConfig.getMaxDisparity())
    depth_parser.setColormap(cv2.COLORMAP_JET)

    visualizer.addTopic("Color", color_output, "images")
    visualizer.addTopic("Left", left_output, "images")
    visualizer.addTopic("Right", right_output, "images")
    visualizer.addTopic("Depth", depth_parser.output, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
