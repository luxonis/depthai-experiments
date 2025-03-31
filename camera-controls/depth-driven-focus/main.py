import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from util.arguments import initialize_argparser
from util.depth_color_transform import DepthColorTransform
from util.depth_driven_focus import DepthDrivenFocus
from util.depth_merger import DepthMerger

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription("luxonis/yunet:640x480")
    platform = device.getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = cam.requestOutput(
        nn_archive.getInputSize(), type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
    )

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = left.requestOutput(
        nn_archive.getInputSize(), type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
    )
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = right.requestOutput(
        nn_archive.getInputSize(), type=dai.ImgFrame.Type.NV12, fps=args.fps_limit
    )

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left_out, right=right_out)
    # print(stereo.initialConfig.getConfidenceThreshold())
    stereo.initialConfig.setConfidenceThreshold(240)
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)

    face_det_nn = pipeline.create(ParsingNeuralNetwork).build(cam, nn_archive)
    depth_merger = pipeline.create(DepthMerger).build(
        face_det_nn.out, stereo.depth, device.readCalibration2()
    )
    depth_color_transform = pipeline.create(DepthColorTransform).build(stereo.disparity)
    depth_color_transform.setMaxDisparity(stereo.initialConfig.getMaxDisparity())

    depth_driven_focus = pipeline.create(DepthDrivenFocus).build(
        control_queue=cam.inputControl.createInputQueue(),
        face_detection=depth_merger.output,
    )

    visualizer.addTopic("Video", color_out, "images")
    visualizer.addTopic("Visualizations", face_det_nn.out, "images")
    visualizer.addTopic("Depth", depth_color_transform.output, "images")
    visualizer.addTopic("Focus distance", depth_driven_focus.output, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
