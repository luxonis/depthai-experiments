import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform()


with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput(
        size=(812, 608), fps=args.fps_limit, type=dai.ImgFrame.Type.NV12
    )

    model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=platform.name
    )
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    output_type = (
        dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p
    )

    img_manip = pipeline.create(dai.node.ImageManipV2)
    img_manip.initialConfig.setOutputSize(
        nn_archive.getInputWidth(),
        nn_archive.getInputHeight(),
        dai.ImageManipConfigV2.ResizeMode.LETTERBOX,
    )
    img_manip.initialConfig.setFrameType(output_type)
    cam_out.link(img_manip.inputImage)

    nn = pipeline.create(ParsingNeuralNetwork).build(img_manip.out, nn_archive)

    visualizer.addTopic("Resized", nn.passthrough, "images")
    visualizer.addTopic("Visualizations", nn.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
