import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

_, args = initialize_argparser()


device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

visualizer = dai.RemoteConnection(httpPort=8082)


with dai.Pipeline(device) as pipeline:
    platform = device.getPlatform()

    model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=platform.name
    )
    archive_path = dai.getModelFromZoo(model_description)
    nn_archive = dai.NNArchive(archivePath=archive_path)

    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(
        (512, 288),
        fps=args.fps_limit,
        type=dai.ImgFrame.Type.BGR888i
        if platform == dai.Platform.RVC4
        else dai.ImgFrame.Type.BGR888p,
    )

    nn = pipeline.create(ParsingNeuralNetwork).build(cam_out, nn_archive)

    visualizer.addTopic("Passthrough", nn.passthrough)
    visualizer.addTopic("Detections", nn.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
