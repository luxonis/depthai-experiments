import depthai as dai
from pathlib import Path
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.yuv2bgr import YUV2BGR

_, args = initialize_argparser()

if args.fps_limit and args.media_path:
    args.fps_limit = None
    print(
        "WARNING: FPS limit is set but media path is provided. FPS limit will be ignored."
    )

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription(args.model)
    platform = pipeline.getDefaultDevice().getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(model_description, apiKey=args.api_key)
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        imageManip = pipeline.create(dai.node.ImageManipV2)
        imageManip.setMaxOutputFrameSize(
            nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
        )
        imageManip.initialConfig.setOutputSize(
            nn_archive.getInputWidth(), nn_archive.getInputHeight()
        )
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if platform == "RVC4":
            imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        replay.out.link(imageManip.inputImage)

        input_node = imageManip.out

    else:
        cam = pipeline.create(dai.node.Thermal).build()

        # Thermal "color" output is YUV, model needs BGR
        yuv2bgr = pipeline.create(YUV2BGR)
        cam.color.link(yuv2bgr.input)

        input_node = yuv2bgr.out

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    parser = nn_with_parser.getParser()
    parser.setConfidenceThreshold(0.6)  # NOTE: Adjust if needed

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
