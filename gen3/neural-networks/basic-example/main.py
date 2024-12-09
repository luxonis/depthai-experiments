import time
from pathlib import Path
import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

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

    model_description = dai.NNModelDescription(args.model_slug)
    platform = pipeline.getDefaultDevice().getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        imageManip = pipeline.create(dai.node.ImageManipV2)
        imageManip.setMaxOutputFrameSize(
            nn_archive.getInputWidth() * nn_archive.getInputHeight() * 3
        )
        imageManip.initialConfig.addResize(
            nn_archive.getInputWidth(), nn_archive.getInputHeight()
        )
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if platform == "RVC4":
            imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        replay.out.link(imageManip.inputImage)

    input_node = (
        imageManip.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        time.sleep(1 / 30)
