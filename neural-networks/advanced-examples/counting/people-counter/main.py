from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.object_counter import ObjectCounter
from utils.parser_bridge import ParserBridge

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

    model_description = dai.NNModelDescription(
        "luxonis/scrfd-person-detection:25g-640x640"
    )
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
        imageManip.initialConfig.setOutputSize(
            nn_archive.getInputWidth(),
            nn_archive.getInputHeight(),
            dai.ImageManipConfigV2.ResizeMode.STRETCH,
        )
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if platform == "RVC4":
            imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        replay.out.link(imageManip.inputImage)

    input_node = (
        imageManip.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )
    parser_bridge = pipeline.create(ParserBridge).build(nn_with_parser.out)
    object_counter = pipeline.create(ObjectCounter).build(
        nn=parser_bridge.out, label_map=["People"]
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough)
    visualizer.addTopic("Visualizations", nn_with_parser.out)
    visualizer.addTopic("Object count", object_counter.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
