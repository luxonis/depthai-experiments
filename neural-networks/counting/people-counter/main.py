from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.object_counter import ObjectCounter

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

    platform = pipeline.getDefaultDevice().getPlatformAsString()
    model_description = dai.NNModelDescription(
        "luxonis/scrfd-person-detection:25g-640x640", platform=platform
    )
    nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(
            dai.ImgFrame.Type.BGR888i
            if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        replay.setSize(nn_archive.getInputWidth(), nn_archive.getInputHeight())

    input_node = (
        replay.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    nn_with_parser: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )
    object_counter = pipeline.create(ObjectCounter).build(
        nn=nn_with_parser.out, label_map=["People"]
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
