from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.segmentation import SegAnnotationNode, DetSegAnntotationNode
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    model_description = dai.NNModelDescription(args.model)
    platform = device.getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            model_description,
            apiKey=args.api_key,
        )
    )

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

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    if args.annotation_mode == "segmentation":
        annotation_node = pipeline.create(SegAnnotationNode).build(
            nn_with_parser.passthrough, nn_with_parser.out
        )
        visualizer.addTopic("Video", annotation_node.output, "images")
    elif args.annotation_mode == "segmentation_with_annotation":
        annotation_node = pipeline.create(DetSegAnntotationNode).build(
            nn_with_parser.passthrough, nn_with_parser.out
        )
        visualizer.addTopic("Video", annotation_node.output, "images")
        visualizer.addTopic("Detections", nn_with_parser.out, "detections")
    else:
        visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
        visualizer.addTopic("Detections", nn_with_parser.out, "detections")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
