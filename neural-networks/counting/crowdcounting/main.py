from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgFrameOverlay, ApplyColormap

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

if args.fps_limit is None:
    args.fps_limit = 1 if platform == "RVC2" else 5
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # crowd counting model
    cc_model_description = dai.NNModelDescription(args.model)
    cc_model_description.platform = platform
    cc_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(cc_model_description))

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(
            dai.ImgFrame.Type.BGR888i
            if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )
        replay.setLoop(True)
        replay.setSize(
            cc_model_nn_archive.getInputWidth(), cc_model_nn_archive.getInputHeight()
        )
    input_node = replay if args.media_path else pipeline.create(dai.node.Camera).build()

    cc_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, cc_model_nn_archive, fps=args.fps_limit
    )

    # crowd density overlay
    color_transform_node = pipeline.create(ApplyColormap).build(cc_nn.out)
    overlay_node = pipeline.create(ImgFrameOverlay).build(
        cc_nn.passthrough,
        color_transform_node.out,
    )

    # annotation
    annotation_node = pipeline.create(AnnotationNode).build(cc_nn.out)

    # visualization
    visualizer.addTopic("VideoOverlay", overlay_node.out)
    visualizer.addTopic("Count", annotation_node.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
