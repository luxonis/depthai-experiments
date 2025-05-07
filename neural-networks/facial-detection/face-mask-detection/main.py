from pathlib import Path
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsFilter,
    ImgDetectionsBridge,
)
from utils.arguments import initialize_argparser

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

# define relevant labels
LABEL_ENCODING = {
    1: "mask",
    3: "no_mask",
}

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # mask detection model
    det_model_description = dai.NNModelDescription(args.model)
    det_model_description.platform = platform
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

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
            det_model_nn_archive.getInputWidth(), det_model_nn_archive.getInputHeight()
        )
    input_node = replay if args.media_path else pipeline.create(dai.node.Camera).build()

    det_nn = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_model_nn_archive, fps=args.fps_limit
    )

    # filter and rename detection labels
    det_process_filter = pipeline.create(ImgDetectionsFilter).build(det_nn.out)
    det_process_filter.setLabels(list(LABEL_ENCODING.keys()), keep=True)
    det_process_bridge = pipeline.create(ImgDetectionsBridge).build(
        det_process_filter.out, label_encoding=LABEL_ENCODING
    )

    # visualization
    visualizer.addTopic("Video", det_nn.passthrough, "images")
    visualizer.addTopic("Detections", det_process_bridge.out, "detections")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
