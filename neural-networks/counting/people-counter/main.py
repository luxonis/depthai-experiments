from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # person detection model
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

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_model_nn_archive, fps=args.fps_limit
    )

    # person detection filter
    classes = det_model_nn_archive.getConfig().model.heads[0].metadata.classes
    labels_to_keep = [classes.index("person")] if "person" in classes else []
    det_filter = pipeline.create(ImgDetectionsFilter).build(
        det_nn.out, labels_to_keep=labels_to_keep, confidence_threshold=0.5
    )

    # annotation
    annotation_node = pipeline.create(AnnotationNode).build(det_filter.out)

    # visualization
    visualizer.addTopic("Video", det_nn.passthrough)
    visualizer.addTopic("PersonDetections", det_filter.out)
    visualizer.addTopic("PersonCount", annotation_node.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
