from pathlib import Path

import depthai as dai
from depthai_nodes import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.parser_bridge import ParserBridge
from utils.people_counter import PeopleCounter
from utils.tracklet_visualizer import TrackletVisualizer

_, args = initialize_argparser()

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

    input_node = replay if args.media_path else pipeline.create(dai.node.Camera).build()

    nn = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive, fps=args.fps_limit
    )

    bridge = pipeline.create(ParserBridge).build(nn=nn.out)

    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack([0])
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setTrackerThreshold(0.4)
    nn.passthrough.link(tracker.inputTrackerFrame)
    nn.passthrough.link(tracker.inputDetectionFrame)
    bridge.output.link(tracker.inputDetections)

    tracklet_visualizer = pipeline.create(TrackletVisualizer).build(
        tracklets=tracker.out,
        labels=nn_archive.getConfigV1().model.heads[0].metadata.classes,
    )

    people_counter = pipeline.create(PeopleCounter).build(
        tracklets=tracker.out, threshold=args.threshold
    )

    visualizer.addTopic("Video", nn.passthrough, "images")
    visualizer.addTopic("Tracklets", tracklet_visualizer.out, "images")
    visualizer.addTopic("People count", people_counter.out, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
