import argparse
from pathlib import Path

import depthai as dai
from depthai_nodes.ml.parsers import SCRFDParser
from host_node.draw_tracklet import DrawTracklet
from host_node.host_display import Display
from host_node.normalize_tracklet import NormalizeTracklet
from host_node.parser_bridge import ParserBridge
from people_counter import PeopleCounter

parser = argparse.ArgumentParser()
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video to use for inference. Otherwise uses the DepthAI color camera",
)
parser.add_argument(
    "-t",
    "--threshold",
    default=0.25,
    type=float,
    help="Minimum distance the person has to move (across the x/y axis) to be considered a real movement. Default: 0.25",
)
args = parser.parse_args()

device = dai.Device()
model_description = dai.NNModelDescription(
    modelSlug="scrfd-person-detection",
    platform=device.getPlatform().name,
    modelVersionSlug="2-5g-640x640",
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(640, 640)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        frame_type = pipeline.create(dai.node.ImageManip)
        frame_type.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        frame_type.setMaxOutputFrameSize(640 * 640 * 3)
        replay.out.link(frame_type.inputImage)

        preview = frame_type.out

    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        preview = cam.requestOutput(size=(640, 640), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    nn.input.setBlocking(False)
    preview.link(nn.input)
    nn_parser = pipeline.create(SCRFDParser)
    nn_parser.setFeatStrideFPN((8, 16, 32, 64, 128))
    nn_parser.setNumAnchors(1)
    nn.out.link(nn_parser.input)
    bridge = pipeline.create(ParserBridge).build(nn=nn_parser.out)

    tracker = pipeline.create(dai.node.ObjectTracker)
    # Track people
    tracker.setDetectionLabelsToTrack([0])
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # Take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setTrackerThreshold(0.4)
    nn.passthrough.link(tracker.inputTrackerFrame)
    nn.passthrough.link(tracker.inputDetectionFrame)
    bridge.output.link(tracker.inputDetections)

    normalize_tracklet = pipeline.create(NormalizeTracklet).build(
        frame=preview, tracklets=tracker.out
    )
    draw_tracklet = pipeline.create(DrawTracklet).build(
        frame=preview, tracklets=normalize_tracklet.output
    )
    draw_tracklet.set_color((0, 0, 255))

    people_counter = pipeline.create(PeopleCounter).build(
        preview=draw_tracklet.output, tracklets=tracker.out, threshold=args.threshold
    )
    display = pipeline.create(Display).build(frames=people_counter.output)

    print("Pipeline created.")
    pipeline.run()
