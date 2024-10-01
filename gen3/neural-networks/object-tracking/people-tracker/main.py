import argparse
from pathlib import Path

import blobconverter
import depthai as dai
from host_people_tracker import PeopleTracker

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

model_description = dai.NNModelDescription(
    modelSlug="scrfd-person-detection", platform="RVC2", modelVersionSlug="2-5g-640x640"
)
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:
    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(544, 320)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)

        preview = replay.out

    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(544, 320)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        preview = cam.preview

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    # nn.setBlobPath(blobconverter.from_zoo(name="person-detection-retail-0013", shaves=7))
    nn.setNNArchive(nn_archive)
    nn.setConfidenceThreshold(0.5)
    nn.input.setBlocking(False)
    preview.link(nn.input)

    tracker = pipeline.create(dai.node.ObjectTracker)
    # Track people
    tracker.setDetectionLabelsToTrack([1])
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # Take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setTrackerThreshold(0.4)
    nn.passthrough.link(tracker.inputTrackerFrame)
    nn.passthrough.link(tracker.inputDetectionFrame)
    nn.out.link(tracker.inputDetections)

    people_tracker = pipeline.create(PeopleTracker).build(
        preview=tracker.passthroughTrackerFrame,
        tracklets=tracker.out,
        threshold=args.threshold,
    )

    print("Pipeline created.")
    pipeline.run()
