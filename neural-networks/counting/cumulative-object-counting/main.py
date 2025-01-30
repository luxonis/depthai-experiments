import argparse
import depthai as dai

from pathlib import Path
from cumulative_object_counting import CumulativeObjectCounting
from host_fps_drawer import FPSDrawer
from host_display import Display

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--video_path",
    type=str,
    default="",
    help="Path to video. If empty OAK-RGB camera is used.",
)
parser.add_argument(
    "-roi", "--roi_position", type=float, default=0.5, help="ROI Position (0-1)"
)
parser.add_argument(
    "-a",
    "--axis",
    default=True,
    action="store_false",
    help="Axis for cumulative counting (default=x axis)",
)
parser.add_argument(
    "-sp",
    "--save_path",
    type=str,
    default="",
    help="Path to save the output. If None output won't be saved",
)
args = parser.parse_args()

model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2")
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

with dai.Pipeline() as pipeline:
    if args.video_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setLoop(False)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setReplayVideoFile(args.video_path)
        replay.setSize((300, 300))
        video_out = replay.out

    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        video_out = cam.requestOutput((300, 300), dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.DetectionNetwork).build(
        input=video_out, nnArchive=dai.NNArchive(archive_path)
    )
    nn.setConfidenceThreshold(0.5)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    objectTracker = pipeline.create(dai.node.ObjectTracker)
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerIdAssignmentPolicy(
        dai.TrackerIdAssignmentPolicy.SMALLEST_ID
    )

    nn.passthrough.link(objectTracker.inputTrackerFrame)
    nn.passthrough.link(objectTracker.inputDetectionFrame)
    nn.out.link(objectTracker.inputDetections)

    counting = pipeline.create(CumulativeObjectCounting).build(
        img_frames=video_out, tracklets=objectTracker.out
    )
    counting.set_axis(args.axis)
    counting.set_roi_position(args.roi_position)

    fps_drawer = pipeline.create(FPSDrawer).build(counting.output)

    if args.save_path:
        img_manip = pipeline.create(dai.node.ImageManip)
        img_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        fps_drawer.output.link(img_manip.inputImage)

        videoEncoder = pipeline.create(dai.node.VideoEncoder).build(img_manip.out)
        videoEncoder.setProfile(dai.VideoEncoderProperties.Profile.H264_MAIN)

        record = pipeline.create(dai.node.RecordVideo)
        record.setRecordVideoFile(Path(args.save_path))
        videoEncoder.out.link(record.input)

    display = pipeline.create(Display).build(fps_drawer.output)
    display.setName("Cumulative Object Counting")

    print("Pipeline created.")
    pipeline.run()
