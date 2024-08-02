from pathlib import Path
import blobconverter
import argparse
import depthai as dai

from cumulative_object_counting import CumulativeObjectCounting
from display import Display

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', type=str, help='File path of .blob file.')
parser.add_argument('-v', '--video_path', type=str, default='',
                    help='Path to video. If empty OAK-RGB camera is used. (default=\'\')')
parser.add_argument('-roi', '--roi_position', type=float,
                    default=0.5, help='ROI Position (0-1)')
parser.add_argument('-a', '--axis', default=True, action='store_false',
                    help='Axis for cumulative counting (default=x axis)')
parser.add_argument('-sp', '--save_path', type=str, default='',
                    help='Path to save the output. If None output won\'t be saved')
args = parser.parse_args()

if args.model is None:
    args.model = blobconverter.from_zoo(name="mobilenet-ssd", shaves=7)

with dai.Pipeline() as pipeline:
    output_size = (300, 300)
    if args.video_path != '':
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setLoop(False)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setReplayVideoFile(args.video_path)
        replay.setSize(output_size)
        video_out = replay.out
        fps = replay.getFps()
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(output_size)
        cam.setInterleaved(False)
        video_out = cam.preview
        fps = cam.getFps()

    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(args.model)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    video_out.link(nn.input)

    objectTracker = pipeline.create(dai.node.ObjectTracker)
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    nn.passthrough.link(objectTracker.inputTrackerFrame)
    nn.passthrough.link(objectTracker.inputDetectionFrame)
    nn.out.link(objectTracker.inputDetections)

    counting = pipeline.create(CumulativeObjectCounting).build(video_out, objectTracker.out)
    counting.set_axis(args.axis)
    counting.set_roi_position(args.roi_position)
    counting.set_show(args.show)

    if args.save_path:
        img_manip = pipeline.create(dai.node.ImageManip)
        img_manip.initialConfig.setResize(320, 320)
        img_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
        counting.output.link(img_manip.inputImage)

        videoEncoder = pipeline.create(dai.node.VideoEncoder).build(img_manip.out)
        videoEncoder.setProfile(dai.VideoEncoderProperties.Profile.H264_MAIN)

        record = pipeline.create(dai.node.RecordVideo)
        record.setRecordVideoFile(Path(args.save_path))
        videoEncoder.out.link(record.input)

    display = pipeline.create(Display).build(counting.output)
    display.set_name("Cumulative Object Counting")
    pipeline.run()