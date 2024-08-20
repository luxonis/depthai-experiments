import depthai as dai
import argparse
from pathlib import Path

from displayPeopleCounter import DisplayPeopleCounter
from hostNodes import FrameEditor, InputsConnector


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', default='depth-people-counting-01', type=str, help="Path to depthai-recording")
args = parser.parse_args()

PATH = Path(args.path).resolve().absolute()
SIZE = (1280, 800)


with dai.Pipeline() as pipeline:

    pipeline.setCalibrationData(dai.CalibrationHandler(str(PATH / 'calib.json')))
    
    left = pipeline.create(dai.node.ReplayVideo)
    left.setReplayVideoFile(PATH / 'left.mp4')
    left.setOutFrameType(dai.ImgFrame.Type.RAW8)
    left.setSize(SIZE)

    right = pipeline.create(dai.node.ReplayVideo)
    right.setReplayVideoFile(PATH / 'right.mp4')
    right.setOutFrameType(dai.ImgFrame.Type.RAW8)
    right.setSize(SIZE)

    left_frame_editor = pipeline.create(FrameEditor, dai.CameraBoardSocket.CAM_B)
    right_frame_editor = pipeline.create(FrameEditor, dai.CameraBoardSocket.CAM_C)

    left.out.link(left_frame_editor.input)
    right.out.link(right_frame_editor.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(left=left_frame_editor.output, right=right_frame_editor.output)

    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    object_tracker = pipeline.create(dai.node.ObjectTracker)
    object_tracker.inputTrackerFrame.setBlocking(True)
    object_tracker.inputDetectionFrame.setBlocking(True)
    object_tracker.inputDetections.setBlocking(True)
    object_tracker.setDetectionLabelsToTrack([1])  # track only person
    object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    connect_node = pipeline.create(InputsConnector) # need one InputQueue for two inputs
    connect_node.output.link(object_tracker.inputDetectionFrame)
    connect_node.output.link(object_tracker.inputTrackerFrame)
    
    disparity_multiplier = 255 / stereo.initialConfig.getMaxDisparity()

    pipeline.create(DisplayPeopleCounter).build(
        depth_in=stereo.disparity,
        tracklets_in=object_tracker.out,
        det_in_q=object_tracker.inputDetections.createInputQueue(),
        frame_in_q=connect_node.input.createInputQueue(),
        disparity_multiplier=disparity_multiplier
    )

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")