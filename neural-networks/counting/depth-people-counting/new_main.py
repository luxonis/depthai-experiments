import depthai as dai
from pathlib import Path
from hostNodes import FrameEditor
from peopleDetector import PeopleDetector

from utils.arguments import initialize_argparser

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

if args.recording:
    PATH = Path(args.recording)
else:
    PATH = (Path(__file__).parent / "resources").resolve().absolute()

SIZE = (1280, 800)


with dai.Pipeline(device) as pipeline:
    pipeline.setCalibrationData(dai.CalibrationHandler(str(PATH / "calib.json")))

    left = pipeline.create(dai.node.ReplayVideo)
    left.setReplayVideoFile(PATH / "left.mp4")
    left.setOutFrameType(dai.ImgFrame.Type.RAW8)
    left.setSize(SIZE)

    right = pipeline.create(dai.node.ReplayVideo)
    right.setReplayVideoFile(PATH / "right.mp4")
    right.setOutFrameType(dai.ImgFrame.Type.RAW8)
    right.setSize(SIZE)

    left_frame_editor = pipeline.create(FrameEditor, dai.CameraBoardSocket.CAM_B)
    right_frame_editor = pipeline.create(FrameEditor, dai.CameraBoardSocket.CAM_C)

    left.out.link(left_frame_editor.input)
    right.out.link(right_frame_editor.input)

    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_frame_editor.output, right=right_frame_editor.output
    )

    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)

    disparity_multiplier = 255 / stereo.initialConfig.getMaxDisparity()

    people_detector = pipeline.create(PeopleDetector).build(
        depth=stereo.disparity,
        disparity_multiplier=disparity_multiplier,
    )

    # object_tracker = pipeline.create(dai.node.ObjectTracker)
    # object_tracker.setDetectionLabelsToTrack([0])  # track only person
    # object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    # object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    # people_detector.out.link(object_tracker.inputDetections)
    # people_detector.out_depth_rgb.link(object_tracker.inputDetectionFrame)
    # people_detector.out_depth_rgb.link(object_tracker.inputTrackerFrame)

    # connect_node = pipeline.create(
    #     InputsConnector
    # )  # need one InputQueue for two inputs
    # connect_node.output.link(object_tracker.inputDetectionFrame)
    # connect_node.output.link(object_tracker.inputTrackerFrame)

    # print("Pipeline created.")
    # pipeline.run()
    # print("Pipeline finished.")

    # apply_colormap = pipeline.create(ApplyColormap).build(
    # stereo.disparity,
    # )

    # visualization
    # visualizer.addTopic("Video", apply_colormap.out)
    visualizer.addTopic("Video", people_detector.out_depth_rgb)
    # visualizer.addTopic("Debug", people_detector.out_debug)
    # visualizer.addTopic("Detections", people_detector.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            print("Got q key. Exiting...")
            break
