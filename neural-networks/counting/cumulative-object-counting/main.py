from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()
visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

if platform != "RVC2":
    raise ValueError(f"This experiment is only supported for RVC2 platform.")


with dai.Pipeline(device) as pipeline:

    # model
    model_description = dai.NNModelDescription(args.model)
    model_description.platform = platform
    model_nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
    model_input_width, model_input_height = (
        model_nn_archive.getInputWidth(),
        model_nn_archive.getInputHeight(),
    )

    # media/camera input
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
        replay.setSize(model_input_width, model_input_height)
        input_node = replay.out
    else:
        cam = pipeline.create(dai.node.Camera).build()
        input_node = cam.requestOutput(
            size=(model_input_width, model_input_height),
            type=frame_type,
            fps=args.fps_limit,
        )

    nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, model_nn_archive
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
