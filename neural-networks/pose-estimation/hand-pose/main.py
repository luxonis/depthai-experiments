from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from utils.process import ProcessDetections

_, args = initialize_argparser()

detection_model_slug: str = "luxonis/mediapipe-palm-detection:192x192"
pose_model_slug: str = "luxonis/mediapipe-hand-landmarker:224x224"

PADDING = 0.1
CONFIDENCE_THRESHOLD = 0.5

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

if not args.fps_limit:
    args.fps_limit = 30.0 if platform == "RVC4" else 6.0

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(detection_model_slug)
    print(f"Platform: {platform}")
    detection_model_description.platform = platform
    detection_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description)
    )

    pose_model_description = dai.NNModelDescription(pose_model_slug)
    pose_model_description.platform = platform
    pose_nn_archive = dai.NNArchive(dai.getModelFromZoo(pose_model_description))
    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )
    connection_pairs = (
        pose_nn_archive.getConfig().model.heads[0].metadata.extraParams["connections"]
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(args.fps_limit)

    else:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (768, 768), dai.ImgFrame.Type.NV12, fps=args.fps_limit
        )
    video_output = replay.out if args.media_path else cam_out

    imageManip = pipeline.create(dai.node.ImageManipV2)
    imageManip.setMaxOutputFrameSize(
        detection_nn_archive.getInputWidth() * detection_nn_archive.getInputHeight() * 3
    )
    imageManip.initialConfig.setOutputSize(
        detection_nn_archive.getInputWidth(),
        detection_nn_archive.getInputHeight(),
        mode=dai.ImageManipConfigV2.ResizeMode.STRETCH,
    )
    imageManip.initialConfig.setFrameType(frame_type)
    video_output.link(imageManip.inputImage)

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        imageManip.out, detection_nn_archive
    )

    detections_processor = pipeline.create(ProcessDetections).build(
        detections_input=detection_nn.out,
        padding=PADDING,
        target_size=(pose_nn_archive.getInputWidth(), pose_nn_archive.getInputHeight()),
    )

    script = pipeline.create(dai.node.Script)
    script.setScriptPath(str(Path(__file__).parent / "utils/script.py"))
    script.inputs["frame_input"].setMaxSize(30)
    script.inputs["config_input"].setMaxSize(30)
    script.inputs["num_configs_input"].setMaxSize(30)

    detection_nn.passthrough.link(script.inputs["frame_input"])
    detections_processor.config_output.link(script.inputs["config_input"])
    detections_processor.num_configs_output.link(script.inputs["num_configs_input"])

    pose_manip = pipeline.create(dai.node.ImageManipV2)
    pose_manip.initialConfig.setOutputSize(
        pose_nn_archive.getInputWidth(), pose_nn_archive.getInputHeight()
    )
    pose_manip.inputConfig.setMaxSize(30)
    pose_manip.inputImage.setMaxSize(30)
    pose_manip.setNumFramesPool(30)
    pose_manip.inputConfig.setWaitForMessage(True)

    script.outputs["output_config"].link(pose_manip.inputConfig)
    script.outputs["output_frame"].link(pose_manip.inputImage)

    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_nn_archive
    )

    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=detection_nn.out,
        padding_factor=PADDING,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        connections_pairs=connection_pairs,
    )
    pose_nn.getOutput(0).link(
        annotation_node.input_keypoints
    )  # First head is for keypoints
    pose_nn.getOutput(1).link(
        annotation_node.input_confidence
    )  # Second head is for confidence score
    pose_nn.getOutput(2).link(
        annotation_node.input_handedness
    )  # Third head is for handedness

    visualizer.addTopic("Video", video_output, "images")
    visualizer.addTopic("Detections", annotation_node.out_detections, "images")
    visualizer.addTopic("Pose", annotation_node.out_pose_annotations, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
