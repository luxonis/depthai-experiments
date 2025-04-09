from pathlib import Path
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, HRNetParser
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from utils.script import generate_script_content
from depthai_nodes.node import TwoStageSync


_, args = initialize_argparser()

detection_model_slug: str = "luxonis/yolov6-nano:r2-coco-512x288"
pose_model_slug: str = args.model

padding = 0.1
valid_labels = [0]
confidence_threshold = 0.5

if args.fps_limit and args.media_path:
    args.fps_limit = None
    print(
        "WARNING: FPS limit is set but media path is provided. FPS limit will be ignored."
    )

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(detection_model_slug)
    platform = device.getPlatform().name
    print(f"Platform: {platform}")
    detection_model_description.platform = platform
    detection_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description)
    )
    classes = detection_nn_archive.getConfig().model.heads[0].metadata.classes

    pose_model_description = dai.NNModelDescription(pose_model_slug)
    pose_model_description.platform = platform
    pose_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(pose_model_description, useCached=False)
    )
    connection_pairs = (
        pose_nn_archive.getConfig()
        .model.heads[0]
        .metadata.extraParams["skeleton_edges"]
    )

    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(6 if platform == "RVC2" else 20)

    else:
        cam = pipeline.create(dai.node.Camera).build()
    input_node = replay if args.media_path else cam

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, detection_nn_archive, fps=args.fps_limit
    )

    script = pipeline.create(dai.node.Script)
    detection_nn.out.link(script.inputs["det_in"])
    detection_nn.passthrough.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=pose_nn_archive.getInputWidth(),
        resize_height=pose_nn_archive.getInputHeight(),
        padding=padding,
        valid_labels=valid_labels,
    )
    script.setScript(script_content)

    pose_manip = pipeline.create(dai.node.ImageManipV2)
    pose_manip.initialConfig.setOutputSize(
        pose_nn_archive.getInputWidth(), pose_nn_archive.getInputHeight()
    )
    pose_manip.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(pose_manip.inputConfig)
    script.outputs["manip_img"].link(pose_manip.inputImage)

    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_nn_archive
    )
    parser: HRNetParser = pose_nn.getParser(0)
    parser.setScoreThreshold(
        0.0
    )  # to get all keypoints so we can draw skeleton. We will filter them later.

    fps = args.fps_limit
    if args.media_path:
        fps = 6 if platform == "RVC2" else 20
    detection_recognitions_sync = pipeline.create(TwoStageSync).build(camera_fps=fps)
    detection_nn.out.link(detection_recognitions_sync.input_detections)
    pose_nn.out.link(detection_recognitions_sync.input_recognitions)

    annotation_node = pipeline.create(AnnotationNode).build(
        detected_recognitions=detection_recognitions_sync.out,
        connection_pairs=connection_pairs,
        valid_labels=valid_labels,
        padding=padding,
        confidence_threshold=confidence_threshold,
    )

    visualizer.addTopic("Video", detection_nn.passthrough, "images")
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
