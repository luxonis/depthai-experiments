from pathlib import Path
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    HRNetParser,
    GatherData,
    ImgDetectionsFilter,
)

from depthai_nodes.node.utils import generate_script_content
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

_, args = initialize_argparser()

detection_model_slug: str = "luxonis/yolov6-nano:r2-coco-512x288"
pose_model_slug: str = args.model

padding = 0.1
fps = args.fps_limit

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(detection_model_slug)
    detection_model_description.platform = platform
    detection_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description, useCached=False)
    )

    valid_labels = [
        detection_nn_archive.getConfig().model.heads[0].metadata.classes.index("person")
    ]

    pose_model_description = dai.NNModelDescription(pose_model_slug)
    pose_model_description.platform = platform
    pose_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(pose_model_description, useCached=False)
    )

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
        replay.setSize(
            detection_nn_archive.getInputWidth(), detection_nn_archive.getInputHeight()
        )
    input_node = (
        replay.out if args.media_path else pipeline.create(dai.node.Camera).build()
    )

    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, detection_nn_archive, fps=args.fps_limit
    )

    script_node = pipeline.create(dai.node.Script)
    detection_nn.out.link(script_node.inputs["det_in"])
    detection_nn.passthrough.link(script_node.inputs["preview"])
    script_content = generate_script_content(
        resize_width=pose_nn_archive.getInputWidth(),
        resize_height=pose_nn_archive.getInputHeight(),
        padding=padding,
        valid_labels=valid_labels,
    )
    script_node.setScript(script_content)

    pose_manip = pipeline.create(dai.node.ImageManipV2)
    pose_manip.initialConfig.setOutputSize(
        pose_nn_archive.getInputWidth(), pose_nn_archive.getInputHeight()
    )
    pose_manip.inputConfig.setWaitForMessage(True)

    script_node.outputs["manip_cfg"].link(pose_manip.inputConfig)
    script_node.outputs["manip_img"].link(pose_manip.inputImage)

    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_nn_archive
    )
    parser: HRNetParser = pose_nn.getParser(0)
    parser.setScoreThreshold(
        0.0
    )  # to get all keypoints so we can draw skeleton. We will filter them later.

    detections_filter = pipeline.create(ImgDetectionsFilter).build(
        detection_nn.out, labels_to_keep=valid_labels
    )

    gather_data_node = pipeline.create(GatherData).build(fps)
    pose_nn.out.link(gather_data_node.input_data)
    detections_filter.out.link(gather_data_node.input_reference)

    skeleton_edges = (
        pose_nn_archive.getConfig()
        .model.heads[0]
        .metadata.extraParams["skeleton_edges"]
    )
    annotation_node = pipeline.create(AnnotationNode).build(
        gather_data_node.out,
        connection_pairs=skeleton_edges,
        valid_labels=valid_labels,
    )

    visualizer.addTopic("Video", detection_nn.passthrough, "images")
    visualizer.addTopic("Detections", detections_filter.out, "images")
    visualizer.addTopic("Pose", annotation_node.out_pose_annotations, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
