from pathlib import Path
import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsBridge,
    ImgDetectionsFilter,
    GatherData,
)
from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
from depthai_nodes.node.utils import generate_script_content

_, args = initialize_argparser()

detection_model_slug = "luxonis/yolov6-nano:r2-coco-512x288"
pose_model_slug = "luxonis/superanimal-landmarker:256x256"

padding = 0.1
valid_labels = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23]

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
        args.fps_limit = 4 if platform == "RVC2" else 20
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        replay.setFps(args.fps_limit)
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

    detections_filter = pipeline.create(ImgDetectionsFilter).build(
        detection_nn.out, labels_to_keep=valid_labels
    )

    detections_bridge = pipeline.create(ImgDetectionsBridge).build(
        detections_filter.out
    )

    gather_data = pipeline.create(GatherData).build(camera_fps=args.fps_limit)
    detections_bridge.out.link(gather_data.input_reference)
    pose_nn.out.link(gather_data.input_data)

    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=gather_data.out,
        connection_pairs=connection_pairs,
        padding=padding,
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
