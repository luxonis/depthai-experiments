import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, MPPalmDetectionParser, DepthMerger
from utils.arguments import initialize_argparser
from utils.adapter import ParserBridge
from utils.annotation_node import AnnotationNode
from utils.detection_merger import DetectionMerger
from utils.detection_label_filter import DetectionLabelFilter
from utils.measure_object_distance import MeasureObjectDistance
from utils.visualize_object_distances import VisualizeObjectDistances
from utils.show_alert import ShowAlert


_, args = initialize_argparser()

object_detection_model_slug = "luxonis/yolov6-nano:r2-coco-512x288"
palm_detection_model_slug = "luxonis/mediapipe-palm-detection:192x192"

DANGEROUS_OBJECTS = ["bottle", "cup"]
FPS_LIMIT = 30

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Check if the device has color, left and right cameras
    available_cameras = device.getConnectedCameras()

    if len(available_cameras) < 3:
        raise ValueError(
            "Device must have 3 cameras (color, left and right) in order to run this experiment."
        )

    object_detection_model_description = dai.NNModelDescription(
        object_detection_model_slug
    )
    platform = device.getPlatform().name
    print(f"Platform: {platform}")
    object_detection_model_description.platform = platform
    object_detection_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(object_detection_model_description)
    )
    classes = object_detection_nn_archive.getConfig().model.heads[0].metadata.classes

    palm_detection_model_description = dai.NNModelDescription(palm_detection_model_slug)
    palm_detection_model_description.platform = platform
    palm_detection_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(palm_detection_model_description, useCached=False)
    )

    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )
    FPS_LIMIT = 10 if platform == "RVC2" else 30

    color_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput(
            object_detection_nn_archive.getInputSize(), fps=FPS_LIMIT
        ),
        right=right_cam.requestOutput(
            object_detection_nn_archive.getInputSize(), fps=FPS_LIMIT
        ),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if platform == "RVC2":
        stereo.setOutputSize(*object_detection_nn_archive.getInputSize())
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)

    camera_output = color_camera.requestOutput(
        (800, 600), dai.ImgFrame.Type.NV12, fps=FPS_LIMIT
    )

    object_detection_manip = pipeline.create(dai.node.ImageManipV2)
    object_detection_manip.initialConfig.setOutputSize(
        512, 288, mode=dai.ImageManipConfigV2.ResizeMode.STRETCH
    )
    object_detection_manip.initialConfig.setFrameType(frame_type)
    camera_output.link(object_detection_manip.inputImage)

    object_detection_nn: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(
        object_detection_manip.out,
        object_detection_nn_archive,
    )
    if platform == "RVC2":
        object_detection_nn.setNNArchive(object_detection_nn_archive, numShaves=7)

    palm_detection_manip = pipeline.create(dai.node.ImageManipV2)
    palm_detection_manip.initialConfig.setOutputSize(
        192, 192, mode=dai.ImageManipConfigV2.ResizeMode.STRETCH
    )
    palm_detection_manip.initialConfig.setFrameType(frame_type)
    camera_output.link(palm_detection_manip.inputImage)

    palm_detection_nn: ParsingNeuralNetwork = pipeline.create(
        ParsingNeuralNetwork
    ).build(
        palm_detection_manip.out,
        palm_detection_nn_archive,
    )
    if platform == "RVC2":
        palm_detection_nn.setNNArchive(palm_detection_nn_archive, numShaves=7)

    parser: MPPalmDetectionParser = palm_detection_nn.getParser(0)
    parser.setConfidenceThreshold(0.7)

    adapter = pipeline.create(ParserBridge)
    palm_detection_nn.out.link(adapter.palm_detection_input)

    detection_depth_merger = pipeline.create(DepthMerger).build(
        output_2d=object_detection_nn.out,
        output_depth=stereo.depth,
        calib_data=device.readCalibration2(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
        shrinking_factor=0.1,
    )
    palm_depth_merger = pipeline.create(DepthMerger).build(
        output_2d=adapter.out,
        output_depth=stereo.depth,
        calib_data=device.readCalibration2(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
        shrinking_factor=0.1,
    )

    # merge both detections into one message
    merge_detections = pipeline.create(DetectionMerger).build(
        detection_depth_merger.output, palm_depth_merger.output
    )
    merge_detections.set_detection_2_label_offset(len(classes))

    # Filter out everything except for dangerous objects and palm
    merged_labels = classes + ["palm"]
    filter_labels = [merged_labels.index(i) for i in DANGEROUS_OBJECTS]
    filter_labels.append(merged_labels.index("palm"))
    detection_filter = pipeline.create(DetectionLabelFilter).build(
        merge_detections.output, filter_labels
    )

    measure_object_distance = pipeline.create(MeasureObjectDistance).build(
        nn=detection_filter.output
    )

    visualize_distances = pipeline.create(VisualizeObjectDistances).build(
        measure_object_distance.output
    )

    show_alert = pipeline.create(ShowAlert).build(
        distances=measure_object_distance.output,
        palm_label=merged_labels.index("palm"),
        dangerous_objects=[merged_labels.index(i) for i in DANGEROUS_OBJECTS],
    )

    annotation_node = pipeline.create(AnnotationNode)
    detection_filter.output.link(annotation_node.detections_input)
    stereo.depth.link(annotation_node.depth_input)

    visualizer.addTopic("Color", camera_output)
    visualizer.addTopic("Detections", annotation_node.out_detections)
    visualizer.addTopic("Distances", visualize_distances.output)
    visualizer.addTopic("Alert", show_alert.output)
    # visualizer.addTopic("Depth", annotation_node.out_depth)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key_pressed = visualizer.waitKey(1)
        if key_pressed == ord("q"):
            pipeline.stop()
            break
