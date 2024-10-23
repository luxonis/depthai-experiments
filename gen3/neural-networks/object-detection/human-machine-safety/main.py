import depthai as dai
from depthai_nodes import MPPalmDetectionParser
from host_node.depth_merger import DepthMerger
from host_node.detection_label_filter import DetectionLabelFilter
from host_node.detection_merger import DetectionMerger
from host_node.measure_object_distance import MeasureObjectDistance
from host_node.visualize_detections import VisualizeDetections
from host_node.visualize_object_distances import VisualizeObjectDistances
from show_alert import ShowAlert

device = dai.Device()

yolo_description = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
yolo_archive_path = dai.getModelFromZoo(yolo_description, useCached=True)
yolo_archive = dai.NNArchive(yolo_archive_path)

palm_detection_description = dai.NNModelDescription(
    modelSlug="mediapipe-palm-detection",
    platform=device.getPlatform().name,
    modelVersionSlug="128x128",
)
palm_detection_archive_path = dai.getModelFromZoo(
    palm_detection_description, useCached=True
)
palm_detection_archive = dai.NNArchive(palm_detection_archive_path)


# If dangerous object is too close to the palm, warning will be displayed
DANGEROUS_OBJECTS = ["bottle", "cup"]

VIDEO_SIZE = (1280, 720)
YOLO_SIZE = (512, 288)
PALM_DETECTION_SIZE = (128, 128)


visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = color_cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)
    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestFullResolutionOutput(fps=10),
        right=right_cam.requestFullResolutionOutput(fps=10),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*VIDEO_SIZE)

    yolo_manip = pipeline.create(dai.node.ImageManipV2)
    yolo_manip.initialConfig.addResize(*YOLO_SIZE)
    color_out.link(yolo_manip.inputImage)

    yolo_nn = pipeline.create(dai.node.SpatialDetectionNetwork)
    yolo_nn.setNNArchive(yolo_archive)
    yolo_nn.setConfidenceThreshold(0.5)
    yolo_nn.input.setBlocking(False)
    yolo_nn.input.setMaxSize(2)
    stereo.depth.link(yolo_nn.inputDepth)
    yolo_manip.out.link(yolo_nn.input)

    palm_detection_manip = pipeline.create(dai.node.ImageManipV2)
    palm_detection_manip.initialConfig.addResize(*PALM_DETECTION_SIZE)
    color_out.link(palm_detection_manip.inputImage)

    palm_detection = pipeline.create(dai.node.NeuralNetwork)
    palm_detection.setNNArchive(palm_detection_archive)
    palm_detection_manip.out.link(palm_detection.input)
    palm_detection.input.setBlocking(False)
    palm_detection.input.setMaxSize(2)
    palm_detection_parser = pipeline.create(MPPalmDetectionParser)
    palm_detection_parser.setScale(128)
    palm_detection_parser.setConfidenceThreshold(0.6)
    palm_detection.out.link(palm_detection_parser.input)

    palm_depth_merger = pipeline.create(DepthMerger).build(
        output_2d=palm_detection_parser.out,
        output_depth=stereo.depth,
        calib_data=device.readCalibration2(),
        depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
    )

    merge_detections = pipeline.create(DetectionMerger).build(
        yolo_nn.out, palm_depth_merger.output
    )
    merge_detections.set_detection_2_label_offset(len(yolo_nn.getClasses()))

    # Filter out everything except for dangerous objects and palm
    merged_labels = yolo_nn.getClasses() + ["palm"]
    filter_labels = [merged_labels.index(i) for i in DANGEROUS_OBJECTS]
    filter_labels.append(merged_labels.index("palm"))
    detection_filter = pipeline.create(DetectionLabelFilter).build(
        merge_detections.output, filter_labels
    )

    measure_object_distance = pipeline.create(MeasureObjectDistance).build(
        nn=detection_filter.output
    )

    visualize_detections = pipeline.create(VisualizeDetections).build(
        detection_filter.output, merged_labels
    )
    visualize_distances = pipeline.create(VisualizeObjectDistances).build(
        measure_object_distance.output
    )

    show_alert = pipeline.create(ShowAlert).build(
        distances=measure_object_distance.output,
        palm_label=merged_labels.index("palm"),
        dangerous_objects=[merged_labels.index(i) for i in DANGEROUS_OBJECTS],
    )

    print("Pipeline created.")
    visualizer.addTopic("Detections", visualize_detections.output)
    visualizer.addTopic("Distances", visualize_distances.output)
    visualizer.addTopic("Alert", show_alert.output)
    visualizer.addTopic("Color", color_out)
    pipeline.start()
    visualizer.registerPipeline(pipeline)
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        if key == ord("q"):
            break
