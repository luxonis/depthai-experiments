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
device.setIrLaserDotProjectorIntensity(1)
yolo_description = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    modelVersionSlug="r2-coco-512x288",
)
palm_model_dimension = 128 if device.getPlatform() == dai.Platform.RVC2 else 192
palm_detection_description = dai.NNModelDescription(
    modelSlug="mediapipe-palm-detection",
    modelVersionSlug=f"{palm_model_dimension}x{palm_model_dimension}",
)

# If dangerous object is too close to the palm, warning will be displayed
DANGEROUS_OBJECTS = ["bottle", "cup"]

VIDEO_SIZE = (512, 288)
FPS = 10

visualizer = dai.RemoteConnection()

with dai.Pipeline(device) as pipeline:
    color_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_out = color_cam.requestOutput(VIDEO_SIZE, dai.ImgFrame.Type.NV12, fps=FPS)
    left_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_cam.requestOutput(VIDEO_SIZE, fps=FPS),
        right=right_cam.requestOutput(VIDEO_SIZE, fps=FPS),
        presetMode=dai.node.StereoDepth.PresetMode.HIGH_ACCURACY,
    )
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*VIDEO_SIZE)

    yolo_nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=color_cam, stereo=stereo, model=yolo_description, fps=FPS
    )
    yolo_nn.setConfidenceThreshold(0.5)
    yolo_nn.input.setBlocking(False)
    yolo_nn.input.setMaxSize(2)

    palm_detection = pipeline.create(dai.node.NeuralNetwork).build(
        input=color_cam,
        modelDesc=palm_detection_description,
        fps=FPS,
    )
    palm_detection.input.setBlocking(False)
    palm_detection.input.setMaxSize(2)
    palm_detection_parser = pipeline.create(MPPalmDetectionParser)
    palm_detection_parser.setScale(palm_model_dimension)
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
