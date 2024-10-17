import depthai as dai
from depthai_nodes.ml.parsers import KeypointParser
from host_node.crop_detection import CropDetection
from host_node.detection_label_filter import DetectionLabelFilter
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.normalize_detections import NormalizeDetections
from host_node.translate_cropped_detection import TranslateCroppedDetection

device = dai.Device()


detection_model_description = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
detection_archive_path = dai.getModelFromZoo(detection_model_description)
detection_nn_archive = dai.NNArchive(detection_archive_path)

pose_model_description = dai.NNModelDescription(
    modelSlug="objectron",
    platform=device.getPlatform().name,
    modelVersionSlug="chair-224x224",
)
pose_archive_path = dai.getModelFromZoo(pose_model_description)
pose_nn_archive = dai.NNArchive(pose_archive_path)

DETECTION_NN_WIDTH, DETECTION_NN_HEIGHT = 512, 288
POSE_NN_WIDTH, POSE_NN_HEIGHT = 224, 224
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
DETECTION_BBOX_PADDING = 0.1

OBJECTRON_LINES = [
    (1, 2),
    (2, 4),
    (1, 3),
    (4, 3),
    (2, 6),
    (1, 5),
    (3, 7),
    (4, 8),
    (6, 8),
    (7, 8),
    (7, 5),
    (5, 6),
]

with dai.Pipeline(device) as pipeline:
    # Create a camera
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A
    )
    video = cam.requestOutput(
        size=(VIDEO_WIDTH, VIDEO_HEIGHT), type=dai.ImgFrame.Type.BGR888p, fps=10
    )

    # Create a Manip that resizes image before detection
    manip_det = pipeline.create(dai.node.ImageManip)
    manip_det.initialConfig.setResize(DETECTION_NN_WIDTH, DETECTION_NN_HEIGHT)
    manip_det.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    video.link(manip_det.inputImage)

    nn_detection = pipeline.create(dai.node.DetectionNetwork)
    nn_detection.build(manip_det.out, detection_nn_archive)
    nn_detection.setConfidenceThreshold(0.5)

    detection_filter = pipeline.create(DetectionLabelFilter).build(
        nn=nn_detection.out, accepted_labels=[nn_detection.getClasses().index("chair")]
    )  # Filter only chairs
    crop_bbox = pipeline.create(CropDetection).build(nn=detection_filter.output)
    crop_bbox.set_resize((POSE_NN_WIDTH, POSE_NN_HEIGHT))
    crop_bbox.set_bbox_padding(DETECTION_BBOX_PADDING)
    crop_bbox_manip = pipeline.create(dai.node.ImageManip)
    crop_bbox_manip.initialConfig.setResize(POSE_NN_WIDTH, POSE_NN_HEIGHT)
    crop_bbox_manip.inputConfig.setWaitForMessage(False)
    manip_det.out.link(crop_bbox_manip.inputImage)
    crop_bbox.output_config.link(crop_bbox_manip.inputConfig)

    nn_pose = pipeline.create(dai.node.NeuralNetwork).build(
        input=crop_bbox_manip.out, nnArchive=pose_nn_archive
    )
    pose_parser = pipeline.create(KeypointParser)
    pose_parser.setNumKeypoints(9)
    pose_parser.setScaleFactor(224)
    nn_pose.out.link(pose_parser.input)
    translate_cropped = pipeline.create(TranslateCroppedDetection).build(
        detection_nn=crop_bbox.detection_passthrough, cropped_nn=pose_parser.out
    )
    translate_cropped.set_bbox_padding(DETECTION_BBOX_PADDING)
    normalize_pose = pipeline.create(NormalizeDetections).build(
        frame=video, nn=translate_cropped.output
    )
    draw_pose = pipeline.create(DrawDetections).build(
        frame=video,
        nn=normalize_pose.output,
        label_map=["chair"],
        lines=OBJECTRON_LINES,
    )
    display_pose = pipeline.create(Display).build(frames=draw_pose.output)
    display_pose.setName("Objectron")

    print("pipeline created")
    pipeline.run()
    print("pipeline finished")
