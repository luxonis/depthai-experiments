import depthai as dai
from depthai_nodes import KeypointParser, YuNetParser
from draw_effect import DrawEffect
from host_node.crop_detection import CropDetection
from host_node.draw_detections import DrawDetections
from host_node.host_display import Display
from host_node.normalize_detections import NormalizeDetections
from host_node.translate_cropped_detection import TranslateCroppedDetection

device = dai.Device()
platform = device.getPlatform()

YUNET_INPUT_SIZE = (640, 640)
FACE_LANDMARK_INPUT_SIZE = (192, 192)
DETECTION_BBOX_PADDING = 0.1

yunet_model = dai.getModelFromZoo(
    dai.NNModelDescription(
        modelSlug="yunet",
        modelVersionSlug=f"{YUNET_INPUT_SIZE[1]}x{YUNET_INPUT_SIZE[0]}",
        platform=platform.name,
    )
)
yunet_archive = dai.NNArchive(yunet_model)

face_landmark_model = dai.getModelFromZoo(
    dai.NNModelDescription(
        modelSlug="mediapipe-face-landmarker",
        modelVersionSlug="192x192",
        platform=platform.name,
    )
)
face_landmark_archive = dai.NNArchive(face_landmark_model)

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    preview = cam.requestOutput(YUNET_INPUT_SIZE, dai.ImgFrame.Type.BGR888p, fps=10)

    face_nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=preview, nnArchive=yunet_archive
    )

    yunet_parser = pipeline.create(YuNetParser)
    yunet_parser.setConfidenceThreshold(0.5)
    face_nn.out.link(yunet_parser.input)

    crop_detection = pipeline.create(CropDetection).build(nn=yunet_parser.out)
    crop_detection.set_bbox_padding(DETECTION_BBOX_PADDING)
    crop_detection.set_resize(FACE_LANDMARK_INPUT_SIZE)
    crop_bbox_manip = pipeline.create(dai.node.ImageManip)
    crop_bbox_manip.initialConfig.setResize(*FACE_LANDMARK_INPUT_SIZE)
    crop_bbox_manip.inputConfig.setWaitForMessage(True)
    crop_detection.output_config.link(crop_bbox_manip.inputConfig)
    preview.link(crop_bbox_manip.inputImage)

    face_landmark_nn = pipeline.create(dai.node.NeuralNetwork).build(
        input=crop_bbox_manip.out, nnArchive=face_landmark_archive
    )
    face_landmark_nn.setNumPoolFrames(4)
    face_landmark_nn.setNumInferenceThreads(2)

    face_landmark_parser = pipeline.create(KeypointParser)
    face_landmark_parser.setNumKeypoints(468)
    face_landmark_parser.setScaleFactor(192)
    face_landmark_nn.out.link(face_landmark_parser.input)

    normalize_detection = pipeline.create(NormalizeDetections).build(
        frame=preview, nn=yunet_parser.out
    )
    translate_detection = pipeline.create(TranslateCroppedDetection).build(
        detection_nn=crop_detection.detection_passthrough,
        cropped_nn=face_landmark_parser.out,
    )
    translate_detection.set_bbox_padding(DETECTION_BBOX_PADDING)
    normalize_landmarks = pipeline.create(NormalizeDetections).build(
        frame=preview, nn=translate_detection.output
    )

    draw_detection = pipeline.create(DrawDetections).build(
        frame=preview, nn=normalize_detection.output, label_map=["face"]
    )
    draw_detection.set_draw_kpts(False)
    draw_detection.set_draw_labels(False)
    draw_detection.set_draw_confidence(False)
    draw_landmarks = pipeline.create(DrawDetections).build(
        frame=draw_detection.output, nn=normalize_landmarks.output, label_map=["face"]
    )
    draw_landmarks.set_kpt_size(2)
    display = pipeline.create(Display).build(frames=draw_landmarks.output)
    display.setName("Landmarks")

    draw_effect = pipeline.create(DrawEffect).build(
        preview=preview, landmarks_nn=normalize_landmarks.output
    )
    display_effect = pipeline.create(Display).build(frames=draw_effect.output_mask)
    display_effect.setName("Mesh")

    print("Pipeline created.")
    pipeline.run()
