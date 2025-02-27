import depthai as dai

from host_display import Display
from depthai_nodes.ml.parsers import PPTextDetectionParser, PaddleOCRParser
from host_process_detections import ProcessDetections
from host_ocr import OCR
from detections_recognitions_sync import DetectionsRecognitionsSync

FPS = 10

device = dai.Device()

# RVC2 models
detection_model_description = dai.NNModelDescription(
    modelSlug="paddle-text-detection", platform="RVC2", modelVersionSlug="256x256"
)
detection_archive_path = dai.getModelFromZoo(detection_model_description)
detection_nn_archive = dai.NNArchive(detection_archive_path)

recognition_model_description = dai.NNModelDescription(
    modelSlug="paddle-text-recognition", platform="RVC2", modelVersionSlug="320x48"
)
# recognition_model_description = dai.NNModelDescription(modelSlug="paddle-text-recognition", platform="RVC2", modelVersionSlug="160x48")
recognition_archive_path = dai.getModelFromZoo(recognition_model_description)
recognition_nn_archive = dai.NNArchive(recognition_archive_path)
classes = recognition_nn_archive.getConfigV1().model.heads[0].metadata.classes


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(256, 256)
    cam.setVideoSize(1024, 1024)  # 4 times larger in both axis
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(FPS)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    # detection_nn.setBlobPath(blobconverter.from_zoo(name="east_text_detection_256x256", zoo_type="depthai", shaves=6, version="2021.4"))
    detection_nn.setNNArchive(detection_nn_archive)
    cam.preview.link(detection_nn.input)

    paddle_det = pipeline.create(PPTextDetectionParser)
    detection_nn.out.link(paddle_det.input)

    process_detections = pipeline.create(ProcessDetections).build(
        frame=cam.video, detections=paddle_det.out
    )

    color_display = pipeline.create(Display).build(process_detections.display)
    color_display.setName("Color camera")

    manip = pipeline.create(dai.node.ImageManip)
    manip.inputConfig.setWaitForMessage(True)
    process_detections.output_config.link(manip.inputConfig)
    process_detections.passthrough.link(manip.inputImage)

    # color_display = pipeline.create(Display).build(manip.out)
    # color_display.setName("Manip")

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    # recognition_nn.setBlobPath(blobconverter.from_zoo(name="text-recognition-0012", shaves=6, version="2021.4"))
    recognition_nn.setNNArchive(recognition_nn_archive)
    recognition_nn.setNumInferenceThreads(2)
    manip.out.link(recognition_nn.input)

    paddle_ocr = pipeline.create(PaddleOCRParser, classes)
    recognition_nn.out.link(paddle_ocr.input)

    recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
    recognition_sync.set_camera_fps(FPS)
    paddle_ocr.out.link(recognition_sync.input_recognitions)
    paddle_det.out.link(recognition_sync.input_detections)

    manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    manip_sync.set_camera_fps(FPS)
    manip.out.link(manip_sync.input_recognitions)
    paddle_det.out.link(manip_sync.input_detections)

    ocr = pipeline.create(OCR).build(
        preview=cam.video,
        manips=manip_sync.output,
        recognitions=recognition_sync.output,
    )
    ocr.inputs["preview"].setBlocking(False)

    print("Pipeline created.")
    print('press "c" to recognize detections')
    pipeline.run()
