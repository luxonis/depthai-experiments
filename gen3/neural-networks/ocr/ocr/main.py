import depthai as dai
import blobconverter

from host_east import East
from host_process_detections import ProcessDetections
from host_ocr import OCR
from detections_recognitions_sync import DetectionsRecognitionsSync

FPS = 10

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera).build()
    cam.setPreviewSize(256, 256)
    cam.setVideoSize(1024, 1024)  # 4 times larger in both axis
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(FPS)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo(name="east_text_detection_256x256", zoo_type="depthai", shaves=6, version="2021.4"))
    cam.preview.link(detection_nn.input)

    east = pipeline.create(East).build(
        video=cam.video,
        nn=detection_nn.out
    )
    east.inputs["video"].setBlocking(False)
    east.inputs["video"].setMaxSize(4)

    process_detections = pipeline.create(ProcessDetections).build(
        frame=cam.video,
        detections=east.output
    )

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(120, 32)
    manip.inputConfig.setWaitForMessage(True)
    process_detections.output_config.link(manip.inputConfig)
    process_detections.passthrough.link(manip.inputImage)

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="text-recognition-0012", shaves=6, version="2021.4"))
    recognition_nn.setNumInferenceThreads(2)
    manip.out.link(recognition_nn.input)

    recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
    recognition_sync.set_camera_fps(FPS)
    recognition_nn.out.link(recognition_sync.input_recognitions)
    east.output.link(recognition_sync.input_detections)

    manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    manip_sync.set_camera_fps(FPS)
    manip.out.link(manip_sync.input_recognitions)
    east.output.link(manip_sync.input_detections)

    ocr = pipeline.create(OCR).build(
        preview=east.passthrough,
        manips=manip_sync.output,
        recognitions=recognition_sync.output
    )
    ocr.inputs["preview"].setBlocking(False)

    print("Pipeline created.")
    pipeline.run()
