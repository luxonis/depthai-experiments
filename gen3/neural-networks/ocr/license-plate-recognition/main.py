import argparse
from pathlib import Path
import blobconverter
import depthai as dai

from detections_recognitions_sync import DetectionsRecognitionsSync
from host_license_plate_recognition import LicensePlateRecognition

parser = argparse.ArgumentParser()
parser.add_argument('-vid', '--video', type=str
                    , help="Path to video file to be used for inference (otherwises uses the DepthAI RGB Cam Input Feed)")
args = parser.parse_args()

FPS = 10

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    if args.video:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.video).resolve().absolute())
        replay.setSize(672, 384)
        replay.setOutFrameType(dai.ImgFrame.Type.BGR888p)
        replay.setFps(FPS)

        preview = replay.out
        shaves = 7

    else:
        cam = pipeline.create(dai.node.ColorCamera).build()
        cam.setPreviewSize(672, 384)
        cam.setInterleaved(False)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(FPS)

        preview = cam.preview
        shaves = 6

    to_nn_manip = pipeline.create(dai.node.ImageManip)
    to_nn_manip.initialConfig.setResize(300, 300)
    to_nn_manip.initialConfig.setKeepAspectRatio(False)
    to_nn_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    preview.link(to_nn_manip.inputImage)

    plate_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    plate_detection_nn.setConfidenceThreshold(0.5)
    plate_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-license-plate-detection-barrier-0106"
                                                          , shaves=shaves, version="2021.4"))
    plate_detection_nn.input.setBlocking(False)
    to_nn_manip.out.link(plate_detection_nn.input)

    car_detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork).build()
    car_detection_nn.setConfidenceThreshold(0.5)
    car_detection_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-detection-adas-0002"
                                                        , shaves=shaves, version="2021.4"))
    car_detection_nn.input.setBlocking(False)
    preview.link(car_detection_nn.input)

    script_plate = pipeline.create(dai.node.Script)
    preview.link(script_plate.inputs["preview"])
    plate_detection_nn.out.link(script_plate.inputs["detections"])
    script_plate.setScript("""
while True:
    frame = node.io["preview"].get()
    detections = node.io["detections"].get().detections
    license_detections = [detection for detection in detections if detection.label == 2]
    
    for idx, detection in enumerate(license_detections):
        cfg = ImageManipConfig()
        cfg.setKeepAspectRatio(False)
        cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        cfg.setFrameType(ImgFrame.Type.BGR888p)
        cfg.setResize(94, 24)
        
        # Send outputs to neural network
        if idx == 0:
            node.io["passthrough"].send(frame)
            cfg.setReusePreviousImage(False)
        else:
            cfg.setReusePreviousImage(True)
        node.io["config"].send(cfg)
    """)

    script_car = pipeline.create(dai.node.Script)
    preview.link(script_car.inputs["preview"])
    car_detection_nn.out.link(script_car.inputs["detections"])
    script_car.setScript("""
while True:
    frame = node.io["preview"].get()
    detections = node.io["detections"].get().detections

    for idx, detection in enumerate(detections):
        cfg = ImageManipConfig()
        cfg.setKeepAspectRatio(False)
        cfg.setCropRect(detection.xmin, detection.ymin, detection.xmax, detection.ymax)
        cfg.setFrameType(ImgFrame.Type.BGR888p)
        cfg.setResize(72, 72)

        # Send outputs to neural network
        if idx == 0:
            node.io["passthrough"].send(frame)
            cfg.setReusePreviousImage(False)
        else:
            cfg.setReusePreviousImage(True)
        node.io["config"].send(cfg)
    """)

    manip_plate = pipeline.create(dai.node.ImageManip)
    manip_plate.inputConfig.setWaitForMessage(True)
    manip_plate.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_plate.initialConfig.setResize(94, 24)
    script_plate.outputs["passthrough"].link(manip_plate.inputImage)
    script_plate.outputs["config"].link(manip_plate.inputConfig)

    manip_car = pipeline.create(dai.node.ImageManip)
    manip_car.inputConfig.setWaitForMessage(True)
    manip_car.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip_car.initialConfig.setResize(72, 72)
    script_car.outputs["passthrough"].link(manip_car.inputImage)
    script_car.outputs["config"].link(manip_car.inputConfig)

    plate_recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    plate_recognition_nn.setBlobPath(blobconverter.from_zoo(name="license-plate-recognition-barrier-0007"
                                                            , shaves=shaves, version="2021.4"))
    manip_plate.out.link(plate_recognition_nn.input)

    car_attribute_nn = pipeline.create(dai.node.NeuralNetwork)
    car_attribute_nn.setBlobPath(blobconverter.from_zoo(name="vehicle-attributes-recognition-barrier-0039"
                                                        , shaves=shaves, version="2021.4"))
    manip_car.out.link(car_attribute_nn.input)

    plate_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    plate_detection_nn.out.link(plate_manip_sync.input_detections)
    manip_plate.out.link(plate_manip_sync.input_recognitions)

    car_manip_sync = pipeline.create(DetectionsRecognitionsSync).build()
    car_detection_nn.out.link(car_manip_sync.input_detections)
    manip_car.out.link(car_manip_sync.input_recognitions)

    plate_recognition_sync = pipeline.create(DetectionsRecognitionsSync).build()
    plate_detection_nn.out.link(plate_recognition_sync.input_detections)
    plate_recognition_nn.out.link(plate_recognition_sync.input_recognitions)

    car_attribute_sync = pipeline.create(DetectionsRecognitionsSync).build()
    car_detection_nn.out.link(car_attribute_sync.input_detections)
    car_attribute_nn.out.link(car_attribute_sync.input_recognitions)

    license_plate_recognition = pipeline.create(LicensePlateRecognition).build(
        preview=preview,
        plate_images=plate_manip_sync.output,
        car_images=car_manip_sync.output,
        plate_recognitions=plate_recognition_sync.output,
        car_attributes=car_attribute_sync.output
    )
    license_plate_recognition.inputs["preview"].setBlocking(False)
    license_plate_recognition.inputs["preview"].setMaxSize(16)
    license_plate_recognition.inputs["plate_images"].setBlocking(False)
    license_plate_recognition.inputs["car_images"].setBlocking(False)
    license_plate_recognition.inputs["plate_recognitions"].setBlocking(False)
    license_plate_recognition.inputs["car_attributes"].setBlocking(False)

    print("Pipeline created.")
    pipeline.run()
