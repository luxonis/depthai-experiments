import depthai as dai
import blobconverter

from pathlib import Path
from os.path import isfile
from download import download_decoder
from host_nanosam_main import NanoSAM

detection_shape = (416, 416)
nn_shape = (1024, 1024)

# Download onnx decoder
if not isfile(Path("onnx_decoder/mobile_sam_mask_decoder.onnx").resolve().absolute()):
    download_decoder()

with dai.Pipeline() as pipeline:

    print("Creating pipeline...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*nn_shape)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setInterleaved(False)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(*detection_shape)
    manip.inputConfig.setWaitForMessage(False)
    cam.preview.link(manip.inputImage)

    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo("yolov6n_coco_416x416", shaves = 6, zoo_type = "depthai", use_cache=True))
    detection_nn.setNumClasses(80)
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors([])
    detection_nn.setAnchorMasks({})
    detection_nn.setIouThreshold(0.45)
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setNumInferenceThreads(1)
    manip.out.link(detection_nn.input)

    embedding_nn = pipeline.create(dai.node.NeuralNetwork)
    embedding_nn.setBlobPath(blobconverter.from_zoo("nanosam_resnet18_image_encoder_1024x1024", shaves = 6
                                                    , zoo_type = "depthai", version="2022.1", use_cache=True))
    embedding_nn.setNumInferenceThreads(1)
    cam.preview.link(embedding_nn.input)

    nanosam = pipeline.create(NanoSAM).build(
        preview=cam.preview,
        detections=detection_nn.out,
        nn=embedding_nn.out
    )
    nanosam.inputs["preview"].setBlocking(False)
    nanosam.inputs["preview"].setMaxSize(3)
    nanosam.inputs["detections"].setBlocking(False)
    nanosam.inputs["detections"].setMaxSize(3)

    print("Pipeline created.")
    pipeline.run()
