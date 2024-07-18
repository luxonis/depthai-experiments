from pathlib import Path
import depthai as dai
from preprocess import Preprocess
from yolox import YoloX

SHAPE = 416
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

with dai.Pipeline() as pipeline:
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_3)

    camera = pipeline.create(dai.node.ColorCamera)
    camera.setPreviewSize(SHAPE, SHAPE)
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    preprocess = pipeline.create(Preprocess).build(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        shape=SHAPE
    )
    preprocess.input.setBlocking(False)
    preprocess.input.setMaxSize(4)
    camera.preview.link(preprocess.input)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(Path("yolox_tiny.blob").resolve().absolute()))
    nn.setNumInferenceThreads(2)
    preprocess.output.link(nn.input)

    yolox = pipeline.create(YoloX).build(camera.preview, nn.out, SHAPE, labelMap)
    pipeline.run()
