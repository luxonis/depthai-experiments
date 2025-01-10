from pathlib import Path

# import blobconverter
import depthai as dai

from deepsort_tracking import DeepsortTracking
from detections_recognitions_sync import DetectionsRecognitionsSync

LABELS = [
    "Person",
    "Bicycle",
    "Car",
    "Motorbike",
    "Aeroplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic Light",
    "Fire Hydrant",
    "Stop Sign",
    "Parking Meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Giraffe",
    "Backpack",
    "Umbrella",
    "Handbag",
    "Tie",
    "Suitcase",
    "Frisbee",
    "Skis",
    "Snowboard",
    "Sports Ball",
    "Kite",
    "Baseball Bat",
    "Baseball Glove",
    "Skateboard",
    "Surfboard",
    "Tennis Racket",
    "Bottle",
    "Wine Glass",
    "Cup",
    "Fork",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Sandwich",
    "Orange",
    "Broccoli",
    "Carrot",
    "Hot Dog",
    "Pizza",
    "Donut",
    "Cake",
    "Chair",
    "Sofa",
    "Pottedplant",
    "Bed",
    "Diningtable",
    "Toilet",
    "Tvmonitor",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell Phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Refrigerator",
    "Book",
    "Clock",
    "Vase",
    "Scissors",
    "Teddy Bear",
    "Hair Drier",
    "Toothbrush",
]

device = dai.Device()
detection_model_description = dai.NNModelDescription(
    modelSlug="yolov6-nano",
    platform=device.getPlatform().name,
    modelVersionSlug="r2-coco-512x288",
)
detection_archive_path = dai.getModelFromZoo(detection_model_description)

recognition_model_description = dai.NNModelDescription(
    modelSlug="mobilenetv2-imagenet-embedder",
    platform="RVC2",
    modelVersionSlug="224x224",
)
recognition_archive_path = dai.getModelFromZoo(recognition_model_description)
recognition_nn_archive = dai.NNArchive(recognition_archive_path)

with dai.Pipeline(device) as pipeline:
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(512, 288)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(15)
    cam.setInterleaved(False)

    detection_nn = pipeline.create(dai.node.DetectionNetwork).build(
        cam.preview, dai.NNArchive(detection_archive_path)
    )
    detection_nn.setConfidenceThreshold(0.5)

    script = pipeline.create(dai.node.Script)
    detection_nn.out.link(script.inputs["detections"])
    cam.preview.link(script.inputs["preview"])
    script.setScriptPath(Path(__file__).parent / "script.py")

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(224, 224)
    recognition_manip.inputConfig.setWaitForMessage(True)
    # recognition_manip.setMaxOutputFrameSize(256*256*3)
    script.outputs["manip_cfg"].link(recognition_manip.inputConfig)
    script.outputs["manip_img"].link(recognition_manip.inputImage)

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    # recognition_nn.setBlob(blobconverter.from_zoo("mobilenetv2_imagenet_embedder_224x224", zoo_type="depthai", shaves=6))
    recognition_nn.setNNArchive(recognition_nn_archive)
    recognition_manip.out.link(recognition_nn.input)
    recognition_nn.input.setBlocking(False)
    recognition_nn.input.setMaxSize(2)

    detection_recognitions_sync = pipeline.create(DetectionsRecognitionsSync).build()
    detection_nn.out.link(detection_recognitions_sync.input_detections)
    recognition_nn.out.link(detection_recognitions_sync.input_recognitions)

    deepsort_tracking = pipeline.create(DeepsortTracking).build(
        cam.video, detection_recognitions_sync.output, LABELS
    )

    pipeline.run()
