from pathlib import Path

import depthai as dai
from deepsort_tracking import DeepsortTracking
from depthai_nodes import ParsingNeuralNetwork
from detections_recognitions_sync import DetectionsRecognitionsSync
from utils.arguments import initialize_argparser

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


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    detection_model_description = dai.NNModelDescription(
        "luxonis/yolov6-nano:r2-coco-512x288", platform=device.getPlatform().name
    )
    detection_model_archive = dai.NNArchive(
        dai.getModelFromZoo(detection_model_description)
    )

    recognition_model_description = dai.NNModelDescription(
        "luxonis/osnet:imagenet-128x256", platform=device.getPlatform().name
    )
    recognition_model_archive = dai.NNArchive(
        dai.getModelFromZoo(recognition_model_description)
    )

    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
        if args.fps_limit:
            replay.setFps(args.fps_limit)
            args.fps_limit = None  # only want to set it once
        imageManip = pipeline.create(dai.node.ImageManipV2)
        imageManip.setMaxOutputFrameSize(
            detection_model_archive.getInputWidth()
            * detection_model_archive.getInputHeight()
            * 3
        )
        imageManip.initialConfig.setOutputSize(
            detection_model_archive.getInputWidth(),
            detection_model_archive.getInputHeight(),
        )
        imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        if device.getPlatform().name == "RVC4":
            imageManip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888i)
        replay.out.link(imageManip.inputImage)
        cam_out = imageManip.out
    else:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput(
            size=(
                detection_model_archive.getInputWidth(),
                detection_model_archive.getInputHeight(),
            ),
            type=dai.ImgFrame.Type.BGR888p,
            fps=args.fps_limit,
        )

    detection_nn = pipeline.create(ParsingNeuralNetwork).build(
        cam_out, detection_model_archive
    )

    script = pipeline.create(dai.node.Script)
    detection_nn.out.link(script.inputs["detections"])
    cam_out.link(script.inputs["preview"])
    script.setScriptPath(Path(__file__).parent / "script.py")

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(256, 128)
    recognition_manip.inputConfig.setWaitForMessage(True)
    script.outputs["manip_cfg"].link(recognition_manip.inputConfig)
    script.outputs["manip_img"].link(recognition_manip.inputImage)

    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setNNArchive(recognition_model_archive)
    recognition_manip.out.link(recognition_nn.input)
    recognition_nn.input.setBlocking(False)
    recognition_nn.input.setMaxSize(2)

    detection_recognitions_sync = pipeline.create(DetectionsRecognitionsSync).build()
    detection_nn.out.link(detection_recognitions_sync.input_detections)
    recognition_nn.out.link(detection_recognitions_sync.input_recognitions)

    deepsort_tracking = pipeline.create(DeepsortTracking).build(
        cam_out, detection_recognitions_sync.output, LABELS
    )

    pipeline.run()
