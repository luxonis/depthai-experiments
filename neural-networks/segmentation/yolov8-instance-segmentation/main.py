from datetime import timedelta
import depthai as dai
from depthai_nodes.ml.parsers import YOLOExtendedParser
from depthai_nodes.ml.messages import ImgDetectionsExtended
import depthai_viewer as viewer
import numpy as np
import cv2
import subprocess
import sys
import colorsys

# Run & initialize the depthai_viewer
try:
    subprocess.Popen(
        [sys.executable, "-m", "depthai_viewer", "--viewer-mode"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
except subprocess.TimeoutExpired:
    pass
viewer.init("Depthai Viewer")
viewer.connect()

FPS = 14


def get_colors(N, pastel_level=0.5):
    colors_arr = []
    for i in range(N):
        hue = i / float(N)
        # Pastel colors have low saturation and high lightness
        saturation = (1 - pastel_level) * 0.5  # Adjust saturation
        lightness = 0.5 + pastel_level * 0.5  # Adjust lightness
        # Convert HLS to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Scale RGB to 0-255 and convert to integers
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        colors_arr.append(rgb)
    # Shuffle in repetable order
    np.random.seed(0)
    np.random.shuffle(colors_arr)
    return colors_arr


classes = [
    "background",  # not coco, but required for segmentation
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
colors = get_colors(len(classes) - 1, 0.3)
colors.insert(0, (0, 0, 0))  # Background color
arr = [
    viewer.AnnotationInfo(id=i, label=classes[i], color=c) for i, c in enumerate(colors)
]
viewer.log_annotation_context("NN/Image", arr, timeless=True)

with dai.Pipeline() as pipeline:
    device: dai.Device = pipeline.getDefaultDevice()
    # print(device.getConnectedCameraFeatures())
    print("Device info: ", device.getDeviceInfo())
    device.setIrLaserDotProjectorIntensity(1.0)
    device.setIrFloodLightIntensity(0.1)

    # Set up camera
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            dai.NNModelDescription(
                # Download model from https://hub.luxonis.com/ai/models/698b881d-2e98-45d0-bc72-1121d2eb2319
                "luxonis/yolov8-instance-segmentation-large:coco-640x480",
                "RVC4",
            )
        )
    )
    detection_nn = pipeline.create(dai.node.NeuralNetwork).build(
        cam.requestOutput(
            (640, 480),
            type=dai.ImgFrame.Type.BGR888i,
            fps=FPS,
            resizeMode=dai.ImgResizeMode.STRETCH,
        ),
        nn_archive,
    )
    # Set up the parser
    parser = YOLOExtendedParser()
    parser.setNumClasses(len(classes) - 1)
    detection_nn.out.link(parser.input)

    # Stereo
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

    stereo.initialConfig.postProcessing.thresholdFilter.minRange = 400
    stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 6_000  # Max 6m
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_5x5)
    stereo.setExtendedDisparity(
        True
    )  # Extended disparity consumes a lot of resources (bottlenecks to 13FPS)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

    monoLeft.requestOutput((1280, 800), type=dai.ImgFrame.Type.NV12, fps=FPS).link(
        stereo.left
    )
    monoRight.requestOutput((1280, 800), type=dai.ImgFrame.Type.NV12, fps=FPS).link(
        stereo.right
    )

    sync = pipeline.create(dai.node.Sync)
    sync.setRunOnHost(True)
    sync.setSyncThreshold(timedelta(seconds=1 / FPS))

    rgb_800 = cam.requestOutput(
        (1280, 960),
        fps=FPS,
        type=dai.ImgFrame.Type.NV12,
        resizeMode=dai.ImgResizeMode.STRETCH,
    )

    align = pipeline.create(dai.node.ImageAlign)
    align.input.setBlocking(False)
    align.input.setMaxSize(2)
    align.inputAlignTo.setBlocking(False)
    align.setOutKeepAspectRatio(False)
    stereo.depth.link(align.input)
    rgb_800.link(align.inputAlignTo)

    align.outputAligned.link(sync.inputs["depth_aligned"])
    sync.inputs["depth_aligned"].setBlocking(False)
    rgb_800.link(sync.inputs["frame"])
    parser.out.link(sync.inputs["parser"])
    monoLeft.requestOutput((640, 400), type=dai.ImgFrame.Type.NV12, fps=FPS).link(
        sync.inputs["left"]
    )
    q = sync.out.createOutputQueue()

    pipeline.start()

    device.setIrLaserDotProjectorIntensity(1.0)
    device = pipeline.getDefaultDevice()
    calibData = device.readCalibration2()
    intrinsics = np.array(
        calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, dai.Size2f(1280, 960)
        )
    )
    rgb_distortion = np.array(
        calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
    )
    mapX, mapY = cv2.initUndistortRectifyMap(
        intrinsics, rgb_distortion, None, intrinsics, (1280, 960), cv2.CV_32FC1
    )

    viewer.log_rigid3(
        "CAM_A", child_from_parent=([0, 0, 0], [1, 0, 0, 0]), xyz="RDF", timeless=True
    )
    viewer.log_pinhole(
        "CAM_A/Image",
        child_from_parent=intrinsics,
        width=1280,
        height=960,
        timeless=True,
    )

    while pipeline.isRunning():
        msgs = q.get()
        nn: ImgDetectionsExtended = msgs["parser"]
        imgFrame_960: dai.ImgFrame = msgs["frame"]
        depthImg: dai.ImgFrame = msgs["depth_aligned"]

        frame_200 = imgFrame_960.getCvFrame()
        undistorted_rgb_200 = cv2.remap(frame_200, mapX, mapY, cv2.INTER_LINEAR)

        depthFrame = depthImg.getFrame()
        viewer.log_depth_image("CAM_A/Image/Depth", depthFrame, meter=1e3)
        viewer.log_image("CAM_A/Image/Color", undistorted_rgb_200[..., ::-1])
        viewer.log_image("CAM_B/Image", msgs["left"].getCvFrame())

        viewer.log_image("NN/Image/Color", imgFrame_960.getCvFrame()[..., ::-1])
        xyxy_boxes = []
        box_class_ids = []
        log_colors = []
        segmentation_image = np.zeros((960, 1280), dtype=np.uint8)
        for i, det in enumerate(nn.detections):
            xmin = int(1280 * (det.x_center - det.width / 2))
            ymin = int(960 * (det.y_center - det.height / 2))
            xmax = int(1280 * (det.x_center + det.width / 2))
            ymax = int(960 * (det.y_center + det.height / 2))
            xyxy_boxes.append([xmin, ymin, xmax, ymax])
            box_class_ids.append(det.label + 1)
            log_colors.append(colors[det.label + 1])

        viewer.log_rects(
            "NN/Image/Color/Detections",
            rects=np.array(xyxy_boxes),
            rect_format=viewer.RectFormat.XYXY,
            colors=log_colors,
            #  identifiers=box_class_ids
        )

        if 0 < len(nn.masks):
            mask = np.copy(nn.masks) + 1
            for i, det in enumerate(nn.detections):
                mask[nn.masks == i] = (
                    det.label + 1
                )  # Use `det.id` to identify regions and map them to `det.label`
            det_mask = cv2.resize(mask, (1280, 960), interpolation=cv2.INTER_NEAREST)
            viewer.log_segmentation_image("NN/Image/Color/Segmentation", det_mask)
