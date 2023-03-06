#!/usr/bin/env python3
import math
import cv2
import depthai as dai
import numpy as np
import blobconverter

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self._bboxColors = np.random.random(size=(256, 3)) * 256
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.6, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.6, self.color, 1, self.line_type)
    def rectangle(self, frame, p1, p2, id):
        cv2.rectangle(frame, p1, p2, (0,0,0), 4)
        cv2.rectangle(frame, p1, p2, self._bboxColors[id], 2)

class TitleHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self._bboxColors = np.random.random(size=(256, 3)) * 256
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.2, self.bg_color, 6, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.2, self.color, 2, self.line_type)
    def rectangle(self, frame, p1, p2, id):
        cv2.rectangle(frame, p1, p2, self._bboxColors[id], 3)

LOGO = cv2.imread('logo.jpeg')
LOGO = cv2.resize(LOGO, (250, 67))

jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom = jet_custom[::-1]
jet_custom[0] = [0, 0, 0]

class HostSync:
    def __init__(self):
        self.dict = {}

    def add_msg(self, name, msg):
        seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        # print(f"Adding {name} with seq `{seq}`")
        self.dict[seq][name] = msg

    def get_msgs(self):
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == 3:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None

def create_bird_frame():
    fov = 68.3
    frame = np.zeros((300, 100, 3), np.uint8)
    cv2.rectangle(frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

    alpha = (180 - fov) / 2
    center = int(frame.shape[1] / 2)
    max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, frame.shape[0]),
        (frame.shape[1], frame.shape[0]),
        (frame.shape[1], max_p),
        (center, frame.shape[0]),
        (0, max_p),
        (0, frame.shape[0]),
    ])
    cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
    return frame
birdseyeframe = create_bird_frame()
MAX_Z = 15000

def draw_bird_frame(frame, x, z, id = None):
    global MAX_Z
    max_x = 5000 #mm
    pointY = frame.shape[0] - int(z / (MAX_Z - 10000) * frame.shape[0]) - 20
    pointX = int(-x / max_x * frame.shape[1] + frame.shape[1]/2)
    if id is not None:
        cv2.putText(frame, str(id), (pointX - 30, pointY + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
    cv2.circle(frame, (pointX, pointY), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)


# Tiny yolo v3/4 label texts
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

syncNN = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.initialControl.setManualFocus(130)
camRgb.setIspScale(2, 3) # Downscale color to match mono
camRgb.setPreviewKeepAspectRatio(False)

spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

nnPath = blobconverter.from_zoo(name="yolov4_tiny_coco_416x416", zoo_type="depthai", shaves=6)
spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.3)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(300)
spatialDetectionNetwork.setDepthUpperThreshold(35000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
spatialDetectionNetwork.setIouThreshold(0.5)

camRgb.preview.link(spatialDetectionNetwork.input)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("detections")
spatialDetectionNetwork.out.link(xoutNN.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb")
    detectionNNQueue = device.getOutputQueue(name="detections")
    depthQueue = device.getOutputQueue(name="depth")

    text = TextHelper()
    title = TitleHelper()
    sync = HostSync()
    display = None

    cv2.namedWindow("Luxonis", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Luxonis",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        if previewQueue.has():
            sync.add_msg("rgb", previewQueue.get())
        if depthQueue.has():
            sync.add_msg("depth", depthQueue.get())
        if detectionNNQueue.has():
            sync.add_msg("detections", detectionNNQueue.get())

        msgs = sync.get_msgs()
        if msgs is not None:
            birds = birdseyeframe.copy()
            detections = msgs["detections"].detections
            #print(msgs)
            #if msgs.get("rgb") is None or msgs.get("depth") is None:
            #    continue
            frame = msgs["rgb"].getCvFrame()
            depthFrame = msgs["depth"].getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, jet_custom)
            height = frame.shape[0]
            width  = frame.shape[1]

            display = frame
            display[:,640:] = depthFrameColor[:,640:]
            display = cv2.flip(display, 1)
            # If the frame is available, draw bounding boxes on it and show the frame
            for detection in detections:
                # Denormalize bounding box
                detection.xmin = 1 - detection.xmin
                detection.xmax = 1 - detection.xmax
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                text.putText(display, str(label), (x2 + 10, y1 + 20))
                text.putText(display, "{:.0f}%".format(detection.confidence*100), (x2 + 10, y1 + 40))
                text.rectangle(display, (x1, y1), (x2, y2), detection.label)
                if detection.spatialCoordinates.z != 0:
                    text.putText(display, "X: {:.2f} m".format(detection.spatialCoordinates.x/1000), (x2 + 10, y1 + 60))
                    text.putText(display, "Y: {:.2f} m".format(detection.spatialCoordinates.y/1000), (x2 + 10, y1 + 80))
                    text.putText(display, "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000), (x2 + 10, y1 + 100))

                draw_bird_frame(birds, detection.spatialCoordinates.x, detection.spatialCoordinates.z)


        if display is not None:
            title.putText(display, 'DEPTH', (30, 50))
            title.putText(display, 'RGB', (width // 2 + 30, 50))
            # add LOGO
            cv2.rectangle(display, (width // 2 - 140, height - 90), (width // 2 + 140, height), (255, 255, 255), -1)
            display[(height - 77):(height-10), (width // 2 - 125):(width // 2 + 125)] = LOGO

            # Birdseye view
            # cv2.imshow("birds", birds)
            cv2.rectangle(display, (0, 210), (100,510), (255, 255, 255), 3)
            display[210:510, 0:100] = birds
            cv2.imshow("Luxonis", cv2.resize(display, (1920, 1080)))

        if cv2.waitKey(1) == ord('q'):
            break
