import cv2
import depthai as dai
import numpy as np

from utility import TextHelper, TitleHelper

JET_CUSTOM = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
JET_CUSTOM = JET_CUSTOM[::-1]
JET_CUSTOM[0] = [0, 0, 0]

LOGO = cv2.imread('logo.jpeg')
LOGO = cv2.resize(LOGO, (250, 67))

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
        

class CombineOutputs(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])

        self.text = TextHelper()
        self.title = TitleHelper()

    def build(self, color: dai.Node.Output, depth: dai.Node.Output, birdseye: dai.Node.Output, detections: dai.Node.Output) -> "CombineOutputs":
        self.link_args(color, depth, birdseye, detections)
        return self

    def process(self, color_frame: dai.ImgFrame, depth_frame: dai.Buffer, bird_frame: dai.ImgFrame, detections: dai.ImgDetections) -> None:
        assert(isinstance(depth_frame, dai.ImgFrame))

        combined_frame = color_frame.getCvFrame()
        depth_frame = depth_frame.getCvFrame()
        bird_frame = bird_frame.getCvFrame()
        detections = detections.detections

        depth_frame = cv2.normalize(depth_frame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
        depth_frame = cv2.equalizeHist(depth_frame)
        depth_frame = cv2.applyColorMap(depth_frame, JET_CUSTOM)
        combined_frame[:, 640:] = depth_frame[:, 640:]

        height = combined_frame.shape[0]
        width = combined_frame.shape[1]
        combined_frame = cv2.flip(combined_frame, 1)

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
            except KeyError:
                label = detection.label

            self.text.putText(combined_frame, str(label), (x2 + 10, y1 + 20))
            self.text.putText(combined_frame, "{:.0f}%".format(detection.confidence*100), (x2 + 10, y1 + 40))
            self.text.rectangle(combined_frame, (x1, y1), (x2, y2), detection.label)
            if detection.spatialCoordinates.z != 0:
                self.text.putText(combined_frame, "X: {:.2f} m".format(detection.spatialCoordinates.x/1000), (x2 + 10, y1 + 60))
                self.text.putText(combined_frame, "Y: {:.2f} m".format(detection.spatialCoordinates.y/1000), (x2 + 10, y1 + 80))
                self.text.putText(combined_frame, "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000), (x2 + 10, y1 + 100))

        if combined_frame is not None:
            self.title.putText(combined_frame, 'DEPTH', (30, 50))
            self.title.putText(combined_frame, 'RGB', (width // 2 + 30, 50))
            # Add Luxonis logo
            cv2.rectangle(combined_frame, (width // 2 - 140, height - 90), (width // 2 + 140, height - 10), (255, 255, 255), -1)
            combined_frame[(height - 82):(height - 15), (width // 2 - 125):(width // 2 + 125)] = LOGO

            # Birdseye view
            cv2.rectangle(combined_frame, (10, 210), (110,510), (255, 255, 255), 3)
            combined_frame[210:510, 10:110] = bird_frame

            output_frame = dai.ImgFrame()
            output_frame.setCvFrame(cv2.resize(combined_frame, (1920, 1080)), dai.ImgFrame.Type.BGR888p)
            self.output.send(output_frame)
