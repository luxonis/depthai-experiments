import cv2
import depthai as dai
import numpy as np
import math
from utility import TextHelper, TitleHelper

MAX_Z = 15000

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
        

class Display(dai.node.HostNode):
    def __init__(self) -> None:
        self.text = TextHelper()
        self.title = TitleHelper()
        cv2.namedWindow("Luxonis", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Luxonis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.jet_custom = self.jet_custom[::-1]
        self.jet_custom[0] = [0, 0, 0]

        self.LOGO = cv2.imread('logo.jpeg')
        self.LOGO = cv2.resize(self.LOGO, (250, 67))

        super().__init__()


    def build(self, cam_rgb : dai.Node.Output, depth : dai.Node.Output, nn_out : dai.Node.Output) -> "Display":
        self.birdseyeframe = self.create_bird_frame()
        self.link_args(cam_rgb, depth, nn_out)
        self.sendProcessingToPipeline(True)
        return self


    def process(self, rgb_frame : dai.ImgFrame, in_depth : dai.Buffer, in_nn : dai.ImgDetections) -> None:
        assert(isinstance(in_depth, dai.ImgFrame))

        birds = self.birdseyeframe.copy()

        frame = rgb_frame.getCvFrame()
        depthFrame = in_depth.getCvFrame()
        detections = in_nn.detections

        depthFrameColor = cv2.normalize(depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
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
            self.text.putText(display, str(label), (x2 + 10, y1 + 20))
            self.text.putText(display, "{:.0f}%".format(detection.confidence*100), (x2 + 10, y1 + 40))
            self.text.rectangle(display, (x1, y1), (x2, y2), detection.label)
            if detection.spatialCoordinates.z != 0:
                self.text.putText(display, "X: {:.2f} m".format(detection.spatialCoordinates.x/1000), (x2 + 10, y1 + 60))
                self.text.putText(display, "Y: {:.2f} m".format(detection.spatialCoordinates.y/1000), (x2 + 10, y1 + 80))
                self.text.putText(display, "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000), (x2 + 10, y1 + 100))

            self.draw_bird_frame(birds, detection.spatialCoordinates.x, detection.spatialCoordinates.z)


        if display is not None:
            self.title.putText(display, 'DEPTH', (30, 50))
            self.title.putText(display, 'RGB', (width // 2 + 30, 50))
            # add self.LOGO
            cv2.rectangle(display, (width // 2 - 140, height - 90), (width // 2 + 140, height), (255, 255, 255), -1)
            display[(height - 77):(height-10), (width // 2 - 125):(width // 2 + 125)] = self.LOGO

            # Birdseye view
            # cv2.imshow("birds", birds)
            cv2.rectangle(display, (0, 210), (100,510), (255, 255, 255), 3)
            display[210:510, 0:100] = birds
            cv2.imshow("Luxonis", cv2.resize(display, (1920, 1080)))

        if cv2.waitKey(1) == ord('q'):
            self.stopPipeline()
    

    def create_bird_frame(self):
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



    def draw_bird_frame(self, frame, x, z, id = None):
        global MAX_Z
        max_x = 5000 #mm
        pointY = frame.shape[0] - int(z / (MAX_Z - 10000) * frame.shape[0]) - 20
        pointX = int(-x / max_x * frame.shape[1] + frame.shape[1]/2)
        if id is not None:
            cv2.putText(frame, str(id), (pointX - 30, pointY + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
        cv2.circle(frame, (pointX, pointY), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
