import numpy as np
import cv2
import depthai as dai

LABEL_MAP = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",          "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "street sign", "stop sign",    "parking meter",
    "bench",          "bird",       "cat",           "dog",           "horse",       "sheep",        "cow",
    "elephant",       "bear",       "zebra",         "giraffe",       "hat",         "backpack",     "umbrella",
    "shoe",           "eye glasses","handbag",       "tie",           "suitcase",    "frisbee",      "skis",
    "snowboard",      "sports ball","kite",          "baseball bat",  "baseball glove","skateboard", "surfboard",
    "tennis racket",  "bottle",     "plate",         "wine glass",    "cup",         "fork",         "knife",
    "spoon",          "bowl",       "banana",        "apple",         "sandwich",    "orange",       "broccoli",
    "carrot",         "hot dog",    "pizza",         "donut",         "cake",        "chair",        "sofa",
    "pottedplant",    "bed",        "mirror",        "diningtable",   "window",      "desk",         "toilet",
    "door",           "tvmonitor",  "laptop",        "mouse",         "remote",      "keyboard",     "cell phone",
    "microwave",      "oven",       "toaster",       "sink",          "refrigerator","blender",      "book",
    "clock",          "vase",       "scissors",      "teddy bear",    "hair drier",  "toothbrush"
]
COLORS = np.random.random(size=(256, 3)) * 256


class MaskRCNN(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, preview: dai.Node.Output, nn: dai.Node.Output
              , threshold: float, region_threshold: float) -> "MaskRCNN":
        self.link_args(preview, nn)
        self.sendProcessingToPipeline(True)

        self._threshold = threshold
        self._region_threshold = region_threshold
        return self

    def process(self, preview: dai.ImgFrame, nn: dai.NNData) -> None:
        frame = preview.getCvFrame()
        boxes = nn.getTensor("DetectionOutput_647").squeeze()
        masks = nn.getTensor("Sigmoid_733")

        self.process_boxes_and_regions(frame, boxes, masks)
        cv2.imshow("Preview", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Pipeline exited.")
            self.stopPipeline()


    def process_boxes_and_regions(self, frame, boxes, masks):
        for i, box in enumerate(boxes):
            if box[0] == -1:
                break

            cls = int(box[1])
            prob = box[2]

            if prob < self._threshold:
                continue

            bbox = frame_norm(frame, box[-4:])
            cv2.rectangle(frame, (bbox[0], bbox[1] - 15), (bbox[2], bbox[1]), COLORS[cls], -1)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[cls], 1)
            cv2.putText(frame, f"{LABEL_MAP[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5)
                        , cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(frame, f"{LABEL_MAP[cls-1]}: {prob:.2f}", (bbox[0] + 5, bbox[1] - 5)
                        , cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)

            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            mask = cv2.resize(masks[i, cls], (bbox_w, bbox_h))
            mask = mask > self._region_threshold

            roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            roi[mask] = roi[mask] * 0.6 + COLORS[cls] * 0.4
            frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi


def frame_norm(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)
