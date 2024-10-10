import time

import cv2
import depthai as dai
import numpy as np

labels = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
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
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
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


class DeviceDecoding(dai.node.HostNode):
    def __init__(self) -> None:
        self._startTime = time.monotonic()
        self._color = (255, 255, 255)
        self._labels = labels
        self._counter = 0
        super().__init__()

    def build(
        self, images: dai.Node.Output, detections: dai.Node.Output
    ) -> "DeviceDecoding":
        self._labels = labels
        self.link_args(images, detections)
        self.sendProcessingToPipeline(True)
        return self

    def process(
        self, img_frame: dai.ImgFrame, img_detections: dai.ImgDetections
    ) -> None:
        frame: np.ndarray = img_frame.getCvFrame()
        cv2.putText(
            frame,
            "NN fps: {:.2f}".format(
                self._counter / (time.monotonic() - self._startTime)
            ),
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            self._color,
        )

        detections = img_detections.detections
        self._counter += 1

        self._displayFrame("Device decoding", frame, detections)

        if cv2.waitKey(1) == ord("q"):
            self.stopPipeline()

    def _displayFrame(
        self, name: str, frame: np.ndarray, detections: list[dai.ImgDetection]
    ):
        color = (255, 0, 0)
        for detection in detections:
            bbox = self._frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            cv2.putText(
                frame,
                self._labels[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.imshow(name, frame)

    def _frameNorm(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
