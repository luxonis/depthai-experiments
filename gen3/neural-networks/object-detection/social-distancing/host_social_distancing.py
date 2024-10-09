import itertools

import cv2
import depthai as dai
import numpy as np

ALERT_THRESHOLD = 0.5
STATE_QUEUE_LENGTH = 30
ALERT_DISTANCE = 2000  # mm


class SocialDistancing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

        self._state_queue = []

    def build(self, frame: dai.ImgFrame, nn: dai.Node.Output) -> "SocialDistancing":
        self.link_args(frame, nn)
        return self

    def process(self, frame: dai.ImgFrame, nn: dai.SpatialImgDetections):
        img = frame.getCvFrame()
        close_detections = set()
        for det1, det2 in itertools.combinations(nn.detections, 2):
            dist = self._calc_distance(det1.spatialCoordinates, det2.spatialCoordinates)
            if dist < ALERT_DISTANCE:
                close_detections.add(det1)
                close_detections.add(det2)

        self._add_state(len(close_detections) > 0)
        if self._should_alert:
            img = self._draw_alert(img)

        img = self._draw_close_detections(img, close_detections)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame.getTimestamp())
        img_frame.setSequenceNum(frame.getSequenceNum())
        self.output.send(img_frame)

    @property
    def _should_alert(self) -> bool:
        return (
            len(self._state_queue) >= STATE_QUEUE_LENGTH
            and (sum(self._state_queue) / len(self._state_queue)) > ALERT_THRESHOLD
        )

    def _add_state(self, is_too_close: bool):
        self._state_queue.append(is_too_close)
        if len(self._state_queue) > STATE_QUEUE_LENGTH:
            self._state_queue.pop(0)

    def _draw_alert(self, img: np.ndarray):
        text = "Too close"
        text_size = 2
        text_thickness = 3
        size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness
        )

        x = img.shape[1] // 2 - size[0] // 2
        y = img.shape[0] // 2 - size[1] // 2
        img = cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 0, 255),
            text_thickness,
        )
        img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
        return img

    def _draw_close_detections(
        self, img: np.ndarray, detections: list[dai.SpatialImgDetection]
    ):
        ellipses = np.zeros_like(img)
        for detection in detections:
            bottom_left = (
                int(np.clip(detection.xmin, 0, 1) * img.shape[1]),
                int(np.clip(detection.ymax, 0, 1) * img.shape[0]),
            )
            bottom_right = (
                int(np.clip(detection.xmax, 0, 1) * img.shape[1]),
                int(np.clip(detection.ymax, 0, 1) * img.shape[0]),
            )
            center = (
                (bottom_left[0] + bottom_right[0]) // 2,
                (bottom_left[1] + bottom_right[1]) // 2,
            )

            length = int(np.linalg.norm(np.array(bottom_left) - np.array(bottom_right)))
            axes = (length // 2, length // 8)

            angle = 0
            ellipses = cv2.ellipse(
                ellipses, center, axes, angle, 0, 360, (0, 0, 255), -1
            )

        alpha = 0.5
        mask = ellipses.astype(bool)
        img[mask] = cv2.addWeighted(img, alpha, ellipses, 1 - alpha, 0)[mask]
        return img

    def _calc_distance(self, p1: dai.Point3f, p2: dai.Point3f) -> float:
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
