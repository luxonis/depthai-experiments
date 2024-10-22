import cv2
import depthai as dai
import numpy as np
from host_node.measure_object_distance import ObjectDistances

DISTANCE_THRESHOLD = 500  # mm
ALERT_THRESHOLD = 0.5
STATE_QUEUE_LENGTH = 10


class ShowAlert(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

        self._state_queue = []

    def build(
        self,
        frame: dai.ImgFrame,
        distances: dai.Node.Output,
        palm_label: int,
        dangerous_objects: list[int],
    ) -> "ShowAlert":
        self.link_args(frame, distances)
        self.palm_label = palm_label
        self.dangerous_objects = dangerous_objects
        return self

    def process(self, frame: dai.ImgFrame, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        img = frame.getCvFrame()
        found_close_dets = False
        for distance in distances.distances:
            if (
                (
                    distance.detection1.label == self.palm_label
                    and distance.detection2.label in self.dangerous_objects
                )
                or (
                    distance.detection2.label == self.palm_label
                    and distance.detection1.label in self.dangerous_objects
                )
            ) and distance.distance < DISTANCE_THRESHOLD:
                found_close_dets = True
                break
        self._state_queue.append(found_close_dets)
        if len(self._state_queue) > STATE_QUEUE_LENGTH:
            self._state_queue.pop(0)
        if self._should_alert:
            img = self._draw_alert(img)

        new_frame = dai.ImgFrame()
        new_frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        new_frame.setTimestamp(frame.getTimestamp())
        new_frame.setSequenceNum(frame.getSequenceNum())
        self.output.send(new_frame)

    @property
    def _should_alert(self) -> bool:
        return sum(self._state_queue) / len(self._state_queue) > ALERT_THRESHOLD

    def _draw_alert(self, img: np.ndarray) -> np.ndarray:
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
