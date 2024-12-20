import cv2
import depthai as dai
import numpy as np
from host_node.measure_object_distance import DetectionDistance, ObjectDistances


class DrawObjectDistances(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.color = (255, 0, 0)

        self._state_queue = []

    def build(
        self, frame: dai.ImgFrame, distances: dai.Node.Output
    ) -> "DrawObjectDistances":
        self.link_args(frame, distances)
        return self

    def process(self, frame: dai.ImgFrame, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        img = frame.getCvFrame()

        img = self._draw_overlay(img, distances)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame.getTimestamp())
        img_frame.setSequenceNum(frame.getSequenceNum())
        self.output.send(img_frame)

    def _draw_overlay(self, img: np.ndarray, distances: ObjectDistances):
        for distance in distances.distances:
            img = self._draw_distance_line(img, distance)
        return img

    def _draw_distance_line(self, img: np.ndarray, distance: DetectionDistance):
        det1 = distance.detection1
        det2 = distance.detection2
        text = str(round(distance.distance / 1000, 1))
        x_start = (det1.xmin + det1.xmax) / 2
        y_start = (det1.ymin + det1.ymax) / 2
        x_end = (det2.xmin + det2.xmax) / 2
        y_end = (det2.ymin + det2.ymax) / 2
        start = self._get_abs_coordinates((x_start, y_start), img.shape)
        end = self._get_abs_coordinates((x_end, y_end), img.shape)
        img = cv2.line(img, start, end, self.color, 2)

        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = (start[0] + end[0] - label_size[0]) // 2
        label_y = (start[1] + end[1] - label_size[1] - 10) // 2
        cv2.putText(
            img, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2
        )
        return img

    def set_color(self, color: tuple[int, int, int]):
        self.color = color

    def _get_abs_coordinates(
        self, point: tuple[float, float], img_size: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            int(np.clip(point[0], 0, 1) * img_size[1]),
            int(np.clip(point[1], 0, 1) * img_size[0]),
        )
