import cv2
import depthai as dai
import numpy as np
from utils.measure_object_distance import DetectionDistance, ObjectDistances

ALERT_THRESHOLD = 0.5
STATE_QUEUE_LENGTH = 30
ALERT_DISTANCE = 1000  # mm


class SocialDistancing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

        self._state_queue = []

    def build(
        self, frame: dai.ImgFrame, distances: dai.Node.Output
    ) -> "SocialDistancing":
        self.link_args(frame, distances)
        return self

    def process(self, frame: dai.ImgFrame, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        img = frame.getCvFrame()

        close_detections = self._get_all_close_detections(distances)
        self._add_state(len(close_detections) > 0)

        img = self._draw_overlay(img, distances, close_detections)

        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(img, dai.ImgFrame.Type.BGR888p)
        img_frame.setTimestamp(frame.getTimestamp())
        img_frame.setSequenceNum(frame.getSequenceNum())
        self.output.send(img_frame)

    def _draw_overlay(
        self,
        img: np.ndarray,
        distances: ObjectDistances,
        close_detections: list[dai.SpatialImgDetection],
    ):
        if self._should_alert:
            img = self._draw_alert(img)

        img = self._draw_close_detections(img, close_detections)
        for distance in distances.distances:
            img = self._draw_distance_line(img, distance)
        return img

    def _get_all_close_detections(
        self, distances: ObjectDistances
    ) -> list[dai.SpatialImgDetection]:
        close_detections: list[dai.SpatialImgDetection] = []
        close_bboxes: list[tuple[float, float, float, float]] = []
        for distance in distances.distances:
            if distance.distance < ALERT_DISTANCE:
                det1_bbox = (
                    distance.detection1.xmin,
                    distance.detection1.ymin,
                    distance.detection1.xmax,
                    distance.detection1.ymax,
                )
                det2_bbox = (
                    distance.detection2.xmin,
                    distance.detection2.ymin,
                    distance.detection2.xmax,
                    distance.detection2.ymax,
                )
                if det1_bbox not in close_bboxes:
                    close_detections.append(distance.detection1)
                    close_bboxes.append(det1_bbox)
                if det2_bbox not in close_bboxes:
                    close_detections.append(distance.detection2)
                    close_bboxes.append(det2_bbox)

        return close_detections

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
            bottom_left = self._get_abs_coordinates(
                (detection.xmin, detection.ymax), img.shape
            )
            bottom_right = self._get_abs_coordinates(
                (detection.xmax, detection.ymax), img.shape
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

    def _draw_distance_line(self, img: np.ndarray, distance: DetectionDistance):
        det1 = distance.detection1
        det2 = distance.detection2
        text = str(round(distance.distance / 1000, 1))
        color = (255, 0, 0)
        x_start = (det1.xmin + det1.xmax) / 2
        y_start = (det1.ymin + det1.ymax) / 2
        x_end = (det2.xmin + det2.xmax) / 2
        y_end = (det2.ymin + det2.ymax) / 2
        start = self._get_abs_coordinates((x_start, y_start), img.shape)
        end = self._get_abs_coordinates((x_end, y_end), img.shape)
        img = cv2.line(img, start, end, color, 2)

        label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = (start[0] + end[0] - label_size[0]) // 2
        label_y = (start[1] + end[1] - label_size[1] - 10) // 2
        cv2.putText(
            img, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        return img

    def _get_abs_coordinates(
        self, point: tuple[float, float], img_size: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            int(np.clip(point[0], 0, 1) * img_size[1]),
            int(np.clip(point[1], 0, 1) * img_size[0]),
        )
