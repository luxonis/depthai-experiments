import depthai as dai
from utils.measure_object_distance import ObjectDistances
from datetime import timedelta
from typing import List

DISTANCE_THRESHOLD = 500  # mm
ALERT_THRESHOLD = 0.3
STATE_QUEUE_LENGTH = 5


class ShowAlert(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

        self._state_queue = []

    def build(
        self,
        distances: dai.Node.Output,
        palm_label: int,
        dangerous_objects: List[int],
    ) -> "ShowAlert":
        self.link_args(distances)
        self.palm_label = palm_label
        self.dangerous_objects = dangerous_objects
        return self

    def process(self, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
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
            img_annotations = self._draw_alert(distances.getTimestamp())
            self.output.send(img_annotations)

    @property
    def _should_alert(self) -> bool:
        return sum(self._state_queue) / len(self._state_queue) > ALERT_THRESHOLD

    def _draw_alert(self, timestamp: timedelta) -> dai.ImgAnnotations:
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()
        pointsAnnotation = dai.PointsAnnotation()
        pointsAnnotation.type = dai.PointsAnnotationType.LINE_LOOP
        pointsAnnotation.outlineColor = dai.Color(1, 0, 0, 1)
        pointsAnnotation.thickness = 10
        pointsAnnotation.points = dai.VectorPoint2f(
            [
                dai.Point2f(0, 0, normalized=True),
                dai.Point2f(1, 0, normalized=True),
                dai.Point2f(1, 1, normalized=True),
                dai.Point2f(0, 1, normalized=True),
            ]
        )
        annotation.points.append(pointsAnnotation)

        text = dai.TextAnnotation()
        text.position = dai.Point2f(0.4, 0.5, True)
        text.text = "Too close!"
        text.fontSize = 64
        text.textColor = dai.Color(1, 1, 1, 1)
        text.backgroundColor = dai.Color(1, 0, 0, 1)
        annotation.texts.append(text)

        img_annotations.annotations.append(annotation)
        img_annotations.setTimestamp(timestamp)

        return img_annotations
