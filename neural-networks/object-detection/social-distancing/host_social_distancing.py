import depthai as dai
from utils.measure_object_distance import ObjectDistances, DetectionDistance

ALERT_THRESHOLD = 0.5
STATE_QUEUE_LENGTH = 15


class SocialDistancing(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

        self._state_queue = []

        self.alert_distance = 1500  # mm

    def build(
        self, distances: dai.Node.Output, alert_distance: float = 1500
    ) -> "SocialDistancing":
        self.alert_distance = alert_distance
        self.link_args(distances)
        return self

    def process(self, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)

        close_detections = self._get_all_close_detections(distances)
        self._add_state(len(close_detections) > 0)

        annotations = self._create_annotations(distances)

        self.output.send(annotations)

    def _create_annotations(self, distances: ObjectDistances):
        annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()

        if self._should_alert:
            self._add_alert_annotation(annotation)

        for distance in distances.distances:
            self._add_distance_annotation(annotation, distance)

        annotations.annotations.append(annotation)
        annotations.setTimestamp(distances.getTimestamp())
        return annotations

    def _add_alert_annotation(self, annotation: dai.ImgAnnotation):
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

    def _add_distance_annotation(
        self, annotation: dai.ImgAnnotation, distance: DetectionDistance
    ):
        det1 = distance.detection1
        det2 = distance.detection2
        pointsAnnotation = dai.PointsAnnotation()
        pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
        x_start = (det1.xmin + det1.xmax) / 2
        y_start = (det1.ymin + det1.ymax) / 2
        x_end = (det2.xmin + det2.xmax) / 2
        y_end = (det2.ymin + det2.ymax) / 2
        points = [
            dai.Point2f(x_start, y_start, normalized=True),
            dai.Point2f(x_end, y_end, normalized=True),
        ]
        pointsAnnotation.points = dai.VectorPoint2f(points)
        pointsAnnotation.outlineColor = dai.Color(0, 0, 1, 1)
        pointsAnnotation.thickness = 2
        annotation.points.append(pointsAnnotation)

        text = f"{round(distance.distance / 1000, 1)} m"
        label_x = (x_start + x_end) / 2
        label_y = (y_start + y_end) / 2 - 0.02
        textAnnotation = dai.TextAnnotation()
        textAnnotation.position = dai.Point2f(label_x, label_y, normalized=True)
        textAnnotation.text = text
        textAnnotation.fontSize = 24
        textAnnotation.textColor = dai.Color(0, 0, 1, 1)
        annotation.texts.append(textAnnotation)

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

    def _get_all_close_detections(
        self, distances: ObjectDistances
    ) -> list[dai.SpatialImgDetection]:
        close_detections: list[dai.SpatialImgDetection] = []
        close_bboxes: list[tuple[float, float, float, float]] = []
        for distance in distances.distances:
            if distance.distance < self.alert_distance:
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
