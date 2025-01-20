import depthai as dai
from .measure_object_distance import ObjectDistances


class VisualizeObjectDistances(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(self, distances: dai.Node.Output) -> "VisualizeObjectDistances":
        self.link_args(distances)
        return self

    def process(self, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        annotations = self._draw_overlay(distances)

        self.output.send(annotations)

    def _draw_overlay(self, distances: ObjectDistances):
        img_annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()
        for distance in distances.distances:
            det1 = distance.detection1
            det2 = distance.detection2
            if not (det1.label == 80 and det2.label in [39, 41]) and not (
                det2.label == 80 and det1.label in [39, 41]
            ):
                continue

            x_start = (det1.xmin + det1.xmax) / 2
            y_start = (det1.ymin + det1.ymax) / 2
            x_end = (det2.xmin + det2.xmax) / 2
            y_end = (det2.ymin + det2.ymax) / 2

            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
            pointsAnnotation.outlineColor = dai.Color(0, 0, 1, 1)
            pointsAnnotation.thickness = 2
            pointsAnnotation.points = dai.VectorPoint2f(
                [
                    dai.Point2f(x_start, y_start, normalized=True),
                    dai.Point2f(x_end, y_end, normalized=True),
                ]
            )
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

            img_annotations.annotations.append(annotation)

        img_annotations.setTimestamp(distances.getTimestamp())

        return img_annotations
