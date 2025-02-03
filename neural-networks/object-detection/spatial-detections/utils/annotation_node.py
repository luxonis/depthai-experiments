import depthai as dai
from depthai_nodes.ml.helpers.constants import OUTLINE_COLOR, TEXT_COLOR
from typing import List


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_detections = self.createInput()
        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.labels = []

    def build(
        self,
        input_detections: dai.Node.Output,
        labels: List[str],
    ) -> "AnnotationNode":
        self.labels = labels
        self.link_args(input_detections)
        return self

    def process(self, detections_message: dai.Buffer) -> None:
        assert isinstance(detections_message, dai.SpatialImgDetections)

        detections_list: List[dai.SpatialImgDetection] = detections_message.detections

        annotations = (
            dai.ImgAnnotations()
        )  # custom annotations for drawing bbox and displaying label + spatial info
        annotation = dai.ImgAnnotation()

        for ix, detection in enumerate(detections_list):
            xmin, ymin, xmax, ymax = (
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            )
            points = [
                dai.Point2f(x=xmin, y=ymin, normalized=True),
                dai.Point2f(x=xmax, y=ymin, normalized=True),
                dai.Point2f(x=xmax, y=ymax, normalized=True),
                dai.Point2f(x=xmin, y=ymax, normalized=True),
            ]

            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_LOOP
            pointsAnnotation.points = dai.VectorPoint2f(points)
            pointsAnnotation.outlineColor = OUTLINE_COLOR
            pointsAnnotation.thickness = 2.0
            annotation.points.append(pointsAnnotation)

            text = dai.TextAnnotation()
            text.position = dai.Point2f(x=xmin + 0.01, y=ymin + 0.2, normalized=True)
            text.text = f"{self.labels[detection.label]} {int(detection.confidence * 100)}% \nx: {detection.spatialCoordinates.x:.2f}mm \ny: {detection.spatialCoordinates.y:.2f}mm \nz:{detection.spatialCoordinates.z:.2f}mm"
            text.fontSize = 12
            text.textColor = TEXT_COLOR
            annotation.texts.append(text)

        annotations.annotations.append(annotation)
        annotations.setTimestamp(detections_message.getTimestamp())
        annotations.setSequenceNum(detections_message.getSequenceNum())

        self.out_annotations.send(annotations)
