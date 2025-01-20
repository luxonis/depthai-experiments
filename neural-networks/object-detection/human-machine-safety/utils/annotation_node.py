import depthai as dai
from depthai_nodes.ml.helpers.constants import (
    OUTLINE_COLOR,
    TEXT_COLOR,
    BACKGROUND_COLOR,
)
from utils.measure_object_distance import ObjectDistances


class AnnotationNode(dai.node.ThreadedHostNode):
    """Transforms ImgDetectionsExtended received from parsers to dai.ImgDetections"""

    def __init__(self) -> None:
        super().__init__()

        self.detections_input = self.createInput()
        self.distances_input = self.createInput()
        self.out_detections = self.createOutput()

    def run(self):
        while self.isRunning():
            try:
                detections_msg: dai.SpatialImgDetections = (
                    self.detections_input.tryGet()
                )
                distances_msg: ObjectDistances = self.distances_input.tryGet()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

            if detections_msg is not None:
                img_annotations = dai.ImgAnnotations()
                annotation = dai.ImgAnnotation()
                for detection in detections_msg.detections:
                    detection: dai.SpatialImgDetection = detection
                    points = [
                        dai.Point2f(detection.xmin, detection.ymin, normalized=True),
                        dai.Point2f(detection.xmax, detection.ymin, normalized=True),
                        dai.Point2f(detection.xmax, detection.ymax, normalized=True),
                        dai.Point2f(detection.xmin, detection.ymax, normalized=True),
                    ]
                    pointsAnnotation = dai.PointsAnnotation()
                    pointsAnnotation.type = dai.PointsAnnotationType.LINE_LOOP
                    pointsAnnotation.points = dai.VectorPoint2f(points)
                    pointsAnnotation.outlineColor = OUTLINE_COLOR
                    pointsAnnotation.thickness = 2.0
                    annotation.points.append(pointsAnnotation)

                    text = dai.TextAnnotation()
                    text.position = points[0]
                    text.text = f"{detection.label} {int(detection.confidence * 100)}%"
                    text.fontSize = 15
                    text.textColor = TEXT_COLOR
                    text.backgroundColor = BACKGROUND_COLOR
                    annotation.texts.append(text)

                img_annotations.annotations.append(annotation)
                img_annotations.setTimestamp(detections_msg.getTimestamp())

                self.out_detections.send(img_annotations)
