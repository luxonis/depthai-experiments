from datetime import timedelta

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
)


class DrawDetections(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.inputDet = self.createInput()
        self.output = self.createOutput()

    def setLabelMap(self, labelMap):
        self.labelMap = labelMap
        
    def build(
        self, nn: dai.Node.Output
        ) -> "DrawDetections":
        
        self.link_args(nn)
        return self
        
    def process(self, detections: dai.Buffer) -> None:
        imgAnnt = dai.ImgAnnotations()
        imgAnnt.setTimestamp(detections.getTimestamp())
        annotation = dai.ImgAnnotation()
        for d in detections:
            
            center = dai.Point2f(d.x_center, d.y_center)
            size = dai.Size2f(d.width, d.height)
            rotated_rect = dai.RotatedRect(center, size, d.angle)
            points = rotated_rect.points
            print("points", points)
            print("points.type", points.type)
            
            pointsAnnotation = dai.PointsAnnotation()
            pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
            pointsAnnotation.points = dai.VectorPoint2f([
                dai.Point2f(detection.xmin, detection.ymin),
                dai.Point2f(detection.xmax, detection.ymin),
                dai.Point2f(detection.xmax, detection.ymax),
                dai.Point2f(detection.xmin, detection.ymax),
            ])
            outlineColor = dai.Color(1.0, 0.5, 0.5, 1.0)
            pointsAnnotation.outlineColor = outlineColor
            fillColor = dai.Color(0.5, 1.0, 0.5, 0.5)
            pointsAnnotation.fillColor = fillColor
            pointsAnnotation.thickness = 2.0
            text = dai.TextAnnotation()
            text.position = dai.Point2f(detection.xmin, detection.ymin)
            text.text = f"{self.labelMap[detection.label]} {int(detection.confidence * 100)}%"
            text.fontSize = 50.5
            textColor = dai.Color(0.5, 0.5, 1.0, 1.0)
            text.textColor = textColor
            backgroundColor = dai.Color(1.0, 1.0, 0.5, 1.0)
            text.backgroundColor = backgroundColor
            annotation.points.append(pointsAnnotation)
            annotation.texts.append(text)

        imgAnnt.annotations.append(annotation)
        self.output.send(imgAnnt)
