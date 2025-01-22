import depthai as dai
from depthai_nodes.ml.helpers.constants import (
    OUTLINE_COLOR,
    TEXT_COLOR,
    BACKGROUND_COLOR,
)
import cv2


class AnnotationNode(dai.node.ThreadedHostNode):
    """Transforms ImgDetectionsExtended received from parsers to dai.ImgDetections"""

    def __init__(self) -> None:
        super().__init__()

        self.detections_input = self.createInput()
        self.depth_input = self.createInput()
        self.out_detections = self.createOutput()
        self.out_depth = self.createOutput()

    def run(self):
        while self.isRunning():
            try:
                detections_msg: dai.SpatialImgDetections = self.detections_input.get()
                depth_msg: dai.ImgFrame = self.depth_input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

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

            depth_map = depth_msg.getFrame()
            colorred_depth_map = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_map, alpha=0.3), cv2.COLORMAP_JET
            )

            depth_frame = dai.ImgFrame()
            depth_frame.setCvFrame(colorred_depth_map, dai.ImgFrame.Type.BGR888i)
            depth_frame.setTimestamp(depth_msg.getTimestamp())
            depth_frame.setSequenceNum(depth_msg.getSequenceNum())

            img_annotations.annotations.append(annotation)
            img_annotations.setTimestamp(detections_msg.getTimestamp())
            # img_annotations.setSequenceNum(detections_msg.getSequenceNum())

            self.out_detections.send(img_annotations)
            self.out_depth.send(depth_frame)
