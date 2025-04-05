import depthai as dai
from depthai_nodes import (
    ImgDetectionsExtended,
    ImgDetectionExtended,
    OUTLINE_COLOR,
)
from typing import List
from depthai_nodes import DetectedRecognitions


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.out_detections = self.createOutput()
        self.out_pose_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.connection_pairs = [[]]
        self.padding = 0.1
        self.valid_labels = [0]
        self.confidence_threshold = 0.5

    def build(
        self,
        detected_recognitions: dai.Node.Output,
        connection_pairs: List[List[int]],
        valid_labels: List[int],
        padding: float,
        confidence_threshold: float,
    ) -> "AnnotationNode":
        self.connection_pairs = connection_pairs
        self.valid_labels = valid_labels
        self.padding = padding
        self.confidence_threshold = confidence_threshold
        self.link_args(detected_recognitions)
        return self

    def process(self, detected_recognitions: dai.Buffer) -> None:
        assert isinstance(detected_recognitions, DetectedRecognitions)

        detections_list: List[
            dai.ImgDetection
        ] = detected_recognitions.img_detections.detections
        img_detections_extended = ImgDetectionsExtended()

        annotations = (
            dai.ImgAnnotations()
        )  # custom annotations for drawing lines between keypoints

        padding = self.padding
        confidence_threshold = self.confidence_threshold

        for ix, detection in enumerate(detections_list):
            if detection.label not in self.valid_labels:  # label not chair
                continue
            img_detection_extended = ImgDetectionExtended()
            center_x = detection.xmin + (detection.xmax - detection.xmin) / 2
            center_y = detection.ymin + (detection.ymax - detection.ymin) / 2
            width = detection.xmax - detection.xmin
            height = detection.ymax - detection.ymin
            angle = 0
            img_detection_extended.rotated_rect = (
                center_x,
                center_y,
                width,
                height,
                angle,
            )
            img_detection_extended.label = detection.label
            img_detection_extended.confidence = detection.confidence

            keypoints_msg = detected_recognitions.recognitions_data[ix]

            slope_x = (detection.xmax + padding) - (detection.xmin - padding)
            slope_y = (detection.ymax + padding) - (detection.ymin - padding)
            xs = []
            ys = []
            confidences = []
            for kp in keypoints_msg.keypoints:
                x = min(max(detection.xmin - padding + slope_x * kp.x, 0.0), 1.0)
                y = min(max(detection.ymin - padding + slope_y * kp.y, 0.0), 1.0)
                xs.append(x)
                ys.append(y)
                confidences.append(kp.confidence)

            annotation = dai.ImgAnnotation()
            for connection in self.connection_pairs:
                pt1_idx, pt2_idx = connection
                if (
                    confidences[pt1_idx] < confidence_threshold
                    or confidences[pt2_idx] < confidence_threshold
                ):
                    continue
                if pt1_idx < len(xs) and pt2_idx < len(xs):
                    x1, y1 = xs[pt1_idx], ys[pt1_idx]
                    x2, y2 = xs[pt2_idx], ys[pt2_idx]
                    pointsAnnotation = dai.PointsAnnotation()
                    pointsAnnotation.type = dai.PointsAnnotationType.LINE_STRIP
                    pointsAnnotation.points = dai.VectorPoint2f(
                        [
                            dai.Point2f(x=x1, y=y1, normalized=True),
                            dai.Point2f(x=x2, y=y2, normalized=True),
                        ]
                    )
                    pointsAnnotation.outlineColor = OUTLINE_COLOR
                    pointsAnnotation.thickness = 1.0
                    annotation.points.append(pointsAnnotation)

            img_detections_extended.detections.append(img_detection_extended)
            annotations.annotations.append(annotation)

        annotations.setTimestamp(detected_recognitions.getTimestamp())
        img_detections_extended.setTimestamp(detected_recognitions.getTimestamp())
        img_detections_extended.transformation = (
            detected_recognitions.img_detections.getTransformation()
        )

        self.out_detections.send(img_detections_extended)
        self.out_pose_annotations.send(annotations)
