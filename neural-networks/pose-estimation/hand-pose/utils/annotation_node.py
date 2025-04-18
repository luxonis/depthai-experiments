import depthai as dai
from depthai_nodes import (
    ImgDetectionsExtended,
    ImgDetectionExtended,
    Keypoints,
    Keypoint,
    Predictions,
    PRIMARY_COLOR,
)
from typing import List
from utils.gesture_recognition import recognize_gesture


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_keypoints = self.createInput()
        self.input_confidence = self.createInput()
        self.input_handedness = self.createInput()
        self.out_detections = self.createOutput()
        self.out_pose_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.confidence_threshold = 0.5
        self.padding_factor = 0.1
        self.connection_pairs = [[]]

    def build(
        self,
        input_detections: dai.Node.Output,
        confidence_threshold: float,
        padding_factor: float,
        connections_pairs: List[List[int]],
    ) -> "AnnotationNode":
        self.confidence_threshold = confidence_threshold
        self.padding_factor = padding_factor
        self.connection_pairs = connections_pairs
        self.link_args(input_detections)
        return self

    def process(self, detections_message: dai.Buffer) -> None:
        assert isinstance(detections_message, ImgDetectionsExtended)

        detections_list: List[ImgDetectionExtended] = detections_message.detections

        new_dets = ImgDetectionsExtended()
        new_dets.transformation = detections_message.transformation

        annotations = dai.ImgAnnotations()
        annotation = dai.ImgAnnotation()

        for ix, detection in enumerate(detections_list):
            keypoints_msg: Keypoints = self.input_keypoints.get()
            confidence_msg: Predictions = self.input_confidence.get()
            handness_msg: Predictions = self.input_handedness.get()

            hand_confidence = confidence_msg.prediction
            handness = handness_msg.prediction

            if hand_confidence < self.confidence_threshold:
                continue

            width = detection.rotated_rect.size.width
            height = detection.rotated_rect.size.height

            xmin = detection.rotated_rect.center.x - width / 2
            xmax = detection.rotated_rect.center.x + width / 2
            ymin = detection.rotated_rect.center.y - height / 2
            ymax = detection.rotated_rect.center.y + height / 2

            padding = self.padding_factor

            slope_x = (xmax + padding) - (xmin - padding)
            slope_y = (ymax + padding) - (ymin - padding)

            new_det = ImgDetectionExtended()
            new_det.rotated_rect = (
                detection.rotated_rect.center.x,
                detection.rotated_rect.center.y,
                detection.rotated_rect.size.width + 2 * padding,
                detection.rotated_rect.size.height + 2 * padding,
                detection.rotated_rect.angle,
            )
            new_det.label = 0
            new_det.confidence = detection.confidence

            xs = []
            ys = []

            kpts = []

            for kp in keypoints_msg.keypoints:
                x = min(max(xmin - padding + slope_x * kp.x, 0.0), 1.0)
                y = min(max(ymin - padding + slope_y * kp.y, 0.0), 1.0)
                new_kpt = Keypoint()
                new_kpt.x = x
                new_kpt.y = y
                kpts.append(new_kpt)
                xs.append(x)
                ys.append(y)

            new_det.keypoints = kpts
            new_dets.detections.append(new_det)

            for connection in self.connection_pairs:
                pt1_ix, pt2_ix = connection
                points_ann = dai.PointsAnnotation()
                points_ann.type = dai.PointsAnnotationType.LINE_STRIP
                points_ann.points = dai.VectorPoint2f(
                    [
                        dai.Point2f(x=xs[pt1_ix], y=ys[pt1_ix], normalized=True),
                        dai.Point2f(x=xs[pt2_ix], y=ys[pt2_ix], normalized=True),
                    ]
                )
                points_ann.outlineColor = PRIMARY_COLOR
                points_ann.thickness = 2.0
                annotation.points.append(points_ann)

            gesture = recognize_gesture([[kpt.x, kpt.y] for kpt in kpts])

            text_ann = dai.TextAnnotation()
            text_x = detection.rotated_rect.center.x - 0.05
            text_y = detection.rotated_rect.center.y - height / 2 - 0.10
            text_ann.position = dai.Point2f(x=text_x, y=text_y, normalized=True)
            text_ann.text = "Left" if handness < 0.5 else "Right"
            text_ann.text += f" {gesture}"
            text_ann.fontSize = 32
            text_ann.textColor = PRIMARY_COLOR
            annotation.texts.append(text_ann)

            annotations.annotations.append(annotation)

        new_dets.setTimestamp(detections_message.getTimestamp())
        new_dets.setSequenceNum(detections_message.getSequenceNum())
        self.out_detections.send(new_dets)

        annotations.setTimestamp(detections_message.getTimestamp())
        annotations.setSequenceNum(detections_message.getSequenceNum())
        self.out_pose_annotations.send(annotations)
