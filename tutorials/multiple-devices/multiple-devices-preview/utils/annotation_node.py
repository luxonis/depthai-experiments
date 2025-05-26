import depthai as dai
from typing import List, Tuple, Optional

DEFAULT_OUTLINE_COLOR_RGBA: Tuple[int, int, int, int] = (0, 1, 0, 1)  # Green


class AnnotationNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self.labels: List[str] = []
        self.outline_color: dai.Color = self._create_dai_color(
            *DEFAULT_OUTLINE_COLOR_RGBA
        )

        self.annotation_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def _create_dai_color(
        self, r: float, g: float, b: float, a: float = 1.0
    ) -> dai.Color:
        color = dai.Color()
        color.r = r
        color.g = g
        color.b = b
        color.a = a
        return color

    def build(
        self,
        input_frame_stream: dai.Node.Output,
        input_detections_stream: dai.Node.Output,
        labels: List[str],
        outline_color_rgba: Optional[Tuple[float, float, float, float]] = None,
    ) -> "AnnotationNode":
        self.labels = labels
        if outline_color_rgba:
            self.outline_color = self._create_dai_color(*outline_color_rgba)

        self.link_args(input_frame_stream, input_detections_stream)
        return self

    def process(
        self, frame_message: dai.ImgFrame, detections_message: dai.ImgDetections
    ) -> None:
        output_annotations_msg = dai.ImgAnnotations()
        output_annotations_msg.setTimestamp(frame_message.getTimestamp())
        output_annotations_msg.setSequenceNum(frame_message.getSequenceNum())

        current_img_annotation_group = dai.ImgAnnotation()

        for detection in detections_message.detections:
            xmin, ymin, xmax, ymax = (
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            )

            rect_points_data = [
                dai.Point2f(xmin, ymin),
                dai.Point2f(xmax, ymin),
                dai.Point2f(xmax, ymax),
                dai.Point2f(xmin, ymax),
            ]

            points_annotation = dai.PointsAnnotation()
            points_annotation.type = dai.PointsAnnotationType.LINE_LOOP
            points_annotation.points = dai.VectorPoint2f(rect_points_data)
            points_annotation.outlineColor = self.outline_color
            points_annotation.thickness = 1.0
            current_img_annotation_group.points.append(points_annotation)

            label_idx = detection.label
            label_text = (
                self.labels[label_idx]
                if 0 <= label_idx < len(self.labels)
                else "Unknown"
            )
            confidence_text = f"{int(detection.confidence * 100)}%"
            full_text = f"{label_text}: {confidence_text}"

            text_annotation = dai.TextAnnotation()
            text_annotation.position = dai.Point2f(xmin + 0.01, ymin + 0.035)
            text_annotation.text = full_text
            text_annotation.fontSize = 10
            text_annotation.textColor = self._create_dai_color(255, 255, 255)
            current_img_annotation_group.texts.append(text_annotation)

        output_annotations_msg.annotations = dai.VectorImgAnnotation(
            [current_img_annotation_group]
        )
        self.annotation_out.send(output_annotations_msg)
