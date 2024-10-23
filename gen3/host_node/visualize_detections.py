from datetime import timedelta

import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
)
from host_node.annotation_builder import AnnotationBuilder


class VisualizeDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.thickness = 2
        self.color = (0, 0, 255)
        self.text_color = (255, 255, 255)
        self.kpt_color = (0, 255, 0)
        self.draw_labels = True
        self.draw_confidence = True
        self.draw_kpts = True
        self.kpt_size = 0.01
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)
            ]
        )

    def build(
        self,
        nn: dai.Node.Output,
        label_map: list[str],
        lines: list[tuple[int, int]] = [],
    ) -> "VisualizeDetections":
        self.label_map = label_map
        self.lines = lines

        self.link_args(nn)
        return self

    def process(self, in_detections: dai.Buffer) -> None:
        assert isinstance(
            in_detections,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
                Keypoints,
                dai.SpatialImgDetections,
            ),
        )

        annotations = self.draw_detections(
            in_detections, in_detections.getTimestamp(), in_detections.getSequenceNum()
        )

        self.output.send(annotations)

    def draw_detections(
        self,
        detections: ImgDetectionsExtended
        | dai.ImgDetections
        | Keypoints
        | dai.SpatialImgDetections,
        timestamp: timedelta,
        sequence_num: int,
    ) -> dai.ImageAnnotations:
        annotation_builder = AnnotationBuilder()
        if isinstance(
            detections,
            (ImgDetectionsExtended, dai.ImgDetections, dai.SpatialImgDetections),
        ):
            self._draw_bboxes(detections.detections, annotation_builder)
        if isinstance(detections, ImgDetectionsExtended) and self.draw_kpts:
            kpts = [i.keypoints for i in detections.detections]
            self._draw_kpts(kpts, annotation_builder)
            self._draw_lines(kpts, self.lines, annotation_builder)
        if isinstance(detections, Keypoints) and self.draw_kpts:
            if len(detections.keypoints) > 0:
                kpts = [(i.x, i.y) for i in detections.keypoints]
                self._draw_kpts([kpts], annotation_builder)
                self._draw_lines([kpts], self.lines, annotation_builder)
        return annotation_builder.build(timestamp, sequence_num)

    def _draw_lines(
        self,
        kpts: list[list[tuple[float, float]]],
        lines: list[tuple[int, int]],
        annotation_builder: AnnotationBuilder,
    ) -> np.ndarray:
        for kpts_det in kpts:
            for line in lines:
                pt1 = (kpts_det[line[0]][0], kpts_det[line[0]][1])
                pt2 = (kpts_det[line[1]][0], kpts_det[line[1]][1])
                color = self.kpt_color + (1,)
                annotation_builder.draw_line(pt1, pt2, color)
        return annotation_builder

    def _draw_kpts(
        self,
        kpts: list[list[tuple[float, float]]],
        annotation_builder: AnnotationBuilder,
    ):
        for kpts_det in kpts:
            for kpt in kpts_det:
                annotation_builder.draw_circle(
                    kpt, self.kpt_size, self.kpt_color + (1,), self.kpt_color + (1,), 1
                )
        return annotation_builder

    def _draw_bboxes(
        self,
        detections: list[dai.ImgDetection]
        | list[ImgDetectionExtended]
        | list[dai.SpatialImgDetection],
        annotation_builder: AnnotationBuilder,
    ) -> np.ndarray:
        for detection in detections:
            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            text = ""
            if self.draw_labels:
                text += self.label_map[detection.label]
                if self.draw_labels:
                    text += " "
            if self.draw_confidence:
                text += f"{int(detection.confidence * 100)}%"
            annotation_builder.draw_text(
                text=text,
                position=(bbox[0], bbox[1]),
                color=self.text_color + (1,),
                background_color=self.color + (1,),
                size=24,
            )
            annotation_builder.draw_rectangle(
                bbox[0:2], bbox[2:4], self.color + (1,), None, 2
            )

        return annotation_builder

    def set_color(self, color: tuple[int, int, int]) -> None:
        self.color = color

    def set_kpt_color(self, kpt_color: tuple[int, int, int]) -> None:
        self.kpt_color = kpt_color

    def set_text_color(self, text_color: tuple[int, int, int]) -> None:
        self.text_color = text_color

    def set_thickness(self, thickness: float) -> None:
        self.thickness = thickness

    def set_kpt_size(self, kpt_size: float) -> None:
        self.kpt_size = kpt_size

    def set_draw_labels(self, draw_labels: bool) -> None:
        self.draw_labels = draw_labels

    def set_draw_confidence(self, draw_confidence: bool) -> None:
        self.draw_confidence = draw_confidence

    def set_draw_kpts(self, draw_kpts: bool) -> None:
        self.draw_kpts = draw_kpts

    def _create_img_frame(
        self,
        frame: np.ndarray,
        type: dai.ImgFrame.Type,
        timestamp: timedelta,
        sequence_num: int,
    ) -> dai.ImgFrame:
        img_frame = dai.ImgFrame()
        img_frame.setCvFrame(frame, type)
        img_frame.setTimestamp(timestamp)
        img_frame.setSequenceNum(sequence_num)
        return img_frame
