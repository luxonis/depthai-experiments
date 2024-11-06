import math
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import (
    Classifications,
    Clusters,
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
    SegmentationMasksSAM,
)
from host_node.annotation_builder import AnnotationBuilder


class VisualizeDetectionsV2(dai.node.HostNode):
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
        self.output_mask = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        nn: dai.Node.Output,
        label_map: list[str] | None = None,
        lines: list[tuple[int, int]] = [],
    ) -> "VisualizeDetectionsV2":
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
                Classifications,
                Clusters,
                SegmentationMasksSAM,
            ),
        )

        annotations = self.draw_detections(
            in_detections, in_detections.getTimestamp(), in_detections.getSequenceNum()
        )

        mask_frame = self.draw_mask(in_detections)
        if mask_frame is not None:
            self.output_mask.send(mask_frame)

        self.output.send(annotations)

    def draw_mask(
        self, detections: SegmentationMasksSAM | ImgDetectionExtended
    ) -> np.ndarray | None:
        if (
            isinstance(detections, ImgDetectionExtended)
            and len(detections.masks.shape) == 2
        ):
            mask = detections.masks + 1
        elif (
            isinstance(detections, SegmentationMasksSAM)
            and len(detections.masks.shape) == 3
        ):
            mask = np.zeros(detections.masks.shape[1:])
            for i, m in enumerate(detections.masks):
                ones = m == 1
                mask[ones] = i + 1
        else:
            return None

        return self._create_img_frame(
            mask.astype(np.uint8),
            dai.ImgFrame.Type.RAW8,
            detections.getTimestamp(),
            detections.getSequenceNum(),
        )

    def draw_detections(
        self,
        detections: ImgDetectionsExtended
        | dai.ImgDetections
        | Keypoints
        | dai.SpatialImgDetections
        | Classifications
        | Clusters,
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
        if isinstance(detections, Classifications):
            label, score = max(
                zip(detections.classes, detections.scores), key=lambda x: x[1]
            )
            self._draw_classification(label, f"{score:.2f}", annotation_builder)
        if isinstance(detections, Clusters):
            self._draw_clusters(detections, annotation_builder)
        if isinstance(detections, SegmentationMasksSAM):
            self._draw_mask_contours(detections, annotation_builder)

        return annotation_builder.build(timestamp, sequence_num)

    def _draw_mask_contours(
        self, detections: SegmentationMasksSAM, annotation_builder: AnnotationBuilder
    ):
        if len(detections.masks.shape) == 3:
            for m in detections.masks:
                binary_img = m.astype(np.uint8) * 255

                contours, _ = cv2.findContours(
                    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                norm_points = contours[0].reshape(-1, 2) / np.array(
                    [m.shape[1], m.shape[0]]
                )
                annotation_builder.draw_polyline(
                    norm_points, self.color + (1,), None, 2, True
                )
        return annotation_builder

    def _draw_clusters(self, clusters: Clusters, annotation_builder: AnnotationBuilder):
        for cluster in clusters.clusters:
            pts = [(pt.x, pt.y) for pt in cluster.points]
            annotation_builder.draw_points(pts, self.color + (1,), 0.01)
            annotation_builder.draw_polyline(pts, self.color + (1,), None, 2, False)
        return annotation_builder

    def _draw_classification(
        self, label: str, score: str, annotation_builder: AnnotationBuilder
    ) -> None:
        annotation_builder.draw_text(
            f"{label} {score}",
            (0.1, 0.1),
            self.text_color + (1,),
            self.color + (1,),
            24,
        )
        return annotation_builder

    def _get_bbox_points_from_img_detection_extended(
        self,
        detection: ImgDetectionExtended,
    ) -> list[tuple[float, float]]:
        # Convert angle to radians
        angle_radians = math.radians(detection.angle)

        # Half-width and half-height
        half_w = detection.width / 2
        half_h = detection.height / 2

        # Create initial corner points relative to the center
        corners = np.array(
            [
                [-half_w, -half_h],  # top-left
                [half_w, -half_h],  # top-right
                [half_w, half_h],  # bottom-right
                [-half_w, half_h],  # bottom-left
            ]
        )

        # Rotation matrix
        rotation_matrix = np.array(
            [
                [math.cos(angle_radians), -math.sin(angle_radians)],
                [math.sin(angle_radians), math.cos(angle_radians)],
            ]
        )

        # Rotate corners
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Translate corners to actual center position
        final_corners: np.ndarray = rotated_corners + np.array(
            [detection.x_center, detection.y_center]
        )
        return final_corners.tolist()

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
            if isinstance(detection, ImgDetectionExtended):
                bbox_points = self._get_bbox_points_from_img_detection_extended(
                    detection
                )
            else:
                bbox_points = [
                    (detection.xmin, detection.ymin),
                    (detection.xmax, detection.ymin),
                    (detection.xmax, detection.ymax),
                    (detection.xmin, detection.ymax),
                ]
            # bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            text = ""
            if self.draw_labels:
                if self.label_map:
                    text += self.label_map[detection.label]
                else:
                    text += str(detection.label)
                if self.draw_labels:
                    text += " "
            if self.draw_confidence:
                text += f"{int(detection.confidence * 100)}%"
            annotation_builder.draw_text(
                text=text,
                position=(bbox_points[0]),
                color=self.text_color + (1,),
                background_color=self.color + (1,),
                size=24,
            )
            annotation_builder.draw_polyline(
                bbox_points, self.color + (1,), None, 2, True
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
        img_frame.setWidth(frame.shape[1])
        img_frame.setHeight(frame.shape[0])
        return img_frame
