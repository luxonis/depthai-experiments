from datetime import timedelta
from typing import List, Tuple, Union
import cv2
import depthai as dai
import numpy as np
from depthai_nodes import (
    ImgDetectionExtended,
    ImgDetectionsExtended,
    Keypoints,
)


class DrawDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.thickness = 2
        self.color = (255, 0, 0)
        self.kpt_color = (0, 255, 0)
        self.draw_labels = True
        self.draw_confidence = True
        self.draw_kpts = True
        self.kpt_size = 4
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self,
        frame: dai.Node.Output,
        nn: dai.Node.Output,
        label_map: List[str],
        lines: List[Tuple[int, int]] = [],
    ) -> "DrawDetections":
        self.label_map = label_map
        self.lines = lines

        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self

    def process(self, in_frame: dai.ImgFrame, in_detections: dai.Buffer) -> None:
        frame = in_frame.getCvFrame()
        assert isinstance(
            in_detections,
            (
                dai.ImgDetections,
                ImgDetectionsExtended,
                Keypoints,
                dai.SpatialImgDetections,
            ),
        )

        out_frame = self.draw_detections(frame, in_detections)
        img = self._create_img_frame(
            out_frame,
            dai.ImgFrame.Type.BGR888p,
            in_frame.getTimestamp(),
            in_frame.getSequenceNum(),
        )

        self.output.send(img)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Union[
            ImgDetectionsExtended,
            dai.ImgDetections,
            Keypoints,
            dai.SpatialImgDetections,
        ],
    ) -> np.ndarray:
        if isinstance(
            detections,
            (ImgDetectionsExtended, dai.ImgDetections, dai.SpatialImgDetections),
        ):
            frame = self._draw_bboxes(frame, detections.detections)
        if isinstance(detections, ImgDetectionsExtended) and self.draw_kpts:
            kpts = [i.keypoints for i in detections.detections]
            frame = self._draw_kpts(frame, kpts)
            frame = self._draw_lines(frame, kpts, self.lines)
        if isinstance(detections, Keypoints) and self.draw_kpts:
            if len(detections.keypoints) > 0:
                kpts = [(i.x, i.y) for i in detections.keypoints]
                frame = self._draw_kpts(frame, [kpts])
                frame = self._draw_lines(frame, [kpts], self.lines)
        return frame

    def _draw_lines(
        self,
        frame: np.ndarray,
        kpts: List[List[Tuple[float, float]]],
        lines: List[Tuple[int, int]],
    ) -> np.ndarray:
        for kpts_det in kpts:
            for line in lines:
                pt1 = (int(kpts_det[line[0]][0]), int(kpts_det[line[0]][1]))
                pt2 = (int(kpts_det[line[1]][0]), int(kpts_det[line[1]][1]))
                frame = cv2.line(frame, pt1, pt2, self.kpt_color, 1)
        return frame

    def _draw_kpts(
        self,
        frame: np.ndarray,
        kpts: List[List[Tuple[float, float]]],
    ):
        for kpts_det in kpts:
            for kpt in kpts_det:
                frame = cv2.circle(
                    frame,
                    (int(kpt[0]), int(kpt[1])),
                    self.kpt_size,
                    self.kpt_color,
                    cv2.FILLED,
                )

        return frame

    def _draw_bboxes(
        self,
        frame: np.ndarray,
        detections: Union[
            List[dai.ImgDetection],
            List[ImgDetectionExtended],
            List[dai.SpatialImgDetection],
        ],
    ) -> np.ndarray:
        for detection in detections:
            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            bbox = (np.clip(np.array(bbox), 0, 1) * bbox).astype(int)
            if self.draw_labels:
                cv2.putText(
                    frame,
                    self.label_map[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    self.color,
                )
                if isinstance(detection, dai.SpatialImgDetection):
                    texts = [
                        f"X: {detection.spatialCoordinates.x / 1000:>4.02f} m",
                        f"Y: {detection.spatialCoordinates.y / 1000:>4.02f} m",
                        f"Z: {detection.spatialCoordinates.z / 1000:>4.02f} m",
                    ]
                    for ind, t in enumerate(texts):
                        cv2.putText(
                            frame,
                            t,
                            (bbox[0] + 10, bbox[1] + 60 + ind * 20),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            self.color,
                        )
            if self.draw_confidence:
                cv2.putText(
                    frame,
                    f"{int(detection.confidence * 100)}%",
                    (bbox[0] + 10, bbox[1] + 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    self.color,
                )
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                self.color,
                self.thickness,
            )

        return frame

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self.color = color

    def set_kpt_color(self, kpt_color: Tuple[int, int, int]) -> None:
        self.kpt_color = kpt_color

    def set_thickness(self, thickness: int) -> None:
        self.thickness = thickness

    def set_kpt_size(self, kpt_size: int) -> None:
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
