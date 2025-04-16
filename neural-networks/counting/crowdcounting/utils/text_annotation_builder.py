from datetime import timedelta
from typing import Tuple, Optional, List
import depthai as dai

Point = Tuple[float, float]
ColorRGBA = Tuple[float, float, float, float]


class TextAnnotationBuilder:
    def __init__(self):
        self.annotation: dai.ImgAnnotation = dai.ImgAnnotation()

    def draw_text(
        self,
        text: str,
        position: Point,
        color: ColorRGBA,
        background_color: Optional[ColorRGBA] = None,
        size: float = 32,
    ):
        text_annot = dai.TextAnnotation()
        text_annot.position = dai.Point2f(position[0], position[1])
        text_annot.text = text
        text_annot.textColor = self._create_color(color)
        text_annot.fontSize = size
        if background_color is not None:
            text_annot.backgroundColor = self._create_color(background_color)
        self.annotation.texts.append(text_annot)
        return self

    def build(self, timestamp: timedelta, sequence_num: int) -> dai.ImgAnnotations:
        annotations_msg = dai.ImgAnnotations()
        annotations_msg.annotations = dai.VectorImgAnnotation([self.annotation])
        annotations_msg.setTimestamp(timestamp)
        annotations_msg.setSequenceNum(sequence_num)
        return annotations_msg

    def _create_color(self, color: ColorRGBA) -> dai.Color:
        c = dai.Color()
        c.a = color[3]
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        return c

    def _create_points_vector(self, points: List[Point]) -> dai.VectorPoint2f:
        return dai.VectorPoint2f([dai.Point2f(pt[0], pt[1]) for pt in points])
