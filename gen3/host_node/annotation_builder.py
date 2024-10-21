from datetime import timedelta

import depthai as dai

Point = tuple[float, float]
ColorRGBA = tuple[float, float, float, float]


class AnnotationBuilder:
    def __init__(self):
        self.annotation: dai.ImageAnnotation = dai.ImageAnnotation()

    def draw_line(self, pt1: Point, pt2: Point, color: ColorRGBA, thickness: float):
        line = dai.PointsAnnotation()
        c = self._create_color(color)
        line.fillColor = c
        line.outlineColor = c
        line.thickness = thickness
        line.type = dai.PointsAnnotationType.LINE_STRIP
        line.points = self._create_points_vector([pt1, pt2])
        self.annotation.points.append(line)
        return self

    def draw_polyline(
        self,
        points: list[Point],
        outline_color: ColorRGBA,
        fill_color: ColorRGBA | None = None,
        thickness: float = 1,
        closed: bool = False,
    ):
        # TODO: Update this to use LINE_STRIP, once it is fixed in DepthAI
        if not closed:
            for i in range(len(points) - 1):
                self.draw_line(points[i], points[i + 1], outline_color, thickness)
            return self

        polyline = dai.PointsAnnotation()
        polyline.outlineColor = self._create_color(outline_color)
        if fill_color is not None:
            polyline.fillColor = self._create_color(fill_color)
        polyline.thickness = thickness
        polyline.type = (
            dai.PointsAnnotationType.LINE_STRIP
        )  # TODO: Update this to POLYGON, once it is fixed in DepthAI
        polyline.points = self._create_points_vector(points)
        self.annotation.points.append(polyline)
        return self

    def draw_points(self, points: list[Point], color: ColorRGBA, size: float = 0.003):
        # TODO: Update this to use dai.PointsAnnotationType.POINTS, once it is fixed in DepthAI
        for pt in points:
            self.draw_circle(pt, size, color, color, 0)
        return self

    def draw_circle(
        self,
        center: Point,
        radius: float,
        outline_color: ColorRGBA,
        fill_color: ColorRGBA | None = None,
        thickness: float = 1,
    ):
        circle = dai.CircleAnnotation()
        circle.outlineColor = self._create_color(outline_color)
        if fill_color is not None:
            circle.fillColor = self._create_color(fill_color)
        circle.thickness = thickness
        circle.diameter = radius * 2
        circle.position = dai.Point2f(center[0], center[1])
        self.annotation.circles.append(circle)
        return self

    def draw_rectangle(
        self,
        top_left: Point,
        bottom_right: Point,
        outline_color: ColorRGBA,
        fill_color: ColorRGBA | None = None,
        thickness: float = 1,
    ):
        points = [
            top_left,
            (bottom_right[0], top_left[1]),
            bottom_right,
            (top_left[0], bottom_right[1]),
        ]
        self.draw_polyline(points, outline_color, fill_color, thickness, closed=True)
        return self

    def draw_text(
        self,
        text: str,
        position: Point,
        color: ColorRGBA,
        background_color: ColorRGBA | None = None,
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

    def build(self, timestamp: timedelta, sequence_num: int) -> dai.ImageAnnotations:
        annotations_msg = dai.ImageAnnotations()
        annotations_msg.annotations = dai.VectorImageAnnotation([self.annotation])
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

    def _create_points_vector(self, points: list[Point]) -> dai.VectorPoint2f:
        return dai.VectorPoint2f([dai.Point2f(pt[0], pt[1]) for pt in points])
