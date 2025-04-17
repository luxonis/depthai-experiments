from datetime import timedelta
from typing import List, Optional, Tuple

import depthai as dai
import numpy as np

Point = Tuple[float, float]
ColorRGBA = Tuple[float, float, float, float]


class AnnotationHelper:
    """Simplifies `dai.ImgAnnotation` creation.

    After calling the desired drawing methods, call the `build` method to create the `ImgAnnotations` message.
    """

    def __init__(self):
        self.annotation: dai.ImgAnnotation = dai.ImgAnnotation()

    def draw_line(
        self, pt1: Point, pt2: Point, color: ColorRGBA, thickness: float
    ) -> "AnnotationHelper":
        """Draws a line between two points.

        @param pt1: Start of the line
        @type pt1: Point
        @param pt2: End of the line
        @type pt2: Point
        @param color: Line color
        @type color: ColorRGBA
        @param thickness: Line thickness
        @type thickness: float
        @return: self
        @rtype: AnnotationHelper
        """
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
        points: List[Point],
        outline_color: ColorRGBA,
        fill_color: Optional[ColorRGBA] = None,
        thickness: float = 1,
        closed: bool = False,
    ) -> "AnnotationHelper":
        """Draws a polyline.

        @param points: List of points of the polyline
        @type points: list[Point]
        @param outline_color: Outline color
        @type outline_color: ColorRGBA
        @param fill_color: Fill color (None for no fill), defaults to None
        @type fill_color: ColorRGBA | None, optional
        @param thickness: Line thickness, defaults to 1
        @type thickness: float, optional
        @param closed: Creates polygon, instead of polyline if True, defaults to False
        @type closed: bool, optional
        @return: self
        @rtype: AnnotationHelper
        """
        points_type = (
            dai.PointsAnnotationType.LINE_STRIP
            if not closed
            else dai.PointsAnnotationType.LINE_LOOP
        )
        points_annot = self._create_points_annotation(
            points, outline_color, fill_color, points_type
        )
        points_annot.thickness = thickness
        self.annotation.points.append(points_annot)
        return self

    def draw_points(
        self, points: List[Point], color: ColorRGBA, thickness: float = 2
    ) -> "AnnotationHelper":
        """Draws points.

        @param points: List of points to draw
        @type points: list[Point]
        @param color: Color of the points
        @type color: ColorRGBA
        @param thickness: Size of the points, defaults to 2
        @type thickness: float, optional
        @return: self
        @rtype: AnnotationHelper
        """
        # TODO: Visualizer currently does not show dai.PointsAnnotationType.POINTS
        points_annot = self._create_points_annotation(
            points, color, None, dai.PointsAnnotationType.POINTS
        )
        points_annot.thickness = thickness
        self.annotation.points.append(points_annot)
        return self

    def draw_circle(
        self,
        center: Point,
        radius: float,
        outline_color: ColorRGBA,
        fill_color: Optional[ColorRGBA] = None,
        thickness: float = 1,
    ) -> "AnnotationHelper":
        """Draws a circle.

        @param center: Center of the circle
        @type center: Point
        @param radius: Radius of the circle
        @type radius: float
        @param outline_color: Outline color
        @type outline_color: ColorRGBA
        @param fill_color: Fill color (None for no fill), defaults to None
        @type fill_color: ColorRGBA | None, optional
        @param thickness: Outline thickness, defaults to 1
        @type thickness: float, optional
        @return: self
        @rtype: AnnotationHelper
        """
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
        fill_color: Optional[ColorRGBA] = None,
        thickness: float = 1,
    ) -> "AnnotationHelper":
        """Draws a rectangle.

        @param top_left: Top left corner of the rectangle
        @type top_left: Point
        @param bottom_right: Bottom right corner of the rectangle
        @type bottom_right: Point
        @param outline_color: Outline color
        @type outline_color: ColorRGBA
        @param fill_color: Fill color (None for no fill), defaults to None
        @type fill_color: ColorRGBA | None, optional
        @param thickness: Outline thickness, defaults to 1
        @type thickness: float, optional
        @return: self
        @rtype: AnnotationHelper
        """
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
        background_color: Optional[ColorRGBA] = None,
        size: float = 32,
    ) -> "AnnotationHelper":
        """Draws text.

        @param text: Text string
        @type text: str
        @param position: Text position
        @type position: Point
        @param color: Text color
        @type color: ColorRGBA
        @param background_color: Background color (None for no background), defaults to
            None
        @type background_color: ColorRGBA | None, optional
        @param size: Text size, defaults to 32
        @type size: float, optional
        @return: self
        @rtype: AnnotationHelper
        """
        text_annot = dai.TextAnnotation()
        text_annot.position = dai.Point2f(position[0], position[1])
        text_annot.text = text
        text_annot.textColor = self._create_color(color)
        text_annot.fontSize = size
        if background_color is not None:
            text_annot.backgroundColor = self._create_color(background_color)
        self.annotation.texts.append(text_annot)
        return self

    def draw_rotated_rect(
        self,
        center: Point,
        size: Tuple[float, float],
        angle: float,
        outline_color: ColorRGBA,
        fill_color: Optional[ColorRGBA] = None,
        thickness: float = 1,
    ) -> "AnnotationHelper":
        """Draws a rotated rectangle.

        @param center: Center of the rectangle
        @type center: Point
        @param size: Size of the rectangle (width, height)
        @type size: tuple[float, float]
        @param angle: Angle of rotation in degrees
        @type angle: float
        @param outline_color: Outline color
        @type outline_color: ColorRGBA
        @param fill_color: Fill color (None for no fill), defaults to None
        @type fill_color: ColorRGBA | None, optional
        @param thickness: Outline thickness, defaults to 1
        @type thickness: float, optional
        @return: self
        @rtype: AnnotationHelper
        """
        points = self._get_rotated_rect_points(center, size, angle)
        self.draw_polyline(points, outline_color, fill_color, thickness, True)
        return self

    def build(self, timestamp: timedelta, sequence_num: int) -> dai.ImgAnnotations:
        """Creates an ImgAnnotations message.

        @param timestamp: Message timestamp
        @type timestamp: timedelta
        @param sequence_num: Message sequence number
        @type sequence_num: int
        @return: Created ImgAnnotations message
        @rtype: dai.ImgAnnotations
        """
        annotations_msg = dai.ImgAnnotations()
        annotations_msg.annotations = dai.VectorImgAnnotation([self.annotation])
        annotations_msg.setTimestamp(timestamp)
        annotations_msg.setSequenceNum(sequence_num)
        return annotations_msg

    def _create_points_annotation(
        self,
        points: List[Point],
        color: ColorRGBA,
        fill_color: Optional[ColorRGBA],
        type: dai.PointsAnnotationType,
    ) -> dai.PointsAnnotation:
        points_annot = dai.PointsAnnotation()
        points_annot.outlineColor = self._create_color(color)
        if fill_color is not None:
            points_annot.fillColor = self._create_color(fill_color)
        points_annot.type = type
        points_annot.points = self._create_points_vector(points)
        return points_annot

    def _create_color(self, color: ColorRGBA) -> dai.Color:
        c = dai.Color()
        c.a = color[3]
        c.r = color[0]
        c.g = color[1]
        c.b = color[2]
        return c

    def _get_rotated_rect_points(
        self, center: Point, size: Tuple[float, float], angle: float
    ) -> List[Point]:
        cx, cy = center
        width, height = size
        angle_rad = np.radians(angle)

        # Half-dimensions
        dx = width / 2
        dy = height / 2

        # Define the corners relative to the center
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])

        # Rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        # Rotate and translate the corners
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([cx, cy])

        # Convert to list of tuples
        return translated_corners.tolist()

    def _create_points_vector(self, points: List[Point]) -> dai.VectorPoint2f:
        return dai.VectorPoint2f([dai.Point2f(pt[0], pt[1]) for pt in points])
