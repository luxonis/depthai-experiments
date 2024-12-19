import depthai as dai
import numpy as np
from host_node.annotation_builder import AnnotationBuilder
from host_node.measure_object_distance import DetectionDistance, ObjectDistances


class VisualizeObjectDistances(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.color = (0, 0, 255)
        self.text_color = (255, 255, 255)

        self._state_queue = []

    def build(self, distances: dai.Node.Output) -> "VisualizeObjectDistances":
        self.link_args(distances)
        return self

    def process(self, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        annotations = self._draw_overlay(distances)

        self.output.send(annotations)

    def _draw_overlay(self, distances: ObjectDistances):
        annotation_builder = AnnotationBuilder()
        for distance in distances.distances:
            self._draw_distance_line(distance, annotation_builder)
        return annotation_builder.build(
            distances.getTimestamp(), distances.getSequenceNum()
        )

    def _draw_distance_line(
        self, distance: DetectionDistance, annotation_builder: AnnotationBuilder
    ):
        det1 = distance.detection1
        det2 = distance.detection2
        text = f"{round(distance.distance / 1000, 1)} m"
        x_start = (det1.xmin + det1.xmax) / 2
        y_start = (det1.ymin + det1.ymax) / 2
        x_end = (det2.xmin + det2.xmax) / 2
        y_end = (det2.ymin + det2.ymax) / 2
        annotation_builder.draw_line(
            (x_start, y_start), (x_end, y_end), color=self.color + (1,), thickness=2
        )
        label_x = (x_start + x_end) / 2
        label_y = (y_start + y_end) / 2 - 0.02
        annotation_builder.draw_text(
            text=text,
            position=(label_x, label_y),
            color=self.text_color + (1,),
            background_color=self.color + (1,),
            size=24,
        )
        return annotation_builder

    def set_color(self, color: tuple[int, int, int]):
        self.color = color

    def set_text_color(self, color: tuple[int, int, int]):
        self.text_color = color

    def _get_abs_coordinates(
        self, point: tuple[float, float], img_size: tuple[int, int]
    ) -> tuple[int, int]:
        return (
            int(np.clip(point[0], 0, 1) * img_size[1]),
            int(np.clip(point[1], 0, 1) * img_size[0]),
        )
