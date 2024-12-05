import depthai as dai
from host_node.annotation_builder import AnnotationBuilder
from host_node.measure_object_distance import ObjectDistances

DISTANCE_THRESHOLD = 500  # mm
ALERT_THRESHOLD = 0.5
STATE_QUEUE_LENGTH = 10


class ShowAlert(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)
            ]
        )

        self._state_queue = []

    def build(
        self,
        distances: dai.Node.Output,
        palm_label: int,
        dangerous_objects: list[int],
    ) -> "ShowAlert":
        self.link_args(distances)
        self.palm_label = palm_label
        self.dangerous_objects = dangerous_objects
        return self

    def process(self, distances: dai.Buffer):
        assert isinstance(distances, ObjectDistances)
        found_close_dets = False
        for distance in distances.distances:
            if (
                (
                    distance.detection1.label == self.palm_label
                    and distance.detection2.label in self.dangerous_objects
                )
                or (
                    distance.detection2.label == self.palm_label
                    and distance.detection1.label in self.dangerous_objects
                )
            ) and distance.distance < DISTANCE_THRESHOLD:
                found_close_dets = True
                break
        self._state_queue.append(found_close_dets)

        annotation_builder = AnnotationBuilder()
        if len(self._state_queue) > STATE_QUEUE_LENGTH:
            self._state_queue.pop(0)
        if self._should_alert:
            self._draw_alert(annotation_builder)

        annot = annotation_builder.build(
            distances.getTimestamp(), distances.getSequenceNum()
        )
        self.output.send(annot)

    @property
    def _should_alert(self) -> bool:
        return sum(self._state_queue) / len(self._state_queue) > ALERT_THRESHOLD

    def _draw_alert(self, annotation_builder: AnnotationBuilder) -> AnnotationBuilder:
        text = "Too close"
        annotation_builder.draw_text(
            text, (0.4, 0.5), (255, 255, 255, 1), (255, 0, 0, 1), 64
        )
        annotation_builder.draw_rectangle((0, 0), (1, 1), (255, 0, 0, 1), None, 10)

        return annotation_builder
