import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import Map2D
from host_node.annotation_builder import AnnotationBuilder


class VisualizeCrowdCount(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageAnnotations, True)
            ]
        )

    def build(self, nn: dai.Node.Output):
        self.link_args(nn)
        return self

    def process(self, detections: dai.Buffer):
        assert isinstance(detections, Map2D)

        count = np.sum(detections.map)
        annotation_builder = AnnotationBuilder()
        text = f"Predicted count: {count:.2f}"
        annotation_builder.draw_text(text, (0.1, 0.1), (255, 255, 255, 1), (0, 0, 0, 1))
        annotation = annotation_builder.build(
            detections.getTimestamp(), detections.getSequenceNum()
        )
        self.output.send(annotation)
