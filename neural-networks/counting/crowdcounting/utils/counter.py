import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import Map2D
from .text_annotation_builder import TextAnnotationBuilder


class CrowdCounter(dai.node.HostNode):
    """A host node that receives map of people detections and outputs the number of people in the crowd.

    Attributes
    ----------
    output : dai.ImgAnnotations
        The output message for the number of people in the crowd.
    """

    def __init__(self):
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def build(self, nn: dai.Node.Output):
        self.link_args(nn)
        return self

    def process(self, detections: dai.Buffer):
        assert isinstance(detections, Map2D)

        count = np.sum(detections.map)
        annotation_builder = TextAnnotationBuilder()
        text = f"Predicted count: {count:.2f}"
        annotation_builder.draw_text(text, (0.1, 0.1), (255, 255, 255, 1), (0, 0, 0, 1))
        annotation = annotation_builder.build(
            detections.getTimestamp(), detections.getSequenceNum()
        )
        self.output.send(annotation)
