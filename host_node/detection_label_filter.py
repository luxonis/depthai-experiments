import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class DetectionLabelFilter(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._accepted_labels = []
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, nn: dai.Node.Output, accepted_labels: list[int]
    ) -> "DetectionLabelFilter":
        self._accepted_labels = accepted_labels
        self.link_args(nn)
        return self

    def process(self, detections: dai.Buffer) -> None:
        assert isinstance(
            detections,
            (dai.ImgDetections, dai.SpatialImgDetections, ImgDetectionsExtended),
        )

        filtered_detections = [
            i for i in detections.detections if i.label in self._accepted_labels
        ]
        img_detections = type(detections)()
        img_detections.detections = filtered_detections
        img_detections.setTimestamp(detections.getTimestamp())
        img_detections.setSequenceNum(detections.getSequenceNum())
        self.output.send(img_detections)
