import depthai as dai


class DetectionLabelFilter(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self._accepted_labels = []
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )

    def build(
        self, nn: dai.Node.Output, accepted_labels: list[int]
    ) -> "DetectionLabelFilter":
        self._accepted_labels = accepted_labels
        self.link_args(nn)
        return self

    def process(self, detections: dai.ImgDetections) -> None:
        filtered_detections: list[dai.ImgDetection] = [
            i for i in detections.detections if i.label in self._accepted_labels
        ]
        img_detections = dai.ImgDetections()
        img_detections.detections = filtered_detections
        img_detections.setTimestamp(detections.getTimestamp())
        img_detections.setSequenceNum(detections.getSequenceNum())
        self.output.send(img_detections)
