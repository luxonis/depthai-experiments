import depthai as dai
from depthai_nodes import ImgDetectionsExtended


class ParserBridge(dai.node.HostNode):
    """Transforms ImgDetectionsExtended received from parsers to dai.ImgDetections"""

    def __init__(self) -> None:
        super().__init__()

        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, nn: dai.Node.Output) -> "ParserBridge":
        self.link_args(nn)
        return self

    def process(self, detections: dai.Buffer) -> None:
        assert isinstance(detections, ImgDetectionsExtended)
        transformed_dets = dai.ImgDetections()
        transformed_dets.setTimestamp(detections.getTimestamp())
        transformed_dets.setSequenceNum(detections.getSequenceNum())
        dets_list = []

        for detection in detections.detections:
            trans_det = dai.ImgDetection()
            if detection.label == -1:
                trans_det.label = 0
            else:
                trans_det.label = detection.label
            trans_det.confidence = detection.confidence
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()
            trans_det.xmin = xmin
            trans_det.ymin = ymin
            trans_det.xmax = xmax
            trans_det.ymax = ymax

            dets_list.append(trans_det)

        transformed_dets.detections = dets_list
        self._out.send(transformed_dets)

    @property
    def out(self) -> dai.Node.Output:
        return self._out

    @property
    def output(self) -> dai.Node.Output:
        return self._out
