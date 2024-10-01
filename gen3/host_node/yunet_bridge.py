import depthai as dai
from depthai_nodes.ml.messages.img_detections import ImgDetectionsExtended


class YuNetBridge(dai.node.HostNode):
    """Transforms messages received from YuNetParser to dai.ImgDetections"""

    def __init__(self) -> None:
        super().__init__()

        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, nn: dai.Node.Output) -> "YuNetBridge":
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
            trans_det.label = detection.label
            trans_det.confidence = detection.confidence
            trans_det.xmin = detection.xmin
            trans_det.ymin = detection.ymin
            trans_det.xmax = detection.xmax
            trans_det.ymax = detection.ymax

            dets_list.append(trans_det)

        transformed_dets.detections = dets_list
        self.output.send(transformed_dets)
