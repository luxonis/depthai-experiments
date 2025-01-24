import depthai as dai
from depthai_nodes.ml.messages.img_detections import ImgDetectionsExtended


class ParserBridge(dai.node.ThreadedHostNode):
    """Transforms ImgDetectionsExtended received from parsers to dai.ImgDetections"""

    def __init__(self) -> None:
        super().__init__()

        self.palm_detection_input = self.createInput()
        self.out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self) -> "ParserBridge":
        self.build()
        return self

    def run(self):
        while self.isRunning():
            try:
                detections: ImgDetectionsExtended = self.palm_detection_input.get()
            except dai.MessageQueue.QueueException:
                break  # Pipeline was stopped

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
            self.out.send(transformed_dets)
