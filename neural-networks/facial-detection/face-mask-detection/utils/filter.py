import depthai as dai
from depthai_nodes import ImgDetectionExtended, ImgDetectionsExtended


class FilterDetections(dai.node.HostNode):
    """A host node that iterates over a list of detections and outputs only those defined in the label encoding.

    ----------
    detections_msg : dai.Input
        Detections input message.
    label_encoding : dict
        Dict of label indexes and their corresponding names. Only the detections with the corresponding label will be outputed.
    output : dai.Output
        The output message for the ImgDetectionsExtended object.
    """

    def __init__(self, label_encoding: dict = None):
        super().__init__()
        self.output = self.createOutput()
        self._label_encoding = label_encoding

    def build(self, detections_msg: dai.Node.Output):
        self.link_args(detections_msg)
        return self

    def set_label_encoding(self, label_encoding: dict):
        self._label_encoding = label_encoding

    def process(self, detections_msg: dai.Buffer):
        assert isinstance(detections_msg, dai.ImgDetections)

        detections = []
        for detection in detections_msg.detections:
            if detection.label in self._label_encoding.keys():
                detections.append(self._process_detection(detection))

        detections_msg_new: ImgDetectionsExtended = ImgDetectionsExtended()
        detections_msg_new.detections = detections

        detections_msg_new.transformation = detections_msg.getTransformation()

        detections_msg_new.setTimestamp(detections_msg.getTimestamp())
        detections_msg_new.setSequenceNum(detections_msg.getSequenceNum())

        self.output.send(detections_msg_new)

    def _process_detection(self, detection: dai.ImgDetection) -> ImgDetectionExtended:
        processed_detection: ImgDetectionExtended = ImgDetectionExtended()

        x_center = (detection.xmin + detection.xmax) / 2
        y_center = (detection.ymin + detection.ymax) / 2
        width = detection.xmax - detection.xmin
        height = detection.ymax - detection.ymin
        angle = 0  # we expect no rotation
        processed_detection.rotated_rect = [x_center, y_center, width, height, angle]

        processed_detection.confidence = detection.confidence

        label = detection.label
        processed_detection.label = label
        processed_detection.label_name = self._label_encoding[label]

        return processed_detection
