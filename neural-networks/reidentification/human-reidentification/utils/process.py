import depthai as dai
from depthai_nodes.message import ImgDetectionExtended, ImgDetectionsExtended


class ProcessDetections(dai.node.HostNode):
    """A host node for processing a list of detections in a two-stage pipeline.
    The node iterates over a list of detections and sends a dai.MessageGroup with
    a list of ImageManipConfigV2 objects that can be executed by the ImageManipV2 node.

    Before use, the target size need to be set with the set_target_size method.
    Attributes
    ----------
    detections_msg : dai.Input
        The input message for the detections.
    config_output : dai.Output
        The output message for the ImageManipConfigV2 objects.
    num_configs_output : dai.Output
        The output message for the number of configs.

    """

    def __init__(self):
        super().__init__()
        self.config_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self.num_configs_output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )

    def build(self, detections_msg: dai.Node.Output):
        self.link_args(detections_msg)
        return self

    def process(self, detections_msg: dai.Buffer):
        assert isinstance(detections_msg, ImgDetectionsExtended)

        detections = detections_msg.detections

        num_detections = len(detections)
        num_cfgs_message = dai.Buffer(num_detections)

        num_cfgs_message.setTimestamp(detections_msg.getTimestamp())
        num_cfgs_message.setSequenceNum(detections_msg.getSequenceNum())
        self.num_configs_output.send(num_cfgs_message)

        for i, detection in enumerate(detections):
            cfg = dai.ImageManipConfigV2()
            detection: ImgDetectionExtended = detection
            rect = detection.rotated_rect

            cfg.addCropRotatedRect(rect, normalizedCoords=True)
            cfg.setOutputSize(self._target_w, self._target_h)
            cfg.setReusePreviousImage(False)
            cfg.setTimestamp(detections_msg.getTimestamp())
            cfg.setSequenceNum(detections_msg.getSequenceNum())
            self.config_output.send(cfg)

    def set_target_size(self, w: int, h: int):
        """Set the target size for the output image."""
        self._target_w = w
        self._target_h = h
