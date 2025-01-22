import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionExtended


class ProcessDetections(dai.node.ThreadedHostNode):
    """A host node for processing a list of detections in a two-stage pipeline.
    The node iterates over a list of detections and sends a dai.MessageGroup with
    a list of ImageManipConfigV2 objects that can be executed by the ImageManipV2 node.

    Before use, the target size need to be set with the set_target_size method.
    Attributes
    ----------
    detections_input : dai.Input
        The input message for the detections.
    config_output : dai.Output
        The output message for the ImageManipConfigV2 objects.
    num_configs_output : dai.Output
        The output message for the number of configs.

    """

    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        self.config_output = self.createOutput()
        self.num_configs_output = self.createOutput()

    def run(self) -> None:
        while self.isRunning():

            img_detections = self.detections_input.get()
            detections = img_detections.detections

            num_detections = len(detections)
            num_cfgs_message = dai.Buffer(num_detections)

            num_cfgs_message.setTimestamp(img_detections.getTimestamp())
            num_cfgs_message.setSequenceNum(img_detections.getSequenceNum())
            self.num_configs_output.send(num_cfgs_message)

            for i, detection in enumerate(detections):
                cfg = dai.ImageManipConfigV2()
                detection: ImgDetectionExtended = detection
                rect = detection.rotated_rect

                cfg.addCropRotatedRect(rect, normalizedCoords=True)
                cfg.setOutputSize(self._target_w, self._target_h)
                cfg.setReusePreviousImage(False)
                cfg.setTimestamp(img_detections.getTimestamp())
                cfg.setSequenceNum(img_detections.getSequenceNum())
                self.config_output.send(cfg)

    def set_target_size(self, w: int, h: int):
        """Set the target size for the output image."""
        self._target_w = w
        self._target_h = h
