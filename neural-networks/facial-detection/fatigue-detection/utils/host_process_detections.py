import depthai as dai
from depthai_nodes import ImgDetectionExtended


class ProcessDetections(dai.node.ThreadedHostNode):
    """A host node for processing a list of detections in a two stage pipeline.
    The node iterates over a list of detections and sends a dai.MessageGroup with
    a list of ImageManipConfigV2 objects that can be executed by the ImageManipV2 node.

    Before use, the source and target sizes need to be set with the set_source_size and set_target_size functions.
    Attributes
    ----------
    detections_input : dai.Input
        The input message for the detections.
    config_output : dai.Output
        The output message for the ImageManipConfigV2 objects.
    w : int
        The width of the input image.
    h : int
        The height of the input image.
    target_w : int
        The width of the output image.
    target_h : int
        The height of the output image.

    """

    def __init__(self):
        super().__init__()
        self.detections_input = self.createInput()
        self.config_output = self.createOutput()

        self._w = 1
        self._h = 1

        self._target_w = 100
        self._target_h = 100

    def run(self) -> None:
        while self.isRunning():
            img_detections = self.detections_input.get()
            detections = img_detections.detections

            # w, h = img_detections.transformation.getSize()
            configs_message = dai.MessageGroup()
            for i, detection in enumerate(detections):
                cfg = dai.ImageManipConfigV2()
                detection: ImgDetectionExtended = detection
                rect = detection.rotated_rect
                rect = rect.denormalize(self.w, self.h)
                cfg.addCropRotatedRect(rect, normalizedCoords=False)
                cfg.setOutputSize(self.target_w, self.target_h)
                cfg.setReusePreviousImage(False)
                cfg.setTimestamp(img_detections.getTimestamp())
                cfg.setSequenceNum(img_detections.getSequenceNum())

                configs_message[str(i + 100)] = cfg

            configs_message.setSequenceNum(img_detections.getSequenceNum())
            configs_message.setTimestamp(img_detections.getTimestamp())
            self.config_output.send(configs_message)

    def set_target_size(self, w: int, h: int):
        """Set the target size for the output image."""
        self._target_w = w
        self._target_h = h

    def set_source_size(self, w: int, h: int):
        """Set the source size for the input image."""
        self._w = w
        self._h = h

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def target_w(self):
        return self._target_w

    @property
    def target_h(self):
        return self._target_h

    @w.setter
    def w(self, w):
        self._w = w

    @h.setter
    def h(self, h):
        self._h = h

    @target_w.setter
    def target_w(self, target_w):
        self._target_w = target_w

    @target_h.setter
    def target_h(self, target_h):
        self._target_h = target_h
