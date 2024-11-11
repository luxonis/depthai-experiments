import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class CropDetections(dai.node.HostNode):
    """Crops the image based on the detection with the highest confidence. Creates dai.ImageManipConfig for cropping and resizing.
    `detection_passthrough` output sends out detection, that is used for cropping."""

    def __init__(self) -> None:
        super().__init__()
        self._resize = None
        self._bbox_padding = 0.1
        self._last_config = dai.ImageManipConfigV2()
        self.output_config = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )
        self.detection_passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, detections: dai.Node.Output) -> "CropDetections":
        self.link_args(detections)
        return self

    def set_resize(self, size: tuple[int, int] | None) -> None:
        """Sets the resize size. If None, the original bbox size is used."""
        self._resize = size
        self._last_config.addResize(*size)

    def set_bbox_padding(self, padding: float) -> None:
        self._bbox_padding = padding

    def process(self, dets_msg: dai.Buffer) -> None:
        assert isinstance(dets_msg, (dai.ImgDetections, ImgDetectionsExtended))
        dets = dets_msg.detections
        if len(dets) == 0:
            # No detections, there is nothing to crop
            self.detection_passthrough.send(dets_msg)
            self.output_config.send(self._last_config)
            return
        
        for detection in dets:
            cfg = dai.ImageManipConfigV2()
            rect = self._get_rect(detection)
            cfg.addCrop(rect, True)

            if self._resize is not None:
                cfg.addResize(*self._resize)

            self._last_config = cfg
            cfg.setTimestamp(dets_msg.getTimestamp())
            try:
                self.output_config.send(cfg)
            except dai.MessageQueue.QueueException as e:
                return

        self.detection_passthrough.send(dets_msg)

    def _get_rect(self, detection: dai.ImgDetection) -> dai.Rect:
        min = dai.Point2f(detection.xmin, detection.ymin)
        max = dai.Point2f(detection.xmax, detection.ymax)
        rect = dai.Rect(min, max)
        return rect