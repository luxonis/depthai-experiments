import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended


class CropDetection(dai.node.HostNode):
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

    def build(self, nn: dai.Node.Output) -> "CropDetection":
        self.link_args(nn)
        self.sendProcessingToPipeline(True)
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
            rect = self._get_rotated_rect(detection)
            cfg.addCropRotatedRect(rect, True)

            if self._resize is not None:
                cfg.addResize(*self._resize)

            self._last_config = cfg
            self.output_config.send(cfg)

        self.detection_passthrough.send(dets_msg)

    def _get_rotated_rect(self, detection: dai.ImgDetection) -> dai.RotatedRect:
        rect = dai.RotatedRect()
        rect.size = dai.Size2f(detection.xmax - detection.xmin + self._bbox_padding, detection.ymax - detection.ymin + self._bbox_padding)
        rect.center = dai.Point2f((detection.xmin + detection.xmax) / 2, (detection.ymin + detection.ymax) / 2)
        return rect