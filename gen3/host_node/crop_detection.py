import depthai as dai
import numpy as np
from depthai_nodes.ml.messages import ImgDetectionsExtended


class CropDetection(dai.node.HostNode):
    """Crops the image based on the detection with the highest confidence. Creates dai.ImageManipConfig for cropping and resizing.
    `detection_passthrough` output sends out detection, that is used for cropping."""

    def __init__(self) -> None:
        super().__init__()
        self._resize = None
        self._bbox_padding = 0.1
        self._last_config = dai.ImageManipConfig()
        self.output_config = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)
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
        self._last_config.setResize(*size)

    def set_bbox_padding(self, padding: float) -> None:
        self._bbox_padding = padding

    def process(self, nn: dai.Buffer) -> None:
        assert isinstance(nn, (dai.ImgDetections, ImgDetectionsExtended))
        dets = nn.detections
        if len(dets) == 0:
            # No detections, there is nothing to crop
            self.detection_passthrough.send(nn)
            self.output_config.send(self._last_config)
            return
        # take detection with highest confidence
        best_detection = sorted(dets, key=lambda d: d.confidence, reverse=True)[0]
        bbox = (
            best_detection.xmin - self._bbox_padding,
            best_detection.ymin - self._bbox_padding,
            best_detection.xmax + self._bbox_padding,
            best_detection.ymax + self._bbox_padding,
        )

        bbox = self._limit_roi(bbox)
        cfg = dai.ImageManipConfig()
        cfg.setCropRect(*bbox)
        cfg.setKeepAspectRatio(False)

        if self._resize is not None:
            cfg.setResize(*self._resize)

        self._last_config = cfg
        self.output_config.send(cfg)

        img_dets = dai.ImgDetections()
        img_dets.detections = [best_detection]
        img_dets.setSequenceNum(nn.getSequenceNum())
        img_dets.setTimestamp(nn.getTimestamp())
        self.detection_passthrough.send(img_dets)

    def _limit_roi(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        return tuple(np.clip(bbox, 0.001, 0.999))
