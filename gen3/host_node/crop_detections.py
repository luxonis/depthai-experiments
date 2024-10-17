import depthai as dai
import numpy as np


class CropDetections(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._resize = None
        self._bbox_padding = 0.1
        self.output_cfg = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)
            ]
        )
        self.detection_passthrough = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, frame: dai.ImgFrame, nn: dai.Node.Output) -> "CropDetections":
        self.link_args(frame, nn)
        self.sendProcessingToPipeline(True)
        return self

    def set_resize(self, size: tuple[int, int] | None) -> None:
        """Sets the resize size. If None, the original bbox size is used."""
        self._resize = size

    def set_bbox_padding(self, padding: float) -> None:
        self._bbox_padding = padding

    def process(self, frame: dai.ImgFrame, nn: dai.Buffer) -> None:
        assert isinstance(nn, dai.ImgDetections)
        dets = nn.detections
        if len(dets) == 0:
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

        self.output_cfg.send(cfg)

        img_dets = dai.ImgDetections()
        img_dets.detections = [best_detection]
        img_dets.setSequenceNum(nn.getSequenceNum())
        img_dets.setTimestamp(nn.getTimestamp())
        self.detection_passthrough.send(img_dets)

    def _limit_roi(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        return tuple(np.clip(bbox, 0.001, 0.999))
