import depthai as dai


class DetectionCropMaker(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.out_cfg = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )
        self.out_img = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self._confidence_threshold = 0

    def build(
        self,
        detections: dai.Node.Output,
        img: dai.Node.Output,
        crop_size: tuple[int, int],
    ) -> "DetectionCropMaker":
        self._crop_size = crop_size
        self.link_args(detections, img)
        return self

    def set_confidence_threshold(self, threshold: float) -> None:
        self._confidence_threshold = threshold

    def process(self, detections: dai.ImgDetections, img: dai.ImgFrame) -> None:
        for detection in detections.detections:
            if detection.confidence > self._confidence_threshold:
                cfg = dai.ImageManipConfigV2()
                rect = self._get_rect(detection)
                cfg.addCrop(rect, True)
                cfg.setOutputSize(
                    self._crop_size[0],
                    self._crop_size[1],
                    dai.ImageManipConfigV2.ResizeMode.STRETCH,
                )

                cfg.setFrameType(img.getType())
                self.out_cfg.send(cfg)
                self.out_img.send(img)

    def _clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def _get_rect(self, detection: dai.ImgDetection) -> dai.Rect:
        xmin, ymin, xmax, ymax = (
            detection.xmin,
            detection.ymin,
            detection.xmax,
            detection.ymax,
        )
        xmin = self._clamp(xmin, 0.001, 0.999)
        ymin = self._clamp(ymin, 0.001, 0.999)
        xmax = self._clamp(xmax, 0.001, 0.999)
        ymax = self._clamp(ymax, 0.001, 0.999)
        min = dai.Point2f(xmin, ymin)
        max = dai.Point2f(xmax, ymax)
        rect = dai.Rect(min, max)
        return rect
