import depthai as dai
from utils.utils import get_boxes


class MaskToBbox(dai.node.HostNode):
    """Transforms mask received in frame of dai.ImgFrame to bounding boxes."""

    def __init__(self):
        super().__init__()
        self._bbox_threshold = 0.2
        self._bitmap_threshold = 0.01
        self._min_size = 1
        self._max_candidates = 75
        self._padding = 5
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgDetections, True)
            ]
        )

    def build(self, nn: dai.Node.Output) -> "MaskToBbox":
        self.link_args(nn)
        return self

    def process(self, mask_frame: dai.ImgFrame) -> None:
        boxes, scores = get_boxes(
            mask_frame.getFrame(),
            self._bitmap_threshold,
            self._bbox_threshold,
            self._min_size,
            self._max_candidates,
        )

        detections = dai.ImgDetections()
        dets = []
        for bbox, score in zip(boxes, scores):
            detection = dai.ImgDetection()
            detection.confidence = score
            detection.xmin = (bbox[0, 0] - self._padding) / mask_frame.getWidth()
            detection.ymin = (bbox[0, 1] - self._padding) / mask_frame.getHeight()
            detection.xmax = (bbox[2, 0] + self._padding) / mask_frame.getWidth()
            detection.ymax = (bbox[2, 1] + self._padding) / mask_frame.getHeight()
            detection.label = 0
            dets.append(detection)
        detections.detections = dets
        detections.setSequenceNum(mask_frame.getSequenceNum())
        detections.setTimestamp(mask_frame.getTimestamp())

        self.output.send(detections)

    def setBboxThreshold(self, threshold: float) -> None:
        self._bbox_threshold = threshold

    def setBitmapThreshold(self, threshold: float) -> None:
        self._bitmap_threshold = threshold

    def setMinSize(self, min_size: int) -> None:
        self._min_size = min_size

    def setMaxCandidates(self, max_candidates: int) -> None:
        self._max_candidates = max_candidates

    def setPadding(self, padding: int) -> None:
        self._padding = padding
