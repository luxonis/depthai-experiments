import depthai as dai
from depthai_nodes.ml.messages import ImgDetectionsExtended

AVG_MAX_NUM = 3


class CropFace(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )

        self.x = []
        self.y = []

    def build(self, detections: dai.Node.Output):
        self.link_args(detections)
        return self

    def process(self, detections: dai.Buffer):
        assert isinstance(detections, ImgDetectionsExtended)

        dets = detections.detections
        if len(dets) == 0:
            return

        coords = dets[0]  # Take first
        xmin, ymin, xmax, ymax = coords.rotated_rect.getOuterRect()

        # Get detection center
        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        x_avg, y_avg = self.average_filter(x, y)

        rect = dai.RotatedRect()
        rect.size = dai.Size2f(0.4, 0.4)
        rect.center = dai.Point2f(x_avg, y_avg)

        cfg = dai.ImageManipConfigV2()
        cfg.addCropRotatedRect(rect, True)
        cfg.setTimestamp(detections.getTimestamp())
        self.output.send(cfg)

    def average_filter(self, x, y):
        self.x.append(x)
        self.y.append(y)

        if AVG_MAX_NUM < len(self.x):
            self.x.pop(0)
        if AVG_MAX_NUM < len(self.y):
            self.y.pop(0)

        x_avg = sum(self.x) / len(self.x)
        y_avg = sum(self.y) / len(self.y)

        return x_avg, y_avg
