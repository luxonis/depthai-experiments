import depthai as dai

full_size = (3840, 2160)  # 4K
zoom_size = (1920, 1080)  # 1080P

size = dai.Size2f(zoom_size[0], zoom_size[1])

AVG_MAX_NUM = 3

limits = [zoom_size[0] // 2, zoom_size[1] // 2]  # xmin and ymin limits
limits.append(full_size[0] - limits[0])  # xmax limit
limits.append(full_size[1] - limits[1])  # ymax limit


class CropFace(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfig, True)
            ]
        )

        self.x = []
        self.y = []

    def build(self, detections):
        self.link_args(detections)
        return self

    def process(self, detections):
        dets = detections.detections
        if len(dets) == 0:
            return

        coords = dets[0]  # Take first
        # Get detection center
        x = (coords.xmin + coords.xmax) / 2 * full_size[0]
        y = (coords.ymin + coords.ymax) / 2 * full_size[1] + 100

        x_avg, y_avg = self.average_filter(x, y)

        rect = dai.RotatedRect()
        rect.size = size
        rect.center = dai.Point2f(x_avg, y_avg)

        cfg = dai.ImageManipConfig()
        cfg.setCropRotatedRect(rect, False)
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

        x_avg = limits[0] if x_avg < limits[0] else x_avg
        y_avg = limits[1] if y_avg < limits[1] else y_avg
        x_avg = limits[2] if limits[2] < x_avg else x_avg
        y_avg = limits[3] if limits[3] < y_avg else y_avg

        return x_avg, y_avg
