import depthai as dai
import time
import os


class SnapsProducer(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def build(
        self,
        rgb: dai.Node.Output,
        detections: dai.Node.Output,
        label_map: list,
        confidence_threshold: float = 0.7,
        labels: list = ["person"],
        time_interval: float = 60.0,
    ) -> None:
        self.link_args(rgb, detections)

        self.em = dai.EventsManager()
        self.em.setLogResponse(True)
        if os.getenv("DEPTHAI_HUB_URL") is not None:
            self.em.setUrl(os.getenv("DEPTHAI_HUB_URL"))

        self.label_map = label_map
        self.confidence_threshold = confidence_threshold
        self.labels = labels
        self.last_update = time.time()
        self.time_interval = time_interval

        return self

    def process(self, rgb: dai.Buffer, detections: dai.ImgDetections):
        for det in detections.detections:
            if (
                det.confidence < self.confidence_threshold
                and self.label_map[det.label] in self.labels
                and time.time() > self.last_update + self.time_interval
            ):
                self.last_update = time.time()
                det_xyxy = [det.xmin, det.ymin, det.xmax, det.ymax]
                extra_data = {
                    "model": "luxonis/yolov6-nano:r2-coco-512x288",
                    "detection_xyxy": str(det_xyxy),
                    "detection_label": str(det.label),
                    "detection_label_str": self.label_map[det.label],
                    "detection_confidence": str(det.confidence),
                }
                self.em.sendSnap(
                    "rgb",
                    rgb,
                    [],
                    ["demo"],
                    extra_data,
                )

                print(f"Event sent: {extra_data}")
